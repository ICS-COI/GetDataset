import os
import cv2
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from colorama import Fore

GAUSS_PSF: int = 0
SIMULATE_PSF: int = 1


# 从成像原理出发
def imaging_process(compress_rate, target_size, lattice_vectors, offset_vector, shift_vector, filepath_list,
                    result_folder, psf_mode=GAUSS_PSF, sigma_exc=2, sigma_det=2, show_steps=False, save=False, save_lake=False,
                    save_background=False, save_lattice=False, save_illumination=False):
    slice_num = shift_vector['scan_dimensions'][0] * shift_vector['scan_dimensions'][1]
    crop_size = np.int32(np.array(target_size) / compress_rate)
    image = np.ones(tuple([slice_num] + list(crop_size)))
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    # 画PSF侧面图像，用真实PSF以及多次高斯进行尝试，顺便找找有没有其他波状函数可以用在这里！
    if psf_mode == SIMULATE_PSF:


    # 照明点和检测点
    dot_exc = dot_det = np.zeros(tuple(crop_size))
    center_pix = (crop_size[0] // 2, crop_size[1] // 2)
    for i in tqdm(range(crop_size[0]), desc=Fore.LIGHTWHITE_EX + "Generate kernel      ",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.LIGHTWHITE_EX)):
        for j in range(crop_size[1]):
            if psf_mode == GAUSS_PSF:
                dot_exc[i][j] = np.exp(-((i - center_pix[0]) ** 2 + (j - center_pix[1]) ** 2) / (2 * sigma_exc ** 2))
            dot_exc[i][j] = np.exp(-((i - center_pix[0]) ** 2 + (j - center_pix[1]) ** 2) / (2 * sigma_det ** 2))
    if show_steps:
        if psf_mode == GAUSS_PSF:
            utils.single_show(dot_exc, "dot_exc")
        utils.single_show(dot_det, "dot_det")


    # 照明点位置
    lattice = get_lattice_image(image, lattice_vectors, offset_vector, shift_vector, show=False)
    if show_steps:
        utils.single_show(lattice, "lattice location")
    if save_lattice:
        result_name = os.path.join(result_folder, 'lattice.tif')
        utils.save_tiff_3d(result_name, lattice)

    # 照明点附着于位置
    lattice = simulate_blur_2d2(lattice, dot_exc, pad=10, pad_flag=utils.PAD_ZERO)
    if show_steps:
        utils.single_show(lattice, "illumination lattice")
    if save_illumination:
        result_name = os.path.join(result_folder, 'illumination.tif')
        utils.save_tiff_3d(result_name, lattice)

    # 检测矩阵生成
    # img_det = np.zeros(target_size)
    det_weight_dict = {}
    for i in tqdm(range(target_size[0]), desc=Fore.LIGHTWHITE_EX + "Generate detect      ",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.LIGHTWHITE_EX)):
        for j in range(target_size[1]):
            # 从中心在center_pix改到中心在（2*i，2*j)
            y = 2 * i - center_pix[0]
            x = 2 * j - center_pix[1]
            # 创建平移矩阵
            mov = np.float32([[1, 0, x],
                              [0, 1, y]])
            # 应用平移矩阵对图像进行平移变换
            det_weight = cv2.warpAffine(dot_det, mov, (crop_size[1], crop_size[0]))
            det_weight_dict[(i, j)] = det_weight.copy()
            # img_det[i, j] = np.sum(det_weight * dot_exc)

            # utils.single_show(det_weight, "det_weight"+str((i,j)))
    # utils.single_show(img_det, "img_det")

    # 合成图像生成
    # 图像
    for idx in range(len(filepath_list)):
        # for idx in range(1):

        # 图像读取
        img = np.float64(cv2.imread(filepath_list[idx], cv2.IMREAD_UNCHANGED)) / 65535
        _, file_name = os.path.split(filepath_list[idx])
        file_name, _ = os.path.splitext(file_name)

        if show_steps:
            utils.single_show(img, "original image_" + file_name)

        # 裁剪图像
        cropped_img = img[img.shape[0] // 2 - crop_size[0] // 2:img.shape[0] // 2 + crop_size[0] // 2,
                      img.shape[1] // 2 - crop_size[1] // 2:img.shape[1] // 2 + crop_size[1] // 2]
        if show_steps:
            utils.single_show(cropped_img, "cropped_img_" + file_name)

        # 扩展维度
        extended_img = np.stack([cropped_img] * slice_num, axis=0)

        # 晶格照明
        latticed_img = extended_img * lattice
        if show_steps:
            utils.single_show(latticed_img, "latticed_img_" + file_name)

        # 检测矩阵检测
        img_det = simulate_detect(img_det_shape=tuple([slice_num] + list(target_size)), latticed_img=latticed_img,
                                  file_name=file_name, det_weight_dict=det_weight_dict)
        if show_steps:
            utils.single_show(img_det, "img_det_" + file_name)

        # 保存
        if save:
            result_name = os.path.join(result_folder, file_name) + '.tif'
            utils.save_tiff_3d(result_name, img_det)

    # 校准
    if save_lake:
        lake_img = lattice
        file_name = "lake"
        lake_det = simulate_detect(img_det_shape=tuple([slice_num] + list(target_size)), latticed_img=lake_img,
                                   file_name=file_name, det_weight_dict=det_weight_dict)
        if show_steps:
            utils.single_show(lake_det, "img_det_lake")
        result_name = os.path.join(result_folder, 'lake.tif')
        utils.save_tiff_3d(result_name, lake_det)

    # 背景
    if save_background:
        background_img = np.zeros(tuple([slice_num] + list(target_size)))

        if show_steps:
            utils.single_show(background_img, "img_det_background")
        result_name = os.path.join(result_folder, 'background.tif')
        utils.save_tiff_3d(result_name, background_img)

    return


def simulate_detect(img_det_shape, latticed_img, file_name, det_weight_dict):
    img_det = np.zeros(img_det_shape)
    for slice_num in tqdm(range(img_det_shape[0]), desc=Fore.LIGHTWHITE_EX + "Detecting image " + file_name,
                          bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.LIGHTWHITE_EX)):
        for i in range(img_det_shape[1]):
            for j in range(img_det_shape[2]):
                img_det[slice_num, i, j] = np.sum(det_weight_dict[(i, j)] * latticed_img[slice_num])
    img_det = img_det / np.max(img_det)

    return img_det


def simulate_degrade(
        image_shape, lattice_vectors, offset_vector, shift_vector, filepath_list, result_folder, illu_sigma,
        illu_sigma2, blur_sigma,
        illu_n_mean, illu_n_sigma, blur_n_mean, blur_n_sigma,
        show_steps=False, save=False, save_gt=False
):
    """
    实现晶格照明、模糊、压缩模拟数据集的生成（注意）

    :param image_shape:目标图像的像素形状*2
    :param lattice_vectors:晶格向量
    :param offset_vector:偏移向量
    :param shift_vector:位移向量
    :param filepath_list:文件名列表
    :param result_folder:输出文件夹
    :param illu_sigma: 照明点高斯核sigma
    :param blur_sigma: 模糊高斯核sigma
    :param show_steps:是否输出图片展示，debug
    :param save:是否保存图片
    :param save_gt: 是否保存ground truth
    :return:
    """
    image = np.zeros(image_shape)
    lattice = get_lattice_image(image, lattice_vectors, offset_vector, shift_vector, show=False)
    if show_steps:
        utils.single_show(lattice, "lattice location")

    lattice = simulate_blur_2d(lattice, sigma=illu_sigma, sigma2=illu_sigma2, noise_flag=utils.BLUR_GP,
                               mean_n=illu_n_mean,
                               sigma_n=illu_n_sigma, pad=10, pad_flag=utils.PAD_ZERO, flag=True)
    if show_steps:
        utils.single_show(lattice, "illumination lattice")

    # for i in range(len(filepath_list)):
    for i in range(3):

        # 图像读取
        img = cv2.imread(filepath_list[i], cv2.IMREAD_UNCHANGED) / 65535
        # img = np.zeros([256, 256])
        # img = np.ones([256, 256])
        if show_steps:
            utils.single_show(img, "original image")

        # 裁剪图像
        cropped_img = img[img.shape[0] // 2 - image_shape[1] // 2:img.shape[0] // 2 + image_shape[1] // 2,
                      img.shape[1] // 2 - image_shape[2] // 2:img.shape[1] // 2 + image_shape[2] // 2]
        if show_steps:
            utils.single_show(cropped_img, "cropped_img")

        # 扩展维度
        extended_img = np.stack([cropped_img] * 224, axis=0)

        # 晶格照明
        latticed_img = extended_img * lattice
        if show_steps:
            utils.single_show(latticed_img, "latticed_img")

        # 图像模糊
        blurred_img = simulate_blur_2d(latticed_img, sigma=blur_sigma, noise_flag=utils.BLUR_GP, mean_n=blur_n_mean,
                                       sigma_n=blur_n_sigma, pad=10, pad_flag=utils.PAD_ZERO)
        if show_steps:
            utils.single_show(blurred_img, "blurred_img")

        # 图像压缩
        compressed_img = compress_img(blurred_img, compress_rate=0.5)
        if show_steps:
            utils.single_show(compressed_img, "compressed_img")

        _, file_name = os.path.split(filepath_list[i])
        file_name, _ = os.path.splitext(file_name)
        # file_name = "background"
        # file_name = "lake"

        if save:
            if not os.path.isdir(result_folder):
                os.makedirs(result_folder)
            result_name = os.path.join(result_folder, file_name) + '.tiff'
            utils.save_tiff_3d(result_name, compressed_img / 1.9)

        if save_gt:
            gt_folder = os.path.join(result_folder, "ground truth")
            if not os.path.isdir(gt_folder):
                os.makedirs(gt_folder)
            gt_name = os.path.join(gt_folder, file_name) + '_gt.tiff'
            utils.save_tiff_2d(gt_name, cropped_img)

    return


def get_lattice_image(img, direct_lattice_vectors, offset_vector, shift_vector, show=False):
    """
    得到整个图像堆栈的晶格点位置图像

    :param img: 基图像
    :param direct_lattice_vectors: 晶格向量
    :param offset_vector: 偏移向量
    :param shift_vector: 位移向量
    :param show: 展示结果图像，debug
    :return:
    """
    if show:
        plt.figure()

    lattice = []

    for s in tqdm(range(img.shape[0]), desc=Fore.LIGHTWHITE_EX + "Generate lattice     ",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.LIGHTWHITE_EX)):
        if show:
            plt.clf()

        img_show = median_filter(np.array(img[s, :, :]), size=3)  # 3x3正方形窗口
        dots = np.zeros(img_show.shape)
        lattice_points = generate_lattice(img_show.shape, direct_lattice_vectors,
                                          center_pix=offset_vector + get_shift(shift_vector, s))
        for lp in lattice_points:
            x, y = np.round(lp).astype(int)
            dots[x, y] = 1
        lattice.append(dots)

        if show:
            plt.imshow(img_show, cmap='gray')
            plt.imshow(dots, cmap='gray')
            plt.title("White dots show the calculated illumination pattern")
            plt.show()
            m = input("Show the next one? y/[n]: ")
            if m != 'y':
                show = False

    return np.array(lattice)


def generate_lattice(image_shape, lattice_vectors, center_pix='image', edge_buffer=2, return_i_j=False):
    """
    根据晶格向量获得单张图片的照明点位置

    :param image_shape: 图像形状
    :param lattice_vectors: 晶格向量
    :param center_pix: 中心坐标，由偏移向量和位移向量在外部计算
    :param edge_buffer: 图像边缘的缓冲区域大小。避免照明点超出图像边界，默认为 2
    :param return_i_j: 是否返回 i 和 j
    :return:
    """
    # 修正判断
    lattice_shift = None
    if isinstance(center_pix, str):
        if center_pix == 'image':
            center_pix = np.array(image_shape) // 2
    else:
        center_pix = np.array(center_pix) - (np.array(image_shape) // 2)
        lattice_components = np.linalg.solve(np.vstack(lattice_vectors[:2]).T, center_pix)
        lattice_components_centered = np.mod(lattice_components, 1)
        lattice_shift = lattice_components - lattice_components_centered
        center_pix = (lattice_vectors[0] * lattice_components_centered[0] +
                      lattice_vectors[1] * lattice_components_centered[1] +
                      np.array(image_shape) // 2)

    num_vectors = int(np.round(1.5 * max(image_shape) / np.sqrt((lattice_vectors[0] ** 2).sum())))  # changed
    lower_bounds = (edge_buffer, edge_buffer)
    upper_bounds = (image_shape[0] - edge_buffer, image_shape[1] - edge_buffer)
    i, j = np.mgrid[-num_vectors:num_vectors, -num_vectors:num_vectors]
    i = i.reshape(i.size, 1)
    j = j.reshape(j.size, 1)
    lp = i * lattice_vectors[0] + j * lattice_vectors[1] + center_pix
    valid = np.all(lower_bounds < lp, 1) * np.all(lp < upper_bounds, 1)
    lattice_points = list(lp[valid])
    if return_i_j:
        return (lattice_points,
                list(i[valid] - lattice_shift[0]),
                list(j[valid] - lattice_shift[1]))
    else:
        return lattice_points


def get_shift(shift_vector, frame_number):
    """
    由位移向量与图像帧数得到位移向量

    :param shift_vector: 位移向量，字典，包括快轴向量、慢轴向量和扫描维数
    :param frame_number: 帧数
    :return: 相应帧数对应的位移向量
    """
    if isinstance(shift_vector, dict):
        """This means we have a 2D shift vector"""
        fast_steps = frame_number % shift_vector['scan_dimensions'][0]
        slow_steps = frame_number // shift_vector['scan_dimensions'][0]
        return (shift_vector['fast_axis'] * fast_steps +
                shift_vector['slow_axis'] * slow_steps)
    else:
        """This means we have a 1D shift vector"""
        return frame_number * shift_vector


def simulate_blur_2d2(image, psf, pad=0, pad_flag=utils.PAD_ZERO):
    """
    对二维图像或三维堆栈进行模拟模糊，卷积方法
    :param image:
    :param psf:
    :param pad:
    :param pad_flag:
    :return:
    """

    psf /= np.max(psf)
    # utils.single_show(psf, "psf")

    if len(image.shape) == 3:

        for j in tqdm(range(image.shape[0]), desc=Fore.LIGHTWHITE_EX + "Generate illumination",
                      bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.LIGHTWHITE_EX)):
            image[j] = utils.blur_2d(image[j], psf, noise_flag=utils.BLUR_ONLY, pad=pad, pad_flag=pad_flag)
    elif len(image.shape) == 2:
        image = utils.blur_2d(image, psf, noise_flag=utils.BLUR_ONLY, pad=pad, pad_flag=pad_flag)

    return image


def simulate_blur_2d(image, sigma, sigma2=10, noise_flag=utils.BLUR_ONLY, mean_n=0., sigma_n=0., pad=0,
                     pad_flag=utils.PAD_ZERO, flag=False):
    """
    对二维图像或三维堆栈进行模拟模糊
    :param image: 清晰图像
    :param sigma: 模糊核sigma
    :param noise_flag: 是否添加噪声：utils.BLUR_ONLY、utils.BLUR_GAUSS、BLUR_POISSON、BLUR_GP
    :param mean_n: 高斯噪声均值
    :param sigma_n: 高斯噪声标准差
    :param pad: pad宽度
    :param pad_flag: 填充方式
    :return: 模糊后图像
    """
    kernel_size = image.shape[1]
    psf = utils.create_2d_gaussian_kernel(kernel_size, sigma)

    if flag:
        psf2 = 2 * utils.create_2d_gaussian_kernel(kernel_size, sigma2)
        psf3 = utils.create_2d_gaussian_kernel(kernel_size, 0.4)
        psf4 = utils.create_2d_gaussian_kernel(kernel_size, 1.5)
        psf5 = 1.2 * utils.create_2d_gaussian_kernel(kernel_size, 4)
        psf6 = 1.5 * utils.create_2d_gaussian_kernel(kernel_size, 6)
        psf = np.maximum(psf, psf2, psf3)
        psf = np.maximum(psf, psf4, psf5)
        psf = np.maximum(psf, psf6)

    psf /= np.max(psf)
    utils.single_show(psf, "psf")

    if len(image.shape) == 3:
        for j in range(image.shape[0]):
            image[j] = utils.blur_2d(image[j], psf, noise_flag=noise_flag, mean=mean_n, sigma=sigma_n, pad=pad,
                                     pad_flag=pad_flag)
    elif len(image.shape) == 2:
        image = utils.blur_2d(image, psf, noise_flag=noise_flag, mean=mean_n, sigma=sigma_n, pad=pad, pad_flag=pad_flag)

    return image


def compress_img(img, compress_rate=0.5):
    """
    使用OpenCV对图像进行压缩的方法

    :param img: 输入图像
    :param compress_rate: 压缩率，默认值为0.5，表示将图像的长和宽都缩小到原来的0.5倍
    """
    # 获取图像的高度和宽度
    t = len(img.shape)
    height, width = img.shape[t - 2:t]

    img_resize = None
    # 使用双三次插值法对图像进行缩放，实现压缩效果
    if len(img.shape) == 3:
        img_resize = []
        for j in range(img.shape[0]):
            img_resize.append(cv2.resize(img[j], (int(height * compress_rate), int(width * compress_rate)),
                                         interpolation=cv2.INTER_AREA))
        img_resize = np.array(img_resize)
    elif len(img.shape) == 2:
        img_resize = cv2.resize(img, (int(height * compress_rate), int(width * compress_rate)),
                                interpolation=cv2.INTER_AREA)

    return img_resize
