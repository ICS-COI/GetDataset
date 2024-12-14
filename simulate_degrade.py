import os
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import simulate_incoherent


def simulate_degrade(
        image_shape, lattice_vectors, offset_vector, shift_vector, filepath_list, result_folder, illu_params,
        blur_params,
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
    :param illu_params: 照明点高斯核sigma
    :param blur_params: 模糊高斯核sigma
    :param show_steps:是否输出图片展示，debug
    :param save:是否保存图片
    :param save_gt: 是否保存ground truth
    :return:
    """
    image = np.zeros(image_shape)
    lattice = get_lattice_image(image, lattice_vectors, offset_vector, shift_vector, show=False)
    if show_steps:
        utils.single_show(lattice, "lattice location")

    lattice = simulate_blur_2d(lattice, params=illu_params, pad=10, pad_flag=utils.PAD_ZERO)
    if show_steps:
        utils.single_show(lattice, "illumination lattice")

    for i in range(1):
        img = np.zeros([256, 256])
        # img = np.ones([256, 256])

        # for i in range(len(filepath_list)):
        # 图像读取
        # img = cv2.imread(filepath_list[i], cv2.IMREAD_UNCHANGED) / 65535
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
        blurred_img = simulate_blur_2d(latticed_img, params=blur_params, noise_flag=utils.BLUR_GP, mean_n=0.02,
                                       sigma_n=0.01, pad=10, pad_flag=utils.PAD_ZERO)
        if show_steps:
            utils.single_show(blurred_img, "blurred_img")

        # 图像压缩
        compressed_img = compress_img(blurred_img, compress_rate=0.5)
        if show_steps:
            utils.single_show(compressed_img, "compressed_img")

        _, file_name = os.path.split(filepath_list[i])
        file_name, _ = os.path.splitext(file_name)
        file_name = "background"
        # file_name = "lake"

        if save:
            if not os.path.isdir(result_folder):
                os.makedirs(result_folder)
            result_name = os.path.join(result_folder, file_name) + '.tiff'
            utils.save_tiff_3d(result_name, compressed_img)

        if save_gt:
            gt_folder = os.path.join(result_folder, "ground truth")
            if not os.path.isdir(gt_folder):
                os.makedirs(gt_folder)
            gt_name = os.path.join(gt_folder, file_name) + '_gt.tiff'
            utils.save_tiff_2d(gt_name, cropped_img)

    return


def simulate_degrade_from_lake(
        lake_path, filepath_list, result_folder, show_steps=False, save=False, save_gt=False
):
    _, lake = cv2.imreadmulti(lake_path, flags=cv2.IMREAD_UNCHANGED)
    lattice = np.float64(lake) / 65535
    if show_steps:
        utils.single_show(lattice, "lattice location")

    for i in range(1):
        # img = np.zeros([128, 128])
        # img = np.ones([128, 128])

        # for i in range(len(filepath_list)):
        # 图像读取
        img = cv2.imread(filepath_list[i], cv2.IMREAD_UNCHANGED) / 65535
        if show_steps:
            utils.single_show(img, "original image")

        # 裁剪图像
        cropped_img = img[img.shape[0] // 2 - lattice.shape[1] // 2:img.shape[0] // 2 + lattice.shape[1] // 2,
                      img.shape[1] // 2 - lattice.shape[2] // 2:img.shape[1] // 2 + lattice.shape[2] // 2]
        if show_steps:
            utils.single_show(cropped_img, "cropped_img")

        # 扩展维度
        extended_img = np.stack([cropped_img] * 224, axis=0)

        # 晶格照明
        latticed_img = extended_img * lattice
        if show_steps:
            utils.single_show(latticed_img, "latticed_img")

        # 图像模糊
        # blurred_img = simulate_blur_2d(latticed_img, params=blur_params, noise_flag=utils.BLUR_GP, mean_n=0.02,
        #                                sigma_n=0.01, pad=10, pad_flag=utils.PAD_ZERO)
        # if show_steps:
        #     utils.single_show(blurred_img, "blurred_img")
        #
        # # 图像压缩
        # compressed_img = compress_img(blurred_img, compress_rate=0.5)
        # if show_steps:
        #     utils.single_show(compressed_img, "compressed_img")

        _, file_name = os.path.split(filepath_list[i])
        file_name, _ = os.path.splitext(file_name)
        # file_name = "background"
        # file_name = "lake"

        if save:
            if not os.path.isdir(result_folder):
                os.makedirs(result_folder)
            result_name = os.path.join(result_folder, file_name) + '.tiff'
            utils.save_tiff_3d(result_name, latticed_img)

        if save_gt:
            gt_folder = os.path.join(result_folder, "ground truth")
            if not os.path.isdir(gt_folder):
                os.makedirs(gt_folder)
            gt_name = os.path.join(gt_folder, file_name) + '_gt.tiff'
            utils.save_tiff_2d(gt_name, cropped_img)

    return


def get_lattice_image(img, direct_lattice_vectors, offset_vector, shift_vector, show=True):
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
    for s in range(img.shape[0]):
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


def simulate_blur_2d(image, params, noise_flag=utils.BLUR_ONLY, mean_n=0., sigma_n=0., pad=0, pad_flag=utils.PAD_ZERO):
    """
    对二维图像或三维堆栈进行模拟模糊
    :param image: 清晰图像
    :param params: 模拟PSF生成参数
    :param noise_flag: 是否添加噪声：utils.BLUR_ONLY、utils.BLUR_GAUSS、BLUR_POISSON、BLUR_GP
    :param mean_n: 高斯噪声均值
    :param sigma_n: 高斯噪声标准差
    :param pad: pad宽度
    :param pad_flag: 填充方式
    :return: 模糊后图像
    """
    _, _, psf = simulate_incoherent.incoh_otf(image, params)
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
