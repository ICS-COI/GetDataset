import os
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


def simulate_degrade_reallocate(
        image_shape, lattice_vectors, offset_vector, shift_vector, filepath_list, result_folder, illu_sigma, blur_sigma,
        detect_radium=4,
        show_steps=False, save=False, save_gt=False, save_illu=False
):
    image = np.zeros(image_shape)
    image2 = np.zeros((224, 128, 128))

    # 每个晶格点位置
    lattice = get_lattice_image(image, lattice_vectors, offset_vector, shift_vector, show=False)
    if show_steps:
        utils.single_show(lattice, "lattice location")

    detect_all = []
    for i in range(len(lattice)):
        coordinates = np.argwhere(lattice[i] == 1)
        location_idx = [tuple(coord) for coord in coordinates]
        detected_i = []
        for (x, y) in location_idx:
            detect_idx = get_circular_region_coordinates_numpy(x, y, detect_radium, image[0].shape)
            circle_idx = []
            for detected_idx in detect_idx:
                show_idx = [detected_idx[0] - x // 2, detected_idx[1] - y // 2]
                if (0 <= detected_idx[0] < image.shape[1] and 0 <= detected_idx[1] < image.shape[2] and
                        0 <= show_idx[0] < image.shape[1] // 2 and 0 <= show_idx[1] < image.shape[2] // 2):
                    circle_idx.append((detected_idx, show_idx))
                    image[tuple([i] + detected_idx)] = 1
                    image2[tuple([i] + show_idx)] = 1
            detected_i.append({"original_idx": [x, y], "image_idx": [x // 2, y // 2], "circle_idx": circle_idx})
        detect_all.append(detected_i)

    if show_steps:
        utils.single_show(image, "lattice detected")
        utils.single_show(image2, "lattice detected2")

    # 高斯核照明掩膜（224*128*128）
    lattice_mask = simulate_blur_2d(lattice, sigma=illu_sigma, pad=10, pad_flag=utils.PAD_ZERO,
                                    noise_flag=utils.BLUR_ONLY)
    # lattice_path = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake_256.tiff"
    # _, lattice = cv2.imreadmulti(lattice_path, flags=cv2.IMREAD_UNCHANGED)
    lattice_mask = np.float64(lattice_mask)
    lattice_mask /= lattice_mask.max()

    lattice_show = compress_img(lattice, compress_rate=0.5)
    if show_steps:
        utils.single_show(lattice_show, "illumination lattice")
    # lattice_mask = simulate_blur_2d1(lattice, sigma=illu_sigma, pad=10, pad_flag=utils.PAD_ZERO, noise_flag=utils.BLUR_GAUSS,
    #                             mean_n=0, sigma_n=0.01)
    # print(lattice.shape, lattice.max(), lattice.min())
    # lattice_path = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake_256.tiff"
    # _, lattice = cv2.imreadmulti(lattice_path, flags=cv2.IMREAD_UNCHANGED)
    # lattice = np.float64(lattice)
    # lattice /= lattice.max()
    #
    # lattice_show = compress_img(lattice, compress_rate=0.5)
    # if show_steps:
    #     utils.single_show(lattice_show, "illumination lattice")
    #
    for i in range(2):
        # img = np.zeros([256, 256])
        # img = np.ones([256, 256])
        #
        #     # for i in range(len(filepath_list)):
        #     # 图像读取
        img = cv2.imread(filepath_list[i], cv2.IMREAD_UNCHANGED) / 65535
        if show_steps:
            utils.single_show(img, "original image")
        #
        # 裁剪图像
        cropped_img = img[img.shape[0] // 2 - image_shape[1] // 2:img.shape[0] // 2 + image_shape[1] // 2,
                      img.shape[1] // 2 - image_shape[2] // 2:img.shape[1] // 2 + image_shape[2] // 2]
        if show_steps:
            utils.single_show(cropped_img, "cropped_img")
        #
        # 扩展维度
        extended_img = np.stack([cropped_img] * 224, axis=0)
        reallocate_img = np.zeros((224, 128, 128))

        # 像素重分配
        for j in range(len(detect_all)):
            for circle in detect_all[j]:
                for (idx1, idx2) in circle["circle_idx"]:
                    reallocate_img[tuple([j] + idx2)] = extended_img[tuple([j] + idx1)]

        if show_steps:
            utils.single_show(reallocate_img, "reallocate_img")

        masked_img = reallocate_img * lattice_show
        if show_steps:
            utils.single_show(masked_img, "masked_img")

        #
        #     # 晶格照明
        #     latticed_img = extended_img * lattice
        #     if show_steps:
        #         utils.single_show(latticed_img, "latticed_img")
        #
        #     # 图像模糊
        #     blurred_img = simulate_blur_2d(latticed_img, sigma=blur_sigma, noise_flag=utils.BLUR_GP, mean_n=0.02,
        #                                    sigma_n=0.01, pad=10, pad_flag=utils.PAD_ZERO)
        #     if show_steps:
        #         utils.single_show(blurred_img, "blurred_img")
        #
        #     # 图像压缩
        #     compressed_img = compress_img(blurred_img, compress_rate=0.5)
        #     if show_steps:
        #         utils.single_show(compressed_img, "compressed_img")
        #
        _, file_name = os.path.split(filepath_list[i])
        file_name, _ = os.path.splitext(file_name)
        # file_name = "background"
        # file_name = "lake"
        #
        if save:
            if not os.path.isdir(result_folder):
                os.makedirs(result_folder)
            result_name = os.path.join(result_folder, file_name) + '.tiff'
            utils.save_tiff_3d(result_name, masked_img)

    #     if save_gt:
    #         gt_folder = os.path.join(result_folder, "ground truth")
    #         if not os.path.isdir(gt_folder):
    #             os.makedirs(gt_folder)
    #         gt_name = os.path.join(gt_folder, file_name) + '_gt.tiff'
    #         utils.save_tiff_2d(gt_name, cropped_img)
    #
    #     if save_illu:
    #         illu_folder = os.path.join(result_folder, "illumination")
    #         if not os.path.isdir(illu_folder):
    #             os.makedirs(illu_folder)
    #         illu_name = os.path.join(illu_folder, file_name) + '_illu.tiff'
    #         utils.save_tiff_3d(illu_name, lattice_show)

    return

# 衰减扩散照明模式(探寻自然的秘密)
def simulate_degrade_spread(
        image_shape, lattice_vectors, offset_vector, shift_vector, filepath_list, result_folder, illu_sigma, blur_sigma,
        detect_radium=10,
        show_steps=False, save=False, save_gt=False, save_illu=False
):
    image = np.zeros(image_shape)
    image2 = np.zeros((224, 128, 128))

    # 每个晶格点位置
    lattice = get_lattice_image(image, lattice_vectors, offset_vector, shift_vector, show=False)
    if show_steps:
        utils.single_show(lattice, "lattice location")

    detect_all = []
    for i in range(len(lattice)):
        coordinates = np.argwhere(lattice[i] == 1)
        location_idx = [tuple(coord) for coord in coordinates]
        detected_i = []
        for (x, y) in location_idx:
            detect_idx = get_circular_region_coordinates_numpy(x, y, detect_radium, image[0].shape)
            circle_idx = []
            for detected_idx in detect_idx:
                show_idx = [detected_idx[0] - x // 2, detected_idx[1] - y // 2]
                if (0 <= detected_idx[0] < image.shape[1] and 0 <= detected_idx[1] < image.shape[2] and
                        0 <= show_idx[0] < image.shape[1] // 2 and 0 <= show_idx[1] < image.shape[2] // 2):
                    circle_idx.append((detected_idx, show_idx))
                    image[tuple([i] + detected_idx)] = 1
                    image2[tuple([i] + show_idx)] = 1
            detected_i.append({"original_idx": [x, y], "image_idx": [x // 2, y // 2], "circle_idx": circle_idx})
        detect_all.append(detected_i)

    if show_steps:
        utils.single_show(image, "lattice detected")

    # # 高斯核照明掩膜（224*128*128）
    # lattice_mask = simulate_blur_2d(lattice, sigma=illu_sigma, pad=10, pad_flag=utils.PAD_ZERO,
    #                                 noise_flag=utils.BLUR_ONLY)
    # # lattice_path = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake_256.tiff"
    # # _, lattice = cv2.imreadmulti(lattice_path, flags=cv2.IMREAD_UNCHANGED)
    # lattice_mask = np.float64(lattice_mask)
    # lattice_mask /= lattice_mask.max()
    #
    # lattice_show = compress_img(lattice, compress_rate=0.5)
    # if show_steps:
    #     utils.single_show(lattice_show, "illumination lattice")
    # # lattice_mask = simulate_blur_2d1(lattice, sigma=illu_sigma, pad=10, pad_flag=utils.PAD_ZERO, noise_flag=utils.BLUR_GAUSS,
    # #                             mean_n=0, sigma_n=0.01)
    # # print(lattice.shape, lattice.max(), lattice.min())
    # # lattice_path = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake_256.tiff"
    # # _, lattice = cv2.imreadmulti(lattice_path, flags=cv2.IMREAD_UNCHANGED)
    # # lattice = np.float64(lattice)
    # # lattice /= lattice.max()
    # #
    # # lattice_show = compress_img(lattice, compress_rate=0.5)
    # # if show_steps:
    # #     utils.single_show(lattice_show, "illumination lattice")
    # #
    # for i in range(2):
    #     # img = np.zeros([256, 256])
    #     # img = np.ones([256, 256])
    #     #
    #     #     # for i in range(len(filepath_list)):
    #     #     # 图像读取
    #     img = cv2.imread(filepath_list[i], cv2.IMREAD_UNCHANGED) / 65535
    #     if show_steps:
    #         utils.single_show(img, "original image")
    #     #
    #     # 裁剪图像
    #     cropped_img = img[img.shape[0] // 2 - image_shape[1] // 2:img.shape[0] // 2 + image_shape[1] // 2,
    #                   img.shape[1] // 2 - image_shape[2] // 2:img.shape[1] // 2 + image_shape[2] // 2]
    #     if show_steps:
    #         utils.single_show(cropped_img, "cropped_img")
    #     #
    #     # 扩展维度
    #     extended_img = np.stack([cropped_img] * 224, axis=0)
    #     reallocate_img = np.zeros((224, 128, 128))
    #
    #     # 像素重分配
    #     for j in range(len(detect_all)):
    #         for circle in detect_all[j]:
    #             for (idx1, idx2) in circle["circle_idx"]:
    #                 reallocate_img[tuple([j] + idx2)] = extended_img[tuple([j] + idx1)]
    #
    #     if show_steps:
    #         utils.single_show(reallocate_img, "reallocate_img")
    #
    #     masked_img = reallocate_img * lattice_show
    #     if show_steps:
    #         utils.single_show(masked_img, "masked_img")
    #
    #     #
    #     #     # 晶格照明
    #     #     latticed_img = extended_img * lattice
    #     #     if show_steps:
    #     #         utils.single_show(latticed_img, "latticed_img")
    #     #
    #     #     # 图像模糊
    #     #     blurred_img = simulate_blur_2d(latticed_img, sigma=blur_sigma, noise_flag=utils.BLUR_GP, mean_n=0.02,
    #     #                                    sigma_n=0.01, pad=10, pad_flag=utils.PAD_ZERO)
    #     #     if show_steps:
    #     #         utils.single_show(blurred_img, "blurred_img")
    #     #
    #     #     # 图像压缩
    #     #     compressed_img = compress_img(blurred_img, compress_rate=0.5)
    #     #     if show_steps:
    #     #         utils.single_show(compressed_img, "compressed_img")
    #     #
    #     _, file_name = os.path.split(filepath_list[i])
    #     file_name, _ = os.path.splitext(file_name)
    #     # file_name = "background"
    #     # file_name = "lake"
    #     #
    #     if save:
    #         if not os.path.isdir(result_folder):
    #             os.makedirs(result_folder)
    #         result_name = os.path.join(result_folder, file_name) + '.tiff'
    #         utils.save_tiff_3d(result_name, masked_img)
    #
    # #     if save_gt:
    # #         gt_folder = os.path.join(result_folder, "ground truth")
    # #         if not os.path.isdir(gt_folder):
    # #             os.makedirs(gt_folder)
    # #         gt_name = os.path.join(gt_folder, file_name) + '_gt.tiff'
    # #         utils.save_tiff_2d(gt_name, cropped_img)
    # #
    # #     if save_illu:
    # #         illu_folder = os.path.join(result_folder, "illumination")
    # #         if not os.path.isdir(illu_folder):
    # #             os.makedirs(illu_folder)
    # #         illu_name = os.path.join(illu_folder, file_name) + '_illu.tiff'
    # #         utils.save_tiff_3d(illu_name, lattice_show)

    return


def get_circular_region_coordinates_numpy(center_x, center_y, radius, image_shape):
    """
    使用numpy获取以(center_x, center_y)为中心，radius为半径的圆形区域内所有像素点坐标，
    并根据输入的图像宽度和高度对坐标进行限制，确保都在图像范围内
    :param center_x: 中心坐标的x值
    :param center_y: 中心坐标的y值
    :param radius: 圆形区域的半径
    :param image_shape: 图像形状
    :return: 圆形区域内像素点坐标列表，每个元素为一个二元组 (x, y)
    """
    # 确定横坐标（x）的有效范围，限制在图像宽度内
    x_min = max(center_x - radius, 0)
    x_max = min(center_x + radius, image_shape[0] - 1)
    x = np.arange(x_min, x_max + 1)

    # 确定纵坐标（y）的有效范围，限制在图像高度内
    y_min = max(center_y - radius, 0)
    y_max = min(center_y + radius, image_shape[1] - 1)
    y = np.arange(y_min, y_max + 1)

    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    mask = distance <= radius
    coordinates = np.column_stack((xx[mask].ravel(), yy[mask].ravel())).tolist()
    return coordinates


# def simulate_degrade(
#         image_shape, lattice_vectors, offset_vector, shift_vector, filepath_list, result_folder, illu_sigma, blur_sigma,
#         show_steps=False, save=False, save_gt=False, save_illu=False
# ):
#     """
#     实现晶格照明、模糊、压缩模拟数据集的生成（注意）
#
#     :param image_shape:目标图像的像素形状*2
#     :param lattice_vectors:晶格向量
#     :param offset_vector:偏移向量
#     :param shift_vector:位移向量
#     :param filepath_list:文件名列表
#     :param result_folder:输出文件夹
#     :param illu_sigma: 照明点高斯核sigma
#     :param blur_sigma: 模糊高斯核sigma
#     :param show_steps:是否输出图片展示，debug
#     :param save:是否保存图片
#     :param save_gt: 是否保存ground truth
#     :param save_illu: 是否保存照明图像
#     :return:
#     """
#     image = np.zeros(image_shape)
#     lattice = get_lattice_image(image, lattice_vectors, offset_vector, shift_vector, show=False)
#     if show_steps:
#         utils.single_show(lattice, "lattice location")
#
#     lattice = simulate_blur_2d1(lattice, sigma=illu_sigma, pad=10, pad_flag=utils.PAD_ZERO, noise_flag=utils.BLUR_GAUSS,
#                                 mean_n=0, sigma_n=0.01)
#     print(lattice.shape, lattice.max(), lattice.min())
#     lattice_path = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake_256.tiff"
#     _, lattice = cv2.imreadmulti(lattice_path, flags=cv2.IMREAD_UNCHANGED)
#     lattice = np.float64(lattice)
#     lattice /= lattice.max()
#
#     lattice_show = compress_img(lattice, compress_rate=0.5)
#     if show_steps:
#         utils.single_show(lattice_show, "illumination lattice")
#
#     for i in range(1):
#         # img = np.zeros([256, 256])
#         # img = np.ones([256, 256])
#
#         # for i in range(len(filepath_list)):
#         # 图像读取
#         img = cv2.imread(filepath_list[i], cv2.IMREAD_UNCHANGED) / 65535
#         if show_steps:
#             utils.single_show(img, "original image")
#
#         # 裁剪图像
#         cropped_img = img[img.shape[0] // 2 - image_shape[1] // 2:img.shape[0] // 2 + image_shape[1] // 2,
#                       img.shape[1] // 2 - image_shape[2] // 2:img.shape[1] // 2 + image_shape[2] // 2]
#         if show_steps:
#             utils.single_show(cropped_img, "cropped_img")
#
#         # 扩展维度
#         extended_img = np.stack([cropped_img] * 224, axis=0)
#
#         # 晶格照明
#         latticed_img = extended_img * lattice
#         if show_steps:
#             utils.single_show(latticed_img, "latticed_img")
#
#         # 图像模糊
#         blurred_img = simulate_blur_2d(latticed_img, sigma=blur_sigma, noise_flag=utils.BLUR_GP, mean_n=0.02,
#                                        sigma_n=0.01, pad=10, pad_flag=utils.PAD_ZERO)
#         if show_steps:
#             utils.single_show(blurred_img, "blurred_img")
#
#         # 图像压缩
#         compressed_img = compress_img(blurred_img, compress_rate=0.5)
#         if show_steps:
#             utils.single_show(compressed_img, "compressed_img")
#
#         _, file_name = os.path.split(filepath_list[i])
#         file_name, _ = os.path.splitext(file_name)
#         # file_name = "background"
#         # file_name = "lake"
#
#         if save:
#             if not os.path.isdir(result_folder):
#                 os.makedirs(result_folder)
#             result_name = os.path.join(result_folder, file_name) + '.tiff'
#             utils.save_tiff_3d(result_name, compressed_img)
#
#         if save_gt:
#             gt_folder = os.path.join(result_folder, "ground truth")
#             if not os.path.isdir(gt_folder):
#                 os.makedirs(gt_folder)
#             gt_name = os.path.join(gt_folder, file_name) + '_gt.tiff'
#             utils.save_tiff_2d(gt_name, cropped_img)
#
#         if save_illu:
#             illu_folder = os.path.join(result_folder, "illumination")
#             if not os.path.isdir(illu_folder):
#                 os.makedirs(illu_folder)
#             illu_name = os.path.join(illu_folder, file_name) + '_illu.tiff'
#             utils.save_tiff_3d(illu_name, lattice_show)
#
#     return


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


def simulate_blur_2d(image, sigma, noise_flag=utils.BLUR_ONLY, mean_n=0., sigma_n=0., pad=0, pad_flag=utils.PAD_ZERO):
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
    # psf = cv2.imread("D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake-psf256.tiff",cv2.IMREAD_UNCHANGED)
    psf /= np.max(psf)
    utils.single_show(psf, "psf")

    if len(image.shape) == 3:
        for j in range(image.shape[0]):
            image[j] = utils.blur_2d(image[j], psf, noise_flag=noise_flag, mean=mean_n, sigma=sigma_n, pad=pad,
                                     pad_flag=pad_flag)
    elif len(image.shape) == 2:
        image = utils.blur_2d(image, psf, noise_flag=noise_flag, mean=mean_n, sigma=sigma_n, pad=pad, pad_flag=pad_flag)

    return image


def simulate_blur_2d1(image, sigma, noise_flag=utils.BLUR_ONLY, mean_n=0., sigma_n=0., pad=0, pad_flag=utils.PAD_ZERO):
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
    # psf = utils.create_2d_gaussian_kernel(kernel_size, sigma)
    psf = np.float64(cv2.imread("D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake-psf2562.tif",
                                cv2.IMREAD_UNCHANGED))
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
