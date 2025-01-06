# import simulate_inco
import os
import cv2
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from colorama import Fore


#
# # import simulate_degrade
# # filename = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/AVG_lake.tif"
# # img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
# # img = np.array(img)
# # img2 = simulate_degrade.compress_img(img,2)
# # cv2.imwrite(r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/AVG_lake2.tif", img2)
#
# # image = np.ones((256, 128, 128))
# # dot_illu = np.zeros_like(image)
# # print(image,image.shape,image.dtype)
# # print(dot_illu,dot_illu.shape,dot_illu.dtype)
#
# image = np.zeros((256, 256))
# image[128:128 + 5, 128:128 + 5] = 1
# utils.single_show(image,"before")
#
# # 定义移动向量（x，y），这里示例向右移动20像素，向下移动10像素，可按需调整
# x = 60
# y = 10
#
# # 创建平移矩阵
# M = np.float32([[1, 0, x],
#                 [0, 1, y]])
#
# # 应用平移矩阵对图像进行平移变换
# moved_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# utils.single_show(moved_image,"after")


# shift_vector = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
#                 'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}
# slice_num = shift_vector['scan_dimensions'][0]*shift_vector['scan_dimensions'][1]
# print(shift_vector['scan_dimensions'][0])
# print(slice_num)
# crop_size = [256,256]
# image = np.ones(tuple([slice_num]+crop_size))
# print(image.shape)

def get_lattice_image(image_shape, direct_lattice_vectors, offset_vector, shift_vector, show=False):
    """
    得到整个图像堆栈的晶格点位置图像

    :param image_shape: 图像大小
    :param direct_lattice_vectors: 晶格向量
    :param offset_vector: 偏移向量
    :param shift_vector: 位移向量
    :param show: 展示结果图像，debug
    :return:
    """
    if show:
        plt.figure()

    slice_num = shift_vector['scan_dimensions'][0] * shift_vector['scan_dimensions'][1]
    lattice = []

    for s in tqdm(range(slice_num), desc=Fore.LIGHTWHITE_EX + "Generate lattice     ",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.LIGHTWHITE_EX)):
        if show:
            plt.clf()

        dots = np.zeros(tuple(image_shape))
        lattice_points = generate_lattice(image_shape, direct_lattice_vectors,
                                          center_pix=offset_vector + get_shift(shift_vector, s))
        for lp in lattice_points:
            x, y = np.round(lp).astype(int)
            dots[x, y] = 1
        lattice.append(dots)

        if show:
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


if __name__ == '__main__':
    lattice_vectors1 = [np.array([36., 0.]), np.array([-9. * 2, - 15.58845727 * 2]),
                        np.array([-9. * 2, 15.58845727 * 2])]
    offset_vector1 = np.array([540.,960.])
    shift_vector1 = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
                     'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}

    # DMD图像生成
    img_shape = [1080, 1920]
    lattice_stack = get_lattice_image(img_shape, lattice_vectors1, offset_vector1, shift_vector1, show=True)
    utils.save_tiff_3d("lattice_stack.tiff",lattice_stack)

