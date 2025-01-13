# import simulate_inco
import os
import cv2
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from colorama import Fore

# import simulate_degrade
#
# filename = r"D:\Files\OneDrive - stu.hit.edu.cn\codes\python\MSIM-MPSS-tiff\data\250107-real_no_lake\r=3-1024.tif"
# _, img = cv2.imreadmulti(filename, flags=cv2.IMREAD_UNCHANGED)
# img = np.float64(img)
# img /= np.max(img)
# print(img.shape)
# img2 = []
# for i in range(len(img)):
#     img2.append(simulate_degrade.compress_img(img[i], 0.125))
#
# img2 = np.array(img2)
# utils.save_tiff_3d(
#     r"D:\Files\OneDrive - stu.hit.edu.cn\codes\python\MSIM-MPSS-tiff\data\250107-real_no_lake\r=3-128.tif", img2)

image = -1.*np.ones((256, 128, 128))
stat = cv2.imwritemulti(r"D:\Files\OneDrive - stu.hit.edu.cn\codes\python\MSIM-MPSS-tiff\data\250107-real_no_lake\background.tif", tuple(image),
                        (int(cv2.IMWRITE_TIFF_RESUNIT), 1, int(cv2.IMWRITE_TIFF_COMPRESSION), 1))
if stat:
    print("Successfully save", r"D:\Files\OneDrive - stu.hit.edu.cn\codes\python\MSIM-MPSS-tiff\data\250107-real_no_lake\background.tif", "!")

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
