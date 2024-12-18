import simulate_inco
import numpy as np
import cv2
import utils
from simulate_degrade import compress_img

# file_name = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/lake.tif"
# _,image = cv2.imreadmulti(file_name, flags=cv2.IMREAD_UNCHANGED)
# image = np.float64(image)
# image/=image.max()
# print(type(image),len(image))
# good_image=[]
# for i in range(len(image)):
#     good_image.append(compress_img(image[i],2))
#     good_image[i] -= np.mean(good_image[i])
#     good_image[i] = np.clip(good_image[i], 0, 1)
# # print(good_image.shape)
# good_image = np.array(good_image)
# utils.single_show(good_image,"good image")
# utils.save_tiff_3d("D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/lake_256.tiff",good_image)

# background = np.zeros((224,128,128))
# utils.save_tiff_3d("D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/background.tiff",background)

# file_name = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/lake_256.tiff"
# _,image = cv2.imreadmulti(file_name, flags=cv2.IMREAD_UNCHANGED)
# image = np.float64(image)
# image/=image.max()
# print(type(image),len(image))
# good_image=[]
# for i in range(len(image)):
#     good_image.append(compress_img(image[i],0.5))
#     # good_image[i] -= np.mean(good_image[i])
#     good_image[i] = np.clip(good_image[i], 0, 1)
# # print(good_image.shape)
# good_image = np.array(good_image)
# utils.single_show(good_image,"good image")
# utils.save_tiff_3d("D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/MSIM/lake_128nb.tiff",good_image)

# 扩展psf
# file_name = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake-psf2.tif"
# # file_name = r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_PSF,blur_sigma=1/tubules488.tiff"
# image = np.float64(cv2.imread(file_name, flags=cv2.IMREAD_UNCHANGED))
# image /= image.max()
# print(image.shape)
# p2d = np.array([118, 118, 118, 118])
# img_pad = np.pad(image, p2d.reshape((2, 2)), "constant")
# utils.single_show(img_pad,"pad psf")
# print(img_pad.shape)
#
# smooth = utils.create_2d_gaussian_kernel(256, 4)
# utils.single_show(smooth,"smooth")
#
# # psf = img_pad-np.mean(image)*0.95
# # psf = np.clip(psf, 0, 1)
# #
# psf = img_pad*smooth
# psf/=psf.max()
# utils.single_show(psf,"psf")
#
# otf = np.log(np.abs(np.fft.fftshift(np.fft.fft2(psf)))+1)
# utils.single_show(otf,"otf")
# utils.save_tiff_2d(r"D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR/result/MSIM_middle/lake-psf2562.tif",psf)


# x = np.zeros((224,256,256))
# m = tuple([0,3,156])
# x[m] = 1
# print(x)


background = np.zeros((224,128,128))
utils.save_tiff_3d(r"D:/Files/OneDrive - stu.hit.edu.cn/codes/python/MSIM-MPSS-tiff/data/241218-relocate/MSIM_reallocate_mask,radium=5,illu_sigma=4/background.tiff",background)