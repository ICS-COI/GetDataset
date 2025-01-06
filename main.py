import os
import numpy as np
from utils import list_suffix_files
import simulate_degrade

if __name__ == '__main__':
    show_steps = True
    data_root = r'D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR'
    # ori_folder = os.path.join(data_root, r'F-actin/GT_all_a')
    ori_folder = os.path.join(data_root, r'MSIM')
    result_folder = os.path.join(data_root, r'result/MSIM_simulate_from_function')

    filepath_list = list_suffix_files(ori_folder, '.tiff')
    #
    # # 晶格参数
    # image_shape = [224, 256, 256]
    # # image = np.zeros(image_shape)
    # lattice_vectors = [np.array([17.92647318 * 2, - 0.13509819 * 2]), np.array([-9.13087358 * 2, - 15.53157036 * 2]),
    #                    np.array([-8.7955996 * 2, 15.66666855 * 2])]
    # offset_vector = np.array([55.07256157 * 2, 80.33416977 * 2])
    # shift_vector = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
    #                 'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}

    lattice_vectors = [np.array([18. * 2, 0.]), np.array([-9. * 2, - 15.58845727 * 2]),
                       np.array([-9. * 2, 15.58845727 * 2])]
    offset_vector = np.array([64. * 2, 64. * 2])
    shift_vector = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
                    'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}

    # illu_sigma = 2.5
    # illu_sigma2 = 8
    # illu_n_mean = 0
    # illu_n_sigma = 0
    # blur_sigma = 2
    # blur_n_mean = 0.1
    # blur_n_sigma = 0.01
    #
    # # 生成退化图像
    # simulate_degrade.simulate_degrade(
    #     image_shape=image_shape, lattice_vectors=lattice_vectors, offset_vector=offset_vector,
    #     shift_vector=shift_vector, filepath_list=filepath_list, result_folder=result_folder, illu_sigma=illu_sigma,
    #     illu_sigma2=illu_sigma2, blur_sigma=blur_sigma, illu_n_mean=illu_n_mean, illu_n_sigma=illu_n_sigma,
    #     blur_n_mean=blur_n_mean, blur_n_sigma=blur_n_sigma, show_steps=True, save=True, save_gt=True
    # )

    # 从成像原理出发生成模糊图像
    image = np.ones((256, 256))
    compress_rate = 0.5
    target_size = [128, 128]
    sigma_exc = 2
    sigma_det = 2
    simulate_degrade.imaging_process(compress_rate, target_size, sigma_exc, sigma_det, lattice_vectors=lattice_vectors,
                                     offset_vector=offset_vector, shift_vector=shift_vector,
                                     filepath_list=filepath_list, show_steps=show_steps)
