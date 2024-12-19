import os
import numpy as np
from utils import list_suffix_files
import simulate_degrade

if __name__ == '__main__':
    show_steps = True
    data_root = r'D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR'
    # ori_folder = os.path.join(data_root, r'F-actin/GT_all_a')
    ori_folder = os.path.join(data_root, r'MSIM')
    result_folder = os.path.join(data_root, r'result/mimic_MSIM_0.4+1.5+2.5+4+6+8_2')

    filepath_list = list_suffix_files(ori_folder, '.tiff')

    # 晶格参数
    image_shape = [224, 256, 256]
    image = np.zeros(image_shape)
    lattice_vectors = [np.array([17.92647318 * 2, - 0.13509819 * 2]), np.array([-9.13087358 * 2, - 15.53157036 * 2]),
                       np.array([-8.7955996 * 2, 15.66666855 * 2])]
    offset_vector = np.array([55.07256157 * 2, 80.33416977 * 2])
    shift_vector = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
                    'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}

    # lattice_vectors = [np.array([18. * 2, 0.]), np.array([-9. * 2, - 15.58845727 * 2]),
    #                    np.array([-9. * 2, 15.58845727 * 2])]
    # offset_vector = np.array([64. * 2, 64. * 2])
    # shift_vector = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
    #                 'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}

    illu_sigma = 2.5
    illu_sigma2 = 8
    illu_n_mean = 0.65
    illu_n_sigma = 0.04
    blur_sigma = 2
    blur_n_mean = 0.65
    blur_n_sigma = 0.03

    # 生成退化图像
    simulate_degrade.simulate_degrade(
        image_shape=image_shape, lattice_vectors=lattice_vectors, offset_vector=offset_vector,
        shift_vector=shift_vector, filepath_list=filepath_list, result_folder=result_folder, illu_sigma=illu_sigma,
        illu_sigma2=illu_sigma2, blur_sigma=blur_sigma, illu_n_mean=illu_n_mean, illu_n_sigma=illu_n_sigma,
        blur_n_mean=blur_n_mean, blur_n_sigma=blur_n_sigma, show_steps=True, save=True, save_gt=True
    )
