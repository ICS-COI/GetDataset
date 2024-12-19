import os
import numpy as np
from utils import list_suffix_files
import simulate_degrade

if __name__ == '__main__':
    show_steps = True
    data_root = r'D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR'
    # ori_folder = os.path.join(data_root, r'F-actin/GT_all_a')
    ori_folder = os.path.join(data_root, r'MSIM')
    result_folder = os.path.join(data_root, r'result/MSIM_reallocate_mask,radium=5,illu_sigma=4')

    filepath_list = list_suffix_files(ori_folder, '.tiff')

    # 晶格参数
    image_shape = [224, 256, 256]
    image = np.zeros(image_shape)
    lattice_vectors = [np.array([17.92647318 * 2, - 0.13509819 * 2]), np.array([-9.13087358 * 2, - 15.53157036 * 2]),
                       np.array([-8.7955996 * 2, 15.66666855 * 2])]
    offset_vector = np.array([55.07256157 * 2, 80.33416977 * 2])
    shift_vector = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
                    'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}
    illu_sigma = 4
    blur_sigma = 2

    # 生成退化图像
    # simulate_degrade.simulate_degrade(
    #     image_shape=image_shape, lattice_vectors=lattice_vectors, offset_vector=offset_vector,
    #     shift_vector=shift_vector, filepath_list=filepath_list, result_folder=result_folder, illu_sigma=illu_sigma,
    #     blur_sigma=blur_sigma, show_steps=True, save=True, save_gt=True, save_illu = True
    # )

    # # 像素重分配图像
    # simulate_degrade.simulate_degrade_reallocate(
    #     image_shape=image_shape, lattice_vectors=lattice_vectors, offset_vector=offset_vector,
    #     shift_vector=shift_vector, filepath_list=filepath_list, result_folder=result_folder, illu_sigma=illu_sigma,
    #     blur_sigma=blur_sigma, detect_radium=5,show_steps=True, save=True, save_gt=True, save_illu=True
    # )

    # 探寻自然的秘密
    simulate_degrade.simulate_degrade_spread(
        image_shape=image_shape, lattice_vectors=lattice_vectors, offset_vector=offset_vector,
        shift_vector=shift_vector, filepath_list=filepath_list, result_folder=result_folder, illu_sigma=illu_sigma,
        blur_sigma=blur_sigma, detect_radium=10, show_steps=True, save=True, save_gt=True, save_illu=True
    )
