import os
import numpy as np
from utils import list_suffix_files
import simulate_degrade

if __name__ == '__main__':
    show_steps = True
    data_root = r'D:/Files/OneDrive - stu.hit.edu.cn/Dataset/BioSR'
    ori_folder = os.path.join(data_root, r'F-actin/GT_all_a')
    result_folder = os.path.join(data_root, r'result/F-actin_3_3')

    filepath_list = list_suffix_files(ori_folder, '.tiff')

    # 晶格参数
    image_shape = [224, 256, 256]
    image = np.zeros(image_shape)
    lattice_vectors = [np.array([17.92647318 * 2, - 0.13509819 * 2]), np.array([-9.13087358 * 2, - 15.53157036 * 2]),
                       np.array([-8.7955996 * 2, 15.66666855 * 2])]
    offset_vector = np.array([55.07256157 * 2, 80.33416977 * 2])
    shift_vector = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
                    'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}
    # illu_sigma = 2.5
    # blur_sigma = 2

    params = {
        'length': 1.3e-3,
        'lambda': 555e-9,
        'wxp': 1e-3,
        'zxp': 123.8e-3,
        'm': 10,
    }

    # 生成退化图像
    simulate_degrade.simulate_degrade(
        image_shape=image_shape, lattice_vectors=lattice_vectors, offset_vector=offset_vector,
        shift_vector=shift_vector, filepath_list=filepath_list, result_folder=result_folder, illu_params=params,
        blur_params=params, show_steps=True, save=True, save_gt=True
    )
