import os
import numpy as np
from utils import list_suffix_files
import simulate_degrade

if __name__ == '__main__':
    show_steps = True
    data_root = r'data'
    # ori_folder = os.path.join(data_root, r'F-actin/GT_all_a')
    ori_folder = os.path.join(data_root, r'MSIM')
    # result_folder = os.path.join(data_root, result/MSIM_simulate_from_function')

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

    # lattice_vectors = [np.array([18. * 2, 0.]), np.array([-9. * 2, - 15.58845727 * 2]),
    #                    np.array([-9. * 2, 15.58845727 * 2])]
    # offset_vector = np.array([64. * 2, 64. * 2])
    # shift_vector = {'fast_axis': np.array([1.12539078 * 2, -0.00991981 * 2]), 'scan_dimensions': (16, 14),
    #                 'slow_axis': np.array([-0.02036928 * 2, -1.1154118 * 2])}

    vector_basis = 30.
    scan_dimension_1 = 15
    scan_dimensions = (scan_dimension_1, int(np.ceil(scan_dimension_1 / 2. * np.sqrt(3))))
    print(scan_dimensions)

    lattice_vectors = [np.array([vector_basis, 0.]), np.array([-vector_basis / 2., -vector_basis / 2 * np.sqrt(3)]),
                       np.array([-vector_basis / 2., vector_basis / 2 * np.sqrt(3)])]
    offset_vector = np.array([128., 128.])
    shift_vector = {'fast_axis': np.array([vector_basis / scan_dimensions[0], 0.]), 'scan_dimensions': scan_dimensions,
                    'slow_axis': np.array([0., -vector_basis / 2 * np.sqrt(3) / scan_dimensions[1]])}

    # # DMD图像生成
    # img_shape = [1080, 1920]
    # slice_num1 = shift_vector1['scan_dimensions'][0] * shift_vector1['scan_dimensions'][1]
    # lattice_stack = get_lattice_image(img_shape, lattice_vectors1, offset_vector1, shift_vector1, show=False)
    # utils.single_show(lattice_stack, "lattice_stack.tiff")
    # utils.save_tiff_3d("result/" + str(scan_dimensions) + "_" + str(vector_basis) + "_" + "0.tiff", lattice_stack)

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
    result_folder = os.path.join(data_root, 'result/MSIM_simulate_from_function_' + str(vector_basis) + "_" + str(
        scan_dimensions) + "_" + str(sigma_exc) + "_" + str(sigma_det))

    simulate_degrade.imaging_process(compress_rate, target_size, sigma_exc, sigma_det, lattice_vectors=lattice_vectors,
                                     offset_vector=offset_vector, shift_vector=shift_vector,
                                     filepath_list=filepath_list, show_steps=show_steps, result_folder=result_folder,
                                     save=True, save_lake=True, save_background=True, save_lattice=True,
                                     save_illumination=True)
