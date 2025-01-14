import os
import numpy as np
from utils import list_suffix_files
import simulate_degrade

if __name__ == '__main__':
    show_steps = True
    data_root = r'data'
    ori_folder = os.path.join(data_root, r'MSIM')

    filepath_list = list_suffix_files(ori_folder, '.tiff')

    image = np.ones((256, 256))
    compress_rate = 0.5
    target_size = [128, 128]
    sigma_exc = 2
    sigma_det = 2

    params = {
        'length': 1.3e-3,
        'lambda': 555e-9,
        'wxp': 1.5e-3,
        'zxp': 123.8e-3,
        'm': 10,
    }

    vector_basis = 30.
    scan_dimension_1 = 15
    scan_dimensions = (scan_dimension_1, int(np.ceil(scan_dimension_1 / 2. * np.sqrt(3))))
    print("Scan dimensions:",scan_dimensions)

    lattice_vectors = [np.array([vector_basis, 0.]), np.array([-vector_basis / 2., -vector_basis / 2 * np.sqrt(3)]),
                       np.array([-vector_basis / 2., vector_basis / 2 * np.sqrt(3)])]
    offset_vector = np.array([128., 128.])
    shift_vector = {'fast_axis': np.array([vector_basis / scan_dimensions[0], 0.]), 'scan_dimensions': scan_dimensions,
                    'slow_axis': np.array([0., -vector_basis / 2 * np.sqrt(3) / scan_dimensions[1]])}

    result_folder = os.path.join(data_root, 'result/MSIM_function_incoherent' + str(vector_basis) + "_" + str(
        scan_dimensions) + "_w=" + str(params['wxp']) + "_" + str(sigma_det))

    simulate_degrade.imaging_process(

        compress_rate=compress_rate,
        target_size=target_size,
        lattice_vectors=lattice_vectors,
        offset_vector=offset_vector,
        shift_vector=shift_vector,
        psf_mode=simulate_degrade.SIMULATE_PSF,
        simulate_params=params,
        sigma_exc=sigma_exc,
        sigma_det=sigma_det,
        filepath_list=filepath_list,
        show_steps=show_steps,
        result_folder=result_folder,
        save=True,
        save_lake=True,
        save_background=True,
        save_lattice=True,
        save_illumination=True
    )
