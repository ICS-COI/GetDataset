import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from colorama import Fore
import simulate_degrade


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

    for s in tqdm(range(slice_num), desc=Fore.LIGHTWHITE_EX + "Generate lattice",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.LIGHTWHITE_EX)):
        if show:
            plt.clf()

        dots = np.zeros(tuple(image_shape))
        lattice_points = simulate_degrade.generate_lattice(image_shape, direct_lattice_vectors,
                                                           center_pix=offset_vector + simulate_degrade.get_shift(
                                                               shift_vector, s))
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


if __name__ == '__main__':
    vector_basis = 24.
    detect_diameter = 5
    step_1 = 15
    scan_dimensions = (step_1, int(np.ceil(step_1 / 2. * np.sqrt(3))))

    lattice_vectors1 = [np.array([vector_basis, 0.]), np.array([-vector_basis / 2., -vector_basis / 2 * np.sqrt(3)]),
                        np.array([-vector_basis / 2., vector_basis / 2 * np.sqrt(3)])]
    offset_vector1 = np.array([540., 960.])
    shift_vector1 = {'fast_axis': np.array([vector_basis / scan_dimensions[0], 0.]), 'scan_dimensions': scan_dimensions,
                     'slow_axis': np.array([0., -vector_basis / 2 * np.sqrt(3) / scan_dimensions[1]])}

    # DMD图像生成
    img_shape = [1080, 1920]
    slice_num1 = shift_vector1['scan_dimensions'][0] * shift_vector1['scan_dimensions'][1]
    lattice_stack = get_lattice_image(img_shape, lattice_vectors1, offset_vector1, shift_vector1, show=False)
    utils.single_show(lattice_stack, "lattice_stack.tiff")
    # utils.save_tiff_3d("result/"+str(scan_dimensions)+"_"+str(vector_basis)+"_"+"0.tiff",lattice_stack)

    detect_all = []

    detect_idx = utils.get_circular_region_coordinates_diameter(0, 0, detect_diameter)
    for i in tqdm(range(len(lattice_stack)), desc=Fore.LIGHTWHITE_EX + "Generate dot    ",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.LIGHTWHITE_EX)):
        coordinates = np.argwhere(lattice_stack[i] == 1)
        location_idx = [tuple(coord) for coord in coordinates]
        for (x, y) in location_idx:
            for detected_idx in detect_idx:
                x_dot = int(x + detected_idx[0])
                y_dot = int(y + detected_idx[1])
                if x_dot < 0 or x_dot >= img_shape[0] or y_dot < 0 or y_dot >= img_shape[1]:
                    continue
                lattice_stack[i, x_dot, y_dot] = 1
    utils.single_show(lattice_stack, str(scan_dimensions) + "_" + str(vector_basis) + "_" + str(detect_diameter))
    utils.save_tiff_3d(
        "result/" + str(scan_dimensions) + "_" + str(vector_basis) + "_" + str(detect_diameter) + ".tiff",
        lattice_stack)
