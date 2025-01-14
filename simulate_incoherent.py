import cv2
import numpy as np
import torch
import math

incoherent_transform_params_hr = {
    'length': 1.3e-3,
    'lambda': 555e-9,
    'wxp': 4.5e-3,
    'zxp': 21.4e-3,
    'm': 20,
}
incoherent_transform_params_lr = {
    'length': 1.3e-3,
    'lambda': 555e-9,
    'wxp': 3.9e-3,
    'zxp': 123.8e-3,
    'm': 10,
}
downsample_factor = (incoherent_transform_params_hr['m'] // incoherent_transform_params_lr['m'])


def square(img):
    """
    从中间裁剪图像为正方形
    :param img: ndarray
    :return:
    """
    width, height = img.shape[:2]
    if width != height:
        # 计算裁剪的边长
        size = min(width, height)
        # 计算裁剪的起始位置
        start_row = int((width - size) / 2)
        start_col = int((height - size) / 2)
        # 裁剪图像为方形
        img_square = img[start_row:start_row + size, start_col:start_col + size]
    else:
        img_square = img
    return img_square


def incoh_otf(img, params):
    """
    计算圆形光学传递函数OTF
    ref. voelz2011computationalfourier chapter7 Imaging and DiffractionLimited Imaging Simulation

    :param img: 待模拟的图像，必须为方形， ranges [0,1]，dtype=float32，shape=(H,W,C)
    :param params: 模拟参数，包括图像物理边长，波长，出口瞳孔半径，出口瞳孔距离
    :return: 传递函数
    """
    width, height = img.shape  # 获取图像样本大小
    length_img = params['length']
    lambda_light = params['lambda']  # 波长
    wxp = params['wxp']  # 出口瞳孔半径
    zxp = params['zxp']  # 出口瞳孔距离
    du = length_img / height  # 采样间隔（米）
    dv = length_img / width
    f0 = wxp / (lambda_light * zxp)  # 相干截止频率

    fu = np.arange(-1 / (2 * du), 1 / (2 * du), 1 / length_img)  # 频率坐标
    fv = np.arange(-1 / (2 * dv), 1 / (2 * dv), 1 / length_img)
    fu_mesh, fv_mesh = np.meshgrid(fu, fv)

    ctf = np.double(np.sqrt(fu_mesh ** 2 + fv_mesh ** 2) / f0 <= 1)  # 圆形传递函数

    otf = np.fft.ifft2(np.abs(np.fft.fft2(np.fft.fftshift(ctf))) ** 2)
    otf = np.abs(otf / otf[0, 0])

    return otf


def incoh_sample(img_hr, params_hr, params_lr):
    """
    对高倍镜图像进行下采样，降低分辨率到低倍镜图像的分辨率
    """
    downsample = 'basic'
    # downsample = 'bilinear'

    m_hr = params_hr['m']
    m_lr = params_lr['m']
    sampling_ratio = math.ceil(m_hr / m_lr)
    if downsample == 'basic':
        img_sampled = img_hr[::sampling_ratio, ::sampling_ratio]
    elif downsample == 'bilinear':
        if type(img_hr) is np.ndarray:
            img_sampled = cv2.resize(img_hr, (img_hr.shape[1] // sampling_ratio, img_hr.shape[0] // sampling_ratio),
                                     interpolation=cv2.INTER_LINEAR)
        elif type(img_hr) is torch.Tensor:
            img_sampled = torch.nn.functional.interpolate(
                img_hr.unsqueeze(0).unsqueeze(0), scale_factor=1 / sampling_ratio, mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            raise ValueError('Invalid image type')
    else:
        raise ValueError('Invalid downsample method')
    return img_sampled


def incoh_imaging(img, params):
    """
    对模拟的理想图像进行非相干模拟成像
    """
    tensor_flag = False
    trans_flag = False
    if isinstance(img, torch.Tensor):
        tensor_flag = True
        img = img.detach().cpu().numpy()  # 转换为NumPy数组
        if img.shape[0] < 6:
            trans_flag = True
            img = np.transpose(img, (1, 2, 0))  # 将通道维度置于最后

    i_g = np.real(img).astype(float)  # 整数转浮点数
    i_g = i_g / np.max(i_g)  # 归一化理想图像
    otf = incoh_otf(i_g, params)  # 计算传递函数

    g_g = np.fft.fft2(np.fft.fftshift(i_g))  # 卷积
    g_i = g_g * otf
    i_i = np.fft.ifftshift(np.fft.ifft2(g_i))
    # 移除残余的虚部，小于0的值
    i_i = np.real(i_i)
    mask = i_i >= 0
    i_i = mask * i_i

    if trans_flag:
        i_i = np.transpose(i_i, (2, 0, 1))
    if tensor_flag:
        i_i = torch.from_numpy(i_i).float()

    return i_i


def incoh_sim(i_i_hr, params_hr, params_lr):
    i_g_hr = i_i_hr
    i_g_lr = incoh_sample(i_g_hr, params_hr, params_lr)
    i_i_lr = incoh_imaging(i_g_lr, params_lr)

    return i_i_lr
