# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved 
#
# @Time    : 2022/4/14 21:03
# @Author  : Xiao Wu

# from Toolbox.wald_utilities import MTF
from wald_utilities import *
#from mtf import MTF_MS
import torch
from torch.nn import functional as F
import numpy as np
# from metrics import D_s_new


def norm(tensor, order=2, axis=None):
    """Computes the l-`order` norm of a tensor.

    Parameters
    ----------
    tensor : tl.tensor
    order : int
    axis : int or tuple

    Returns
    -------
    float or tensor
        If `axis` is provided returns a tensor.
    """
    # handle difference in default axis notation
    if axis == ():
        axis = None

    if order == 'inf':
        return torch.max(torch.abs(tensor), dim=axis)
    if order == 1:
        return torch.sum(torch.abs(tensor), dim=axis)
    elif order == 2:
        return torch.sqrt(torch.sum(torch.abs(tensor) ** 2, dim=axis))
    else:
        return torch.sum(torch.abs(tensor) ** order, dim=axis) ** (1 / order)


def indexes_evaluation_fs(sr, lrms, ms, pan, L, th_values, sensor='none', ratio=6, mode='QS'):
    if th_values:
        sr[sr > 2 ** L] = 2 ** L
        sr[sr < 0] = 0

    if mode == 'QS':
        QNR_index, D_lambda, D_S = QS(sr, ms, lrms, pan, ratio)
        return QNR_index, D_lambda, D_S


def img_ssim(img1, img2, block_size):
    img1 = img1.float()
    img2 = img2.float()

    _, channel, h, w = img1.size()
    N = block_size ** 2
    sum2filter = torch.ones([channel, 1, block_size, block_size]).cuda()
    # print(img1.shape, sum2filter.shape)
    img1_sum = F.conv2d(img1, sum2filter, groups=channel)
    img2_sum = F.conv2d(img2, sum2filter, groups=channel)
    img1_sq_sum = F.conv2d(img1 * img1, sum2filter, groups=channel)
    img2_sq_sum = F.conv2d(img2 * img2, sum2filter, groups=channel)
    img12_sum = F.conv2d(img1 * img2, sum2filter, groups=channel)
    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    quality_map = torch.ones_like(denominator)
    two = 2 * torch.ones_like(denominator)
    zeros = torch.zeros_like(denominator)
    # zeros_2 = torch.zeros_like(img12_sq_sum_mul)
    # index = (denominator1 == 0) and (img12_sq_sum_mul != 0)
    # quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]

    quality_map = torch.where((denominator1 == zeros).float() + (img12_sq_sum_mul != zeros).float() == two,
                              2 * img12_sum_mul / img12_sq_sum_mul, quality_map)
    # index = denominator != 0
    # quality_map[index] = numerator[index] / denominator[index]
    quality_map = torch.where(denominator != zeros, numerator / denominator, quality_map)


    return quality_map.mean(2).mean(2)


def Qavg(im1, im2, S):
    Q_orig = img_ssim(im1, im2, S)
    Q_avg = torch.mean(Q_orig)

    return Q_avg


def D_lambda_k(sr, ms, ratio, sensor, S):
    assert sr.shape == ms.shape, print("ms shape is not equal to sr shape")

    H, W, nbands = sr.shape

    if H % S != 0 or W % S != 0:
        raise ValueError("H, W must be multiple of the block size")

    fused_degraded = MTF(sr, sensor, ratio, nbands)

    ms = ms.permute(2, 0, 1).unsqueeze(0)
    Q2n_index = Qavg(ms, fused_degraded, S)
    Dl = 1 - Q2n_index

    return Dl



def QS(sr, ms, lrms, pan,pan_filt,img_range, ratio, device,sensor='none',
       beta=1, alpha=1, q=1, p=1, S=32):

    D_lambda_index = D_lambda_k(sr.transpose(1, 0), ms.transpose(1, 0), ratio, sensor, S)


    D_s_index =D_s_new(sr, ms, lrms, pan,pan_filt,device)


    QNR_index = (1 - D_lambda_index) ** alpha * (1 - D_s_index) ** beta

    return QNR_index, D_lambda_index, D_s_index



def D_s_new(I_F, I_MS, I_MS_LR, I_PAN, pan_filt, device, ratio=4, S=32, q=1):
    flag_orig_paper = 0  # if 0, Toolbox 1.0, otherwise, original QNR paper 

    if I_F.shape != I_MS.shape:
        raise ValueError('The two input images must have the same dimensions')

    N, M, Nb = I_F.shape

    if N % S != 0 or M % S != 0:
        raise ValueError('The number of rows and columns must be multiples of the block size')

    D_s_index = 0.0
    for i in range(Nb):
        band1 = I_F[:, :, i]
        band2 = I_PAN
        fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S, loc[1]:loc[1]+S])
        Qmap_high = blockproc(band1, (S, S), fun_uqi)
        Q_high = torch.mean(Qmap_high)

        if flag_orig_paper == 0:
            # 选项1
            band1 = I_MS[:, :, i]
            band2 = pan_filt
            fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S, loc[1]:loc[1]+S])
            Qmap_low = blockproc(band1, (S, S), fun_uqi)
        else:
            # 选项2
            band1 = I_MS_LR[:, :, i]
            band2 = pan_filt
            fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S//ratio, loc[1]:loc[1]+S//ratio])
            Qmap_low = blockproc(band1, (S//ratio, S//ratio), fun_uqi)

        Q_low = torch.mean(Qmap_low)
        D_s_index += torch.abs(Q_high - Q_low) ** q

    D_s_index = (D_s_index / Nb) ** (1 / q)

    return D_s_index



import torch
import torch.nn.functional as F
from skimage.util import view_as_windows
from scipy import ndimage

def uqi(x, y):
    x = x.flatten().double()  # 转为 double 类型
    y = y.flatten().double()
    mx = torch.mean(x)
    my = torch.mean(y)
    C = torch.cov(torch.stack([x, y]))

    if C.shape == (2, 2):
        Q = 4 * C[0, 1] * mx * my / ((C[0, 0] + C[1, 1]) * (mx ** 2 + my ** 2))
        return Q
    else:
        return torch.tensor(0.0, device=x.device)  # 处理协方差矩阵不是2x2的情况

# def blockproc(image, block_size, func):
#     blocks = view_as_windows(image.detach().cpu().numpy(), block_size, step=block_size)  # 使用 NumPy 处理窗口
#     result = torch.zeros((blocks.shape[0], blocks.shape[1]), device=image.device)
    
#     for i in range(blocks.shape[0]):
#         for j in range(blocks.shape[1]):
#             block = torch.tensor(blocks[i, j], device=image.device)
#             result[i, j] = func(block, (i * block_size[0], j * block_size[1]))
    
#     return result



def blockproc(image, block_size, func):
    # 获取图像的高度和宽度
    height, width = image.shape
    
    # 计算块的行数和列数
    blocks_per_row = height // block_size[0]
    blocks_per_col = width // block_size[1]

    # 创建一个用于存储结果的张量
    result = torch.zeros((blocks_per_row, blocks_per_col), device=image.device)
    
    # 遍历所有块
    for i in range(blocks_per_row):
        for j in range(blocks_per_col):
            # 提取每个块
            block = image[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]]
            
            # 对该块应用传入的函数
            result[i, j] = func(block, (i * block_size[0], j * block_size[1]))
    
    return result


