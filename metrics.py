import numpy as np
from numpy.linalg import norm
import math
from scipy.ndimage.filters import sobel
from scipy.stats import pearsonr
import cv2
import torch
from scipy.ndimage import zoom

def sam_old(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """


    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    sam_rad = np.zeros(x_pred.shape[0:2])
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y]
            tmp_true = x_true[x, y]
            norm_pred=norm (tmp_pred)
            norm_true=norm(tmp_true)
            if(norm_pred!=0)and (norm_true!=0):
                a=tmp_true*tmp_pred
                temp=np.sum(a)/norm(tmp_pred)/norm(tmp_true)
                if temp>1.0:
                    temp=1.0
                sam_rad[x, y] = np.arccos(temp)
            else:
                sam_rad[x,y]=0.0

    sam_deg = sam_rad.mean()
    return sam_deg


def SAM(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """


    assert x_true.ndim ==3 and x_true.shape == x_pred.shape
    dot_sum = np.sum(x_true*x_pred,axis=2)
    norm_true = norm(x_true, axis=2)
    norm_pred = norm(x_pred, axis=2)

    res = np.arccos(dot_sum/norm_pred/norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x,y) in zip(is_nan[0], is_nan[1]):
        res[x,y]=0

    sam = np.mean(res)
    return sam

def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')
    scc = 0.0
    for i in range(ms.shape[2]):
        a = (ps_sobel[:,:,i]).reshape(ms.shape[0]*ms.shape[1])
        b = (ms_sobel[:,:,i]).reshape(ms.shape[0]*ms.shape[1])
        #print (pearsonr(ps_sobel, ms_sobel))
        scc += pearsonr(a, b)[0]
        #scc += (np.sum(ps_sobel*ms_sobel)/np.sqrt(np.sum(ps_sobel*ps_sobel))/np.sqrt(np.sum(ms_sobel*ms_sobel)))

    return scc/ms.shape[2]

def CC(ms, ps):
    cc = 0.0
    for i in range(ms.shape[2]):
        a = (ps[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        b = (ms[:, :, i]).reshape(ms.shape[0] * ms.shape[1])
        # print (pearsonr(ps_sobel, ms_sobel))
        cc += pearsonr(a, b)[0]


    return cc / ms.shape[2]

def Q4(ms, ps):
    def conjugate(a):
        sign = -1 * np.ones(a.shape)
        sign[0,:]=1
        return a*sign
    def product(a, b):
        a = a.reshape(a.shape[0],1)
        b = b.reshape(b.shape[0],1)
        R = np.dot(a, b.transpose())
        r = np.zeros(4)
        r[0] = R[0, 0] - R[1, 1] - R[2, 2] - R[3, 3]
        r[1] = R[0, 1] + R[1, 0] + R[2, 3] - R[3, 2]
        r[2] = R[0, 2] - R[1, 3] + R[2, 0] + R[3, 1]
        r[3] = R[0, 3] + R[1, 2] - R[2, 1] + R[3, 0]
        return r
    imps = np.copy(ps)
    imms = np.copy(ms)
    vec_ps = imps.reshape(imps.shape[1]*imps.shape[0], imps.shape[2])
    vec_ps = vec_ps.transpose(1,0)

    vec_ms = imms.reshape(imms.shape[1]*imms.shape[0], imms.shape[2])
    vec_ms = vec_ms.transpose(1,0)

    m1 = np.mean(vec_ps, axis=1)
    d1 = (vec_ps.transpose(1,0)-m1).transpose(1,0)
    s1 = np.mean(np.sum(d1*d1, axis=0))

    m2 = np.mean(vec_ms, axis=1)
    d2 = (vec_ms.transpose(1, 0) - m2).transpose(1, 0)
    s2 = np.mean(np.sum(d2 * d2, axis=0))


    Sc = np.zeros(vec_ms.shape)
    d2 = conjugate(d2)
    for i in range(vec_ms.shape[1]):
        Sc[:,i] = product(d1[:,i], d2[:,i])
    C = np.mean(Sc, axis=1)

    Q4 = 4 * np.sqrt(np.sum(m1*m1) * np.sum(m2*m2) * np.sum(C*C)) / (s1 + s2) / (np.sum(m1 * m1) + np.sum(m2 * m2))
    return Q4

def RMSE(ms, ps):
    d = (ms - ps)**2

    rmse = np.sqrt(np.sum(d)/(d.shape[0]*d.shape[1]))
    return rmse

def ERGAS(ms, ps, ratio=4):
    m, n, d = ms.shape
    summed = 0.0
    for i in range(d):
        summed += (RMSE(ms[:,:,i], ps[:,:,i]))**2 / np.mean(ps[:,:,i])**2

    ergas = 100 * (1 / ratio) *np.sqrt(summed/d)
    return ergas

def UIQC(ms, ps):
    l = ms.shape[2]
    uiqc = 0.0
    for i in range(l):
        uiqc += Q(ms[:,:,i], ps[:,:,i])

    return uiqc/4

def Q(a, b):
    a = a.reshape(a.shape[0]*a.shape[1])
    b = b.reshape(b.shape[0]*b.shape[1])
    temp=np.cov(a,b)
    d1 =  temp[0,0]
    cov = temp[0,1]
    d2 = temp[1,1]
    m1 = np.mean(a)
    m2 = np.mean(b)
    Q = 4*cov*m1*m2/(d1+d2)/(m1**2+m2**2)

    return Q

def D_lamda(ps, l_ms):
    '''
    ps：[H,W,C]
    l_ms:[H/4 ,W/4 ,C]  注意还没有上采样。即数据集中的ms，而不是lms和gt
    '''
    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j!=i:
                #print(np.abs(Q(ps[:, :, i], ms[:, :, j]) - Q(l_ps[:, :, i], l_ms[:, :, j])))
                sum += np.abs(Q(ps[:, :, i], ps[:, :, j]) - Q(l_ms[:, :, i], l_ms[:, :, j]))
    return sum/L/(L-1)


import numpy as np
from scipy import ndimage
from skimage.util import view_as_windows

def uqi(x, y):
    x = x.flatten().astype(np.double)
    y = y.flatten().astype(np.double)
    mx = np.mean(x)
    my = np.mean(y)
    C = np.cov(x, y)
    
    if C.shape == (2, 2):
        Q = 4 * C[0, 1] * mx * my / ((C[0, 0] + C[1, 1]) * (mx**2 + my**2))
        return Q
    else:
        return 0  # handle the case where covariance matrix is not 2x2

def blockproc(image, block_size, func):
    blocks = view_as_windows(image, block_size, step=block_size)
    result = np.zeros((blocks.shape[0], blocks.shape[1]))
    
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            result[i, j] = func(block, (i * block_size[0], j * block_size[1]))
    
    return result

def D_lambda_new(I_F, I_MS, I_MS_LR, S=32, ratio=4, p=1):
    flag_orig_paper = 0  # if 0, Toolbox 1.0, otherwise, original QNR paper 

    if I_F.shape != I_MS.shape:
        raise ValueError('The two input images must have the same dimensions')

    N, M, Nb = I_F.shape

    if N % S != 0:
        raise ValueError('The number of rows must be multiple of the block size')

    if M % S != 0:
        raise ValueError('The number of columns must be multiple of the block size')

    D_lambda_index = 0
    for i in range(Nb - 1):
        for j in range(i + 1, Nb):
            if flag_orig_paper == 0:
                # Opt. 1 (as toolbox 1.0)
                band1 = I_MS[:, :, i]
                band2 = I_MS[:, :, j]
                fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S, loc[1]:loc[1]+S])
                Qmap_exp = blockproc(band1, (S, S), fun_uqi)
            else:
                # Opt. 2 (as paper QNR)
                band1 = I_MS_LR[:, :, i]
                band2 = I_MS_LR[:, :, j]
                fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S//ratio, loc[1]:loc[1]+S//ratio])
                Qmap_exp = blockproc(band1, (S//ratio, S//ratio), fun_uqi)
            
            Q_exp = np.mean(Qmap_exp)
            
            band1 = I_F[:, :, i]
            band2 = I_F[:, :, j]
            fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S, loc[1]:loc[1]+S])
            Qmap_fused = blockproc(band1, (S, S), fun_uqi)
            Q_fused = np.mean(Qmap_fused)
            D_lambda_index += abs(Q_fused - Q_exp) ** p
    
    s = (Nb**2 - Nb) / 2
    D_lambda_index = (D_lambda_index / s) ** (1 / p)

    return D_lambda_index

# Example usage:
# I_F, I_MS, I_MS_LR should be numpy arrays of the same shape with dimensions (N, M, Nb)
# S is the block size
# ratio is the scale ratio between the high-resolution and low-resolution images
# p is the power parameter


import numpy as np
from scipy import ndimage
from skimage.transform import resize

from scipy.ndimage import convolve

def interp23tap(image, ratio):

    image = np.transpose(image, (2, 0, 1))
    
    b,r,c = image.shape

    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    
    first = 1
    # for z in range(1,np.int(np.log2(ratio))+1):
    for z in range(1, int(ratio / 2) + 1):
        I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        if first:
            I1LRU[:, 1:I1LRU.shape[1]:2, 1:I1LRU.shape[2]:2]=image
            first = 0
        else:
            I1LRU[:,0:I1LRU.shape[1]:2,0:I1LRU.shape[2]:2]=image
        
        for ii in range(0,b):
            t = I1LRU[ii,:,:]
            # # 对图像的行进行滤波
            # t = convolve(t, BaseCoeff[:, None], mode='wrap')
            # # 对图像的列进行滤波
            # t = convolve(t.T, BaseCoeff[:, None], mode='wrap').T
            for j in range(0,t.shape[0]):
                t[j,:]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
            for k in range(0,t.shape[1]):
                t[:,k]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
            I1LRU[ii,:,:]=t
        image = I1LRU
        
    re_image=np.transpose(I1LRU, (1, 2, 0))
        
    return re_image


def D_s_new(I_F, I_MS, I_MS_LR, I_PAN, ratio=4, S=32, q=1):
    flag_orig_paper = 0  # if 0, Toolbox 1.0, otherwise, original QNR paper 

    if I_F.shape != I_MS.shape:
        raise ValueError('The two input images must have the same dimensions')

    N, M, Nb = I_F.shape

    if N % S != 0:
        raise ValueError('The number of rows must be multiple of the block size')

    if M % S != 0:
        raise ValueError('The number of columns must be multiple of the block size')

    if flag_orig_paper == 0:
        # Opt. 1 (as toolbox 1.0) 
        scaled_image = cv2.resize(I_PAN, (int(I_PAN.shape[1] / ratio), int(I_PAN.shape[0] / ratio)), interpolation=cv2.INTER_CUBIC)  #INTER_CUBIC   INTER_LINEAR
        
        pan_filt = interp23tap(resize(I_PAN, (int(I_PAN.shape[0] / ratio), int(I_PAN.shape[1] / ratio)), order=3,anti_aliasing=True), ratio)
    else:
        # Opt. 2 (as paper QNR)
        pan_filt = resize(I_PAN, (I_PAN.shape[0] // ratio, I_PAN.shape[1] // ratio))

    D_s_index = 0
    for i in range(Nb):
        band1 = I_F[:, :, i]
        band2 = I_PAN
        fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S, loc[1]:loc[1]+S])
        Qmap_high = blockproc(band1, (S, S), fun_uqi)
        Q_high = np.mean(Qmap_high)
        
        if flag_orig_paper == 0:
            # Opt. 1 (as toolbox 1.0)
            band1 = I_MS[:, :, i]
            band2 = pan_filt
            fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S, loc[1]:loc[1]+S])
            Qmap_low = blockproc(band1, (S, S), fun_uqi)
        else:
            # Opt. 2 (as paper QNR)
            band1 = I_MS_LR[:, :, i]
            band2 = pan_filt
            fun_uqi = lambda block, loc: uqi(block, band2[loc[0]:loc[0]+S//ratio, loc[1]:loc[1]+S//ratio])
            Qmap_low = blockproc(band1, (S//ratio, S//ratio), fun_uqi)
        
        Q_low = np.mean(Qmap_low)
        D_s_index += abs(Q_high - Q_low) ** q

    D_s_index = (D_s_index / Nb) ** (1 / q)

    return D_s_index

# Example usage:
# I_F, I_MS, I_MS_LR, I_PAN should be numpy arrays with appropriate dimensions
# ratio is the scale ratio between the high-resolution and low-resolution images
# S is the block size
# q is the power parameter



def D_s(ps, l_ms, pan):
    
    '''
    ps：[H,W,C]
    l_ms:[H/4 ,W/4 ,C]  注意还没有上采样
    pan:[H,W,C]
    '''
    L = ps.shape[2]
    l_pan = cv2.pyrDown(pan)
    l_pan = cv2.pyrDown(l_pan)
    sum = 0.0
    for i in range(L):
        sum += np.abs(Q(ps[:,:,i], pan) - Q(l_ms[:,:,i], l_pan))
    return sum/L


def q2n(I_GT, I_F, Q_blocks_size=32, Q_shift=32):
    N1, N2, N3 = I_GT.shape
    size2 = Q_blocks_size

    stepx = int(np.ceil(N1 / Q_shift))
    stepy = int(np.ceil(N2 / Q_shift))

    if stepy <= 0:
        stepy = 1
        stepx = 1

    est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2

    if est1 != 0 or est2 != 0:
        refref = []
        fusfus = []

        for i in range(N3):
            a1 = I_GT[:, :, i]
            ia1 = np.pad(a1, ((0, est1), (0, est2)), mode='reflect')
            refref.append(ia1)

        I_GT = np.stack(refref, axis=2)

        for i in range(N3):
            a2 = I_F[:, :, i]
            ia2 = np.pad(a2, ((0, est1), (0, est2)), mode='reflect')
            fusfus.append(ia2)

        I_F = np.stack(fusfus, axis=2)

    I_F = I_F.astype(np.uint16)
    I_GT = I_GT.astype(np.uint16)

    N1, N2, N3 = I_GT.shape

    if ((np.ceil(np.log2(N3))) - np.log2(N3)) != 0:
        Ndif = int(2**(np.ceil(np.log2(N3))) - N3)
        dif = np.zeros((N1, N2, Ndif), dtype=np.uint16)
        I_GT = np.concatenate((I_GT, dif), axis=2)
        I_F = np.concatenate((I_F, dif), axis=2)

    N1, N2, N3 = I_GT.shape

    valori = np.zeros((stepx, stepy, N3))

    for j in range(stepx):
        for i in range(stepy):
            patch_GT = I_GT[(j * Q_shift):((j * Q_shift) + Q_blocks_size), 
                            (i * Q_shift):((i * Q_shift) + size2), :]
            patch_F = I_F[(j * Q_shift):((j * Q_shift) + Q_blocks_size), 
                          (i * Q_shift):((i * Q_shift) + size2), :]
            o = onions_quality(patch_GT, patch_F, Q_blocks_size)
            valori[j, i, :] = o

    Q2n_index_map = np.sqrt(np.sum(valori**2, axis=2))
    Q2n_index = np.mean(Q2n_index_map)

    return Q2n_index


def onions_quality(dat1, dat2, size1):
    dat1 = dat1.astype(np.float64)
    dat2 = dat2.astype(np.float64)
    dat2 = np.concatenate((dat2[:, :, :1], -dat2[:, :, 1:]), axis=2)
    N3 = dat1.shape[2]
    size2 = size1

    # Block normalization
    for i in range(N3):
        a1, s, t = norm_blocco(dat1[:, :, i])
        dat1[:, :, i] = a1
        if s == 0:
            if i == 0:
                dat2[:, :, i] = dat2[:, :, i] - s + 1
            else:
                dat2[:, :, i] = -(-dat2[:, :, i] - s + 1)
        else:
            if i == 0:
                dat2[:, :, i] = ((dat2[:, :, i] - s) / t) + 1
            else:
                dat2[:, :, i] = -(((-dat2[:, :, i] - s) / t) + 1)

    m1 = np.zeros(N3)
    m2 = np.zeros(N3)

    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = np.zeros((size1, size2))
    mod_q2 = np.zeros((size1, size2))

    for i in range(N3):
        m1[i] = np.mean(dat1[:, :, i])
        m2[i] = np.mean(dat2[:, :, i])
        mod_q1m += m1[i]**2
        mod_q2m += m2[i]**2
        mod_q1 += dat1[:, :, i]**2
        mod_q2 += dat2[:, :, i]**2

    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)

    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2
    int1 = (size1 * size2) / ((size1 * size2) - 1) * np.mean(mod_q1**2)
    int2 = (size1 * size2) / ((size1 * size2) - 1) * np.mean(mod_q2**2)
    termine3 = int1 + int2 - (size1 * size2) / ((size1 * size2) - 1) * (mod_q1m**2 + mod_q2m**2)

    mean_bias = 2 * termine2 / termine4
    if termine3 == 0:
        q = np.zeros((1, 1, N3))
        q[:, :, -1] = mean_bias
    else:
        cbm = 2 / termine3
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        qv = np.zeros(N3)
        for i in range(N3):
            qv[i] = (size1 * size2) / ((size1 * size2) - 1) * np.mean(qu[:, :, i])
        q = qv - (size1 * size2) / ((size1 * size2) - 1) * qm
        q = q * mean_bias * cbm

    return q

def norm_blocco(x):
    a = np.mean(x)
    c = np.std(x)

    if c == 0:
        c = np.finfo(float).eps

    y = ((x - a) / c) + 1
    return y, a, c

def onion_mult(onion1, onion2):
    N = len(onion1)

    if N > 1:
        L = N // 2

        a = onion1[:L]
        b = onion1[L:]
        b = np.concatenate(([b[0]], -b[1:]))
        c = onion2[:L]
        d = onion2[L:]
        d = np.concatenate(([d[0]], -d[1:]))

        if N == 2:
            ris = np.concatenate((a * c - d * b, a * d + c * b))
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, np.concatenate(([b[0]], -b[1:])))
            ris3 = onion_mult(np.concatenate(([a[0]], -a[1:])), d)
            ris4 = onion_mult(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate((aux1, aux2))
    else:
        ris = onion1 * onion2

    return ris

def onion_mult2D(onion1, onion2):
    N3 = onion1.shape[2]

    if N3 > 1:
        L = N3 // 2

        a = onion1[:, :, :L]
        b = onion1[:, :, L:]
        b = np.concatenate((b[:, :, :1], -b[:, :, 1:]), axis=2)
        c = onion2[:, :, :L]
        d = onion2[:, :, L:]
        d = np.concatenate((d[:, :, :1], -d[:, :, 1:]), axis=2)

        if N3 == 2:
            ris = np.concatenate((a * c - d * b, a * d + c * b), axis=2)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, np.concatenate((b[:, :, :1], -b[:, :, 1:]), axis=2))
            ris3 = onion_mult2D(np.concatenate((a[:, :, :1], -a[:, :, 1:]), axis=2), d)
            ris4 = onion_mult2D(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate((aux1, aux2), axis=2)
    else:
        ris = onion1 * onion2

    return ris
