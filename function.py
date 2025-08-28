import torch.nn as nn
import matplotlib.pyplot as plt
import os
from os.path import join
import time
import torch.utils.data as data
import h5py
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import math
from scipy import io as sio
from PIL import Image
import metrics as mtc
import cv2

# load data_hp

def get_edge(data):  # for training: HxWxC
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5), normalize=True)
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5), normalize=True)
    return rs


class Dataset_Pro_hp(data.Dataset):
    def __init__(self, file_path, img_scale):
        super(Dataset_Pro_hp, self).__init__()
        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3=8806x8x64x64

        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)

        ms1 = data["ms"][...]  # NxCxHxW=0,1,2,3
        ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / img_scale  # NxHxWxC
        ms1_tmp = get_edge(ms1)  # NxHxWxC
        self.ms_hp = torch.from_numpy(ms1_tmp).permute(0, 3, 1, 2)  # NxCxHxW:

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / img_scale  # NxHxWx1
        pan1 = np.squeeze(pan1, axis=3)  # NxHxW
        pan_hp_tmp = get_edge(pan1)  # NxHxW
        pan_hp_tmp = np.expand_dims(pan_hp_tmp, axis=3)  # NxHxWx1
        self.pan_hp = torch.from_numpy(pan_hp_tmp).permute(0, 3, 1, 2)  # Nx1xHxW:
        print(
            f"gt: {self.gt.size()}, lms: {self.lms.size()}, pan_hp: {self.pan_hp.size()}, ms_hp: {self.ms_hp.size()} with {img_scale}")

    #####必要函数
    def __getitem__(self, index):
        return {'gt': self.gt[index, :, :, :].float(),
                'lms': self.lms[index, :, :, :].float(),
                'ms_hp': self.ms_hp[index, :, :, :].float(),
                'pan_hp': self.pan_hp[index, :, :, :].float()}

        #####必要函数

    def __len__(self):
        return self.gt.shape[0]

def load_dataset_H5_hp(file_path, scale, use_cuda=True):
    data = h5py.File(file_path)  # NxHxWxC
    shape_list = []

    pan1 = torch.from_numpy(data['pan'][...] / scale).float()
    ms1 = torch.from_numpy(data['ms'][...] / scale).float()

    lms = torch.from_numpy(data['lms'][...] / scale).float()#.permute(0, 3, 1, 2)

    ms2 = data["ms"][...]    
    ms2 = np.array(ms2.transpose(0, 2, 3, 1), dtype=np.float32) / scale  # NxHxWxC
    ms2_tmp = get_edge(ms2)
    ms_hp = torch.from_numpy(ms2_tmp).permute(0, 3, 1, 2)  # NxCxHxW

    pan = np.squeeze(data['pan'][...])
    pan = pan[:, np.newaxis, :, :]  # NxCxHxW (C=1)
    pan_hp = torch.from_numpy(get_edge(pan / scale)).float()#.permute(0, 3, 1, 2)  # Nx1xHxW:
    if data.get('gt', None) is None:
        gt = torch.from_numpy(data['lms'][...]).float()
    else:
        gt = torch.from_numpy(data['gt'][...]).float()

    return {'lms': lms,
            'ms_hp': ms_hp,
            'pan_hp': pan_hp,
            'pan': pan1,
            'ms': ms1,
            'gt': gt.permute([0, 2, 3, 1])
            }


class MultiExmTest_h5_hp(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, suffix='.h5'):
        super(MultiExmTest_h5_hp, self).__init__()

        self.img_scale = img_scale
        print(f"loading MultiExmTest_h5_hp: {file_path} with {img_scale}")
        # 一次性载入到内存
        if 'hp' not in dataset_name:
            # data = load_dataset_H5(file_path, img_scale, False)
            data = load_dataset_H5_hp(file_path, img_scale, False)

        # elif 'hp' in dataset_name:
        #     file_path = file_path.replace('_hp', '')
        #     data = load_dataset_H5_hp(file_path, img_scale, False)

        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError
        
        if suffix == '.mat':
            self.lms = data['lms'].permute(0, 3, 1, 2)  # CxHxW = 8x256x256
            self.ms = data['ms'].permute(0, 3, 1, 2)  # CxHxW= 8x64x64
            self.pan = data['pan'].unsqueeze(1)
            self.gt = data['gt'].permute(0, 3, 1, 2)
        else:
            self.lms = data['lms']
            self.ms = data['ms']
            self.pan = data['pan']
            self.ms_hp = data['ms_hp']
            self.pan_hp = data['pan_hp']
            self.gt = data['gt']

        print(f"lms: {self.lms.shape}, ms: {self.ms.shape}, pan: {self.pan.shape}, gt: {self.gt.shape}, ms_hp: {self.ms_hp.shape}, pan_hp: {self.pan_hp.shape}")

    def __getitem__(self, item):
        return {'lms': self.lms[item, ...],
                'ms': self.ms[item, ...],
                'pan': self.pan[item, ...],
                'gt': self.gt[item, ...],
                'ms_hp': self.ms_hp[item, ...],
                'pan_hp': self.pan_hp[item, ...]
                }

    def __len__(self):
        return self.gt.shape[0]


# load data
class Dataset_Pro(data.Dataset):
    def __init__(self, file_path, img_scale):
        super(Dataset_Pro, self).__init__()

        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter

        # t = ms1[0,0,0,1]

        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:


    #####必要函数
    def __getitem__(self, index):
        return {'gt':self.gt[index, :, :, :].float(),
               'lms':self.lms[index, :, :, :].float(),
               'ms':self.ms[index, :, :, :].float(),
               'pan':self.pan[index, :, :, :].float()}

            #####必要函数
    def __len__(self):
        return self.gt.shape[0]

class Dataset_Pro_ZS(data.Dataset):
    def __init__(self, file_path, img_scale):
        super(Dataset_Pro_ZS, self).__init__()

        data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")
        # # tensor type:
        # gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        # gt1 = np.array(gt1, dtype=np.float32) / img_scale
        # self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter

        # t = ms1[0,0,0,1]

        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:


    #####必要函数
    def __getitem__(self, index):
        return {
               'lms':self.lms[index, :, :, :].float(),
               'ms':self.ms[index, :, :, :].float(),
               'pan':self.pan[index, :, :, :].float()}

            #####必要函数
    def __len__(self):
        return self.ms.shape[0]


class Dataset_Pro_ZS_new(data.Dataset):
    def __init__(self, file_path, img_scale, name):
        super(Dataset_Pro_ZS_new, self).__init__()

        dataset = h5py.File(file_path, 'r')

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")

        ms = np.array(dataset['ms'][name], dtype=np.float32)
        lms = np.array(dataset['lms'][name], dtype=np.float32)
        pan = np.array(dataset['pan'][name], dtype=np.float32) 

        # lms_max_value = np.max(lms)
        # print("整个lms的最大值:", lms_max_value)
        # lms_min_value = np.min(lms)
        # print("整个lms的最小值:", lms_min_value)

        ms = torch.from_numpy(ms).float() / img_scale
        lms = torch.from_numpy(lms).float() / img_scale
        pan = torch.from_numpy(pan).float() / img_scale

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms

        # LMS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.lms_crops = MS_crop(ms)
        self.lms_crops = lms

        # PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        # self.pan_crops = PAN_crop(pan)
        self.pan_crops = pan

    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops

    def __len__(self):
        return 1

from skimage.transform import resize
from metrics import interp23tap

class Dataset_Pro_ZS_new_np(data.Dataset):
    def __init__(self, file_path, img_scale, name):
        super(Dataset_Pro_ZS_new_np, self).__init__()

        dataset = h5py.File(file_path, 'r')

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")

        ms = np.array(dataset['ms'][name], dtype=np.float32)
        lms = np.array(dataset['lms'][name], dtype=np.float32)
        pan = np.array(dataset['pan'][name], dtype=np.float32) 


        I_PAN = np.transpose(pan, (1, 2, 0))
        ratio =4
        pan_filt = interp23tap(resize(I_PAN, (int(I_PAN.shape[0] / ratio), int(I_PAN.shape[1] / ratio)), order=3,anti_aliasing=True), ratio)



        ms = torch.from_numpy(ms).float() / img_scale
        lms = torch.from_numpy(lms).float() / img_scale
        pan = torch.from_numpy(pan).float() / img_scale
        pan_filt = torch.from_numpy(pan_filt).float()

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms

        # LMS_crop = torchvision.transforms.TenCrop(lms.shape[1] / 2)
        # self.lms_crops = LMS_crop(lms)
        self.lms_crops = lms

        # PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        # self.pan_crops = PAN_crop(pan)
        self.pan_crops = pan

        self.pan_filt = pan_filt



    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops, self.pan_filt

    def __len__(self):
        return 1


class Dataset_Pro_ZS_new_np_DA(data.Dataset):
    def __init__(self, file_path, img_scale, name):
        super(Dataset_Pro_ZS_new_np_DA, self).__init__()

        dataset = h5py.File(file_path, 'r')

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")

        ms = np.array(dataset['ms'][name], dtype=np.float32)
        lms = np.array(dataset['lms'][name], dtype=np.float32)
        pan = np.array(dataset['pan'][name], dtype=np.float32) 


        I_PAN = np.transpose(pan, (1, 2, 0))
        ratio =4
        pan_filt = interp23tap(resize(I_PAN, (int(I_PAN.shape[0] / ratio), int(I_PAN.shape[1] / ratio)), order=3,anti_aliasing=True), ratio)



        ms = torch.from_numpy(ms).float() / img_scale
        lms = torch.from_numpy(lms).float() / img_scale
        pan = torch.from_numpy(pan).float() / img_scale
        pan_filt = torch.from_numpy(pan_filt).float()

        MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        self.ms_crops = MS_crop(ms)
        # self.ms_crops = ms

        LMS_crop = torchvision.transforms.TenCrop(lms.shape[1] / 2)
        self.lms_crops = LMS_crop(lms)
        # self.lms_crops = lms

        PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        self.pan_crops = PAN_crop(pan)
        # self.pan_crops = pan

        pan_filt= pan_filt.permute(2, 0, 1)
        PAN_filt_crop = torchvision.transforms.TenCrop(pan_filt.shape[1] / 2)
        
        pan_filt = PAN_filt_crop(pan_filt)
        # self.pan_filt = pan_filt

        self.pan_filt= pan_filt

        self.num_crops = len(self.ms_crops)

    def __getitem__(self, index):
        crop_index = index % self.num_crops # 确保索引在范围内
        return (self.ms_crops[crop_index], 
                self.lms_crops[crop_index], 
                self.pan_crops[crop_index],
                self.pan_filt[crop_index]
                )

    def __len__(self):
        return self.num_crops




class Dataset_Pro_ZS_new_RR(data.Dataset):
    def __init__(self, file_path, img_scale, name):
        super(Dataset_Pro_ZS_new_RR, self).__init__()

        dataset = h5py.File(file_path, 'r')

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")

        ms = np.array(dataset['ms'][name], dtype=np.float32) 
        lms = np.array(dataset['lms'][name], dtype=np.float32)
        pan = np.array(dataset['pan'][name], dtype=np.float32) 

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms

        # LMS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.lms_crops = MS_crop(ms)
        self.lms_crops = lms

        # PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        # self.pan_crops = PAN_crop(pan)
        self.pan_crops = pan

    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops

    def __len__(self):
        return 1



# 数据增强
import torchvision
class Dataset_Pro_ZS_new_DA(data.Dataset):
    def __init__(self, file_path, img_scale, name):
        super(Dataset_Pro_ZS_new_DA, self).__init__()

        dataset = h5py.File(file_path, 'r')

        print(f"loading Dataset_Pro: {file_path} with {img_scale}")

        ms = np.array(dataset['ms'][name], dtype=np.float32)
        lms = np.array(dataset['lms'][name], dtype=np.float32)
        pan = np.array(dataset['pan'][name], dtype=np.float32)

        self.ms = torch.from_numpy(ms).float() / img_scale
        self.lms = torch.from_numpy(lms).float() / img_scale
        self.pan = torch.from_numpy(pan).float() / img_scale

        MS_crop = torchvision.transforms.TenCrop(self.ms.shape[1] / 2)
        self.ms_crops = MS_crop(self.ms)
        # self.ms_crops = ms

        LMS_crop = torchvision.transforms.TenCrop(self.lms.shape[1] / 2)
        self.lms_crops = LMS_crop(self.lms)
        # self.lms_crops = lms

        PAN_crop = torchvision.transforms.TenCrop(self.pan.shape[1] / 2)
        self.pan_crops = PAN_crop(self.pan)
        # self.pan_crops = pan

        self.num_crops = len(self.ms_crops)

    def __getitem__(self, index):
        crop_index = index % self.num_crops # 确保索引在范围内
        return (self.ms_crops[crop_index], 
                self.lms_crops[crop_index], 
                self.pan_crops[crop_index])

    def __len__(self):
        return self.num_crops



def load_dataset_H5_ZS(file_path, scale, use_cuda=True):
    data = h5py.File(file_path)  # NxCxHxW
    print(data.keys())
    # tensor type:
    if use_cuda:
        lms = torch.from_numpy(data['lms'][...] / scale).cuda().float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / scale).cuda().float()  # CxHxW= 8x64x64
        pan = torch.from_numpy(data['pan'][...] / scale).cuda().float()  # HxW = 256x256

        # gt = torch.from_numpy(data['gt'][...]).cuda().float()

    else:
        lms = torch.from_numpy(data['lms'][...] / scale).float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / scale).float()  # CxHxW= 8x64x64
        pan = torch.from_numpy(data['pan'][...] / scale).float()  # HxW = 256x256
        # if data.get('gt', None) is None:
        #     gt = torch.from_numpy(data['lms'][...]).float()
        # else:
        #     gt = torch.from_numpy(data['gt'][...]).float()

    return {'lms': lms,
            'ms': ms,
            'pan': pan
            }

class MultiExmTest_h5_ZS(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, suffix='.h5'):
        super(MultiExmTest_h5_ZS, self).__init__()

        # self.scale = 2047.0
        # if 'gf' in dataset_name:
        #     self.scale = 1023.0
        self.img_scale = img_scale
        print(f"loading MultiExmTest_h5: {file_path} with {img_scale}")
        # 一次性载入到内存
        if 'hp' not in dataset_name:
            data = load_dataset_H5_ZS(file_path, img_scale, False)  # 测试数据读取时，与训练数据集不同，没有对gt进行归一化处理，而其他三个变量都进行了归一化

        # elif 'hp' in dataset_name:
        #     file_path = file_path.replace('_hp', '')
        #     data = load_dataset_H5_hp(file_path, img_scale, False)

        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError
        if suffix == '.mat':  # pan NxHxW   ms NxHxWxC
            self.lms = data['lms'].permute(0, 3, 1, 2)  # NxCxHxW = 8x256x256
            self.ms = data['ms'].permute(0, 3, 1, 2)  # NxCxHxW= 8x64x64
            self.pan = data['pan'].unsqueeze(1)  # pan Nx1xHxW
            # self.gt = data['gt'].permute(0, 3, 1, 2)
        else:
            self.lms = data['lms']
            self.ms = data['ms']  # NxCxHxW
            self.pan = data['pan']
            # self.gt = data['gt']  # NxHxWxC

        print(f"lms: {self.lms.shape}, ms: {self.ms.shape}, pan: {self.pan.shape}")

    def __getitem__(self, item):
        return {'lms': self.lms[item, ...],
                'ms': self.ms[item, ...],
                'pan': self.pan[item, ...]
                
                }

    def __len__(self):
        return self.ms.shape[0]


class MultiExmTest_h5_ZS_new(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, name,suffix='.h5'):
        super(MultiExmTest_h5_ZS_new, self).__init__()

        dataset = h5py.File(file_path, 'r')

        # print(f"loading Dataset_Pro: {file_path} with {img_scale}")

        ms = np.array(dataset['ms'][name], dtype=np.float32) / img_scale
        lms = np.array(dataset['lms'][name], dtype=np.float32) / img_scale
        pan = np.array(dataset['pan'][name], dtype=np.float32) / img_scale

        if dataset.get('gt', None) is None:
            gt = np.array(dataset['lms'][name], dtype=np.float32)
        else:
            gt = np.array(dataset['gt'][name], dtype=np.float32)

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()
        gt = torch.from_numpy(gt).float()


        self.gt_crops = gt

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms

        # LMS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.lms_crops = MS_crop(ms)
        self.lms_crops = lms

        # PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        # self.pan_crops = PAN_crop(pan)
        self.pan_crops = pan

        # print(f"lms: {self.lms_crops.shape}, ms: {self.ms_crops.shape}, pan: {self.pan_crops.shape}")


    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops, self.gt_crops

    def __len__(self):
        return 1


class MultiExmTest_h5_ZS_new_np(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, name,suffix='.h5'):
        super(MultiExmTest_h5_ZS_new_np, self).__init__()

        dataset = h5py.File(file_path, 'r')

        # print(f"loading Dataset_Pro: {file_path} with {img_scale}")

        ms = np.array(dataset['ms'][name], dtype=np.float32) / img_scale
        lms = np.array(dataset['lms'][name], dtype=np.float32) / img_scale
        pan = np.array(dataset['pan'][name], dtype=np.float32) / img_scale

        if dataset.get('gt', None) is None:
            gt = np.array(dataset['lms'][name], dtype=np.float32)
        else:
            gt = np.array(dataset['gt'][name], dtype=np.float32)

        I_PAN = np.transpose(pan*img_scale, (1, 2, 0))
        ratio =4
        pan_filt = interp23tap(resize(I_PAN, (int(I_PAN.shape[0] / ratio), int(I_PAN.shape[1] / ratio)), order=3,anti_aliasing=True), ratio)

        pan_filt = torch.from_numpy(pan_filt).float()


        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()
        gt = torch.from_numpy(gt).float()


        self.gt_crops = gt

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms

        # LMS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.lms_crops = MS_crop(ms)
        self.lms_crops = lms

        # PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        # self.pan_crops = PAN_crop(pan)
        self.pan_crops = pan

        # print(f"lms: {self.lms_crops.shape}, ms: {self.ms_crops.shape}, pan: {self.pan_crops.shape}")
        self.pan_filt = pan_filt

    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops, self.gt_crops, self.pan_filt

    def __len__(self):
        return 1




class MultiExmTest_h5_ZS_new_RR(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, name,suffix='.h5'):
        super(MultiExmTest_h5_ZS_new_RR, self).__init__()

        dataset = h5py.File(file_path, 'r')

        # print(f"loading Dataset_Pro: {file_path} with {img_scale}")

        ms = np.array(dataset['ms'][name], dtype=np.float32)
        lms = np.array(dataset['lms'][name], dtype=np.float32) 
        pan = np.array(dataset['pan'][name], dtype=np.float32) 

        if dataset.get('gt', None) is None:
            gt = np.array(dataset['lms'][name], dtype=np.float32)
        else:
            gt = np.array(dataset['gt'][name], dtype=np.float32)

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()
        gt = torch.from_numpy(gt).float()


        self.gt_crops = gt

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms

        # LMS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.lms_crops = MS_crop(ms)
        self.lms_crops = lms

        # PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        # self.pan_crops = PAN_crop(pan)
        self.pan_crops = pan

        # print(f"lms: {self.lms_crops.shape}, ms: {self.ms_crops.shape}, pan: {self.pan_crops.shape}")


    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops, self.gt_crops

    def __len__(self):
        return 1





class MultiExmTest_h5_ZS_new_DA(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, name,suffix='.h5'):
        super(MultiExmTest_h5_ZS_new_DA, self).__init__()

        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'][name], dtype=np.float32) / img_scale
        lms = np.array(dataset['lms'][name], dtype=np.float32) / img_scale
        pan = np.array(dataset['pan'][name], dtype=np.float32) / img_scale

        if dataset.get('gt', None) is None:
            gt = np.array(dataset['lms'][name], dtype=np.float32)
        else:
            gt = np.array(dataset['gt'][name], dtype=np.float32)

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()
        gt = torch.from_numpy(gt).float()


        self.gt_crops = gt

        MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        self.ms_crops = MS_crop(ms)
        # self.ms_crops = ms

        LMS_crop = torchvision.transforms.TenCrop(lms.shape[1] / 2)
        self.lms_crops = LMS_crop(lms)
        # self.lms_crops = lms

        PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        self.pan_crops = PAN_crop(pan)

        self.num_crops = len(self.ms_crops)

    def __getitem__(self, index):
        crop_index = index % self.num_crops # 确保索引在范围内
        return (self.ms_crops[crop_index], 
                self.lms_crops[crop_index], 
                self.pan_crops[crop_index])

    def __len__(self):
        return self.num_crops


class MultiExmTest_h5_ZS_new_DA_RR(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, name,suffix='.h5'):
        super(MultiExmTest_h5_ZS_new_DA_RR, self).__init__()

        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'][name], dtype=np.float32) / img_scale
        lms = np.array(dataset['lms'][name], dtype=np.float32) / img_scale
        pan = np.array(dataset['pan'][name], dtype=np.float32) / img_scale

        if dataset.get('gt', None) is None:
            gt = np.array(dataset['lms'][name], dtype=np.float32)
        else:
            gt = np.array(dataset['gt'][name], dtype=np.float32)

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()
        gt = torch.from_numpy(gt).float()


        # self.gt_crops = gt

        GT_crop = torchvision.transforms.TenCrop(gt.shape[1] / 2)
        self.gt_crops = GT_crop(gt)

        MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        self.ms_crops = MS_crop(ms)
        # self.ms_crops = ms

        LMS_crop = torchvision.transforms.TenCrop(lms.shape[1] / 2)
        self.lms_crops = LMS_crop(lms)
        # self.lms_crops = lms

        PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        self.pan_crops = PAN_crop(pan)

        self.num_crops = len(self.ms_crops)

    def __getitem__(self, index):
        crop_index = index % self.num_crops # 确保索引在范围内
        return (self.ms_crops[crop_index], 
                self.lms_crops[crop_index], 
                self.pan_crops[crop_index],
                self.gt_crops[crop_index])

    def __len__(self):
        return self.num_crops



def load_dataset_H5(file_path, scale, use_cuda=True):
    data = h5py.File(file_path)  # NxCxHxW
    print(data.keys())
    # tensor type:
    if use_cuda:
        lms = torch.from_numpy(data['lms'][...] / scale).cuda().float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / scale).cuda().float()  # CxHxW= 8x64x64
        pan = torch.from_numpy(data['pan'][...] / scale).cuda().float()  # HxW = 256x256

        gt = torch.from_numpy(data['gt'][...]).cuda().float()

    else:
        lms = torch.from_numpy(data['lms'][...] / scale).float()  # CxHxW = 8x64x64

        ms = torch.from_numpy(data['ms'][...] / scale).float()  # CxHxW= 8x64x64
        pan = torch.from_numpy(data['pan'][...] / scale).float()  # HxW = 256x256
        if data.get('gt', None) is None:
            gt = torch.from_numpy(data['lms'][...]).float()
        else:
            gt = torch.from_numpy(data['gt'][...]).float()

    return {'lms': lms,
            'ms': ms,
            'pan': pan,
            'gt': gt.permute([0, 2, 3, 1])}

class MultiExmTest_h5(Dataset):

    def __init__(self, file_path, dataset_name, img_scale, suffix='.h5'):
        super(MultiExmTest_h5, self).__init__()

        # self.scale = 2047.0
        # if 'gf' in dataset_name:
        #     self.scale = 1023.0
        self.img_scale = img_scale
        print(f"loading MultiExmTest_h5: {file_path} with {img_scale}")
        # 一次性载入到内存
        if 'hp' not in dataset_name:
            data = load_dataset_H5(file_path, img_scale, False)  # 测试数据读取时，与训练数据集不同，没有对gt进行归一化处理，而其他三个变量都进行了归一化

        # elif 'hp' in dataset_name:
        #     file_path = file_path.replace('_hp', '')
        #     data = load_dataset_H5_hp(file_path, img_scale, False)

        else:
            print(f"{dataset_name} is not supported in evaluation")
            raise NotImplementedError
        if suffix == '.mat':  # pan NxHxW   ms NxHxWxC
            self.lms = data['lms'].permute(0, 3, 1, 2)  # NxCxHxW = 8x256x256
            self.ms = data['ms'].permute(0, 3, 1, 2)  # NxCxHxW= 8x64x64
            self.pan = data['pan'].unsqueeze(1)  # pan Nx1xHxW
            self.gt = data['gt'].permute(0, 3, 1, 2)
        else:
            self.lms = data['lms']
            self.ms = data['ms']  # NxCxHxW
            self.pan = data['pan']
            self.gt = data['gt']  # NxHxWxC

        print(f"lms: {self.lms.shape}, ms: {self.ms.shape}, pan: {self.pan.shape}, gt: {self.gt.shape}")

    def __getitem__(self, item):
        return {'lms': self.lms[item, ...],
                'ms': self.ms[item, ...],
                'pan': self.pan[item, ...],
                'gt': self.gt[item, ...]
                }

    def __len__(self):
        return self.gt.shape[0]



# save flag and loss
def visualize(train_loss, valid_loss, network_name, dataset_name,record_dir):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, '-', label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, '--', label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='%s_%s Early Stopping Checkpoint' % (network_name, dataset_name))

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Non-blocking display of the plot 图形显示不会阻塞程序的执行
    plt.show(block=False)

    fig.savefig(join(record_dir, 'loss_plot_%s_%s.png' % (network_name, dataset_name)), bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close(fig)


def visualize_new(train_loss, valid_loss, network_name, dataset_name,record_dir,name):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, '-', label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, '--', label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='%s_%s Early Stopping Checkpoint' % (network_name, dataset_name))

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Non-blocking display of the plot 图形显示不会阻塞程序的执行
    plt.show(block=False)

    fig.savefig(join(record_dir, 'loss_plot_%s_%s_%s.png' % (network_name, dataset_name,name)), bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close(fig)





def visualize2(train_loss, valid_loss, network_name, dataset_name,record_dir,lr):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, '-', label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, '--', label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='%s_%s Early Stopping Checkpoint' % (network_name, dataset_name))

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Non-blocking display of the plot 图形显示不会阻塞程序的执行
    plt.show(block=False)

    fig.savefig(join(record_dir, 'loss_plot_%s_%s_%s.png' % (network_name, dataset_name,lr)), bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close(fig)



# functions for save
def linstretch(images, tol=None):
    '''
    NM = N*M;
    for i=1:3
        b = reshape(double(uint16(ImageToView(:,:,i))),NM,1);
        [hb,levelb] = hist(b,max(b)-min(b));
        chb = cumsum(hb);#沿第一个非单一维运算。matlab矩阵顺序 HxWxC,列的累计和
        t(1)=ceil(levelb(find(chb>NM*tol(i,1), 1 )));
        t(2)=ceil(levelb(find(chb<NM*tol(i,2), 1, 'last' )));
        %t(2) = 1;
        b(b<t(1))=t(1);
        b(b>t(2))=t(2);
        b = (b-t(1))/(t(2)-t(1));
        ImageToView(:,:,i) = reshape(b,N,M);
    end
    '''

    if tol is None:
        tol = [0.01, 0.99]
    if images.ndim == 3:
        h, w, channels = images.shape
    else:
        images = np.expand_dims(images, axis=-1)
        h, w, channels = images.shape
    N = h * w
    for c in range(channels):
        image = np.float32(np.round(images[:, :, c])).reshape(N, 1)
        hb, levelb = np.histogram(image, bins=math.ceil(image.max() - image.min()))
        chb = np.cumsum(hb, 0)
        levelb_center = levelb[:-1] + (levelb[1] - levelb[0]) / 2
        lbc_min, lbc_max = levelb_center[chb > N * tol[0]][0], levelb_center[chb < N * tol[1]][-1]
        image = np.clip(image, a_min=lbc_min, a_max=lbc_max)
        image = (image - lbc_min) / (lbc_max - lbc_min)
        images[..., c] = np.reshape(image, (h, w))

    images = np.squeeze(images)

    return images

def showimage8(images, unnormlize=2047.0, first_channel=False):
    assert images.shape[1] >= 3, print("input images format is not suitable")

    if isinstance(images, torch.Tensor):
        unnormlize = np.where(max(float(torch.max(images)), 1.0) > 1.0, 1.0, unnormlize)
        if first_channel:
            images = images.permute(1, 2, 0)
        output = images[..., [0, 2, 4]] * torch.tensor(unnormlize)
        output = torch.clamp(output, 0, 2047)
        output = output.cpu().detach().numpy()

    norm_image = linstretch(output)
    return norm_image[:, :, ::-1]

def showimage8_new(images, unnormlize, dataset_name, first_channel=False):
    assert images.shape[1] >= 3, print("input images format is not suitable")
    unnormlize2 = int(unnormlize)

    if isinstance(images, torch.Tensor):
        unnormlize = np.where(max(float(torch.max(images)), 1.0) > 1.0, 1.0, unnormlize)
        if first_channel:
            images = images.permute(1, 2, 0)

        if dataset_name == 'wv3' or dataset_name == 'wv2': # 默认只有'wv3'/'wv2'是8维，其他都是4维
           output = images[..., [0, 2, 4]] * torch.tensor(unnormlize)
        else:
           output = images[..., [0, 1, 2]] * torch.tensor(unnormlize)

        output = torch.clamp(output, 0, unnormlize2)
        output = output.cpu().detach().numpy()

    norm_image = linstretch(output)
    return norm_image[:, :, ::-1]


# save output(mat/png)
def save_results2_new(idx, save_model_output, output,network_name,dataset_name,other):
    save_name = os.path.join(f"{save_model_output}","{}_{}_{}_{}.mat".format(network_name,dataset_name,other,idx))  #FusionNet_wv3_rs_ny_
    sio.savemat(save_name, {'sr': output.cpu().detach().numpy()})

    if dataset_name == 'wv3' or dataset_name == 'wv2':
        output = output[..., [0, 2, 4]]
    else:
        output = output[..., [0, 1, 2]]
    output = output.cpu().detach().numpy()
    output = linstretch(output)
    output = output[:, :, ::-1]

    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}__{}.png".format(network_name,dataset_name,other,idx))
    image.save(save_name2)

def save_results2_new2(idx, save_model_output, output,network_name,st_name,dataset_name,other):
    save_name = os.path.join(f"{save_model_output}","{}_{}_{}_{}.mat".format(st_name,dataset_name,other,idx))  #FusionNet_wv3_rs_ny_
    sio.savemat(save_name, {'sr': output.cpu().detach().numpy()})

    if dataset_name == 'wv3' or dataset_name == 'wv2':
        output = output[..., [0, 2, 4]]
    else:
        output = output[..., [0, 1, 2]]
    output = output.cpu().detach().numpy()
    output = linstretch(output)
    output = output[:, :, ::-1]

    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}_{}_{}.png".format(st_name,network_name,dataset_name,other,idx))
    image.save(save_name2)


def save_results2_new_RR(idx, save_model_output, output,network_name,ST_name,dataset_name,other):
    save_name = os.path.join(f"{save_model_output}","{}_{}_{}_{}.mat".format(ST_name,dataset_name,other,idx))  #FusionNet_wv3_rs_ny_
    sio.savemat(save_name, {'sr': output})

    if dataset_name == 'wv3' or dataset_name == 'wv2':
        output = output[..., [0, 2, 4]]
    else:
        output = output[..., [0, 1, 2]]
    # output = output
    output = linstretch(output)
    output = output[:, :, ::-1]

    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}_{}_{}.png".format(ST_name,network_name,dataset_name,other,idx))
    image.save(save_name2)


# # Test(HxWxC)_qb_data_fr86.mat
def save_results2_new_RR_mat(idx, save_model_output, gt,ms,lms,pan,dataset_name):
    save_name = os.path.join(f"{save_model_output}","Test(HxWxC)_{}_data{}.mat".format(dataset_name,idx))  #FusionNet_wv3_rs_ny_
    sio.savemat(save_name, {'gt': gt,'ms':ms,'lms':lms,'pan':pan})




def save_results5_new(idx, save_model_output, output,network_name,dataset_name,other):
    # save_name = os.path.join(f"{save_model_output}","{}_{}_{}_{}.mat".format(network_name,dataset_name,other,idx))  #FusionNet_wv3_rs_ny_
    # sio.savemat(save_name, {'sr': output.cpu().detach().numpy()})

    if dataset_name == 'wv3' or dataset_name == 'wv2':
        output = output[..., [0, 2, 4]]
    else:
        output = output[..., [0, 1, 2]]
    output = output.cpu().detach().numpy()
    output = linstretch(output)
    output = output[:, :, ::-1]

    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}_ms_{}.png".format(network_name,dataset_name,other,idx))
    image.save(save_name2)

def save_results6_new(idx, save_model_output, output,network_name,dataset_name,other):
    # save_name = os.path.join(f"{save_model_output}","{}_{}_{}_{}.mat".format(network_name,dataset_name,other,idx))  #FusionNet_wv3_rs_ny_
    # sio.savemat(save_name, {'sr': output.cpu().detach().numpy()})

    if dataset_name == 'wv3' or dataset_name == 'wv2':
        output = output[..., [0, 2, 4]]
    else:
        output = output[..., [0, 1, 2]]
    output = output.cpu().detach().numpy()
    output = linstretch(output)
    output = output[:, :, ::-1]

    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}_ms_hp_{}.png".format(network_name,dataset_name,other,idx))
    image.save(save_name2)


def save_results2(idx, save_model_output, output,network_name,dataset_name,other):
    save_name = os.path.join(f"{save_model_output}","{}_{}_{}_{}.mat".format(network_name,dataset_name,other,idx))  #FusionNet_wv3_rs_ny_
    sio.savemat(save_name, {'sr': output.cpu().detach().numpy()})

    output = showimage8(output)
    # print(type(output), output.max(), output.min(), output.dtype)
    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}__{}.png".format(network_name,dataset_name,other,idx))
    image.save(save_name2)

# save gt(png)
def save_results3_new(idx, save_model_output, output,network_name,dataset_name,other):
    # save_name = os.path.join(f"{save_model_output}","{}_{}_{}_gt_{}.mat".format(network_name,dataset_name,other,idx))
    # sio.savemat(save_name, {'gt': output.cpu().detach().numpy()})

    if dataset_name == 'wv3' or dataset_name == 'wv2':
        output = output[..., [0, 2, 4]]
    else:
        output = output[..., [0, 1, 2]]
    output = output.cpu().detach().numpy()
    output = linstretch(output)
    output = output[:, :, ::-1]

    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}_gt_{}.png".format(network_name,dataset_name,other,idx))
    image.save(save_name2)

def save_results8_new(idx, save_model_output, output,network_name,dataset_name,other):
    # save_name = os.path.join(f"{save_model_output}","{}_{}_{}_lms_{}.mat".format(network_name,dataset_name,other,idx))
    # sio.savemat(save_name, {'gt': output.cpu().detach().numpy()})

    if dataset_name == 'wv3' or dataset_name == 'wv2':
        output = output[..., [0, 2, 4]]
    else:
        output = output[..., [0, 1, 2]]
    output = output.cpu().detach().numpy()
    output = linstretch(output)
    output = output[:, :, ::-1]

    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}_lms_{}.png".format(network_name,dataset_name,other,idx))
    image.save(save_name2)

def save_results3(idx, save_model_output, output,network_name,dataset_name,other):
    save_name = os.path.join(f"{save_model_output}","{}_{}_{}_gt_{}.mat".format(network_name,dataset_name,other,idx))
    sio.savemat(save_name, {'gt': output.cpu().detach().numpy()})

    output = showimage8(output)
    # print(type(output), output.max(), output.min(), output.dtype)
    output*=255
    output = output.astype(np.uint8)
    image = Image.fromarray(output)
    save_name2 = os.path.join(f"{save_model_output}", "{}_{}_{}_gt__{}.png".format(network_name,dataset_name,other,idx))
    image.save(save_name2)


# Initialization 
def truncated_normal_(tensor, mean=0.0, std=1.0):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm
    def calculate_fan(shape, factor=2.0, mode='FAN_IN', uniform=False):
        # 64 9 3 3 -> 3 3 9 64
        # 64 64 3 3 -> 3 3 64 64
        if shape:
            # fan_in = float(shape[1]) if len(shape) > 1 else float(shape[0])
            # fan_out = float(shape[0])
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == 'FAN_IN':
            # Count only number of input connections.
            n = fan_in
        elif mode == 'FAN_OUT':
            # Count only number of output connections.
            n = fan_out
        elif mode == 'FAN_AVG':
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        if uniform:
            raise NotImplemented
            # # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            # limit = math.sqrt(3.0 * factor / n)
            # return random_ops.random_uniform(shape, -limit, limit,
            #                                  dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = math.sqrt(1.3 * factor / n)
        return fan_in, fan_out, trunc_stddev

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        # fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        x = x.permute(3, 2, 1, 0)  # .permute(2, 3, 1, 0)
        fan_in, fan_out, trunc_stddev = calculate_fan(x.shape)
        # print(trunc_stddev) # debug
        # if mode == "fan_in":
        #     scale /= max(1., fan_in)
        # elif mode == "fan_out":
        #     scale /= max(1., fan_out)
        # else:
        #     scale /= max(1., (fan_in + fan_out) / 2.)
        # if distribution == "normal" or distribution == "truncated_normal":
        #     # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        #     stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, trunc_stddev)  # 0.001)
        x = x.permute(3, 2, 0, 1)
        # print(x.min(), x.max())) # debug
        return x  # /10*1.28

    variance_scaling(tensor)

    return tensor
