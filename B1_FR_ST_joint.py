# coding=utf-8

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
import metrics7_6 as mtc7_6
from function import *
from network import *
from wald_utilities import *
from torch.autograd import Variable


# img_rang = 2047 for 11-bit WV3, WV4 and QB data; img_range = 1023 for 10-bit GF2 data
# nbands波段数，影响网络的第一层和最后一层通道数   WV3是8；QB、GF2是4
# 数据集命名 train/valid_{dataset_name}.h5、test_{dataset_name}_RR/FR.h5
# 网络输出结果保存：FusionNet_wv3_rs_ny_  其中origin_test时other为os_ny，否则为rs_ny

def run_experiment(network_name, dataset_name, train_batch_size, n_epochs, patience, lr, log_freq):
    import sys
    from contextlib import redirect_stdout

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, message):
            for f in self.files:
                f.write(message)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    record_dir = './results/%s_ZS/ST-joint/record_FR_%s_%s_%s_MTF_QS/' % (network_name, network_name, dataset_name, lr)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    log_file_path = rf"{record_dir}{network_name}_{lr}_{n_epochs}_{patience}_MTF_QS_new2.txt"

    # 打开文件并重定向 stdout
    with open(log_file_path, 'w') as f:
        with redirect_stdout(Tee(sys.stdout, f)):

            print(
                "Running {} on {} with batch size {}, {} epochs, patience {}, learning rate {}, log frequency {}".format(
                    network_name, dataset_name, train_batch_size, n_epochs, patience, lr, log_freq))

            # 存储所有性能指标
            performance_metrics = {
                'D_lambda': [],
                'D_s': [],
                'QNR': [],
                'Q2n': [],
                'SAM': [],
                'ERGAS': [],
                'time_FR_train': [],
                'time_FR_test': [],
                'time_RR_test': []

            }

            # 根据 dataset_name 设置 range 的起始值和结束值
            if dataset_name == 'wv3':
                start, end = 271, 291
                # start, end = 271,273

                if network_name == 'FusionNet':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                elif network_name == 'APNN':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                elif network_name == 'PNN':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                elif network_name == 'PanNet':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                else:
                    raise ValueError(f"Unknown network_name: {network_name}")


            elif dataset_name == 'qb':
                start, end = 0, 20
                # start, end = 0, 2

                if network_name == 'FusionNet':
                    alpha = 1  # QNR loss
                    beta = 5  # pixel loss
                elif network_name == 'APNN':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                elif network_name == 'PNN':
                    alpha = 1  # QNR loss
                    beta = 5  # pixel loss
                elif network_name == 'PanNet':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                else:
                    raise ValueError(f"Unknown network_name: {network_name}")

            elif dataset_name == 'gf2':
                # start, end = 134, 154
                start, end = 131, 132

                if network_name == 'FusionNet':
                    alpha = 1  # QNR loss
                    beta = 20  # pixel loss
                elif network_name == 'APNN':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                elif network_name == 'PNN':
                    alpha = 1  # QNR loss
                    beta = 5  # pixel loss
                elif network_name == 'PanNet':
                    alpha = 1  # QNR loss
                    beta = 5  # pixel loss
                else:
                    raise ValueError(f"Unknown network_name: {network_name}")

            elif dataset_name == 'wv2':
                start, end = 0, 20
                # start, end = 0,2

                if network_name == 'FusionNet':
                    alpha = 1  # QNR loss
                    beta = 5  # pixel loss
                elif network_name == 'APNN':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                elif network_name == 'PNN':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                elif network_name == 'PanNet':
                    alpha = 1  # QNR loss
                    beta = 10  # pixel loss
                else:
                    raise ValueError(f"Unknown network_name: {network_name}")


            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            for name in range(start, end):

                ############################################################
                # 超参数设置
                ############################################################

                ## 超参数设置
                gpu = '0'  # 选择用哪张GPU卡
                flag = True  # visualize
                sampler = None
                dim_cut = 21

                lr_decay_freq = 500  # lr衰减 1/10

                print("Training on sample {}, with alpha {}, beta {}".format(name + 1, alpha, beta))

                def get_dataset_properties(dataset_name):
                    if dataset_name in ['wv3', 'wv2']:
                        img_range = 2047.0
                        nbands = 8
                    elif dataset_name == 'qb':
                        img_range = 2047.0
                        nbands = 4
                    elif dataset_name == 'gf2':
                        img_range = 1023.0
                        nbands = 4
                    else:
                        raise ValueError("Unknown dataset name: {}".format(dataset_name))

                    return img_range, nbands

                img_range, nbands = get_dataset_properties(dataset_name)

                def get_sensor(dataset_name):
                    if dataset_name == 'wv3':
                        sensor = 'WV3'
                    elif dataset_name == 'wv2':
                        sensor = 'WV2'
                    elif dataset_name == 'qb':
                        sensor = 'QB'
                    elif dataset_name == 'gf2':
                        sensor = 'GF2'
                    else:
                        raise ValueError("Unknown dataset name: {}".format(dataset_name))

                    return sensor

                sensor = get_sensor(dataset_name)

                def create_network(network_name, spectral_num):
                    if network_name == 'DiCNN':
                        return DiCNN(spectral_num=spectral_num)
                    elif network_name == 'DRPNN':
                        return DRPNN(spectral_num=spectral_num)
                    elif network_name == 'MSDCNN':
                        return MSDCNN(spectral_num=spectral_num)
                    elif network_name == 'FusionNet':
                        return FusionNet(spectral_num=spectral_num)
                    elif network_name == 'APNN':
                        return APNN(spectral_num=spectral_num)
                    elif network_name == 'PNN':
                        return PNN(spectral_num=spectral_num)
                    elif network_name == 'PanNet':
                        return PanNet(spectral_num=spectral_num)
                    else:
                        raise ValueError(f"Unknown network name: {network_name}")

                ## 文件夹设置
                data_dir = './data/test_data'
                # record_dir = './results/%s_ZS/record_RR_%s_%s/' % (network_name, network_name, dataset_name)
                model_dir = './results/%s_ZS/ST-joint/model_FR_%s_%s_%s_MTF_QS_%s/' % (
                network_name, network_name, dataset_name, lr, name)
                net_checkpoint_path = join(model_dir, 'net_checkpoint_FR_%s_%s_%s_MTF_QS_%s_final.pth' % (
                network_name, dataset_name, lr, name))
                net_checkpoint_path_test = join(model_dir, 'net_checkpoint_FR_%s_%s_%s_MTF_QS_%s_final.pth' % (
                network_name, dataset_name, lr, name))
                # testsample_dir = './results/%s_ZS/test_samples_FR_%s_%s_%s/' % (network_name, network_name, dataset_name,lr)
                origin_testsample_dir = './results/%s_ZS/ST-joint/origin_test_samples_FR_%s_%s_%s_MTF_QS/' % (
                network_name, network_name, dataset_name, lr)
                testsample_dir = './results/%s_ZS/ST-joint/test_samples_RR_%s_%s_%s_MTF_QS/' % (
                network_name, network_name, dataset_name, lr)

                ## 创建文件夹
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                if not os.path.exists(testsample_dir):
                    os.makedirs(testsample_dir)
                if not os.path.exists(origin_testsample_dir):
                    os.makedirs(origin_testsample_dir)

                ## Device configuration
                os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu
                print(torch.cuda.is_available())
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                print(device)

                # set seed
                seed = 0
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
                np.random.seed(seed)  # Numpy module.
                random.seed(seed)  # Python random module.
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

                ############################################################
                # load data
                ############################################################

                training_dataset = Dataset_Pro_ZS_new_np('/'.join([data_dir, f'TestData_{dataset_name}_fr.h5']),
                                                         img_scale=img_range, name=name)
                training_dataloaders = DataLoader(dataset=training_dataset, num_workers=0, batch_size=train_batch_size,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  drop_last=True)  # put training data to DataLoader for batches
                validation_dataset = Dataset_Pro_ZS_new_np('/'.join([data_dir, f'TestData_{dataset_name}_fr.h5']),
                                                           img_scale=img_range, name=name)
                validation_dataloaders = DataLoader(validation_dataset, batch_size=train_batch_size,
                                                    shuffle=False, num_workers=1, drop_last=False, sampler=sampler)

                origin_test_dataset = MultiExmTest_h5_ZS_new_np('/'.join([data_dir, f'TestData_{dataset_name}_fr.h5']),
                                                                dataset_name,
                                                                img_scale=img_range, name=name)
                origin_test_dataloaders = DataLoader(origin_test_dataset, batch_size=1,
                                                     shuffle=False, num_workers=1, drop_last=False, sampler=sampler)

                ############################################################
                # load network
                ############################################################

                net = create_network(network_name, spectral_num=nbands).to(device)

                mae = nn.L1Loss().to(device)
                # mae = nn.MSELoss().to(device)

                # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)  #PNN
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)  # weight_decay的值根据网络类型而定

                def adjust_learning_rate(lr, epoch, freq):
                    """Sets the learning rate to the initial LR decayed by 10"""
                    lr = lr * (0.1 ** (epoch // freq))
                    return lr

                if (torch.cuda.device_count() > 1):
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    net = nn.DataParallel(net)


                def net_eval_origin_new2(dataloader, model, sample_dir, name, epoch):
                    model.eval()

                    i = 0
                    D_lambda = []
                    D_s = []
                    QNR = []

                    with torch.no_grad():
                        for iteration, batch in enumerate(dataloader, 1):
                            i += 1
                            ms, lms, pan, pan_filt = Variable(batch[0]).to(device), \
                                Variable(batch[1]).to(device), \
                                Variable(batch[2]).to(device), \
                                Variable(batch[4]).to(device)

                            psh = model(lms, pan)

                            if network_name == 'PNN':
                                psh = psh
                            else:
                                psh = psh + lms

                            result_our = torch.squeeze(psh).permute(1, 2, 0)
                            result_our = torch.clip(result_our, 0, 1)
                            x = result_our * img_range

                            blur = torch.squeeze(ms).permute(1, 2, 0)
                            blur = blur * img_range

                            pan = pan[0].permute(1, 2, 0)
                            pan = pan * img_range

                            blur2 = torch.squeeze(lms).permute(1, 2, 0)
                            blur2 = blur2 * img_range

                            pan_filt = pan_filt[0]

                            QNR_index, D_lambda_index, D_s_index = mtc7_6.QS(x, blur2, blur, pan, pan_filt, img_range,
                                                                             4, device, sensor=sensor, S=32)

                            D_lambda.append(D_lambda_index.cpu().detach().numpy())
                            D_s.append(D_s_index.cpu().detach().numpy())
                            QNR.append(QNR_index.cpu().detach().numpy())

                    print("D_lambda: %.4lf±%.4lf" % (np.mean(D_lambda), np.std(D_lambda)))
                    print("D_s: %.4lf±%.4lf" % (np.mean(D_s), np.std(D_s)))
                    print("QNR: %.4lf±%.4lf" % (np.mean(QNR), np.std(QNR)))

                def net_eval_new2(dataloader, model, sample_dir, name, epoch):
                    model.eval()

                    i = 0
                    SAM = []
                    Q2n = []
                    ERGAS = []

                    with torch.no_grad():
                        # for data in dataloader:
                        for iteration, batch in enumerate(dataloader, 1):
                            i += 1
                            ms, lms, pan = Variable(batch[0]).to(device), \
                                Variable(batch[1]).to(device), \
                                Variable(batch[2]).to(device)

                            lrms_lr = wald_protocol_v1(lms, pan, 4, sensor, nbands)
                            pan_lr = wald_protocol_v2(ms, pan, 4, sensor, nbands)

                            psh_lr = net(lrms_lr, pan_lr)

                            if network_name == 'PNN':
                                psh_lr = psh_lr
                            else:
                                psh_lr = psh_lr + lrms_lr

                            result_our = torch.squeeze(psh_lr).permute(1, 2, 0)
                            result_our = torch.clip(result_our, 0, 1)
                            x = result_our * img_range

                            y = ms.cuda().squeeze(0).permute(1, 2, 0)
                            y = y * img_range

                            x = x.cpu().detach().numpy()
                            y = y.cpu().detach().numpy()

                            x = x[dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut, :]
                            y = y[dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut, :]

                            Q2n.append(mtc.q2n(y, x))
                            ERGAS.append(mtc.ERGAS(y, x))
                            SAM.append(mtc.SAM(y, x) * 180 / np.pi)

                    SAM = np.array(SAM)
                    ERGAS = np.array(ERGAS)
                    Q2n = np.array(Q2n)

                    print("epoch: %s" % epoch)
                    print("Q2n: %.4lf±%.4lf" % (np.mean(Q2n), np.var(Q2n)))
                    print("ERGAS: %.4lf±%.4lf " % (np.mean(ERGAS), np.var(ERGAS)))
                    print("SAM: %.4lf±%.4lf" % (np.mean(SAM), np.var(SAM)))

                ############################################################
                # train
                ############################################################

                def net_train_new(net, patience, n_epochs, lr):
                    print('Begin net Training!')
                    train_losses, valid_losses = [], []
                    avg_train_losses, avg_valid_losses = [], []
                    counter = 0
                    best_val_loss = np.inf
                    time_sum = 0
                    # start = time.time()  # 记录每轮训练的开始时刻

                    # net_train_loss_record = open('%s/net_train_loss_record_%s_%s.txt' % (record_dir, network_name, dataset_name), "w")
                    for epoch in range(1, n_epochs + 1):
                        start = time.time()  # 记录每轮训练的开始时刻
                        net.train()
                        i = 0

                        lr = adjust_learning_rate(lr, epoch - 1, lr_decay_freq)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        for iteration, batch in enumerate(training_dataloaders, 1):
                            i += 1
                            optimizer.zero_grad()
                            ms, lms, pan, pan_filt = Variable(batch[0]).to(device), \
                                Variable(batch[1]).to(device), \
                                Variable(batch[2]).to(device), \
                                Variable(batch[3]).to(device)

                            lrms = lms  # torch.Size([1, 4, 512, 512])

                            psh = net(lrms, pan)
                            if network_name == 'PNN':
                                psh = psh
                            else:
                                psh = psh + lms

                            lrms_lr = wald_protocol_v1(lms, pan, 4, sensor, nbands)
                            pan_lr = wald_protocol_v2(ms, pan, 4, sensor, nbands)

                            psh_lr = net(lrms_lr, pan_lr)

                            if network_name == 'PNN':
                                psh_lr = psh_lr
                            else:
                                psh_lr = psh_lr + lrms_lr

                            pixel_loss = mae(psh_lr, ms)

                            result_our = torch.squeeze(psh).permute(1, 2, 0)
                            result_our = torch.clip(result_our, 0, 1)
                            x = result_our * img_range

                            blur = torch.squeeze(ms).permute(1, 2, 0)
                            blur = blur * img_range

                            pan = pan[0].permute(1, 2, 0)
                            pan = pan * img_range

                            blur2 = torch.squeeze(lms).permute(1, 2, 0)
                            blur2 = blur2 * img_range

                            pan_filt = pan_filt[0]

                            QNR_val, D_lambda_val, D_s_val = mtc7_6.QS(x, blur2, blur, pan, pan_filt, img_range, 4,
                                                                       device, sensor=sensor, S=32)

                            QNR_loss = 1 - QNR_val

                            # 观察a和b的值，当迭代5-10轮后，如果这两个值趋于等量，说明超参数alpha和beta合适
                            a = alpha * QNR_loss
                            b = beta * pixel_loss

                            loss = alpha * QNR_loss + beta * pixel_loss

                            train_losses.append(loss.item())
                            loss.backward()
                            optimizer.step()

                        epoch_time = (time.time() - start)
                        time_sum += epoch_time

                        net.eval()
                        with torch.no_grad():
                            for iteration, batch in enumerate(validation_dataloaders, 1):
                                ms, lms, pan, pan_filt = Variable(batch[0]).to(device), \
                                    Variable(batch[1]).to(device), \
                                    Variable(batch[2]).to(device), \
                                    Variable(batch[3]).to(device)

                                lrms = lms  # torch.Size([1, 4, 512, 512])

                                psh = net(lrms, pan)

                                if network_name == 'PNN':
                                    psh = psh
                                else:
                                    psh = psh + lms

                                lrms_lr = wald_protocol_v1(lms, pan, 4, sensor, nbands)
                                pan_lr = wald_protocol_v2(ms, pan, 4, sensor, nbands)

                                psh_lr = net(lrms_lr, pan_lr)

                                if network_name == 'PNN':
                                    psh_lr = psh_lr
                                else:
                                    psh_lr = psh_lr + lrms_lr

                                pixel_loss = mae(psh_lr, ms)

                                result_our = torch.squeeze(psh).permute(1, 2, 0)
                                result_our = torch.clip(result_our, 0, 1)
                                x = result_our * img_range

                                blur = torch.squeeze(ms).permute(1, 2, 0)
                                blur = blur * img_range

                                pan = pan[0].permute(1, 2, 0)
                                pan = pan * img_range

                                blur2 = torch.squeeze(lms).permute(1, 2, 0)
                                blur2 = blur2 * img_range

                                pan_filt = pan_filt[0]

                                QNR_val, D_lambda_val, D_s_val = mtc7_6.QS(x, blur2, blur, pan, pan_filt, img_range, 4,
                                                                           device, sensor=sensor, S=32)

                                QNR_loss = 1 - QNR_val

                                loss = alpha * QNR_loss + beta * pixel_loss

                                valid_losses.append(loss.item())

                        train_loss = np.average(train_losses)
                        valid_loss = np.average(valid_losses)
                        avg_train_losses.append(train_loss)
                        avg_valid_losses.append(valid_loss)

                        # 检查并保存最佳模型
                        if valid_loss < best_val_loss:
                            best_val_loss = valid_loss
                            torch.save(net.state_dict(), net_checkpoint_path)
                            counter = 0
                        else:
                            counter += 1
                            if counter >= patience:
                                print("Early stopping triggered")
                                break

                        if (epoch + 1) % log_freq == 0:
                            print('\n')
                            print_msg = (
                                    f'Net Training [{epoch}/{n_epochs}] ' + f'train_loss: {train_loss:.10f} ' + f'valid_loss: {valid_loss:.10f}\n')
                            print(print_msg)

                            net.load_state_dict(torch.load(net_checkpoint_path))  # 及时修改net_checkpoint_path_test
                            pretrained_net2 = net
                            origin_testsample_dir2 = './results/%s_ZS/ST-joint/origin_test_samples_%s_%s_%s_%s_MTF_QS_%s/' % (
                            network_name, network_name, dataset_name, epoch, lr, name)
                            testsample_dir2 = './results/%s_ZS/ST-joint/test_samples_%s_%s_%s_%s_MTF_QS_%s/' % (
                            network_name, network_name, dataset_name, epoch, lr, name)

                            net_checkpoint_path2 = join(model_dir, 'net_checkpoint_%s_%s_%s_%s_MTF_QS_%s.pth' % (
                            network_name, dataset_name, epoch, lr, name))
                            torch.save(net.state_dict(), net_checkpoint_path2)
                            net_eval_origin_new2(origin_test_dataloaders, pretrained_net2, origin_testsample_dir2,
                                                 'origin_test_fused', epoch)
                            net_eval_new2(origin_test_dataloaders, pretrained_net2, testsample_dir2, 'test_fused',
                                          epoch)

                        train_losses = []
                        valid_losses = []

                    time_sum = time_sum
                    print('\n')
                    print("time_FR_train: %.4lf" % time_sum)

                    net.load_state_dict(torch.load(net_checkpoint_path))
                    print('Finished train!')
                    return net, avg_train_losses, avg_valid_losses, time_sum

                ########################## only test 时注释

                pretrained_net, train_loss, valid_loss, time_train = net_train_new(net, patience, n_epochs, lr)

                performance_metrics['time_FR_train'].append(time_train)

                # save flag and loss
                if flag:
                    visualize_new(train_loss, valid_loss, network_name, dataset_name, record_dir, name)

                train_loss, valid_loss = np.array(train_loss), np.array(valid_loss)
                np.save('%s/net_avg_train_losses_record_%s_%s_%s_%s.npy' % (
                record_dir, network_name, dataset_name, lr, name), train_loss)
                np.save('%s/net_avg_valid_losses_record_%s_%s_%s_%s.npy' % (
                record_dir, network_name, dataset_name, lr, name), valid_loss)

                ############################################################
                # test
                ############################################################

                def net_eval_origin_new3(dataloader, model, sample_dir, name2):
                    model.eval()
                    print('Save %s......' % name2)

                    i = 0

                    with torch.no_grad():
                        for iteration, batch in enumerate(dataloader, 1):
                            i += 1
                            ms, lms, pan, pan_filt = Variable(batch[0]).to(device), \
                                Variable(batch[1]).to(device), \
                                Variable(batch[2]).to(device), \
                                Variable(batch[4]).to(device)

                            t1 = time.time()
                            psh = model(lms, pan)

                            if network_name == 'PNN':
                                psh = psh
                            else:
                                psh = psh + lms

                            t2 = time.time()

                            result_our = torch.squeeze(psh).permute(1, 2, 0)
                            result_our = torch.clip(result_our, 0, 1)
                            x = result_our * img_range

                            blur = torch.squeeze(ms).permute(1, 2, 0)
                            blur = blur * img_range

                            pan = pan[0].permute(1, 2, 0)
                            pan = pan * img_range

                            blur2 = torch.squeeze(lms).permute(1, 2, 0)
                            blur2 = blur2 * img_range

                            pan_filt = pan_filt[0]

                            QNR_index, D_lambda_index, D_s_index = mtc7_6.QS(x, blur2, blur, pan, pan_filt, img_range,
                                                                             4, device, sensor=sensor, S=32)

                            D_lambda = D_lambda_index.cpu().detach().numpy()
                            D_s = D_s_index.cpu().detach().numpy()
                            QNR = QNR_index.cpu().detach().numpy()

                            st_name = 'ST-joint'
                            save_results2_new2(name + 1, sample_dir, x, network_name, st_name, dataset_name, 'os')


                    time_FR_test = t2 - t1

                    print("time_FR_test: %.4lf" % time_FR_test)
                    print("D_lambda: %.4lf" % D_lambda)
                    print("D_s: %.4lf" % D_s)
                    print("QNR: %.4lf" % QNR)

                    return {
                        'time_FR_test': time_FR_test,
                        'D_lambda': D_lambda,
                        'D_s': D_s,
                        'QNR': QNR,
                    }

                def net_eval_new3(dataloader, model, sample_dir, name2):
                    model.eval()
                    print('Save %s......' % name2)

                    i = 0

                    with torch.no_grad():
                        for iteration, batch in enumerate(dataloader, 1):
                            i += 1
                            ms, lms, pan = Variable(batch[0]).to(device), \
                                Variable(batch[1]).to(device), \
                                Variable(batch[2]).to(device)

                            lrms_lr = wald_protocol_v1(lms, pan, 4, sensor, nbands)
                            pan_lr = wald_protocol_v2(ms, pan, 4, sensor, nbands)

                            t1 = time.time()
                            psh_lr = net(lrms_lr, pan_lr)
                            # psh_lr = psh_lr + lrms_lr
                            if network_name == 'PNN':
                                psh_lr = psh_lr
                            else:
                                psh_lr = psh_lr + lrms_lr

                            t2 = time.time()

                            result_our = torch.squeeze(psh_lr).permute(1, 2, 0)
                            result_our = torch.clip(result_our, 0, 1)
                            x = result_our * img_range

                            y = ms.cuda().squeeze(0).permute(1, 2, 0)
                            y = y * img_range

                            x = x.cpu().detach().numpy()
                            y = y.cpu().detach().numpy()

                            st_name = 'ST-joint'
                            save_results2_new_RR(name + 1, sample_dir, x, network_name, st_name, dataset_name, 'rs')

                            x = x[dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut, :]
                            y = y[dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut, :]

                            Q2n = mtc.q2n(y, x)
                            ERGAS = mtc.ERGAS(y, x)
                            SAM = mtc.SAM(y, x) * 180 / np.pi


                    time_RR_test = t2 - t1

                    print("time_RR_test: %.4lf" % time_RR_test)
                    print("Q2n: %.4lf" % Q2n)
                    print("ERGAS: %.4lf" % ERGAS)
                    print("SAM: %.4lf" % SAM)

                    return {
                        'time_RR_test': time_RR_test,
                        'Q2n': Q2n,
                        'ERGAS': ERGAS,
                        'SAM': SAM,
                    }

                # ########################### only test 时取消注释
                # net.load_state_dict(torch.load(net_checkpoint_path_test))   #及时修改net_checkpoint_path_test
                # pretrained_net = net
                # ###########################

                metrics = net_eval_origin_new3(origin_test_dataloaders, pretrained_net, origin_testsample_dir,
                                               'origin_test_fused')

                performance_metrics['D_lambda'].append(metrics['D_lambda'])
                performance_metrics['D_s'].append(metrics['D_s'])
                performance_metrics['QNR'].append(metrics['QNR'])
                performance_metrics['time_FR_test'].append(metrics['time_FR_test'])

                metrics = net_eval_new3(origin_test_dataloaders, pretrained_net, testsample_dir, 'test_fused')

                performance_metrics['ERGAS'].append(metrics['ERGAS'])
                performance_metrics['Q2n'].append(metrics['Q2n'])
                performance_metrics['SAM'].append(metrics['SAM'])
                performance_metrics['time_RR_test'].append(metrics['time_RR_test'])

                print('End!')
                print('\n')

            # 计算所有样本的平均值和方差
            print("\n=== Overall Performance ===")
            for metric in ['D_lambda', 'D_s', 'QNR', 'time_FR_train', 'time_FR_test', 'Q2n', 'ERGAS', 'SAM',
                           'time_RR_test']:
                mean_val = np.mean(performance_metrics[metric])
                std_val = np.std(performance_metrics[metric])
                print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    # 额外输出到控制台
    print("所有信息已保存到", log_file_path)