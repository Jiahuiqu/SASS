import random

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.utils
from sklearn import preprocessing


def print_data(gt, class_count):
    gt_reshape = np.reshape(gt, [-1])
    each = np.zeros(class_count)
    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        each[i] = samplesCount
        print('第' + str(i + 1) + '类的个数为' + str(samplesCount))
    print(each)


def get_dataset(dataset):
    data_HSI = []
    data_LiDAR = []
    gt = []
    class_count = 0
    dataset_name = ''

    if dataset == '2012houston':
        data_HSI_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/2012houston/HSI_data.mat')
        data_HSI = data_HSI_mat['HSI_data']
        data_LiDAR_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/2012houston/LiDAR_data.mat')
        data_LiDAR = data_LiDAR_mat['LiDAR_data']

        gt_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/2012houston/All_Label.mat')
        gt = gt_mat['All_Label']

        # 参数预设
        class_count = 15  # 样本类别数
        dataset_name = "Huston2013"  # 数据集名称
        pass

    if dataset == 'trento':
        data_HSI_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/trento/HSI_data.mat')
        data_HSI = data_HSI_mat['HSI_data']
        data_LiDAR_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/trento/LiDAR_data.mat')
        data_LiDAR = data_LiDAR_mat['LiDAR_data']

        gt_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/trento/All_Label.mat')
        gt = gt_mat['All_Label']

        # 参数预设
        class_count = 6  # 样本类别数
        dataset_name = "Trento"  # 数据集名称
        pass
    if dataset == 'muufl':
        data_HSI_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/muufl/HSI.mat')
        data_HSI = data_HSI_mat['data']

        data_LiDAR_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/muufl/LiDAR_DEM.mat')
        data_LiDAR = data_LiDAR_mat['LiDAR_DEM']

        gt_mat = sio.loadmat('/media/xd132/USER/ZLJ/dataset/muufl/HSI.mat')
        gt = gt_mat['gt']

        # 参数预设
        class_count = 11  # 样本类别数
        dataset_name = "MUUFL"  # 数据集名称
        pass

    return [np.array(data_HSI), np.array(data_LiDAR), gt, class_count,dataset_name]


def data_standard(data_HSI, data_LiDAR):
    height, width, bands = data_HSI.shape  # 原始高光谱数据的三个维度

    data_HSI = np.reshape(data_HSI, [height * width, bands])  # 将数据转为HW * B
    minMax = preprocessing.StandardScaler()
    data_HSI = minMax.fit_transform(data_HSI)  # 这两行用来归一化数据，归一化时需要进行数据转换
    data_HSI = np.reshape(data_HSI, [height, width, bands])  # 将数据转回去 H * W * B

    data_LiDAR = np.reshape(data_LiDAR, [height * width, 1])  # 将数据转为HW * B
    minMax = preprocessing.StandardScaler()
    data_LiDAR = minMax.fit_transform(data_LiDAR)  # 这两行用来归一化数据，归一化时需要进行数据转换
    data_LiDAR = np.reshape(data_LiDAR, [height, width, 1])  # 将数据转回去 H * W * B
    return [data_HSI, data_LiDAR]


def data_partition(class_count, gt, train_ratio):
    train_rand_idx = np.array([]).astype('int64')
    height, width = gt.shape
    gt_reshape = np.reshape(gt, [-1])

    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        num = np.ceil(samplesCount * train_ratio).astype('int32') if train_ratio< 1 else train_ratio
        real_train_samples_per_class = num if num < samplesCount else samplesCount
        rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
        rand_idx = random.sample(rand_list, real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
        rand_real_idx_per_class = idx[rand_idx]
        train_rand_idx = np.concatenate((train_rand_idx, rand_real_idx_per_class), axis=0)


    train_data_index = train_rand_idx
    train_data_index = set(train_data_index)
    all_data_index = [i for i in range(len(gt_reshape))]
    all_data_index = set(all_data_index)

    # 背景像元的标签
    background_idx = np.where(gt_reshape == 0)[-1]
    background_idx = set(background_idx)
    test_data_index = all_data_index - background_idx
    unlabel_data_index = test_data_index - train_data_index

    # 将训练集 验证集 无标签 整理
    test_data_index = np.array(list(test_data_index))
    train_data_index = np.array(list(train_data_index))
    unlabel_data_index = np.array(list(unlabel_data_index))

    # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)
    train_samples_gt[train_data_index] = gt_reshape[train_data_index]
    train_label = np.reshape(train_samples_gt, [height, width])

    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    test_samples_gt[test_data_index] = gt_reshape[test_data_index]
    test_label = np.reshape(test_samples_gt, [height, width])  # 测试样本图

    # 获取无标签样本的标签图
    unlabel_samples_gt = np.zeros(gt_reshape.shape)
    unlabel_samples_gt[unlabel_data_index] = gt_reshape[unlabel_data_index]
    unlabel = np.reshape(unlabel_samples_gt, [height, width])

    return [train_label, test_label, unlabel]


def getpatch(label, data, patchsize, device):
    bands = data.shape[2]
    [ind1, ind2] = np.where(label != 0)  # ind1和ind2是符合where中条件的点的横纵坐标
    Num = len(ind1)  # 训练样本的个数
    pad_width = patchsize // 2  # 除法取整
    Patch_data = np.empty((Num, bands, patchsize, patchsize), dtype='float32')
    ind3 = ind1 + pad_width  # 因为在图像四周打了padding，所以横纵坐标要加上打padding的像素个数
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        # 取第i个训练patch，取一个立方体
        patch = data[(ind3[i] - pad_width):(ind3[i] + pad_width + 1),
                (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
        # conv的输入是NCHW，现在的patch是[H,W,C]，要把patch转成[C,H,W]形状
        patch = np.reshape(patch, (patchsize * patchsize, bands))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (bands, patchsize, patchsize))
        Patch_data[i] = patch
    Patch_data = torch.from_numpy(Patch_data).to(device)
    return Patch_data


def gen_cnn_data(data_HSI, data_LiDAR, patchsize, train_label, test_label, device):
    # ##### 给HSI和LiDAR打padding #####
    pad_width = patchsize // 2
    data_HSI_pad = np.pad(data_HSI, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 'symmetric')
    data_LiDAR_pad = np.pad(data_LiDAR, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), 'symmetric')

    # #### 构建高光谱的训练集和测试集 #####
    TrainPatch_HSI = getpatch(train_label, data_HSI_pad, patchsize, device)
    UnlabelPatch_HSI = getpatch(test_label, data_HSI_pad, patchsize, device)
    print('Training size and testing size of HSI are:', TrainPatch_HSI.shape, 'and', UnlabelPatch_HSI.shape)

    # #### 构建LiDAR的训练集和测试集 #####
    TrainPatch_LiDAR = getpatch(train_label, data_LiDAR_pad, patchsize, device)
    UnlabelPatch_LIDAR = getpatch(test_label, data_LiDAR_pad, patchsize, device)
    print('Training size and testing size of LiDAR are:', TrainPatch_LiDAR.shape, 'and', UnlabelPatch_LIDAR.shape)

    TestPatch_HSI = torch.cat([TrainPatch_HSI, UnlabelPatch_HSI], dim=0)
    TestPatch_LiDAR = torch.cat([TrainPatch_LiDAR, UnlabelPatch_LIDAR], dim=0)

    return TestPatch_HSI, TestPatch_LiDAR
