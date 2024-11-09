# import dgl
import argparse
import os
import time
from datetime import datetime

import dgl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
from thop import profile, clever_format
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import NetWork
import dataset
import utils
from utils import set_seed, analy, test_all

# device_ids = [0, 1]
print('\n')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parse = argparse.ArgumentParser()
parse.add_argument('--dataset', default='muufl', choices=['2012houston', 'trento', 'houston2018', 'muufl'],
                   help='select dataset')
parse.add_argument('--seed', default=28, help='random seed')
parse.add_argument('--train_num', default=0.003, help='the number of train set')
parse.add_argument('--batchsize', default=32, help='batch-size')
parse.add_argument('--test_batchsize', default=5000, help='test-batch-size')
parse.add_argument('--patchsize', default=13, help='the size of patch')
parse.add_argument('--lr', default=0.0001, help='learning rate')
parse.add_argument('--channels', default=32, help='the first layer channel numbers')

parse.add_argument('--train_epoch', default=300, help='epochs')
parse.add_argument('--select_epoch', default=100, help='when select')
parse.add_argument('--circle_epoch', default=50, help='the iteration')

args = parse.parse_args()
times = 1
# ###################### 超参预设 ######################

module_para = './module/{}/train_num_{} best_acc.pkl'.format(args.dataset, args.train_num)
if not os.path.exists('./module/{}/'.format(args.dataset)):
    os.makedirs('./module/{}/'.format(args.dataset))

# ###################### 加载数据集 ######################
samples_type = ['ratio', 'same_num'][0]  # 训练集按照 0-按比例取训练集 1-按每类个数取训练集
# 选择数据集
datasets = args.dataset

# 加载数据
[data_HSI, data_LiDAR, gt, class_count, dataset_name] = dataset.get_dataset(datasets)
th = 1 / class_count
# 源域和目标域数据信息
height, width, bands = data_HSI.shape

# 数据标准化
[data_HSI, data_LiDAR] = dataset.data_standard(data_HSI, data_LiDAR)
# 给LiDAR降一个维度
# data_LiDAR = data_LiDAR[:, :, 0]

# 打印每类样本个数
print('#####源域样本个数#####')
dataset.print_data(gt, class_count)


# ###################### 参数初始化 ######################

def train(TestPatch_HSI, TestPatch_LiDAR, label_index, unlabel_index, train_label, test_label):
    set_seed(args.seed)
    best_acc = 0

    # 构建数据集
    test_data = TensorDataset(TestPatch_HSI[unlabel_index], TestPatch_LiDAR[unlabel_index], test_label[unlabel_index])
    test_data = DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False)
    all_data = TensorDataset(TestPatch_HSI, TestPatch_LiDAR, test_label)
    all_data = DataLoader(all_data, batch_size=args.test_batchsize, shuffle=False, drop_last=False)

    # 构建SAGE网络
    net_SASS = NetWork.SASS(bands, args.channels, class_count, args.patchsize, batch=True).cuda(device)

    # ###################### 训练 ######################
    optimizer = torch.optim.Adam(net_SASS.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(args.train_epoch + 1):

        # 参数初始化
        net_SASS.train()
        acc_num = 0
        loss_class = []
        loss_mmd = []
        loss_diff = []
        loss_recon = []
        loss = []

        if epoch % 50 == 0 or (epoch - args.select_epoch) % args.circle_epoch == 1:
            train_data = utils.dataset(TestPatch_HSI, TestPatch_LiDAR, train_label)
            train_data = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)

        for i, (HSI_lbl, LiDAR_lbl, label_lbl, HSI_ulb, LiDAR_ulb) in enumerate(train_data):
            result_lbl, data_lbl, dict_lbl = net_SASS(HSI_lbl, LiDAR_lbl, 'label', 'train')
            result_ubl, data_ubl, dict_ubl = net_SASS(HSI_ulb, LiDAR_ulb, 'unlabel', 'train')

            # 计算分类损失
            loss_cls = loss_fun(result_lbl, label_lbl - 1)
            loss_class.append(loss_cls.item())

            result = torch.argmax(result_lbl, dim=1) + 1
            acc_num += torch.where(label_lbl == result)[-1].shape[0]

            # 对齐域不变特征
            loss_mmd_HSI = utils.mmd_loss(dict_lbl['hsi'], dict_ubl['hsi'])
            loss_mmd_LiDAR = utils.mmd_loss(dict_lbl['lidar'], dict_ubl['lidar'])
            loss_mmd_global = utils.mmd_loss(dict_lbl['global'], dict_ubl['global'])
            loss_mmd_fusion = utils.mmd_loss(data_lbl, data_ubl)
            loss_mmd_temp = loss_mmd_fusion + 0.1 * (loss_mmd_HSI + loss_mmd_LiDAR + loss_mmd_global)
            loss_mmd.append(loss_mmd_temp.item())

            loss_diff_temp = dict_lbl['diff'] + dict_ubl['diff']
            loss_recon_temp = dict_lbl['recon'] + dict_ubl['recon']
            loss_diff.append(loss_diff_temp.item())
            loss_recon.append(loss_recon_temp.item())

            # 参数更新
            loss_temp = loss_cls + 0.01 * loss_mmd_temp + 0.01 * loss_diff_temp + 0.01 * loss_recon_temp
            loss.append(loss_temp.item())

            optimizer.zero_grad()
            loss_temp.backward()
            optimizer.step()

        acc_train = acc_num / label_index.shape[0]

        # 输出训练结果
        out = PrettyTable()
        print('epoch:{:0>3d}'.format(epoch))
        out.add_column("loss", ['value'])
        out.add_column('acc_train', ['{:.4f}'.format(acc_train)])
        out.add_column('train loss', ['{:.4f}'.format(np.mean(loss) if len(loss) > 0 else 0)])
        out.add_column('class loss', ['{:.4f}'.format(np.mean(loss_class)) if len(loss_class) > 0 else 0])
        out.add_column('domain loss', ['{:.4f}'.format(np.mean(loss_mmd)) if len(loss_mmd) > 0 else 0])
        out.add_column('diff loss', ['{:.4f}'.format(np.mean(loss_diff)) if len(loss_diff) > 0 else 0])
        out.add_column('recon loss', ['{:.4f}'.format(np.mean(loss_recon)) if len(loss_recon) > 0 else 0])
        print(out)

        # 利用验证集保存最优网络结果
        if epoch % 1 == 0:
            # a = time.time()
            net_SASS.eval()
            result, _, test_label_v1 = test_all(net_SASS, test_data)
            loss_val = loss_fun(result, test_label_v1.long() - 1)
            print('| val loss:%.4f' % loss_val)
            result = torch.argmax(result, dim=1)
            accuracy, each_aa, aa, kappa = utils.analy(test_label_v1, result)
            print('| test accuracy: %.4f' % accuracy, '| aa:', aa)
            if best_acc < accuracy:
                best_acc = accuracy
                # best_oa = aa
                torch.save(net_SASS.state_dict(), module_para)
            print('best_acc:{}'.format(best_acc))
            # oa, aa, kappa, each_aa = test(net_SASS, TestPatch_HSI, TestPatch_LiDAR, test_label)
            # print('测试时间为{}s'.format(time.time() - a))

        if epoch != args.train_epoch and (
                epoch - args.select_epoch) % args.circle_epoch == 0 and epoch >= args.select_epoch:

            net_SASS.load_state_dict(torch.load(module_para))
            net_SASS.eval()
            result, data, test_label = test_all(net_SASS, all_data)
            # 构建图
            if test_label.shape[0] > 30000:
                edge = utils.gen_A_coo(data, k=20)
            else:
                edge = utils.get_A_k(data, k=20)

            edge = torch.from_numpy(edge)
            edge = edge.long().to(device)
            graph = dgl.graph((edge[0, :], edge[1, :])).to(device)
            graph.ndata['feat'] = data
            edge_feat_dim = 1  # dim same as node feature dim
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim).to(device)

            net_dis = NetWork.dis(args.channels).to(device)
            optimizer_d = torch.optim.Adam(net_dis.parameters(), lr=0.0005)

            for i in range(200):
                # a = time.time()
                bin = net_dis(graph, data)
                # loss_focal = loss_fun_ac(bin, lbl)
                loss_BCE = utils.active_loss(bin, label_index, unlabel_index)
                if i % 10 == 0:
                    print('the loss of {:0>3d} epoch: {:.4f}'.format(i, loss_BCE))

                optimizer_d.zero_grad()
                loss_BCE.backward()
                optimizer_d.step()

            net_dis.eval()
            bin = net_dis(graph, data)
            bin = bin.view(-1)
            bin_unlabel = bin[unlabel_index]

            train_label = utils.select(bin_unlabel, result, train_label)
            label_index = torch.where(train_label != 0)[-1]
            unlabel_index = torch.where(train_label == 0)[-1]

    net_SASS.load_state_dict(torch.load(module_para))
    net_SASS.eval()
    torch.save(net_SASS.state_dict(),
               './module/{}/train_num_{}_{}_{}'.format(args.dataset, args.train_num, best_acc,
                                                       formatted_datetime))
    result, data, test_true = test_all(net_SASS, test_data)
    result = torch.argmax(result, dim=1)
    oa, each_aa, aa, kappa = analy(test_true, result)
    print('OA: %.2f' % oa, 'AA: %.2f' % aa, 'kappa: %.2f' % kappa)
    print(each_aa)

    test(net_SASS, TestPatch_HSI, TestPatch_LiDAR, test_label)


def test(net_SASS, TestPatch_HSI, TestPatch_LiDAR, test_label):
    net_SASS.load_state_dict(torch.load(module_para))
    net_SASS.eval()
    torch.cuda.synchronize()

    test_data = TensorDataset(TestPatch_HSI, TestPatch_LiDAR, test_label)
    test_data = DataLoader(test_data, batch_size=args.test_batchsize, shuffle=False, drop_last=False)
    result, data, test_label = test_all(net_SASS, test_data)
    result = torch.argmax(result, dim=1)
    oa, each_aa, aa, kappa = analy(test_label, result)
    print('OA: %.4f' % oa, 'AA: %.4f' % aa, 'kappa: %.4f' % kappa)
    print(each_aa)

    Testlabel = np.zeros([height, width])
    Testlabel = np.reshape(Testlabel, [height * width])
    Testlabel[test_label_index] = result.cpu().detach().numpy() + 1
    Testlabel = np.reshape(Testlabel, [height, width])
    sio.savemat('./result/{}_{}.mat'.format(args.dataset, oa), {'data': Testlabel})


current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
set_seed(args.seed)
# 对源域样本进行划分，得到训练、测试、验证集
[train_label, test_label, unlabel] = dataset.data_partition(class_count, gt, args.train_num)
TestPatch_HSI, TestPatch_LiDAR = dataset.gen_cnn_data(data_HSI, data_LiDAR, args.patchsize, train_label, unlabel,
                                                      device)
# 获得新的索引
train_label = np.reshape(train_label, (height * width))
label_index = np.where(train_label != 0)[-1]
label = train_label[label_index]
unlabel = np.reshape(unlabel, (height * width))
unlabel_index = np.where(unlabel != 0)[-1]
unlabel = unlabel[unlabel_index]
test_label = np.reshape(test_label, (height * width))
test_label_index = np.where(test_label != 0)[-1]
test_label = test_label[test_label_index]

all_index = np.concatenate([label_index, unlabel_index], axis=0)
len1 = label_index.shape[0]
index_index = np.argsort(all_index)

all_index = np.sort(all_index)
label_index = np.where(index_index < len1)[-1]
unlabel_index = np.where(index_index >= len1)[-1]
train_label = np.zeros(all_index.shape[0]).astype(np.int64)
train_label[label_index] = label

# 送进gpu
TestPatch_HSI = TestPatch_HSI[index_index].to(device)
TestPatch_LiDAR = TestPatch_LiDAR[index_index].to(device)
label_index = torch.from_numpy(label_index).to(device)
unlabel_index = torch.from_numpy(unlabel_index).to(device)
train_label = torch.from_numpy(train_label).long().to(device)
test_label = torch.from_numpy(test_label).long().to(device)

# 训练集样本个数
for i in range(1, train_label.max().item() + 1):
    print('第{}类训练样本的个数为{}：'.format(i, torch.where(train_label == i)[0].shape[0]))
# 进行训练
train(TestPatch_HSI, TestPatch_LiDAR, label_index, unlabel_index, train_label, test_label)
