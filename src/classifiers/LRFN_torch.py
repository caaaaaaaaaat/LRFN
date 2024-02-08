import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time
from src.utils.utils import create_directory
from src.utils.utils import get_test_loss_acc
from src.utils.utils import save_models
from src.utils.utils import log_history
from src.utils.utils import calculate_metrics
from src.utils.utils import save_logs
from src.utils.utils import model_predict
from src.utils.utils import plot_epochs_metric
import os
import torchvision
import torch.nn.functional as F


class LRFN(nn.Module):
    # return IFCN_torch.IFCN(1, 16, 32, 96, 16, 32, 64, 192, 32, nb_classes), IFCN_torch

    def __init__(self, in_channels, out1_channels_1, out1_channels_2, out1_channels_3,
                 out1_channels_4, out2_channels_1, out2_channels_2, out2_channels_3,
                 out2_channels_4, num_class):
        super(LRFN, self).__init__()  # 调用父类的构造函数

        in_channels_module1 = in_channels
        self.module1 = Inception_module(in_channels, out1_channels_1, out1_channels_2,
                                        out1_channels_3, out1_channels_4, 1, 3)

        in_channels_module2 = out1_channels_1 + out1_channels_3 + out1_channels_4
        self.module2 = Inception_module(in_channels_module2, out2_channels_1, out2_channels_2,
                                        out2_channels_3, out2_channels_4, 1, 3)

        in_channels_module3 = out2_channels_1 + out2_channels_3 + out2_channels_4
        self.module3 = Inception_module(in_channels_module3, out1_channels_1, out1_channels_2,
                                        out1_channels_3, out1_channels_4, 1, 3)

        self.global_ave_pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_channels_module2, num_class)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.global_ave_pooling(x).squeeze()

        output = self.linear(x)

        return output, x


class Inception_module(nn.Module):

    def __init__(self, in_channels, out_channels_1, out_channels_2,
                 out_channels_3, out_channels_4, block_num, dilation, **kwargs):
        super(Inception_module, self).__init__()

        self.scconv = SCN_block(in_channels, out_channels_1)

        self.conv_3_3_a = BasicConv2d(in_channels, out_channels_3, 1)
        self.conv_3_3_b = _make_layer(BasicBlock, out_channels_3, out_channels_3, block_num=block_num, stride=1,
                                      dilation=dilation)

        self.conv_pool = ASPP(in_channels, out_channels_4, [12, 24, 36])

        self.conv_3pool_1_3 = nn.MaxPool2d(3, 1, 3 // 2)
        self.conv_3pool_1_1 = BasicConv2d(in_channels, out_channels_4, 1)

    def forward(self, x):
        branch_1 = self.scconv(x)

        branch_2 = self.conv_3_3_a(x)
        branch_2 = self.conv_3_3_b(branch_2)

        branch_3 = self.conv_pool(x)

        module_outputs = [branch_1, branch_2, branch_3]

        return torch.cat(module_outputs, 1)


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


def _make_layer(block, inplanes, planes, block_num, stride=1, dilation=1,
                new_level=False, residual=True):

    '''
    :param block: block模板
    :param plane: 每个模块中间运算的维度，一般等于输出维度/4
    :param block_num/blocks: 重复次数
    :param stride: 步长
    :return:
    '''

    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = list()
    layers.append(block(
        inplanes, planes, stride, downsample,
        dilation=(1, 1) if dilation == 1 else (
            dilation // 2 if new_level else dilation, dilation),
        residual=residual))

    inplanes = planes
    for i in range(1, block_num):
        layers.append(block(inplanes, planes, residual=residual,
                            dilation=(dilation, dilation)))

    return nn.Sequential(*layers)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                      padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            norm_layer(planes),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=None):
        super(SCBottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality  # 按输出通道分成两部分
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False),
            norm_layer(group_width),
        )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(
            group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = torch.cat([out_a, out_b], dim=1)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)

        return out


class SCN_block(nn.Module):  # 未使用深度可分离卷积

    def __init__(self, in_planes, out_planes, ):
        super(SCN_block, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.SCBottleneck = SCBottleneck(in_planes, out_planes, norm_layer=norm_layer)

    def forward(self, x):
        x = self.SCBottleneck(x)

        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )  # nn.Dropout(0.3)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def train_op(classifier_obj, EPOCH, batch_size, LR, train_x, train_y,
             test_x, test_y, output_directory_models,
             model_save_interval, test_split,
             save_best_train_model=True,
             save_best_test_model=True):
    # prepare training_data
    BATCH_SIZE = int(min(train_x.shape[0]/8, batch_size))
    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False
    torch_dataset = Data.TensorDataset(
        torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset=torch_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   drop_last=drop_last_flag
                                   )

    # init lr&train&test loss&acc log
    lr_results = []
    loss_train_results = []
    accuracy_train_results = []
    loss_test_results = []
    accuracy_test_results = []

    # prepare optimizer&scheduler&loss_function
    optimizer = torch.optim.Adam(classifier_obj.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                           patience=50,
                                                           min_lr=0.000002, verbose=True) #min_lr=0.000002
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    # save init model
    output_directory_init = output_directory_models+'init_model.pkl'
    # save only the init parameters
    torch.save(classifier_obj.state_dict(), output_directory_init)

    training_duration_logs = []
    start_time = time.time()
    for epoch in range(EPOCH):

        # loss_sum_train = torch.tensor(0)
        # true_sum_train = torch.tensor(0)

        for step, (x, y) in enumerate(train_loader):

            batch_x = x.cpu()
            batch_y = y.cpu()
            #batch_x = x.cuda()
            #batch_y = y.cuda()
            output_bc = classifier_obj(batch_x)[0]

            # # cal the num of correct prediction per batch
            # pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()
            # true_num_bc = torch.sum(pred_bc == batch_y).data

            # cal the sum of pre loss per batch
            loss = loss_function(output_bc, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test per epoch
        classifier_obj.eval()
        loss_train, accuracy_train = get_test_loss_acc(
            classifier_obj, loss_function, train_x, train_y, 15)
        loss_test, accuracy_test = get_test_loss_acc(
            classifier_obj, loss_function, test_x, test_y, test_split)
        classifier_obj.train()

        # update lr
        scheduler.step(loss_train)
        lr = optimizer.param_groups[0]['lr']

        ###################################### dropout#####################################
        # loss_train, accuracy_train = get_loss_acc(classifier_obj.eval(), loss_function, train_x, train_y, test_split)

        # loss_test, accuracy_test = get_loss_acc(classifier_obj.eval(), loss_function, test_x, test_y, test_split)

        # classifier_obj.train()
        ##################################################################################

        # log lr&train&test loss&acc per epoch
        lr_results.append(lr)
        loss_train_results.append(loss_train)
        accuracy_train_results.append(accuracy_train)
        loss_test_results.append(loss_test)
        accuracy_test_results.append(accuracy_test)

        # print training process
        if (epoch+1) % 1 == 0:
            print('Epoch:', (epoch+1), '|lr:', lr,
                  '| train_loss:', loss_train,
                  '| train_acc:', accuracy_train,
                  '| test_loss:', loss_test,
                  '| test_acc:', accuracy_test)

        training_duration_logs = save_models(classifier_obj, output_directory_models,
                                             loss_train, loss_train_results,
                                             accuracy_test, accuracy_test_results,
                                             model_save_interval, epoch, EPOCH,
                                             start_time, training_duration_logs,
                                             save_best_train_model, save_best_test_model)

    # save last_model
    output_directory_last = output_directory_models+'last_model.pkl'
    # save only the init parameters
    torch.save(classifier_obj.state_dict(), output_directory_last)

    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results,
                          loss_test_results, accuracy_test_results)

    return (history, training_duration_logs)
