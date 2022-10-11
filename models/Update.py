#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):

    def __init__(self, args, dataset=None, idxs=None):
        '''
            dataset：数据集
            idxs：属于本客户端的数据在数据集中的编号
        '''
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()  # 交叉熵损失
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)   # batch_size 每次训练样本个数
        # 由以上代码可知，ldr_train为：客户端数据以batch_size分组的结果

    def train(self, net):
        # 在训练模型时会在前面加上
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):  # 本地每个轮次
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()     # 将模型的参数梯度初始化为0
                log_probs = net(images)     # 前向传播计算预测值（Server共享的模型）
                loss = self.loss_func(log_probs, labels)    # 计算当前损失
                loss.backward()             # 反向传播计算梯度
                optimizer.step()            # 更新所有参数
                if self.args.verbose and batch_idx % 10 == 0:       # 若开启了详细打印，则每十组样本打印一次
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

