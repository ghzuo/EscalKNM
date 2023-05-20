#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2023
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2023-05-20 13:55:16
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-05-20 13:59:24
'''

import torch


def fit(net, data_iter, loss=torch.nn.MSELoss(), opt=None,
        n_epochs=3, lr=0.03):
    if opt is None:
        opt = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            opt.zero_grad()
            l.backward()
            opt.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))


def pack_data(features, labels, batch_size=10, shuffle=True, njobs=2):
    # 将训练数据的特征和标签组合
    dataset = torch.utils.data.TensorDataset(features, labels)

    return torch.utils.data.DataLoader(
        dataset=dataset,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=shuffle,  # 要不要打乱数据 (打乱比较好)
        num_workers=njobs,  # 多线程来读数据
    )
