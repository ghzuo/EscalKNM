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
@Last Modified Time: 2023-05-24 20:34:08
'''

import numpy as np
import torch


class EncoderNet:
    def __init__(self, net, info=1, loss=torch.nn.MSELoss(), batch_size=10,
                 shuffle=True, njobs=2):
        self.net = net
        self.info = info
        self.loss = loss
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.njobs = njobs

    def infoLevel(self, info):
        self.info = info

    def conv_data(self, fft, kappa):
        features = torch.tensor(fft.Xx, dtype=torch.float)
        labels = torch.tensor(fft.multi_exiFFT(kappa), dtype=torch.float)
        return features, labels

    def get_kfold_data(self, kfold, i, X, y):
        fold_size = X.shape[0] // kfold
        if i == 0:
            X_valid, y_valid = X[:fold_size], y[:fold_size]
            X_train, y_train = X[fold_size:], y[fold_size:]
        elif i == kfold - 1:
            val_start = i * fold_size
            X_valid, y_valid = X[val_start:], y[val_start:]
            X_train, y_train = X[:val_start], y[:val_start]
        else:
            val_start = i * fold_size
            val_end = val_start + fold_size
            X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
            X_train = torch.cat((X[:val_start], X[val_end:]), dim=0)
            y_train = torch.cat((y[:val_start], y[val_end:]), dim=0)
        return X_train, y_train, X_valid, y_valid

    def train(self, X, y, opt=None, n_epochs=3, lr=0.03):
        # set dataset
        data = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X, y),
            batch_size=self.batch_size,  # mini batch size
            shuffle=self.shuffle,
            num_workers=self.njobs,  # multi-thread read
        )

        # set the opter
        if opt is None:
            opt = torch.optim.SGD(self.net.parameters(), lr=lr)

        # do training
        for epoch in range(1, n_epochs + 1):
            for X, y in data:
                output = self.net(X)
                ls = self.loss(output, y.view(-1, 1))
                opt.zero_grad()
                ls.backward()
                opt.step()
            if (self.info > 1):
                print('epoch %d, loss: %f' % (epoch, ls.item()))

    def validate(self, X_val, y_val, prompt="Validate:"):
        output = self.net(X_val)
        ls = self.loss(output, y_val.view(-1, 1))
        q = ls.item()
        t = torch.var(y_val).item()
        r = np.sqrt(1 - q/t) if q < t else 0
        if self.info > 0:
            print(prompt, r, t, q, sep=" ")
        return r

    def preTrain(self, fft, kfold=0, n_epochs=10, lr=0.05, kappa=2):
        X, y = self.conv_data(fft, kappa)
        if (kfold == 0):
            self.train(X, y, lr=lr, n_epochs=n_epochs)
            self.validate(X, y, f"Validate for PreTrain (kappa={kappa}):")
        else:
            for i in range(0, kfold):
                X_T, y_T, X_V, y_V = self.get_kfold_data(kfold, i, X, y)
                self.train(X_T, y_T, lr=lr, n_epochs=n_epochs)
                self.validate(X_V, y_V, f"Summary for fold ({i+1}/{kfold}):")

    def score(self, fft, kmax=50, n_epochs=3, lr=0.03):
        # type the kappa
        kmax = min([kmax, len(fft.F[0])])
        qmc = np.array([range(2, kmax), np.zeros(kmax-2)]).T
        for item in qmc:
            X, y = self.conv_data(fft, item[0])
            self.train(X, y, lr=lr, n_epochs=n_epochs)
            item[1] = self.validate(X, y,
                                    f"Validate for Score (kappa={item[0]}):")

        # the result
        result = {'list': qmc}
        imax = np.argmax(qmc[:, 1])
        result['KappaMax'], result['qmcMax'] = qmc[imax]
        return result

    def scale(self, fft, kappa, n_epochs=3, lr=0.03):
        # retrain the model
        X, y = self.conv_data(fft, kappa)
        self.train(X, y, lr=lr, n_epochs=n_epochs)
        self.validate(X, y, f"Validate for Scaling (kappa={kappa}):")

        # add hook on features layer
        features_in = []
        handle = self.net.features.register_forward_hook(
            hook=lambda module, fin, fout: features_in.append(fin))
        # get the effect energy and features
        result = {
            'Ee': [self.net(torch.tensor(x, dtype=torch.float))
                   .detach().numpy() for x in fft.X]
            if fft.X is list
            else self.net(torch.tensor(fft.X, dtype=torch.float)
                          ).detach().numpy()}
        result['X'] = (features_in[0][0] *
                       self.net.features.weight).detach().numpy()
        result['A'] = self.net.features.weight.detach().numpy()[0, :]
        handle.remove()
        return result
