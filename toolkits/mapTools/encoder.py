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
@Last Modified Time: 2023-06-06 19:56:19
'''

import numpy as np
import torch


def normlize(X):
    Xm = X.mean(dim=0)
    Xs = X.std(dim=0)
    Xx = (X - Xm)/Xs
    return Xx, Xm, Xs


def deNormlize(Xx, Xm, Xs):
    return Xx*Xs + Xm


class EncoderNet:
    def __init__(self, net, fft, info=1,
                 loss=torch.nn.MSELoss(),
                 batch_size=10, shuffle=True, njobs=2,
                 optim=torch.optim.Adam, lr=0.001):
        self.net = net
        self.fft = fft
        self.info = info
        self.loss = loss
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.njobs = njobs
        self.optim = optim(self.net.parameters(), lr)
        self.X = torch.tensor(self.fft.Xx, dtype=torch.float)

    def infoLevel(self, info):
        self.info = info

    def setOptim(self, optim=torch.optim.Adam, lr=0.001):
        self.optim = optim(self.net.parameters(), lr)

    def setY(self, kappa):
        self.y, self.ym, self.ys = normlize(
            torch.tensor(self.fft.multi_exiFFT(kappa), dtype=torch.float))

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

    def train(self, X, y, n_epochs=3):
        # set dataset
        data = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X, y),
            batch_size=self.batch_size,  # mini batch size
            shuffle=self.shuffle,
            num_workers=self.njobs,  # multi-thread read
        )

        # do training
        for epoch in range(1, n_epochs + 1):
            for X, y in data:
                output = self.net(X)
                ls = self.loss(output, y.view(-1, 1))
                self.optim.zero_grad()
                ls.backward()
                self.optim.step()
            if (self.info > 1):
                print('epoch %d, loss: %f' % (epoch, ls.item()))

    def validate(self, X_val, y_val, prompt="Validate:"):
        output = self.net(X_val)
        ls = self.loss(output, y_val.view(-1, 1))
        q = ls.item()
        r = np.sqrt(1 - q) if q < 1 else 0
        if self.info > 0:
            print(prompt, r, q, sep=" ")
        return r

    def preTrain(self, kfold=0, n_epochs=5, kappa=2):
        self.setY(kappa)
        if (kfold == 0):
            self.train(self.X, self.y, n_epochs=n_epochs)
            self.validate(
                self.X, self.y, f"Validate for PreTrain (kappa={kappa}):")
        else:
            for i in range(0, kfold):
                X_T, y_T, X_V, y_V = self.get_kfold_data(
                    kfold, i, self.X, self.y)
                self.train(X_T, y_T, n_epochs=n_epochs)
                self.validate(X_V, y_V, f"Summary for fold ({i+1}/{kfold}):")

    def score(self, kmax=50, kmin=2, nk=100, n_epochs=2):
        # type the kappa
        kmax = len(self.fft.F[0]) if kmax < kmin else min(
            [kmax, len(self.fft.F[0])])
        kmin = max(kmin, 2)
        ksep = int((kmax - kmin)/nk)+1
        klist = np.arange(kmin, kmax, ksep)
        qmc = np.array([klist, np.zeros(len(klist))]).T
        for item in qmc:
            self.setY(item[0])
            self.train(self.X, self.y, n_epochs=n_epochs)
            item[1] = self.validate(self.X, self.y,
                                    f"Validate for Score (kappa={item[0]}):")

        # the result
        result = {'list': qmc}
        imax = np.argmax(qmc[:, 1])
        result['KappaMax'], result['qmcMax'] = qmc[imax]
        return result

    def scale(self, kappa, n_epochs=2):
        # retrain the model
        self.setY(kappa)
        self.train(self.X, self.y, n_epochs=n_epochs)
        self.validate(self.X, self.y, f"Validate for Scaling (kappa={kappa}):")

        # add hook on features layer
        features = []
        handle = self.net.output.register_forward_hook(
            hook=lambda module, fin, fout: features.append(fin))
        # get the effect energy and features
        Ee = deNormlize(self.net(self.X), self.ym, self.ys).detach().numpy()
        result = {
            "Ee": np.split(Ee, len(self.fft.X)) if self.fft.X is list else Ee,
            'X': (features[0] * self.net.output.weight).detach().numpy()
            if self.net.__name__ == 'LR'
            else features[0][0].detach().numpy(),
            'A': self.net.output.weight.detach().numpy()[0]
        }
        handle.remove()
        return result


# the square layer
class Square(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(in_features*2, out_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features,))

    def forward(self, x):
        xx = torch.square(x)
        return torch.matmul(torch.concat((xx, x), 1),
                            self.weight.detach()) + self.bias.detach()


# the linear regression
class LR(torch.nn.Module):
    def __init__(self, nInput, **kwargs):
        super(LR, self).__init__(**kwargs)
        self.__name__ = 'LR'
        self.output = torch.nn.Linear(nInput, 1)

    def forward(self, x):
        return self.output(x)


# the multi-layer Perceptron
class MLP(torch.nn.Module):
    def __init__(self, nInput, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.__name__ = 'MLP'
        nHidden = nInput
        actfunc = torch.nn.ELU
        self.input = torch.nn.Linear(nInput, nHidden)
        self.act1 = actfunc()
        self.hidden = torch.nn.Linear(nHidden, 2)
        self.act2 = actfunc()
        self.output = Square(2, 1)
        # self.output = torch.nn.Linear(2, 1)
        self.act3 = actfunc()

    def forward(self, x):
        out1 = self.act1(self.input(x))
        out2 = self.act2(self.hidden(out1 + x))
        return self.output(out2)
