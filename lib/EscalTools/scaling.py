#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:32:17
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-09-01 11:45:23
'''


import numpy as np


class Efft:
    def __init__(self, X, E):
        self.X = X
        if all([type(X) is list, type(E) is list]):
            Xx = np.concatenate(X)
            self.Xx = np.c_[np.ones(len(Xx)), Xx]
            self.F = [self.exFFT(td) for td in E]
        else:
            self.Xx = np.c_[np.ones(len(X)), X]
            self.F = [self.exFFT(E)]

    def exFFT(self, tdata):
        exdata = np.append(tdata, tdata[len(tdata)-1::-1])  # even extension
        return np.fft.rfft(exdata)

    def exiFFT(self, fdata, kappa):
        noutput = len(fdata) - 1
        exdata = np.fft.irfft(fdata[0:kappa], 2*noutput)
        return exdata[0:noutput]

    def multi_exiFFT(self, fdlist, kappa):
        return np.concatenate([self.exiFFT(fd, kappa) for fd in fdlist])

    def q_multi_cor(self, X, Y):
        d = Y - np.mean(Y)
        t = np.dot(d.T, d)
        q = np.linalg.lstsq(X, Y, rcond=None)[1][0]
        return np.sqrt(1 - q/t)

    def qmc_lf(self, kappa, F, X):
        return self.q_multi_cor(X, self.multi_exiFFT(F, kappa))

    def check(self, kmax=50):
        # type the kappa
        kmax = min([kmax, len(self.F[0])])
        qmc = np.array([range(2, kmax), np.zeros(kmax-2)]).T
        for item in qmc:
            item[1] = self.qmc_lf(int(item[0]), self.F, self.Xx)

        # the result
        result = {}
        result['list'] = qmc
        imax = np.argmax(qmc[:, 1])
        result['KappaMax'], result['qmcMax'] = qmc[imax]
        return result

    def rescale(self, kappa):
        result = {}
        result['Ef'] = [self.exiFFT(f, int(kappa)) for f in self.F]
        res = np.linalg.lstsq(self.Xx, self.multi_exiFFT(self.F, int(kappa)))
        result['A'] = res[0][1:]
        if self.X is list:
            result['X'] = [x*result['A'] for x in self.X]
        else:
            result['X'] = self.X*result['A']
        return result
