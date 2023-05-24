#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2023
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2023-05-20 11:06:05
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-05-23 16:54:11
'''

import numpy as np


class LinearRegress:
    def __init__(self):
        pass

    def q_multi_cor(self, X, Y):
        t = np.var(Y)
        q = np.linalg.lstsq(X, Y, rcond=None)[1][0]/len(Y)
        return np.sqrt(1 - q/t), t, q

    def score(self, fft, kmax=50, quiet=True):
        # type the kappa
        kmax = min([kmax, len(fft.F[0])])
        qmc = np.array([range(2, kmax), np.zeros(kmax-2)]).T
        for item in qmc:
            item[1], t, q = self.q_multi_cor(
                fft.Xx, fft.multi_exiFFT(int(item[0])))
            if (not quiet):
                print(item[0], item[1], t, q, sep="\t")

        # the result
        result = {'list': qmc}
        imax = np.argmax(qmc[:, 1])
        result['KappaMax'], result['qmcMax'] = qmc[imax]
        return result

    def scale(self, fft, kappa):
        res = np.linalg.lstsq(fft.Xx, fft.multi_exiFFT(int(kappa)))
        result = {'A': res[0][1:]}
        if fft.X is list:
            result['X'] = [x*result['A'] for x in fft.X]
            result['Ee'] = [np.dot(x, result['A']) for x in fft.X]
        else:
            result['X'] = fft.X*result['A']
            result['Ee'] = np.dot(fft.X, result['A'])
        return result
