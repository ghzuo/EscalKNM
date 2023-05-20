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
@Last Modified Time: 2023-05-20 12:45:17
'''

import numpy as np


def q_multi_cor(X, Y):
    d = Y - np.mean(Y)
    t = np.dot(d.T, d)
    q = np.linalg.lstsq(X, Y, rcond=None)[1][0]
    return np.sqrt(1 - q/t)


def score(fft, kmax=50):
    # type the kappa
    kmax = min([kmax, len(fft.F[0])])
    qmc = np.array([range(2, kmax), np.zeros(kmax-2)]).T
    for item in qmc:
        item[1] = q_multi_cor(fft.Xx, fft.multi_exiFFT(int(item[0])))

    # the result
    result = {'list': qmc}
    imax = np.argmax(qmc[:, 1])
    result['KappaMax'], result['qmcMax'] = qmc[imax]
    return result


def scale(fft, kappa):
    result = {'Ef': fft.list_exiFFT(int(kappa))}
    res = np.linalg.lstsq(fft.Xx, fft.multi_exiFFT(int(kappa)))
    result['A'] = res[0][1:]
    if fft.X is list:
        result['X'] = [x*result['A'] for x in fft.X]
    else:
        result['X'] = fft.X*result['A']
    return result
