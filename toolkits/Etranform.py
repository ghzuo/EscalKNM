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
@Last Modified Time: 2023-05-20 11:10:11
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
        exdata = np.append(tdata, tdata[-1::-1])  # even extension
        return np.fft.rfft(exdata)

    def exiFFT(self, fdata, kappa):
        noutput = len(fdata) - 1
        exdata = np.fft.irfft(fdata[:kappa], 2*noutput)
        return exdata[:noutput]

    def list_exiFFT(self, kappa):
        return [self.exiFFT(f, kappa) for f in self.F]

    def multi_exiFFT(self, kappa):
        return np.concatenate(self.list_exiFFT(kappa))
