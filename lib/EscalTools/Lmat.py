#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:30:05
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-08-22 17:44:39
'''

import numpy as np


def Lmat(Smat):
    Dmat = np.diag(np.sum(Smat, axis=0).A1)
    Lmat = Dmat - Smat
    return Lmat


def NLmat_rw(Smat):
    Dmat = np.diag(np.sum(Smat, axis=0).A1)
    Lmat = Dmat - Smat
    Nmat = np.linalg.inv(Dmat) * Lmat
    return Nmat


def NLmat_sym(Smat):
    Dmat = np.diag(np.sum(Smat, axis=0).A1)
    Lmat = Dmat - Smat
    iDmat = np.linalg.inv(Dmat)
    iDmatSqrt = np.sqrt(iDmat)
    Nmat = iDmatSqrt * Lmat * iDmatSqrt
    return Nmat
