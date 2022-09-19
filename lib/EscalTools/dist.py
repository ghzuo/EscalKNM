#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:25:17
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-08-22 22:06:52
'''


import numpy as np


# for distance matrix
def expm(data, lam=0.01):
    mat = []
    for i in data:
        for j in data:
            mat.append(np.exp(- lam*np.linalg.norm(i - j)))
    mat = np.matrix(mat).reshape(len(data), len(data))
    return mat


def manhattan(data):
    mat = []
    for i in data:
        for j in data:
            mat.append(np.sum([abs(vi-vj) for vi, vj in zip(i, j)]))
    mat = np.matrix(mat).reshape(len(data), len(data))
    return mat


def euclidean(data):
    mat = []
    for i in data:
        mat.append(np.linalg.norm(data - i, axis=1))
    return np.matrix(mat)


def norm_rows(Mx):
    norm = np.linalg.norm(Mx, axis=1)
    norm = norm.reshape([len(norm), 1])
    Mx = Mx/norm
    return Mx


def cosin(data, plus=False):
    if plus:
        data = np.concatenate([data, np.ones(data.shape)], axis=1)
    data = norm_rows(data)
    mat = np.dot(data, data.T)
    mat = (mat + 1)/2
    return mat
