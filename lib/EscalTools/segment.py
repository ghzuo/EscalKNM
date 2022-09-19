#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:22:39
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-08-25 11:18:22
'''


import numpy as np
from EscalTools.dist import euclidean


# for sample the cluster point
def first(data, step=100):
    return np.matrix(data[::step])


def medoid(data, step=100, width=200, dist=euclidean):
    mat = np.array([]).reshape(0, data.shape[1])
    ndx = (len(data)-width)//step + 1
    for i in range(ndx):
        idx = i * step
        segment = data[idx:idx+width]
        dmat = dist(segment)
        dsum = np.array(np.sum(dmat, axis=0)).reshape(width)
        mat = np.append(mat, segment[np.where(dsum == np.min(dsum))], axis=0)
    return np.matrix(mat)


def mean(data, step=100, width=200):
    mat = []
    ndx = (len(data)-width)//step + 1
    for i in range(ndx):
        idx = i * step
        mat.append(data[idx:idx+width].mean(axis=0))
    return np.matrix(mat)


def median(data, step=100, width=200):
    mat = []
    ndx = (len(data)-width)//step + 1
    for i in range(ndx):
        idx = i * step
        mat.append(np.median(data[idx:idx+width], axis=0))
    return np.matrix(mat)
