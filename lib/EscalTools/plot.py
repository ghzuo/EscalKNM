#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-07-03 16:18:21
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-08-28 20:27:09
'''


# Basic set of Python Data Analysis
from mpl_toolkits.axes_grid import make_axes_locatable
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def implot(mat, ax=None):
    if ax is None:
        ax = plt.subplot()
    im = ax.imshow(mat, interpolation='nearest', cmap=cm.jet)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)


def coefplot(ys, ax=None):
    if ax is None:
        ax = plt.subplot()
    xs = np.arange(1, len(ys)+1)
    ax.plot(xs, ys)
    for x, y in zip(xs, ys):
        ax.plot(x, y, 'o', ms=20, lw=2, alpha=0.7, mfc='yellow')
        ax.text(x, y, '%d' % (int(x)), fontsize=12,
                horizontalalignment='center', verticalalignment='center')


def efplot(E0, Ef, Ey):
    plt.figure(22, figsize=(16, 8))
    plt.subplot(121)
    scale = (np.sqrt((np.var(E0)/np.var(Ey)))//10)*10
    plt.plot((E0-np.mean(E0))/scale, 'c-',
             label="Raw Data (/{:.0f})".format(scale))
    plt.plot(Ey-np.mean(Ey), "r-", label="Effective Energy")
    Efmean = np.mean(Ef)
    plt.plot(Ef[0]-Efmean, "b-", lw=5, label="iFFT Energy")
    xbeg = len(Ef[0])
    for ef in Ef[1:]:
        xend = xbeg + len(ef)
        plt.plot(range(xbeg, xend), ef-Efmean, "b-", lw=5)
        xbeg = xend
    plt.legend()

    plt.subplot(222)
    sns.histplot(Ey, kde=True, color='red')
    plt.subplot(224)
    sns.histplot(E0, kde=True, color='cyan')


def clplot(score, cl, lab):
    # plot data
    plt.figure(22, figsize=(20, 10))
    ax = plt.subplot(121)
    score.plot(marker="o", ax=ax)

    plt.subplot(222)
    plt.plot(cl, "ro-", label=lab)
    plt.legend()

    plt.subplot(224)
    pd.value_counts(cl).plot.bar(rot=0)
