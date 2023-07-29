#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2023
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2023-07-29 12:54:32
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-07-29 13:35:14
'''

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


def basicSet():
    sns.set_theme(style='white', font_scale=2.2,
                  rc={'font.family': "Serif",
                        'font.weight': "bold",
                        'lines.markersize': 15,
                        'xtick.top': True,
                        'xtick.bottom': True,
                        'ytick.left': True,
                        'ytick.direction': "in",
                        'xtick.direction': "in",
                        'axes.xmargin': 0.005})


def trajPlot(ax, rms, E0, Ey):
    # plot rms trajectory
    ax[0].plot(rms['time'], rms['rms'])
    ax[0].set_yticklabels(["", 0.0, 5.0, 10.0])
    ax[0].set_ylabel("RMSD(Å)", fontsize=25, weight="bold")
    ax[0].set_xticklabels([])
    ax[0].margins(y=0.1)
    ax[0].set_ylim(-0.05, 1.01)

    # plot the energy trajectory
    scale = (np.sqrt((np.var(E0)/np.var(Ey)))//10)*10
    ax[1].plot((E0-np.mean(E0))/scale, 'c-',
               label="Raw Data (/{:.0f})".format(scale))
    ax[1].plot(Ey-np.mean(Ey), "r-", label="Effective Energy")
    ax[1].set_ylim(-65, 65)
    ax[1].set_ylabel("Energy", fontsize=25, weight="bold")
    ax[1].set_xticklabels([0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_xlabel("Time($\mu$s)", fontsize=25, weight="bold")


def stateplot(ax, Smaty, clx=None):
    im = ax.imshow(Smaty, interpolation='nearest', cmap=cm.jet)
    ax.set_xlim(0, Smaty.shape[0]+1)
    ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Time($\mu$s)", fontsize=30, weight="bold")
    ax.invert_yaxis()
    ax.set_ylim(0, Smaty.shape[1]+1)
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Time($\mu$s)", fontsize=30, weight="bold")
    divider = make_axes_locatable(ax)
    rax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=rax)

    # plot cluster traj
    if clx is not None:
        istat = pd.DataFrame(np.unique(clx, return_index=True)
                             ).T.sort_values(by=[1])
        mx = pd.Series(index=istat[0], data=np.arange(0, len(istat))+1)
        clt = [1]
        clt[1:] = list(map(lambda e: mx[e], clx))
        clt.append(clt[-1])
        tax = divider.append_axes("top", size="25%", pad=0.1)
        tax.plot(clt, "bo-", ms=8)
        tax.set_ylabel("State", fontsize=25, weight="bold")
        tax.set_xticklabels([])
        tax.margins(y=0.1)


def dihSaliPlot(ax, sal):
    _ = ax.boxplot(sal.T, patch_artist=True, showfliers=False,
                   showmeans=True, widths=0.75,
                   medianprops={'lw': 2, 'color': 'orange'},
                   meanprops={'marker': 'o', 'mec': 'yellow',
                              'mfc': 'yellow', 'markersize': 10})
    ax.set_xlabel("Feature", fontsize=30, weight="bold")
    # ax.set_xticklabels([1,"",3,"",5,"",7,"",9,"",11,"",13,"",15,""])
    ax.set_xticklabels(["" if s == "" else f'${s}_{i}$' for i in range(
        1, 9) for s in ["\phi", "\psi"]])
    ax.xaxis.set_tick_params(rotation=-45)
    ax.set_ylabel("Saliency Score", fontsize=30, weight="bold")
