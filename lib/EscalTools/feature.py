#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:36:20
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-08-22 17:45:01
'''

import numpy as np


def dih2X(dih):
    # set the dih
    nres = dih['Residue'].nunique()
    dih['frame'] = dih.index // nres
    dih['index'] = dih.index % nres
    dih['Psi'] = dih['Psi']*np.pi/180
    dih['Phi'] = dih['Phi']*np.pi/180

    # set the dataframe of X
    X = dih[["frame", "index"]]
    X['PhiCos'] = np.cos(dih['Phi'])
    X['PhiSin'] = np.sin(dih['Phi'])
    X['PsiCos'] = np.cos(dih['Psi'])
    X['PsiSin'] = np.sin(dih['Psi'])
    return X.melt(id_vars=["frame", "index"],
                  value_vars=["PhiCos", "PhiSin", "PsiCos", "PsiSin"]
                  ).pivot_table(index="frame", columns=["index", "variable"])
