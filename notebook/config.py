#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-09-28 10:22:36
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-09-28 10:31:14
'''


import os

# the super directory of data
gmxpath = os.path.expanduser('~') + "/data/gromacs/"
superpath = gmxpath + "data.Villin/2f4k/"

# the subdirectories
unfoldpath = f"{superpath}unfolding/"
datapath   = f"{superpath}htSeed/unfolding.360.step50/clust-gromos_0.6/298/"
