#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-07-05 12:37:15
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-09-26 09:58:27
'''


import sys
import gmxTools as gt


gt.doTask("rama", sys.argv[1:])
