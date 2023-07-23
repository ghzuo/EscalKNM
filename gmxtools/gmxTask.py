#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-07-03 19:29:45
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-07-19 11:17:02
'''

import sys
import toolkits
import caRMS
import rama
import caDist
import potential


# All function and label
funct = {'rama': rama.comlist,
         'potential': potential.comlist,
         'caRMS': caRMS.comlist,
         'caDist': caDist.comlist
         }

# Check the setting
if len(sys.argv) == 1:
    print("Please input the task\n")
    exit(1)
tasks = funct.keys() if (sys.argv[1].lower() == "all") \
    else sys.argv[1].split(',')
theOpts = toolkits.parseOpts(sys.argv[2:])
joblist = []
for task in tasks:
    if task in funct:
        joblist.extend(funct[task](theOpts))
    else:
        print("Unknow task: ", task)

# execute jobs
toolkits.do_jobs(joblist, theOpts)
