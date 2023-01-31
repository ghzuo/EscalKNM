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
@Last Modified Time: 2023-01-31 19:14:40
'''

import sys
import base
import potential
import rama
import rms


def comlist(task, theOpts):
    if task == "potential":
        return potential.comlist(theOpts)
    elif task == 'rama':
        return rama.comlist(theOpts)
    elif task == 'CaRMS':
        return rms.comlist(theOpts)
    else:
        print("Unknow task: ", task)
        return []


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please input the task\n")
        exit(1)
    tasks = []
    if (sys.argv[1] == "All"):
        tasks = ["potential", "rama", "CaRMS"]
    else:
        tasks = sys.argv[1].split(',')
    theOpts = base.parseOpts(sys.argv[2:])
    joblist = []
    for task in tasks:
        joblist.extend(comlist(task, theOpts))
    base.do_jobs(joblist, theOpts)
