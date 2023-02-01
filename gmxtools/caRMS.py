#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-07-05 12:39:17
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-02-01 16:11:58
'''


import sys
import toolkits


def comlist(opts):
    clist = []
    for fn in toolkits.find_file(f"./**/{opts['traj']}.xtc"):
        cstr = "echo '3\n3\n' | gmx rms -f " + fn
        cstr += " -o " + fn.replace(".xtc", "-CaRMS.xvg")
        if 'tpr' in opts:
            cstr += " -s " + opts['tpr']
        else:
            cstr += " -s " + fn.replace(".xtc", ".tpr")
        clist.append(cstr)
    return clist


if __name__ == "__main__":
    theOpts = toolkits.parseOpts(sys.argv[1:])
    joblist = comlist(theOpts)
    toolkits.do_jobs(joblist, theOpts)
