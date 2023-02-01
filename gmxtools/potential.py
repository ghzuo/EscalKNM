#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-07-05 12:39:56
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-02-01 16:12:16
'''


import sys
import toolkits


def comlist(opts):
    clist = []
    for fn in toolkits.find_file(f"./**/{opts['traj']}.edr"):
        ofile = fn.replace(".edr", "-potential.xvg")
        cstr = f"echo 11 | gmx energy -f {fn} -o {ofile}"
        clist.append(cstr)
    return clist


if __name__ == "__main__":
    theOpts = toolkits.parseOpts(sys.argv[1:])
    joblist = comlist(theOpts)
    toolkits.do_jobs(joblist, theOpts)
