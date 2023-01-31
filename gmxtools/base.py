#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2023
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2023-01-31 15:28:14
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-01-31 16:13:29
'''

import os
import sys
import getopt
import glob
import subprocess
from asyncio.subprocess import DEVNULL


def find_file(key):
    return glob.iglob(key, recursive=True)


def do_jobs(joblist, theOpts):
    if theOpts['show']:
        show_jobs(joblist, theOpts['NJOBS'])
    else:
        run_jobs(joblist, theOpts['NJOBS'])


def run_jobs(joblist, NJOBS):
    if (len(joblist) == 0):
        print("No Jobs will run!")
        exit

    np = 0
    for job in joblist:
        pid = os.fork()
        if pid == 0:
            subprocess.call(job, shell=True,
                            stdout=DEVNULL,
                            stderr=DEVNULL)
            exit(0)
        else:
            np = np + 1
            if np > NJOBS:
                os.wait()
    os.waitpid(0, 0)


def show_jobs(clist, NJOBS):
    print("The Max of Processes is {:d}".format(NJOBS))
    if (len(clist) == 0):
        print("No Jobs will run!")
    else:
        [print(repr(c)) for c in clist]


def parseOpts(argv):
    theOpts = {'NJOBS': 20, "show": False, "traj": "*"}
    try:
        opts, args = getopt.getopt(argv, "s:N:n:I:t:S",
                                   ["tpr=", "njobs=", "ndx=",
                                    "typein=", "traj=", "show"])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-s", "--tpr"):
            theOpts['tpr'] = arg
        elif opt in ("-N", "--njobs"):
            theOpts['NJOBS'] = int(arg)
        elif opt in ("-n", "--ndx"):
            theOpts['ndx'] = arg
        elif opt in ("-t", "traj"):
            theOpts['traj'] = arg
        elif opt in ('--typein'):
            theOpts['typein'] = arg
        elif opt in ("-S", "--show"):
            theOpts['show'] = True
        else:
            assert False
    return theOpts
