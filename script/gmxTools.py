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
@Last Modified Time: 2022-09-26 10:09:28
'''


import os
import sys
import glob
import getopt
import subprocess
from asyncio.subprocess import DEVNULL


def find_file(key):
    return glob.iglob(key, recursive=True)


def run_jobs(joblist, NJOBS=40):
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


def show_jobs(clist, NJOBS=40):
    for com in clist:
        print(repr(com))
    print("run jobs in {:d} processes".format(NJOBS))


def rama_clist():
    clist = []
    for fn in find_file("./**/*.xtc"):
        ofile = fn.replace(".xtc", "-rama.xvg")
        tprfile = fn.replace(".xtc", ".tpr")
        cstr = f"gmx rama -f {fn} -o {ofile} -s {tprfile}"
        clist.append(cstr)
    return clist


def potential_clist():
    clist = []
    for fn in find_file("./**/*.edr"):
        ofile = fn.replace(".edr", "-potential.xvg")
        cstr = f"echo 11 | gmx energy -f {fn} -o {ofile}"
        clist.append(cstr)
    return clist


def caRMS_clist(opts):
    clist = []
    for fn in find_file("./**/*.xtc"):
        cstr = "echo '3\n3\n' | gmx rms -f " + fn
        cstr += " -o " + fn.replace(".xtc", "-CaRMS.xvg")
        if 'tpr' in opts:
            cstr += " -s " + opts['tpr']
        else:
            cstr += " -s " + fn.replace(".xtc", ".tpr")
        clist.append(cstr)
    return clist


def parseOpts(argv):
    theOpts = {'NJOBS': 36}
    try:
        opts, args = getopt.getopt(argv, "s:N:n:I:",
                                   ["tpr=", "njobs=", "ndx=", "typein="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-s", "--tpr"):
            theOpts['tpr'] = arg
        elif opt in ("-N", "--njobs"):
            theOpts['NJOBS'] = int(arg)
        elif opt in ("-n", "--ndx"):
            theOpts['ndx'] = arg
        elif opt in ('--typein'):
            theOpts['typein'] = arg
        else:
            assert False
    return theOpts


def comlist(task, theOpts):
    if task == "potential":
        return potential_clist()
    elif task == 'rama':
        return rama_clist()
    elif task == 'CaRMS':
        return caRMS_clist(theOpts)
    else:
        print("Unknow task: ", task)
        return []


def doTask(task, argv):
    theOpts = parseOpts(argv)
    joblist = comlist(task, theOpts)
    run_jobs(joblist, theOpts['NJOBS'])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please input the task\n")
        exit(1)
    tasks = []
    if (sys.argv[1] == "All"):
        tasks = ["potential", "rama", "CaRMS"]
    else:
        tasks = sys.argv[1].split(',')
    theOpts = parseOpts(sys.argv[2:])
    joblist = []
    for task in tasks:
        joblist.extend(comlist(task, theOpts))
    run_jobs(joblist, theOpts['NJOBS'])
