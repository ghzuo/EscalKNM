#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2023
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2023-06-30 20:16:56
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-06-30 21:29:37
'''


import sys
import os
import mdtraj as mdt
import toolkits


def genNdxFile(grofile, ndxfile):
    gro = mdt.load(grofile)
    ndx = gro.top.select("name == 'CA'")
    ndxlist = []
    for i in range(len(ndx)-1):
        ndxi = ndx[i] + 1
        for j in range(i+2, len(ndx)):
            ndxj = ndx[j] + 1
            ndxlist.extend((str(ndxi), str(ndxj)))
    with open(ndxfile, 'w') as file:
        file.write('[ Ca-pair ]\n')
        for count, x in enumerate(ndxlist, start=1):
            file.write(" %4s" % x)
            if count % 15 == 0:
                file.write("\n")


def comlist(opts):
    ndxfile = "dist.ndx"
    clist = []
    for fn in toolkits.find_file(f"./**/*{opts['traj']}.xtc"):
        if not os.path.isfile(ndxfile):
            grofile = fn.replace(".xtc", ".gro")
            genNdxFile(grofile, ndxfile)
        ofile = fn.replace(".xtc", "-CaDist.xvg")
        tprfile = fn.replace(".xtc", ".tpr")
        cstr = f"gmx distance -f {fn} -oall {ofile} -s {tprfile} \
            -n {ndxfile} -select 0"
        clist.append(cstr)
    return clist


if __name__ == "__main__":
    theOpts = toolkits.parseOpts(sys.argv[1:])
    joblist = comlist(theOpts)
    toolkits.do_jobs(joblist, theOpts)
