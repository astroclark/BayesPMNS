#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
"""

import os, sys
import glob
from optparse import OptionParser
import cPickle as pickle
import numpy as np


def parser():

    # --- Command line input
    parser = OptionParser()

    parser.add_option("-r", "--results-dir", default="./", type=str)
    parser.add_option("-g", "--glob-pat", default="posterior_V1H1L1", type=str)
    parser.add_option("-o", "--out-name", default="alldata", type=str)
    parser.add_option("-p", "--param-name", default="frequency", type=str)

    (opts,args) = parser.parse_args()

    return opts, args

opts, args = parser()

possampsfiles=glob.glob('%s/%s*.dat'%(opts.results_dir, opts.glob_pat))
if not possampsfiles:
    print 'no files matching %s/%s*.dat'%(opts.results_dir, opts.glob_pat)
    print 'exiting.'
outfile=opts.out_name + '_' + opts.glob_pat + '.pickle'
outfile=outfile.replace('*','')

possamps=[]
detstats=[]

for p, possampfile in enumerate(possampsfiles):

    if p==0:
        f = open(possampfile, 'r')
        params = f.readline().split('\t')[:-1]
        colnum = params.index(opts.param_name)

    possamps.append(np.loadtxt(possampfile, usecols=[colnum], skiprows=1))
    detstats.append(np.loadtxt(possampfile+"_B.txt"))

pickle.dump((detstats,possamps), open(outfile, "wb"))
