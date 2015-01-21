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
import cPickle as pickle
import numpy as np


def parser():

    # --- Command line input
    parser = OptionParser()

    parser.add_option("-d", "--web-dir", default="./", type=str)
    parser.add_option("-o", "--out-name", default="alldata", type=str)
    parser.add_option("-p", "--pos-samps", default="posterior_samples.dat",
            type=str)
    parser.add_option("-s", "--sub-dir", default="V1H1L1")

    (opts,args) = parser.parse_args()

    return opts, args


resultsdirs=os.listdir(opts.web_dir)

freqsamps=[]
hrsssamps=[]
logBsn=[]
logBci=[]

for r, resdir in enumerate(resultsdirs):

        try:
            freqdata = np.loadtxt('./%s/%s/%s'%(resdir, opts.sub_dir,
                opts.pos_samps),
                    skiprows=1, usecols=[1])
            allfreqdata.append(freqdata)
        except:
            continue


pickle.dump(allfreqdata, open("allfreqdata.pickle", "wb"))
