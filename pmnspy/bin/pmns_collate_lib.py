#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <clark@physics.umass.edu>
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

from __future__ import division
import os,sys
import numpy as np

import cPickle as pickle

import glob


# -------------------------------
# Load results

waveform_name=sys.argv[1]
filepattern='LIB-PMNS_waveform-%s'%waveform_name

# XXX: Hardcoding
distance=float(sys.argv[2])

# Identify files
globpattern = filepattern+'*seed*distance-%.1f*pickle'%distance
datafiles = glob.glob(globpattern)

logBs   = np.zeros(len(datafiles))
netSNRs = np.zeros(len(datafiles))
freq_pdfs = []
freq_maxL = np.zeros(len(datafiles))
freq_low  = np.zeros(len(datafiles))
freq_upp  = np.zeros(len(datafiles))
freq_area = np.zeros(len(datafiles))

# Load each file
for d, datafile in enumerate(datafiles):
    print >> sys.stdout, "Loading %d of %d (%s)"%(d, len(datafiles), datafile)

    try:
        (this_logB, _, _, _, this_netSNR, _, _, freq_axis, this_freq_pdf,
                this_freq_maxL, this_freq_low, this_freq_upp, this_freq_area) = \
                        pickle.load(open(datafile))
    except pickle.UnpicklingError:
        continue

    logBs[d] = this_logB
    netSNRs[d] = this_netSNR
    freq_pdfs.append(this_freq_pdf)
    freq_maxL[d] = this_freq_maxL
    freq_low[d] = this_freq_low
    freq_upp[d] = this_freq_upp
    freq_area[d] = this_freq_area

pickle.dump((logBs, netSNRs, freq_pdfs, freq_axis, freq_maxL, freq_low,
    freq_upp, freq_area), open(filepattern+'_distance-%.1f.pickle'%distance, 'wb'))

print >> sys.stdout, "written: %s"%filepattern+'_distance-%.1f.pickle'%distance
print >> sys.stdout, "DONE."


