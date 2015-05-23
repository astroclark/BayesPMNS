#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2015-2016 James Clark <clark@physics.umass.edu>
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

import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as pl

def parse_file(filename):

    f = open(filename,'r')
    nresults = len(f.readlines())
    f.close()

    f = open(filename,'r')

    names = ['Detector', 'EoS_name', 'job_number', 'SNR', 'scale_factor',
            'overlap', 'maximised_overlap']

    parsed_results = dict()
    parsed_results['H1'] = overlap_result(nresults, 'H1')
    parsed_results['L1'] = overlap_result(nresults, 'L1')
    parsed_results['V1'] = overlap_result(nresults, 'V1')

    for l, line in enumerate(f.readlines()):

        if line.split()[0]=='H1':
            parsed_results['H1'].append(line)
        elif line.split()[0]=='L1':
            parsed_results['L1'].append(line)
        elif line.split()[0]=='V1':
            parsed_results['V1'].append(line)

    return parsed_results
        

class overlap_result:

    def __init__(self, nresults, Detector):

        self.snr = []
        self.overlap = []
        self.max_overlap = []
        self.scale_factor = []
        self.Detector = Detector

    def append(self, line):

        names = ['Detector', 'EoS_name', 'job_number', 'SNR', 'scale_factor',
                'overlap', 'maximised_overlap']

        values = line.split()

        self.snr.append(float(values[names.index('SNR')]))
        self.overlap.append(float(values[names.index('overlap')]))
        self.max_overlap.append(float(values[names.index('maximised_overlap')]))
        self.scale_factor.append(float(values[names.index('scale_factor')]))

def maximise_over_detectors(parsed_results):

    nresults = len(parsed_results['H1'].snr)

    max_overlaps = np.zeros(nresults)
    net_snr = np.zeros(nresults)
    scale_factors = np.copy(parsed_results['H1'].scale_factor)

    for i in xrange(nresults):

        snrs = np.array([parsed_results['H1'].snr[i],
            parsed_results['L1'].snr[i], parsed_results['V1'].snr[i]])

        det_max_overlaps = np.array([parsed_results['H1'].max_overlap[i],
            parsed_results['L1'].max_overlap[i],
            parsed_results['V1'].max_overlap[i]])

        max_overlaps[i] = det_max_overlaps[np.argmax(snrs)]

        #net_snr[i] = snrs.max()
        net_snr[i] = np.sqrt(sum(snrs**2))


    return max_overlaps, net_snr, scale_factors

# --------------
# MAIN
data_file = sys.argv[1]
data_files = ['overlap_dd213.dat', 'overlap_nl313.dat', 'overlap_sfho.dat',
'overlap_tm1.dat', 'overlap_apr.dat', 'overlap_dd216.dat',
'overlap_nl319.dat', 'overlap_sfhx.dat', 'overlap_tma.dat']


f1, ax1 = pl.subplots()
f2, ax2 = pl.subplots()

all_overlaps=[]
small_scale_overlaps = []

for data_file in data_files:

    parsed_results = parse_file(data_file)
    max_overlaps, net_snr, scale_factors = maximise_over_detectors(parsed_results)

    label=data_file.replace('.dat','')
    label=label.replace('overlap_','')
    print label
    ax1.plot(net_snr, max_overlaps, '.', label=label)

    # --------------
    # PLOTTING
    all_overlaps.append(max_overlaps)

    # XXX: stick with the smallest scale factor for now
    idx = scale_factors==min(np.unique(scale_factors))
    small_scale_overlaps.append(max_overlaps[idx])

ax1.legend(loc='lower right')
ax1.set_xlabel('network SNR of reconstruction')
ax1.set_ylabel('Match')
ax1.minorticks_on()

f1.savefig('recSNR_vs_match.png')

small_scale_overlaps=np.concatenate(small_scale_overlaps)
small_scale_overlaps=small_scale_overlaps[np.isfinite(small_scale_overlaps)]

ax2.hist(small_scale_overlaps, histtype='stepfilled', alpha=0.5, bins=50)
ax2.set_xlabel('Reconstruction Match (all EOS) @ 4 Mpc (sky-averaged)')
ax2.set_xlim(0,1)

f2.savefig('small_scale_matches.png')

pl.show()


