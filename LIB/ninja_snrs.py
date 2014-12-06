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
#np.seterr(all="raise", under="ignore")
import matplotlib
#matplotlib.use("Agg")

from scipy import signal, optimize, special, stats

import pmns_utils
from pylal import Fr
import lal
import lalsimulation as lalsim
import pycbc.filter
import pycbc.psd

# ------------------------
# Setup

datalen=2

# set up channels
sites=['H', 'L', 'V', 'G']
channels=[site+'1:STRAIN' for site in sites]

# read frame data
data = {}

# ------------------------
#
# Load Data
#
times = np.loadtxt(sys.argv[1])
frame = sys.argv[2]
distance = 5.0
waveform_name = 'shen_135135_lessvisc'

waveform=pmns_utils.Waveform(waveform_name)
waveform.reproject_waveform()
waveform.compute_characteristics()
optimal_snr = waveform.snr_plus * 20.0 / distance


#
# Extract data
#
for s,site in enumerate(sites):
    framedata = Fr.frgetvect(frame, channels[s])
    data[site] = pycbc.types.TimeSeries(framedata[0],
            delta_t=framedata[3][0], epoch=framedata[1])

data_times = data[site].sample_times


#
# Identify injection times in this frame
#
times = times[(times>=framedata[1]) *
        (times<=framedata[1]+len(framedata[0])*framedata[3][0])]


#
# Compute SNRs
#
snrs={}
for s,site in enumerate(sites):

    snrs[site] = np.zeros(len(times))

    for t,time in enumerate(times):
        print '%d of %d for %s'%(t, len(times), site)

        tmp = data[site]
        idx = (data_times.data[:]>time-0.5*datalen)*\
                (data_times.data[:]<time+0.5*datalen)

        tmpdat = pycbc.types.TimeSeries(data[site].data[idx],
                delta_t=data[site].delta_t)

        print '... computing snr ...'

        if site=='H' or site=='L':
            psd = \
                    pycbc.psd.aLIGOZeroDetHighPower(len(tmpdat.to_frequencyseries()),
                            tmpdat.to_frequencyseries().delta_f,
                            low_freq_cutoff=10)
        elif site=='V':
            psd = \
                    pycbc.psd.AdvVirgo(len(tmpdat.to_frequencyseries()),
                            tmpdat.to_frequencyseries().delta_f,
                            low_freq_cutoff=10)
        elif site=='G':
            psd = \
                    pycbc.psd.GEOHF(len(tmpdat.to_frequencyseries()),
                            tmpdat.to_frequencyseries().delta_f,
                            low_freq_cutoff=10)

        snrs[site][t] = pycbc.filter.sigma(tmpdat, psd=psd,
                low_frequency_cutoff=1000)



#
# Write output
#
f=open('snrs.dat')
for i in xrange(len(times)):
    f.writelines('%f %f %f %f %f\n'%
            optimal_snr, snrs['H'][i],
            snrs['L'][i],
            snrs['V'][i]
            snrs['G'][i])

f.close()







