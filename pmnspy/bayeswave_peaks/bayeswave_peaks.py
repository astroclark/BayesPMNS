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
import os,sys
import numpy as np
from optparse import OptionParser

import pycbc.types

import peakutils

from matplotlib import pyplot as pl

def wave_psd(wave_sample, srate=8192):
    """
    Return the |H(f)| for the waveform sample
    """

    pywave = pycbc.types.TimeSeries(wave_sample, delta_t=1.0/srate)
    psd = pywave.to_frequencyseries()

    psd.data = abs(psd)

    return psd

def identify_peaks(peaks):
    """
    Return best guess of f20, fspiral and fpeak
    """

    if len(peaks)==3:
        fpeak=peaks[2]
        fspiral=peaks[1]
        f20=peaks[0]
    elif len(peaks)==2:
        fpeak=peaks[1]
        fspiral=np.nan
        f20=np.nan
    else:
        fpeak=np.nan
        fspiral=np.nan
        f20=np.nan

    return fpeak, fspiral, f20


def parser():
    """
    Parser for input (command line and ini file)
    """

    # --- cmd line
    parser = OptionParser()
    parser.add_option("-H", "--hanford-waves", default=None, type=str)
    parser.add_option("-L", "--livingston-waves", default=None, type=str)
    parser.add_option("-V", "--virgo-waves", default=None, type=str)
    parser.add_option("-s", "--srate", default=8192., type=float)
    parser.add_option("-m", "--min-dist", default=100., type=float)

    (opts,args) = parser.parse_args()
    
    return opts, args

#
# Input
#
opts, args = parser()

#
# Load data
#
if opts.hanford_waves is not None:
    hanford_waves = np.loadtxt(opts.hanford_waves)

## XXX: work with median waveform for testing
#hanford_waves = [np.median(hanford_waves, axis=0)]

h_peak_freqs = []

fpeak = np.zeros(len(hanford_waves))
fspiral = np.zeros(len(hanford_waves))
f20 = np.zeros(len(hanford_waves))

for w in xrange(len(hanford_waves)):
    print >> sys.stdout, "Finding peaks for H waveform %d/%d"%(
            w, len(hanford_waves))

    h_psd = wave_psd(hanford_waves[w])

    #
    # Peak detection
    #
    h_peak_indices = peakutils.indexes(h_psd.data, min_dist=opts.min_dist,
            thres=0.5)
    h_peak_freqs.append(h_psd.sample_frequencies[h_peak_indices])

    fpeak[w], fspiral[w], f20[w] = identify_peaks(h_peak_freqs[w])

    if len(hanford_waves)==1:
        f, ax = pl.subplots()
        ax.semilogy(h_psd.sample_frequencies, h_psd)
        ax.semilogy(h_psd.sample_frequencies[h_peak_indices], h_psd[h_peak_indices], 'ro')
        ax.set_xlim(999, 4096)
 
        pl.show()

# Concatenate to get posterior on peak locations
h_peak_freqs = np.concatenate(h_peak_freqs)


