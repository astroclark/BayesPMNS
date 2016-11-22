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
    parser.add_option("-W", "--waveforms", default=None, type=str)
    parser.add_option("-s", "--srate", default=8192., type=float)
    parser.add_option("-m", "--min-dist", default=50., type=float)
    parser.add_option("-v", "--verbose", default=False, action="store_true")
    parser.add_option("-T", "--peak-threshold", default=0.3, type=float)

    (opts,args) = parser.parse_args()
    
    return opts, args

#
# Input
#
opts, args = parser()

#
# Load data
#
waves = np.loadtxt(opts.waveforms)
if len(np.shape(waves))==1:
    waves = [waves]


## XXX: work with median waveform for testing
#waves = [np.median(waves, axis=0)]

peak_freqs = []

fpeak = np.zeros(len(waves))
fspiral = np.zeros(len(waves))
f20 = np.zeros(len(waves))

for w in xrange(len(waves)):
    if opts.verbose:
        print >> sys.stdout, "Finding peaks for waveform %d/%d"%(
                w, len(waves))

    psd = wave_psd(waves[w])

    #
    # Peak detection
    #
    peak_indices = peakutils.indexes(psd.data, min_dist=opts.min_dist,
            thres=opts.peak_threshold)
    peak_freqs.append(psd.sample_frequencies[peak_indices])

    #
    # Peak identification
    #
    fpeak[w], fspiral[w], f20[w] = identify_peaks(peak_freqs[w])

#   if len(waves)==1:
#       f, ax = pl.subplots()
#       ax.semilogy(psd.sample_frequencies, psd)
#       ax.semilogy(psd.sample_frequencies[peak_indices], psd[peak_indices], 'ro')
#       ax.set_xlim(999, 4096)
#
#       pl.show()

#
# Remove nans
#
fpeak = fpeak[~np.isnan(fpeak)]
fspiral = fspiral[~np.isnan(fspiral)]
f20 = f20[~np.isnan(f20)]

#
# Write results txt
#
f=open(opts.waveforms.replace('waveform','waveform_FreqPeaks'),'w')
f.writelines('# fpeak_mean fpeak_std fspiral_mean fspiral_std f20_mean f20_std\n')
f.writelines('%f %f %f %f %f %f\n'%(
    np.mean(fpeak), np.std(fpeak),
    np.mean(fspiral), np.std(fspiral),
    np.mean(f20), np.std(f20)))
f.close()





