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
#from scipy import signal, optimize, special, stats

#import pmns_utils
#import pmns_simsig as simsig

#import lal
#import lalsimulation as lalsim

from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.filter import overlap, sigma, sigmasq
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc import types


def fmoment(n, htilde, psd, frange):
    """
    Compute the n-th moment of frequency
    """

    # Get frequency series of waveform
    f  = htilde.get_sample_frequencies()

    fhtilde = types.FrequencySeries(initial_array=htilde.data * f.data**n,
            delta_f=htilde.delta_f)

    snrsq = sigmasq(htilde=htilde, psd=psd, low_frequency_cutoff=frange[0],
            high_frequency_cutoff=frange[1])

    return (1.0/snrsq) * overlap(fhtilde, htilde, psd=psd,
            low_frequency_cutoff=frange[0], high_frequency_cutoff=frange[1],
            normalized=False)

def effbandwidth(htilde, psd, frange):
    """
    Compute the effective bandwidth
    """
    sigmaf2 = fmoment(2, htilde, psd, frange) \
            - fmoment(1, htilde, psd, frange)**2

    return np.sqrt(sigmaf2)

def timing_uncertainty(htilde, psd, frange):
    
    sigmaf = effbandwidth(htilde, psd, frange)

    snr = sigma(htilde=htilde, psd=psd, low_frequency_cutoff=frange[0],
            high_frequency_cutoff=frange[1])

    return 1.0 / (2.0*np.pi*snr*sigmaf)


# 
# Calculations
#


f_low = 10
sample_rate = 8192
f_high = sample_rate / 2

# Generate the two waveforms to compare
print  "Generating waveform..."
if 1:
    approx="TaylorT4"
    hp, hc = get_td_waveform(approximant=approx,
                             mass1=1.35,
                             mass2=1.35,
                             f_lower=f_low,
                             delta_t=1.0/sample_rate,
                             distance=50)
    delta_f = 1.0 / hp.duration
    flen = len(hp)/2 + 1
    htilde = hp.to_frequencyseries()

else:
    approx="TaylorF2RedSpinTidal"
    #approx="SpinTaylorF2"
    #approx="IMRPhenomP"
    delta_f = 0.5
    htilde, _ = get_fd_waveform(approximant=approx,
                             mass1=1.35,
                             mass2=1.35,
                             f_lower=f_low,
                             f_final=5000,
                             inclination=0.0,
                             delta_f=delta_f,
                             distance=50)
    flen=len(htilde)

# Generate the aLIGO ZDHP PSD
psd = aLIGOZeroDetHighPower(flen, delta_f, f_low) 

# Compute effective bandwidth

print  "Computing effective bandwidth..."
print "----"

print "SNR:"
print sigma(htilde=htilde, psd=psd, low_frequency_cutoff=f_low,
            high_frequency_cutoff=f_high)

print  "Effective bandwidth (Hz):"

print effbandwidth(htilde, psd=psd, frange=[f_low, f_high])

print  "Timing uncertainty (ms):"
print 1e3*timing_uncertainty(htilde, psd=psd, frange=[f_low, f_high])

print  "max frequency of template (Hz):"
print max(htilde.get_sample_frequencies())




