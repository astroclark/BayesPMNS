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
#np.seterr(all="raise", under="ignore")
import matplotlib
#matplotlib.use("Agg")

import lal
import lalsimulation as lalsim
import pycbc.types

import pmns_utils

import pylab as pl

def apply_taper(TimeSeries,newlen=16384):
    """
    Smoothly taper the start of the data in TimeSeries using LAL tapering
    routines

    Also zero pads
    """

    # Create and populate lal time series
    tmp=lal.CreateREAL8TimeSeries('tmp', lal.LIGOTimeGPS(), 0.0,
            TimeSeries.delta_t, lal.StrainUnit, newlen)
            #TimeSeries.delta_t, lal.StrainUnit, len(TimeSeries))
    tmp.data.data = np.zeros(newlen)
    tmp.data.data[0:len(TimeSeries)]=TimeSeries.data

    # Taper
    lalsim.SimInspiralREAL8WaveTaper(tmp.data, lalsim.SIM_INSPIRAL_TAPER_START)

    return pycbc.types.TimeSeries(tmp.data.data, delta_t=TimeSeries.delta_t)

# ------------------- MAIN ----------------------

waveform_names = ['nl3_135135', 'shen_135135', 'dd2_135135', 'apr_135135', 'tma_135135']
fmin=2000
fmax=8192

pl.figure()
for waveform_name in waveform_names:
    print ''
    print '--- %s ---'%waveform_name
    waveform = pmns_utils.Waveform('%s_lessvisc'%waveform_name)

    # Get optimal hplus
    waveform.reproject_waveform()

    h = apply_taper(waveform.hplus)
    h_s = pycbc.filter.sigma(h, low_frequency_cutoff=fmin,
            high_frequency_cutoff=fmax)

    #
    # Spectrum
    #
    hf = h.to_frequencyseries()
    H = abs(h.to_frequencyseries())
    freqs = h.to_frequencyseries().sample_frequencies.data


    # reduce
    H=H[(freqs>=fmin)*(freqs<=fmax)]
    freqs=freqs[(freqs>=fmin)*(freqs<=fmax)]

    idx = np.argmax(H)

    # Gaussian peaks
    sigmas = 50 + 10*np.random.randn(1e3)
    mu = freqs[idx]


    for s in sigmas:
        y = np.exp(-0.5*(freqs-mu)**2 / s**2)
        y *= H[idx]*max(y)
        pl.plot(freqs,y, 'grey')

    pl.plot(freqs,H, 'k', label=waveform_name)

pl.show()





