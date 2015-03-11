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
from matplotlib import pyplot as pl
import pmns_pca_utils
import pmns_utils

import lal
import pycbc.types


waveform_names=['apr_135135_lessvisc',
                'shen_135135_lessvisc',
                'dd2_135135_lessvisc' ,
                'dd2_165165_lessvisc' ,
                'nl3_1919_lessvisc' ,
                'nl3_135135_lessvisc' ,
                'tm1_135135_lessvisc' ,
                'tma_135135_lessvisc' ,
                'sfhx_135135_lessvisc',
                'sfho_135135_lessvisc']

npcs = len(waveform_names)

#
# Build Catalogues
#
fshift_center=1000
print "building catalogues"
(freqaxis, low_cat, high_cat, shift_cat, original_cat, fpeaks, low_sigmas,
        high_sigmas) = pmns_pca_utils.build_catalogues(waveform_names,
                fshift_center)
delta_f = np.diff(freqaxis)[0]

# Convert to magnitude/phase
full_mag, full_phase = pmns_pca_utils.complex_to_polar(original_cat)
shift_mag, shift_phase = pmns_pca_utils.complex_to_polar(shift_cat)
low_mag, low_phase = pmns_pca_utils.complex_to_polar(low_cat)
high_mag, high_phase = pmns_pca_utils.complex_to_polar(high_cat)

waveform = pmns_utils.Waveform(waveform_names[1])
waveform.reproject_waveform()

# Window
peakidx = np.argmax(abs(waveform.hplus.data))
win=lal.CreateTukeyREAL8Window(len(waveform.hplus),0.25)
waveform.hplus.data *= win.data.data

wavdata = np.zeros(16384)
wavdata[:len(waveform.hplus.data)] = np.copy(waveform.hplus.data)
waveform_TD = pycbc.types.TimeSeries(wavdata, delta_t=waveform.hplus.delta_t)
waveform_FD = waveform_TD.to_frequencyseries()

mag = abs(waveform_FD.data)
phase = np.unwrap(np.angle(waveform_FD.data))


# ******** #
# Plotting #
# ******** #
imageformats=['png','eps','pdf']

#
# Plot Catalogues
#

# Magnitude
f, ax = pl.subplots(ncols=3,figsize=(15,5))

peakidx = np.argmax(waveform.hplus.data)
timeaxis = waveform.hplus.sample_times - waveform.hplus.sample_times[peakidx]

ax[0].plot(1000*timeaxis, waveform.hplus, label='Shen (1.35+1.35)')
ax[0].set_xlim(min(1000*timeaxis),max(1000*timeaxis))
ax[0].set_xlabel('Time [ms]')
ax[0].set_ylabel('h(t)')
ax[0].minorticks_on()
ax[0].grid()

ax[1].semilogy(freqaxis, mag, label='Shen (1.35+1.35)')
ax[1].set_xlim(900, 4096)
ax[1].set_ylim(1e-27, 1e-23)
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('|H(f)|')
ax[1].minorticks_on()
ax[1].grid()

# Phase
ax[2].plot(freqaxis, phase,  label='Shen (1.35+1.35)')
ax[2].set_xlim(900, 5000)
ax[2].set_xlabel('Frequency [Hz]')
ax[2].set_ylabel('arg[H(f)]')
ax[2].minorticks_on()
ax[2].grid()

f.tight_layout()
pl.show()
for fileformat in imageformats:
    f.savefig('example_waveform.%s'%fileformat)


