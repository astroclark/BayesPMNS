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
import pycbc.filter


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

#
# PCA
#
print "Performing PCA"
shift_pca = pmns_pca_utils.pca_magphase(shift_cat, freqaxis, flow=10)
full_pca = pmns_pca_utils.pca_magphase(original_cat, freqaxis, flow=10)

# Reconstruct the Shen with 1 PC
firstPC=pmns_pca_utils.unshift_waveform(shift_pca, [1,1], fpeaks[1],
        target_freqs=freqaxis, waveform_num=1, fcenter=fshift_center,
        delta_f=delta_f)

firstPC_FD = pycbc.types.FrequencySeries(firstPC, delta_f=delta_f)
firstPC_sigma = pycbc.filter.sigma(firstPC_FD)
firstPC_FD.data /= firstPC_sigma

# Reconstruct the Shen with all PCS
allPC=pmns_pca_utils.unshift_waveform(shift_pca, [npcs,npcs], fpeaks[1],
        target_freqs=freqaxis, waveform_num=1, fcenter=fshift_center,
        delta_f=delta_f)
allPC_FD = pycbc.types.FrequencySeries(allPC, delta_f=delta_f)
allPC_sigma = pycbc.filter.sigma(allPC_FD)
allPC_FD.data /= allPC_sigma

# Get the example waveform from the original catalogue
waveform_FD = pycbc.types.FrequencySeries(original_cat[:,1], delta_f=delta_f)
waveform_sigma = pycbc.filter.sigma(waveform_FD)
waveform_FD.data /= waveform_sigma

# Now Get TD versions
firstPC_TD  = firstPC_FD.to_timeseries()
allPC_TD    = allPC_FD.to_timeseries()
waveform_TD = waveform_FD.to_timeseries()

# Put the data in the middle of a timeseries for convenient plotting
nright=2000
nleft=500
firstPC_TD_plot  = np.zeros(len(firstPC_TD))
firstPC_TD_plot[8192:8192+nright] = firstPC_TD.data[:nright]
firstPC_TD_plot[8192-nleft:8192] = firstPC_TD.data[len(firstPC_TD)-nleft:]

allPC_TD_plot  = np.zeros(len(allPC_TD))
allPC_TD_plot[8192:8192+nright] = allPC_TD.data[:nright]
allPC_TD_plot[8192-nleft:8192] = allPC_TD.data[len(allPC_TD)-nleft:]

waveform_TD_plot  = np.zeros(len(waveform_TD))
waveform_TD_plot[8192:8192+nright] = waveform_TD.data[:nright]
waveform_TD_plot[8192-nleft:8192] = waveform_TD.data[len(waveform_TD)-nleft:]


# Get max match offsets for plotting
firstPCmatch = pycbc.filter.match(waveform_FD, firstPC_FD,
        low_frequency_cutoff=1000)
allPCmatch = pycbc.filter.match(waveform_FD, allPC_FD,
        low_frequency_cutoff=1000)

firstPCoffset = 16384 - firstPCmatch[1] - 1
allPCoffset = 16384 - allPCmatch[1] - 1



# ******** #
# Plotting #
# ******** #
imageformats=['png','eps','pdf']


# First PC
f, ax = pl.subplots(nrows=2,figsize=(7,10))

peakidx  = np.argmax(waveform_TD_plot)
timeaxis = np.arange(0, 1, 1./16384)
timeaxis -= timeaxis[peakidx]

ax[0].plot(1000*timeaxis, waveform_TD_plot, label='Shen (1.35+1.35): Original',
        color='k', linewidth=2)

peakidx  = np.argmax(waveform_TD_plot) - firstPCoffset
timeaxis = np.arange(0, 1, 1./16384)
timeaxis -= timeaxis[peakidx]

# Time Domain
ax[0].plot(1000*timeaxis, firstPC_TD_plot, label='Shen (1.35+1.35): 1st PC',
        color='r')
ax[0].set_xlim(-5,25)
ax[0].set_xlabel('Time [ms]')
ax[0].set_ylabel('h(t)')
ax[0].minorticks_on()
ax[0].grid()
ax[0].legend()
ax[0].set_title("Match using 1st PC: %.3f"%firstPCmatch[0])

# F-domain Magnitude
ax[1].semilogy(freqaxis, abs(waveform_FD), label='Shen (1.35+1.35): Original',
        color='k', linewidth=2)
ax[1].semilogy(freqaxis, abs(firstPC_FD), label='Shen (1.35+1.35): 1st PC',
        color='r', linewidth=1)
ax[1].set_xlim(900, 4096)
ax[1].set_ylim(1e-4, 1e-1)
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('|H(f)|')
ax[1].minorticks_on()
ax[1].grid()
#ax[1].legend()

# Phase
#   ax[2].plot(freqaxis, np.unwrap(np.angle(waveform_FD)),  
#           label='Shen (1.35+1.35): Original', color='k', linewidth=2)
#   ax[2].plot(freqaxis, np.unwrap(np.angle(firstPC_FD)),
#           label='Shen (1.35+1.35): 1st PC', color='r')
#   ax[2].set_xlim(900, 4096)
#   ax[2].set_xlabel('Frequency [Hz]')
#   ax[2].set_ylabel('arg[H(f)]')
#   ax[2].minorticks_on()
#   ax[2].grid()

f.tight_layout()
pl.show()
for fileformat in imageformats:
    f.savefig('firstPC_example_waveform_reconstruction.%s'%fileformat)


# All PCs
f, ax = pl.subplots(nrows=2,figsize=(7,10))

peakidx  = np.argmax(waveform_TD_plot)
timeaxis = np.arange(0, 1, 1./16384)
timeaxis -= timeaxis[peakidx]

ax[0].plot(1000*timeaxis, waveform_TD_plot, label='Shen (1.35+1.35): Original',
        color='k', linewidth=2)

peakidx  = np.argmax(waveform_TD_plot) - allPCoffset
timeaxis = np.arange(0, 1, 1./16384)
timeaxis -= timeaxis[peakidx]

# Time Domain
ax[0].plot(1000*timeaxis, -1*allPC_TD_plot, label='Shen (1.35+1.35): All PCs',
        color='r')
ax[0].set_xlim(-5,25)
ax[0].set_xlabel('Time [ms]')
ax[0].set_ylabel('h(t)')
ax[0].minorticks_on()
ax[0].grid()
ax[0].legend()
ax[0].set_title("Match using All PCs: %.3f"%allPCmatch[0])

# F-domain Magnitude
ax[1].semilogy(freqaxis, abs(waveform_FD), label='Shen (1.35+1.35): Original',
        color='k', linewidth=2)
ax[1].semilogy(freqaxis, abs(allPC_FD), label='Shen (1.35+1.35): All PCs',
        color='r', linewidth=1)
ax[1].set_xlim(900, 4096)
ax[1].set_ylim(1e-4, 1e-1)
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('|H(f)|')
ax[1].minorticks_on()
ax[1].grid()
#ax[1].legend()

#   # Phase
#   ax[2].plot(freqaxis, np.unwrap(np.angle(waveform_FD)),  
#           label='Shen (1.35+1.35): Original', color='k', linewidth=2)
#   ax[2].plot(freqaxis, np.unwrap(np.angle(allPC_FD)),
#           label='Shen (1.35+1.35): All PCs', color='r')
#   ax[2].set_xlim(900, 4096)
#   ax[2].set_xlabel('Frequency [Hz]')
#   ax[2].set_ylabel('arg[H(f)]')
#   ax[2].minorticks_on()
#   ax[2].grid()

f.tight_layout()
pl.show()
for fileformat in imageformats:
    f.savefig('allPCs_example_waveform_reconstruction.%s'%fileformat)
