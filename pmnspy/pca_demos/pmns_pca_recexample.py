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
pmns_pca_matches.py

Script to produce matches from PCA of post-merger waveforms
"""

from __future__ import division
import os,sys
import cPickle as pickle
import numpy as np

from matplotlib import pyplot as pl

import lal
import pycbc.types
import pycbc.filter
from pycbc.psd import aLIGOZeroDetHighPower

import pmns_utils as pu
import pmns_pca_utils as ppca

def compute_match(recwav, targetwav, flow=1000.0):

    # make psd
    flen = len(targetwav.sample_frequencies)
    psd = aLIGOZeroDetHighPower(flen, targetwav.delta_f,
            low_freq_cutoff=flow)

    # --- Including time/phase maximisation:
    match, max_loc = pycbc.filter.match(recwav, targetwav, psd=psd,
            low_frequency_cutoff=flow)

    return match, max_loc, psd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA
#

waveform_name = sys.argv[1]
npcs=int(sys.argv[2])

#
# Create PMNS PCA instance for this catalogue
#

nTsamples=16384

all_waveform_names=['apr_135135_lessvisc',
                'shen_135135_lessvisc',
                'dd2_135135_lessvisc' ,
                'dd2_165165_lessvisc' ,
                'nl3_135135_lessvisc' ,
                'nl3_1919_lessvisc'   ,
                'tma_135135_lessvisc' ,
                'sfhx_135135_lessvisc',
                'sfho_135135_lessvisc']#,

                #'tm1_135135_lessvisc' ,

# Remove the test waveform from the catalogue
#waveform_names = \
#        [ name for name in all_waveform_names if not name==waveform_name ]
waveform_names = all_waveform_names

pmpca = ppca.pmnsPCA(waveform_names, low_frequency_cutoff=1000, fcenter=2710,
        nTsamples=nTsamples)


print "Building %s"%waveform_name

#
# Create test waveform
#
waveform = pu.Waveform(waveform_name+"_lessvisc", distance=50)
waveform.reproject_waveform()
waveform.compute_characteristics()

# Get this waveform's amplitude and SNR
targetAmp = waveform.hplus.max()
targetH = waveform.hplus.to_frequencyseries()

# Standardise
waveform_FD, target_fpeak, _ = \
        ppca.condition_spectrum(waveform.hplus.data,
                nsamples=nTsamples)

# Take FFT:
waveform_FD = ppca.unit_hrss(waveform_FD.data,
        delta=waveform_FD.delta_f, domain='frequency')

#
# Reconstruct as a function of nPCs, with different frequencies
#

# --- F-domain reconstruction
print "Taking projection"
fd_reconstruction = pmpca.reconstruct_freqseries(waveform_FD.data,
        npcs=npcs, this_fpeak=target_fpeak)['recon_spectrum']

# Invert the reconstruction to time-domain (also do this to the target since
# we've padded it)
td_reconstruction = fd_reconstruction.to_timeseries()
td_target = waveform_FD.to_timeseries()


#
# Rescale amplitudes to original values
#
td_target.data *= targetAmp / td_target.max()
td_reconstruction.data *= targetAmp / td_reconstruction.max()

fd_target = td_target.to_frequencyseries()
fd_reconstruction = td_reconstruction.to_frequencyseries()

#
# Get match and alignment for plotting
#
psd = pu.make_noise_curve(f_low=10, flen=len(fd_target),
        delta_f=fd_target.delta_f, noise_curve='aLIGO')
this_snr = pycbc.filter.sigma(fd_target, psd=psd, low_frequency_cutoff=1000)

match, snr_max_loc = pycbc.filter.match(td_target, td_reconstruction, psd=psd,
        low_frequency_cutoff=1000.0)

print this_snr, match

#td_reconstruction = pycbc.filter.highpass(td_reconstruction, frequency=1000,
#        filter_order=12)
#td_target = pycbc.filter.highpass(td_target, frequency=1000,
#        filter_order=12)

#
# Plotting
#

target_max_loc = td_target.max_loc()[1]
target_times = td_target.sample_times - \
        td_target.sample_times[target_max_loc]

rec_max_loc = np.argmax(td_reconstruction)
rec_times = np.copy(target_times)

# Get offset

# (Need the extra 1 sample offset for 1PC)
#offset = target_max_loc - snr_max_loc
offset = target_max_loc - snr_max_loc #- 1
rec_times += rec_times[offset] 


f, ax = pl.subplots(figsize=(8,6), nrows=2)
ax[0].plot(target_times, td_target, color='r', linestyle='--', linewidth=2, \
        label='Target (TM1 1.35+1.35)')
ax[0].plot(rec_times, td_reconstruction, color='k', linestyle='-', linewidth=1,
        label='Reconstruction ${M}$=%.2f'%match)
#ax[0].plot(rec_times, -1*td_reconstruction, color='k', linestyle='-', linewidth=1,
#        label='Reconstruction ${M}$=%.2f'%match)

ax[0].set_xlim(-2.5e-3,1.5e-2)
ax[0].set_ylim(-5e-22,5e-22)
ax[0].minorticks_on()
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('h$_+$(t) @ 50 Mpc')
ax[0].legend()

ax[1].semilogy(fd_target.sample_frequencies,
        2*abs(fd_target)*np.sqrt(fd_target.sample_frequencies), color='r',
        linewidth=2, linestyle='--')
ax[1].semilogy(fd_reconstruction.sample_frequencies,
        2*abs(fd_reconstruction)*np.sqrt(fd_reconstruction.sample_frequencies),
        color='k', linewidth=1, linestyle='-')

ax[1].semilogy(psd.sample_frequencies, np.sqrt(psd), label='aLIGO', color='grey')

ax[1].set_xlim(999, 4096)
ax[1].set_ylim(1e-24,1e-22)
ax[1].minorticks_on()
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('2|H$_+$($f$)|$\sqrt{f}$ & $\sqrt{S(f)}$')
ax[1].legend()

f.tight_layout()

pl.show()





