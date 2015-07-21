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
pmns_pca_recexample.py

Make a pair of reconstruction examples
"""

from __future__ import division
import os,sys
import numpy as np
import cPickle as pickle

from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pycbc.filter

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_pca as ppca

nTsamples=16384
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Results
#
pickle_file = sys.argv[1]

waveform_data, pmpca, magnitude_euclidean, phase_euclidean, matches, \
        delta_fpeak, delta_R16 = pickle.load(open(pickle_file, "rb"))

#
# Get indices and labels of example waveform
#
eos_example = 'tm1'
mass_example = '135135'
viscosity = 'lessvisc'

label = eos_example.upper()

if "LOO" in pickle_file:
    npcs=1
else:
    npcs=waveform_data.nwaves

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test Waveform
#

print "Building test waveform"

#
# Create test waveform
#
waveform = pwave.Waveform(eos=eos_example, mass=mass_example,
        viscosity=viscosity, distance=50)
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PC reconstruction
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

psd = pwave.make_noise_curve(f_low=10, flen=len(fd_reconstruction),
        delta_f=fd_reconstruction.delta_f, noise_curve='aLIGO')

target_snr = pycbc.filter.sigma(td_target, psd=psd, low_frequency_cutoff=1000)
rec_snr = pycbc.filter.sigma(fd_reconstruction, psd=psd, low_frequency_cutoff=1000)

#td_reconstruction.data *= targetAmp / td_reconstruction.max()
td_reconstruction.data *= target_snr / rec_snr

fd_target = td_target.to_frequencyseries()
fd_reconstruction = td_reconstruction.to_frequencyseries()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Match Calculation
#

#
# Get match and alignment for plotting
#

this_snr = pycbc.filter.sigma(fd_target, psd=psd, low_frequency_cutoff=1000)

match, snr_max_loc = pycbc.filter.match(td_target, td_reconstruction, psd=psd,
        low_frequency_cutoff=1000.0)

print this_snr, match


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plots
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
#ax[0].plot(rec_times, td_reconstruction, color='k', linestyle='-', linewidth=1,
#        label='Reconstruction $\mathcal{M}$=%.2f'%match)
ax[0].plot(rec_times, -1*td_reconstruction, color='k', linestyle='-', linewidth=1,
        label='Reconstruction $\mathcal{M}$=%.2f'%match)

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


