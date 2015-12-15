#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
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
pmns_pca_loomatches.py

Script to produce matches from PCA of post-merger waveforms.  This version
computes the matches when the test waveform is removed from the training data
(leave-one-out strategy)
"""

from __future__ import division
import os,sys
import cPickle as pickle
import numpy as np

from matplotlib import pyplot as pl

import pycbc.types
import pycbc.noise

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata
from pmns_utils import pmns_pca as ppca


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Generate Signal
#


#
# Create the list of dictionaries which comprises our catalogue
#
mass='135135'
eos='shen'

fmin=1000.0
#
# Create test waveform
#
waveform = pwave.Waveform(eos=eos, mass=mass, viscosity='lessvisc')
waveform.reproject_waveform()
waveform.compute_characteristics()

hplus = pycbc.types.TimeSeries(np.zeros(16384),
        delta_t=waveform.hplus.delta_t)
hplus.data[:len(waveform.hplus)] = np.copy(waveform.hplus.data)
#
#   hplus, _ = pycbc.waveform.get_sgburst_waveform(q=100,frequency=waveform.fpeak,
#           delta_t=1./16384, amplitude=1,hrss=1)
#   hplus.resize(16384)
#   hplus = pycbc.types.TimeSeries(hplus.data, delta_t=1./16384)

Hplus = hplus.to_frequencyseries()

#
# Construct PSD
#
psd = pwave.make_noise_curve(fmax=Hplus.sample_frequencies.max(),
        delta_f=Hplus.delta_f, noise_curve='aLIGO')

#
# Compute Full SNR
#
target_sigma = float(sys.argv[2])
full_snr = pycbc.filter.sigma(Hplus, psd=psd, low_frequency_cutoff=fmin)
Hplus.data *= target_sigma/full_snr

#noise=pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=101)
#noisy_data = noise + Hplus
#sys.exit()

#
# Get the fisher estimate
#
_, _, _, _, delta_fpeak, _= \
        pickle.load(open(sys.argv[1], "rb"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Generate PCA template
#
#
# Create PMNS PCA instance for the reduced catalogue
#
waveform_data = pdata.WaveData(viscosity='lessvisc', mass='135135')

# Remove testwave_name from the catalogue:
reduced_waveform_data = waveform_data.copy() # make a copy

# Grab the mass-eos waveform we want to study
for w,wave in enumerate(waveform_data.waves):
    if wave['mass']==mass and wave['eos']==eos:
        waveidx=w
delta_fpeak_fisher=delta_fpeak[0][waveidx]

reduced_waveform_data.remove_wave(waveform_data.waves[waveidx])

if 1:
    pmpca = ppca.pmnsPCA(reduced_waveform_data,
            low_frequency_cutoff=1000.0, fcenter=2710,
            nTsamples=16384)


    # Process signal for projection
    # Standardise
    waveform_FD, target_fpeak, _ = ppca.condition_spectrum(
            waveform.hplus.data, nsamples=16384)

    # Normalise
    waveform_FD = ppca.unit_hrss(waveform_FD.data,
            delta=waveform_FD.delta_f, domain='frequency')

# XXX: Begin Loop over noise realisations
fig, ax = pl.subplots(nrows=1)

#fpeaks = np.arange(fmin,4000, 5) 
fpeaks = np.arange(2000, 3000, 1)
nnoise = 5
sigma=np.zeros(shape=(nnoise,len(fpeaks)))

sh_inner=np.zeros(shape=(nnoise,len(fpeaks)))
hh_inner=np.zeros(shape=(nnoise,len(fpeaks)))
ss_inner=np.zeros(shape=(nnoise,len(fpeaks)))
for n in xrange(nnoise):

    print 'noise realisation %d of %d'%(n+1, nnoise)

    #
    # Generate Noise
    #
    noise=pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=n)
    noisy_data = noise + Hplus


    # XXX: BEGIN LOOP OVER FPEAKS
    for f,this_fpeak in enumerate(fpeaks):

        fd_reconstruction = pmpca.reconstruct_freqseries(waveform_FD.data,
                npcs=1, this_fpeak=this_fpeak)

#       td_tmplt, _ = \
#               pycbc.waveform.get_sgburst_waveform(q=100,frequency=this_fpeak,
#                       delta_t=1./16384, amplitude=1,hrss=1)
#       td_tmplt.resize(16384)
#       td_tmplt = pycbc.types.TimeSeries(td_tmplt.data, delta_t=1./16384)
#       fd_tmplt = td_tmplt.to_frequencyseries()

        fd_tmplt=fd_reconstruction['recon_spectrum'] 

        tmplt_snr=pycbc.filter.sigma(fd_tmplt, psd=psd,
                low_frequency_cutoff=fmin)
        fd_tmplt.data *= target_sigma/tmplt_snr


        sh_inner[n,f] = 2*np.real(pycbc.filter.overlap_cplx(fd_tmplt, noisy_data, psd=psd,
               low_frequency_cutoff=fmin, high_frequency_cutoff=4000., normalized=False))

        hh_inner[n,f] = 2*np.real(pycbc.filter.overlap_cplx(fd_tmplt, fd_tmplt, psd=psd,
               low_frequency_cutoff=fmin, high_frequency_cutoff=4000., normalized=False))

        ss_inner[n,f] = 2*np.real(pycbc.filter.overlap_cplx(noisy_data, noisy_data, psd=psd,
               low_frequency_cutoff=fmin, high_frequency_cutoff=4000., normalized=False))


        sigma[n,f]=sh_inner[n,f] - 0.5*ss_inner[n,f] - 0.5*hh_inner[n,f]

    ax.plot(fpeaks, np.exp((sigma[n,:]-max(sigma[n,:]))), color='grey')

ax.axvline(waveform.fpeak,color='r')
ax.axvline(waveform.fpeak-delta_fpeak_fisher, color='r', linestyle='--')
ax.axvline(waveform.fpeak+delta_fpeak_fisher, color='r', linestyle='--')

#ax.set_xlim(waveform.fpeak-1.1*delta_fpeak_fisher, waveform.fpeak+1.1*delta_fpeak_fisher)
fig.tight_layout()


pl.show()

























