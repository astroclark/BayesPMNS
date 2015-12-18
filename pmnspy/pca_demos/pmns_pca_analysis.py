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

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata
from pmns_utils import pmns_pca as ppca

# ________________ - Local Defs - ________________ #

def compute_inner(recwav, targetwav, target_snr, psd, flow=1000.0):

    # make sure amplitudes are scaled correctly
    recwav_snr = pycbc.filter.sigma(recwav, psd=psd,
            low_frequency_cutoff=flow)
    recwav *= target_snr/recwav_snr

    targetwav_snr = pycbc.filter.sigma(targetwav, psd=psd,
            low_frequency_cutoff=flow)

    targetwav_tmp = target_snr/targetwav_snr * targetwav

    diff = targetwav_tmp - recwav

    # --- Including time/phase maximisation:
    rhodiff = pycbc.filter.match(diff, diff, psd=psd, low_frequency_cutoff=flow,
            v1_norm=1.0, v2_norm=1.0)[0]

    return rhodiff

def propagate_deltaF(f, delta_f, a=1.099, b=-8.574, c=28.07):
    return np.sqrt( ((2*a*f+b)*delta_f)**2 )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA

# XXX: Hardcoding
nTsamples=16384
low_frequency_cutoff=1500
fcenter=2710
deltaF=1
noise_curve="aLIGO"
target_snr=float(sys.argv[1])
loo=True
#loo=False

eos="all"
mass="135135"
viscosity="lessvisc"

#
# Build filename
#
if loo==True:
    picklename="matches_LOO_%s_targetsnr-%.2f_eos-%s_mass-%s_viscosity-%s.pickle"%(
            noise_curve, target_snr, eos, mass, viscosity)
else:
    picklename="matches_ALL_%s_targetsnr-%.2f_eos-%s_mass-%s_viscosity-%s.pickle"%(
            noise_curve, target_snr, eos, mass, viscosity)


# XXX: should probably fix this at the module level..
if eos=="all": eos=None
if mass=="all": mass=None
if viscosity=="all": viscosity=None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialise
#

#
# Create the list of dictionaries which comprises our catalogue
#
waveform_data = pdata.WaveData(eos=eos,viscosity=viscosity, mass=mass)
if loo==True:
    maxnpcs=waveform_data.nwaves-1
else:
    maxnpcs=waveform_data.nwaves

#
# Create PMNS PCA instance for the full catalogue
#
pmpca = ppca.pmnsPCA(waveform_data, low_frequency_cutoff=low_frequency_cutoff,
        fcenter=fcenter, nTsamples=nTsamples)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Match Calculations
#

print "Setting up match and Fisher analysis"

#
# Exact matches (include test waveform in training data)
#

matches=np.zeros(shape=(waveform_data.nwaves, maxnpcs))
magnitude_euclidean=np.zeros(shape=(waveform_data.nwaves, maxnpcs))
phase_euclidean=np.zeros(shape=(waveform_data.nwaves, maxnpcs))

# Fisher error estimates
delta_fpeak  = np.zeros(shape=(waveform_data.nwaves,maxnpcs))
delta_R16 = np.zeros(shape=(waveform_data.nwaves,maxnpcs))

for w, wave in enumerate(waveform_data.waves):

    print "Matching %s, %s ,%s (%d of %d)"%(
            wave['eos'], wave['mass'], wave['viscosity'], w+1,
            waveform_data.nwaves)

    if loo:
        print ""
        print "Building Reduced Catalogue"
        print ""
        # Remove testwave_name from the catalogue:
        reduced_waveform_data = waveform_data.copy() # make a copy
        reduced_waveform_data.remove_wave(wave)      # remove this waveform
        
        #
        # Create PMNS PCA instance for the reduced catalogue
        #
        pmpca = ppca.pmnsPCA(reduced_waveform_data,
                low_frequency_cutoff=low_frequency_cutoff, fcenter=fcenter,
                nTsamples=nTsamples)

    #
    # Create test waveform
    #
    waveform = pwave.Waveform(eos=wave['eos'], mass=wave['mass'],
            viscosity=wave['viscosity'])
    waveform.reproject_waveform()
    waveform.compute_characteristics()

    # Standardise
    waveform_FD, target_fpeak, _ = ppca.condition_spectrum(
            waveform.hplus.data, nsamples=nTsamples)

    # Normalise
    waveform_FD = ppca.unit_hrss(waveform_FD.data,
            delta=waveform_FD.delta_f, domain='frequency')
    
    #
    # Construct PSD
    #
    psd = pwave.make_noise_curve(fmax=waveform_FD.sample_frequencies.max(),
            delta_f=waveform_FD.delta_f, noise_curve=noise_curve)

    # Fisher matrix: expand likelihood about target_fpeak +/- 0.5*deltaF
    fpeak1 = target_fpeak+0.5*deltaF
    fpeak2 = target_fpeak-0.5*deltaF

    #
    # Compute results as functions of #s of PCs
    #
    for n, npcs in enumerate(xrange(1,waveform_data.nwaves)):

        fd_reconstruction = \
                pmpca.reconstruct_freqseries(waveform_FD.data,
                        npcs=npcs, wfnum=w)
        #
        # --- FOMs
        #

        # Reconstruction fidelity
        matches[w,n]=pycbc.filter.match(fd_reconstruction['recon_spectrum'],
                waveform_FD, psd=psd,
                low_frequency_cutoff=low_frequency_cutoff)[0]


        magnitude_euclidean[w,n]=fd_reconstruction['magnitude_euclidean']
        phase_euclidean[w,n]=fd_reconstruction['phase_euclidean']


        # Parameter estimation: Fisher frequency errors
        fd_reconstruction1 = pmpca.reconstruct_freqseries(waveform_FD.data,
                npcs=npcs, this_fpeak=fpeak1)
        reconstruction1 = fd_reconstruction1['recon_spectrum'] 

        fd_reconstruction2 = pmpca.reconstruct_freqseries(waveform_FD.data,
                npcs=npcs, this_fpeak=fpeak2)
        reconstruction2 = fd_reconstruction2['recon_spectrum'] 

        diffSNR = compute_inner(reconstruction1, reconstruction2,
                target_snr=target_snr, psd=psd)

        delta_fpeak[w,n] = abs(fpeak2-fpeak1) / diffSNR
        delta_R16[w,n] = propagate_deltaF(target_fpeak/1e3,
                delta_fpeak[w,n]/1e3)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save results for plotting seperately
#

pickle.dump([waveform_data, pmpca, magnitude_euclidean, phase_euclidean, matches,
    delta_fpeak, delta_R16], open(picklename, "wb"))



