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
import scipy.optimize

from matplotlib import pyplot as pl

import pycbc.types
import pycbc.noise

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata
from pmns_utils import pmns_pca as ppca

def neglogL(fpeak):#, signal_data=None):
    """
    Compute the phase-maximised negative log-likelihood
    """

    if (fpeak<1000.0) or (fpeak>4000.0):
        return np.inf

    fd_reconstruction = pmpca.reconstruct_freqseries(waveform_FD.data,
            npcs=NPCs, this_fpeak=fpeak)

    fd_tmplt=fd_reconstruction['recon_spectrum'] 

    tmplt_energy=pycbc.filter.sigmasq(fd_tmplt, low_frequency_cutoff=fmin)
    fd_tmplt.data *= np.sqrt(signal_energy/tmplt_energy)


    sigma = pycbc.filter.match(fd_tmplt, noisy_data,
            low_frequency_cutoff=fmin, psd=psd, v1_norm=1.0, v2_norm=1.0)[0]

    return -2*sigma



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input 

#fisher_filename=sys.argv[2]

mass='135135'
eos='dd2' 

NPCs=int(sys.argv[1]) #48
fmin=1000 

if sys.argv[2]=="LOO":
    LOO=True
else:LOO=False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Generate Signal
#

#
# Build the injected waveform
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
target_sigma = 5.0
full_snr = pycbc.filter.sigma(Hplus, psd=psd, low_frequency_cutoff=fmin)
Hplus.data *= target_sigma/full_snr

# Energy
signal_energy = pycbc.filter.sigmasq(Hplus, low_frequency_cutoff=fmin)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Generate PCA template
#
#
# Create PMNS PCA instance for the reduced catalogue
#
waveform_data = pdata.WaveData(viscosity='lessvisc')

# Remove testwave_name from the catalogue:
reduced_waveform_data = waveform_data.copy() # make a copy

# Grab the mass-eos waveform we want to study
for w,wave in enumerate(waveform_data.waves):
    if wave['mass']==mass and wave['eos']==eos:
        waveidx=w

#sigma_fpeak_fisher=sigma_fpeak[NPCs-1][waveidx]

if LOO:
    reduced_waveform_data.remove_wave(waveform_data.waves[waveidx])

pmpca = ppca.pmnsPCA(reduced_waveform_data,
        low_frequency_cutoff=fmin, fcenter=2710,
        nTsamples=16384)

# Process signal for projection
# Standardise
waveform_FD, target_fpeak, _ = ppca.condition_spectrum(
        waveform.hplus.data, nsamples=16384)

# Normalise
waveform_FD = ppca.unit_hrss(waveform_FD.data,
        delta=waveform_FD.delta_f, domain='frequency')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Check nominal reconstruction fidelity

fd_reconstruction_true = pmpca.reconstruct_freqseries(waveform_FD.data,
        npcs=NPCs, this_fpeak=waveform.fpeak)

rec_energy=pycbc.filter.sigmasq(fd_reconstruction_true['recon_spectrum'],
        low_frequency_cutoff=fmin)

fd_reconstruction_true['recon_spectrum'].data *= \
        np.sqrt(signal_energy/rec_energy)

overlap_true = pycbc.filter.overlap_cplx(Hplus,
        fd_reconstruction_true['recon_spectrum'], psd=psd,
        low_frequency_cutoff=fmin)

match_true = pycbc.filter.match(Hplus,
        fd_reconstruction_true['recon_spectrum'], psd=psd,
        low_frequency_cutoff=fmin)


nominal_reconstruction_snr = pycbc.filter.sigma(
        fd_reconstruction_true['recon_spectrum'], psd=psd,
        low_frequency_cutoff=fmin)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute The Fisher Estimate For fpeak Uncertainty

deltaF = 1
fpeak1 = target_fpeak+0.5*deltaF
fpeak2 = target_fpeak-0.5*deltaF

fd_reconstruction1 = pmpca.reconstruct_freqseries(waveform_FD.data,
        npcs=NPCs, this_fpeak=fpeak1)
reconstruction1 = fd_reconstruction1['recon_spectrum'] 
reconstruction1_sigma = pycbc.filter.sigma(reconstruction1, psd=psd,
        low_frequency_cutoff=fmin)
reconstruction1.data *= target_sigma/reconstruction1_sigma

fd_reconstruction2 = pmpca.reconstruct_freqseries(waveform_FD.data,
        npcs=NPCs, this_fpeak=fpeak2)
reconstruction2 = fd_reconstruction2['recon_spectrum'] 
reconstruction2_sigma = pycbc.filter.sigma(reconstruction2, psd=psd,
        low_frequency_cutoff=fmin)
reconstruction2.data *= target_sigma/reconstruction2_sigma

diff = reconstruction2 - reconstruction1
inner_diff = pycbc.filter.match(diff, diff, psd=psd, low_frequency_cutoff=fmin,
        v1_norm=1.0, v2_norm=1.0)[0]
#inner_diff = pycbc.filter.overlap(diff, diff, psd=psd, low_frequency_cutoff=fmin,
#        normalized=False)

sigma_fpeak = abs(fpeak2-fpeak1) / np.sqrt(inner_diff)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOISE REALISATIONS

nnoise = 500
sigma=np.zeros(shape=nnoise)
fpeak_maxL=np.zeros(shape=nnoise)

for n in xrange(nnoise):

    print 'noise realisation %d of %d'%(n+1, nnoise)

    #
    # Generate Noise
    #

    noise=pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=n)
    noisy_data = noise + Hplus


    # Optimize match over fpeak
    result = scipy.optimize.fmin(neglogL, x0=waveform.fpeak,
            full_output=True, retall=True, disp=True)


    fpeak_maxL[n] = result[0]
    sigma[n] = result[1]
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save results
outfile = 'maxLfpeak'+'_'+eos+'_'+mass+'_'+'SNR-'+str(int(target_sigma))
np.savez(outfile, fpeak_maxL=fpeak_maxL, sigma=sigma, SNR=target_sigma, eos=eos,
        mass=mass)

sys.exit()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show distribution of max-likelihood fpeaks

f, ax = pl.subplots()
#bins = np.arange(1000, 4000, 25)
bins = 25
ax.hist(fpeak_maxL, bins=bins, normed=True, histtype='stepfilled', alpha=0.5)
ax.set_xlabel('Max-likelihood f$_{\mathrm{peak}}$')
ax.set_ylabel('Normalised count')

ax.axvline(waveform.fpeak, label='Target Value', color='r')
ax.axvline(waveform.fpeak+sigma_fpeak, label='Fisher', color='r', linestyle='--')
ax.axvline(waveform.fpeak-sigma_fpeak, color='r', linestyle='--')

ax.axvline(np.mean(fpeak_maxL), color='g', label='mean')
ax.axvline(np.mean(fpeak_maxL)+np.std(fpeak_maxL), color='g', label='1$\sigma$',
        linestyle='--')
ax.axvline(np.mean(fpeak_maxL)-np.std(fpeak_maxL), color='g',
        linestyle='--')

ax.legend(loc='upper left')
ax.minorticks_on()

ax.set_xlim(1000, 4000)


pl.show()

sys.exit()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot reconstructions


f, ax = pl.subplots(nrows=3, sharex=True)


ax[0].plot(Hplus.sample_frequencies, np.real(Hplus), label='Injected')
ax[0].plot(fd_reconstruction_true['sample_frequencies'],
        np.real(fd_reconstruction_true['recon_spectrum']), label='fpeak rec')

ax[1].plot(Hplus.sample_frequencies, np.imag(Hplus), label='Injected')
ax[1].plot(fd_reconstruction_true['sample_frequencies'],
        np.imag(fd_reconstruction_true['recon_spectrum']), label='fpeak rec')

ax[2].plot(Hplus.sample_frequencies, abs(Hplus), label='Injected')
ax[2].plot(fd_reconstruction_true['sample_frequencies'],
        abs(fd_reconstruction_true['recon_spectrum']), label='fpeak rec')



#f, ax = pl.subplots(nrows=3, sharex=True)

for n in xrange(nnoise):

    fpeak_maxL[n]=fpeaks[np.argmax(sigma[n,:])]

    fd_reconstruction_maxL = pmpca.reconstruct_freqseries(waveform_FD.data,
            npcs=NPCs, this_fpeak=fpeak_maxL[n])

    tmplt_energy=pycbc.filter.sigmasq(fd_reconstruction_maxL['recon_spectrum'], low_frequency_cutoff=fmin)
    fd_reconstruction_maxL['recon_spectrum'].data *= np.sqrt(signal_energy/tmplt_energy)

    # XXX
    #fd_reconstruction_maxL['recon_spectrum'].data *= np.exp(1j*np.pi)

    overlap_maxL[n] = pycbc.filter.overlap_cplx(fd_reconstruction_maxL['original_spectrum'],
            fd_reconstruction_maxL['recon_spectrum'], psd=psd,
            low_frequency_cutoff=fmin)


    match_maxL[n] = pycbc.filter.match(fd_reconstruction_maxL['original_spectrum'],
            fd_reconstruction_maxL['recon_spectrum'], psd=psd,
            low_frequency_cutoff=fmin)[0]



    ax[0].plot(Hplus.sample_frequencies,
            np.real(fd_reconstruction_maxL['recon_spectrum']),
            label='fpeak maxL')

    ax[1].plot(Hplus.sample_frequencies,
            np.imag(fd_reconstruction_maxL['recon_spectrum']), 
            label='fpeak maxL')

    ax[2].plot(Hplus.sample_frequencies,
            abs(fd_reconstruction_maxL['recon_spectrum']),
            label='fpeak_maxL')

ax[0].set_ylabel('real [H(f)]')
ax[0].axvline(waveform.fpeak,color='r')
ax[0].set_xlim(1000, 4000)
ax[0].legend(loc='lower right')
ax[1].set_ylabel('imag [H(f)]')
ax[1].axvline(waveform.fpeak,color='r')
ax[1].set_xlim(1000, 4000)
ax[1].legend(loc='lower right')
ax[2].set_ylabel('abs [H(f)]')
ax[2].axvline(waveform.fpeak,color='r')
ax[2].set_xlim(1000, 4000)
ax[2].legend(loc='lower right')



pl.show()

