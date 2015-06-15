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


def compute_inner(recwav, targetwav, target_snr, psd, flow=1000.0):

    # make psd
    # XXX: DANGER HARDCODING
#    flen = len(targetwav.sample_frequencies)
#    psd = aLIGOZeroDetHighPower(flen, targetwav.delta_f,
#            low_freq_cutoff=flow)


    # make sure amplitudes are scaled correctly
    recwav_snr = pycbc.filter.sigma(recwav, psd=psd,
            low_frequency_cutoff=flow)
    recwav *= target_snr/recwav_snr

    targetwav_snr = pycbc.filter.sigma(targetwav, psd=psd,
            low_frequency_cutoff=flow)

    targetwav_tmp = target_snr/targetwav_snr * targetwav

    diff = targetwav_tmp - recwav

    # --- Including time/phase maximisation:
    #rhodiff = pycbc.filter.sigma(diff, psd=psd,
    #        low_frequency_cutoff=flow)
    rhodiff = pycbc.filter.match(diff, diff, psd=psd, low_frequency_cutoff=flow,
            v1_norm=1.0, v2_norm=1.0)[0]
#
#    print rhodiff
#
#    sys.exit()

    return rhodiff

def propagate_deltaF(f, delta_f, a=1.099, b=-8.574, c=28.07):
    return np.sqrt( ((2*a*f+b)*delta_f)**2 )

def bar_deltaF(delta_fpeaks, waveform_names):
    labels=[name.replace('_lessvisc','') for name in waveform_names]

    ind = np.arange(len(waveform_names))
    width = 0.4 
    gap   = 0.05

    f, ax = pl.subplots(ncols=1)#,figsize=(6,8))
    f.subplots_adjust(bottom=0.1,top=.95,left=0.2)
    matchbar_exact = ax.barh(ind, delta_fpeaks, height=width, color='k',
            linewidth=0.5)

    ax.set_xlabel('$\delta$ f$_{\\rm peak}$ [Hz]')
    ax.set_ylabel('Waveform')
    ax.set_yticks(ind+width)
    ytickNames = pl.setp(ax, yticklabels=labels)
    #pl.setp(ytickNames, rotation=45)
    #ax.set_xlim(0.0,10)
    ax.set_ylim(-.5,len(waveform_names))
    ax.minorticks_on()
    #ax1.legend(loc='lower right')
    ax.grid(linestyle='-',color='grey')
    ax.xaxis.grid(True, linestyle=':', which='minor', color='grey')
    ax.set_axisbelow(True)
    f.tight_layout()

    return f, ax

def bar_deltaR(delta_radii, waveform_names):
    labels=[name.replace('_lessvisc','') for name in waveform_names]

    ind = np.arange(len(waveform_names))
    width = 0.4 
    gap   = 0.05

    f, ax = pl.subplots(ncols=1)#,figsize=(6,8))
    f.subplots_adjust(bottom=0.1,top=.95,left=0.2)
    matchbar_exact = ax.barh(ind, delta_radii*1000, height=width, color='k',
            linewidth=0.5)

    ax.set_xlabel('$\delta$ R$_{\\rm 1.6}$ [m]')
    ax.set_ylabel('Waveform')
    ax.set_yticks(ind+width)
    ytickNames = pl.setp(ax, yticklabels=labels)
    #pl.setp(ytickNames, rotation=45)
    #ax.set_xlim(0.0,10)
    ax.set_ylim(-.5,len(waveform_names))
    ax.minorticks_on()
    #ax1.legend(loc='lower right')
    ax.grid(linestyle='-',color='grey')
    ax.xaxis.grid(True, linestyle=':', which='minor', color='grey')
    ax.set_axisbelow(True)
    f.tight_layout()

    return f, ax
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA
#
# Produces: 1) Explained variance calculation
#           2) Matches for waveforms reconstructed from their own PCs

print "Setting up exact match analysis"

waveform_names=['apr_135135_lessvisc',
                'shen_135135_lessvisc',
                'dd2_135135_lessvisc' ,
                'dd2_165165_lessvisc' ,
                'nl3_135135_lessvisc' ,
                'nl3_1919_lessvisc'   ,
                'tm1_135135_lessvisc' ,
                'tma_135135_lessvisc' ,
                'sfhx_135135_lessvisc',
                'sfho_135135_lessvisc']#,


catlen = len(waveform_names)
nTsamples=16384

#Distance=float(sys.argv[1])

noise_curve=sys.argv[1]
target_snr=float(sys.argv[2])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Realistic catalogues (leave-one-out)
#
# Produces: 1) matches for waveforms reconstructed from the OTHER waveforms' PCs 

print "Setting up leave-one-out match analysis"

#
# Frequency errors with realistic training data
#

delta_fpeak_real = np.zeros(shape=(len(waveform_names)))
delta_radius_real = np.zeros(shape=(len(waveform_names)))
for w,testwav_name in enumerate(waveform_names):
    print "Analysing %s (real match)"%testwav_name

    # Remove testwave_name from the catalogue:
    waveform_names_reduced = \
            [ name for name in waveform_names if not name==testwav_name ]
    
    #
    # Create PMNS PCA instance for this catalogue
    #
    pmpca = ppca.pmnsPCA(waveform_names_reduced, low_frequency_cutoff=1000,
            fcenter=2710, nTsamples=nTsamples)

    #
    # Create test waveform
    #
    testwav_waveform = pu.Waveform(testwav_name)
    testwav_waveform.reproject_waveform()
    testwav_waveform.compute_characteristics()
    Hplus = testwav_waveform.hplus.to_frequencyseries()


    # Standardise
    testwav_waveform_FD, target_fpeak, _ = \
            ppca.condition_spectrum(testwav_waveform.hplus.data,
                    nsamples=nTsamples)

    # Take FFT:
    testwav_waveform_FD = ppca.unit_hrss(testwav_waveform_FD.data,
            delta=testwav_waveform_FD.delta_f, domain='frequency')

    #
    # Construct PSD
    #
    psd = pu.make_noise_curve(f_low=10, flen=len(testwav_waveform_FD),
            delta_f=testwav_waveform_FD.delta_f, noise_curve=noise_curve)

    #
    # Reconstruct as a function of nPCs, with different frequencies
    #

    # --- F-domain reconstruction
    deltaFs = [1]#np.logspace(np.log10(0.001),0,10)
    deltaFpeak = np.zeros(len(deltaFs))
    for f,deltaF in enumerate(deltaFs):

        fpeak1 = target_fpeak+0.5*deltaF
        fpeak2 = target_fpeak-0.5*deltaF

        fd_reconstruction1 = pmpca.reconstruct_freqseries(testwav_waveform_FD.data,
                npcs=1, this_fpeak=fpeak1)
        reconstruction1 = fd_reconstruction1['recon_spectrum'] 

        fd_reconstruction2 = pmpca.reconstruct_freqseries(testwav_waveform_FD.data,
                npcs=1, this_fpeak=fpeak2)
        reconstruction2 = fd_reconstruction2['recon_spectrum'] 

        diffSNR = compute_inner(reconstruction1, reconstruction2,
                target_snr=target_snr, psd=psd)

        deltaFpeak[f] = abs(fpeak2-fpeak1) / diffSNR

    delta_fpeak_real[w] = np.mean(deltaFpeak)
    delta_radius_real[w] = propagate_deltaF(target_fpeak/1e3, delta_fpeak_real[f]/1e3)


# -------
# FIGURES
f, ax = bar_deltaF(delta_fpeak_real, waveform_names)
ax.set_title("Frequency Error: training data excludes the test waveform")
f.tight_layout()

f, ax = bar_deltaR(delta_radius_real, waveform_names)
ax.set_title("Radius Error: training data excludes the test waveform")
f.tight_layout()

print "-------------------"
print "Min/Max delta fpeak: %.3f/%.3f"%(min(delta_fpeak_real),
        max(delta_fpeak_real))

print "Min/Max delta_R1.6: %.3f/%.3f"%(1000*min(delta_radius_real),
        1000*max(delta_radius_real))

pl.show()














