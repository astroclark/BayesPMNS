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
import pycbc.psd 

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata

# ________________ - Local Defs - ________________ #

def compute_rate(sensemon_range, bns_rate_perL10perMyr=(0.6,60,600)):

    bns_rateperL10_perYr = np.array(bns_rate_perL10perMyr)/1e6

    cumlum_data = pwave.CL_vs_PhysDist()
    if sensemon_range>cumlum_data[-2,0]:
        Ng = (4./3)*np.pi*(sensemon_range**3)*0.0116
        cumlum_in_range = Ng/1.7
    else:

        # Get cumulative blue-light luminosity in L10 data:
        cumlum_data = pwave.CL_vs_PhysDist()

        # Interpolate to this range
        cumlum_in_range = np.interp(sensemon_range, cumlum_data[:,0],
                cumlum_data[:,1])

    rates = [cumlum_in_range * rate for rate in bns_rateperL10_perYr]

    return rates



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialise
#

noise_curve=sys.argv[1]
horizon_snr=5
reference_distance=50
ndet=1

eos="all"
mass="all"
viscosity="lessvisc"

#
# Build filename
#
filename="snrs_%s_horizonsnr-%.2f_eos-%s_mass-%s_viscosity-%s.txt"%(
        noise_curve, horizon_snr, eos, mass, viscosity)


# XXX: should probably fix this at the module level..
if eos=="all": eos=None
if mass=="all": mass=None
if viscosity=="all": viscosity=None

#
# Create the list of dictionaries which comprises our catalogue
#
waveform_data = pdata.WaveData(eos=eos,viscosity=viscosity, mass=mass)

#
# Create Waveforms and compute SNRs
#

f = open(filename, "w")

f.writelines("# rho_full rho_post Dhor Dsens Rate\n")

rho_min=100
rho_max=0


for w, wave in enumerate(waveform_data.waves):

    print "SNR for %s, %s ,%s (%d of %d)"%(
            wave['eos'], wave['mass'], wave['viscosity'], w+1,
            waveform_data.nwaves)

    #
    # Create test waveform
    #
    waveform = pwave.Waveform(eos=wave['eos'], mass=wave['mass'],
            viscosity=wave['viscosity'], distance=reference_distance)
    waveform.reproject_waveform()

    hplus_padded = pycbc.types.TimeSeries(np.zeros(16384),
            delta_t=waveform.hplus.delta_t)
    hplus_padded.data[:len(waveform.hplus)] = np.copy(waveform.hplus.data)
    Hplus = hplus_padded.to_frequencyseries()

    #
    # Construct PSD
    #
    psd = pwave.make_noise_curve(fmax=Hplus.sample_frequencies.max(),
            delta_f=Hplus.delta_f, noise_curve=noise_curve)

    #
    # Compute Full SNR
    #
    full_snr = pycbc.filter.sigma(Hplus, psd=psd, low_frequency_cutoff=1000)

    #
    # Compute post-merger only SNR Signal windowed at maximum
    #
    hplus_post = \
            pycbc.types.TimeSeries(pwave.window_inspiral(hplus_padded.data),
                    delta_t=hplus_padded.delta_t)
    Hplus_post = hplus_post.to_frequencyseries()
    
    post_snr = pycbc.filter.sigma(Hplus_post, psd=psd, low_frequency_cutoff=1000)

    horizon_distance = np.sqrt(ndet)*reference_distance * full_snr / horizon_snr
    sensemon_range   = horizon_distance / 2.26
    rates = compute_rate(sensemon_range)

    wave_name = "%s-%s-%s"%(wave['eos'], wave['mass'], wave['viscosity'])
    print "%.2f %.2f %.2f %.2f %.2f"%(full_snr, post_snr, horizon_distance,
            sensemon_range, rates[1])
    f.writelines("%s %.2f %.2f %.2f %.2f %.2f\n"%(wave_name, full_snr, post_snr,
        horizon_distance, sensemon_range, rates[1]))

    # Record min/max waveforms & foms
    if full_snr<rho_min:
        min_wave = wave_name
        min_foms = (wave_name, full_snr, horizon_distance, rates[1])
        rho_min = full_snr
    if full_snr>rho_max:
        max_wave = wave_name
        max_foms = (wave_name, full_snr, horizon_distance, rates[1])
        rho_max = full_snr



print '---------------'
print 'SUMMARY'
print min_foms
print max_foms


f.close()






