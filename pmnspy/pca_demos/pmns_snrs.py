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

import pmns_utils as pu

def compute_rate(sensemon_range, bns_rate_perL10perMyr=(0.6,60,600)):

    bns_rateperL10_perYr = np.array(bns_rate_perL10perMyr)/1e6

    cumlum_data = pu.CL_vs_PhysDist()
    if sensemon_range>cumlum_data[-2,0]:
        Ng = (4./3)*np.pi*(sensemon_range**3)*0.0116
        cumlum_in_range = Ng/1.7
    else:

        # Get cumulative blue-light luminosity in L10 data:
        cumlum_data = pu.CL_vs_PhysDist()

        # Interpolate to this range
        cumlum_in_range = np.interp(sensemon_range, cumlum_data[:,0],
                cumlum_data[:,1])

    rates = [cumlum_in_range * rate for rate in bns_rateperL10_perYr]

#   pl.figure()
#   pl.loglog(cumlum_data[:,0], cumlum_data[:,1])
#   pl.axvline(sensemon_range,color='r')
#   pl.show()
#   sys.exit()

    return rates

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loop over waveforms and PSDs, compute SNRs
#

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

#
# Create Waveform
#

horizon_snr=5
reference_distance=50
ndet=1

f = open('%s_snrs.dat'%sys.argv[1],'w')

f.writelines("# rho Dhor Dsens Rate\n")

rho_min=100
rho_max=0


for w, wave_name in enumerate(waveform_names):

    print "Analysing %s"%wave_name

    #
    # Create test waveform
    #
    waveform = pu.Waveform(wave_name, distance=reference_distance)
    waveform.reproject_waveform()
    Hplus = waveform.hplus.to_frequencyseries()

    #
    # Construct PSD
    #
    psd = pu.make_noise_curve(f_low=10, flen=len(Hplus), delta_f=Hplus.delta_f,
            noise_curve=sys.argv[1])
    #
    # Compute SNR
    #
    this_snr = pycbc.filter.sigma(Hplus, psd=psd, low_frequency_cutoff=1000)

    horizon_distance = np.sqrt(ndet)*reference_distance * this_snr / horizon_snr
    sensemon_range   = horizon_distance / 2.26

    rates = compute_rate(sensemon_range)

    print "%.2f %.2f %.2f %.2f"%(this_snr, horizon_distance,
            sensemon_range, rates[1])
    f.writelines("%s %.2f %.2f %.2f %.2f\n"%(wave_name, this_snr, horizon_distance,
        sensemon_range, rates[1]))

    # Record min/max waveforms & foms
    if this_snr<rho_min:
        min_wave = wave_name
        min_foms = (this_snr, horizon_distance, rates[1])
        rho_min = this_snr
    if this_snr>rho_max:
        max_wave = wave_name
        max_foms = (this_snr, horizon_distance, rates[1])
        rho_max = this_snr

#   f, ax = pl.subplots()
#   ax.semilogy(psd.sample_frequencies, np.sqrt(psd), label=sys.argv[1])
#   ax.semilogy(Hplus.sample_frequencies, 2*np.sqrt(Hplus.sample_frequencies.data)
#       * abs(Hplus.data))
#   ax.set_xlim(800, 4096)
#   pl.show()


print '---------------'
print 'SUMMARY'
print min_foms
print max_foms


f.close()






