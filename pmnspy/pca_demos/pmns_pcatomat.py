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

import numpy as np
import pmns_utils as pu
import pmns_pca_utils as ppca
import scipy.io as sio


#
# Create PMNS PCA instance for this catalogue
#

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

nTsamples=16384

pmpca = ppca.pmnsPCA(waveform_names, low_frequency_cutoff=1000, fcenter=2710,
        nTsamples=nTsamples)

# --- Generate PC coefficients for storage
magnitude_betas=np.zeros(shape=(len(waveform_names),len(waveform_names)))
phase_betas=np.zeros(shape=(len(waveform_names),len(waveform_names)))
for w, testwav_name in enumerate(waveform_names):
    print "Projecting %s"%testwav_name

    testwav_waveform = pu.Waveform(testwav_name)
    testwav_waveform.reproject_waveform()

    # Standardise
    testwav_waveform_FD, fpeak, _ = \
            ppca.condition_spectrum(testwav_waveform.hplus.data,
                    nsamples=nTsamples)

    # Normalise
    testwav_waveform_FD = ppca.unit_hrss(testwav_waveform_FD.data,
            delta=testwav_waveform_FD.delta_f, domain='frequency')

    projection = pmpca.project_freqseries(testwav_waveform_FD)

    magnitude_betas[w,:] = projection['magnitude_betas']
    phase_betas[w,:]     = projection['phase_betas']




outputdata = {'sample_frequencies':pmpca.sample_frequencies,
        'time_domain_waveforms':pmpca.cat_timedomain,
        'fcenter':2710.0,
        'fpeaks':pmpca.fpeaks,
        'frequency_domain_waveforms':pmpca.cat_orig,
        'freqency_domain_waveforms_aligned':pmpca.cat_align,
        'magnitude_spectrum_global_mean':pmpca.pca['magnitude_mean'],
        'magnitude_principal_components':pmpca.pca['magnitude_pca'].components_,
        'magnitude_coeffficients':magnitude_betas,
        'timefreq_global_mean':pmpca.pca['timefreq_mean'],
        'timefreq_principal_components':pmpca.pca['timefreq_pca'].components_}

sio.savemat('postmergerpca.mat', outputdata)
