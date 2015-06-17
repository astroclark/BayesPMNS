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

import pycbc.types

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata
from pmns_utils import pmns_pca as ppca

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA

# XXX: Hardcoding
nTsamples=16384
low_frequency_cutoff=1000
fcenter=2710

print "Setting up exact match analysis"

#
# Create the list of dictionaries which comprises our catalogue
#
waveform_data = pdata.WaveData(viscosity='lessvisc', mass='135135')

#
# Create PMNS PCA instance for this catalogue
#
pmpca = ppca.pmnsPCA(waveform_data, low_frequency_cutoff=low_frequency_cutoff,
        fcenter=fcenter, nTsamples=nTsamples)

#
# Exact matches (include test waveform in training data)
#
exact_matches=np.zeros(shape=(waveform_data.nwaves, waveform_data.nwaves))

exact_magnitude_euclidean=np.zeros(shape=(waveform_data.nwaves, waveform_data.nwaves))
exact_phase_euclidean=np.zeros(shape=(waveform_data.nwaves, waveform_data.nwaves))


for w, wave in enumerate(waveform_data.waves):

    print "Matching %s, %s (%s) [%d of %d]"%(wave['eos'], wave['mass'],
            wave['viscosity'], w, waveform_data.nwaves)

    #
    # Create test waveform
    #
    testwav_waveform = pwave.Waveform(eos=wave['eos'], mass=wave['mass'],
            viscosity=wave['viscosity'])
    testwav_waveform.reproject_waveform()

    # Standardise
    testwav_waveform_FD, fpeak, _ = \
            ppca.condition_spectrum(testwav_waveform.hplus.data,
                    nsamples=nTsamples)

    # Normalise
    testwav_waveform_FD = ppca.unit_hrss(testwav_waveform_FD.data,
            delta=testwav_waveform_FD.delta_f, domain='frequency')

    # Time-frequency map
    testwav_waveform_TF = ppca.build_cwt(pycbc.types.TimeSeries(
        testwav_waveform.hplus.data, delta_t=testwav_waveform.hplus.delta_t))

    #
    # Reconstruct as a function of number of principal components
    #
    for n, npcs in enumerate(xrange(1,waveform_data.nwaves+1)):

        # --- F-domain reconstruction
        fd_reconstruction = pmpca.reconstruct_freqseries(testwav_waveform_FD.data,
                npcs=npcs, wfnum=w)

        # --- FOMs
        exact_matches[w,n]=fd_reconstruction['match_aligo']
        exact_magnitude_euclidean[w,n]=fd_reconstruction['magnitude_euclidean']
        exact_phase_euclidean[w,n]=fd_reconstruction['phase_euclidean']



# ***** Plot Results ***** #

#
# Exact Matches
#
f, ax = ppca.image_euclidean(exact_magnitude_euclidean, waveform_data,
        title="Magnitudes: Euclidean Distance")

f, ax = ppca.image_euclidean(exact_phase_euclidean, waveform_data,
        title="Phases: Euclidean Distance")

f, ax = ppca.image_matches(exact_matches, waveform_data, mismatch=False,
        title="Match: training data includes the test waveform")


# -- Bars of the 90% variance matches
#f, ax = ppca.bar_matches(exact_matches, waveform_names, npcs=1)

pl.show()
