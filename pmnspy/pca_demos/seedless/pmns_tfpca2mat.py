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
pmns_pcatomat.py

Script to do a PCA TF analysis for seedless clustering input
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
deltaF=1.0
noise_curve="aLIGO"
target_snr=5
#loo=True
loo=False

eos="all"
mass="135135"
viscosity="lessvisc"


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

#
# Create PMNS PCA instance for the full catalogue
#
pmpca = ppca.pmnsPCA(waveform_data, low_frequency_cutoff=low_frequency_cutoff,
        fcenter=fcenter, nTsamples=nTsamples)

sys.exit()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Dump to mat file
#

#
# Extract relevant data
#


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
