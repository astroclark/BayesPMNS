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
"""

from __future__ import division
import os,sys

import numpy as np
from matplotlib import pyplot as pl
import pmns_pca_utils
import pmns_utils

import lal
import pycbc.types
import pycbc.filter


waveform_names=['apr_135135_lessvisc',
                'dd2_135135_lessvisc' ,
                'dd2_165165_lessvisc' ,
                'nl3_1919_lessvisc' ,
                'nl3_135135_lessvisc' ,
                'tm1_135135_lessvisc' ,
                'tma_135135_lessvisc' ,
                'sfhx_135135_lessvisc',
                'sfho_135135_lessvisc']

npcs = len(waveform_names)

#
# Build Catalogues
#
fshift_center=1000
print "building catalogues"
(freqaxis, low_cat, high_cat, shift_cat, original_cat, fpeaks, low_sigmas,
        high_sigmas) = pmns_pca_utils.build_catalogues(waveform_names,
                fshift_center)
delta_f = np.diff(freqaxis)[0]

# Convert to magnitude/phase
shift_mag, shift_phase = pmns_pca_utils.complex_to_polar(shift_cat)

#
# PCA
#
print "Performing PCA"
shift_pca = pmns_pca_utils.pca_magphase(shift_cat, freqaxis, flow=10)


#
# Load test waveform
#
test_waveform = pmns_utils.Waveform('shen_135135_lessvisc')
test_waveform.reproject_waveform()
test_waveform.hplus.data /= pycbc.filter.sigma(test_waveform.hplus,
        low_frequency_cutoff=0.0)

test_hplus_data = np.zeros(16384)
test_hplus_data[:len(test_waveform.hplus)] = np.copy(test_waveform.hplus.data)
test_hplus = pycbc.types.TimeSeries(test_hplus_data, delta_t=1./16384)
test_hplus.data /= pycbc.filter.sigma(test_hplus,
        low_frequency_cutoff=10.0)

test_Hplus = test_hplus.to_frequencyseries()

test_mag, test_phase = pmns_pca_utils.complex_to_polar(test_Hplus.data)
