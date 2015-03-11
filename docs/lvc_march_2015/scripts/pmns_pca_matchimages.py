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


waveform_names=['apr_135135_lessvisc',
                'shen_135135_lessvisc',
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
full_mag, full_phase = pmns_pca_utils.complex_to_polar(original_cat)
shift_mag, shift_phase = pmns_pca_utils.complex_to_polar(shift_cat)

#
# PCA
#
print "Performing PCA"
shift_pca = pmns_pca_utils.pca_magphase(shift_cat, freqaxis, flow=10)
full_pca = pmns_pca_utils.pca_magphase(original_cat, freqaxis, flow=10)

#
# Compute idealised minimal matches
#
print "Computing all matches"
full_matches_ideal = pmns_pca_utils.idealised_matches(original_cat, full_pca, delta_f=delta_f, flow=1000)
shift_matches_ideal = pmns_pca_utils.idealised_matches(shift_cat, shift_pca,
        delta_f=delta_f, flow=10) # careful with this one's flow!

unshift_matches_ideal = pmns_pca_utils.unshifted_matches(original_cat, shift_pca, fpeaks,
        freqaxis, fcenter=fshift_center)


# ******** #
# Plotting #
# ******** #
imageformats=['png','eps','pdf']

#
# Plot Matches
#


# -----------
# Match Plots

# Original waveforms
fig, ax = pmns_pca_utils.image_matches(full_matches_ideal, waveform_names,
        title="Matches From PCA With Original Spectra")

for fileformat in imageformats:
    fig.savefig('fullspec_ideal_matches.%s'%fileformat)


# Shifted waveforms
fig, ax = pmns_pca_utils.image_matches(shift_matches_ideal, waveform_names,
        title="Matches From PCA With Shifted Spectra")

for fileformat in imageformats:
    fig.savefig('shiftspec_ideal_matches.%s'%fileformat)


# UN-shifted waveforms
fig, ax = pmns_pca_utils.image_matches(unshift_matches_ideal, waveform_names,
        title="Matches From PCA With Un-Shifted Spectra")

for fileformat in imageformats:
    fig.savefig('unshiftspec_ideal_matches.%s'%fileformat)






