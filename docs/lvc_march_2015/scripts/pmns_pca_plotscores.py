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



# ******** #
# Plotting #
# ******** #
imageformats=['png','eps','pdf']

#
# Plot Catalogues
#

# Magnitude
f, ax = pl.subplots(nrows=2,figsize=(7,10))
ax[0].plot(freqaxis, full_pca['magnitude_scores'], label='full spectrum')
ax[0].set_xlim(900, 5000)
#ax[0].set_ylim(1e-4, 1e-1)
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel('|H(f)|')
ax[0].minorticks_on()
ax[0].set_title('PC Scores: Original Spectra (magnitude)')
ax[0].grid()

ax[1].plot(freqaxis, shift_pca['magnitude_scores'], label='shifted spectrum')
ax[1].axvline(fshift_center, color='r', linewidth=2)
ax[1].set_xlim(300, fshift_center+1000)
#ax[1].set_ylim(1e-4, 1e-1)
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('|H(f)|')
ax[1].minorticks_on()
ax[1].set_title('PC Scores: Shifted Spectra (magnitude)')
ax[1].grid()

f.tight_layout()
pl.show()
for fileformat in imageformats:
    f.savefig('pcs_magnitude_overlay.%s'%fileformat)

# Phase
f, ax = pl.subplots(nrows=2,figsize=(7,10))
ax[0].plot(freqaxis, full_pca['phase_scores'], label='full spectrum')
ax[0].set_xlim(900, 5000)
#ax[0].set_ylim(1e-4, 1e-1)
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel('arg[H(f)]')
ax[0].minorticks_on()
ax[0].set_title('PC Scores: Original Spectra (phase)')
ax[0].grid()

ax[1].plot(freqaxis, shift_pca['phase_scores'], label='shifted spectrum')
ax[1].axvline(fshift_center, color='r', linewidth=2)
ax[1].set_xlim(300, fshift_center+1000)
#ax[1].set_ylim(1e-4, 1e-1)
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('arg[H(f)]')
ax[1].minorticks_on()
ax[1].set_title('PC Scores: Shifted Spectra (phase)')
ax[1].grid()

f.tight_layout()
pl.show()
for fileformat in imageformats:
    f.savefig('pcs_phase_overlay.%s'%fileformat)


