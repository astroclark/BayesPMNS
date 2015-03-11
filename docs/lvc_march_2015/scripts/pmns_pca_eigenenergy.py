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

f, ax = pl.subplots(ncols=1,figsize=(7,5))
ax.plot(range(1,npcs+1), 100*full_pca['magnitude_eigenergy'], 
     label='original magnitude spectrum')
ax.plot(range(1,npcs+1), 100*shift_pca['magnitude_eigenergy'], 
     label='shifted magnitude spectrum')
ax.plot(range(1,npcs+1), 100*full_pca['phase_eigenergy'],
     label='original phase spectrum', color='b', linestyle='--')
ax.plot(range(1,npcs+1), 100*shift_pca['phase_eigenergy'],
        label='shifted phase spectrum', color='g', linestyle='--')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('% Variance Explained')
ax.minorticks_on()
ax.grid()
ax.legend(loc='lower right')


#ax[1].plot(range(1,npcs+1), 100*full_pca['phase_eigenergy'],
#        label='full phase spectrum')
#ax[1].plot(range(1,npcs+1), 100*shift_pca['phase_eigenergy'],
#        label='shifted phase spectrum')
#ax[1].set_xlabel('Number of Principal Components (phase)')
#ax[1].set_ylabel('% Variance Explained')
#ax[1].minorticks_on()
#ax[1].grid()
#ax[1].legend(loc='lower right')


f.tight_layout()
pl.show()
for fileformat in imageformats:
    f.savefig('eigenergy.%s'%fileformat)

