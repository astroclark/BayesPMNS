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

import pmns_utils as pu
import pmns_pca_utils as ppca


#################################################
# MAIN


#
# Full waveform catalogue
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
catlen = len(waveform_names)

#
# Create PMNS PCA instance for this catalogue
#
pmpca = ppca.pmnsPCA(waveform_names)


#
# Create test waveform
#
testwav_name = 'shen_135135_lessvisc'
testwav_waveform = pu.Waveform(testwav_name)
testwav_waveform.reproject_waveform()

# Standardise
testwav_waveform_FD, fpeak = ppca.condition_spectrum(testwav_waveform.hplus.data)

# Normalise
testwav_waveform_FD = ppca.unit_hrss(testwav_waveform_FD.data,
        delta=testwav_waveform_FD.delta_f, domain='frequency')


#
# Reconstruct 
#
matches=np.zeros(catlen)
matches_align=np.zeros(catlen)

for n, npcs in enumerate(xrange(1,catlen+1)):
    reconstruction = pmpca.reconstruct(testwav_waveform_FD.data, npcs=npcs)
    matches_align[n]=reconstruction['match_aligo_align'][0]
    matches[n]=reconstruction['match_aligo'][0]

print matches
sys.exit()

#sys.exit()
#
# Plotting
#

#
# Matches
#
f, ax = pl.subplots(figsize=(7,5))
ax.plot(range(1,catlen+1), matches_align, label='Aligned spectra')
ax.plot(range(1,catlen+1), matches, label='Unaligned spectra')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Match')
ax.minorticks_on()
ax.grid()
ax.legend(loc='lower right')
f.tight_layout()
pl.show()

sys.exit()
#
# Eigenenergy
#
catlen=len(waveform_names)

f, ax = pl.subplots(ncols=1,figsize=(7,5))

ax.plot(range(1,catlen+1), 100*pmpca.pca['magnitude_eigenergy'], 
     label='magnitudes')

ax.plot(range(1,catlen+1), 100*pmpca.pca['phase_eigenergy'],
     label='phases', color='b', linestyle='--')

ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('% Variance Explained')
ax.minorticks_on()
ax.grid()
ax.legend(loc='lower right')
f.tight_layout()
pl.show()

sys.exit()
