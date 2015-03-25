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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA
#
# Produces: 1) Explained variance calculation
#           2) Matches for waveforms reconstructed from their own PCs

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

#waveform_names=['apr_135135_lessvisc',
#                'shen_135135_lessvisc',
#                'shen_1215',
#                'dd2_135135_lessvisc' ,
#                'dd2_165165_lessvisc' ,
#                'nl3_1919_lessvisc' ,
#                'nl3_135135_lessvisc' ,
#                'tm1_135135_lessvisc' ,
#                'tma_135135_lessvisc' ,
#                'sfhx_135135_lessvisc',
#                'sfho_135135_lessvisc',
#                'tm1_1215',
#                'gs1_135135',
#                'gs2_135135',
#                'sly4_135135',
#                'ls220_135135',
#                'ls375_135135']


catlen = len(waveform_names)

#
# Create PMNS PCA instance for this catalogue
#

pmpca = ppca.pmnsPCA(waveform_names, low_frequency_cutoff=0)

#
# Exact matches (include test waveform in training data)
#
exact_matches=np.zeros(shape=(catlen, catlen))
exact_matches_align=np.zeros(shape=(catlen, catlen))

exact_dotmatch_magnitude=np.zeros(shape=(catlen, catlen))
exact_dotmatch_phase=np.zeros(shape=(catlen, catlen))

for w,testwav_name in enumerate(waveform_names):

    print "Analysing %s (exact match)"%testwav_name

    #
    # Create test waveform
    #
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
    for n, npcs in enumerate(xrange(1,catlen+1)):
        reconstruction = pmpca.reconstruct(testwav_waveform_FD.data, npcs=npcs)

        exact_matches_align[w,n]=reconstruction['match_noweight_align']
        exact_matches[w,n]=reconstruction['match_noweight']

        exact_dotmatch_magnitude[w,n]=reconstruction['dotmatch_magnitude']
        exact_dotmatch_phase[w,n]=reconstruction['dotmatch_phase']


# ***** Plot Results ***** #

#
# Exact Matches
#
#f, ax = ppca.image_matches(exact_matches, waveform_names, mismatch=True,
#        title="Reconstructing including the test waveform")

f, ax = ppca.image_matches(exact_dotmatch_magnitude, waveform_names,
        mismatch=False, title="Magnitudes")

f, ax = ppca.image_matches(exact_dotmatch_phase, waveform_names,
        mismatch=False, title="Phases")

f, ax = ppca.image_matches(exact_matches_align, waveform_names, mismatch=False,
        title="Reconstructing including the test waveform")
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Realistic catalogues
#
# Produces: 1) matches for waveforms reconstructed from the OTHER waveforms' PCs 

print "Setting up realistic match analysis"

#
# Exact matches (include test waveform in training data)
#

real_matches=np.zeros(shape=(catlen, catlen-1))
real_matches_align=np.zeros(shape=(catlen, catlen-1))

for w,testwav_name in enumerate(waveform_names):
    print "Analysing %s (real match)"%testwav_name

    # Remove testwave_name from the catalogue:
    waveform_names_reduced = \
            [ name for name in waveform_names if not name==testwav_name ]
    
    #
    # Create PMNS PCA instance for this catalogue
    #
    pmpca = ppca.pmnsPCA(waveform_names_reduced)

    #
    # Create test waveform
    #
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
    for n, npcs in enumerate(xrange(1,catlen-1+1)):
        reconstruction = pmpca.reconstruct(testwav_waveform_FD.data, npcs=npcs)

        real_matches_align[w,n]=reconstruction['match_aligo_align'][0]
        real_matches[w,n]=reconstruction['match_aligo'][0]


# ***** Plot Results ***** #

#
# Realistic Matches
#
f, ax = ppca.image_matches(real_matches, waveform_names, mismatch=True,
        title="Reconstructing excluding the test waveform")
f, ax = ppca.image_matches(real_matches, waveform_names, mismatch=False,
        title="Reconstructing excluding the test waveform")



pl.show()

sys.exit()
