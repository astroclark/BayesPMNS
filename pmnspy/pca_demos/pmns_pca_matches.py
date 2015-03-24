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

#
# Create PMNS PCA instance for this catalogue
#
pmpca = ppca.pmnsPCA(waveform_names)

#
# Create test waveform
#
noncat_name = 'shen_135135_lessvisc'
noncat_waveform = pu.Waveform(noncat_name)
noncat_waveform.reproject_waveform()

# Standardise
noncat_waveform_FD, fpeak = ppca.condition_spectrum(noncat_waveform.hplus.data)

# Normalise
noncat_waveform_FD = ppca.unit_hrss(noncat_waveform_FD.data,
        delta=noncat_waveform_FD.delta_f, domain='frequency')

#
# Reconstruct 
#
for npcs in xrange(1,11):
    reconstruction = pmpca.reconstruct(noncat_waveform_FD.data, npcs=npcs)
    print reconstruction['match_aligo']
    print reconstruction['match_noweight_align']
    del reconstruction

sys.exit()
#
# Plotting
#

#sys.exit()

f, ax = pl.subplots()

ax.plot(reconstruction['sample_frequencies'],
        abs(reconstruction['original_spectrum_align']), color='r',
        linewidth=2)

ax.plot(reconstruction['sample_frequencies'],
        abs(reconstruction['recon_spectrum_align']), color='k')

ax.set_xlim(0,1500)

pl.show()
