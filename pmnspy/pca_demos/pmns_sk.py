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

print 'generate catalogue'

pmpca = ppca.pmnsPCA(waveform_names, low_frequency_cutoff=1000, fcenter=2710)


print 'extract data and perform pca'

mags = np.copy(pmpca.magnitude_align)
phis = np.copy(pmpca.phase)

phifits = np.copy(pmpca.pca['phase_fits'])

from sklearn.decomposition import PCA 
from sklearn.decomposition import SparsePCA 

mag_pca = PCA(whiten=False)#, n_components=8193)
mag_pca.fit(mags)

phase_pca = PCA(whiten=False)
phase_pca.fit(phis)

cat_align = np.copy(pmpca.cat_align)
cplx_pca = PCA()
cplx_pca.fit(np.real(cat_align))


print 'generate test waveform'
 
testwav_waveform = pu.Waveform(waveform_names[1])
testwav_waveform.reproject_waveform()

# Standardise
testwav_waveform_FD, fpeak = ppca.condition_spectrum(testwav_waveform.hplus.data)

# Normalise
testwav_waveform_FD = ppca.unit_hrss(testwav_waveform_FD.data,
        delta=testwav_waveform_FD.delta_f, domain='frequency')

testmag = abs(testwav_waveform_FD)
testphi = ppca.phase_of(testwav_waveform_FD)

testmag_align = ppca.shift_vec(testmag, testwav_waveform_FD.sample_frequencies,
        fpeak=fpeak, fcenter=2710).real

testspec_align = ppca.shift_vec(np.real(testwav_waveform_FD),
        testwav_waveform_FD.sample_frequencies, fpeak=fpeak, fcenter=2710).real

print 'reconstruct test wav'

#testbetas = np.concatenate(cplx_pca.transform(testspec_align))
testbetas = np.concatenate(mag_pca.transform(testmag_align))

rec = np.zeros(np.shape(mags)[1])
for b, beta in enumerate(testbetas):
    #rec += beta * cplx_pca.components_[b,:]
    rec += beta * mag_pca.components_[b,:]

pl.figure()
pl.plot(testmag_align, label='target')
#pl.plot(testspec_align.real, label='target')
pl.plot(rec, label='rec')
#pl.plot(rec+cplx_pca.mean_, label='rec+mean')
pl.plot(rec+mag_pca.mean_, label='rec+mean')
pl.legend()
pl.show()





