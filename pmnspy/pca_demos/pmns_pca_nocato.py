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
import pycbc.filter
import pycbc.types
import lal

# ----
#
# Waveform catalogue
#

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
(freqaxis, shift_cat, original_cat, fpeaks) = \
        pmns_pca_utils.build_catalogues(waveform_names, fshift_center)
delta_f = np.diff(freqaxis)[0]

# Convert to magnitude/phase
shift_mag, shift_phase = pmns_pca_utils.complex_to_polar(shift_cat)

#
# PCA
#
print "Performing PCA"
shift_pca = pmns_pca_utils.pca_magphase(shift_cat, freqaxis, flow=10)

#
# Generate non-catalogue waveform
#
noncat_name = 'shen_135135_lessvisc'
noncat_waveform = pmns_utils.Waveform(noncat_name)
noncat_waveform.reproject_waveform()
noncat_waveform.compute_characteristics()

peakidx = np.argmax(abs(noncat_waveform.hplus.data))
win=lal.CreateTukeyREAL8Window(len(noncat_waveform.hplus),0.25)
noncat_waveform.hplus.data *= win.data.data

# zero-pad and FFT
zpdata = np.zeros(16384)
zpdata[:len(noncat_waveform.hplus)] = np.copy(noncat_waveform.hplus.data)
zpdata = pycbc.types.TimeSeries(zpdata, delta_t=1./16384)

zpdata_spec = zpdata.to_frequencyseries()
zpdata_mag, zpdata_phase = pmns_pca_utils.complex_to_polar(zpdata_spec)

# Shift and center the noncat waveform
nocat_shift = pmns_pca_utils.shift_vec(zpdata_spec, freqaxis,
        noncat_waveform.fpeak)

# Normalise
nocat_shift_spec = pycbc.types.FrequencySeries(nocat_shift, delta_f=zpdata_spec.delta_f)
nocat_sigma = pycbc.filter.sigma(nocat_shift_spec)
nocat_shift /= nocat_sigma

nocat_mag_shift, nocat_phase_shift = pmns_pca_utils.complex_to_polar(nocat_shift)

nocat_mag_cent = np.copy(nocat_mag_shift) - shift_pca['mean_mag']
nocat_phase_cent = np.copy(nocat_phase_shift) - shift_pca['mean_phase']

#
# Project onto catalogue PCs
#
magpcs = shift_pca['magnitude_pcs']
phasepcs = shift_pca['phase_pcs']

magscores = shift_pca['magnitude_scores']
phasescores = shift_pca['phase_scores']

nocat_mag_betas   = np.dot(nocat_mag_cent, magpcs)
#nocat_mag_betas   = shift_pca['magnitude_betas'][1,:]
nocat_phase_betas = np.dot(nocat_phase_cent, phasepcs)

print 'doing reconstructions'

#
# Reconstruct
#

#   mag_rec_shift = np.copy(shift_pca['mean_mag'])
#   phase_rec_shift = np.copy(shift_pca['mean_phase'])
#   for s in xrange(len(waveform_names)):
#       mag_rec_shift += nocat_mag_betas[s] * magpcs[:,s] 
#
#   mag_rec_shift   /= np.sqrt(np.dot(mag_rec_shift,mag_rec_shift))
#   nocat_mag_shift /= np.sqrt(np.dot(nocat_mag_shift,nocat_mag_shift))
#   nocat_mag_cent  /= np.sqrt(np.dot(nocat_mag_cent,nocat_mag_cent))
#
#   f, ax= pl.subplots(nrows=1)
#   ax.plot(freqaxis, nocat_mag_shift, color='r', linewidth=2)
#   #ax.plot(freqaxis, nocat_mag_cent, color='r', linewidth=2)
#   ax.plot(freqaxis, mag_rec_shift, color='k')
#   ax.set_xlim(0,1500)
#
#pl.show()
#sys.exit()


npcs=range(1,len(waveform_names)+1)
f, ax = pl.subplots(nrows = len(waveform_names), figsize=(10,15))
for n in xrange(len(waveform_names)):

    mag_rec_shift = np.copy(shift_pca['mean_mag'])
    phase_rec_shift = np.copy(shift_pca['mean_phase'])
    print 'reconstruccting with %d pcs'%(n+1)
    for s in xrange(n+1):
        print 'summing ',s
        mag_rec_shift += nocat_mag_betas[s] * magpcs[:,s] 
        phase_rec_shift += nocat_phase_betas[s] * phasepcs[:,s] 

    mag_rec = pmns_pca_utils.unshift_vec(mag_rec_shift, freqaxis,
            noncat_waveform.fpeak)

    mag_rec_shift   /= np.sqrt(np.dot(mag_rec_shift,mag_rec_shift))
    nocat_mag_shift /= np.sqrt(np.dot(nocat_mag_shift,nocat_mag_shift))

    mag_rec         /= np.sqrt(np.dot(mag_rec, mag_rec))


    ax[n].plot(freqaxis, nocat_mag_shift, color='r', linewidth=2)
    ax[n].plot(freqaxis, mag_rec_shift, color='k')

    ax[n].set_xlim(0,1500)


pl.show()










