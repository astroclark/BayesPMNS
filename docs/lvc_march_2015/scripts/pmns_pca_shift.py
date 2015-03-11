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
(freqaxis, low_cat, high_cat, shift_cat, original_cat, fpeaks, low_sigmas,
        high_sigmas) = pmns_pca_utils.build_catalogues(waveform_names,
                fshift_center)
delta_f = np.diff(freqaxis)[0]

# Convert to magnitude/phase
shift_mag, shift_phase = pmns_pca_utils.complex_to_polar(shift_cat)
original_mag, original_phase = pmns_pca_utils.complex_to_polar(original_cat)

#
# PCA
#
print "Performing PCA"
shift_pca = pmns_pca_utils.pca_magphase(shift_cat, freqaxis, flow=10)

#
# Compute idealised minimal matches
#
print "Computing all matches"
unshift_matches_ideal = pmns_pca_utils.unshifted_matches(original_cat, shift_pca, fpeaks,
        freqaxis, fcenter=fshift_center, flow=1000)

shift_matches_ideal = pmns_pca_utils.idealised_matches(shift_cat, shift_pca,
        delta_f=delta_f, flow=10)

# print some diagnostics
nshift=0
for i in xrange(npcs):
    for j in xrange(npcs):
        if unshift_matches_ideal[i,j] < 0.95: nshift+=1
print "Number of UNSHIFT < 0.9: ", nshift


#pl.figure()
#pl.plot(shift_phase)
#pl.show()

#sys.exit()
# -----
imageformats=['png','eps','pdf']


# Shifted
fig, ax = pmns_pca_utils.image_matches(shift_matches_ideal, waveform_names,
        title="Matches For Full Aligned Spectrum PCA")

for fileformat in imageformats:
    fig.savefig('shiftspec_ideal_matches.%s'%fileformat)

pl.show()

# UN shifted
fig, ax = pmns_pca_utils.image_matches(unshift_matches_ideal, waveform_names,
        title="Matches For Shifted Full Spectrum PCA")

for fileformat in imageformats:
    fig.savefig('unshiftspec_ideal_matches.%s'%fileformat)

pl.show()
sys.exit()




#   #
#   # Brute-force beta maximisation
#   #
#   print "Maximising Betas"
#   fixed_args = original_cat[:,0], shift_pca, fpeaks[0], freqaxis, fshift_center, delta_f
#   npcs = 1
#
#   x = pmns_pca_utils.emcee_maximise(ndim=2*npcs, fixed_params=fixed_args)
#   sys.exit()
#
#   unshifted_rec_cat = pmns_pca_utils.unshifted_rec_cat(shift_pca, [3,3], fpeaks,
#           freqaxis, fshift_center)
#
#   #best_fit = pmns_pca_utils.unshifted_template(x['x'][:len(x['x'])/2], x['x'][len(x['x'])/2:],
#   #        shift_pca, fpeaks[0], freqaxis, fshift_center, delta_f=1.0)
#
#   best_fit = pmns_pca_utils.unshifted_template(x['x'][:len(x['x'])/2], x['x'][len(x['x'])/2:],
#           shift_pca, fpeaks[0], freqaxis, fshift_center, delta_f=1.0)
#
#   #pl.figure()
#   #pl.plot(freqaxis, original_mag[:,0])
#   #pl.plot(freqaxis, abs(unshifted_rec_cat[:,0]))
#   #pl.plot(freqaxis, abs(best_fit))
#   #pl.show()
#
#   print unshift_matches_ideal[0,0]
#   print 1-x['fun']
#   sys.exit()
    

# -----
