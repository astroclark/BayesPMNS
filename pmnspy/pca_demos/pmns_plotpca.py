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
pmns_plotpca.py

Script to load results from pmns_pca_analysis.py and make nice plots
"""

from __future__ import division
import os,sys
import cPickle as pickle
import numpy as np

from matplotlib import pyplot as pl

from pmns_utils import pmns_plots as pplots

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Results
#
pickle_file = sys.argv[1]
waveform_data, _, _, matches, delta_fpeak, delta_R16 = \
        pickle.load(open(pickle_file, "rb"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reduce matches, deltas to lessvisc waveforms
#

#   # Count the number of lessvisc signals and record their indices
#   nlessvisc=0
#   lessviscidx=[]
#   for w,wave in enumerate(waveform_data.waves):
#       if wave['viscosity'] == 'lessvisc': 
#           nlessvisc+=1
#           lessviscidx.append(w)
#
#   matches     = np.zeros(shape=(nlessvisc, waveform_data.nwaves))
#   delta_fpeak = np.zeros(shape=(nlessvisc, waveform_data.nwaves))
#   delta_R16   = np.zeros(shape=(nlessvisc, waveform_data.nwaves))
#   for w in xrange(nlessvisc):
#
#       matches[w, :]     = all_matches[lessviscidx[w],:]
#       delta_fpeak[w, :] = all_delta_fpeak[lessviscidx[w],:]
#       delta_R16[w, :]   = all_delta_R16[lessviscidx[w],:]
#

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot Results 

#
# Image plots
#
#f, ax = pplots.image_euclidean(magnitude_euclidean, waveform_data)
#f, ax = pplots.image_euclidean(phase_euclidean, waveform_data)
#f, ax = pplots.image_matches(matches, waveform_data, mismatch=False)

#
# Line plots
#

# Reconstruction quality
#f, ax = pplots.plot_fidelity_by_npc(magnitude_euclidean, waveform_data,
#        ylabel=r'$||A - A_r||$', legloc=None)
#f, ax = pplots.plot_fidelity_by_npc(phase_euclidean, waveform_data,
#        ylabel=r'$||\phi - \phi_r||$', legloc=None)
f, ax = pplots.plot_fidelity_by_npc(matches, waveform_data, legloc='lower right')

#f, ax = pplots.plot_delta_by_npc(delta_fpeak, waveform_data, 
#        ylabel=r"$\delta f_{\rm peak}$ [Hz]", legloc=None)
#f, ax = pplots.plot_delta_by_npc(delta_R16, waveform_data,
#        ylabel=r"$\delta R_{1.6}$ [km]", legloc=None)


pl.show()
