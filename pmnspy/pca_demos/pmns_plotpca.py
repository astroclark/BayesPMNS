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
waveform_data, pmpca, magnitude_euclidean, phase_euclidean, matches, \
        delta_fpeak, delta_R16 = pickle.load(open(pickle_file, "rb"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot Results 

#
# Image plots
#
f, ax = pplots.image_euclidean(magnitude_euclidean, waveform_data)
f, ax = pplots.image_euclidean(phase_euclidean, waveform_data)
f, ax = pplots.image_matches(matches, waveform_data, mismatch=False)


pl.show()
