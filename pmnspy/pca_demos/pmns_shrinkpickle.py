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
pmns_shrinkpickle.py

Script to load results from pmns_pca_analysis.py and re-save just the
matches/errors for faster plotting
"""

from __future__ import division
import os,sys
import cPickle as pickle

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Results
#
pickle_file = sys.argv[1]
waveform_data, pmpca, magnitude_euclidean, phase_euclidean, matches, \
        delta_fpeak, delta_R16 = pickle.load(open(pickle_file, "rb"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save results for plotting seperately
#

picklename = pickle_file.replace(".pickle","_reduced.pickle")

pickle.dump([waveform_data, magnitude_euclidean, phase_euclidean, matches,
    delta_fpeak, delta_R16], open(picklename, "wb"))
