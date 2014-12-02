#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <james.clark@physics.gatech.edu>
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
#np.seterr(all="raise", under="ignore")
import matplotlib
#matplotlib.use("Agg")

from scipy import signal, optimize, special, stats

import pmns_utils
#import pmns_simsig as simsig

import lal
import lalsimulation as lalsim
import pycbc.filter

import pylab as pl

# --------------------------------------------------------------------
# Data Generation

#
# Generate Signal Data
#

# Signal
print ''
print '--- %s ---'%sys.argv[1]
waveform = pmns_utils.Waveform('%s_lessvisc'%sys.argv[1])
waveform.reproject_waveform()

# --------------------------------------------------------------------
# Tapering
#
# We'll look at the properties of the waveform as a function of the amount of
# data we retain around the inspiral peak
# 
# Seconds before peak to retain: negative will truncate waveform (then taper)
# waveform *prior* to peak, positive will truncate *after* the peak

#
# Find time-domain peak
#
peak_idx = np.argmax(abs(waveform.hplus.
