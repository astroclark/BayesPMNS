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
pmns_freqs.py

Script to plot and analyse f-domain pmns signals
"""

from __future__ import division
import os,sys
import numpy as np

from matplotlib import pyplot as pl

import pycbc.types
from pycbc.waveform import utils as wfutils

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata

import romspline


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Waveform Generation
#

eos="tm1"
mass="135135"

# Hardcoded, fixed delta_t is fine for Bauswein et al:
delta_t = 1./16384

#
# Create waveform catalog (EOS, mass & file path)
#

waveform_data = pdata.WaveData(eos=eos, mass=mass, viscosity='lessvisc')

#
# Pull out quadrupole data
#
quadrupole_data = pwave.get_quadrupole_data(waveform_data.waves[0]['data'])

#
# Generate Hlm's
#

# Hlm is a dictionary:
#In [8]: Hlm.keys()
#Out[8]: ['l=2, m=2', 'l=2, m=-2']
Hlm = pwave.construct_Hlm(*quadrupole_data[1:])

# Turn this into amp/phase 
HlmAmp   = dict()
HlmPhase = dict()
for key in Hlm.keys():
    hplus  = pycbc.types.TimeSeries(np.real(Hlm[key]), delta_t=delta_t)
    hcross = pycbc.types.TimeSeries(-1*np.imag(Hlm[key]), delta_t=delta_t)
    HlmAmp[key] = wfutils.amplitude_from_polarizations(hplus, hcross)
    HlmPhase[key] = wfutils.phase_from_polarizations(hplus, hcross)

# ROMspline usage:
#    spline = romSpline.ReducedOrderSpline(time_data, amplitude/phase_data,
#            verbose=False)

# XXX: next steps
# 1) fix / understand units/normalisation

# 2) use romspline module to make... rom splines:
# https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/Waveforms/NR/InjectionInfrastructure

# 3) use examples in lalapps_nest2pos to make hdf5 groups for output files:
# https://github.com/johnveitch/lalsuite/blob/h5utils/lalapps/src/inspiral/posterior/lalapps_nest2pos.py#L155





