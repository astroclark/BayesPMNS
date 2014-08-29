#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <clark@physics.umass.edu>
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
matplotlib.use("Agg")
from matplotlib import pyplot as pl

from scipy import signal, optimize, special, stats

import pmns_utils
import pmns_simsig as simsig

import lal
import lalsimulation as lalsim
import pycbc.filter

from pylal import Fr

import emcee
import mpmath


# --------------------------------------------------------------------
# Data Generation

#
# Generate Signal Data
#

# Signal
print 'generating waveform...'
waveform = pmns_utils.Waveform('apr_135135_lessvisc')
waveform.compute_characteristics()

# Extrinsic parameters
geocent_peak_time = 1090606564

ext_params = simsig.ExtParams(distance=10, ra=0.0, dec=0.0,
        polarization=0.0, inclination=0.0, phase=0.0,
        geocent_peak_time=geocent_peak_time)

# Construct the time series for these params
waveform.make_wf_timeseries(theta=ext_params.inclination,
        phi=ext_params.phase)

#
# Generate IFO data
#

datalen=6.0

# Make sure we inject the signal in the middle!
epoch = geocent_peak_time - 0.5*datalen

seed=0
#seed=np.random.random_integers(1000)
#seed=int(sys.argv[2])

# XXX Sanity check: data segment must contain full extent of max signal duration
max_sig=100e-3
if max_sig > geocent_peak_time + 0.5*datalen:
    print >> sys.stderr, "templates will lie outside data segment: extend data length"
    sys.exit()

print >> sys.stdout, "generating detector responses & noise..."

det1_data = simsig.DetData(det_site="H1", noise_curve='aLIGO', waveform=waveform,
        ext_params=ext_params, duration=datalen, seed=seed, epoch=epoch,
        f_low=10.0)

#det2_data = simsig.DetData(det_site="L1", noise_curve='aLIGO', waveform=waveform,
#        ext_params=ext_params, duration=datalen, seed=seed+1, epoch=epoch,
#        f_low=10.0)

#det3_data = simsig.DetData(det_site="V1", noise_curve='adVirgo', waveform=waveform,
#        ext_params=ext_params, duration=datalen, seed=seed+2, epoch=epoch,
#        f_low=10.0)


# --------------------------------------------------------------------
# Write Frames
filename = 'H-H1_HMNS_APR135135_OPTIMAL-%d-%d.gwf'%(epoch, datalen)
datadict={}
datadict['name']='H1:LDAS-STRAIN'
datadict['data']=det1_data.td_signal.data
datadict['start']=det1_data.td_signal.start_time
datadict['dx']=det1_data.td_signal.delta_t
Fr.frputvect(filename, [datadict], verbose=True)

# --------------------------------------------------------------------
# Plot data
if 1:
    pl.figure()
    pl.semilogy(det1_data.td_response.to_frequencyseries().sample_frequencies,
            abs(det1_data.td_response.to_frequencyseries()))
    pl.semilogy(det1_data.td_signal.to_frequencyseries().sample_frequencies,
            abs(det1_data.td_signal.to_frequencyseries()),'r')
    pl.xlim(1000,5000)
    pl.ylim(1e-25,1e-19)

    pl.figure()
    pl.plot(det1_data.td_response.sample_times, det1_data.td_response.data)
    pl.plot(det1_data.td_signal.sample_times, det1_data.td_signal.data,'r')

    pl.show()

    sys.exit()


