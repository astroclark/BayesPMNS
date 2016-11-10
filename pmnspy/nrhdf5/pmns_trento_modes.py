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
import hashlib
import numpy as np
import glob

from matplotlib import pyplot as pl

import pycbc.types
from pycbc.waveform import utils as wfutils
from pycbc import pnutils
#import pycbc.filter
import scipy.signal
import lal


import h5py
#import romspline as romSpline
import romSpline


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Waveform Generation
#

eos="apr4"
mass="142159"
total_mass=1.42+1.59
mass1=1.42
mass2=1.59

datadir='/home/jclark/Projects/gwaveforms/BNS/APR4/mb_1.42_1.59_d45/bns_apr4_4p_mb_1.42_1.59_irrot.ninja'

globpattern='bns_apr4_4p_mb_1.42_1.59_irrot*asc'
ninjafiles=glob.glob(os.path.join(datadir,globpattern))

# Hardcoded, fixed delta_t is fine for Bauswein et al:
delta_t = 1./16384
f_lower_hz = 1000.0 # waveform not valid below here really; could put this on the
                    # pipeline to handle...



#startFreqHz = startFreq / (lal.TWOPI * massTotal * lal.MTSUN_SI)
f_lower = f_lower_hz * (lal.TWOPI * total_mass * lal.MTSUN_SI)

with h5py.File(eos+'_'+mass+'.h5','w') as fd:

    #
    # Set metadata
    #

    mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(mass1, mass2)
    fd.attrs.create('NR_group', 'Trento')
    fd.attrs.create('name', 'Trento:BNS:%s' %
            (eos+'_'+mass))
    hashtag = hashlib.md5()
    hashtag.update(fd.attrs['name'])
    fd.attrs.create('hashtag', hashtag.digest())
    fd.attrs.create('f_lower_at_1MSUN', f_lower)
    fd.attrs.create('eta', eta)
    fd.attrs.create('spin1x', 0.0)
    fd.attrs.create('spin1y', 0.0)
    fd.attrs.create('spin1z', 0.0)
    fd.attrs.create('spin2x', 0.0)
    fd.attrs.create('spin2y', 0.0)
    fd.attrs.create('spin2z', 0.0)
    fd.attrs.create('coa_phase', 0.0)
    fd.attrs.create('mass1', mass1/total_mass)
    fd.attrs.create('mass2', mass2/total_mass)


    for ninjafile in ninjafiles:
        print ninjafile

        lm = ninjafile.split('_')[-1].replace('.asc','')
        l=float(lm.split('m')[0].replace('l',''))
        m=float(lm.split('m')[-1])

        timesGeom, hplusGeom, hcrossGeom = np.loadtxt(ninjafile, unpack=True)

        #
        # Reverse engineer for sanity check and resampling
        #
        scalefac = lal.MRSUN_SI / (100e6*lal.PC_SI)
        times = timesGeom*lal.MTSUN_SI

        native_delta_t = np.diff(times)[0]
        hplus = hplusGeom*scalefac
        hcross = hcrossGeom*scalefac

        target_nsamp = times[-1]/delta_t
        hplus_16384, times_16384 = scipy.signal.resample(hplus, num=target_nsamp,
                t=times)
        hcross_16384, _ = scipy.signal.resample(hcross, num=target_nsamp,
                t=times)
        times_16384 = times_16384[:len(hplus_16384)]

        massMpc = total_mass*lal.MRSUN_SI / (lal.PC_SI*100.0e6)

        hplusMpc  = pycbc.types.TimeSeries(hplus/massMpc, delta_t=delta_t)
        hcrossMpc = pycbc.types.TimeSeries(hcross/massMpc, delta_t=delta_t)
        times_M = times / (lal.MTSUN_SI * total_mass)
 
        HlmAmp   = wfutils.amplitude_from_polarizations(hplusMpc,
                hcrossMpc).data
        HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data 

        if l!=2 or m!=2:
            HlmAmp = np.zeros(len(HlmAmp))
            HlmPhase = np.zeros(len(HlmPhase))
 
        print 'fitting spline'
        sAmph = romSpline.ReducedOrderSpline(times_M, HlmAmp, verbose=False)
        sPhaseh = romSpline.ReducedOrderSpline(times_M, HlmPhase, verbose=False)
      
        gramp = fd.create_group('amp_l%d_m%d' %(l,m))
        sAmph.write(gramp)
        
        grphase = fd.create_group('phase_l%d_m%d' %(l,m))
        sPhaseh.write(grphase)

 





