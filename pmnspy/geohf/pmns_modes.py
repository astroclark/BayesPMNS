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

from matplotlib import pyplot as pl

import pycbc.types
from pycbc.waveform import utils as wfutils
from pycbc import pnutils
import lal

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata

import h5py
#import romspline as romSpline
import romSpline


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Waveform Generation
#

eos="tm1"
mass="135135"
total_mass=2*1.35
mass1=1.35
mass2=1.35

# Hardcoded, fixed delta_t is fine for Bauswein et al:
delta_t = 1./16384
f_lower_hz = 1000.0 # waveform not valid below here really; could put this on the
                 # pipeline to handle...
#startFreqHz = startFreq / (lal.TWOPI * massTotal * lal.MTSUN_SI)
f_lower = f_lower_hz * (lal.TWOPI * total_mass * lal.MTSUN_SI)

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

Hlm = pwave.construct_Hlm(*quadrupole_data[1:])



wavelen=len(Hlm['l=2, m=2'])
times=np.arange(0, wavelen*delta_t, delta_t)
times_M = times / (lal.MTSUN_SI * total_mass)
# GEOMETERIZE  HLM (might be already!) see line 241+ of pmns_waveform.py


l=2
with h5py.File(eos+'_'+mass+'.h5','w') as fd:

    #
    # Set metadata
    #

    mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(mass1, mass2)
    fd.attrs.create('NR_group', 'Bauswein')
    fd.attrs.create('name', 'Bauswein:BNS:%s' %
            (waveform_data.eos+'_'+waveform_data.mass+'_'+waveform_data.viscosity))
    hashtag = hashlib.md5()
    hashtag.update(fd.attrs['name'])
    fd.attrs.create('hashtag', hashtag.digest())
    fd.attrs.create('f_lower_at_1MSUN', f_lower)
    fd.attrs.create('eta', eta)
    fd.attrs.create('LNhatx', 0.0)
    fd.attrs.create('LNhaty', 0.0)
    fd.attrs.create('LNhatz', 0.0)
    fd.attrs.create('lnhatx', 0.0)
    fd.attrs.create('lnhaty', 0.0)
    fd.attrs.create('lnhatz', 0.0)
    fd.attrs.create('nhatx', 0.0)
    fd.attrs.create('nhaty', 0.0)
    fd.attrs.create('nhatz', 0.0)
    fd.attrs.create('spin1z', 0.0)
    fd.attrs.create('spin1x', 0.0)
    fd.attrs.create('spin1y', 0.0)
    fd.attrs.create('spin2z', 0.0)
    fd.attrs.create('spin2x', 0.0)
    fd.attrs.create('spin2y', 0.0)
    fd.attrs.create('coa_phase', 0.0)
    fd.attrs.create('mass1', mass1/total_mass)
    fd.attrs.create('mass2', mass2/total_mass)


    for m in range(-l,l+1):

        if abs(m)!=2:
            HlmAmp=np.zeros(wavelen)
            HlmPhase=np.zeros(wavelen)
        else:
            key="l=%d, m=%d"%(l,m)

            hplus  = pycbc.types.TimeSeries(np.real(Hlm[key]), delta_t=delta_t)
            hcross = pycbc.types.TimeSeries(-1*np.imag(Hlm[key]), delta_t=delta_t)

            massMpc =  total_mass * lal.MRSUN_SI / (  20.0e6)
            HlmAmp   = massMpc*wfutils.amplitude_from_polarizations(hplus, hcross).data
            HlmPhase = wfutils.phase_from_polarizations(hplus, hcross).data 

        sAmph = romSpline.ReducedOrderSpline(times_M, HlmAmp, verbose=True)
        sPhaseh = romSpline.ReducedOrderSpline(times_M, HlmPhase, verbose=True)
      
        gramp = fd.create_group('amp_l%d_m%d' %(l,m))
        sAmph.write(gramp)
        
        grphase = fd.create_group('phase_l%d_m%d' %(l,m))
        sPhaseh.write(grphase)






