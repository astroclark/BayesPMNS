#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
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
pmns_multifit.py

Explore 3 ringdown fits from Bauswein 2015
"""

from __future__ import division
import os,sys
import numpy as np
from scipy import signal

import cma

from matplotlib import pyplot as pl

import lal
import pycbc.types
import pycbc.filter
import pycbc.psd 

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata

# ________________ - Local Defs - ________________ #

def compute_rate(sensemon_range, bns_rate_perL10perMyr=(0.6,60,600)):

    bns_rateperL10_perYr = np.array(bns_rate_perL10perMyr)/1e6

    cumlum_data = pwave.CL_vs_PhysDist()
    if sensemon_range>cumlum_data[-2,0]:
        Ng = (4./3)*np.pi*(sensemon_range**3)*0.0116
        cumlum_in_range = Ng/1.7
    else:

        # Get cumulative blue-light luminosity in L10 data:
        cumlum_data = pwave.CL_vs_PhysDist()

        # Interpolate to this range
        cumlum_in_range = np.interp(sensemon_range, cumlum_data[:,0],
                cumlum_data[:,1])

    rates = [cumlum_in_range * rate for rate in bns_rateperL10_perYr]

    return rates

def frequencies(fpeak):
    """
    Compute fspiral and f2-0 in Bauswein 15
    """

    # Compute frequencies (equation 8, 9)
    fspiral = 1000*(0.8058*fpeak/1000 - 0.1895)
    f20 = 1000*(1.0024*fpeak/1000 - 1.0798)

    return fspiral, f20

def multi_ring_template(t0, Apeak, Aspiral, A20, fpeak, taupeak, tauspiral,
        tau20, phipeak, phispiral, phi20, delta_t=1./16384, datalen=16384):
    """
    Build the template described by equation 7 in Bauswein 15
    """

    t = np.arange(0,datalen*delta_t,delta_t)

    fspiral, f20 = frequencies(fpeak)

    peak = Apeak * np.exp(-(t-t0)/taupeak) * \
            np.cos(2*np.pi*fpeak*(t-t0) + phipeak)
    peak[t<t0]=0.0
    spiral = Aspiral * np.exp(-(t-t0)/tauspiral) * \
            np.cos(2*np.pi*fspiral*(t-t0) + phispiral)
    spiral[t<t0]=0.0
    axi = A20 * np.exp(-(t-t0)/tau20) * \
            np.cos(2*np.pi*f20*(t-t0) + phi20)
    axi[t<t0]=0.0


    return peak+spiral+axi


def init_guess(hplus):
    """
    Generate initial guess at waveform parameters
    """

    Hplus = hplus.to_frequencyseries()

    # frequencies
    fpeak = Hplus.sample_frequencies[np.argmax(abs(Hplus))]
    fspiral, f20 = frequencies(fpeak)


    # Damping time
    hcross = np.imag(signal.hilbert(hplus.data))
    ampseries = abs(hplus - 1j*hcross)
    taupeak = hplus.sample_times[np.argmin(abs(ampseries -
        max(ampseries)/np.exp(1)))] - hplus.sample_times[np.argmax(ampseries)]
    tauspiral = 0.5*taupeak
    tau20 = 0.5*taupeak

    # Amplitudes
   #Apeak = abs(Hplus[np.argmin(abs(Hplus.sample_frequencies-fpeak))])
   #Aspiral = abs(Hplus[np.argmin(abs(Hplus.sample_frequencies-fspiral))])
   #A20 = abs(Hplus[np.argmin(abs(Hplus.sample_frequencies-f20))])
    Apeak = max(ampseries)
    Aspiral = 0.5*Apeak
    A20 = 0.5*Apeak

    # start time
    t0 = hplus.sample_times[np.argmax(ampseries)]

    # phase
    phipeak = 0.0
    phispiral = 2*np.pi*np.random.rand()
    phi20 = 2*np.pi*np.random.rand()

    return np.array([t0, Apeak, Aspiral, A20, fpeak, taupeak, tauspiral, tau20, phipeak,
            phispiral, phi20])

def min_overlap(x):
    """
    Compute 1-overlap between the waveform and the template (for
    minimisation)
    """


    x = code(x,'decode')
    t0, Apeak, Aspiral, A20, fpeak, taupeak, tauspiral, tau20, phipeak, phispiral, phi20 = x

#
#   if (Apeak>0) * (Aspiral>0) * (A20>0) * (fpeak>1000) * (fpeak<4000) \
#           * (taupeak>1e-3) * (tauspiral>5e-5) * (tauspiral>5e-5) \
#           * (phipeak>0) * (phipeak<2*np.pi) * (phispiral>0) \
#           * (phipeak<2*np.pi) * (phi20>0) * (phi20<2*np.pi):
#
#               template = multi_ring_template(*x)
#
#               if sum(np.isnan(template))>0:
#                   cost = np.NaN
#               else:
#           #    template_ts = pycbc.types.TimeSeries(template, delta_t=1./16384)
#           #    overlap = pycbc.filter.overlap(hplus, template_ts, psd=psd,
#           #            low_frequency_cutoff=1000, high_frequency_cutoff=4096,
#           #            normalized=True)
#                   #cost = sum(template - hplus)**2
#                   cost = 1-np.dot(template,hplus)
#
#   else:
#       cost=np.NaN

    template = multi_ring_template(*x)
#    template_ts = pycbc.types.TimeSeries(template, delta_t=1./16384)
 
#   overlap = pycbc.filter.overlap(hplus, template_ts, psd=psd,
#           low_frequency_cutoff=1000, high_frequency_cutoff=4096,
#           normalized=True)
#   return -1*float(overlap)

    cost = sum((template-hplus)**2)

    return cost
    

    

encode = lambda x,low,high: (x-low)*10.0/(high-low)
decode = lambda x,low,high: low + (high-low) * x/10.0

def code(xin, operation):
    """
    Scale the parameters
    """

    xout = np.zeros(len(xin))
    if operation=='encode':
        xout[0]  = encode(xin[0], 0.0, 0.1)
        xout[1]  = encode(xin[1], 0.01, 1.0)
        xout[2]  = encode(xin[2], 0.01, 1.0)
        xout[3]  = encode(xin[3], 0.01, 1.0)
        xout[4]  = encode(xin[4], 1000, 4000)
        xout[5]  = encode(xin[5], 1e-3, 1e-1)
        xout[6]  = encode(xin[6], 1e-3, 1e-1)
        xout[7]  = encode(xin[7], 1e-3, 1e-1)
        xout[8]  = encode(xin[8], 0, 2*np.pi)
        xout[9]  = encode(xin[9], 0, 2*np.pi)
        xout[10]  = encode(xin[10], 0, 2*np.pi)
    elif operation=='decode':
        xout[0]  = decode(xin[0], 0.0, 0.1)
        xout[1]  = decode(xin[1], 0.01, 1.0)
        xout[2]  = decode(xin[2], 0.01, 1.0)
        xout[3]  = decode(xin[3], 0.01, 1.0)
        xout[4]  = decode(xin[4], 1000, 4000)
        xout[5]  = decode(xin[5], 1e-3, 1e-1)
        xout[6]  = decode(xin[6], 1e-3, 1e-1)
        xout[7]  = decode(xin[7], 1e-3, 1e-1)
        xout[8]  = decode(xin[8], 0, 2*np.pi)
        xout[9]  = decode(xin[9], 0, 2*np.pi)
        xout[10]  = decode(xin[10], 0, 2*np.pi)

    return xout



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialise
#

noise_curve=sys.argv[1]
horizon_snr=5
reference_distance=50
ndet=1

eos="dd2"
mass="135135"
viscosity="oldvisc"


# XXX: should probably fix this at the module level..
if eos=="all": eos=None
if mass=="all": mass=None
if viscosity=="all": viscosity=None

#
# Create the list of dictionaries which comprises our catalogue
#
waveform_data = pdata.WaveData(eos=eos,viscosity=viscosity, mass=mass)

#
# Initialise standard devs
#
sigma0 = 0.05

#
# Create Waveforms and compute params
#
for w, wave in enumerate(waveform_data.waves):

    print "Matching for %s, %s ,%s (%d of %d)"%(
            wave['eos'], wave['mass'], wave['viscosity'], w+1,
            waveform_data.nwaves)

    #
    # Create test waveform
    #
    waveform = pwave.Waveform(eos=wave['eos'], mass=wave['mass'],
            viscosity=wave['viscosity'], distance=reference_distance)
    waveform.reproject_waveform()

    hplus = pycbc.types.TimeSeries(np.zeros(16384),
            delta_t=waveform.hplus.delta_t)
    hplus.data[:len(waveform.hplus)] = np.copy(waveform.hplus.data)

    hplus.data /= np.linalg.norm(hplus.data)

    Hplus = hplus.to_frequencyseries()

    sys.exit()
    #
    # Construct PSD
    #
    psd = pwave.make_noise_curve(fmax=Hplus.sample_frequencies.max(),
            delta_f=Hplus.delta_f, noise_curve=noise_curve)

    #
    # Get params
    #
    x0 = init_guess(hplus)
    x0 = code(x0, 'encode')
    x = cma.fmin(min_overlap, x0, sigma0=sigma0)

    result = code(x[0], 'decode')



    sys.exit()




