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
#matplotlib.use("Agg")

from scipy import signal, optimize, special, stats

import pmns_utils

import lal
import lalsimulation as lalsim
import pycbc.filter
from pycbc.psd import aLIGOZeroDetHighPower

import pylab as pl

def SG_template(frequency, quality, delta_t=1./16384, epoch=0.0, datalen=16384):
    """
    Return a pycbc timeseries object with a sine-Gaussian with hrss=1,
    ellipticity=1, polar angle=0
    """

    hp, _ = lalsim.SimBurstSineGaussian(quality, frequency, 1.0/np.sqrt(2), 1.0, 0,
            delta_t)

    tmparr=np.zeros(datalen)
    tmparr[:hp.data.length]=hp.data.data[:]

    return pycbc.types.TimeSeries(tmparr, delta_t=delta_t, epoch=epoch) 

def mismatch(intrinsic_params, data):

    # variable params
    f0, Q = intrinsic_params

    if (f0low<f0<f0upp) and (10 < Q < 1000):

        # Generate template
        tmplt_raw = SG_template(f0, Q)

        # Get uniform lengths
        strainarr = np.zeros(16384)
        strainarr[:len(data)] = np.array(data)
        strain = pycbc.types.TimeSeries(strainarr, delta_t=1./16384)

        tmpltarr = np.zeros(16384)
        tmpltarr[:len(tmplt_raw)] = np.array(tmplt_raw)
        tmplt = pycbc.types.TimeSeries(tmpltarr, delta_t=1./16384)

        try:

            psd = aLIGOZeroDetHighPower(len(strain.to_frequencyseries()),
                    strain.to_frequencyseries().delta_f, 10) 

            match = pycbc.filter.match(tmplt,strain, 
                    low_frequency_cutoff=fmin, high_frequency_cutoff=fmax,
                    psd=psd)[0]

        except ZeroDivisionError:
            match = 0

        return 1-match

    else:

        return 1

def min_mismatch(init_params, data):

    x0 = [
            init_params['frequency'],
            init_params['quality']

            ]

    err_func = lambda x0: mismatch(x0, data)

    return optimize.minimize(err_func, x0=x0,
            method='nelder-mead')

def apply_taper(TimeSeries,newlen=16384):
    """
    Smoothly taper the start of the data in TimeSeries using LAL tapering
    routines

    Also zero pads
    """

    # Create and populate lal time series
    tmp=lal.CreateREAL8TimeSeries('tmp', lal.LIGOTimeGPS(), 0.0,
            TimeSeries.delta_t, lal.StrainUnit, newlen)
            #TimeSeries.delta_t, lal.StrainUnit, len(TimeSeries))
    tmp.data.data = np.zeros(newlen)
    tmp.data.data[0:len(TimeSeries)]=TimeSeries.data

    # Taper
    lalsim.SimInspiralREAL8WaveTaper(tmp.data, lalsim.SIM_INSPIRAL_TAPER_START)

    return pycbc.types.TimeSeries(tmp.data.data, delta_t=TimeSeries.delta_t)


# --------------------------------------------------------------------
# Data Generation
global f0low
global f0upp
global fmin
global fmax
f0low=1500
f0upp=4000
fmin=1000
fmax=8192

#
# Generate Signal Data
#


waveform_names=['apr_135135_lessvisc',
                'shen_135135_lessvisc',
                'dd2_135135_lessvisc' ,
                'dd2_165165_lessvisc' ,
                'nl3_135135_lessvisc' ,
                'nl3_1919_lessvisc'   ,
                'tm1_135135_lessvisc' ,
                'tma_135135_lessvisc' ,
                'sfhx_135135_lessvisc',
                'sfho_135135_lessvisc']


#
# Waveform
#
print ''
print '--- %s ---'%sys.argv[1]
#waveform = pmns_utils.Waveform('%s_lessvisc'%sys.argv[1])
matches=[]
for name in waveform_names:
    waveform = pmns_utils.Waveform(name)
    waveform.compute_characteristics()

    # Get optimal hplus
    waveform.reproject_waveform()

    h = apply_taper(waveform.hplus)
    h_s = pycbc.filter.sigma(h, low_frequency_cutoff=fmin,
            high_frequency_cutoff=fmax)

    #
    # Spectrum
    #
    H = abs(h.to_frequencyseries())**2
    freqs = h.to_frequencyseries().sample_frequencies.data


    # reduce
    H=H[(freqs>=fmin)*(freqs<=fmax)]
    freqs=freqs[(freqs>=fmin)*(freqs<=fmax)]

    # --------------------------------------------------------------------
    #
    # Match Calculations
    #
    init_guess={}

    idx=(freqs>f0low)*(freqs<f0upp)
    init_guess['frequency'] = freqs[idx][np.argmax(H[idx])]
    init_guess['quality']   = 100

    # Get match for full waveform
    mismatch_full = min_mismatch(init_guess, h)

    print '---------------------------'
    print 'Full Waveform:'
    print mismatch_full
    matches.append(1-mismatch_full['fun'])


sys.exit()
# best fitting template
match_tmplt = SG_template(mismatch_full['x'][0], mismatch_full['x'][1])
match_tmplt.data *= h_s

match_tmplt_F=match_tmplt.to_frequencyseries()

# --------------------------------------------------------------------
#
# With Windowing
#

#
# Find time-domain peak and zero-out data prior to peak time+deltaT
#

delay=0e-3
tzero_idx = h.max_loc()[1] + delay / h.delta_t

hnew = pycbc.types.TimeSeries(h,delta_t=h.delta_t)
hnew.data[:tzero_idx]=0.0
hnew = apply_taper(hnew)
hnew_s = pycbc.filter.sigma(hnew, low_frequency_cutoff=fmin,
        high_frequency_cutoff=fmax)

# spectrum
Hnew = abs(hnew.to_frequencyseries())**2
freqsnew = hnew.to_frequencyseries().sample_frequencies.data
Hnew=Hnew[(freqsnew>=fmin)*(freqsnew<=fmax)]
freqsnew=freqsnew[(freqsnew>=fmin)*(freqsnew<=fmax)]

# use best fit params from full waveform as initial guess
init_guess['frequency'] = mismatch_full['x'][0]
init_guess['quality'] = mismatch_full['x'][1]
mismatch_restricted = min_mismatch(init_guess, hnew)

# best fitting template
match_tmplt_restricted = SG_template(mismatch_restricted['x'][0], mismatch_full['x'][1])
match_tmplt_restricted.data*=hnew_s

match_tmplt_restricted_F=match_tmplt_restricted.to_frequencyseries()

print '---------------------------'
print 'Restricted waveform: '
print mismatch_restricted


#
# Plotting / results
#
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

f, ax = pl.subplots()

ax.semilogy(freqs,H,label='full', color='k')
ax.semilogy(freqsnew,Hnew,label='truncated at merger', color='k', linestyle='--')
ax.semilogy(match_tmplt_F.sample_frequencies, abs(match_tmplt_F)**2, 
        label='SG match=%.2f (full)'%(1-mismatch_full['fun']), color='r')
ax.semilogy(match_tmplt_restricted_F.sample_frequencies, abs(match_tmplt_F)**2, 
        label='SG match=%.2f (truncated)'%(1-mismatch_restricted['fun']),
        color='m', linestyle='--')

ax.set_xlim(fmin,min(fmax,2*mismatch_full['x'][0]))
#ax.set_ylim(1e-51,1e-45)
ax.set_ylim(0.01*max(H), 10*max(H))
ax.legend(loc='lower right')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'|H(f)|$^2$ @ 20 Mpc / Hz$^{-1}$')

axins = inset_axes(ax, width="40%", height="40%", loc=1, borderpad=1)

axins.plot(freqs,H,label='full', color='k', linestyle='-')
axins.plot(freqsnew,Hnew,label='truncated', color='k', linestyle='--')
axins.plot(match_tmplt_F.sample_frequencies, abs(match_tmplt_F)**2, 
        label='best fit (full, match=%.2f)'%(1-mismatch_full['fun']), color='r',
        linestyle='-')
axins.plot(match_tmplt_restricted_F.sample_frequencies, abs(match_tmplt_F)**2, 
        label='best fit (truncated, match=%.2f)'%(1-mismatch_restricted['fun']),
        color='r', linestyle='--')

axins.set_xlim(mismatch_full['x'][0]-250, mismatch_full['x'][0]+250)


pl.show()


