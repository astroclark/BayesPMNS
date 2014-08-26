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
import pmns_simsig as simsig

import lal
import lalsimulation as lalsim
import pycbc.filter

import pylab as pl

import emcee
import mpmath

#   def damped_chirp(tmplt_len, Amp, f0, tau, df, phi0=0):
#       """
#       """
#       # Get time stamps for non-zero part of signal
#       time_axis = np.arange(0, tmplt_len / 16384.0, 1/16384.0)
#
#       # Frequency evolution
#       f1 = f0+df
#
#       cycles = signal.chirp(time_axis, f0=f0, t1=tau, f1=f1, method='linear',
#               phi=phi0)
#
#       decay = np.exp(-0.5*(time_axis/tau)**2)
#       #decay = np.exp(-0.5*(time_axis/tau))
#
#       #win = lal.CreateTukeyREAL8Window(len(decay), beta)
#       #return time_axis, np.real(Amp*cycles*decay)*win.data.data
#       return time_axis, np.real(Amp*cycles*decay)

class DampedChirpParams:
    """
    """

    def __init__(self, tmplt_len=16384, Amp=1.0, f0=2000, tau=1e-3, df=0):
        self.tmplt_len=tmplt_len
        self.Amp=Amp
        self.f0=f0
        self.tau=tau
        self.df=df

def damped_chirp_tmplt(det_data, int_params, ext_params):
    """
    Build a template for the detector in det_data from the intrinsic and
    extrinsic parameteres in int_params and ext_params
    """

    #
    # Compute polarisations for damped chirp
    #

#   _, hp = damped_chirp(int_params.tmplt_len,
#           int_params.Amp, int_params.f0, int_params.tau, int_params.df)
#
#   _, hc = damped_chirp(int_params.tmplt_len,
#           int_params.Amp, int_params.f0, int_params.tau, int_params.df,
#           phi0=90.0)

    # Get the epoch for the start of the time series
    epoch = ext_params.geocent_peak_time# - \
            #np.argmax(hp)*det_data.td_response.delta_t
            # XXX: why don't we need to subtract this bit...?

    Q = 2.0*np.pi*int_params.tau*int_params.f0

    hp, hc = lalsim.SimBurstChirplet(Q, int_params.f0,
            int_params.df, int_params.Amp, 0, 0, 1.0/16384)
    hp.data.data[0:hp.data.length/2] = 0.0
    hc.data.data[0:hp.data.length/2] = 0.0

    hplus = lal.CreateREAL8TimeSeries('hplus', epoch, 0,
            det_data.td_noise.delta_t, lal.StrainUnit, hp.data.length)
    hplus.data.data=np.copy(hp.data.data)
    del hp
 
    hcross = lal.CreateREAL8TimeSeries('hcross', epoch, 0,
            det_data.td_noise.delta_t, lal.StrainUnit, hc.data.length)
    hcross.data.data=np.copy(hc.data.data)
    del hc


#   try:
#       time_delay = lal.TimeDelayFromEarthCenter(det_data.det_site.location,
#               ext_params.ra, ext_params.dec, ext_params.geocent_peak_time)
#   except RuntimeError:
#       time_delay = lal.TimeDelayFromEarthCenter(det_data.det_site.location,
#               ext_params.ra, ext_params.dec, -1e4)


    # --- Put the polarisations into LAL TimeSeries
#   hplus = lal.CreateREAL8TimeSeries('hplus', epoch, 0,
#           det_data.td_noise.delta_t, lal.StrainUnit, len(hp))
#   hplus.data.data=np.copy(hp)
#   del hp
#
#   hcross = lal.CreateREAL8TimeSeries('hcross', epoch, 0,
#           det_data.td_noise.delta_t, lal.StrainUnit, len(hc))
#   hcross.data.data=np.copy(hc)
#   del hc

    #
    # Project polarisations down to detector
    #
    tmplt = lalsim.SimDetectorStrainREAL8TimeSeries(hplus, hcross,
            ext_params.ra, ext_params.dec, ext_params.polarization,
            det_data.det_site) 
    del hplus, hcross

    # Scale for distance (waveforms extracted at 20 Mpc)
    tmplt.data.data *= 20.0 / ext_params.distance

    #
    # Finally make this the same size as the data (useful for fitting)
    #
    #lal.ResizeREAL8TimeSeries(tmplt, 0, len(det_data.td_response.data))
    no_noise = lal.CreateREAL8TimeSeries('blah', det_data.td_noise.start_time, 0,
            det_data.td_noise.delta_t, lal.StrainUnit, 
            int(det_data.td_noise.duration / det_data.td_noise.delta_t))
    no_noise.data.data = np.zeros(int(det_data.td_noise.duration / det_data.td_noise.delta_t))

    tmplt = lal.AddREAL8TimeSeries(no_noise, tmplt)

    return pycbc.types.timeseries.TimeSeries(\
            initial_array=np.copy(tmplt.data.data), delta_t=tmplt.deltaT,
            epoch=tmplt.epoch)

def mismatch(vary_args, *fixed_args):

    # variable params
    f0, df, tau = vary_args

    #if (abs(f0 - waveform.fpeak)<100) and (1e-3 < tau < 5e-2):
    if (1500<f0<4000) and (1e-3 < tau < 5e-2):

        # fixed params
        distance, ra, dec, polarization, inclination = fixed_args

        # setup structures
        int_params = DampedChirpParams(f0=f0, tau=tau, df=df)

        ext_params = simsig.ExtParams( distance=distance, ra=ra, dec=dec,
                polarization=polarization, inclination=inclination, phase=0.0,
                geocent_peak_time=0.25)

        tmplt_td = damped_chirp_tmplt(det1_data, int_params, ext_params)

        try:
            match = pycbc.filter.match(tmplt_td,det1_data.td_signal,det1_data.psd,
                    low_frequency_cutoff=1000, high_frequency_cutoff=5000)[0]
        except ZeroDivisionError:
            match = 0

        return 1-match

    else:

        return 1

def match(int_params, ext_params):

    tmplt_td = damped_chirp_tmplt(det1_data, int_params, ext_params)

    try:
        match = pycbc.filter.match(tmplt_td,det1_data.td_signal,det1_data.psd,
                low_frequency_cutoff=flow, high_frequency_cutoff=fupp)[0]
    except ZeroDivisionError:
        match = 0

    return match


def fmin_wrap(det_data, int_params0, ext_params):

    x0 = [
            int_params0.f0,
            int_params0.df,
            int_params0.tau,
            ]


    fixed_args = (
            ext_params.distance,
            ext_params.ra,
            ext_params.dec,
            ext_params.polarization,
            ext_params.inclination
            )

    # Use mismatch for quick, easy maximisation over time / phase
    result_theory = optimize.minimize(mismatch, x0=x0, args=fixed_args,
            method='nelder-mead')

    return result_theory



# --------------------------------------------------------------------
# Data Generation

#
# Generate Signal Data
#

# Signal
print ''
print '--- %s ---'%sys.argv[1]
waveform = pmns_utils.Waveform('%s'%sys.argv[1])
waveform.compute_characteristics()

# Extrinsic parameters
ext_params = simsig.ExtParams(20.0, ra=0.0, dec=0.0,
        polarization=0.0, inclination=0.0, phase=0.0, geocent_peak_time=0.25)

# Frequency range for SNR around f2 peak - we'll go between 1, 5 kHz for the
# actual match calculations, though
flow=waveform.fpeak-150
fupp=waveform.fpeak+150

# Construct the time series for these params
waveform.make_wf_timeseries(theta=ext_params.inclination,
        phi=ext_params.phase)

#
# Generate IFO data
#

#print >> sys.stdout, "generating detector data objects..."

det1_data = simsig.DetData(det_site="H1", noise_curve='aLIGO',
        waveform=waveform, ext_params=ext_params, duration=0.5, seed=0,
        epoch=0.0, f_low=10.0)


# --------------------------------------------------------------------

#print 'broad band pre-conditioning snr: ', \
#        pycbc.filter.sigma(det1_data.td_signal,det1_data.psd,1000,5000)
#print 'f2 pre-conditioning snr: ', \
#        pycbc.filter.sigma(det1_data.td_signal,det1_data.psd,flow,fupp)


# --- high-pass
high_pass=1
knee=1000
if high_pass:
#    print >> sys.stdout, "applying high pass filter..."
    # Signal-only
    det1_data.td_signal = pycbc.filter.highpass(det1_data.td_signal, knee,
            filter_order=20, attenuation=0.9)

#print 'broad band high-passed snr: ', \
#        pycbc.filter.sigma(det1_data.td_signal,det1_data.psd,1000, 5000)
#print 'f2 high-passed snr: ', \
#        pycbc.filter.sigma(det1_data.td_signal,det1_data.psd,flow,fupp)

# --------------------------------------------------------------------
#
# The overlap calculations
#

# Initial guesses for intrinsic params for the max-match calculation
int_params0=DampedChirpParams(tau=.005, f0=waveform.fpeak, df=5000.0)

#print >> sys.stdout, "Attempting to maximise overlap..."

# --- minimise:
#for beta in np.arange(0,1,0.1):
result_theory = fmin_wrap(det1_data, int_params0, ext_params)

# --- resulting waveform:
int_params=DampedChirpParams(f0=result_theory['x'][0], df=result_theory['x'][1],
        tau=result_theory['x'][2])

result_wave=damped_chirp_tmplt(det1_data, int_params, ext_params)

#print result_theory
#print ''
print 'maximal match (broadband): ', 1-result_theory['fun']
p = result_wave.to_frequencyseries()
print 'maximal match (narrowband): ', match(int_params, ext_params)
print ''
print 'systematic frequency error from PSDs: ', \
        waveform.fpeak - p.sample_frequencies[np.argmax(abs(p.data)**2)]
print 'systematic frequency error from f, fpeak ', \
        waveform.fpeak - int_params.f0
print ''

Q = 2.0*np.pi*int_params.tau*int_params.f0
print 'f0=%.2f, df=%.2f, tau=%.2e, Q=%.2f'%(int_params.f0, int_params.df,
        int_params.tau, Q)

#print 'Result: ', result_theory

print ''

f=open('AsymChirplet_%s.txt'%sys.argv[1], "w")
f.writelines("%.2f %.2f %.2f %.2f %.2f %.2f %.2e %.2f\n"%(
    1-result_theory['fun'], match(int_params, ext_params), 
    waveform.fpeak - p.sample_frequencies[np.argmax(abs(p.data)**2)],
    waveform.fpeak - int_params.f0,
    int_params.f0, int_params.df, int_params.tau, Q))
f.close()


sys.exit()
# --------------------------------------------------------------------
# Diagnostic plots
pl.figure()
pl.plot(det1_data.td_signal.to_frequencyseries().sample_frequencies,
        abs(det1_data.td_signal.to_frequencyseries())\
                /max(abs(det1_data.td_signal.to_frequencyseries())),'k')
pl.plot(result_wave.to_frequencyseries().sample_frequencies,
        abs(result_wave.to_frequencyseries())\
                /max(abs(result_wave.to_frequencyseries())),'r')

f,a=pl.subplots(nrows=2)
a[0].plot(det1_data.td_signal.to_frequencyseries().sample_frequencies,
        np.real(det1_data.td_signal.to_frequencyseries())\
                /max(np.real(det1_data.td_signal.to_frequencyseries())),'k')
a[1].plot(det1_data.td_signal.to_frequencyseries().sample_frequencies,
        np.imag(det1_data.td_signal.to_frequencyseries())\
                /max(np.imag(det1_data.td_signal.to_frequencyseries())),'k')

a[0].plot(result_wave.to_frequencyseries().sample_frequencies,
        np.real(result_wave.to_frequencyseries())/max(np.real(result_wave.to_frequencyseries())),
        'r')
a[1].plot(result_wave.to_frequencyseries().sample_frequencies,
        np.imag(result_wave.to_frequencyseries())/max(np.imag(result_wave.to_frequencyseries()))
        ,'r')

a[0].set_ylim(-1,1)
a[1].set_ylim(-1,1)

pl.show()

