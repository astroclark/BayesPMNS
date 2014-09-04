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

from scipy import signal, optimize, special, stats

import pmns_utils
import pmns_simsig as simsig

import lal
import lalsimulation as lalsim
import pycbc.filter

import pylab as pl

import emcee
import mpmath


class DampedChirpParams:
    """
    """

    def __init__(self, tmplt_len=16384, Amp=1.0, f0=2000, tau=1e-3, df=0, phi0=0):
        self.tmplt_len=tmplt_len
        self.Amp=Amp
        self.f0=f0
        self.tau=tau
        self.df=df
        self.phi0=phi0

def damped_chirp_tmplt(det_data, int_params, ext_params):
    """
    Build a template for the detector in det_data from the intrinsic and
    extrinsic parameteres in int_params and ext_params
    """

    #
    # Compute polarisations for damped chirp
    #

#   # Get the epoch for the start of the time series
    epoch = ext_params.geocent_peak_time# - \

    # --- lalsimulation version
    Q = 2.0*np.pi*int_params.tau*int_params.f0
 
    hp, hc = lalsim.SimBurstSineGaussian(Q, int_params.f0,
            int_params.Amp, 0, 0, 1.0/16384)

 
    sg_center = 0.5*hp.data.length / 16384.0
    hplus = lal.CreateREAL8TimeSeries('hplus', epoch-sg_center, 0,
            det_data.td_noise.delta_t, lal.StrainUnit, hp.data.length)
    hplus.data.data=np.copy(hp.data.data)
    del hp
 
    hcross = lal.CreateREAL8TimeSeries('hcross', epoch-sg_center, 0,
            det_data.td_noise.delta_t, lal.StrainUnit, hc.data.length)
    hcross.data.data=np.copy(hc.data.data)
    del hc

 
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
    f0, tau = vary_args

    #if (abs(f0 - waveform.fpeak)<100) and (1e-3 < tau < 5e-2):
    if (1500<f0<4000) and (1e-3 < tau < 5e-2):

        # fixed params
        distance, ra, dec, polarization, inclination = fixed_args

        # setup structures
        int_params = DampedChirpParams(f0=f0, tau=tau)

        ext_params = simsig.ExtParams( distance=distance, ra=ra, dec=dec,
                polarization=polarization, inclination=inclination, phase=0.0,
                geocent_peak_time=0.25)

        tmplt_td = damped_chirp_tmplt(det1_data, int_params, ext_params)

        try:
            match = pycbc.filter.match(tmplt_td,det1_data.td_signal,det1_data.psd,
                    low_frequency_cutoff=2000, high_frequency_cutoff=5000)[0]
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
            int_params0.tau
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
#waveform = pmns_utils.Waveform('%s_lessvisc'%sys.argv[1])
waveform = pmns_utils.Waveform('%s_lessvisc'%sys.argv[1])
waveform.compute_characteristics()

# Extrinsic parameters
ext_params = simsig.ExtParams(float(sys.argv[2]), ra=0.0, dec=0.0,
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

#noise_curve='aLIGO'
#noise_curve='ligo3_basePSD'
noise_curve=sys.argv[3]
det1_data = simsig.DetData(det_site="H1", noise_curve=noise_curve,
        waveform=waveform, ext_params=ext_params, duration=0.5, seed=0,
        epoch=0.0, f_low=10.0, taper=True)


# --------------------------------------------------------------------


# --- high-pass
high_pass=0
knee=1000
if high_pass:
    #print >> sys.stdout, "applying high pass filter..."
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
int_params0=DampedChirpParams(tau=.005, f0=waveform.fpeak,
        Amp=max(abs(waveform.hplus.data.data)))

#print >> sys.stdout, "Attempting to maximise overlap..."

# --- minimise:
#for beta in np.arange(0,1,0.1):
result_theory = fmin_wrap(det1_data, int_params0, ext_params)

# --- resulting waveform:
int_params=DampedChirpParams( f0=result_theory['x'][0],
        tau=result_theory['x'][1])

result_wave=damped_chirp_tmplt(det1_data, int_params, ext_params)

#print result_theory
#print ''
print 'maximal match (broadband): ', 1-result_theory['fun']
sys.exit()
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

f=open('SG_%s.txt'%sys.argv[1], "w")
f.writelines("%.2f %.2f %.2f %.2f %.2f %.2f %.2e %.2f\n"%(
    1-result_theory['fun'], match(int_params, ext_params), 
    waveform.fpeak - p.sample_frequencies[np.argmax(abs(p.data)**2)],
    waveform.fpeak - int_params.f0,
    int_params.f0, int_params.df, int_params.tau, Q))
f.close()

result_wave=damped_chirp_tmplt(det1_data, int_params, ext_params)

# --------------------------------------------------------------------
#
# Frequency error estimation
#

#
# Find MaxL Time
#

# We need the correct time-offset to line the templates up.  We can get this
# from the SNR timeseries:
snr=pycbc.filter.matched_filter(result_wave, det1_data. td_signal,
        det1_data.psd, low_frequency_cutoff=1000, high_frequency_cutoff=5000)
tstart_idx=abs(snr).max_loc()[1]

# Now realign the template time-series
maxL_tmplt=pycbc.types.TimeSeries(initial_array=np.zeros(len(result_wave.data)),
      delta_t=result_wave.delta_t)
maxL_tmplt.data[tstart_idx:]=result_wave.data[0:-tstart_idx]

#
# Find MaxL Amplitude
#

# Finally, just get the amplitude by fitting this model signal
# (and divide by 1e-40 to handle small numbers)
idx=np.concatenate(np.argwhere((det1_data.fd_signal.sample_frequencies.data>1000) *
        (det1_data.fd_signal.sample_frequencies.data<5000)))
residsq = lambda Amp: 100*sum(abs(det1_data.fd_signal.data[idx] -
    Amp*maxL_tmplt.to_frequencyseries().data[idx])**2 / det1_data.psd.data[idx])

x0=max(abs(det1_data.td_signal.data))
result = optimize.minimize(residsq, x0=x0, method='nelder-mead')
print result
maxL_tmplt.data *= result['x']


#
# Compute Fisher error estimate
#

waveform_residuals = pycbc.types.TimeSeries(initial_array=det1_data.td_signal -
        maxL_tmplt, delta_t = det1_data.delta_t)
#waveform_residuals = pycbc.types.FrequencySeries(initial_array=det1_data.fd_signal -
#        maxL_tmplt.to_frequencyseries(), delta_f = det1_data.fd_signal.delta_f)

residuals_snrsq = pycbc.filter.sigmasq(waveform_residuals, psd=det1_data.psd,
        low_frequency_cutoff=1000, high_frequency_cutoff=5000)

deltafsq = (waveform.fpeak - int_params.f0)**2 / residuals_snrsq
print 'innner product delta = %.2f'%residuals_snrsq
print 'deltaF = %.2f'%np.sqrt(deltafsq)

pl.figure()
pl.plot(det1_data.td_signal.data)
pl.plot(maxL_tmplt)
pl.show()

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

#sys.exit()
#def amp_mismatch(Amp, vec1, vec2):
#    """
#    vec2 is time series to rescale, vec1 is the target
#    """
#
#    vec2_scaled=pycbc.types.TimeSeries(initial_array=Amp*vec2.data,
#            delta_t=vec2.delta_t)
#
#    overlap=pycbc.filter.overlap(vec1, vec2_scaled,
#                psd=det1_data.psd, low_frequency_cutoff=2000,
#                high_frequency_cutoff=5000, normalized=False)
#    optimal_overlap=pycbc.filter.overlap(vec1, vec1,
#                psd=det1_data.psd, low_frequency_cutoff=2000,
#                high_frequency_cutoff=5000, normalized=False)
#
#    amp_mm=1-overlap/optimal_overlap
#    print optimal_overlap
#
#    print amp_mm
#
#    return amp_mm
#
#result = optimize.minimize(amp_mismatch, x0=max(abs(det1_data.td_signal.data)),
#        args=(det1_data.td_signal, maxL_tmplt), method='nelder-mead')
#
#print result

