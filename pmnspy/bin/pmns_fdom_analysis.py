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

def damped_chirp(tmplt_len, Amp, f0, tau, df, phi0=0):
    """
    """
    # Get time stamps for non-zero part of signal
    time_axis = np.arange(0, tmplt_len / 16384.0, 1/16384.0)

    # Frequency evolution
    f1 = f0+df

    cycles = signal.chirp(time_axis, f0=f0, t1=tau, f1=f1, method='linear',
            phi=phi0)

    decay = np.exp(-0.5*(time_axis/tau)**2)

    return time_axis, np.real(Amp*cycles*decay)

class DampedChirpParams:
    """
    """

    def __init__(self, tmplt_len, Amp, f0, tau, df):
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

    _, hp = damped_chirp(int_params.tmplt_len,
            int_params.Amp, int_params.f0, int_params.tau, int_params.df)

    _, hc = damped_chirp(int_params.tmplt_len,
            int_params.Amp, int_params.f0, int_params.tau, int_params.df,
            phi0=90.0)

    # Get the epoch for the start of the time series
    epoch = ext_params.geocent_peak_time# - \
            #np.argmax(hp)*det_data.td_response.delta_t
            # XXX: why don't we need to subtract this bit...?

    try:
        time_delay = lal.TimeDelayFromEarthCenter(det_data.det_site.location,
                ext_params.ra, ext_params.dec, ext_params.geocent_peak_time)
    except RuntimeError:
        time_delay = lal.TimeDelayFromEarthCenter(det_data.det_site.location,
                ext_params.ra, ext_params.dec, -1e4)


    # --- Put the polarisations into LAL TimeSeries
    hplus = lal.CreateREAL8TimeSeries('hplus', epoch, 0,
            det_data.td_noise.delta_t, lal.StrainUnit, len(hp))
    hplus.data.data=np.copy(hp)
    del hp

    hcross = lal.CreateREAL8TimeSeries('hcross', epoch, 0,
            det_data.td_noise.delta_t, lal.StrainUnit, len(hc))
    hcross.data.data=np.copy(hc)
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

    # fixed params
    #Amp, f0, tau, df, geocent_peak_time = vary_args
    Amp, f0, tau, geocent_peak_time = vary_args

    # variable params
    #distance, ra, dec, polarization, inclination, phase = fixed_args
    df, distance, ra, dec, polarization, inclination, phase = fixed_args

    # setup structures
    int_params = DampedChirpParams(max_sig_len, Amp, f0, tau, df)

    ext_params = simsig.ExtParams( distance=distance, ra=ra, dec=dec,
            polarization=polarization, inclination=inclination, phase=phase,
            geocent_peak_time=geocent_peak_time )

    tmplt_td = damped_chirp_tmplt(det1_data, int_params, ext_params)

    try:
        match = pycbc.filter.match(tmplt_td,det1_data.td_signal,det1_data.psd,
                low_frequency_cutoff=1000, high_frequency_cutoff=5000)[0]
    except ZeroDivisionError:
        match = 0

    return 1-match

#def lnprob(vary_args, *fixed_args):
def logl(vary_args, *fixed_args):

    # variable params
    #Amp, f0, tau, df, geocent_peak_time = vary_args
    Amp, f0, tau, geocent_peak_time = vary_args

    # fixed params
    df, distance, ra, dec, polarization, inclination, phase = fixed_args

    # set up structures
    int_params = DampedChirpParams(max_sig_len, Amp, f0, tau, df)

    ext_params = simsig.ExtParams( distance=distance, ra=ra, dec=dec,
            polarization=polarization, inclination=inclination, phase=phase,
            geocent_peak_time=geocent_peak_time )

    # Generate time-domain template
    tmplt_td = damped_chirp_tmplt(det1_data, int_params, ext_params)

    data_fd  = det1_data.fd_response
    tmplt_fd = tmplt_td.to_frequencyseries()


    # --- Phase marginalised: T1300326, Veitch & del Pozzo
    if 1:
        dd = abs(pycbc.filter.overlap(data_fd, data_fd, psd=det1_data.psd,
                low_frequency_cutoff=1000, high_frequency_cutoff=5000,
                normalized=False))

        hh = abs(pycbc.filter.overlap(tmplt_fd, tmplt_fd, psd=det1_data.psd,
                low_frequency_cutoff=1000, high_frequency_cutoff=5000,
                normalized=False))

        dh = abs(pycbc.filter.overlap(tmplt_fd, data_fd, psd=det1_data.psd,
                low_frequency_cutoff=1000, high_frequency_cutoff=5000,
                normalized=False))

        logl_val = - 0.5*(float(hh) + float(dd)) + \
                mpmath.log(mpmath.besseli(0, float(dh)))

    else:

        idx = (det1_data.fd_response.sample_frequencies.data>1000) * \
                (det1_data.fd_response.sample_frequencies.data<5000)

        TwoDeltaToverN = 2.0 * det1_data.td_response.delta_t / \
            len(det1_data.td_response.data)

        sigma = np.sqrt(det1_data.psd.data[idx])
        data_fd_norm  = data_fd.data[idx]/sigma
        tmplt_fd_norm = tmplt_fd.data[idx]/sigma

        dd = abs(np.dot(data_fd_norm, data_fd_norm.conjugate()))
        hh = abs(np.dot(tmplt_fd_norm, tmplt_fd_norm.conjugate()))
        dh = abs(np.dot(data_fd_norm, tmplt_fd_norm.conjugate()))

        logl_val = - 0.5*(float(hh) + float(dd)) + \
                mpmath.log(mpmath.besseli(0, float(dh)))

    return logl_val

def noise_ev(det1_data):

    # --- Phase marginalised: T1300326, Veitch & del Pozzo
    data_fd  = det1_data.fd_response
    if 1:
        dd = abs(pycbc.filter.overlap(data_fd, data_fd, psd=det1_data.psd,
                low_frequency_cutoff=1000, high_frequency_cutoff=5000,
                normalized=False))

        logl_val = - 0.5*float(dd)


    elif 0:
        idx = (det1_data.fd_response.sample_frequencies.data>1000) * \
                (det1_data.fd_response.sample_frequencies.data<5000)


        TwoDeltaToverN = 2.0 * det1_data.td_response.delta_t / \
            len(det1_data.td_response.data)

        dataReal = np.real(data_fd.data[idx]) #/ det1_data.td_response.delta_t
        dataImag = np.imag(data_fd.data[idx]) #/ det1_data.td_response.delta_t

        D = TwoDeltaToverN*sum( (dataReal*dataReal + dataImag*dataImag)/det1_data.psd.data[idx] )

        logl_val = -D

    else:

        idx = (det1_data.fd_response.sample_frequencies.data>1000) * \
                (det1_data.fd_response.sample_frequencies.data<5000)

        sigma = np.sqrt(det1_data.psd.data[idx])
        data_fd_norm  = data_fd.data[idx]/sigma

        dd = abs(np.dot(data_fd_norm,  data_fd_norm.conjugate()))
        logl_val = - 0.5*float(dd)

    return logl_val

def logpost(vary_args, *fixed_args):

    # variable params
    #Amp, f0, tau, df, geocent_peak_time = vary_args
    Amp, f0, tau, geocent_peak_time = vary_args

    # fixed params
    #distance, ra, dec, polarization, inclination, phase = fixed_args
    df, distance, ra, dec, polarization, inclination, phase = fixed_args

    int_params = DampedChirpParams(max_sig_len, Amp, f0, tau, df)

    ext_params = simsig.ExtParams( distance=distance, ra=ra, dec=dec,
            polarization=polarization, inclination=inclination, phase=phase,
            geocent_peak_time=geocent_peak_time )

    logprior = lnprior(vary_args)
    if logprior == -np.inf:
        return -np.inf

    logl_val = logl(vary_args, *fixed_args)

    return logl_val + logprior



def fmin_wrap(det_data, int_params0, ext_params):

    x0 = [
            int_params0.Amp,
            int_params0.f0,
            int_params0.tau,
            float(ext_params.geocent_peak_time)
            ]
            #int_params0.df,


    fixed_args = (
            int_params0.df,
            ext_params.distance,
            ext_params.ra,
            ext_params.dec,
            ext_params.polarization,
            ext_params.inclination,
            ext_params.phase
            )

    # Use mismatch for quick, easy maximisation over time / phase
    result_theory = optimize.minimize(mismatch, x0=x0, args=fixed_args,
            method='Nelder-Mead', options={'maxiter':1000})
    print result_theory
    print ''
    print 'maximal match: ', 1-result_theory['fun']
    print '(expected) systematic frequency error: ', result_theory['x'][1] - waveform.fpeak
    print ''
    #sys.exit()

    print 'noise evidence: ', noise_ev(det1_data)

    print 'initiating emcee'

    simple=False
    if simple:
        #ndim, nwalkers = 7, 1000
        ndim, nwalkers = 4, 200
        pos0 = np.zeros(shape=(nwalkers, ndim))

        print 'drawing initial samples...'
        pos0 = draw_init_samp(nwalkers)
        #pos0[:,1] = 2000+np.random.randn(np.shape(pos0)[0])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpost, args=(fixed_args),
                threads=4, a=1.5)
        #print 'burning...'
        #sampler.run_mcmc(pos0, 100)
        print 'sampling...'
        sampler.reset()
        sampler.run_mcmc(pos0, 100)

    else:
        ndim, nwalkers, ntemps = 4, 100, 10
        sampler = emcee.PTSampler(ntemps=ntemps, nwalkers=nwalkers, dim=ndim,
                logl=logl, logp=lnprior, loglargs=(fixed_args), threads=2,
                a=1.5)#, betas=np.logspace(0,np.log(1e-2), base=np.e, num=ntemps))

        pos0 = np.zeros(shape=(ntemps, nwalkers, ndim))

        print 'drawing initial samples...'
        for i in xrange(ntemps):
            pos0[i, :, :] = draw_init_samp(nwalkers)

        print 'burning...'
        for pos, lnprob, lnlike in sampler.sample(pos0, iterations=10):
            pass
        sampler.reset()

        print 'sampling...'
        for pos, lnprob, lnlike in sampler.sample(pos, lnprob0=lnprob,
                lnlike0=lnlike, iterations=10):
        #for pos, lnprob, lnlike in sampler.sample(pos0, iterations=10):
            pass


    return result_theory, sampler

def lnprior(x):
    #Amp, f0, tau, df, geocent_peak_time = x
    Amp, f0, tau, geocent_peak_time = x

    Amprange=[1e-23, 1e-20]
    f0range=[1500, 4000]
    taurange=[1e-3, max_sig / 3.0]
    #timerange=[0.05, 0.2]
    timerange=ext_params.geocent_peak_time + [- 0.5*3*time_prior_width,
            0.5*3*time_prior_width]

    if (min(f0range)<f0<max(f0range)) and (min(taurange)<tau<max(taurange)) \
            and (min(timerange)<geocent_peak_time<max(timerange)) \
            and (min(Amprange) < Amp < max(Amprange)):
        #return 0.0

        #amp_prior = -1*np.log(np.diff(Amprange))

        #ampnorm = lambda a, b: np.log(b/a)
        #amp_prior = -1*np.log(Amp)

        ampnorm = lambda a, b: (b-a)/(a*b)
        amp_prior = -np.log(ampnorm(1e-23,1e-20))-2*np.log(Amp)

        freq_prior = -1*np.log(np.diff(f0range))
        tau_prior = -1*np.log(np.diff(taurange))
        time_prior = -1*np.log(np.diff(timerange))

        return amp_prior + freq_prior + tau_prior + time_prior

    else:
        return -np.inf


def draw_init_samp(N):
#    import scipy.stats as stats
    #a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    trunclims = lambda low,mean,std,upp: \
            ((low - mean) / std, (upp - mean) / std)

    # --- Amp
    #amp_samp = 10**(-22 + (-20 - -22)*np.random.rand(N))
    amp_samp = 1e-23 + (1e-20-1e-23)*np.random.rand(N)

    # --- Freq
    freq_samp = 1500 + (4000 - 1500) * np.random.rand(N)
    #freq_samp = 3000 + 10*np.random.randn(N)

    # --- tau
    tau_samp = 1e-3 + (max_sig / 3.0 - 1e-3) * np.random.rand(N)

    # --- df
    #df_samp = 0.0 + 10*np.random.randn(N)

    # --- time
    center=ext_params.geocent_peak_time
    upper=center-3*cbc_delta_t
    lower=center+3*cbc_delta_t
    time_prior_sigma = cbc_delta_t,
    a, b = trunclims(lower, center,  time_prior_sigma, upper)
    time_samp=stats.truncnorm.rvs(a, b, loc=center, scale=time_prior_sigma, size=N)


    return np.transpose(np.array([amp_samp, freq_samp, tau_samp, time_samp]))


# --------------------------------------------------------------------
# Data Generation

#
# Generate Signal Data
#

# Signal
print 'generating waveform...'
waveform = pmns_utils.Waveform('dd2_135135_lessvisc')
waveform.compute_characteristics()

# Extrinsic parameters
ext_params = simsig.ExtParams(distance=float(sys.argv[1]), ra=0.0, dec=0.0,
        polarization=0.0, inclination=0.0, phase=0.0, geocent_peak_time=0.25)

# Construct the time series for these params
waveform.make_wf_timeseries(theta=ext_params.inclination,
        phi=ext_params.phase)

#
# Generate IFO data
#
# Need to be careful.  The latest, longest template must lie within the data
# segment.   note also that the noise length needs to be a power of 2.

max_sig=100e-3
cbc_delta_t=10e-3 # std of gaussian with cbc timing error
time_prior_width=3*cbc_delta_t
max_sig_len=int(np.ceil(max_sig*16384))

datalen=2**(np.ceil(np.log2( 2*(time_prior_width + max_sig) )))
print >> sys.stdout, "using datalen=", datalen

# Make sure we inject the signal in the middle!
ext_params.geocent_peak_time = 0.5*datalen

#seed=np.random.random_integers(1000)
seed=int(sys.argv[2])

# XXX Sanity check: data segment must contain full extent of max signal duration
if max_sig > 0.5*datalen:
    print >> sys.stderr, "templates will lie outside data segment: extend data length"
    sys.exit()

print >> sys.stdout, "generating detector responses & noise..."

det1_data = simsig.DetData(det_site="H1", noise_curve='aLIGO', waveform=waveform,
        ext_params=ext_params, duration=datalen, seed=seed, epoch=0.0,
        f_low=10.0)

# --------------------------------------------------------------------
# Data Conditioning

# --------------------------------------------------------------------
#
# XXX: Validation... 
#

validate=True
if validate:

    #ext_params.geocent_peak_time=0.0
    int_params_test=DampedChirpParams(tmplt_len=max_sig_len, Amp=1, tau=.005,
            df=0., f0=2000)

    targetSNR=float(sys.argv[1])

    # Make new signal at desired SNR
    det1_data.td_signal = damped_chirp_tmplt(det1_data, int_params_test, ext_params)

    currentSNR = pycbc.filter.sigma(det1_data.td_signal, psd=det1_data.psd,
            low_frequency_cutoff=1000)

    det1_data.td_signal.data *= targetSNR/currentSNR

    det1_data.td_response = damped_chirp_tmplt(det1_data, int_params_test, ext_params) 
    det1_data.td_response.data *= targetSNR/currentSNR

    # add the old noise to the new signal
    det1_data.td_response.data += det1_data.td_noise.data

    det1_data.make_fdomain()

# --------------------------------------------------------------------

print 'pre-conditioning snr: ', \
        pycbc.filter.sigma(det1_data.td_signal,det1_data.psd,1000,5000)


# --- high-pass
high_pass=1
knee=1000
if high_pass:
    print >> sys.stdout, "applying high pass filter..."
    # Signal-only
    det1_data.td_signal.data = pycbc.filter.highpass(det1_data.td_signal, knee,
            filter_order=20, attenuation=0.9)
    # Signal-plus-noise
    det1_data.td_response.data = pycbc.filter.highpass(det1_data.td_response,
            knee, filter_order=20, attenuation=0.9)

print 'high-passed snr: ', \
        pycbc.filter.sigma(det1_data.td_signal,det1_data.psd,1000,5000)

# --------------------------------------------------------------------
# Plot data
if 0:
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
# --------------------------------------------------------------------
# Parameter Estimation!

#
# Preliminary estimate using scipy.optimize
#

# Initial guesses for intrinsic params for the max-match calculation
if validate:
    int_params0 = int_params_test
else:
    int_params0=DampedChirpParams(tmplt_len=max_sig_len,
        Amp=max(abs(det1_data.td_response.data)), tau=.005, df=0.,
        f0=(5000-1000)*np.random.rand()+1000)

print >> sys.stdout, "Attempting to maximise overlap..."

result_theory, sampler = fmin_wrap(det1_data, int_params0, ext_params)

print 'signal evidence (at %.2f Mpc): %f (+/- %f)'%(\
        ext_params.distance,
        sampler.thermodynamic_integration_log_evidence(fburnin=0.2)[0],
        sampler.thermodynamic_integration_log_evidence(fburnin=0.2)[1])

print 'signal vs noise: ', \
        sampler.thermodynamic_integration_log_evidence(fburnin=0.2)[0]-noise_ev(det1_data)

samples = sampler.chain[0,...]
samples[:,0]=np.log10(samples[:,0])

#hev=np.log(stats.hmean(np.exp(np.concatenate(sampler.lnprobability[0,:,:]))))
#print 'signal evidence (hmean): ', hev
#print 'signal vs noise (hmean): ', hev - noise_ev(det1_data)
#pl.close('all')


#import triangle
#fig = triangle.corner(samples, labels=["$A$", "$f_0$", "$\\tau$", "$t_0$"])#,
        #truths=[m_true, b_true, np.log(f_true)])

#pl.savefig('corner_%d.ps'%seed)

import cPickle as pickle
if validate:
    pickle.dump([samples,
        sampler.thermodynamic_integration_log_evidence(fburnin=0)[0],
        noise_ev(det1_data)], open("results_snr-%s_seed-%s_validate.pickle"%(sys.argv[1],
            sys.argv[2]), "wb"))
else:
    pickle.dump([samples,
        sampler.thermodynamic_integration_log_evidence(fburnin=0)[0],
        noise_ev(det1_data)], open("results_dist-%s_seed-%s.pickle"%(sys.argv[1],
            sys.argv[2]), "wb"))

#
# Get timing info - we'll use this for windowing and, later, for the search
#
#h_time_delay = lal.TimeDelayFromEarthCenter(det1_data.det_site.location,
#        ext_params.ra, ext_params.dec, ext_params.geocent_peak_time)

#h_peak_time = ext_params.geocent_peak_time + h_time_delay

#h_peak_idx = np.argmin(abs(det1_data.td_signal.sample_times.data-h_peak_time))


