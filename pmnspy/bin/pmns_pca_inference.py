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
"""

from __future__ import division
import os,sys
import numpy as np
np.seterr(all="raise", under="ignore")
from optparse import OptionParser
import ConfigParser
import random
import time
import cPickle as pickle

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

import lal
import pycbc.filter
import pmns_utils
import pmns_simsig as simsig
import pmns_pca_utils as ppca
from pylal import antenna

from sklearn.neighbors.kde import KernelDensity

def parser():

    # --- Command line input
    parser = OptionParser()
    parser.add_option("-w", "--waveform-name", default="dd2_135135", type=str)
    parser.add_option("-D", "--fixed-distance", type=float, default=None)
    parser.add_option("-S", "--init-seed", type=int, default=101)
    parser.add_option("-N", "--ninject", type=int, default=None)
    parser.add_option("-o", "--output-dir", type=str, default=None)

    #parser.add_option("-Dmin", "--min-distance")
    #parser.add_option("-Dmax", "--max-distance")

    (opts,args) = parser.parse_args()

    # --- Read config file
    cp = ConfigParser.ConfigParser()
    cp.read(args[0])


    return opts, args, cp


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def PCtemplatePSD(theta):
    """
    Construct a template for the power spectrum from the beta-weighted linear
    superposition of the PCs in PCarray
    """
    hrss, fpeak, betas = theta

    # Build spectrum
    template_spectrum = np.copy(pca_result.pca['magnitude_mean'])
    for b, beta in enumerate(betas):
        template_spectrum += betas[b] * \
                pca_result.pca['magnitude_pca'].components_[b,:]

    # Re-align spectrum to this fpeak
    template_spectrum = ppca.shift_vec(template_spectrum,
            pca_result.sample_frequencies, fcenter=fpeak, fpeak=fcenter).real

    # Scale to this hrss
    hrss2 = hrss*hrss
    norm  = np.inner(template_spectrum, template_spectrum)
    template_spectrum *= hrss2/norm

    return template_spectrum

#
# Probability functions
#
def lnprob(theta, data, psd):
    """
    The log-posterior
    """
    lp = lnprior(theta)
    if not np.isinfinite(lp):
        return -np.inf
    return lp + lnlike(theta, data, psd)

from scipy.stats import ncx2
def lnlike(theta, data, psd):
    """
    The log-likelihood.  Non-central chiquared with 2 dof, noncentrality given
    by the template
    """

    # Construct the template (non-centrality), normalised by the noise variance
    template = PCtemplatePSD(theta) / psd

    # Normalise by the noise variance
    data_normed = data / psd

    # likelihood of each frequency
    lnlikelihoods = ncx2.logpdf(data_normed, 2, data_normed)

    inband = (pca_result.sample_frequencies>1000)*(pca_result<4000)
    return sum(lnlikelihoods[inband])

def lnprior(theta):
    """
    the log-prior probability of the parameters theta
    """
    hrss, fpeak, betas = theta
    if 1e-24 < hrss < 1e-19 and 1500 < fpeak < 4000 and (abs(betas)<0.1).all():
        # XXX: betas probably shouldn't be independent, surely???  We know that
        # the first PC should have a much larger coefficient than the second PC
        # etc...
        return 0.0
    return -np.inf


# --- End Defs

#####################################################################
# Input
USAGE='''%prog [options] [args]
Run external analysis programs on internally generated post-merger injections
with Gaussian noise.
'''

opts, args, cp = parser()

# Data configuration
datalen=cp.getfloat('analysis', 'datalength')
flow=cp.getfloat('analysis','flow')
srate=cp.getfloat('analysis','srate')

seed=opts.init_seed
seed+=random.randint(0,lal.GPSTimeNow())

epoch=0.0
trigtime=0.5*datalen+epoch

if opts.output_dir is None:
    outdir=cp.get('program', 'output-dir')

ts0=time.time()

######################################################################
# Data Generation

print >> sys.stdout, '''
-------------------------------------
Beginning pmns_pca_inference:

1) Generating Data
'''

#
# Generate Signal Data
#

print >> sys.stdout, "generating waveform..."
waveform = pmns_utils.Waveform('%s_lessvisc'%opts.waveform_name)
waveform.compute_characteristics()

te=time.time()
print >> sys.stdout, "...waveform construction took: %f sec"%(te-ts0)


# ------------------------------------------------------------------------------------
# --- Set up timing for injection: inject in center of segment
# Need to be careful.  The latest, longest template must lie within the data
# segment.   note also that the noise length needs to be a power of 2.

max_sig=100e-3
cbc_delta_t=10e-3 # std of gaussian with cbc timing error
time_prior_width=3*cbc_delta_t
max_sig_len=int(np.ceil(max_sig*16384))

# Sanity check: data segment must contain full extent of max signal
# duration, when the signal is injected in the middle of the segment
#if max_sig > 0.5*datalen:
    #print >> sys.stderr, "templates will lie outside data segment: extend data length"
    #sys.exit()

# ------------------------------------------------------------------------------------
# Begin Loop over injections (if required)
#seed+=random.randint(0,lal.GPSTimeNow())

if opts.ninject is not None:
    # Get distance from command line if present
    ninject=opts.ninject
else:
    # Read from config file
    ninject = cp.getint('injections', 'ninject')

for i in xrange( ninject ):

    # -----------------------------
    # --- Generate injection params
    #

    # Distance
    if opts.fixed_distance is not None:
        # Get distance from command line if present
        distance=opts.fixed_distance
    else:
        # Read from config file
        if cp.get('injections', 'dist-distr')=='fixed':
            distance = cp.getfloat('injections', 'fixed-dist')

    # Sky angles
    inj_ra  = -1.0*np.pi + 2.0*np.pi*np.random.random()
    inj_dec = -0.5*np.pi + np.arccos(-1.0 + 2.0*np.random.random())
    inj_pol = 2.0*np.pi*np.random.random()
    inj_inc = 0.5*(-1.0*np.pi + 2.0*np.pi*np.random.random())
    inj_phase = 2.0*np.pi*random.random()

    # Antenna response
    det1_fp, det1_fc, det1_fav, det1_qval = antenna.response(
            epoch, inj_ra, inj_dec, inj_inc, inj_pol, 
            'radians', cp.get('analysis', 'ifo1'))

    if cp.getboolean('injections', 'inj-overhead'):
        # set the injection distance to that which yields an effective distance
        # equal to the targeted fixed-dist
        inj_distance = det1_qval*distance
    else:
        inj_distance = np.copy(distance)

    # --- End injection params

    # -----------------------------------------------
    # --- Project waveform onto these extrinsic params
    # Extrinsic parameters
    ext_params = simsig.ExtParams(distance=inj_distance, ra=inj_ra, dec=inj_dec,
            polarization=inj_pol, inclination=inj_inc, phase=inj_phase,
            geocent_peak_time=trigtime)

    # Construct the time series for these params
    waveform.reproject_waveform(theta=ext_params.inclination,
            phi=ext_params.phase)

    # -----------------
    #
    # Generate IFO data
    #

    ts=time.time()
    print >> sys.stdout, "generating detector responses & noise..."

    det1_data = simsig.DetData(det_site=cp.get('analysis','ifo1'),
            noise_curve=cp.get('analysis','noise-curve'), waveform=waveform,
            ext_params=ext_params, duration=datalen, seed=seed, epoch=epoch,
            f_low=flow, taper=cp.getboolean('analysis','taper-inspiral'))

    # Compute optimal SNR for injection
    det1_optSNR=pycbc.filter.sigma(det1_data.td_signal, psd=det1_data.psd,
            low_frequency_cutoff=flow, high_frequency_cutoff=0.5*srate)


    te=time.time()

    print >> sys.stdout, "Injected SNR=%.2f"%det1_optSNR
    print >> sys.stdout, "...data generation took %f sec"%(te-ts)
    print >> sys.stdout, "Total elapsed time: %f sec"%(te-ts0)

    ######################################################################
    # Initialise Principal Components
    print >> sys.stdout, '''
2) Initialising PCA
    '''


    waveform_names=['apr_135135_lessvisc',
                    'shen_135135_lessvisc',
                    'dd2_135135_lessvisc' ,
                    'dd2_165165_lessvisc' ,
                    'nl3_135135_lessvisc' ,
                    'nl3_1919_lessvisc'   ,
                    'tm1_135135_lessvisc' ,
                    'tma_135135_lessvisc' ,
                    'sfhx_135135_lessvisc',
                    'sfho_135135_lessvisc']#,
    catlen = len(waveform_names)

    #
    # Create PMNS PCA instance for this catalogue
    #
    fcenter=2710
    pca_result = ppca.pmnsPCA(waveform_names, low_frequency_cutoff=1000,
            fcenter=fcenter, nTsamples=len(det1_data.td_response))

    te=time.time()
    print >> sys.stdout, "...PCA took: %f sec"%(te-ts0)


    ######################################################################
    # PCA-based Spectral Analysis

    print >> sys.stdout, '''
-------------------------------------
3) Fitting spectra
    '''

    # XXX: READY TO MCMCMCMCMCMCMCMCMCMC



print >> sys.stdout, "FINISHED; total elapsed time: %f"%(te-ts0)





