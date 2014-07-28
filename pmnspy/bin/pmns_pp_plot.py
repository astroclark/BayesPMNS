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
from matplotlib import pyplot as pl
#matplotlib.use("Agg")

#from scipy import signal, optimize, special, stats
from scipy import optimize
from scipy.stats.mstats import mquantiles

import pmns_utils
#import pmns_simsig as simsig

#import lal
#import lalsimulation as lalsim
#import pycbc.filter

import triangle

import cPickle as pickle


import glob

def compute_efficiency(k,N,b=True):

    if b:
        # Bayesian treatment
        classify_efficiencyilon=(k+1)/(N+2)
        stdev_classify_efficiencyilon=np.sqrt(classify_efficiencyilon*(1-classify_efficiencyilon)/(N+3))
    else:
        # Binomial treatment
        if N==0:
            classify_efficiencyilon=0.0
            stdev_classify_efficiencyilon=0.0
        else:
            classify_efficiencyilon=k/N
            stdev_classify_efficiencyilon=np.sqrt(classify_efficiencyilon*(1-classify_efficiencyilon)/N)
    return (classify_efficiencyilon,stdev_classify_efficiencyilon)

def sigmoid(x,width,asymmetry,x50):

    if x[0]>x[1]:
        y = 1 - 1 / (1 + (x / x50)**(-(1 + asymmetry**np.tanh(x/x50)) / width))
    elif x[0]<x[1]:
        y = 1 / (1 + (x / x50)**(-(1 + asymmetry**np.tanh(x/x50)) / width))
    else:
        print "error in sigmoid(); first and second x-values equal"

    return y

def fit_sigmoid(x,y,yerr):

    # initial guesses
    width=0.5*(max(x)-min(x))
    asymmetry=1.0
    x50=x[abs(y-0.5)==min(abs(y-0.5))][0]
    e0 = [width, asymmetry, x50]

    # fit sigmoid
    try:
        sigparams, sigparams_cov = optimize.curve_fit(sigmoid, x, y, e0,
                sigma=yerr)
    except RuntimeError:
        sigparams = e0
        sigparams_cov = np.zeros(shape=(len(e0),len(e0)))

    return sigparams, sigparams_cov

def freq_prior_points(a=1500, b=4000, N=1e4):
    median = a+(b-a) / 2.0
    width  = 0.68*(b-a)
    return median, width

# -------------------------------
# Load results

datafiles = sys.argv[1]

# XXX: Hardcoding
distances=np.array([5, 10, 15, 20])
wf = pmns_utils.Waveform('dd2_135135_lessvisc')
wf.compute_characteristics()

logBs = []
freq_pdfs = []
maxfreqs = []
freqintvals = []
freqwidths = []

for d,datafile in enumerate(datafiles.split(',')):

    # --- Data extraction
    print >> sys.stdout, "loading %s..."%datafile
    freq_axis, freq_pdfs_tmp, freq_estimates, all_sig_evs, all_noise_evs =\
            pickle.load(open(datafile))

    # Bayes factors at each distance
    logBs.append(all_sig_evs - all_noise_evs)

    # Frequency recovery at each distance
    freq_pdfs.append(freq_pdfs_tmp)

    maxfreqs.append(freq_estimates[0])
    freqintvals.append(np.array([freq_estimates[1], freq_estimates[2]]))
    freqwidths.append(freq_estimates[2] - freq_estimates[1])


# -------------------------------
# Computation & plots
import mpmath
odds2pos = lambda logB: 1.0/(1.0+1.0/np.exp(logB))

logBthresh=np.interp(0.9, odds2pos(np.arange(0, 10, 1e-4)), np.arange(0, 10,
    1e-4))
logBthresh=0.0

eps = np.zeros(len(datafiles.split(',')))
delta_eps = np.zeros(len(datafiles.split(',')))

freqerrs = []
freqerrs_found = []
freqwidths_found = []
freq_pdfs_found = []

for b in xrange(len(distances)):

    # --- Efficiency Calculation
    k = sum(logBs[b]>logBthresh)
    N = len(logBs[b])
    eps[b], delta_eps[b] = compute_efficiency(k,N)

    # --- Frequency Recovery
    if N>=10:
        freqerrs.append((maxfreqs[b]-wf.fpeak))

    # only include found things where Nfound is > 10
    if k>=10:
        found_indices=np.concatenate(np.argwhere(logBs[b]>logBthresh))
        freqerrs_found.append((maxfreqs[b][found_indices]-wf.fpeak))
        freqwidths_found.append(freqwidths[b][found_indices])
        freq_pdfs_found.append(freq_pdfs[b])


# -------------------------------
# Plots


"""
Plots to make:

1) logB vs Distance (box plot)
2) 'efficiency' vs Distance (efficiency: logB>thresh)
3) frequency error vs distance, error bars with 1-sigma conf interval
"""

# --- Bayes facotrs
bayesboxfig, bayesboxax = pl.subplots()
bayesbp = pl.boxplot(logBs, notch=True)
pl.setp(bayesbp['boxes'], color='black')
pl.setp(bayesbp['whiskers'], color='black')
bayesboxax.set_ylabel('log B')
bayesboxax.set_xlabel('Distance [Mpc]')
pl.axhline(logBthresh,color='r')
pl.setp(bayesboxax, xticklabels=distances)

pl.savefig('bayes-boxes.png')
pl.savefig('bayes-boxes.eps')

# --- Efficiency
epsfig, epsax = pl.subplots()

sigparams = fit_sigmoid(distances[::-1], eps[::-1], delta_eps[::-1])
dist_fit = np.arange(0, 20, 0.1)[::-1]
eps_fit = sigmoid(dist_fit,*sigparams[0]) 

epsp = pl.errorbar(distances, eps, yerr=delta_eps, linestyle='none', color='r')
epsfitp = pl.plot(dist_fit, eps_fit, color='k')

epsax.set_xlabel('Distance [Mpc]')
epsax.set_ylabel('Efficiency')
epsax.set_ylim(0,1)
epsax.set_xlim(0,25)

pl.savefig('efficiency_logB-%.2f.png'%logBthresh)
pl.savefig('efficiency_logB-%.2f.eps'%logBthresh)

# --- Frequency recovery: all PDFs

for i in xrange(len(distances)):
    pdffig, pdfax = pl.subplots()

    im=pdfax.imshow(np.transpose(freq_pdfs[i]), aspect='auto',
            interpolation='nearest', extent=np.transpose([1500, 4000, 0,
                np.shape(freq_pdfs[i])[1]-1]), cmap='gnuplot', vmin=0,
            vmax=0.01)
    pdfax.set_xlabel('Frequency [Hz]')
    pdfax.set_ylabel('Injection #')
    #pdfax.set_xlim(2500,2700)
    pdfax.set_title('Distance = %.2f Mpc'%distances[i])
    pdfax.axvline(wf.fpeak, color='r', linestyle='--')
    colbar=pdffig.colorbar(im)
    colbar.set_label('p(f|D)')

    pl.savefig('freqpdfs_dist-%.2f.png'%distances[i])
    pl.savefig('freqpdfs_dist-%.2f.eps'%distances[i])



# --- Frequency recovery: errors
"""
a) boxplot of frequency error for found and all injections
"""

# expectations from prior:
prior_median_f, prior_f_width = freq_prior_points()
exp_err = (wf.fpeak-prior_median_f)

# Inter-quartile range
x=1500+2500*np.random.rand(1e5)
q=mquantiles(x,prob=[0.25, 0.75])
exp_err_low = (wf.fpeak - q[0])
exp_err_upp = (q[1] - wf.fpeak)


# - All
freqerrboxfig, freqerrboxax = pl.subplots() 

freqerrbp=freqerrboxax.boxplot(freqerrs, notch=True, bootstrap=1000)
pl.axhline(exp_err_low, color='r', linestyle='-')
pl.axhline(exp_err, color='r', linestyle='--')
pl.axhline(exp_err_upp, color='r', linestyle='--')

#freqerrboxax.set_yscale('log')
#freqerrboxax.set_ylim(1e-3, 1e4)
pl.setp(freqerrbp['boxes'], color='black')
pl.setp(freqerrbp['whiskers'], color='black')
freqerrboxax.set_title('All Injections')
freqerrboxax.set_ylabel('Frequency Error [Hz]')
freqerrboxax.set_xlabel('Distance [Mpc]')
pl.setp(freqerrboxax, xticklabels=distances)

pl.savefig('freqerr.png')
pl.savefig('freqerr.eps')

# - 'Found'
freqerrboxfig, freqerrboxax = pl.subplots() 

freqerrbp = freqerrboxax.boxplot(freqerrs_found, notch=True, bootstrap=1000)
pl.axhline(exp_err_low, color='r', linestyle='-')
pl.axhline(exp_err, color='r', linestyle='--')
pl.axhline(exp_err_upp, color='r', linestyle='--')

#freqerrboxax.set_yscale('log')
#freqerrboxax.set_ylim(1e-3, 1e4)
pl.setp(freqerrbp['boxes'], color='black')
pl.setp(freqerrbp['whiskers'], color='black')
freqerrboxax.set_title('Injections with log B>%.2f'%logBthresh)
freqerrboxax.set_ylabel('Frequency Error [Hz]')
freqerrboxax.set_xlabel('Distance [Mpc]')
pl.setp(freqerrboxax, xticklabels=distances)

pl.savefig('freqerr_logB-%.2f.png'%logBthresh)
pl.savefig('freqerr_logB-%.2f.eps'%logBthresh)

# --- Frequency recovery: widths
"""
a) boxplot of frequency width for found and all injections
"""
# - 'All'
freqwidthboxfig, freqwidthboxax = pl.subplots() 

freqwidthbp=freqwidthboxax.boxplot(freqwidths, notch=True, bootstrap=1000)
pl.axhline(prior_f_width, color='r', linestyle='--')

freqwidthboxax.set_ylim(0,3000)
pl.setp(freqwidthbp['boxes'], color='black')
pl.setp(freqwidthbp['whiskers'], color='black')
freqwidthboxax.set_title('All Injections')
freqwidthboxax.set_ylabel('Frequency width [Hz]')
freqwidthboxax.set_xlabel('Distance [Mpc]')
pl.setp(freqwidthboxax, xticklabels=distances)

pl.savefig('freqwidth.png')
pl.savefig('freqwidth.eps')

# - 'Found'
freqwidthboxfig, freqwidthboxax = pl.subplots() 

freqwidthbp=freqwidthboxax.boxplot(freqwidths_found, notch=True, bootstrap=1000)
pl.axhline(prior_f_width, color='r', linestyle='--')

freqwidthboxax.set_ylim(0, 3000)
pl.setp(freqwidthbp['boxes'], color='black')
pl.setp(freqwidthbp['whiskers'], color='black')
freqwidthboxax.set_title('Injections with log B>%.2f'%logBthresh)
freqwidthboxax.set_ylabel('Frequency width [Hz]')
freqwidthboxax.set_xlabel('Distance [Mpc]')
pl.setp(freqwidthboxax, xticklabels=distances)

pl.savefig('freqwidth_logB-%.2f.png'%logBthresh)
pl.savefig('freqwidth_logB-%.2f.eps'%logBthresh)

#pl.show()


    


