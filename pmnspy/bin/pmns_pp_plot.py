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

import cPickle as pickle
import glob

#from scipy import signal, optimize, special, stats
from scipy import optimize,stats
from scipy.stats.mstats import mquantiles

import pmns_utils

#import triangle


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


waveform_name=sys.argv[1]
datafiles=sys.argv[2]

# XXX: Hardcoding
distances=np.array([5, 7.5, 10])

wf = pmns_utils.Waveform(waveform_name+'_lessvisc')
wf.compute_characteristics()

logBs = []
netSNRs = []
freq_pdfs = []
maxfreqs = []
meanfreqs = []
credintvals = []
credintwidths = []

confintvals=[]
confintwidths=[]

for d,datafile in enumerate(datafiles.split(',')):

    # --- Data extraction
    print >> sys.stdout, "loading %s..."%datafile

    (this_logBs, this_netSNRs, this_freq_pdfs, freq_axis, this_freq_maxL,
            this_freq_low, this_freq_upp, this_freq_area) = \
                    pickle.load(open(datafile))

    # Bayes factors at each distance
    logBs.append(this_logBs)
    netSNRs.append(this_netSNRs)

    # Frequency recovery at each distance
    freq_pdfs.append(this_freq_pdfs)

    # Compute freq. expectation value
    #mf=np.zeros(shape=np.shape(freq_pdfs_tmp)[1])
    #for m in xrange(np.shape(freq_pdfs_tmp)[1]):
    #    mf[m] = np.trapz(freq_pdfs_tmp[:,m]*freq_axis, freq_axis)
    #meanfreqs.append(mf)

    # Bayesian credible intervals and MAP estimate
    maxfreqs.append(this_freq_maxL)
    credintvals.append(np.array([this_freq_low, this_freq_upp]))
    credintwidths.append(this_freq_upp - this_freq_low)

    # 1-sigma Frequentist confidence intervals for MAP estimate
    confintvals.append(stats.scoreatpercentile(maxfreqs[d], [15.87,84.13]))
    confintwidths.append(confintvals[d][1]-confintvals[d][0])

# -------------------------------
# Computation & plots
#import mpmath
odds2pos = lambda logB: 1.0/(1.0+1.0/np.exp(logB))
#logBthresh=np.interp(0.9, odds2pos(np.arange(0, 10, 1e-4)), np.arange(0, 10,
#    1e-4))

logBthresh=0.0

eps = np.zeros(len(datafiles.split(',')))
delta_eps = np.zeros(len(datafiles.split(',')))

freqerrs = []
freqerrs_found = []
credintwidths_found = []
freq_pdfs_found = []

maxfreqs_found=[]
meanfreqs_found=[]

confintvals_found=[]
confintwidths_found=[]

found_distances=[]

# --- Efficiency/missed/found loop
for b in xrange(len(distances)):

    freq_pdfs_found_tmp = []

    # --- Efficiency Calculation
    #delta_f = abs(maxfreqs[b]-wf.fpeak)
    #k = sum(delta_f<10)
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
        credintwidths_found.append(credintwidths[b][found_indices])

        for i in found_indices:
            freq_pdfs_found_tmp.append(freq_pdfs[b][i])
        freq_pdfs_found.append(freq_pdfs_found_tmp)

        maxfreqs_found.append(maxfreqs[b][found_indices])
        #meanfreqs_found.append(meanfreqs[b][found_indices])

        found_distances.append(distances[b])

        # 1-sigma Frequentist confidence intervals for MAP estimate
        confintvals_found.append(stats.scoreatpercentile(maxfreqs[b][found_indices],
            [15.87,100-15.87]))
        confintwidths_found.append(confintvals_found[b][1]-confintvals_found[b][0])


# XXX: testing

# -------------------------------
# Plots

"""
Plots to make:

1) logB vs Distance (box plot)
2) 'efficiency' vs Distance (efficiency: logB>thresh)
3) frequency error vs distance, error bars with 1-sigma cred interval
"""

# --- Bayes factors
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

epsp = pl.errorbar(distances, eps, yerr=delta_eps, ecolor='r', color='k')
#epsp = pl.errorbar(distances, eps, yerr=delta_eps, linestyle='none', color='r')
#epsfitp = pl.plot(dist_fit, eps_fit, color='k')

epsax.set_xlabel('Distance [Mpc]')
epsax.set_ylabel('Fraction with log B>%.2f'%logBthresh)
epsax.minorticks_on()
epsax.grid(linestyle='--')
epsax.set_ylim(0,1)
epsax.set_xlim(0,25)

pl.savefig('efficiency_logB-%.2f.png'%logBthresh)
pl.savefig('efficiency_logB-%.2f.eps'%logBthresh)

# --- Frequency recovery: all PDFs

for d in xrange(len(distances)):
#   pdffig, pdfax = pl.subplots()
#
#   im=pdfax.imshow(np.transpose(freq_pdfs[i]), aspect='auto',
#           interpolation='nearest', extent=np.transpose([1500, 4000, 0,
#               np.shape(freq_pdfs[i])[1]-1]), cmap='gnuplot', vmin=0,
#           vmax=0.01)
#   pdfax.set_xlabel('Frequency [Hz]')
#   pdfax.set_ylabel('Injection #')
#   #pdfax.set_xlim(2500,2700)
#   pdfax.set_title('Distance = %.2f Mpc'%distances[i])
#   pdfax.axvline(wf.fpeak, color='r', linestyle='--')
#   colbar=pdffig.colorbar(im)
#   colbar.set_label('p(f|D)')
#
#   pl.savefig('freqpdfs_dist-%.2f.png'%distances[i])
#   pl.savefig('freqpdfs_dist-%.2f.eps'%distances[i])
#

    pdflines, pdflinesax = pl.subplots()
    for i in xrange(np.shape(freq_pdfs[d])[0]):
        pdflinesax.plot(freq_axis,freq_pdfs[d][i], color='grey', linewidth=0.001)
    pdflinesax.set_yscale('log')
    pdflinesax.set_ylim(1e-5,1) 
    pdflinesax.set_xlabel('Frequency [Hz]')
    pdflinesax.set_ylabel('p(f|D)')
    pdflinesax.axvline(wf.fpeak, color='r', linestyle='--')
    pdflinesax.axhline(1.0/(4000-1500), color='k', linestyle='--')
    pdflinesax.set_title('Distance = %.2f Mpc'%distances[d])
 
    pl.savefig('freqpdflines_dist-%.2f.png'%distances[d])
    pl.savefig('freqpdflines_dist-%.2f.eps'%distances[d])
 

    pdflines, pdflinesax = pl.subplots()
    for i in xrange(np.shape(freq_pdfs_found[d])[0]):
        pdflinesax.plot(freq_axis,freq_pdfs_found[d][i], color='grey', linewidth=0.001)
    pdflinesax.set_yscale('log')
    pdflinesax.set_ylim(1e-5,1) 
    pdflinesax.set_xlabel('Frequency [Hz]')
    pdflinesax.set_ylabel('p(f|D)')
    pdflinesax.axvline(wf.fpeak, color='r', linestyle='--')
    pdflinesax.axhline(1.0/(4000-1500), color='k', linestyle='--')
    pdflinesax.set_title('log B > %.2f, Distance = %.2f Mpc'%(logBthresh,
        distances[d]))

    pl.savefig('freqpdflines_dist-%.2f_logB-%.2f.png'%(distances[d], logBthresh))
    pl.savefig('freqpdflines_dist-%.2f_logB-%.2f.eps'%(distances[d], logBthresh))



# --- Frequency recovery: errors

# Inter-quartile range
x=1500+2500*np.random.rand(1e5)
q=mquantiles(x,prob=[0.1, 0.5, 0.9])
exp_err = wf.fpeak - q[1]
exp_err_upp = (wf.fpeak - q[2])
exp_err_low = (wf.fpeak - q[0])

#
# Plot frequentist confidence intervals as errorbars
#
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# --- All injections
freqerrfig, freqerrax = pl.subplots()

# performance for logB>0
upps=np.zeros(len(distances))
lows=np.zeros(len(distances))
centers=np.zeros(len(distances))
for d in xrange(len(distances)):

    upps[d]=confintvals[d][1] - wf.fpeak
    lows[d]=wf.fpeak - confintvals[d][0]
    centers[d]=wf.fpeak - np.median(maxfreqs[d])

p = freqerrax.errorbar(distances, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 90% CI')
freqerrax.set_xlim(0.9*min(distances), 1.1*max(distances))

freqerrax.axhline(0,color='r',linestyle='-', label='error=0 Hz')
freqerrax.axhline(exp_err,color='grey',linestyle='-', label='expected error from prior')
freqerrax.axhline(exp_err_low,color='grey',linestyle='--',label='90% CI from prior')
freqerrax.axhline(exp_err_upp,color='grey',linestyle='--')
freqerrax.legend(loc='upper left')

#freqerrax.set_ylim(-100,100)
#freqerrax.legend()
freqerrax.axhline(0,color='r',linestyle='-')
freqerrax.minorticks_on()
freqerrax.set_title('All Injections')
freqerrax.set_ylabel('Frequency Error [Hz]')
freqerrax.set_xlabel('Distance [Mpc]')


pl.savefig('freqerr.png')
pl.savefig('freqerr.eps')


# --- Injections with logB>0
freqerrfig, freqerrax = pl.subplots()

# performance for logB>0
upps=np.zeros(len(confintvals_found))
lows=np.zeros(len(confintvals_found))
centers=np.zeros(len(confintvals_found))
for d in xrange(len(confintvals_found)):

    upps[d]=confintvals_found[d][1] - wf.fpeak
    lows[d]=wf.fpeak - confintvals_found[d][0]
    centers[d]=wf.fpeak - np.median(maxfreqs_found[d])

p = freqerrax.errorbar(found_distances, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 90% CI')
freqerrax.set_xlim(0.9*min(distances), 1.1*max(distances))

freqerrax.axhline(0,color='r',linestyle='-', label='error=0 Hz')
freqerrax.axhline(exp_err,color='grey',linestyle='-', label='expected error from prior')
freqerrax.axhline(exp_err_low,color='grey',linestyle='--',label='90% CI from prior')
freqerrax.axhline(exp_err_upp,color='grey',linestyle='--')
freqerrax.legend(loc='upper left')

#freqerrax.set_ylim(-100,100)
#freqerrax.legend()
freqerrax.axhline(0,color='r',linestyle='-')
freqerrax.minorticks_on()
freqerrax.set_title('Injections with log B>%.2f'%logBthresh)
freqerrax.set_ylabel('Frequency Error [Hz]')
freqerrax.set_xlabel('Distance [Mpc]')

ins_ax = inset_axes(freqerrax, width="60%", height="30%", loc=4)

ins_ax.errorbar(found_distances, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 90% CI')

ins_ax.set_xlim(0.9*min(distances), 1.1*max(distances))
ylims=ins_ax.get_ylim()
ins_ax.set_ylim(ylims)
ins_ax.axhline(0,color='r',linestyle='-')
ins_ax.axhline(exp_err,color='grey',linestyle='-', label='expected error from prior')
ins_ax.axhline(exp_err_low,color='grey',linestyle='--')
ins_ax.axhline(exp_err_upp,color='grey',linestyle='--')
ins_ax.minorticks_on()
ins_ax.set_xticklabels('')

pl.savefig('freqerr_logB-%.2f.png'%logBthresh)
pl.savefig('freqerr_logB-%.2f.eps'%logBthresh)



pl.show()
sys.exit()


# --- Frequency recovery: widths
"""
a) boxplot of frequency width for found and all injections
"""
# - 'All'
freqwidthboxfig, freqwidthboxax = pl.subplots() 

freqwidthbp=freqwidthboxax.boxplot(credintwidths, notch=True, bootstrap=1000)
pl.axhline(prior_f_width, color='r', linestyle='--')

freqwidthboxax.set_ylim(0,3000)
pl.setp(freqwidthbp['boxes'], color='black')
pl.setp(freqwidthbp['whiskers'], color='black')
freqwidthboxax.set_title('All Injections')
freqwidthboxax.set_ylabel('Frequency width [Hz]')
freqwidthboxax.set_xlabel('Distance [Mpc]')
pl.setp(freqwidthboxax, xticklabels=distances)

pl.savefig('freqcredwidth.png')
pl.savefig('freqcredwidth.eps')

# - 'Found'
freqwidthboxfig, freqwidthboxax = pl.subplots() 

freqwidthbp=freqwidthboxax.boxplot(credintwidths_found, notch=True, bootstrap=1000)
pl.axhline(prior_f_width, color='r', linestyle='--')

freqwidthboxax.set_ylim(0, 3000)
pl.setp(freqwidthbp['boxes'], color='black')
pl.setp(freqwidthbp['whiskers'], color='black')
freqwidthboxax.set_title('Injections with log B>%.2f'%logBthresh)
freqwidthboxax.set_ylabel('Frequency width [Hz]')
freqwidthboxax.set_xlabel('Distance [Mpc]')
pl.setp(freqwidthboxax, xticklabels=distances)

pl.savefig('freqcredwidth_logB-%.2f.png'%logBthresh)
pl.savefig('freqcredwidth_logB-%.2f.eps'%logBthresh)

#pl.show()


    


