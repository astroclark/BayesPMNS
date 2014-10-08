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

def cdf(values, nbins=100, binrange=None, complementary=False):
    if binrange==None:
        binrange=(min(values[values>-np.inf]),max(values[values<np.inf]))
    bins=np.linspace(binrange[0],binrange[-1],nbins)

    delta=np.diff(bins)[0]
    values=np.clip(values,binrange[0],binrange[-1])

    bins=np.sort(values)
    counts, bin_edges = np.histogram(values, bins=bins, normed=False)

    cdf = np.cumsum(counts)
    
    if complementary:
        distr = 1-cdf/float(max(cdf))
    else:
        distr = cdf/float(max(cdf))

    #bin_centers=np.resize(bin_edges,len(bin_edges)-1)+delta
    bin_centers=np.resize(bin_edges,len(bin_edges)-1)#+delta
     
    return bin_centers,distr

# -------------------------------
# Load results


waveform_name=sys.argv[1]
#datafiles=sys.argv[2]

# XXX: Hardcoding
#distances=np.array([5, 7.5, 10, 12.5, 15, 17.5, 20])
distances=np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100])

odds2pos = lambda logB: 1.0/(1.0+1.0/np.exp(logB))
#logBthresh=np.interp(0.9, odds2pos(np.arange(0, 10, 1e-4)), np.arange(0, 10,
#    1e-4))

found_criterion=sys.argv[2]

logBthresh=np.log(float(sys.argv[3]))
netSNRthresh=float(sys.argv[3])

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

#datafiles=glob.glob('LIB-PMNS_waveform-%s_distance-*.pickle'%waveform_name)

for d,dist in enumerate(distances):

    datafile='LIB-PMNS_waveform-%s_distance-%.1f.pickle'%(waveform_name, dist)

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
    mf=np.zeros(shape=np.shape(this_freq_pdfs)[0])
    for m in xrange(np.shape(this_freq_pdfs)[0]):
        mf[m] = np.trapz(this_freq_pdfs[m]*freq_axis, freq_axis)
    meanfreqs.append(mf)

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

eps = np.zeros(len(distances))
delta_eps = np.zeros(len(distances))

# The probability of recovering the frequency to within 50 Hz
eps_freq = np.zeros(len(distances))
delta_eps_freq = np.zeros(len(distances))
eps_freq_found = []
delta_eps_freq_found = []

credintwidths_found = []
freq_pdfs_found = []

maxfreqs_found=[]
meanfreqs_found=[]

confintvals_found=[]
confintwidths_found=[]

found_distances=[]

# --- Efficiency/missed/found loop
c=0
for b in xrange(len(distances)):

    freq_pdfs_found_tmp = []

    # --- Efficiency Calculation
    #delta_f = abs(maxfreqs[b]-wf.fpeak)
    #k = sum(delta_f<10)
    if found_criterion=='snr':
        k = sum(netSNRs[b]>netSNRthresh)
    elif found_criterion=='logB':
        k = sum(logBs[b]>logBthresh)

    N = len(logBs[b])
    eps[b], delta_eps[b] = compute_efficiency(k,N)

    k = sum(abs(wf.fpeak-maxfreqs[b])<50.0)
    eps_freq[b], delta_eps_freq[b] = compute_efficiency(k,N)

    # --- Frequency Recovery

    # only include found things where Nfound is > 10
    if k>=10:
        #found_indices=np.concatenate(np.argwhere(logBs[b]>logBthresh))
        found_indices=np.concatenate(np.argwhere(netSNRs[b]>netSNRthresh))

        credintwidths_found.append(credintwidths[b][found_indices])

        for i in found_indices:
            freq_pdfs_found_tmp.append(freq_pdfs[b][i])
        freq_pdfs_found.append(freq_pdfs_found_tmp)

        maxfreqs_found.append(maxfreqs[b][found_indices])
        meanfreqs_found.append(meanfreqs[b][found_indices])

        k = sum(abs(wf.fpeak-maxfreqs_found[c])<50.0)
        N = sum(netSNRs[b]>netSNRthresh)
        e, d = compute_efficiency(k,N)
        eps_freq_found.append(e)
        delta_eps_freq_found.append(d)

        found_distances.append(distances[b])

        # 1-sigma Frequentist confidence intervals for MAP estimate
        confintvals_found.append(stats.scoreatpercentile(maxfreqs_found[c],
            [15.87,100-15.87]))

        confintwidths_found.append(confintvals_found[c][1]-confintvals_found[c][0])

        c+=1


# Averaged results
all_maxfreqs = np.concatenate(maxfreqs)
all_maxfreqs_int = stats.scoreatpercentile(all_maxfreqs, [25.0,100-25.0])
all_maxfreqs_std = np.std(all_maxfreqs)

all_maxfreqs_found = np.concatenate(maxfreqs_found)
all_maxfreqs_found_int = stats.scoreatpercentile(all_maxfreqs_found,
        [25.0,100-25.0])
all_maxfreqs_found_std = np.diff(all_maxfreqs_found_int)#np.std(all_maxfreqs_found)

all_freqerrs = wf.fpeak-all_maxfreqs
all_freqerrs_int = stats.scoreatpercentile(all_freqerrs, [25.0,100-25.0])
all_freqerrs_std = np.std(all_freqerrs)

all_freqerrs_found = wf.fpeak-all_maxfreqs_found
all_freqerrs_found_int = stats.scoreatpercentile(all_freqerrs_found,
        [25.0,100-25.0])
all_freqerrs_found_std = np.diff(all_freqerrs_found_int)#np.std(all_freqerrs_found)

all_absfreqerrs = abs(wf.fpeak-all_maxfreqs)
all_absfreqerrs_int = stats.scoreatpercentile(all_absfreqerrs, [25.0,100-25.0])
all_absfreqerrs_std = np.diff(all_absfreqerrs_int)#np.std(all_absfreqerrs)

all_absfreqerrs_found = abs(wf.fpeak-all_maxfreqs_found)
all_absfreqerrs_found_int = stats.scoreatpercentile(all_absfreqerrs_found,
        [25.0,100-25.0])
all_absfreqerrs_found_std = np.diff(all_absfreqerrs_found_int)#np.std(all_absfreqerrs_found)

f=open('frequency_recovery.txt', 'w')
f.writelines('# All frequencies\n')
f.writelines('# median_fpeak 25th 75th std\n')
f.writelines('%.2f %.2f %.2f %.2f\n\n'%(np.median(all_maxfreqs),
    all_maxfreqs_int[0], all_maxfreqs_int[1], all_maxfreqs_std))

f.writelines('# Found frequencies\n')
f.writelines('# median_fpeak 25th 75th std\n')
f.writelines('%.2f %.2f %.2f %.2f\n\n'%(np.median(all_maxfreqs_found),
    all_maxfreqs_found_int[0], all_maxfreqs_found_int[1],
    all_maxfreqs_found_std))

f.writelines('# All frequencies ERRORS\n')
f.writelines('# median_fpeak 25th 75th std\n')
f.writelines('%.2f %.2f %.2f %.2f\n\n'%(np.median(all_freqerrs),
    all_freqerrs_int[0], all_freqerrs_int[1], all_freqerrs_std))

f.writelines('# Found frequencies ERRORS\n')
f.writelines('# median_fpeak 25th 75th std\n')
f.writelines('%.2f %.2f %.2f %.2f\n\n'%(np.median(all_freqerrs_found),
    all_freqerrs_found_int[0], all_freqerrs_found_int[1],
    all_freqerrs_found_std))

f.close()
#sys.exit()

# XXX: TODO: add abs values

# -------------------------------
# Plots

# --- Distance Averaged Frequency Measurements
ffig, fax = pl.subplots(figsize=(10,5),ncols=2)

bins = np.arange(min(all_maxfreqs), max(all_maxfreqs), 5)
fax[0].hist(all_maxfreqs, bins=bins, normed=True, histtype='stepfilled', alpha=0.5)
fax[0].set_xlim(min(bins),max(bins))
fax[0].axvline(wf.fpeak,color='r',label='Target')
fax[0].axvline(all_maxfreqs_int[0],linestyle='--',
        color='k',label='%.2f'%all_maxfreqs_int[0])
fax[0].axvline(all_maxfreqs_int[1],linestyle='--',
        color='k',label='%.2f'%all_maxfreqs_int[1])
fax[0].axvline(np.median(all_maxfreqs),linestyle='-',
        color='k',label='%.2f'%np.median(all_maxfreqs))
fax[0].set_xlabel('Frequency [Hz]')
fax[0].set_ylabel('Normalised Count')
fax[0].set_title('All Injections')
fax[0].legend()

fax[1].hist(all_maxfreqs_found, bins=bins, normed=True, histtype='stepfilled',
        alpha=0.5)
fax[1].set_xlim(min(bins),max(bins))
fax[1].axvline(wf.fpeak,color='r',label='Target')
fax[1].axvline(all_maxfreqs_found_int[0],linestyle='--',
        color='k',label='%.2f'%all_maxfreqs_found_int[0])
fax[1].axvline(all_maxfreqs_found_int[1],linestyle='--',
        color='k',label='%.2f'%all_maxfreqs_found_int[1])
fax[1].axvline(np.median(all_maxfreqs_found),linestyle='-',
        color='k',label='%.2f'%np.median(all_maxfreqs_found))
fax[1].set_xlabel('Frequency [Hz]')
#fax[1].set_ylabel('Normalised Count')
fax[1].set_title('Found Injections')
fax[1].legend()

ffig.savefig('recovered_freqs.eps')
ffig.savefig('recovered_freqs.png')

#ffig.tight_layout()

# --- Distance Averaged Frequency Measurements: errors
ffig, fax = pl.subplots(figsize=(10,5),ncols=2)

bins = np.arange(min(all_freqerrs), max(all_freqerrs), 5)
fax[0].hist(all_freqerrs, bins=bins, normed=True, histtype='stepfilled', alpha=0.5)
fax[0].set_xlim(min(bins),max(bins))
fax[0].axvline(0,color='r',label='Target')
fax[0].axvline(all_freqerrs_int[0],linestyle='--',
        color='k',label='%.2f'%all_freqerrs_int[0])
fax[0].axvline(all_freqerrs_int[1],linestyle='--',
        color='k',label='%.2f'%all_freqerrs_int[1])
fax[0].axvline(np.median(all_freqerrs),linestyle='-',
        color='k',label='%.2f'%np.median(all_freqerrs))
fax[0].set_xlabel('Frequency Error [Hz]')
fax[0].set_ylabel('Normalised Count')
fax[0].set_title('All Injections')
fax[0].legend(loc='upper left')

fax[1].hist(all_freqerrs_found, bins=bins, normed=True, histtype='stepfilled',
        alpha=0.5)
fax[1].set_xlim(min(bins),max(bins))
fax[1].axvline(0,color='r',label='Target')
fax[1].axvline(all_freqerrs_found_int[0],linestyle='--',
        color='k',label='%.2f'%all_freqerrs_found_int[0])
fax[1].axvline(all_freqerrs_found_int[1],linestyle='--',
        color='k',label='%.2f'%all_freqerrs_found_int[1])
fax[1].axvline(np.median(all_freqerrs_found),linestyle='-',
        color='k',label='%.2f'%np.median(all_freqerrs_found))
fax[1].set_xlabel('Frequency Error [Hz]')
#fax[1].set_ylabel('Normalised Count')
fax[1].set_title('Found Injections')
fax[1].legend(loc='upper left')

ffig.savefig('recovered_freqs_errors.eps')
ffig.savefig('recovered_freqs_errors.png')

# --- Distance Averaged Frequency Measurements: errors
ffig, fax = pl.subplots(figsize=(10,5),ncols=2)

bins = np.arange(min(all_absfreqerrs), max(all_absfreqerrs), 5)
fax[0].hist(all_absfreqerrs, bins=bins, normed=True, histtype='stepfilled', alpha=0.5)
fax[0].set_xlim(min(bins),max(bins))
fax[0].axvline(0,color='r',label='Target')
fax[0].axvline(all_absfreqerrs_int[0],linestyle='--',
        color='k',label='%.2f'%all_absfreqerrs_int[0])
fax[0].axvline(all_absfreqerrs_int[1],linestyle='--',
        color='k',label='%.2f'%all_absfreqerrs_int[1])
fax[0].axvline(np.median(all_absfreqerrs),linestyle='-',
        color='k',label='%.2f'%np.median(all_absfreqerrs))
fax[0].set_xlabel('Frequency Error [Hz]')
fax[0].set_ylabel('Normalised Count')
fax[0].set_title('All Injections')
fax[0].legend(loc='upper right')

fax[1].hist(all_absfreqerrs_found, bins=bins, normed=True, histtype='stepfilled',
        alpha=0.5)
fax[1].set_xlim(min(bins),max(bins))
fax[1].axvline(0,color='r',label='Target')
fax[1].axvline(all_absfreqerrs_found_int[0],linestyle='--',
        color='k',label='%.2f'%all_absfreqerrs_found_int[0])
fax[1].axvline(all_absfreqerrs_found_int[1],linestyle='--',
        color='k',label='%.2f'%all_absfreqerrs_found_int[1])
fax[1].axvline(np.median(all_absfreqerrs_found),linestyle='-',
        color='k',label='%.2f'%np.median(all_absfreqerrs_found))
fax[1].set_xlabel('Frequency Error [Hz]')
#fax[1].set_ylabel('Normalised Count')
fax[1].set_title('Found Injections')
fax[1].legend(loc='upper right')

ffig.savefig('recovered_absfreqs_errors.eps')
ffig.savefig('recovered_absfreqs_errors.png')

#ffig.tight_layout()


# --- Efficiency: signals above significance threshold
epsfig, epsax = pl.subplots()

sigparams = fit_sigmoid(distances[::-1], eps[::-1], delta_eps[::-1])
dist_fit = np.arange(0, 200, 0.1)[::-1]
eps_fit = sigmoid(dist_fit,*sigparams[0]) 

# Get (APPROXIMATE) sensitive distance
x=dist_fit[::-1]/2.26
y=eps_fit[::-1]

Vsens = np.trapz(4*np.pi*x*x*y, x=x)
Dsens = ( 3/(4.*np.pi) * Vsens )**(1.0/3.0)
print Vsens, Dsens

epsp = pl.errorbar(distances, eps, yerr=delta_eps, ecolor='r', color='r',
        label='Above detection threshold',
        marker='^',mfc='r')
#epsp = pl.errorbar(distances, eps, yerr=delta_eps, linestyle='none', color='r')
epsfitp = pl.plot(dist_fit, eps_fit, color='k', label='Dsens=%.2f'%Dsens)

epsax.set_xlabel('Distance [Mpc]')
epsax.set_ylabel('Fraction Statistically Significant')
epsax.legend(loc='upper right')
epsax.minorticks_on()
epsax.grid(linestyle='--')
epsax.set_ylim(0,1)
epsax.set_xlim(0,100)

pl.savefig('efficiency.png')
pl.savefig('efficiency.eps')

if 0:
    # --- Efficiency: frequency accuracy
    #epsfig, epsax = pl.subplots()

    epsp = pl.errorbar(distances, eps_freq, yerr=delta_eps_freq, ecolor='m',
            color='m', label='within 50 Hz (no threshold)', linestyle=':',
            marker='^',mfc='m')
    epsp = pl.errorbar(found_distances, eps_freq_found, yerr=delta_eps_freq_found,
            ecolor='g', color='g', linestyle='--', label='Fraction of found events which are within 50 Hz',
            marker='^',mfc='g')
    #epsp = pl.errorbar(distances, eps, yerr=delta_eps, linestyle='none', color='r')
    #epsfitp = pl.plot(dist_fit, eps_fit, color='k')
    epsax.legend(loc='lower left')

    epsax.set_xlabel('Distance [Mpc]')
    epsax.set_ylabel('Fraction Found')
    epsax.minorticks_on()
    epsax.grid(linestyle='--')
    epsax.set_ylim(0,1)
    epsax.set_xlim(0,25)

    pl.savefig('efficiency_accuracy.png')
    pl.savefig('efficiency_accuracy.eps')

print 'exiting after efficiency'
sys.exit()
# --------------------------------------------------------------------
# --- Frequency recovery: errors in Distance


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

    upps[d], lows[d] = wf.fpeak - confintvals[d]
    centers[d]=wf.fpeak - np.median(maxfreqs[d])
    lows[d] = centers[d]-lows[d]
    upps[d] = upps[d]-centers[d]

    t = freqerrax.plot(distances[d]*np.ones(len(maxfreqs[d])),
            wf.fpeak-maxfreqs[d], color='b', marker=',', linestyle='none')

p = freqerrax.errorbar(distances, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 68% CI')
freqerrax.set_xlim(0.9*min(distances), 1.1*max(distances))

freqerrax.axhline(0,color='r',linestyle='-', label='error=0 Hz')
freqerrax.axhline(wf.fpeak-0.5*(4000+1750),color='k',linestyle='--', label='Guess')
freqerrax.legend(loc='upper left')

ins_ax = inset_axes(freqerrax, width="60%", height="30%", loc=4)
ins_ax.errorbar(distances, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 68% CI')

ins_ax.set_xlim(0.9*min(distances), 1.1*max(distances))
ylims=ins_ax.get_ylim()
#ins_ax.set_ylim(ylims)
ins_ax.set_ylim(-100, 100)
ins_ax.axhline(0,color='r',linestyle='-')
ins_ax.minorticks_on()
ins_ax.set_xticklabels('')

#freqerrax.set_ylim(-100,100)
#freqerrax.legend()
freqerrax.axhline(0,color='r',linestyle='-')
freqerrax.minorticks_on()
freqerrax.set_title('All Injections')
freqerrax.set_ylabel('Frequency Error [Hz]')
freqerrax.set_xlabel('Distance [Mpc]')


pl.savefig('freqerr.png')
pl.savefig('freqerr.eps')


# --- 'found' Injections
freqerrfig, freqerrax = pl.subplots()

upps=np.zeros(len(confintvals_found))
lows=np.zeros(len(confintvals_found))
centers=np.zeros(len(confintvals_found))
for d in xrange(len(confintvals_found)):

    upps[d], lows[d] = wf.fpeak - confintvals_found[d]
    centers[d]=wf.fpeak - np.median(maxfreqs_found[d])
    lows[d] = centers[d]-lows[d]
    upps[d] = upps[d]-centers[d]

    t = freqerrax.plot(found_distances[d]*np.ones(len(maxfreqs_found[d])),
            wf.fpeak-maxfreqs_found[d], color='b', marker=',', linestyle='none')

p = freqerrax.errorbar(found_distances, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 68% CI')

freqerrax.set_xlim(0.9*min(distances), 1.1*max(distances))

freqerrax.axhline(0,color='r',linestyle='-', label='error=0 Hz')
freqerrax.axhline(wf.fpeak-0.5*(4000+1750),color='k',linestyle='--', label='Guess')
freqerrax.legend(loc='upper left')

#freqerrax.set_ylim(-100,100)
#freqerrax.legend()
freqerrax.axhline(0,color='r',linestyle='-')
freqerrax.minorticks_on()
freqerrax.set_title('Found Injections')
freqerrax.set_ylabel('Frequency Error [Hz]')
freqerrax.set_xlabel('Distance [Mpc]')

ins_ax = inset_axes(freqerrax, width="60%", height="30%", loc=4)

ins_ax.errorbar(found_distances, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 68% CI')

ins_ax.set_xlim(0.9*min(distances), 1.1*max(distances))
ylims=ins_ax.get_ylim()
#ins_ax.set_ylim(ylims)
ins_ax.set_ylim(-100, 100)
ins_ax.axhline(0,color='r',linestyle='-')
ins_ax.minorticks_on()
ins_ax.set_xticklabels('')

pl.savefig('freqerr_found.png')
pl.savefig('freqerr_found.eps')



#pl.show()
#sys.exit()



#pl.show()

# --------------------------------------------------------------------
# --- Frequency recovery: errors as function of recovered SNR

    
freqerrfig, freqerrax = pl.subplots()

# performance for logB>0
upps=np.zeros(len(distances))
lows=np.zeros(len(distances))
centers=np.zeros(len(distances))
median_snrs=np.zeros(len(distances))
for d in xrange(len(distances)):

    median_snrs[d] = np.median(netSNRs[d])
    upps[d], lows[d] = wf.fpeak - confintvals[d]
    centers[d]=wf.fpeak - np.median(maxfreqs[d])
    lows[d] = centers[d]-lows[d]
    upps[d] = upps[d]-centers[d]

    t = freqerrax.plot(netSNRs[d], wf.fpeak-maxfreqs[d], color='b', marker=',',
            linestyle='none')

p = freqerrax.errorbar(median_snrs, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 68% CI')
freqerrax.set_xlim(0.9*min(median_snrs), 1.1*max(median_snrs))

freqerrax.axhline(0,color='r',linestyle='-', label='error=0 Hz')
freqerrax.legend(loc='lower right')

#freqerrax.set_ylim(-100,100)
#freqerrax.legend()
freqerrax.axhline(0,color='r',linestyle='-')
freqerrax.minorticks_on()
freqerrax.set_title('All Injections')
freqerrax.set_ylabel('Frequency Error [Hz]')
freqerrax.set_xlabel('Recovered SNR')


pl.savefig('freqerr_netSNR.png')
pl.savefig('freqerr_netSNR.eps')

# --------------------------------------------------------------------
# --- Frequency recovery: errors as function of logB

    
freqerrfig, freqerrax = pl.subplots()

# performance for logB>0
upps=np.zeros(len(distances))
lows=np.zeros(len(distances))
centers=np.zeros(len(distances))
median_logBs=np.zeros(len(distances))
for d in xrange(len(distances)):

    upps[d], lows[d] = wf.fpeak - confintvals[d]
    centers[d]=wf.fpeak - np.median(maxfreqs[d])
    lows[d] = centers[d]-lows[d]
    upps[d] = upps[d]-centers[d]

    median_logBs[d] = np.median(logBs[d])

    t = freqerrax.plot(logBs[d], wf.fpeak-maxfreqs[d], color='b', marker=',',
            linestyle='none')

p = freqerrax.errorbar(median_logBs, centers, yerr=[lows, upps], linestyle='none',
        fmt='^', ecolor='k', mfc='r', mec='k', label='median & 68% CI')
freqerrax.set_xlim(0.9*min(median_logBs), 1.1*max(median_logBs))

freqerrax.axhline(0,color='r',linestyle='-', label='error=0 Hz')
freqerrax.legend(loc='lower right')

#freqerrax.set_ylim(-100,100)
#freqerrax.legend()
freqerrax.axhline(0,color='r',linestyle='-')
freqerrax.minorticks_on()
freqerrax.set_title('All Injections')
freqerrax.set_ylabel('Frequency Error [Hz]')
freqerrax.set_xlabel('log B')


pl.savefig('freqerr_logBs.png')
pl.savefig('freqerr_logBs.eps')

# -----------------
# LogB vs SNR
logBfig, logBax = pl.subplots()
for i in range(len(distances)): 
    p=logBax.plot(netSNRs[i],logBs[i],'.', label='distance=%.2f'%distances[i])

# Fit curve
x=np.concatenate(netSNRs)
y=np.concatenate(logBs)

idx=np.argsort(x)
x=x[idx]
y=y[idx]

p = np.polyfit(x,y,deg=2)
yfit=p[0]*x**2 + p[1]*x + p[0]

if 1:
    # --- Bayes factors
    bayesboxfig, bayesboxax = pl.subplots()
    bayesbp = pl.boxplot(logBs, notch=True)
    pl.setp(bayesbp['boxes'], color='black')
    pl.setp(bayesbp['whiskers'], color='black')
    bayesboxax.set_ylabel('log B')
    bayesboxax.set_xlabel('Distance [Mpc]')
    #pl.axhline(logBthresh,color='r')
    pl.setp(bayesboxax, xticklabels=distances)

    pl.savefig('bayes-boxes.png')
    pl.savefig('bayes-boxes.eps')


    # --- Frequency recovery: all PDFs

    for d in xrange(len(distances)):

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
     

    for d in xrange(len(found_distances)):

        pdflines, pdflinesax = pl.subplots()
        for i in xrange(np.shape(freq_pdfs_found[d])[0]):
            pdflinesax.plot(freq_axis,freq_pdfs_found[d][i], color='grey', linewidth=0.001)
        pdflinesax.set_yscale('log')
        pdflinesax.set_ylim(1e-5,1) 
        pdflinesax.set_xlabel('Frequency [Hz]')
        pdflinesax.set_ylabel('p(f|D)')
        pdflinesax.axvline(wf.fpeak, color='r', linestyle='--')
        pdflinesax.axhline(1.0/(4000-1500), color='k', linestyle='--')
        pdflinesax.set_title('Found, Distance = %.2f Mpc'%found_distances[d])

        pl.savefig('freqpdflines_dist-%.2f_found.png'%(distances[d]))
        pl.savefig('freqpdflines_dist-%.2f_found.eps'%(distances[d]))


if 0:
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
    freqwidthboxax.set_title('Found Injections')
    freqwidthboxax.set_ylabel('Frequency width [Hz]')
    freqwidthboxax.set_xlabel('Distance [Mpc]')
    pl.setp(freqwidthboxax, xticklabels=distances)

    pl.savefig('freqcredwidth_found.png')
    pl.savefig('freqcredwidth_found.eps')


