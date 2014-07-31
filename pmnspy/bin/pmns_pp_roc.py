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
from scipy import optimize,stats
from scipy.stats.mstats import mquantiles

import pmns_utils

import cPickle as pickle


def compute_efficiency(k,N,b=True):

    if b:
        # Bayesian treatment
        classify_efficiencyilon=(k+1)/(N+2)
        stdev_classify_efficiencyilon=1.64*np.sqrt(classify_efficiencyilon*(1-classify_efficiencyilon)/(N+3))
    else:
        # Binomial treatment
        if N==0:
            classify_efficiencyilon=0.0
            stdev_classify_efficiencyilon=0.0
        else:
            classify_efficiencyilon=k/N
            stdev_classify_efficiencyilon=1.64*np.sqrt(classify_efficiencyilon*(1-classify_efficiencyilon)/N)
    return (classify_efficiencyilon,stdev_classify_efficiencyilon)

def get_err_bars(intervals, centers):
    upps=np.zeros(max(np.shape(intervals)))
    lows=np.zeros(max(np.shape(intervals)))

    for c in xrange(len(upps)):
        upps[c]=intervals[1][c] - centers[c]
        lows[c]=centers[c] - intervals[0][c]

    return lows, upps


# -------------------------------
# Load results

#datafiles = sys.argv[1]
datafile = sys.argv[1]

# XXX: Hardcoding
wf = pmns_utils.Waveform('dd2_135135_lessvisc')
wf.compute_characteristics()

# --- Data extraction
print >> sys.stdout, "loading %s..."%datafile
freq_axis, freq_pdfs_tmp, freq_estimates, all_sig_evs, all_noise_evs =\
        pickle.load(open(datafile))

logBs = all_sig_evs - all_noise_evs

#logBthreshs = np.linspace(min(logBs), max(logBs), 100)
#logBthresh_range = stats.scoreatpercentile(logBs, [5,95])
#logBthreshs = np.linspace(min(logBthresh_range), max(logBthresh_range), 10)
logBthreshs=np.arange(-0.5,0.5+0.05,0.05)

# Preallocation
freqerrs=np.zeros(len(logBthreshs))+np.nan
#confints=np.zeros(len(logBthreshs))+np.nan
confintwidths=np.zeros(len(logBthreshs))+np.nan
confints=np.zeros(shape=(2,len(logBthreshs)))+np.nan
epsilon=np.zeros(len(logBthreshs))+np.nan
stdeps=np.zeros(len(logBthreshs))+np.nan

#
# Calculations
#

# Loop over thresholds
for b, logBthresh in enumerate(logBthreshs):

    # Find fraction of events with logB>threshold (epsilon)

    k = sum(logBs>logBthresh)
    N = len(logBs)
    epsilon[b], stdeps[b] = compute_efficiency(k,N)


    # Median frequency error
    if k==0: continue
    found_indices=np.concatenate(np.argwhere(logBs>logBthresh))
    errtmp = wf.fpeak - freq_estimates[0]
    freqerrs[b]=np.median(errtmp[found_indices])
    confints[:,b]=stats.scoreatpercentile(errtmp[found_indices], [10,90])
    confintwidths[b]=abs(np.diff(stats.scoreatpercentile(errtmp[found_indices],
        [10,90])))



#
# Results figure
#
fig1, ax1 = pl.subplots()

lows, upps = get_err_bars(confints, freqerrs)

ax1.errorbar(logBthreshs, freqerrs, yerr=[lows, upps], linestyle='-', color='k',
        fmt='^', ecolor='k', mfc='k', mec='k', label='median $\pm$ 90% CI')
xlims=ax1.get_xlim()
ylims=ax1.get_ylim()
ax1.errorbar(-100, -100, -100, marker='v', mfc='grey', mec='k', linestyle='-',
        label='median $\pm$ 90% CI', color='k')
#ax1.axhline(0, color='k', linestyle='--')
ax1.axvline(0, color='k', linestyle='--')
ax1.minorticks_on()
ax1.set_xlabel('log B threshold')
ax1.set_ylabel('Frequency Error [Hz]')
ax1.legend(loc='lower right')
ax1.set_xlim(xlims)
ax1.set_ylim(ylims)
ax1.set_xlim(min(xlims)-0.1,max(xlims)+0.1)
ax1.set_title('%s'%datafile)
#ax1.grid()

ax2 = ax1.twinx()
ax2.set_ylabel('Dismissal probability')

ax2.errorbar(logBthreshs+0.01, 1-epsilon, stdeps, marker='v', mfc='grey', mec='k',
        color='grey')

ax2.set_xlim(min(xlims)-0.1,max(xlims)+0.1)
ax2.set_ylim(0,1)
ax2.minorticks_on()

pl.savefig('%s'%(datafile.replace('.pickle','_ROC.eps')))
pl.savefig('%s'%(datafile.replace('.pickle','_ROC.png')))


