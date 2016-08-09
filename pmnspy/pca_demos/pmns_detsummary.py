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
pmns_detsummary.py

Script to load results from pmns_pca_analysis.py and make nice plots to
summarise match and parameter estimation results for the different detectors
"""

from __future__ import division
import os,sys
import numpy as np

from matplotlib import pyplot as pl

from pmns_utils import pmns_plots as pplots

pl.rcParams.update({'axes.labelsize': 18})
pl.rcParams.update({'xtick.labelsize':18})
pl.rcParams.update({'ytick.labelsize':18})
pl.rcParams.update({'legend.fontsize':18})

def detector_names(npzfile):
    """
    Parse the detector name from the npz file name and return string for plot
    labels
    """

    IFO = npzfile.split('_')[0]

    labels=dict()
    labels['aLIGO'] = 'aLIGO'
    labels['A+'] = 'A+'
    labels['A++'] = 'A++'
    labels['CE1'] = 'CE'
    labels['ET-D'] = 'ET-D'
    labels['Voyager'] = 'LV'

    return labels[IFO]

def bar_summary(data, labels, ylims=None, ylabel=None):
    """
    Produce a bar plot summarising the results in data.  Bar height lies at the
    median, error bars show 10th, 90th percentiles
    """

    print np.shape(np.percentile(data, [10, 50, 90], axis=1))
    print np.shape(np.nanpercentile(data, [10, 50, 90], axis=1))

    low_val, height, upp_val = np.nanpercentile(data, [10, 50, 90], axis=1).T

    bar_width = 0.4
    index = np.arange(len(labels))

    f, ax = pl.subplots()
    ax.bar(index+bar_width, height, width=bar_width, yerr=[height-low_val,
        upp_val-height], ecolor='black', facecolor='grey')

    ax.set_xticks(index+1.5*bar_width)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Instrument')

    if ylims is not None:
        ax.set_ylim(ylims)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return f, ax

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Results
#
npz_files = sys.argv[1].split(',')

# Load first file to get number of waveforms for preallocation
data = np.load(npz_files[0])
nwaves = len(data['match_modes'])

instrument_labels=[]

all_delta_fpeaks=np.zeros(shape=(len(npz_files), nwaves))
all_delta_R16=np.zeros(shape=(len(npz_files), nwaves))
all_delta_R16_total=np.zeros(shape=(len(npz_files), nwaves))
all_matches=np.zeros(shape=(len(npz_files), nwaves))

for p, npz_file in enumerate(npz_files):

    instrument_labels.append(detector_names(npz_file))

    data = np.load(npz_files[p])

    all_matches[p,:] = data['match_modes']
    #all_delta_fpeaks[p,:] = abs(data['fpeak_targets'] - data['fpeak_modes'])
    all_delta_fpeaks[p,:] = data['fpeak_std']
    all_delta_R16[p,:] = 1000*data['deltaR16_statistical']

    for r in xrange(len(all_delta_R16[p, :])):
        all_delta_R16_total[p, r] = np.sqrt(all_delta_R16[p,r]**2 +
                (1000*data['deltaR16_systematic'][r])**2)

    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot Results

# Match
f, ax = bar_summary(all_matches, instrument_labels, ylims=(.8,1),
        ylabel=r'Match')
f.savefig('multidet_match_summary.eps')
f.savefig('multidet_match_summary.png')

# Delta fpeak
f, ax = bar_summary(all_delta_fpeaks, instrument_labels, ylims=None,
        ylabel=r'Frequency Error [Hz]')
f.savefig('multidet_deltaFpeak_summary.eps')
f.savefig('multidet_deltaFpeak_summary.png')

# Delta R16 statistical
f, ax = bar_summary(all_delta_R16, instrument_labels, ylims=None,
        ylabel=r'Radius Error [m]')
f.savefig('multidet_deltaR16statistical_summary.eps')
f.savefig('multidet_deltaR16statistical_summary.png')

# Delta R16
f, ax = bar_summary(all_delta_R16_total, instrument_labels, ylims=None,
        ylabel=r'Radius Error [m]')
f.savefig('multidet_deltaR16_summary.eps')
f.savefig('multidet_deltaR16_summary.png')

pl.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Print Results

print "------------------------"
print "Matches"
print "10, 50, 90 percentiles:"
for d, det in enumerate(instrument_labels):
    low_val, height, upp_val = np.nanpercentile(all_matches, [10, 50, 90],
            axis=1).T
    print "%s: %.2f %.2f %.2f"%(det, low_val[d], height[d], upp_val[d])

print "------------------------"
print "delta fpeak"
print "10, 50, 90 percentiles:"
for d, det in enumerate(instrument_labels):
    low_val, height, upp_val = np.nanpercentile(all_delta_fpeaks, [10, 50, 90],
            axis=1).T
    print "%s: %.1f %.1f %.1f"%(det, low_val[d], height[d], upp_val[d])

print "------------------------"
print "delta R16 Statistical"
print "10, 50, 90 percentiles:"
for d, det in enumerate(instrument_labels):
    low_val, height, upp_val = np.nanpercentile(all_delta_R16, [10, 50, 90],
            axis=1).T
    print "%s: %.1f %.1f %.1f"%(det, low_val[d], height[d], upp_val[d])

print "------------------------"
print "delta R16"
print "10, 50, 90 percentiles:"
for d, det in enumerate(instrument_labels):
    low_val, height, upp_val = np.nanpercentile(all_delta_R16_total, [10, 50,
        90], axis=1).T
    print "%s: %.1f %.1f %.1f"%(det, low_val[d], height[d], upp_val[d])



