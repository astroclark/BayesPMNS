#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2015-2016 James Clark <clark@physics.umass.edu>
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
import cPickle as pickle
import numpy as np

from matplotlib import pyplot as pl

from pmns_utils import pmns_plots as pplots

def detector_names(picklefile):
    """
    Parse the detector name from the pickle file name and return string for plot
    labels
    """

    IFO = picklefile.split('_')[2]

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

    low_val, height, upp_val = np.percentile(data, [10, 50, 90], axis=1)

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
pickle_files = sys.argv[1].split(',')

# redundant load just to get the number of waveforms for pre-allocation:
waveform_data, _, _, _, _, _ = pickle.load(open(pickle_files[0], "rb"))

instrument_labels=[]
all_delta_fpeaks=np.zeros(shape=(len(pickle_files), waveform_data.nwaves))
all_delta_R16=np.zeros(shape=(len(pickle_files), waveform_data.nwaves))
all_matches=np.zeros(shape=(len(pickle_files), waveform_data.nwaves))

for p, pickle_file in enumerate(pickle_files):

    instrument_labels.append(detector_names(pickle_file))

    _, _, _, matches, delta_fpeak, delta_R16 = pickle.load(open(pickle_file, "rb"))

    # matches is (nwaveforms, npcs); use 1st PC only
    all_matches[p, :] = matches[:,0]
    all_delta_fpeaks[p, :] = delta_fpeak[:,0]
    all_delta_R16[p, :] = delta_R16[:,0]

    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot Results

# Match
f, ax = bar_summary(all_matches, instrument_labels, ylims=(.8,1),
        ylabel=r'Match')
f.savefig('multidet_match_summary.eps')

# Delta fpeak
f, ax = bar_summary(all_delta_fpeaks, instrument_labels, ylims=None,
        ylabel=r'Frequency Error [Hz]')
f.savefig('multidet_deltaFpeak_summary.eps')

# Delta R16
f, ax = bar_summary(all_delta_R16, instrument_labels, ylims=None,
        ylabel=r'Radius Error [km]')
f.savefig('multidet_deltaR16_summary.eps')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Print Results

print "------------------------"
print "Matches"
print "10, 50, 90 percentiles:"
for d, det in enumerate(instrument_labels):
    low_val, height, upp_val = np.percentile(all_matches, [10, 50, 90], axis=1)
    print "%s: %.2f %.2f %.2f"%(det, low_val[d], height[d], upp_val[d])

print "------------------------"
print "delta fpeak"
print "10, 50, 90 percentiles:"
for d, det in enumerate(instrument_labels):
    low_val, height, upp_val = np.percentile(all_delta_fpeaks, [10, 50, 90], axis=1)
    print "%s: %.1f %.1f %.1f"%(det, low_val[d], height[d], upp_val[d])

print "------------------------"
print "delta R16"
print "10, 50, 90 percentiles:"
for d, det in enumerate(instrument_labels):
    low_val, height, upp_val = np.percentile(all_delta_R16, [10, 50, 90], axis=1)
    print "%s: %.1f %.1f %.1f"%(det, 1000*low_val[d], 1000*height[d], 1000*upp_val[d])




