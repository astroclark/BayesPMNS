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

import triangle

import cPickle as pickle

import glob

from sklearn.neighbors import KernelDensity

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def compute_confints(x_axis,pdf,alpha):
    """
    Confidence Intervals From KDE
    """

    # Make sure PDF is correctly normalised
    pdf /= np.trapz(pdf,x_axis)

    # --- initialisation
    peak = freq_axis[np.argmax(pdf)]

    # Initialisation
    area=0.

    i=0 
    j=0 

    x_axis_left=x_axis[(x_axis<peak)][::-1]
    x_axis_right=x_axis[x_axis>peak]

    while area <= alpha:

        x_axis_current=x_axis[(x_axis>=x_axis_left[i])*(x_axis<=x_axis_right[j])]
        pdf_current=pdf[(x_axis>=x_axis_left[i])*(x_axis<=x_axis_right[j])]

        area=np.trapz(pdf_current,x_axis_current)

        if i<len(x_axis_left)-1: i+=1
        if j<len(x_axis_right)-1: j+=1

    low_edge, upp_edge = x_axis_left[i], x_axis_right[j]

    return peak,low_edge,upp_edge,area

# ----------------------
# Load Data

globpattern=sys.argv[1]
resultsfiles=glob.glob(globpattern)
if len(resultsfiles)==0:
    print >> sys.stderr, "error, no files found matching pattern: %s"%globpattern
    sys.exit()

# Load first file to get dimensions
samples, sig_ev, noise_ev = pickle.load(open(resultsfiles[0]))

#freq_samples = np.zeros(shape=(np.shape(np.concatenate(samples[:,:,1]))[0],
#    len(resultsfiles)))
all_sig_evs = np.zeros(len(resultsfiles))
all_noise_evs = np.zeros(len(resultsfiles))

freq_estimates = np.zeros(shape=(3, len(resultsfiles)))

# Loop through results matching glob pattern
freq_axis = np.arange(1500, 4000, 0.1)
freq_bw = 0.1

freq_pdfs = np.zeros(shape=(len(freq_axis), len(resultsfiles)))

for r, resfile in enumerate(resultsfiles):
    print 'loading %d of %d'%(r, len(resultsfiles))

    # Load current data
    samples, sig_ev, noise_ev = pickle.load(open(resfile))

    # Populate arrays
    freq_samples = np.concatenate(samples[:,:,1])
    all_sig_evs[r] = sig_ev
    all_noise_evs[r] = noise_ev

    # Construct frequency PDF kde
    freq_pdfs[:,r] = kde_sklearn(x=freq_samples, x_grid=freq_axis, bandwidth=5,
            algorithm='kd_tree') 

    # Get max-likelihood & confidence intervals
    freq_estimates[0, r], freq_estimates[1, r], freq_estimates[2, r], _ = \
            compute_confints(freq_axis, freq_pdfs[:,r], 0.68)


# Save results
import cPickle as pickle
savename=globpattern.replace('*','')
pickle.dump((freq_axis, freq_pdfs, freq_estimates, all_sig_evs, all_noise_evs),
        open("ensemble_%s.pickle"%savename,"wb"))





