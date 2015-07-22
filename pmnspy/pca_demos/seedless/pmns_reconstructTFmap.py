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
pmns_reconstructTFmap.py

Script to load results from pmns_tfpca2mat.py and produce figures showing an
example TF map and its reconstruction.

The goal is to demonstrate the peak alignment procedure.

Produces 4 figures:
    1) The original example TF map (continuous wavelet transform)
    2) The feature-aligned version of that map
    3) The reconstruction of the feature-aligned map using the first principal
    component
    4) The de-aligned, single-PC reconstructed TF map

    (MUST not rely on pmns modules!)

"""

from __future__ import division
import os,sys
import cPickle as pickle
import numpy as np

from matplotlib import pyplot as pl

from scipy import io as sio

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Func defs

def shift_vec(vector, target_freqs, fpeak, fcenter=1000.0):
    """
    Perform the geometric frequency alignment which defines the PMNS PCA
    approach
    """

    # Frequency shift
    fshift = fcenter / fpeak
    false_freqs = target_freqs * fshift

    aligned_vector = np.interp(target_freqs, false_freqs, vector)

    return aligned_vector

def align_tfmap(timefreqmap, scales, freqs, fpeak):
    """
    Align the time-frequency map such that the wavelet scale which corresponds
    to the fpeak lies at 0.25x the maximum scale.

    In other words, 'squeeze' the TF map so that the fpeak is aligned to a
    common value (in this case a quarter of the maximum scale)
    """

    outputmap = np.copy(timefreqmap)

    peak_scale = scales[abs(freqs-fpeak).argmin()]

    # shift columns
    for c in xrange(np.shape(outputmap)[1]):
        outputmap[:,c] = shift_vec(outputmap[:,c], scales, peak_scale,
                0.25*scales.max())

    return outputmap

def dealign_tfmap(timefreqmap, scales, freqs, fpeak):
    """
    The inverse of align_tfmap(); simply switches the order of the scaling
    operation
    """
    outputmap = np.copy(timefreqmap)

    peak_scale = scales[abs(freqs-fpeak).argmin()]

    # shift columns
    for c in xrange(np.shape(outputmap)[1]):

        outputmap[:,c] = shift_vec(outputmap[:,c], scales, 0.25*scales.max(),
                peak_scale)

    return outputmap


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialise
#
timefreqpca = sio.loadmat(sys.argv[1])

# SIO loads the mat file as a dictionary; here's a trick to populate the
# workspace with the contents of that dictionary:
for key in timefreqpca.keys():
    vars()[key] = timefreqpca[key]

# Fix the shapes of some things:
fpeaks = np.concatenate(fpeaks)
timefreq_scales = np.concatenate(timefreq_scales)
timefreq_frequencies = np.concatenate(timefreq_frequencies)
timefreq_times = np.concatenate(timefreq_times)

# fpeaks contains the peak frequencies of the waveforms in the catalogue; we'll
# use the first one
fpeak = fpeaks[0]

# Select out example map
example_timefreqmap = timefreq_maps[0]
aligned_example_timefreqmap = align_tfmap(example_timefreqmap, timefreq_scales,
        timefreq_frequencies, fpeak)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build Reconstruction

# timefreq_betas is an NxN matrix of projection coefficients where rows
# correspond to the waveform being projected and columns correspond to the
# principal component.

aligned_timefreq_reconstruction = np.zeros(shape=np.shape(timefreq_mean))

# XXX: setting Npcs=1 here but keeping the loop for generality
npcs=1

# Sum the PCs
for n in xrange(npcs):
    aligned_timefreq_reconstruction += timefreq_betas[0,n] * \
            timefreq_principal_components[n,:,:]

# De-center (add the mean)
aligned_timefreq_reconstruction += timefreq_mean

# De-align (rescale the frequency)
timefreq_reconstruction = dealign_tfmap(aligned_timefreq_reconstruction,
        timefreq_scales, timefreq_frequencies, fpeak)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot results

f, ax = pl.subplots(nrows=2,ncols=2,figsize=(15,10), sharex=True)

# Example map
maxcol = 0.45*example_timefreqmap.max()
vmin, vmax = 0, maxcol
collevs=np.linspace(vmin, vmax, 100)
ax[0][0].contourf(timefreq_times, timefreq_frequencies, example_timefreqmap,
        levels=collevs, extend='both')
ax[0][0].set_xlim(0.04,0.08)
ax[0][0].set_ylim(1000, 4096)
ax[0][0].grid()
ax[0][0].set_title('Example Time-Frequency Map (step 1)')

# Aligned example
maxcol = 0.45*aligned_example_timefreqmap.max()
vmin, vmax = 0, maxcol
collevs=np.linspace(vmin, vmax, 100)
ax[1][0].contourf(timefreq_times, timefreq_frequencies, aligned_example_timefreqmap,
        levels=collevs, extend='both')
ax[1][0].set_xlim(0.04,0.08)
ax[1][0].set_ylim(152, 650)
ax[1][0].grid()
ax[1][0].set_title('Aligned Example Time-Frequency Map (step 2)')

# Reconstruction
maxcol = 0.45*timefreq_reconstruction.max()
vmin, vmax = 0, maxcol
collevs=np.linspace(vmin, vmax, 100)
ax[0][1].contourf(timefreq_times, timefreq_frequencies, timefreq_reconstruction,
        levels=collevs, extend='both')
ax[0][1].set_xlim(0.04,0.08)
ax[0][1].set_ylim(1000, 4096)
ax[0][1].grid()
ax[0][1].set_title('Reconstructed Time-Frequency Map (step 4)')

# Aligned reconstruction
maxcol = 0.45*aligned_timefreq_reconstruction.max()
vmin, vmax = 0, maxcol
collevs=np.linspace(vmin, vmax, 100)
ax[1][1].contourf(timefreq_times, timefreq_frequencies,
        aligned_timefreq_reconstruction, levels=collevs, extend='both')
ax[1][1].set_xlim(0.04,0.08)
ax[1][1].set_ylim(152, 650)
ax[1][1].grid()
ax[1][1].set_title('Aligned Reconstructed Time-Frequency Map (step 3)')

for i in xrange(2):
    ax[1][i].set_xlabel('Time [s]')
    for j in xrange(2):
        ax[i][j].set_ylabel('Frequency [Hz]')

#pl.subplots_adjust(hspace=0)
f.tight_layout()

pl.show()


