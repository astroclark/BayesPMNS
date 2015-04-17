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
pmns_pca_matches.py

Script to produce matches from PCA of post-merger waveforms
"""

from __future__ import division
import os,sys
import cPickle as pickle
import numpy as np

from matplotlib import pyplot as pl

import pmns_utils as pu
import pmns_pca_utils as ppca

from sklearn.decomposition import PCA

n_row, n_col = 2, 3
n_components = n_row*n_col

def plot_tfpcs(title, images, n_col=n_col, n_row=n_row):
    pl.figure(figsize=(4. * n_col, 4.52 * n_row))
    pl.suptitle(title, size=16)
    for i, comp in enumerate(images):
        pl.subplot(n_row, n_col, i + 1)

        vmin, vmax = -comp.max(), comp.max()
        collevs=np.linspace(vmin, vmax, 100)
        pl.contourf(times-0.5*max(times), frequencies,
                comp.reshape(image_shape), cmap=pl.cm.gnuplot2_r, levels=collevs)
        pl.xlim(-0.005,0.025)
        pl.ylim(min(frequencies),
                2*frequencies[abs(scales-0.5*max(scales)).argmin()])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# THINGS TO PLOT IN THIS SCRIPT
#
# 1a) Plots of original & centered magnitude spectra
# 1b) Plot of mean magnitude spectrum
# 1c) Plot of explained variances
# 1d) Plots of magnitude PCs

# 2a) Plots of original & centered TF maps
# 2b) Plot of mean tfmap
# 2c) Plot of explained variances
# 2d) Plots of TF PCs

# Just go up to ~3 PCs (or whatever's required for 90% variance)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA
#
# Produces: 1) Explained variance calculation
#           2) Matches for waveforms reconstructed from their own PCs

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


#
# Create PMNS PCA instance for this catalogue
#

nTsamples=16384

pmpca = ppca.pmnsPCA(waveform_names, low_frequency_cutoff=1000, fcenter=2710,
        nTsamples=nTsamples)

#
# Plot Sample spectra
#

#
# Plot spectral PCs
#

#
# Plot Sample TF maps
#

#
# Plot TF PCs
#

image_shape = pmpca.align_image_cat[0].shape
times = pmpca.map_times
frequencies = pmpca.map_frequencies
scales = pmpca.map_scales

plot_tfpcs('EigenMaps',
        pmpca.pca['timefreq_pca'].components_[:n_components])




