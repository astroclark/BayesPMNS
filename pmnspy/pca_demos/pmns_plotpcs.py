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

def plot_timeseries(ax,x,y,xlims=(-0.0025,0.025),ylims=(-15,15)):

    idx = np.argmax(y)
    ax.plot(x-x[idx], y, color='k')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel('Time [s]')
#    ax.set_ylabel('h$_+$(t)')
    ax.minorticks_on()

    #xticks = np.arange(min(xlims),max(xlims),0.005)
    xticks = np.arange(0,max(xlims),0.005)
    ax.set_xticks(xticks)

    yticks=ax.get_yticks()
    ax.set_yticklabels([])
    ax.grid()

    return yticks

def plot_freqseries(ax,x,y,xlims=(999,4096),ylims=(0, 0.035)):

    ax.plot(x, y, color='k')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel('Frequency [Hz]')
#    ax.set_ylabel('h$_+$(t)')
    ax.minorticks_on()

    label_locs=[1000, 1500, 2000, 2500, 3000, 3500, 4000]
    ax.set_xticks(label_locs)
    labels=[' ', '1500', '2000', '2500', '3000', '3500', ' ']
    ax.set_xticklabels(labels)

    yticks=ax.get_yticks()
    ax.set_yticklabels([])

    ax.grid()

    return yticks

def plot_freqpc(ax,x,y,xlims=(999,4096),ylims=(0, 0.035)):

    ax.plot(x, y, color='k')
    ax.set_xlim(xlims)
#    ax.set_ylim(ylims)
    ax.set_xlabel('Frequency [arb. units]')
#    ax.set_ylabel('h$_+$(t)')
    ax.minorticks_on()

    ax.set_xticklabels([])

    yticks=ax.get_yticks()
    ax.set_yticklabels([])

    ax.grid()

    return yticks

def plot_timefreqmap(ax, x, y, z, xlims=(-0.0025,0.025), ylims=(999,4096), clims=None):

    if clims is None:
        vmin, vmax = 0, z.max()
    else:
        vmin, vmax = clims
    collevs=np.linspace(vmin, vmax, 100)
    cont = ax.contourf(x-0.5*x.max(), y, z, cmap=pl.cm.gnuplot2_r, levels=collevs)

#    c = pl.colorbar(cont, ax=ax, orientation='horizontal')

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.minorticks_on()

    yticks=[int(val) for val in ax.get_yticks()]
    ax.set_yticklabels([])

    ax.set_xlabel('Time [s]')

    xticks = np.arange(0,max(xlims),0.005)
    ax.set_xticks(xticks)
    ax.grid()


    return yticks, cont

def plot_timefreqpc(ax, x, y, z, xlims=(-0.0025,0.025), ylims=(128,300),
        clims=None):

    if clims is None:
        vmin, vmax = -z.max(), z.max()
    else:
        vmin, vmax = clims

    collevs=np.linspace(vmin, vmax, 100)
    tfmap = ax.contourf(x-0.5*x.max(), y, z, cmap=pl.cm.gnuplot2_r, levels=collevs)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.minorticks_on()


    #ax.set_ylabel('Frequency [arb. units]')
    ax.set_yticklabels([])
    ax.set_xlabel('Time [s]')

    xticks = np.arange(0,max(xlims),0.005)
    ax.set_xticks(xticks)
    ax.grid()

    return tfmap

#   pl.xlim(-0.005,0.025)
#   pl.ylim(min(frequencies),
#           2*frequencies[abs(scales-0.5*max(scales)).argmin()])
#
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
# Plot Sample waveforms
#
nexamples=3
fcat, axcat = pl.subplots(ncols=nexamples, nrows=3, figsize=(10,8))
for w in xrange(3):

    tyticks  = plot_timeseries(axcat[0][w], pmpca.sample_times, pmpca.cat_timedomain[w,:])
    fyticks  = plot_freqseries(axcat[1][w], pmpca.sample_frequencies, abs(pmpca.cat_orig[w,:]))
    tfyticks, tfmap = plot_timefreqmap(axcat[2][w], pmpca.map_times,
            pmpca.map_frequencies, pmpca.original_image_cat[w])

    c = pl.colorbar(tfmap, orientation='horizontal', ax=axcat[2][w],
            shrink=0.9, pad=.25)
    c.set_ticks(np.arange(0,1,0.1))



# Prettify
axcat[0][0].set_ylabel('h$_+$(t)')
axcat[0][0].set_yticklabels(tyticks)

axcat[1][0].set_ylabel('|H$_+$(f)|')
axcat[1][0].set_yticklabels(fyticks)

axcat[2][0].set_ylabel('Frequency [Hz]')
axcat[2][0].set_yticklabels(tfyticks)


fcat.suptitle('Example Waveforms')
fcat.tight_layout()
fcat.subplots_adjust(wspace=0.0)

fcat.savefig('example_waveforms.png')


#
# Plot Explained Variance
#
f, ax = pl.subplots(figsize=(5,4))
ax.plot(range(1,len(waveform_names)+1),
        np.cumsum(pmpca.pca['magnitude_pca'].explained_variance_ratio_),
        label='Magnitude spectra')
ax.plot(range(1,len(waveform_names)+1),
        np.cumsum(pmpca.pca['timefreq_pca'].explained_variance_ratio_),
        label='Time-Frequency maps')
ax.minorticks_on()
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.legend(loc='lower right')
ylims=ax.get_ylim()
ax.set_ylim(min(ylims),1)
ax.grid()
f.tight_layout()

f.savefig('explained_variance.png')


#
# Plot global means
#
f, ax = pl.subplots(ncols=2, figsize=(10,4))

# Magnitude spectrum
ax[0].plot(pmpca.sample_frequencies, pmpca.pca['magnitude_mean'], color='k')
ax[0].set_xlim(999,4096)
ax[0].minorticks_on()
ax[0].grid()
ax[0].set_xlabel('Frequency [arb. units]')
ax[0].set_xticklabels([])
ax[0].set_ylabel('|H$_+$(f)|')

# TF map
vmin, vmax = 0, pmpca.pca['timefreq_mean'].max()
collevs=np.linspace(vmin, vmax, 100)
times = pmpca.map_times
freqs = pmpca.map_frequencies
image_shape = pmpca.original_image_cat[0].shape
meanmap = pmpca.pca['timefreq_mean'].reshape(image_shape)

cont = ax[1].contourf(times-0.5*times.max(), freqs, meanmap, cmap=pl.cm.gnuplot2_r,
        levels=collevs)

c = pl.colorbar(cont, orientation='horizontal', ax=ax[1],
        shrink=0.9, pad=.25)
c.set_ticks(np.arange(0,1,0.1))

xlims=(-0.0025,0.025)
ylims=(128,300)
ax[1].set_xlim(xlims)
ax[1].set_ylim(ylims)
ax[1].minorticks_on()
ax[1].set_yticklabels([])
ax[1].grid()

ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Frequency [arb. units]')

f.suptitle('Global Averages')
f.tight_layout()

f.savefig('global_means.png')

#
# Plot centered spectra & maps
#
f, ax = pl.subplots(nrows=2, ncols=3, figsize=(10,5))
for w in xrange(3):
    fyticks = plot_freqpc(ax[0][w], pmpca.sample_frequencies,
            pmpca.magnitude_align[w,:] - pmpca.pca['magnitude_mean'])

    #ax[0][w].set_ylim(-0.01,0.01)

    clims = (-0.4,0.4)
    tfmap = plot_timefreqpc(ax[1][w], pmpca.map_times, pmpca.map_frequencies,
            pmpca.align_image_cat[w]-pmpca.pca['timefreq_mean'].reshape(image_shape),
            clims=clims)

    c = pl.colorbar(tfmap, orientation='horizontal', ax=ax[1][w],
            shrink=0.9, pad=.25)
    c.set_ticks(np.arange(clims[0],clims[1]+0.2,0.2))

ax[0][0].set_yticklabels(fyticks)

f.tight_layout()
f.subplots_adjust(wspace=0.0)
f.suptitle('Centered Data')

f.savefig('centered_example_waveforms.png')

pl.show()

#
# Plot spectral PCs
#
f, ax = pl.subplots(nrows=2, ncols=3, figsize=(10,5))
for w in xrange(3):

    fyticks = plot_freqpc(ax[0][w], pmpca.sample_frequencies,
            pmpca.pca['magnitude_pca'].components_[w,:])

    clims = (-0.025,0.025)
    tfmap = plot_timefreqpc(ax[1][w], pmpca.map_times, pmpca.map_frequencies,
            pmpca.pca['timefreq_pca'].components_[w,:].reshape(image_shape),clims=clims)

    c = pl.colorbar(tfmap, orientation='horizontal', ax=ax[1][w],
            shrink=0.9, pad=.25)

    c.set_ticks(np.arange(-1,1,0.01))

ax[0][0].set_ylabel('PC amplitude')
ax[0][0].set_yticklabels(fyticks)
ax[1][0].set_ylabel('Frequency \n [arb. units]')

f.suptitle('Principal Components (1-3)')
f.tight_layout()
f.subplots_adjust(wspace=0.0)

f.savefig('top3_principal_components.png')

pl.show()


#
# Save data
#
import scipy.io as sio
outputdata = {'time_domain_waveforms':pmpca.cat_timedomain,
        'frequency_domain_waveforms':pmpca.cat_orig,
        'freqency_domain_waveforms_aligned':pmpca.cat_align,
        'magnitude_spectrum_global_mean':pmpca.pca['magnitude_mean'],
        'magnitude_principal_components':pmpca.pca['magnitude_pca'].components_,
        'timefreq_global_mean':pmpca.pca['timefreq_mean'],
        'timefreq_principal_components':pmpca.pca['timefreq_pca'].components_}
sio.savemat('postmergerpca.mat', outputdata)





