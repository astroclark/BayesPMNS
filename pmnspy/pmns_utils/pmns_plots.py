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
pmns_plots.py

Various plotting tools for non-trivial PMNS plots
"""

import numpy as np
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as pl
import matplotlib.patches as mpatches

def image_matches(match_matrix, waveform_data, title=None, mismatch=False,
        figsize=None):

    if mismatch:
        match_matrix = 1-match_matrix
        text_thresh = 0.1
        clims = (0,0.2)
        bar_label = '1-$\mathcal{M}$'
    else:
        text_thresh = 0.85
        clims = (0.75,1.0)
        bar_label = '$\mathcal{M}$'

    fig, ax = pl.subplots()

    if figsize is not None:
        fig.set_size_inches(figsize)

    nwaves = np.shape(match_matrix)[0]
    npcs = np.shape(match_matrix)[1]

    im = ax.imshow(match_matrix, interpolation='nearest', origin='lower',
            aspect='auto')

    for x in xrange(nwaves):
        for y in xrange(npcs):
            if match_matrix[x,y]<text_thresh:
                ax.text(y, x, '%.2f'%(match_matrix[x,y]), \
                    va='center', ha='center', color='w')
            else:
                ax.text(y, x, '%.2f'%(match_matrix[x,y]), \
                    va='center', ha='center', color='k')

    ax.set_xticks(range(0,npcs))
    ax.set_yticks(range(0,nwaves))

    xlabels=range(1,npcs+1)
    ax.set_xticklabels(xlabels)

    ylabels=[]
    for wave in waveform_data.waves:
        if wave['viscosity']=='lessvisc':
            ylabels.append("%s, %s"%(wave['eos'], wave['mass']))
        else:
            ylabels.append("%s$^{\dagger}$, %s"%(wave['eos'], wave['mass']))
    ax.set_yticklabels(ylabels)


    ax.set_yticklabels(ylabels)

    im.set_clim(clims)
    im.set_cmap('gnuplot2')

    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Waveform type')

    if title is not None:
        ax.set_title(title)

    #c=pl.colorbar(im, ticks=np.arange(clims[0],clims[1]+0.05,0.05),
    c=pl.colorbar(im, orientation='horizontal')
    c.set_label(bar_label)

    fig.tight_layout()

    return fig, ax

def image_euclidean(euclidean_matrix, waveform_data, title=None, clims=None,
        figsize=None):

    #clims = (0.0,0.10)
    if clims is None:
        clims = (0.0, euclidean_matrix.max())
    text_thresh = 0.5*max(clims)
    bar_label = '$||\Phi - \Phi_r||$'

    fig, ax = pl.subplots()
    if figsize is not None:
        fig.set_size_inches(figsize)

    nwaves = np.shape(euclidean_matrix)[0]
    npcs = np.shape(euclidean_matrix)[1]

    im = ax.imshow(euclidean_matrix, interpolation='nearest', origin='lower',
            aspect='auto')

    for x in xrange(nwaves):
        for y in xrange(npcs):
            if euclidean_matrix[x,y]<text_thresh:
                ax.text(y, x, '%.2f'%(euclidean_matrix[x,y]), \
                    va='center', ha='center', color='k')
            else:
                ax.text(y, x, '%.2f'%(euclidean_matrix[x,y]), \
                    va='center', ha='center', color='w')

    ax.set_xticks(range(0,npcs))
    ax.set_yticks(range(0,nwaves))

    xlabels=range(1,npcs+1)
    ax.set_xticklabels(xlabels)

    
    ylabels=[]
    for wave in waveform_data.waves:
        if wave['viscosity']=='lessvisc':
            ylabels.append("%s, %s"%(wave['eos'], wave['mass']))
        else:
            ylabels.append("%s$^{\dagger}$, %s"%(wave['eos'], wave['mass']))
    ax.set_yticklabels(ylabels)

    im.set_clim(clims)
    im.set_cmap('gnuplot2_r')

    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Waveform type')

    if title is not None:
        ax.set_title(title)

    c=pl.colorbar(im, orientation='horizontal')
    c.set_label(bar_label)

    fig.tight_layout()

    return fig, ax


def plot_fidelity_by_npc(fidelity_matrix, waveform_data, title=None,
        figsize=None, ylabel = '$\mathcal{M}$', legloc=None):


    # Calculations
    min_fidelity = fidelity_matrix.min(axis=0)
    max_fidelity = fidelity_matrix.max(axis=0)
    mean_fidelity = fidelity_matrix.mean(axis=0)
    median_fidelity = np.median(fidelity_matrix, axis=0)
    rms_fidelity = np.sqrt(np.mean(np.square(fidelity_matrix), axis=0))

    low, upp = np.percentile(fidelity_matrix, [10, 90], axis=0)  

    fig, ax = pl.subplots()
    if figsize is not None:
        fig.set_size_inches(figsize)

    center = ax.step(range(1,len(min_fidelity)+1), mean_fidelity, color='r',
            label='mean')

    ax.bar(range(1,len(min_fidelity)+1), bottom=low, height=upp-low,
            color='lightgrey', label='10th, 90th percentile',
            edgecolor='lightgrey', width=1)

    lims=ax.step(range(1,len(min_fidelity)+1), min_fidelity, color='k', linestyle='--',
            label='min/max')

    ax.step(range(1,len(min_fidelity)+1), max_fidelity, color='k', linestyle='--')

    ax.minorticks_on()
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel(ylabel)

    ax.set_xlim(1,len(fidelity_matrix))

    leg = ax.legend(loc=legloc)


    return fig, ax

def plot_delta_by_npc(delta_matrix, waveform_data, ylabel="$\delta$",
        title=None, figsize=None, legloc=None):

    # Calculations
    min_delta = delta_matrix.min(axis=0)
    max_delta = delta_matrix.max(axis=0)
    mean_delta = delta_matrix.mean(axis=0)
    median_delta = np.median(delta_matrix, axis=0)
    rms_delta = np.sqrt(np.mean(np.square(delta_matrix), axis=0))

    low, upp = np.percentile(delta_matrix, [10, 90], axis=0)  

    fig, ax = pl.subplots()
    if figsize is not None:
        fig.set_size_inches(figsize)

    center = ax.step(range(1,len(min_delta)+1), mean_delta, color='r', label='mean')

    ax.bar(range(1,len(min_delta)+1), bottom=low, height=upp-low,
            color='lightgrey', label='10th, 90th percentile',
            edgecolor='lightgrey', width=1)

    lims=ax.step(range(1,len(min_delta)+1), min_delta, color='k', linestyle='--',
            label='min/max')

    ax.step(range(1,len(min_delta)+1), max_delta, color='k', linestyle='--')


    ax.minorticks_on()

    ax.set_xlabel('Number of PCs')
    ax.set_ylabel(ylabel)

    ax.set_xlim(1,len(delta_matrix))

    leg=ax.legend(loc=legloc)


    return fig, ax

def main():

    print 'nothing here'

#
# End definitions
#
if __name__ == "__main__":
    main()






