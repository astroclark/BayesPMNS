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

def image_matches(match_matrix, waveform_data, title=None, mismatch=False):

    if mismatch:
        match_matrix = 1-match_matrix
        text_thresh = 0.1
        clims = (0,0.2)
        bar_label = '1-$\mathcal{M}$'
    else:
        text_thresh = 0.85
        clims = (0.75,1.0)
        bar_label = '$\mathcal{M}$'

    #fig, ax = pl.subplots(figsize=(15,8))
    #fig, ax = pl.subplots(figsize=(8,4))
    fig, ax = pl.subplots()
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

def image_euclidean(euclidean_matrix, waveform_data, title=None, clims=None):

    #clims = (0.0,0.10)
    if clims is None:
        clims = (0.0, euclidean_matrix.max())
    text_thresh = 0.5*max(clims)
    bar_label = '$||\Phi - \Phi_r||$'

    #fig, ax = pl.subplots(figsize=(15,8))
    #fig, ax = pl.subplots(figsize=(7,7))
    fig, ax = pl.subplots()
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


def plot_matches_by_npc(match_matrix, waveform_data, title=None,
        mismatch=False):

    if mismatch:
        match_matrix = 1-match_matrix
        ylabel = '1-$\mathcal{M}$'
    else:
        ylabel = '$\mathcal{M}$'

    # Calculations
    min_matches = match_matrix.min(axis=0)
    max_matches = match_matrix.max(axis=0)
    mean_matches = match_matrix.mean(axis=0)
    median_matches = np.median(match_matrix, axis=0)
    rms_matches = np.sqrt(np.mean(np.square(match_matrix), axis=0))

    low, upp = np.percentile(match_matrix, [10, 90], axis=0)  

    fig, ax = pl.subplots()

    ax.plot(range(1,len(min_matches)+1), mean_matches, color='k', label='mean')
    #ax.plot(range(1,len(min_matches)+1), median_matches, color='g', label='median')
    #ax.plot(range(1,len(min_matches)+1), rms_matches, color='r', label='rms')
    
    ax.fill_between(x=range(1,len(min_matches)+1), y1=low,
            y2=upp, color='lightgrey', label='min/max')

    ax.plot(range(1,len(min_matches)+1), min_matches, color='k', linestyle='--',
            label='min/max')
    ax.plot(range(1,len(min_matches)+1), max_matches, color='k', linestyle='--')


    return fig, ax


def main():

    print 'nothing here'

#
# End definitions
#
if __name__ == "__main__":
    main()






