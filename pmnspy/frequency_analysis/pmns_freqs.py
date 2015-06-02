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
pmns_freqs.py

Script to plot and analyse f-domain pmns signals
"""

from __future__ import division
import os,sys
import numpy as np

from matplotlib import pyplot as pl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pycbc.types
import pmns_utils as pu
import pmns_pca_utils as ppca


def tripleplot(cwt_result):

    signal = cwt_result['analysed_data']

    # Open the figure
    fig, ax_cont = pl.subplots(figsize=(10,5))

    vmin, vmax = 0, cwt_result['map'].max()
    collevs=np.linspace(vmin, vmax, 100)

    # Characteristic times
    tmax = signal.sample_times[np.argmax(signal.data)]

    # --- CWT
    ax_cont.contourf(cwt_result['times']-tmax, cwt_result['frequencies'],
            cwt_result['map'], levels=collevs, cmap=cm.gnuplot2_r)

    #ax_cont.set_yscale('log')
    ax_cont.set_xlabel('Time [s]')
    ax_cont.set_ylabel('Frequency [Hz]')

    divider = make_axes_locatable(ax_cont)

    # --- Time-series
    ax_ts = divider.append_axes("top", 1.5, sharex=ax_cont)
    ax_ts.plot(signal.sample_times-tmax, signal.data, color='k')

    # --- Fourier spectrum
    signal_frequency_spectrum = signal.to_frequencyseries()
    ax_fs = divider.append_axes("right", 1.5, sharey=ax_cont)
    ax_fs.semilogx(abs(signal_frequency_spectrum.data),
            signal_frequency_spectrum.sample_frequencies, color='k')

    # --- Mark features
    ax_fs.set_xlabel('|H$_+$(f)|')
    ax_ts.set_ylabel('h$_+$(t)')

    ax_cont.axvline(0,color='k')#,linewidth=2)
    ax_ts.axvline(0,color='k')#,linewidth=2)

    # Adjust axis limits
    ax_fs.set_ylim(800, 4096)
    ax_cont.set_ylim(800, 4096)
    ax_ts.set_xlim(-2e-3, 2.5e-2)
    ax_cont.set_xlim(-2e-3, 2.5e-2)

    ax_cont.set_yscale('log')
    ax_fs.set_yscale('log')
    ax_cont.set_yticks(np.arange(1000,5000,1000))
    ax_cont.set_yticklabels(np.arange(1000,5000,1000))

    ax_ts.minorticks_on()
    ax_fs.minorticks_on()
    ax_cont.minorticks_on()

   #ax_ts.grid(linestyle='-', color='grey')
   #ax_fs.grid(linestyle='-', color='grey')
   #ax_cont.grid(linestyle='-', color='grey')

    ax_cont.invert_yaxis()
 
    # Clean up tick text
    pl.setp(ax_ts.get_xticklabels()+ax_fs.get_yticklabels(),visible=False)

    ax_cont.xaxis.get_ticklabels()[-1].set_visible(False)

    for label in ax_fs.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    ax_ts.yaxis.get_ticklabels()[0].set_visible(False)


    #ax_fs.tick_params(axis='both', which='major', labelsize=8)
    #ax_fs.tick_params(axis='x', which='major', labelsize=8)

    fig.tight_layout()

    return fig, ax_cont, ax_ts, ax_fs


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

w = waveform_names.index(sys.argv[1]+'_lessvisc')

#
# Create PMNS PCA instance for this catalogue
#

nTsamples=16384

waveform = pu.Waveform(waveform_names[w])
waveform.reproject_waveform()
waveform.compute_characteristics()
Hplus = waveform.hplus.to_frequencyseries()

cwt_result = ppca.build_cwt(waveform.hplus, mother_freq=2, fmin=500)

fig, ax_cont, ax_ts, ax_fs = tripleplot(cwt_result)

# XXX: find frequency peaks in ASD
from scipy import signal
peakind = signal.find_peaks_cwt(abs(Hplus.data), np.arange(1,10,.1))
#pl.figure()
#pl.plot(Hplus.sample_frequencies, abs(Hplus.data))
linestyles=[':', '--']
p=0
for line in peakind:
    if abs(Hplus.data[line])<0.1*abs(Hplus.data).max() or Hplus.sample_frequencies[line] < 1000:
        continue
    else:
        ax_cont.axhline(Hplus.sample_frequencies[line], color='k',
                linestyle=linestyles[p], linewidth=2,
                label='%.2f'%Hplus.sample_frequencies[line])
        ax_fs.axhline(Hplus.sample_frequencies[line], color='k', linewidth=2,
                linestyle=linestyles[p],
                label='%.2f'%Hplus.sample_frequencies[line])
        ax_cont.legend()
        p+=1


# XXX: Get instantaneous freq. of max power
#maxfreqs = np.zeros(len(cwt_result['times']))
#for m in xrange(len(maxfreqs)):
#    maxfreqs[m] = cwt_result['frequencies'][np.argmax(cwt_result['map'])[:,m]]

pl.show()



