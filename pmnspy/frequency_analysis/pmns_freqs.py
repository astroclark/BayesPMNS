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

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata
from pmns_utils import pmns_pca as ppca


def tripleplot(cwt_result):

    signal = cwt_result['analysed_data']

    Z = np.copy(cwt_result['map'])

    for c in xrange(np.shape(Z)[1]):
        Z[:,c] /= np.log10(np.sqrt(cwt_result['frequencies']) / 2)

    # Open the figure
    fig, ax_cont = pl.subplots(figsize=(10,5))

    #maxcol = 0.45*Z.max()
    maxcol = 0.8*Z.max()
    vmin, vmax = 0, maxcol
    collevs=np.linspace(vmin, vmax, 100)

    # Characteristic times
    tmax = signal.sample_times[np.argmax(signal.data)]

    # --- CWT
    c = ax_cont.contourf(cwt_result['times']-tmax, cwt_result['frequencies'], Z,
            levels=collevs, extend='both')

    c.cmap.set_over('k')
    #c.set_cmap('gnuplot2_r')

    #ax_cont.set_yscale('log')
    ax_cont.set_xlabel('Time [s]')
    ax_cont.set_ylabel('Frequency [Hz]')

    divider = make_axes_locatable(ax_cont)

    # --- Time-series
    ax_ts = divider.append_axes("top", 0.8, sharex=ax_cont)
    ax_ts.plot(signal.sample_times-tmax, signal.data, color='k')

    # --- Fourier spectrum
    signal_frequency_spectrum = signal.to_frequencyseries()
    ax_fs = divider.append_axes("right", 3.0, sharey=ax_cont)
    x = 2*abs(signal_frequency_spectrum.data[1:])*np.sqrt(signal_frequency_spectrum.sample_frequencies.data[1:])
    y = signal_frequency_spectrum.sample_frequencies[1:]
    ax_fs.semilogx(x, y, color='k')

    #
    # Construct PSD
    #
    psd = pwave.make_noise_curve(fmax=signal_frequency_spectrum.sample_frequencies.max(),
            delta_f=signal_frequency_spectrum.delta_f, noise_curve='aLIGO')
    ax_fs.semilogx(np.sqrt(psd.data), psd.sample_frequencies, color='k',
            linestyle='--', label='aLIGO')
    ax_fs.legend()

#   pl.figure()
#   pl.plot(y, x)
#   pl.show()
#   sys.exit()

    # --- Mark features
    ax_fs.set_xlabel('2|H$_+$($f$)|$\sqrt{f}$ & $\sqrt{S(f)}$')
    ax_ts.set_ylabel('h$_+$(t)')

    #ax_cont.axvline(0,color='r')#,linewidth=2)
    #ax_ts.axvline(0,color='r')#,linewidth=2)

    #ax_cont.set_yscale('log')
    #ax_fs.set_yscale('log')
    #ax_cont.set_yticks(np.arange(1000,5000,1000))
    #ax_cont.set_yticklabels(np.arange(1000,5000,1000))

    # Adjust axis limits
    ax_fs.set_ylim(900, 3096)
    ax_cont.set_ylim(900, 3096)
    ax_ts.set_xlim(-2e-3, 1.5e-2)
    ax_cont.set_xlim(-2e-3, 1.5e-2)

    ax_fs.set_xlim(3e-24, 2.5e-22)

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

#    for label in ax_fs.xaxis.get_ticklabels()[::2]:
#        label.set_visible(False)

    ax_ts.yaxis.get_ticklabels()[0].set_visible(False)


    #ax_fs.tick_params(axis='both', which='major', labelsize=8)
    #ax_fs.tick_params(axis='x', which='major', labelsize=8)

    panel_times = [-0.1218e-3, 0.3593e-3, 1.559e-3, 4.606e-3]
    for pt in panel_times:
        ax_ts.axvline(pt, color='r')
        ax_cont.axvline(pt, color='r')

    fig.tight_layout()

    return fig, ax_cont, ax_ts, ax_fs


#   pl.xlim(-0.005,0.025)
#   pl.ylim(min(frequencies),
#           2*frequencies[abs(scales-0.5*max(scales)).argmin()])
#

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Waveform Generation
#

eos="tm1"
mass="135135"


#
# Create the list of dictionaries which comprises our catalogue
#
#waveform_data = pdata.WaveData(eos=eos, mass=mass, viscosity='lessvisc')

waveform = pwave.Waveform(eos=eos, mass=mass, viscosity="lessvisc")

waveform.reproject_waveform()

Hplus = waveform.hplus.to_frequencyseries()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wavelet analysis
#
cwt_result = ppca.build_cwt(waveform.hplus, mother_freq=3, fmin=500)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting
#

fig, ax_cont, ax_ts, ax_fs = tripleplot(cwt_result)

# XXX: find frequency peaks in ASD
from scipy import signal
peakind = signal.find_peaks_cwt(abs(Hplus.data), np.arange(1,10,1))
#pl.figure()
#pl.plot(Hplus.sample_frequencies, abs(Hplus.data))
linestyles=10*[':', '--', '-']
p=0
for line in peakind:
    if abs(Hplus.data[line])<0.1*abs(Hplus.data).max() or Hplus.sample_frequencies[line] < 1500:
        continue
    else:
        ax_cont.axhline(Hplus.sample_frequencies[line], color='r',
                linestyle='-', linewidth=2,
                label='%.2f'%Hplus.sample_frequencies[line])
        ax_fs.axhline(Hplus.sample_frequencies[line], color='r', linewidth=2,
                linestyle='-',
                label='%.2f'%Hplus.sample_frequencies[line])
        #ax_cont.legend()
        p+=1


# XXX: Specgram
#   pl.figure()
#   (Pxx, freqs, bins, im) = pl.specgram( waveform.hplus.data,
#           Fs=waveform.hplus.sample_rate, window=None, noverlap=32, NFFT=64)
#   pl.close()
#   maxcol = 0.5*Pxx.max()
#   vmin, vmax = 0, maxcol
#   collevs=np.linspace(vmin, vmax, 100)
#
#   f, ax = pl.subplots()
#   ax.imshow(np.log10(Pxx), extent=[min(bins), max(bins),
#       min(freqs), max(freqs)], aspect='auto', interpolation='lanczos',
#       origin='lower')
#   ax.invert_yaxis()
#
pl.show()

