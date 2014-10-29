#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2014-2015 James Clark <james.clark@ligo.org>
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

import os
import sys 

__author__ = "James Clark <james.clark@ligo.org>"

import numpy as np
import scipy.signal as signal
import scipy.io as sio

from matplotlib import pyplot as pl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


import pmns_utils
import pmns_simsig as simsig

import cwt

def scale2hertz(motherf0, scale, sample_rate):
    fNy=0.5*sample_rate
    return 2*fNy * motherf0 / scale

def freq2scale(freqHz, motherf0, sample_rate):
    fNy  = 0.5*sample_rate
    freq = freqHz / (2*fNy)
    return motherf0/freq 

def set_scales(sample_rate, motherf0, deltat=2, deltaj=.1):

    s0 = freq2scale(0.5*sample_rate, 0.849, sample_rate)
    N = len(data)
    J = (1./deltaj) * np.log2(N*deltat/s0)

    return s0*2**( (np.arange(J))*deltaj)



# --------------------------------------------------------------------
# Data Generation

#
# Generate Signal Data
#

# Signal
print ''
print '--- %s ---'%sys.argv[1]
waveform = pmns_utils.Waveform('%s'%sys.argv[1])
waveform.compute_characteristics()

# Extrinsic parameters
ext_params = simsig.ExtParams(20.0, ra=0.0, dec=0.0,
        polarization=0.0, inclination=0.0, phase=0.0, geocent_peak_time=0.25)

# Frequency range for SNR around f2 peak - we'll go between 1, 5 kHz for the
# actual match calculations, though
flow=waveform.fpeak-150
fupp=waveform.fpeak+150

# Construct the time series for these params
waveform.make_wf_timeseries(theta=ext_params.inclination,
        phi=ext_params.phase)


det1_data = simsig.DetData(det_site="H1", noise_curve='aLIGO',
        waveform=waveform, ext_params=ext_params, duration=0.5, seed=0,
        epoch=0.0, f_low=10.0, taper=False)
        #epoch=0.0, f_low=10.0, taper=True)

#det1_data.td_signal = pycbc.filter.highpass(det1_data.td_signal, knee,
#        filter_order=20, attenuation=0.9)

# get data
sigdat=det1_data.td_signal.trim_zeros()

# highpass
import pycbc.filter
knee=1000
#sigdat = pycbc.filter.highpass(sigdat, knee,
#        filter_order=20, attenuation=0.9)

data = np.copy(sigdat.data)
time = sigdat.sample_times.data

sample_rate = 1.0/np.diff(time)[0]

########################
# Wavelets Decomposition

#
# CWT from pyCWT
#

#scales = np.logspace(np.log2(1), np.log2(256), base=2)

motherfreq = 2
scalerange = motherfreq/(np.array([10,0.5*sample_rate])*(1.0/sample_rate))

scales = np.arange(scalerange[1],scalerange[0])

mother_wavelet = cwt.Morlet(len_signal = len(data), scales = scales,
        sampf=sample_rate, f0=motherfreq)

#mother_wavelet = cwt.SDG(len_signal = len(data), scales = scales, normalize = True,
#        fc = 'center')

wavelet = cwt.cwt(data, mother_wavelet)

# --- Plotting

# convert to pseudo frequency
freqs = scale2hertz(mother_wavelet.fc, scales, sample_rate)

#collevs=np.linspace(0, max(map(max,abs(wavelet.coefs)**2)), 100)

fpeakmin=2000.0
collevs=np.linspace(0, 5*max(wavelet.get_wps()[freqs>fpeakmin]), 100)

fig, ax_cont = pl.subplots(figsize=(10,5))
c=ax_cont.contourf(time,freqs,np.abs(wavelet.coefs)**2, levels=collevs,
        cmap=cm.gnuplot2)


ax_cont.set_xlim(min(time),max(time))
ax_cont.set_ylim(1,0.5*sample_rate)
ax_cont.set_xlabel('Time [s]')
ax_cont.set_ylabel('Frequency [Hz]')

#pl.show()
#sys.exit()

divider = make_axes_locatable(ax_cont)

# time-series
ax_ts = divider.append_axes("top", 1.2, sharex=ax_cont)
ax_ts.plot(time, data)
ax_cont.set_xlim(min(time),max(time))
ax_ts.set_ylim(-1.1*max(abs(data)), 1.1*max(abs(data)))

# fourier spectrum
#freq_fourier, Pxx = signal.periodogram(data, fs=sample_rate)
ax_fs = divider.append_axes("right", 1.2, sharey=ax_cont)
#ax_fs.semilogx(Pxx,freq_fourier)
ax_fs.semilogx(wavelet.get_wps(),freqs)
ax_fs.set_ylim(1,0.5*sample_rate)
#ax_fs.set_xlim(0.01*max(Pxx),1.1*max(Pxx))
ax_fs.set_xlim(0.01*max(wavelet.get_wps()),1.1*max(wavelet.get_wps()))

pl.setp(ax_ts.get_xticklabels()+ax_fs.get_yticklabels(),visible=False)

pl.tight_layout()

########################
# Fourier Decomposition

from matplotlib import mlab

pl.figure()
S, F, T = mlab.specgram(data, NFFT=int(0.1*len(data)), Fs=sample_rate,
        window=mlab.window_hanning, noverlap=int(.5*0.1*len(data)))
pl.close()
Time=T+min(time)

#collevs=np.linspace(0, max(map(max,S)), 100)
fpeakmin=2000.0

freq_fourier, Pxx = signal.periodogram(data, fs=sample_rate)
collevs=np.linspace(0, 5*max(Pxx[freq_fourier>fpeakmin]), 100)


fig, ax_cont = pl.subplots(figsize=(10,5))

#ax_cont.contourf(Time, F, S, levels=collevs, cmap=cm.gnuplot2)
extent = [min(Time), max(Time), min(F), max(F)]
ax_cont.imshow(S, extent=extent, aspect='auto', origin='lower',
        cmap=cm.gnuplot2, interpolation='nearest')

ax_cont.set_xlim(min(Time),max(Time))
ax_cont.set_ylim(1,0.5*sample_rate)
ax_cont.set_xlabel('Time [s]')
ax_cont.set_ylabel('Frequency [Hz]')

divider = make_axes_locatable(ax_cont)

# time-series
ax_ts = divider.append_axes("top", 1.2, sharex=ax_cont)
ax_ts.plot(time, data)
ax_cont.set_xlim(min(time),max(time))
ax_ts.set_ylim(-1.1*max(abs(data)), 1.1*max(abs(data)))

# fourier spectrum
freq_fourier, Pxx = signal.periodogram(data, fs=sample_rate)
ax_fs = divider.append_axes("right", 1.2, sharey=ax_cont)
ax_fs.semilogx(Pxx,freq_fourier)
ax_fs.set_ylim(1,0.5*sample_rate)
ax_fs.set_xlim(0.01*max(Pxx),1.1*max(Pxx))

pl.setp(ax_ts.get_xticklabels()+ax_fs.get_yticklabels(),visible=False)

pl.tight_layout()


pl.show()



#   #
#   # CWT in scipy
#   #
#   scales  = np.arange(1,256+1)
#   wavelet = signal.ricker
#   cwtcmatr = signal.cwt(data, wavelet, widths)
#
#
#   #
#   # DWT in pyrange(0,2*numpy.pi,numpy.pi/8.)
#   #data = numpy.sin(x**2)
#   scales = numpy.arange(10)
#
#   mother_wavelet = SDG(len_signal = len(data), scales = np.arange(10), normalize = True, fc ='center')
#
#   wavelet = cwt(data, mother_wavelet)
#   wavelet.scalogram(origin = 'bottom')
#
#
#
#   # --- pywt stuff
#   import pywt
#
#   wavelet = 'db4'
#   level = 4
#   order = "freq"  # "normal"
#   interpolation = 'nearest'
#   cmap = cm.hot
#
#   # --- perform wavelet decomposition at levels 1->maxlevel
#   #wp = pywt.WaveletPacket(data, wavelet, 'sym', maxlevel=level)
#   wd = pywt.wavedec(data, wavelet, 'sym', level=level)
#   sys.exit()
#
#   # --- Get the nodes of the decomposition at the level you specified
#   #nodes = wp.get_level(level, order=order)
#
#   # Get the time & frequency resolution at this level
#   Fres = sample_rate / (2**level)
#   Tres = 1.0/Fres
#   time_axis = np.arange(min(time), max(time)+Tres, Tres)
#
#   #labels = [n.path for n in nodes]
#   scales=len(nodes)
#   freqs = pywt.scal2frq(wavelet, (1+np.arange(0,scales))[::-1], 1./sample_rate)
#   labels = ['%.2f'%f for f in freqs]
#   values = np.array([n.data for n in nodes], 'd')
#   values = abs(values)**2
#
#
#   # plotting:
#
#   f = pl.figure()
#   f.subplots_adjust(hspace=0.2, bottom=.03, left=.07, right=.97, top=.92)
#   pl.subplot(2, 1, 1)
#   pl.title("linchirp signal")
#   pl.plot(time, data, 'b')
#
#   ax = pl.subplot(2, 1, 2)
#   pl.title("Wavelet packet coefficients at level %d" % level)
#   pl.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto",
#       origin="lower", extent=[0, max(time), 0, max(freqs)])
#
#   pl.show()

