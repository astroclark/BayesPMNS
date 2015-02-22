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
"""

from __future__ import division
import os,sys
import numpy as np
import scipy.linalg

import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as pl

import lal
import lalsimulation as lalsim
import pmns_utils

import pycbc.types
import pycbc.filter
from pycbc.psd import aLIGOZeroDetHighPower

def pca(catalogue):

    U, S, Vt = scipy.linalg.svd(catalogue, full_matrices=False)
    V = Vt.T

    # sort the PCs by descending order of the singular values (i.e. by the
    # proportion of total variance they explain)
    ind = np.argsort(S)[::-1]
    U = U[:, ind]
    S = S[ind]
    V = V[:, ind]

    return U, S, V

def apply_window(data):
    win = lal.CreateTukeyREAL8Window(len(data), 0.9)
    return data*win.data.data

def eigenenergy(eigenvalues):
    """
    """
    eigenvalues=abs(eigenvalues)
    # See:
    # http://en.wikipedia.org/wiki/Principal_component_analysis#Compute_the_cumulative_energy_content_for_each_eigenvector
    gp = sum(eigenvalues)
    gj=np.zeros(len(eigenvalues))
    for e,i in enumerate(eigenvalues):
        for a in range(e+1):
            gj[e]+=eigenvalues[a]
    gj/=gp

    return gj

def catwave_to_spectrum(catwave, catfreqs, freqaxis, freq_low, freq_high):
    """
    Reconstruct the original spectrum from the data in catwave which may have
    been shifted
    """

    # populate a new spectrum with the waveform data at the correct location
    full_spectrum = np.zeros(np.shape(catwave), dtype=complex)

    # Indices for location of spectral data
    idx = (freqaxis>=freq_low) * (freqaxis<freq_high)
    catidx = (catfreqs>=freq_low) * (catfreqs<freq_high)

    full_spectrum[idx] = catwave[catidx]

    return full_spectrum


def comp_match(timeseries1, timeseries2, delta_t=1./16384, flow=10., fhigh=8192,
        weighted=False):
    """ 
    """

    tmp1 = pycbc.types.TimeSeries(initial_array=timeseries1, delta_t=delta_t)
    tmp2 = pycbc.types.TimeSeries(initial_array=timeseries2, delta_t=delta_t)

    if weighted:

        # make psd
        flen = len(tmp1.to_frequencyseries())
        delta_f = np.diff(tmp1.to_frequencyseries().sample_frequencies)[0]
        psd = aLIGOZeroDetHighPower(flen, delta_f, low_freq_cutoff=flow)


        return pycbc.filter.match(tmp1, tmp2, psd=psd,
                low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)[0]

    else:

        return pycbc.filter.match(tmp1, tmp2, low_frequency_cutoff=flow,
                high_frequency_cutoff=fhigh)[0]


def taper(input_data):
    """  
    """

    timeseries = lal.CreateREAL8TimeSeries('blah', 0.0, 0,
            1.0/16384, lal.StrainUnit, int(len(input_data)))

    lalsim.SimInspiralREAL8WaveTaper(timeseries.data,
        lalsim.SIM_INSPIRAL_TAPER_START)

    return timeseries.data.data


#def pmns_template(low_component, high_component, high_location):


#
# Waveform catalogue
#
waveform_names=['apr_135135_lessvisc',
                'shen_135135_lessvisc',
                'dd2_135135_lessvisc' ,
                'dd2_165165_lessvisc' ,
                'nl3_135135_lessvisc' ,
                'nl3_1919_lessvisc'   ,
                'tm1_135135_lessvisc' ,
                'tma_135135_lessvisc' ,
                'sfhx_135135_lessvisc',
                'sfho_135135_lessvisc']

#               'gs1_135135',
#               'gs2_135135',
#               'ls220_135135',
#               'ls375_135135',
#               'sly4_135135'
#               ]

npcs = len(waveform_names)

# Preallocate arrays for low and high frequency catalogues
low_cat  = np.zeros(shape=(4097, len(waveform_names)), dtype=complex)
high_cat = np.zeros(shape=(4097, len(waveform_names)), dtype=complex)
full_cat = np.zeros(shape=(4097, len(waveform_names)), dtype=complex)
time_cat = np.zeros(shape=(8192, len(waveform_names)))

low_comp = [1000., 2000.]
high_comp = [2000., 5000.]

peak_width = 500.

align_idx=2048
for w, name in enumerate(waveform_names):

    # Create waveform
    waveform = pmns_utils.Waveform(name)
    waveform.compute_characteristics()
    waveform.reproject_waveform()
    htmp = waveform.hplus 
    htmp.data = taper(htmp.data)

    # Use unit-norm waveforms
    htmp.data /= waveform.hrss_plus

    # High-pass at 1 kHz
    htmp = pycbc.filter.highpass(htmp, 1000)

    # window for a smooth start
    #htmp.data=apply_window(htmp.data)

    # Zero-pad
    signal = pycbc.types.TimeSeries(np.zeros(8192), delta_t=htmp.delta_t)
    signal.data[:len(htmp)] = htmp.data
    time_cat[:,w] = signal.data

    signal_spectrum = signal.to_frequencyseries() 

    # Select out the low and high frequency parts
    idx_low = (signal_spectrum.sample_frequencies.data>=low_comp[0])*\
            (signal_spectrum.sample_frequencies.data<low_comp[1])
    idx_high = (signal_spectrum.sample_frequencies.data>waveform.fpeak-0.5*peak_width) *\
            (signal_spectrum.sample_frequencies.data<waveform.fpeak+0.5*peak_width)

    # window for smoothness
    #low_part = apply_window(signal_spectrum[idx_low])
    #high_part= apply_window(signal_spectrum[idx_high])
    low_part = signal_spectrum[idx_low]
    high_part= signal_spectrum[idx_high]

    # align peaks for high-frequency component
    aligned = np.zeros(4097, dtype=complex)
    peak_idx=np.argmax(abs(high_part))
    start_idx = align_idx - peak_idx
    aligned[start_idx:start_idx+len(high_part)] = high_part

    # add to the catalogues
    low_cat[:len(low_part),w] = low_part
    high_cat[:,w] = aligned
    full_cat[:len(signal_spectrum),w] = signal_spectrum

    # PCA conditioning
#   for n in xrange(np.shape(low_cat)[1]):
#       low_cat[:,n]  -= np.mean(low_cat, axis=1)
#       high_cat[:,n] -= np.mean(high_cat, axis=1)
#       full_cat[:,n] -= np.mean(full_cat, axis=1)


#
# PCA
#
Ulow, Slow, Vlow = pca(low_cat)
Uhigh, Shigh, Vhigh = pca(high_cat)
Ufull, Sfull, Vfull = pca(full_cat)


# PCs in time-domain for testing / diagnostics
PClow = np.zeros(shape=(8192, npcs))
for n in xrange(npcs):

    # populate a new spectrum with the waveform data at the correct location!

    # Get time domain PCs for low-frequencies
    freqaxis = signal_spectrum.sample_frequencies.data
    freq_low = low_comp[0]
    freq_high = low_comp[1]

    low_cat_freqs = np.arange(low_comp[0], \
            low_comp[0] + len(low_cat[:,n])*signal_spectrum.delta_f,
            signal_spectrum.delta_f)
    
    full_spectrum = catwave_to_spectrum(Ulow[:,n], low_cat_freqs, freqaxis,
            freq_low, freq_high)

    tmp = pycbc.types.FrequencySeries(full_spectrum,
            delta_f=signal_spectrum.delta_f)

    PClow[:,n] = tmp.to_timeseries()

for w in xrange(npcs-1):
    print comp_match(PClow[:,0], time_cat[:,w], fhigh=2000)

sys.exit()

#####################################################
# Plotting

#
# PCA diagnostics
#
low_eigenergies = eigenenergy(Slow)
high_eigenergies = eigenenergy(Shigh)
full_eigenergies = eigenenergy(Sfull)

f, ax = pl.subplots()
npcs = np.arange(len(Slow))+1
ax.plot(npcs, full_eigenergies, label='full spectrum')
ax.plot(npcs, low_eigenergies, label='low-F component')
ax.plot(npcs, high_eigenergies, label='high-F component')

ax.legend(loc='lower right')
ax.set_ylim(0,1)
ax.minorticks_on()

#pl.show()
#sys.exit()

#
# Plot catalogue
#
f, ax = pl.subplots(nrows=2, ncols=2)
ax[0][0].plot(np.real(low_cat))
ax[0][1].plot(np.imag(low_cat))
ax[1][0].plot(np.real(high_cat))
ax[1][1].plot(np.real(high_cat))


ax[0][0].set_xlim(0,500)
ax[0][1].set_xlim(0,500)
ax[1][0].set_xlim(2048-250,2048+250)
ax[1][1].set_xlim(2048-250,2048+250)

#
# Plot PCs
#
f, ax = pl.subplots(nrows=3, ncols=2)
ax[0][0].plot(np.real(Ulow))
ax[0][1].plot(np.imag(Ulow))
ax[1][0].plot(np.real(Uhigh))
ax[1][1].plot(np.real(Uhigh))
ax[2][0].plot(np.fft.ifft(Ulow))
ax[2][1].plot(np.fft.ifft(Uhigh))


ax[0][0].set_xlim(0,500)
ax[0][1].set_xlim(0,500)
ax[1][0].set_xlim(2048-250,2048+250)
ax[1][1].set_xlim(2048-250,2048+250)

pl.show()


