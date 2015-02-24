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
    U = U[:,ind]
    S = S[ind]
    V = V[:,ind]

    # See e.g.,:
    # http://en.wikipedia.org/wiki/Principal_component_analysis#Singular_value_decomposition

    # Score matrix:
    PCs = U * S

    return PCs, V, S**2 #Betas

def apply_window(data, padding):
    """
    Padding: number of samples for roll-off
    """
    beta = 2*float(padding) / len(data)
    win = lal.CreateTukeyREAL8Window(len(data), beta)
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

    full_spectrum[idx] = np.copy(catwave[catidx])

    return full_spectrum


def comp_match(freqseries1, freqseries2, delta_f=2.0, flow=10., fhigh=8192,
        weighted=False):
    """ 
    """

    tmp1 = pycbc.types.FrequencySeries(initial_array=freqseries1, delta_f=delta_f)
    tmp2 = pycbc.types.FrequencySeries(initial_array=freqseries2, delta_f=delta_f)

    if weighted:

        # make psd
        flen = len(tmp1)
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
        lalsim.SIM_INSPIRAL_TAPER_STARTEND)
        #lalsim.SIM_INSPIRAL_TAPER_START)

    return timeseries.data.data


def reconstruct_signal(magPCs, magBetas, phasePCs, phaseBetas, nPCs):

    mag = np.zeros(np.shape(magPCs)[0], dtype=complex)
    phase = np.zeros(np.shape(phasePCs)[0], dtype=complex)
    for n in xrange(nPCs):
        mag += magBetas[n] * magPCs[:,n]
        phase += phaseBetas[n] * phasePCs[:,n]

    reconstruction = mag*np.exp(1j*phase)

    return reconstruction/scipy.linalg.norm(reconstruction)

def build_hf_component(signal_freqs, peak_width, \
        magPCs, magBetas, phasePCs, phaseBetas, use_npcs, fpeak):

    # Reconstruct the peak
    reconstruction = reconstruct_signal(magPCs, magBetas, phasePCs,
            phaseBetas, use_npcs)

    # Determine the range of frequencies spanned by the data in catalogue form
    delta_f = np.diff(signal_freqs)[0]
    catfreq_low  = fpeak - 0.5*len(reconstruction)*delta_f
    catfreq_high = fpeak + 0.5*len(reconstruction)*delta_f
    cat_freqs = np.arange(catfreq_low, catfreq_high, delta_f)

    # Determine the desired range of frequencies
    freq_low  = fpeak - 0.5*peak_width
    freq_high = fpeak + 0.5*peak_width

    hf_component = catwave_to_spectrum(reconstruction, cat_freqs, signal_freqs,
            freq_low, freq_high)

    return hf_component

def stitch_highlow(lf_component, hf_component):
    full_spectrum = np.zeros(len(lf_component), dtype=complex)

    lf_nonzero = np.argwhere(abs(lf_component)>0)
    hf_nonzero = np.argwhere(abs(hf_component)>0)

    # populate output array
    full_spectrum[lf_nonzero] = lf_component[lf_nonzero]
    full_spectrum[hf_nonzero] += hf_component[hf_nonzero]

    return full_spectrum

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def align_peaks(data, outlen):
    aligned_data = np.zeros(outlen, dtype=complex)
    peak_idx = np.argmax(abs(data))
    start_idx = align_idx - peak_idx
    aligned_data[start_idx:start_idx+len(data)] = np.copy(data)
    
    return aligned_data

def complex_to_polar(catalogue):

    magnitudes = np.zeros(shape=np.shape(catalogue))
    phases = np.zeros(shape=np.shape(catalogue))
    for c in xrange(np.shape(catalogue)[1]):
        magnitudes[:,c] = abs(catalogue[:,c])
        phases[:,c] = np.unwrap(np.angle(catalogue[:,c]))

    return magnitudes, phases


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
fpeaks = np.zeros(npcs)

# Preallocate arrays for low and high frequency catalogues
low_cat  = np.zeros(shape=(4097, len(waveform_names)), dtype=complex)
high_cat = np.zeros(shape=(4097, len(waveform_names)), dtype=complex)
full_cat = np.zeros(shape=(4097, len(waveform_names)), dtype=complex)
early_full_cat = np.zeros(shape=(4097, len(waveform_names)), dtype=complex)
late_full_cat = np.zeros(shape=(4097, len(waveform_names)), dtype=complex)
time_cat = np.zeros(shape=(8192, len(waveform_names)))
early_time_cat = np.zeros(shape=(8192, len(waveform_names)))
late_time_cat = np.zeros(shape=(8192, len(waveform_names)))

low_comp = [1000., 2000.]
high_comp = [2000., 5000.]

peak_width = 500.
delay = 1e-3

align_idx=2048
for w, name in enumerate(waveform_names):

    # Create waveform
    waveform = pmns_utils.Waveform(name)
    waveform.compute_characteristics()
    waveform.reproject_waveform()
    waveform.hplus.data = taper(waveform.hplus.data)
    fpeaks[w] = waveform.fpeak

    # Use unit-norm waveforms
    waveform.hplus.data /= waveform.hrss_plus

    # Find merger
    tpeak = np.floor(np.argmax(waveform.hplus.data) +
            delay/waveform.hplus.delta_t)

    # High-pass at 1 kHz
    #waveform.hplus = pycbc.filter.highpass(waveform.hplus, 1000)

    # Zero-pad / divide into early / late signals
    signal = pycbc.types.TimeSeries(np.zeros(8192), delta_t=waveform.hplus.delta_t)
    signal.data[:len(waveform.hplus)] = np.copy(waveform.hplus.data)
    time_cat[:,w] = np.copy(signal.data)

    early_signal = pycbc.types.TimeSeries(np.zeros(8192), delta_t=waveform.hplus.delta_t)
    early_signal.data[:tpeak] = np.copy(waveform.hplus.data[:tpeak])
    early_signal.data = apply_window(early_signal.data, delay/waveform.hplus.delta_t)
    early_time_cat[:,w] = np.copy(early_signal.data)

    late_signal = pycbc.types.TimeSeries(np.zeros(8192), delta_t=waveform.hplus.delta_t)
    late_signal.data[tpeak:tpeak+len(waveform.hplus.data[tpeak:])] = \
            np.copy(waveform.hplus.data[tpeak:])
    late_signal.data = apply_window(late_signal.data, delay/waveform.hplus.delta_t)
    late_time_cat[:,w] = np.copy(late_signal.data)

    signal_spectrum = signal.to_frequencyseries() 
    early_signal_spectrum = early_signal.to_frequencyseries() 
    late_signal_spectrum  = late_signal.to_frequencyseries() 

    # Select out the low and high frequency parts
    idx_low = (signal_spectrum.sample_frequencies.data>=low_comp[0])*\
            (signal_spectrum.sample_frequencies.data<low_comp[1])
    idx_high = (signal_spectrum.sample_frequencies.data>waveform.fpeak-0.5*peak_width) *\
            (signal_spectrum.sample_frequencies.data<waveform.fpeak+0.5*peak_width)

    # Chop out the high-frequency part and align the peak
    high_component = signal_spectrum[idx_high]
    low_component  = signal_spectrum[idx_low]
    #high_component = late_signal_spectrum[idx_high]
    #low_component  = early_signal_spectrum[idx_low]

    #
    # Feature Alignment
    #
    peak_spectrum = align_peaks(high_component, 4097)

    #
    # Smooth out edges introduced from truncating features
    # 
    smooth=False
    if smooth:

        idx = range(int(align_idx-0.5*peak_width/signal_spectrum.delta_f),
                int(align_idx+0.5*peak_width/signal_spectrum.delta_f))

        peak_spectrum = apply_window(peak_spectrum[idx])
        peak_spectrum = align_peaks(peak_spectrum, 4097)
        #low_component = apply_window(low_component)

    # Build the low-freq-only spectrum
    low_spectrum  = np.zeros(4097,dtype=complex)
    low_spectrum[idx_low] = np.copy(low_component)

    # Populate the catalogues
    low_cat[:,w]  = np.copy(low_spectrum)
    high_cat[:,w] = np.copy(peak_spectrum)
    full_cat[:,w] = np.copy(signal_spectrum.data)
    early_full_cat[:,w] = np.copy(early_signal_spectrum.data)
    late_full_cat[:,w] = np.copy(late_signal_spectrum.data)

    # PCA conditioning
#   for n in xrange(np.shape(low_cat)[1]):
#       low_cat[:,n]  -= np.mean(low_cat, axis=1)
#       high_cat[:,n] -= np.mean(high_cat, axis=1)
#       full_cat[:,n] -= np.mean(full_cat, axis=1)

#       low_cat[:,n]  /= np.std(low_cat, axis=1)
#       high_cat[:,n] /= np.std(high_cat, axis=1)
#       full_cat[:,n] /= np.std(full_cat, axis=1)

#
# PCA

low_mags, low_phases = complex_to_polar(low_cat)
magPCs_low,  magBetas_low,  magS_low = pca(low_mags)
phasePCs_low,  phaseBetas_low,  phaseS_low = pca(low_phases)

high_mags, high_phases = complex_to_polar(high_cat)
magPCs_high,  magBetas_high,  magS_high = pca(high_mags)
phasePCs_high,  phaseBetas_high,  phaseS_high = pca(high_phases)

full_mags, full_phases = complex_to_polar(full_cat)
magPCs_full,  magBetas_full,  magS_full = pca(full_mags)
phasePCs_full,  phaseBetas_full,  phaseS_full = pca(full_phases)

# Amplitude & Phase-maximised match
low_matches = np.zeros(shape=(len(waveform_names),npcs))
high_matches = np.zeros(shape=(len(waveform_names),npcs))
full_matches = np.zeros(shape=(len(waveform_names),npcs))
synth_matches = np.zeros(shape=(len(waveform_names),npcs))

# Dot products (for sanity)
low_dots = np.zeros(shape=(len(waveform_names),npcs))
high_dots = np.zeros(shape=(len(waveform_names),npcs))
full_dots = np.zeros(shape=(len(waveform_names),npcs))
synth_dots = np.zeros(shape=(len(waveform_names),npcs))

freqaxis = signal_spectrum.sample_frequencies.data



for n in xrange(len(waveform_names)):

    print '%d of %d'%(n, len(waveform_names))

    # Loop over the number of pcs to use
    for u, use_npcs in enumerate(xrange(1,11)):

        full_rec_spectrum = \
                reconstruct_signal(magPCs_full, magBetas_full[n,:], \
                phasePCs_full, phaseBetas_full[n,:], use_npcs)

        low_rec_spectrum = \
                reconstruct_signal(magPCs_low, magBetas_low[n,:], \
                phasePCs_low, phaseBetas_low[n,:], use_npcs)

        low_rec_spectrum_0 = \
                reconstruct_signal(magPCs_low, magBetas_low[n,:], \
                phasePCs_low, phaseBetas_low[n,:], 0)

        high_rec_simple = \
                reconstruct_signal(magPCs_high, magBetas_high[n,:], \
                phasePCs_high, phaseBetas_high[n,:], use_npcs)

        high_rec_spectrum = \
                build_hf_component(freqaxis, peak_width, \
                magPCs_high, magBetas_high[n,:], \
                phasePCs_high, phaseBetas_high[n,:], \
                use_npcs, fpeaks[n])

        synth_rec_spectrum = stitch_highlow(low_rec_spectrum, high_rec_spectrum)

        # XXX: Look closely at frequency ranges for matches

        # Low-Freq component: compute match
        #low_matches[n,u] = comp_match(low_rec_spectrum, early_full_cat[:,n],
        low_matches[n,u] = comp_match(low_rec_spectrum, full_cat[:,n],
                flow=low_comp[0], fhigh=low_comp[1], weighted=True)
                #flow=1000, fhigh=8192, weighted=True)

        low_dots[n,u] = \
                np.vdot(low_rec_spectrum/np.linalg.norm(low_rec_spectrum),
                        low_cat[:,n]/np.linalg.norm(low_cat[:,n]))

        # High-Freq component: compute match around the peak
        #high_matches[n,u] = comp_match(high_rec_spectrum, late_full_cat[:,n],
        high_matches[n,u] = comp_match(high_rec_spectrum, full_cat[:,n],
                flow=fpeaks[n]-0.5*peak_width, fhigh=fpeaks[n]+0.5*peak_width,
                weighted=True)
                #flow=1000, fhigh=8192,
                #weighted=True)

        high_dots[n,u] = \
                np.vdot(high_rec_simple/np.linalg.norm(high_rec_simple),
                        high_cat[:,n]/np.linalg.norm(high_cat[:,n]))

        # Full-spectrum (naive)
        full_matches[n,u] = comp_match(full_rec_spectrum, full_cat[:,n],
                flow=1000, fhigh=8192, weighted=True)

        full_dots[n,u] = \
                np.vdot(full_rec_spectrum/np.linalg.norm(full_rec_spectrum),
                        full_cat[:,n]/np.linalg.norm(full_cat[:,n]))
 
        # Full synthesised signal: compute match
        synth_matches[n,u] = comp_match(synth_rec_spectrum, full_cat[:,n],
                flow=1000, fhigh=8192, weighted=True)
#
# PCA diagnostics
#
mag_low_eigenergies = eigenenergy(magS_low)
mag_high_eigenergies = eigenenergy(magS_high)
mag_full_eigenergies = eigenenergy(magS_full)
phase_low_eigenergies = eigenenergy(phaseS_low)
phase_high_eigenergies = eigenenergy(phaseS_high)
phase_full_eigenergies = eigenenergy(phaseS_full)

# Get minimal matches
low_minimal_match = np.zeros(npcs)
for m in xrange(npcs):
    low_minimal_match[m] = min(low_matches[:,m])

high_minimal_match = np.zeros(npcs)
for m in xrange(npcs):
    high_minimal_match[m] = min(high_matches[:,m])

full_minimal_match = np.zeros(npcs)
for m in xrange(npcs):
    full_minimal_match[m] = min(full_matches[:,m])

synth_minimal_match = np.zeros(npcs)
for m in xrange(npcs):
    synth_minimal_match[m] = min(synth_matches[:,m])


#####################################################
# Plotting

f, ax = pl.subplots(nrows=2, figsize=(8,6), sharex=True)
npcs_axis = range(1,11)

# Eignenergies
ax[0].plot(npcs_axis, mag_full_eigenergies, label='full spectrum', color='b')
ax[0].plot(npcs_axis, mag_low_eigenergies, label='low-F component', color='g')
ax[0].plot(npcs_axis, mag_high_eigenergies, label='high-F component', color='r')

ax[0].plot(npcs_axis, phase_full_eigenergies, color='b', linestyle='--')
ax[0].plot(npcs_axis, phase_low_eigenergies, color='g', linestyle='--')
ax[0].plot(npcs_axis, phase_high_eigenergies, color='r', linestyle='--')

ax[0].legend(loc='lower right')
#ax[0].set_ylim(0,1)
#ax[0].set_xlabel('Number of PCs')
ax[0].set_ylabel('Eigenenergy')
ax[0].minorticks_on()

# Minimal match
ax[1].plot(npcs_axis, full_minimal_match, label='full spectrum')
ax[1].plot(npcs_axis, low_minimal_match, label='low-F component')
ax[1].plot(npcs_axis, high_minimal_match, label='high-F component')
ax[1].plot(npcs_axis, synth_minimal_match, label='synthesised waveform')
ax[1].legend(loc='lower right')
#ax[1].set_ylim(0,1)
ax[1].set_xlabel('Number of PCs')
ax[1].set_ylabel('Minimal Match')
ax[1].minorticks_on()

pl.show()
sys.exit()

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
ax[0][0].plot(np.real(PCs_low))
ax[0][1].plot(np.imag(PCs_low))
ax[1][0].plot(np.real(Uhigh))
ax[1][1].plot(np.real(Uhigh))
ax[2][0].plot(np.fft.ifft(PCs_low))
ax[2][1].plot(np.fft.ifft(Uhigh))


ax[0][0].set_xlim(0,500)
ax[0][1].set_xlim(0,500)
ax[1][0].set_xlim(2048-250,2048+250)
ax[1][1].set_xlim(2048-250,2048+250)

pl.show()


