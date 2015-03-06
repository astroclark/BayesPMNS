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
import scipy.sparse.linalg
from scipy import signal

import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as pl

import lal
import lalsimulation as lalsim
import pmns_utils

import pycbc.types
import pycbc.filter
from pycbc.psd import aLIGOZeroDetHighPower

def pca(catalogue, nonzeroidx=None):

    # Shift and scale
    #means = np.mean(catalogue, axis=1)
    #stds = np.std(catalogue, axis=1)
    #for o,obs in enumerate(xrange(np.shape(catalogue)[1])):
    #    catalogue[:,o] -= means
        #idx=stds>0
        #catalogue[idx,o] /= stds[idx]

    # work with non-zero parts
    if nonzeroidx is not None:
        U, S, Vt = scipy.linalg.svd(catalogue[nonzeroidx,:], full_matrices=True)
    else:
        U, S, Vt = scipy.linalg.svd(catalogue, full_matrices=True)
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
    if nonzeroidx is not None:
        PCs = np.zeros(shape=np.shape(catalogue))
        PCs[nonzeroidx,:] = np.copy(U * S)
    else:
        PCs = U * S

    return PCs, V, S**2 #Betas

def pcacov(catalogue):

    # Shift and scale
    #means = np.mean(catalogue, axis=1)
    #stds = np.std(catalogue, axis=1)
    #for o,obs in enumerate(xrange(np.shape(catalogue)[1])):
    #    catalogue[:,o] -= means
        #idx=stds>0
        #catalogue[idx,o] /= stds[idx]

    H = np.matrix(catalogue)

    # --- 1) Compute H^T * H - left side of equation (5) in 0810.5707
    HTH = H.T * H

    # --- 2) Compute eigenvectors (V) and eigenvalues (S) of H^T * H
    #S, V = np.linalg.eigh(HTH)
    S, V = np.linalg.eig(HTH)

    # --- 3) Sort eigenvectors in descending order
    idx = np.argsort(S)[::-1]
    V = V[:,idx]
    S = S[idx]

    # --- 4) Sorted Eigenvectors of covariance matrix, C = PCs
    # To get these, note:  H.(H^T.H) = M*C.H, since C = 1/M * H.H^T
    # so H. eigenvectors of HTH are the eigenvectors of C
    # i..e.,  C.U = s*U
    #         (1/M)*H.H^T . U = s*U
    #         U = H.V, V = eigenvectors of H^T.H
    U = H*V 

    U = np.array(U)
    S = np.array(S)
    V = np.array(V)

    # normalise PCs
    #for i in xrange(np.shape(U)[1]):
    #    U[:,i] /= np.linalg.norm(U[:,i])

    return U, V, S

def taper(input_data):
    """  
    """

    timeseries = lal.CreateREAL8TimeSeries('blah', 0.0, 0,
            1.0/16384, lal.StrainUnit, int(len(input_data)))

    lalsim.SimInspiralREAL8WaveTaper(timeseries.data,
        lalsim.SIM_INSPIRAL_TAPER_STARTEND)
        #lalsim.SIM_INSPIRAL_TAPER_START)

    return timeseries.data.data

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


def comp_match(freqseries1, freqseries2, delta_f=1.0, flow=10., fhigh=8192,
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




def reconstruct_signal(magPCs, magBetas, phasePCs, phaseBetas, nMagPCs, nPhasePCs):

    mag = np.zeros(np.shape(magPCs)[0], dtype=complex)
    phase = np.zeros(np.shape(phasePCs)[0], dtype=complex)

    for n in xrange(nMagPCs):
        mag += magBetas[n] * magPCs[:,n]
    for n in xrange(nPhasePCs):
        phase += phaseBetas[n] * phasePCs[:,n]

    reconstruction = mag*np.exp(1j*phase)

    return reconstruction#/scipy.linalg.norm(reconstruction)


def complex_to_polar(catalogue):

    magnitudes = np.zeros(shape=np.shape(catalogue))
    phases = np.zeros(shape=np.shape(catalogue))
    for c in xrange(np.shape(catalogue)[1]):
        magnitudes[:,c] = abs(catalogue[:,c])
        phases[:,c] = np.unwrap(np.angle(catalogue[:,c]))

    return magnitudes, phases

def build_hf_component(magPCs, magBetas, phasePCs, phaseBetas, nMagPCs,
        nPhasePCs, fpeak, delta_f, fhet=4000.0):

    # Reconstruct:
    reconstruction = reconstruct_signal(magPCs, magBetas, phasePCs, phaseBetas,
            nMagPCs, nPhasePCs)

    # Re-position the spectrum
    reconstruction_FD = pycbc.types.FrequencySeries(reconstruction,
            delta_f=delta_f)

    specdata = np.zeros(len(reconstruction), dtype=complex)

    # Find where the peak should be
    peakidx = np.argmin(abs(reconstruction_FD.sample_frequencies-fpeak))

    # Start populating the output spectrum at this location
    false_freqs = reconstruction_FD.sample_frequencies.data - (fhet-fpeak)
    specdata[:sum(false_freqs>=0)] = reconstruction[false_freqs>=0]
#
#    pl.figure()
#   pl.plot(reconstruction_FD.sample_frequencies, abs(reconstruction))
#    pl.plot(false_freqs, abs(reconstruction))
#    pl.plot(false_freqs[pos_false_freqs], abs(reconstruction[pos_false_freqs]))
#    pl.plot(reconstruction_FD.sample_frequencies, abs(specdata))
#    pl.axvline(fpeak, color='r')
#    pl.show()
#    sys.exit()
#
#   print specdata


    return specdata

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
                'sfho_135135_lessvisc']#,
#               'gs1_135135',
#               'gs2_135135',
#               'ls220_135135',
#               'ls375_135135',
#               'sly4_135135'
#               ]

npcs = len(waveform_names)
fpeaks = np.zeros(npcs)

# Preallocate arrays for low and high frequency catalogues
low_cat  = np.zeros(shape=(8193, len(waveform_names)), dtype=complex)
high_cat_aligned = np.zeros(shape=(8193, len(waveform_names)), dtype=complex)
high_cat = np.zeros(shape=(8193, len(waveform_names)), dtype=complex)
full_cat = np.zeros(shape=(8193, len(waveform_names)), dtype=complex)
time_cat = np.zeros(shape=(16384, len(waveform_names)))


fhet = 4096.0
peak_width = 500.

knee_freq=2000
butter_b_low, butter_a_low = signal.butter(8, knee_freq / 8192.0, btype='lowpass')
butter_b_band, butter_a_band = signal.butter(8, \
        [(fhet-0.5*peak_width) / 8192.0, (fhet+0.5*peak_width) / 8192.0],
            btype='bandpass')

align_idx=4096
for w, name in enumerate(waveform_names):

    print 'loading ', name

    # Create waveform
    waveform = pmns_utils.Waveform(name)
    waveform.reproject_waveform()
    waveform.compute_characteristics()
    waveform.hplus.data = taper(waveform.hplus.data)
    fpeaks[w] = waveform.fpeak

    # Use unit-norm waveforms
    waveform.hplus.data /= waveform.hrss_plus

    # High-pass at 1 kHz
    waveform.hplus = pycbc.filter.highpass(waveform.hplus, 1000)

    # Zero-pad
    rawdata = np.zeros(16384)
    tmptimes = np.arange(0,1,1./16384)
    rawdata[:len(waveform.hplus)] = np.copy(waveform.hplus.data)
    time_cat[:,w] = np.copy(rawdata)

    original_signal = pycbc.types.TimeSeries(np.copy(rawdata), delta_t = waveform.hplus.delta_t)
    original_signal_spectrum = original_signal.to_frequencyseries()

    # --- Construct Low frequency catalogue
    lowdata = signal.filtfilt(butter_b_low, butter_a_low, np.copy(rawdata))
    low_signal = pycbc.types.TimeSeries(lowdata, delta_t=waveform.hplus.delta_t)
    low_signal_spectrum = low_signal.to_frequencyseries()

    # --- Construct High frequency catalogue
    # Heterodyne
    highdata = np.copy(rawdata)
    fmix = fhet - waveform.fpeak
    mixsignal = np.sin(2*np.pi*fmix*tmptimes)
    highdata *= mixsignal

    high_signal = pycbc.types.TimeSeries(highdata,
            delta_t=waveform.hplus.delta_t)

    # f-domain filter
    high_signal_spectrum = high_signal.to_frequencyseries() 
    idx_passfreqs = (high_signal_spectrum.sample_frequencies.data>fhet-0.5*peak_width) *\
            (high_signal_spectrum.sample_frequencies.data<fhet+0.5*peak_width)
    high_signal_spectrum.data[~idx_passfreqs] = 0.0

    # t-domain filter
#   high_signal = signal.filtfilt(butter_b_band, butter_a_band,
#       mixsignal*rawdata)
#    high_signal = pycbc.filter.highpass(high_signal, fhet-0.5*peak_width,
#           filter_order=12)
#   high_signal = pycbc.types.TimeSeries(high_signal,
#           delta_t=waveform.hplus.delta_t)
#   high_signal_spectrum = high_signal.to_frequencyseries() 
#
    # Populate the catalogues
    low_cat[:,w]  = np.copy(low_signal_spectrum.data)
    high_cat[:,w]  = np.copy(high_signal_spectrum.data)
    full_cat[:,w] = np.copy(original_signal_spectrum.data)

    # Tidy up
    del waveform, rawdata, highdata, lowdata


freqaxis = original_signal_spectrum.sample_frequencies.data

#pl.figure()
#pl.plot(freqaxis, abs(full_cat))
#pl.plot(freqaxis, abs(low_cat))
#pl.plot(freqaxis, abs(high_cat))
#pl.show()
#sys.exit()
#
#pl.figure()
#pl.plot(freqaxis, abs(high_cat))
#pl.show()
#
#pl.figure()
#pl.plot(freqaxis, abs(low_cat))
#pl.show()

#sys.exit()
#
# Polar data
#
low_mags, low_phases = complex_to_polar(low_cat)
high_mags, high_phases = complex_to_polar(high_cat)
full_mags, full_phases = complex_to_polar(full_cat)

#
# Phase fit
#

#
# PCA

magPCs_low,  magBetas_low,  magS_low = pca(low_mags, low_mags[:,0]>0)
phasePCs_low,  phaseBetas_low,  phaseS_low = pca(low_phases, low_mags[:,0]>0)

magPCs_high,  magBetas_high,  magS_high = pca(high_mags, high_mags[:,0]>0)
phasePCs_high,  phaseBetas_high,  phaseS_high = pca(high_phases, high_mags[:,0]>0)

magPCs_full,  magBetas_full,  magS_full = pca(full_mags)
phasePCs_full,  phaseBetas_full,  phaseS_full = pca(full_phases)

# Amplitude & Phase-maximised match
low_matches = np.zeros(shape=(len(waveform_names),npcs))
high_matches = np.zeros(shape=(len(waveform_names),npcs))
full_matches = np.zeros(shape=(len(waveform_names),npcs))

# Dot products (for sanity)
low_dots = np.zeros(shape=(len(waveform_names),npcs))
high_dots = np.zeros(shape=(len(waveform_names),npcs))
full_dots = np.zeros(shape=(len(waveform_names),npcs))

for n in xrange(len(waveform_names)):

    print '%d of %d'%(n, len(waveform_names))

    # Loop over the number of pcs to use
    for u, use_npcs in enumerate(xrange(1,npcs+1)):

        use_n_mag_pcs = np.copy(use_npcs)
        use_n_phase_pcs = np.copy(use_npcs)

        full_rec_spectrum = \
                reconstruct_signal(magPCs_full, magBetas_full[n,:],
                phasePCs_full, phaseBetas_full[n,:], use_n_mag_pcs,
                use_n_phase_pcs)

        low_rec_spectrum = \
                reconstruct_signal(magPCs_low, magBetas_low[n,:],
                phasePCs_low, phaseBetas_low[n,:], use_n_mag_pcs,
                use_n_phase_pcs)

        high_rec_spectrum = build_hf_component(magPCs_high, magBetas_high[n,:],
                phasePCs_high, phaseBetas_high[n,:], use_n_mag_pcs,
                use_n_phase_pcs, fpeaks[n], delta_f =
                high_signal_spectrum.delta_f)

        # Also reconstruct the high-freq part without frequency shifting, so we
        # can assess the high-freq only reconstruction fidelity
        high_rec_spectrum_simple = \
                reconstruct_signal(magPCs_high, magBetas_high[n,:],
                phasePCs_high, phaseBetas_high[n,:], use_n_mag_pcs,
                use_n_phase_pcs)

        synth_rec_spectrum = low_rec_spectrum + high_rec_spectrum

        # Low-Freq component: compute match
        low_matches[n,u] = comp_match(low_rec_spectrum, low_cat[:,n],
                flow=1000, fhigh=8192, weighted=False)

        low_dots[n,u] = \
                np.vdot(low_rec_spectrum/np.linalg.norm(low_rec_spectrum),
                        low_cat[:,n]/np.linalg.norm(low_cat[:,n]))

        # High-Freq component: compute match around the peak
        high_matches[n,u] = comp_match(high_rec_spectrum_simple, high_cat[:,n],
                flow=1000, fhigh=8192,
                weighted=True)

        high_dots[n,u] = \
                np.vdot(high_rec_spectrum_simple/np.linalg.norm(high_rec_spectrum_simple),
                        high_cat[:,n]/np.linalg.norm(high_cat[:,n]))

        # Full-spectrum (naive)
        full_matches[n,u] = comp_match(full_rec_spectrum, full_cat[:,n],
                flow=1000, fhigh=8192, weighted=True)

        full_dots[n,u] = \
                np.vdot(full_rec_spectrum/np.linalg.norm(full_rec_spectrum),
                        full_cat[:,n]/np.linalg.norm(full_cat[:,n]))

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

oneD=True
if oneD:
    f, ax = pl.subplots(nrows=2, figsize=(8,6), sharex=True)
    npcs_axis = range(1,npcs+1)
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

    # 1D-Minimal match
    ax[1].plot(npcs_axis, full_minimal_match, label='full spectrum')
    ax[1].plot(npcs_axis, low_minimal_match, label='low-F component')
    ax[1].plot(npcs_axis, high_minimal_match, label='high-F component')
    ax[1].legend(loc='lower right')
    #ax[1].set_ylim(0,1)
    ax[1].set_xlabel('Number of PCs')
    ax[1].set_ylabel('Minimal Match')
    ax[1].minorticks_on()

    pl.show()

#sys.exit()

#
# Now look at fully synthesised results
#
synth_matches = np.zeros(shape=(len(waveform_names), len(waveform_names)+1,
    len(waveform_names)+1))

for n in xrange(len(waveform_names)):

    print '%d of %d'%(n, len(waveform_names))

    # Loop over the number of pcs to use for low and high parts
    for u, low_use_npcs in enumerate(xrange(0,npcs+1)):

        low_use_n_mag_pcs = np.copy(low_use_npcs)
        low_use_n_phase_pcs = np.copy(low_use_npcs)

        for v, high_use_npcs in enumerate(xrange(0,npcs+1)):

        #for n in xrange(9):

            high_use_n_mag_pcs = np.copy(high_use_npcs)
            high_use_n_phase_pcs = np.copy(high_use_npcs)

        #   low_use_n_mag_pcs = 3
        #   low_use_n_phase_pcs = 3
        #   high_use_n_mag_pcs = 3
        #   high_use_n_phase_pcs = 3

            if u==0 and v==0:
                synth_matches[n,u,v] = 0.0
            else:
                if u==0:
                    low_rec_spectrum = np.zeros(8193)
                else:
                    low_rec_spectrum = \
                            reconstruct_signal(magPCs_low, magBetas_low[n,:], \
                            phasePCs_low, phaseBetas_low[n,:],
                            low_use_n_mag_pcs, low_use_n_phase_pcs)


                if v==0:
                    high_rec_spectrum = np.zeros(8193)
                else:
                    high_rec_spectrum = \
                            build_hf_component(magPCs_high, magBetas_high[n,:],
                                    phasePCs_high, phaseBetas_high[n,:],
                                    high_use_n_mag_pcs, high_use_n_phase_pcs,
                                    fpeaks[n], delta_f =
                                    high_signal_spectrum.delta_f)

                # Full synthesised signal: compute match
                if v>0:
                    peak_height = abs(full_cat[np.argmin(abs(freqaxis-fpeaks[n])), n])
                    high_rec_spectrum *= peak_height/max(abs(high_rec_spectrum))
                synth_rec_spectrum = low_rec_spectrum + high_rec_spectrum

            synth_matches[n,u,v] = comp_match(synth_rec_spectrum, full_cat[:,n],
                    flow=1000, fhigh=8192, weighted=True)

synth_minimal_match = np.zeros(shape=(npcs+1,npcs+1))
for m in xrange(npcs+1):
    for n in xrange(npcs+1):
        synth_minimal_match[m,n] = min(synth_matches[:,m,n])
        print waveform_names[np.argmin(synth_matches[:,m,n])],synth_minimal_match[m,n]

if 0:
    #####################################################
    # Ad Hoc matches

    test_waveform = pmns_utils.Waveform('apr_135135_lessvisc')

    test_waveform.reproject_waveform()
    test_waveform.compute_characteristics()
    test_signal = pycbc.types.TimeSeries(np.zeros(16384), delta_t=waveform.hplus.delta_t)
    test_signal.data[:len(test_waveform.hplus)] = np.copy(test_waveform.hplus.data)

    for w in range(len(waveform_names)):

        low_rec_spectrum = reconstruct_signal(magPCs_low, magBetas_low[w,:],
                phasePCs_low, phaseBetas_low[w,:], 1,
                low_use_n_phase_pcs)

        high_rec_spectrum = build_hf_component(freqaxis, peak_width, magPCs_high,
                magBetas_high[w,:], phasePCs_high, phaseBetas_high[w,:],
                high_use_n_mag_pcs, 1, waveform.fpeak)

        #high_rec_spectrum = np.zeros(len(low_rec_spectrum))

        synth_rec_spectrum = stitch_highlow(low_rec_spectrum, high_rec_spectrum)

        print comp_match(synth_rec_spectrum, test_signal.to_frequencyseries(),
                                flow=1000, fhigh=8192, weighted=True)

#####################################################
# Plotting



# 2D-Minimal Match for synthesised waveforms
fig, ax = pl.subplots()

#text portion
xvals = range(0,len(waveform_names)+1)
yvals = range(0,len(waveform_names)+1)

im = ax.imshow(np.transpose(synth_minimal_match), interpolation='nearest', \
        extent=[min(xvals)-0.5,max(xvals)+0.5,min(yvals)-0.5,max(yvals)+0.5], origin='lower')

for x,xval in enumerate(xvals):
    for y,yval in enumerate(yvals):
        if synth_minimal_match[x,y]<0.85:
            ax.text(xval, yval, '%.2f'%(synth_minimal_match[x,y]), \
                va='center', ha='center', color='w')
        else:
            ax.text(xval, yval, '%.2f'%(synth_minimal_match[x,y]), \
                va='center', ha='center')

ax.set_xticks(xvals)
ax.set_yticks(yvals)

im.set_clim(0.75,1)
im.set_cmap('bone')
c=pl.colorbar(im, ticks=np.arange(0.75,1.05,0.05))

ax.set_xlabel('# of low-freq PCs')
ax.set_ylabel('# of high-freq PCs')

pl.show()
#sys.exit()
