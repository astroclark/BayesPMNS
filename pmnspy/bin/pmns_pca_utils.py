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
import cPickle as pickle
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

def pca_by_svd(catalogue):
    """
    Perform principle component analysis via singular value decomposition
    """

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
    PCs = U * S

    return PCs, V, S**2 

def pca_magphase(catalogue):
    """
    Do PCA with magnitude and phase parts of the complex waveforms in catalogue
    """

    magnitudes, phases = complex_to_polar(catalogue)

    pcs_magphase = {}

    pcs_magphase['magnitude_pcs'], pcs_magphase['magnitude_betas'], \
            pcs_magphase['magnitude_eigs'] = pca_by_svd(magnitudes)
    pcs_magphase['phase_pcs'], pcs_magphase['phase_betas'], \
            pcs_magphase['phase_eigs'] = pca_by_svd(phases)

    pcs_magphase['magnitude_eigenergy'] = \
            eigenergy(pcs_magphase['magnitude_eigs'])
    pcs_magphase['phase_eigenergy'] = \
            eigenergy(pcs_magphase['phase_eigs'])

    return pcs_magphase


def reconstruct_signal_ampphase(pcs_magphase, nMagPCs, nPhasePCs, waveform_num):

    """
    Build the reconstructed signal from magnitude and phase components
    """


    magPCs = pcs_magphase['magnitude_pcs']
    magBetas = pcs_magphase['magnitude_betas']
    phasePCs = pcs_magphase['phase_pcs']
    phaseBetas = pcs_magphase['phase_betas']

    mag = np.zeros(np.shape(magPCs)[0], dtype=complex)
    phase = np.zeros(np.shape(phasePCs)[0], dtype=complex)

    for n in xrange(nMagPCs):
        mag += magBetas[waveform_num,n] * magPCs[:,n]
    for n in xrange(nPhasePCs):
        phase += phaseBetas[waveform_num,n] * phasePCs[:,n]

    reconstruction = mag*np.exp(1j*phase)

    return reconstruction

def complex_to_polar(catalogue):
    """
    Convert the complex Fourier spectrum to an amplitude and phase
    """

    magnitudes = np.zeros(shape=np.shape(catalogue))
    phases = np.zeros(shape=np.shape(catalogue))
    for c in xrange(np.shape(catalogue)[1]):
        magnitudes[:,c] = abs(catalogue[:,c])
        phases[:,c] = np.unwrap(np.angle(catalogue[:,c]))

    return magnitudes, phases


def build_hf_component(reconstructed_spec, fpeak, delta_f, fcenter=4000.0):

    # Re-position the spectrum
    reconstruction_FD = pycbc.types.FrequencySeries(reconstruction,
            delta_f=delta_f)

    specdata = np.zeros(len(reconstruction), dtype=complex)

    # Find where the peak should be
    peakidx = np.argmin(abs(reconstruction_FD.sample_frequencies-fpeak))

    # Start populating the output spectrum at this location
    false_freqs = reconstruction_FD.sample_frequencies.data - (fcenter-fpeak)
    specdata[:sum(false_freqs>=0)] = reconstruction[false_freqs>=0]

def taper(input_data, delta_t=1./16384):
    """  
    Apply a taper to the start/end of the data in input_data
    """

    timeseries = lal.CreateREAL8TimeSeries('blah', 0.0, 0,
            delta_t, lal.StrainUnit, int(len(input_data)))

    lalsim.SimInspiralREAL8WaveTaper(timeseries.data,
        lalsim.SIM_INSPIRAL_TAPER_STARTEND)

    return timeseries.data.data

def build_catalogues(waveform_names):
    """
    Build the low, high and full catalogues, including data conditioning for the
    waveforms in the list of names waveform_names
    """

    sample_freq=16384
    delta_t=1./16384
    nTsamples=16384
    nFsamples=nTsamples/2 + 1
    times=np.arange(0, delta_t*nTsamples, delta_t)

    fcenter = 4096.0  # center high-frequency peaks here
    peak_width = 500. # retain this much data around the peak for the high-freq
                      # component

    knee_freq=2000 # low pass the data here to form the low-freq component
                   # catalogue

    # Setup butterworth filters
    butter_b_low, butter_a_low = signal.butter(8, knee_freq / (0.5*sample_freq), btype='lowpass')
    butter_b_high, butter_a_high = signal.butter(8, 900 / (0.5*sample_freq),
            btype='highpass')

    butter_b_band, butter_a_band = signal.butter(8, \
            [(fcenter-0.5*peak_width) / (0.5*sample_freq), \
            (fcenter+0.5*peak_width) / (0.5*sample_freq)],
                btype='bandpass')

    # Preallocation
    low_cat  = np.zeros(shape=(nFsamples, len(waveform_names)), dtype=complex)
    high_cat = np.zeros(shape=(nFsamples, len(waveform_names)), dtype=complex)
    full_cat = np.zeros(shape=(nFsamples, len(waveform_names)), dtype=complex)
    fpeaks = np.zeros(len(waveform_names))

    for w, name in enumerate(waveform_names):

        # Create waveform instance
        waveform = pmns_utils.Waveform(name)
        waveform.reproject_waveform()
        waveform.compute_characteristics()
        fpeaks[w] = np.copy(waveform.fpeak)

        # Zero-pad
        rawdata = np.zeros(nTsamples)
        rawdata[:len(waveform.hplus)] = taper(waveform.hplus.data)

        # High-pass at 1 kHz
        #rawdata = signal.filtfilt(butter_b_high, butter_a_high, rawdata)

        del waveform

        # Unit-normalise
        rawdata /= np.linalg.norm(rawdata)

        original_signal = pycbc.types.TimeSeries(np.copy(rawdata),
                delta_t=delta_t) 
        original_signal_spectrum = original_signal.to_frequencyseries()

        full_cat[:,w] = np.copy(original_signal_spectrum.data)

        #
        # Extract low frequency data
        #

        # TD filter:
        low_cat[:,w] = pycbc.types.TimeSeries(
                signal.filtfilt(butter_b_low, butter_a_low, rawdata),
                delta_t=delta_t).to_frequencyseries() 

        #
        # Extract high frequency data
        #
        highdata = np.zeros(len(original_signal_spectrum), dtype=complex)

        # Alignment:
        false_freqs = original_signal_spectrum.sample_frequencies.data \
                - (fpeaks[w]-fcenter)
        startidx = np.argmin(abs(original_signal_spectrum.sample_frequencies -
                false_freqs[0]))
        highdata[startidx:] = original_signal_spectrum.data[:len(highdata)-startidx]

        # IFFT & TD Bandpass (for smoothness)
        highdata_fd = pycbc.types.FrequencySeries(highdata, 
                delta_f=original_signal_spectrum.delta_f)
        highdata_td = highdata_fd.to_timeseries()
        highdata_td.data = signal.filtfilt(butter_b_band, butter_a_band,
                highdata_td.data)
        high_cat[:,w] = highdata_td.to_frequencyseries()


    return (original_signal_spectrum.sample_frequencies, 
            low_cat, high_cat, full_cat)

def idealised_matches(catalogue, principle_components, delta_f=1.0, flow=1000,
        fhigh=8192):
    """
    Compute the matches between the waveforms in the catalogue and the
    ideal reconstructions, where we use the training data as the test and
    consider full, high and low catalogues seperately
    """
    
    nwaveforms=np.shape(catalogue)[1]

    # Amplitude & Phase-maximised match
    matches = np.zeros(shape=(np.shape(catalogue)[1], np.shape(catalogue)[1]))
    # NOTE: columns = waveforms, rows=Npcs

    # Loop over waveforms
    for column_idx in xrange(nwaveforms):

        # Loop over the number of pcs to use
        for u, use_npcs in enumerate(xrange(1,nwaveforms+1)):

            reconstruction = reconstruct_signal_ampphase(principle_components,
                    use_npcs, use_npcs, column_idx)

            matches[u, column_idx] = comp_match(reconstruction,
                    catalogue[:,column_idx],
                    delta_f=delta_f, flow=flow)

    return matches


def eigenergy(eigenvalues):
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



# *******************************************************************************88
def main():

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

    npcs = len(waveform_names)

    #
    # Build Catalogues
    #
    print "building catalogues"
    (freqaxis, low_cat, high_cat, full_cat) = build_catalogues(waveform_names)
    delta_f = np.diff(freqaxis)[0]

    # Convert to magnitude/phase
    full_mag, full_phase = complex_to_polar(full_cat)
    low_mag, low_phase = complex_to_polar(low_cat)
    high_mag, high_phase = complex_to_polar(high_cat)


    #
    # PCA
    #
    print "Performing PCA"
    high_pca = pca_magphase(high_cat)
    low_pca = pca_magphase(low_cat)
    full_pca = pca_magphase(full_cat)


    #
    # Compute idealised minimal matches
    #
    print "Computing all matches"
    full_matches_ideal = idealised_matches(full_cat, full_pca, delta_f=delta_f, flow=1000)
    low_matches_ideal = idealised_matches(low_cat, low_pca, delta_f=delta_f, flow=1000)
    high_matches_ideal = idealised_matches(high_cat, high_pca, delta_f=delta_f, flow=1000)










    # ******** #
    # Plotting #
    # ******** #


    #
    # Plot Catalogues
    #

    # Magnitude
    f, ax = pl.subplots(nrows=3)
    ax[0].plot(freqaxis, full_mag, label='full spectrum')
    ax[0].set_xlim(900, 5000)
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('|H(f)|')
    ax[0].minorticks_on()
    ax[0].set_title('Full Spectrum (magnitude)')

    ax[1].plot(freqaxis, low_mag, label='low-frequency components')
    ax[1].set_xlim(900, 2500)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('|H(f)|')
    ax[1].minorticks_on()
    ax[1].set_title('Low Frequency Components (magnitude)')

    ax[2].plot(freqaxis, high_mag, label='high-frequency components')
    ax[2].set_xlim(3500, 4500)
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('|H(f)|')
    ax[2].minorticks_on()
    ax[2].set_title('High Frequency Components, aligned to 4096 Hz (magnitude)')
    
    f.tight_layout()

    # Phase
    f, ax = pl.subplots(nrows=3)
    ax[0].plot(freqaxis, full_phase, label='full spectrum')
    ax[0].set_xlim(900, 5000)
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('arg[H(f)]')
    ax[0].minorticks_on()
    ax[0].set_title('Full Spectrum (phase)')

    ax[1].plot(freqaxis, low_phase, label='low-frequency components')
    ax[1].set_xlim(900, 2500)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('arg[H(f)]')
    ax[1].minorticks_on()
    ax[1].set_title('Low Frequency Components (phase)')

    ax[2].plot(freqaxis, high_phase, label='high-frequency components')
    ax[2].set_xlim(3500, 4500)
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('arg[H(f)]')
    ax[2].minorticks_on()
    ax[2].set_title('High Frequency Components, aligned to 4096 Hz (phase)')

    f.tight_layout()

    pl.show()

    #
    # Plot Magnitude PCs
    #
    f, ax = pl.subplots(nrows=3)
    ax[0].plot(freqaxis, abs(full_pca['magnitude_pcs']), label='full spectrum')
    ax[0].set_xlim(900, 5000)
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('|H(f)|')
    ax[0].minorticks_on()
    ax[0].set_title('Full Spectrum Principle Components (magnitude)')

    ax[1].plot(freqaxis, abs(low_pca['magnitude_pcs']), label='low-frequency components')
    ax[1].set_xlim(900, 2500)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('|H(f)|')
    ax[1].minorticks_on()
    ax[1].set_title('Low-frequency Principle Components (magnitude)')

    ax[2].plot(freqaxis, abs(high_pca['magnitude_pcs']), label='high-frequency components')
    ax[2].set_xlim(3500, 4500)
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('|H(f)|')
    ax[2].minorticks_on()
    ax[2].set_title('High-frequency Principle Components (magnitude)')
    f.tight_layout()
    pl.show()

    #
    # Plot Phase PCs
    #
    f, ax = pl.subplots(nrows=3)
    ax[0].plot(freqaxis, full_pca['phase_pcs'], label='full spectrum')
    ax[0].set_xlim(900, 5000)
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('Phase PCs')
    ax[0].set_ylabel('arg[H(f)]')
    ax[0].minorticks_on()
    ax[0].set_title('Full Spectrum Principle Components (phase)')

    ax[1].plot(freqaxis, low_pca['phase_pcs'], label='low-frequency components')
    ax[1].set_xlim(900, 2500)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('arg[H(f)]')
    ax[1].minorticks_on()
    ax[1].set_title('Low-frequency Principle Components (phase)')

    ax[2].plot(freqaxis, high_pca['phase_pcs'], label='high-frequency components')
    ax[2].set_xlim(3500, 4500)
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('arg[H(f)]')
    ax[2].minorticks_on()
    ax[2].set_title('High-frequency Principle Components (phase)')
    f.tight_layout()
    pl.show()

    #
    # PCA Diagnostics
    #

    # Eignenergies
    f, ax = pl.subplots(nrows=1)#, figsize=(8,6), sharex=True)
    npcs_axis = range(1,npcs+1)

    ax.plot(npcs_axis, full_pca['magnitude_eigenergy'], label='full spectrum')
    ax.plot(npcs_axis, low_pca['magnitude_eigenergy'],
         label='low frequencies')
    ax.plot(npcs_axis, high_pca['magnitude_eigenergy'],
         label='high frequencies')

    ax.plot(npcs_axis, full_pca['phase_eigenergy'], linestyle='--')
    ax.plot(npcs_axis, low_pca['phase_eigenergy'], linestyle='--')
    ax.plot(npcs_axis, high_pca['phase_eigenergy'], linestyle='--')

    ax.legend(loc='lower right')
    #ax.set_ylim(0,1)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Eigenenergy')
    ax.minorticks_on()

    f.tight_layout()
    pl.show()

    # -----

    # Full spectrum Matches: rows=npcs, cols=waveforms
    fig, ax = pl.subplots()

    #text portion
    xvals = range(1,len(waveform_names)+1)
    yvals = range(1,len(waveform_names)+1)

    im = ax.imshow(np.transpose(full_matches_ideal), interpolation='nearest', \
            extent=[min(xvals)-0.5,max(xvals)+0.5,min(yvals)-0.5,max(yvals)+0.5], origin='lower')

    for x,xval in enumerate(xvals):
        for y,yval in enumerate(yvals):
            if full_matches_ideal[x,y]<0.85:
                ax.text(xval, yval, '%.2f'%(full_matches_ideal[x,y]), \
                    va='center', ha='center', color='w')
            else:
                ax.text(xval, yval, '%.2f'%(full_matches_ideal[x,y]), \
                    va='center', ha='center')

    ax.set_xticks(xvals)
    ax.set_yticks(yvals)

    ylabels=[name.replace('_lessvisc','') for name in waveform_names]
    ax.set_yticklabels(ylabels)

    im.set_clim(0.75,1)
    im.set_cmap('bone')

    ax.set_xlabel('Number of PCs')

    ax.set_title("Matches For Full Spectrum PCA")

    fig.tight_layout()
    c=pl.colorbar(im, ticks=np.arange(0.75,1.05,0.05))
    pl.show()

    #
    # Low spectrum Matches: rows=npcs, cols=waveforms
    #

    fig, ax = pl.subplots()

    #text portion
    xvals = range(1,len(waveform_names)+1)
    yvals = range(1,len(waveform_names)+1)

    im = ax.imshow(np.transpose(low_matches_ideal), interpolation='nearest', \
            extent=[min(xvals)-0.5,max(xvals)+0.5,min(yvals)-0.5,max(yvals)+0.5], origin='lower')

    for x,xval in enumerate(xvals):
        for y,yval in enumerate(yvals):
            if low_matches_ideal[x,y]<0.85:
                ax.text(xval, yval, '%.2f'%(low_matches_ideal[x,y]), \
                    va='center', ha='center', color='w')
            else:
                ax.text(xval, yval, '%.2f'%(low_matches_ideal[x,y]), \
                    va='center', ha='center')

    ax.set_xticks(xvals)
    ax.set_yticks(yvals)

    ax.set_yticklabels(ylabels)

    im.set_clim(0.75,1)
    im.set_cmap('bone')

    ax.set_xlabel('Number of PCs')

    ax.set_title("Matches For Low Frequency PCA")

    fig.tight_layout()
    c=pl.colorbar(im, ticks=np.arange(0.75,1.05,0.05))
    pl.show()


    #
    # High spectrum Matches: rows=npcs, cols=waveforms
    #

    fig, ax = pl.subplots()

    #text portion
    xvals = range(1,len(waveform_names)+1)
    yvals = range(1,len(waveform_names)+1)

    im = ax.imshow(np.transpose(high_matches_ideal), interpolation='nearest', \
            extent=[min(xvals)-0.5,max(xvals)+0.5,min(yvals)-0.5,max(yvals)+0.5], origin='lower')

    for x,xval in enumerate(xvals):
        for y,yval in enumerate(yvals):
            if high_matches_ideal[x,y]<0.85:
                ax.text(xval, yval, '%.2f'%(high_matches_ideal[x,y]), \
                    va='center', ha='center', color='w')
            else:
                ax.text(xval, yval, '%.2f'%(high_matches_ideal[x,y]), \
                    va='center', ha='center')

    ax.set_xticks(xvals)
    ax.set_yticks(yvals)

    ax.set_yticklabels(ylabels)

    im.set_clim(0.75,1)
    im.set_cmap('bone')

    ax.set_xlabel('Number of PCs')

    ax.set_title("Matches For High Frequency PCA")

    fig.tight_layout()
    c=pl.colorbar(im, ticks=np.arange(0.75,1.05,0.05))

    pl.show()

    return 0


#
# End definitions
#
if __name__ == "__main__":
    main()






