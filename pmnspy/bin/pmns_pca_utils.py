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
from scipy import optimize

import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as pl

import lal
import lalsimulation as lalsim
import pmns_utils

import pycbc.types
import pycbc.filter
from pycbc.psd import aLIGOZeroDetHighPower

# _________________ FUNCTIONS  _________________ #

def pca_by_svd(matrix):
    """
    Perform principle component analysis via singular value decomposition
    """

    U, S, Vt = scipy.linalg.svd(matrix, full_matrices=True)

    V = Vt.T

    # sort the PCs by descending order of the singular values (i.e. by the
    # proportion of total variance they explain)
    #ind = np.argsort(S)[::-1]
    ind = np.argsort(S)[::-1]
    U = U[:,ind]
    S = S[ind]
    V = V[:,ind]

    # See e.g.,:
    # http://en.wikipedia.org/wiki/Principal_component_analysis#Singular_value_decomposition

    # Score matrix:
    PC_scores = U * S

    return PC_scores, U, V, S**2 

def pca_magphase(complex_catalogue):
    """
    Do PCA with magnitude and phase parts of the complex waveforms in complex_catalogue
    """

    magnitudes, phases = complex_to_polar(complex_catalogue)

    for w in xrange(np.shape(complex_catalogue)[1]):
        phases[:,w] = signal.detrend(phases[:,w])

    mean_mag   = np.zeros(np.shape(complex_catalogue)[0])
    mean_phase = np.zeros(np.shape(complex_catalogue)[0])
    for s in xrange(np.shape(complex_catalogue)[0]):
        mean_mag[s]   = np.mean(magnitudes[s,:])
        mean_phase[s] = np.mean(phases[s,:])


    for w in xrange(np.shape(complex_catalogue)[1]):
        #phases[:,w] = signal.detrend(phases[:,w])
        magnitudes[:,w] -= mean_mag
        phases[:,w] -= mean_phase

    pcs_magphase = {}

    pcs_magphase['magnitude_scores'], pcs_magphase['magnitude_pcs'], \
            pcs_magphase['magnitude_betas'], pcs_magphase['magnitude_eigs'] = \
            pca_by_svd(magnitudes)

    pcs_magphase['phase_scores'], pcs_magphase['phase_pcs'], \
            pcs_magphase['phase_betas'], pcs_magphase['phase_eigs'] = \
            pca_by_svd(phases)

    pcs_magphase['magnitude_eigenergy'] = \
            eigenergy(pcs_magphase['magnitude_eigs'])
    pcs_magphase['phase_eigenergy'] = \
            eigenergy(pcs_magphase['phase_eigs'])

    pcs_magphase['mean_mag'] = mean_mag
    pcs_magphase['mean_phase'] = mean_phase

    return pcs_magphase



def complex_to_polar(catalogue):
    """
    Convert the complex Fourier spectrum to an amplitude and phase
    """

    magnitudes = np.zeros(shape=np.shape(catalogue))
    phases = np.zeros(shape=np.shape(catalogue))
    if len(np.shape(catalogue))==1:
        return abs(catalogue), np.unwrap(np.angle(catalogue))

    for c in xrange(np.shape(catalogue)[1]):
        magnitudes[:,c] = abs(catalogue[:,c])
        phases[:,c] = np.unwrap(np.angle(catalogue[:,c]))

    return magnitudes, phases



def taper(input_data, delta_t=1./16384):
    """  
    Apply a taper to the start/end of the data in input_data
    """

    timeseries = lal.CreateREAL8TimeSeries('blah', 0.0, 0,
            delta_t, lal.StrainUnit, int(len(input_data)))

    lalsim.SimInspiralREAL8WaveTaper(timeseries.data,
        lalsim.SIM_INSPIRAL_TAPER_START)
        #lalsim.SIM_INSPIRAL_TAPER_STARTEND)

    return timeseries.data.data

def build_catalogues(waveform_names, fshift_center):
    """
    Build the low, high and full catalogues, including data conditioning for the
    waveforms in the list of names waveform_names
    """

    sample_freq=16384
    delta_t=1./sample_freq
    nTsamples=1*sample_freq
    nFsamples=nTsamples/2 + 1
    times=np.arange(0, delta_t*nTsamples, delta_t)

    # Preallocation
    aligned_cat = np.zeros(shape=(nFsamples, len(waveform_names)), dtype=complex)
    original_cat = np.zeros(shape=(nFsamples, len(waveform_names)), dtype=complex)
    fpeaks = np.zeros(len(waveform_names))

    for w, name in enumerate(waveform_names):

        #
        # Create waveform instance: pmns_utils
        #
        waveform = pmns_utils.Waveform(name)
        waveform.reproject_waveform()

        # Waveform conditioning
        original_spectrum, fpeaks[w] = condition_spectrum(waveform.hplus.data)

        del waveform

        # Normalise to unit hrss
        original_spectrum = unit_hrss(original_spectrum,
                delta=original_spectrum.delta_f, domain='frequency')

        # Add to catalogue
        original_frequencies = np.copy(original_spectrum.sample_frequencies)
        original_cat[:,w] = np.copy(original_spectrum.data)

        # Feature alignment
        aligned_spectrum = shift_vec(original_spectrum.data, original_frequencies,
                fpeaks[w], fshift_center)

        # Populate catalogue with reconstructed, normalised aligned spectrum
        aligned_cat[:,w] = unit_hrss(aligned_spectrum,
                delta=original_spectrum.delta_f, domain='frequency') 

    return (original_frequencies, aligned_cat, original_cat, fpeaks)

def condition_spectrum(waveform_timeseries, delta_t=1./16384, nsamples=16384):
    """
    Zero-pad, window and FFT a time-series to return a frequency series of
    standard length (16384 samples)
    """

    # Time-domain Window
    win=lal.CreateTukeyREAL8Window(len(waveform_timeseries),0.25)
    waveform_timeseries *= win.data.data

    # Zero-pad
    paddata = np.zeros(nsamples)
    paddata[:len(waveform_timeseries)] = np.copy(waveform_timeseries)

    # FFT
    timeseries = pycbc.types.TimeSeries(initial_array=paddata, delta_t=delta_t)
    freqseries = timeseries.to_frequencyseries()

    # Locate fpeak
    high_idx = freqseries.sample_frequencies.data>=2000 
    high_freq = freqseries.sample_frequencies.data[high_idx] 
    fpeak = high_freq[np.argmax(abs(freqseries[high_idx]))]
    
    return freqseries, fpeak

def unit_hrss(data, delta, domain):
    """
    Normalise the data to have unit hrss.  delta is the unit spacing (delta_t or
    delta_f) and domain is one of 'frequency' or 'time' to determine the pycbc
    type
    """

    allowed_domains=['time', 'frequency']
    if domain not in ['time', 'frequency']:
        print >> sys.stderr, "ERROR: domain must be in ", allowed_domains
        sys.exit()

    if domain=='time':
        timeseries = pycbc.types.TimeSeries(initial_array=data, delta_t=delta)
        sigma = pycbc.filter.sigma(timeseries)
        timeseries.data/=sigma
        return timeseries

    elif domain=='frequency':
        freqseries = pycbc.types.FrequencySeries(initial_array=data, delta_f=delta)
        sigma = pycbc.filter.sigma(freqseries)
        freqseries.data/=sigma
        return freqseries


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
        weighted=True):
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


def shift_vec(vector, target_freqs, fpeak, fcenter=1000.0):

    # Frequency shift
    fshift = fcenter / fpeak
    false_freqs = target_freqs * fshift

    aligned_spec_real = np.interp(target_freqs, false_freqs, np.real(vector))
    aligned_spec_imag = np.interp(target_freqs, false_freqs, np.imag(vector))

    aligned_vector = aligned_spec_real + 1j*aligned_spec_imag

    return aligned_vector


def unshift_vec(vector, target_freqs, fpeak, fcenter=1000.0):

    # Frequency shift
    fshift = fpeak / fcenter
    false_freqs = target_freqs * fshift

    unaligned_spec_real = np.interp(target_freqs, false_freqs, np.real(vector))
    unaligned_spec_imag = np.interp(target_freqs, false_freqs, np.imag(vector))

    unaligned_vector = unaligned_spec_real + 1j*unaligned_spec_imag

    return unaligned_vector


# _________________ CLASSES  _________________ #

class pmnsPCA:
    """
    An object with a catalogue and principal component decomposition of
    post-merger waveforms

    """

    def __init__(self, waveform_list, fcenter=1000):

        #
        # Build Catalogues
        #
        self.fcenter=fcenter
        self.waveform_list=waveform_list

        print "Building catalogue"
        (self.sample_frequencies, self.cat_align, self.cat_orig,
                self.fpeaks)=build_catalogues(self.waveform_list, self.fcenter)
        self.delta_f = np.diff(self.sample_frequencies)[0]

        #
        # PCA
        #
        print "Performing PCA"

        # Convert to magnitude/phase
        # XXX: need this?  happens in pca anyway
        self.magnitudes_align, self.phases_align = \
                complex_to_polar(self.cat_align)

        # Do PCA
        self.pca = pca_magphase(self.cat_align)

    def project(self, freqseries, this_fpeak=None):
        """
        Project the frequency series freqseries onto the principal components to
        represent that waveform in the new basis.  The projection yields the
        coefficients {beta} such that the freqseries can be reconstructed as a
        linear combination of the beta-weighted PCs

        Procedure:
        1) Align test spectrum (peak) to 1kHz
        2) Normalise test spectrum to unit hrss
        3) Convert to polar representation
        4) Center the test spectrum
        5) Take projection

        """

        if this_fpeak==None:
            # Locate fpeak
            # Note: we'll assume the peak we're aligning to is >2kHz.  This
            # avoids any low frequency stuff.
            high_idx = self.sample_frequencies>=2000 
            high_freq = self.sample_frequencies[high_idx] 
            this_fpeak = high_freq[np.argmax(abs(freqseries[high_idx]))]

        # Dictionary to hold input and results of projection
        projection = dict()
        projection['freqseries'] = np.copy(freqseries)

        # Align test spectrum
        freqseries_align = shift_vec(freqseries, self.sample_frequencies,
                this_fpeak)
        
        # Normalise test spectrum
        freqseries_align = unit_hrss(freqseries_align, delta=self.delta_f,
                domain='frequency')

        projection['freqseries_align'] = np.copy(freqseries_align)

        # Complex to polar
        magnitude_align, phase_align = complex_to_polar(freqseries_align)

        # Center test spectrum
        magnitude_cent = magnitude_align - self.pca['mean_mag']
        phase_cent = phase_align - self.pca['mean_phase']

        #self.projection['freqseries_align'] = np.copy(freqseries_align)


        # Finally, project test spectrum onto PCs (dot product between data and PCs)
        projection['betas_magnitude'] = np.dot(magnitude_cent,
                self.pca['magnitude_pcs'])
        projection['betas_phase'] = np.dot(phase_cent,
                self.pca['phase_pcs'])

        return projection

    def reconstruct(self, freqseries, npcs=1, this_fpeak=None):
        """
        Reconstruct the waveform in freqseries using <npcs> principal components
        from the catalogue

        Procedure:
        1) Reconstruct the centered spectra (phase and mag) from the
        beta-weighted PCs
        2) Un-center the spectra (add the mean back on)
        """

        # Get projection:
        projection = self.project(freqseries)

        # Initialise reconstructions
        recon_magnitude_align = np.copy(self.pca['mean_mag'])
        recon_phase_align = np.copy(self.pca['mean_phase'])

        # Reconstruct
        for i in xrange(npcs):
            print i
            recon_magnitude_align += projection['betas_magnitude'][i]*\
                    self.pca['magnitude_pcs'][:,i]
            recon_phase_align += projection['betas_phase'][i]*\
                    self.pca['phase_pcs'][:,i]


        f, ax = pl.subplots()
        ax.plot(self.sample_frequencies, abs(projection['freqseries_align']),
                color='r', linewidth=2)
        ax.plot(self.sample_frequencies, recon_magnitude_align, color='k')

        ax.set_xlim(0,1500)

        #print projection['betas_magnitude']

        pl.show()




# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXX: LIKELY JUNK

def reconstruct_signal_ampphase(pcs_magphase, nMagPCs, nPhasePCs, waveform_num):

    """
    Build the reconstructed signal from magnitude and phase components
    """

    magScores = pcs_magphase['magnitude_scores']
    magBetas = pcs_magphase['magnitude_betas']
    phasePCs = pcs_magphase['phase_scores']
    phaseBetas = pcs_magphase['phase_betas']

    mag = np.zeros(np.shape(magScores)[0], dtype=complex)
    phase = np.zeros(np.shape(phasePCs)[0], dtype=complex)

    for n in xrange(nMagPCs):
        mag += magBetas[waveform_num,n] * magScores[:,n]
    for n in xrange(nPhasePCs):
        phase += phaseBetas[waveform_num,n] * phasePCs[:,n]

    mag += pcs_magphase['mean_mag']
    phase += pcs_magphase['mean_phase']
    #mag *= pcs_magphase['std_mag']
    #phase *= pcs_magphase['std_phase']

    reconstruction = mag*np.exp(1j*phase)

    return reconstruction

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
    for w in xrange(nwaveforms):

        # Loop over the number of pcs to use
        for n in xrange(nwaveforms):

            reconstruction = reconstruct_signal_ampphase(principle_components,
                    n+1, n+1, w)

            matches[w, n] = comp_match(reconstruction, catalogue[:,w],
                    delta_f=delta_f, flow=flow)

    return matches

def unaligned_rec_cat(pcs, npcs, fpeaks, freqaxis, fcenter):

    """
    Reconstruct the catalogue using npcs and the aligned waveforms
    """
    
    rec_cat = np.zeros(shape=(np.shape(pcs['magnitude_scores'])), dtype=complex)

    for w in xrange(len(fpeaks)):

        rec_cat[:, w] = unshift_waveform(pcs, [npcs[0], npcs[1]], fpeaks[w],
                freqaxis, waveform_num=w, fcenter=fcenter)

    return rec_cat

def unshift_waveform(aligned_pcs, npcs, fpeak, target_freqs, waveform_num=0,
        fcenter=3000., delta_f=1.0):
    """
    Reconstruct the aligned waveform and shift it back to the original peak
    frequency.  npcs is a tuple with the number of [mag, phase] PCs to use
    """

    reconstruction = reconstruct_signal_ampphase(aligned_pcs, npcs[0], npcs[1], waveform_num)


    #fshift = fcenter / fpeak

    fshift = fpeak / fcenter
    false_freqs = target_freqs * fshift

    aligned_spec_real = np.interp(target_freqs, false_freqs,
            np.real(reconstruction))

    aligned_spec_imag = np.interp(target_freqs, false_freqs,
            np.imag(reconstruction))

    aligned_reconstruction = aligned_spec_real + 1j*aligned_spec_imag

    return aligned_reconstruction


def unaligned_matches(catalogue, pcs, fpeaks, freqaxis, delta_f=1.0, flow=1000,
        fhigh=8192, fcenter=3000):
    """
    Compute the matches between the waveforms in the catalogue and the
    aligned spectrum reconstructions, where we use the training data as the test and
    consider full, high and low catalogues seperately
    """
    
    nwaveforms=np.shape(catalogue)[1]

    # Amplitude & Phase-maximised match
    matches = np.zeros(shape=(np.shape(catalogue)[1], np.shape(catalogue)[1]))
    # NOTE: columns = waveforms, rows=Npcs

    # Loop over the number of pcs to use
    for n in xrange(nwaveforms):

        rec_cat = unaligned_rec_cat(pcs, [n+1, n+1], fpeaks, freqaxis,
                fcenter=fcenter)

        # Loop over waveforms
        for w in xrange(nwaveforms):

            matches[w,n] = comp_match(rec_cat[:,w], catalogue[:,w],
                    delta_f=delta_f, flow=flow)

    return matches




def unaligned_template(magBetas, phaseBetas, pcs, fpeak, target_freqs,
        fcenter=3000., delta_f=1.0):
    """
    Build an template with arbitrary PC amplitudes
    """

    # Basic reconstruction
    magScores   = pcs['magnitude_scores']
    phasePCs = pcs['phase_scores']

    mag = np.zeros(np.shape(magScores)[0], dtype=complex)
    phase = np.zeros(np.shape(phasePCs)[0], dtype=complex)

    for n in xrange(len(magBetas)):
        mag += magBetas[n] * magScores[:,n]
    for n in xrange(len(phaseBetas)):
        phase += phaseBetas[n] * phasePCs[:,n]

    reconstruction = mag*np.exp(1j*phase)

    # Frequency shift
    fshift = fpeak / fcenter
    false_freqs = target_freqs * fshift

    aligned_spec_real = np.interp(target_freqs, false_freqs,
            np.real(reconstruction))

    aligned_spec_imag = np.interp(target_freqs, false_freqs,
            np.imag(reconstruction))

    aligned_reconstruction = aligned_spec_real + 1j*aligned_spec_imag

    return aligned_reconstruction


def match(vary_args, *fixed_args):

    # variable params
    magbetas = vary_args[0:len(vary_args)/2]
    phasebetas = vary_args[len(vary_args)/2:-1]

    #if sum(abs(magbetas)>10) or sum(abs(phasebetas)>10): return -np.inf
    #else:

    # fixed params
    test_waveform, aligned_pcs, fpeak, target_freqs, fcenter, delta_f = fixed_args

    # Generate template
    tmplt = unaligned_template(magbetas, phasebetas, aligned_pcs, fpeak,
            target_freqs, fcenter, delta_f=1.0)

    try:
        match = comp_match(tmplt, test_waveform, delta_f=delta_f, flow=1000)

    except ZeroDivisionError:
        match = -np.inf

    return np.log(match)



def image_matches(match_matrix, waveform_names, title=None):

    fig, ax = pl.subplots(figsize=(15,8))
    #fig, ax = pl.subplots(figsize=(7,5))
    npcs = len(waveform_names)

    im = ax.imshow(match_matrix, interpolation='nearest', origin='lower')

    for x in xrange(npcs):
        for y in xrange(npcs):
            if match_matrix[x,y]<0.85:
                ax.text(y, x, '%.2f'%(match_matrix[x,y]), \
                    va='center', ha='center')
            else:
                ax.text(y, x, '%.2f'%(match_matrix[x,y]), \
                    va='center', ha='center', color='w')

    ax.set_xticks(range(0,npcs))
    ax.set_yticks(range(0,npcs))

    xlabels=range(1,npcs+1)
    ax.set_xticklabels(xlabels)

    ylabels=[name.replace('_lessvisc','') for name in waveform_names]
    ax.set_yticklabels(ylabels)

    im.set_clim(0.75,1)
    im.set_cmap('gnuplot2_r')

    ax.set_xlabel('Number of PCs')

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    c=pl.colorbar(im, ticks=np.arange(0.75,1.05,0.05))#, orientation='horizontal')

    return fig, ax

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# *******************************************************************************
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
 
#   waveform_names=['apr_135135_lessvisc',
#                   'shen_135135_lessvisc',
#                   'shen_1215',
#                   'dd2_135135_lessvisc' ,
#                   'dd2_165165_lessvisc' ,
#                   'nl3_1919_lessvisc' ,
#                   'nl3_135135_lessvisc' ,
#                   'tm1_135135_lessvisc' ,
#                   'tma_135135_lessvisc' ,
#                   'sfhx_135135_lessvisc',
#                   'sfho_135135_lessvisc',
#                   'tm1_1215',
#                   'gs1_135135',
#                   'gs2_135135',
#                   'sly4_135135',
#                   'ls220_135135',
#                   'ls375_135135']
#


    #
    # Create PMNS PCA instance for this catalogue
    #
    pmpca = pmnsPCA(waveform_names)


    #
    # Create test waveform
    #
    noncat_name = 'shen_135135_lessvisc'
    noncat_waveform = pmns_utils.Waveform(noncat_name)
    noncat_waveform.reproject_waveform()

    # Standardise
    noncat_waveform_FD, fpeak = condition_spectrum(noncat_waveform.hplus.data)

    # Normalise
    noncat_waveform_FD = unit_hrss(noncat_waveform_FD.data,
            delta=noncat_waveform_FD.delta_f, domain='frequency')

    # Reconstructions 
    pmpca.reconstruct(noncat_waveform_FD.data, npcs=5)


    sys.exit()











    npcs = len(waveform_names)

    #
    # Build Catalogues
    #
    print "building catalogues"
    fshift_center = 1000
    (freqaxis, shift_cat, original_cat, fpeaks) = build_catalogues(waveform_names, fshift_center)
    delta_f = np.diff(freqaxis)[0]

    # Convert to magnitude/phase
    full_mag, full_phase = complex_to_polar(original_cat)
    shift_mag, shift_phase = complex_to_polar(shift_cat)


    #
    # PCA
    #
    print "Performing PCA"
    shift_pca = pca_magphase(shift_cat, freqaxis, flow=10)
    full_pca = pca_magphase(original_cat, freqaxis, flow=10)



    #
    # Compute idealised minimal matches
    #
    print "Computing all matches"
    full_matches_ideal = idealised_matches(original_cat, full_pca, delta_f=delta_f, flow=1000)
    shift_matches_ideal = idealised_matches(shift_cat, shift_pca,
            delta_f=delta_f, flow=10) # careful with this one's flow!

    unshift_matches_ideal = unaligned_matches(original_cat, shift_pca, fpeaks,
            freqaxis, fcenter=fshift_center)


    # ******** #
    # Plotting #
    # ******** #
    imageformats=['png','eps','pdf']


    #
    # Plot Catalogues
    #

    # Magnitude
    f, ax = pl.subplots(nrows=2,figsize=(7,15))
    ax[0].plot(freqaxis, full_mag, label='full spectrum')
    ax[0].set_xlim(1000, 5000)
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('|H(f)|')
    ax[0].minorticks_on()
    ax[0].set_title('Full Spectrum (magnitude)')

    ax[1].plot(freqaxis, shift_mag, label='aligned spectrum')
    ax[1].set_xlim(0, fshift_center+1000)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('|H(f)|')
    ax[1].minorticks_on()
    ax[1].set_title('aligned Spectrum (magnitude)')

    
    f.tight_layout()
    for fileformat in imageformats:
        f.savefig('catalogue_magnitude_overlay.%s'%fileformat)

    # Phase
    f, ax = pl.subplots(nrows=2,figsize=(7,15))
    a=0
    ax[a].plot(freqaxis, full_phase, label='full spectrum')
    ax[a].set_xlim(1000, 5000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('Full Spectrum (phase)')

    a+=1
    ax[a].plot(freqaxis, shift_phase, label='aligned spectrum')
    ax[a].set_xlim(0, fshift_center+1000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('aligned Spectrum (phase)')

    f.tight_layout()
    for fileformat in imageformats:
        f.savefig('catalogue_phase_overlay.%s'%fileformat)

    #pl.show()

    #
    # Plot Magnitude PCs
    #
    f, ax = pl.subplots(nrows=2,figsize=(7,15))
    a=0
    ax[a].plot(freqaxis, abs(full_pca['magnitude_scores']), label='full spectrum')
    ax[a].set_xlim(1000, 5000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('|H(f)|')
    ax[a].minorticks_on()
    ax[a].set_title('Full Spectrum Principle Components (magnitude)')

    a+=1
    ax[a].plot(freqaxis, abs(shift_pca['magnitude_scores']), label='aligned spectrum')
    ax[a].set_xlim(0, fshift_center+1000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('|H(f)|')
    ax[a].minorticks_on()
    ax[a].set_title('aligned Spectrum Principle Components (magnitude)')

    f.tight_layout()
    for fileformat in imageformats:
        f.savefig('pcs_magnitude_overlay.%s'%fileformat)
    #pl.show()

    #
    # Plot Phase PCs
    #
    f, ax = pl.subplots(nrows=2,figsize=(7,15))
    a=0
    ax[a].plot(freqaxis, full_pca['phase_scores'], label='full spectrum')
    ax[a].set_xlim(1000, 5000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('Phase PCs')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('Full Spectrum Principle Components (phase)')

    a+=1
    ax[a].plot(freqaxis, shift_pca['phase_scores'], label='aligned spectrum')
    ax[a].set_xlim(0, fshift_center+1000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('Phase PCs')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('aligned Spectrum Principle Components (phase)')

    f.tight_layout()
    for fileformat in imageformats:
        f.savefig('pcs_phase_overlay.%s'%fileformat)
    #pl.show()

    #
    # PCA Diagnostics
    #

    # Eignenergies
    f, ax = pl.subplots(nrows=1)#, figsize=(8,6), sharex=True)
    npcs_axis = range(1,npcs+1)

    ax.plot(npcs_axis, full_pca['magnitude_eigenergy'], label='full spectrum',
            color='b')
    ax.plot(npcs_axis, shift_pca['magnitude_eigenergy'], 
            label='aligned spectrum', color='g')

    ax.plot(npcs_axis, full_pca['phase_eigenergy'], linestyle='--', color='b')
    ax.plot(npcs_axis, shift_pca['phase_eigenergy'], linestyle='--', color='g')

    ax.legend(loc='lower right')
    #ax.set_ylim(0,1)
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Explained Variance (\%)')
    ax.minorticks_on()

    f.tight_layout()
    for fileformat in imageformats:
        f.savefig('eigenenergy.%s'%fileformat)
    #pl.show()


    # -----------
    # Match Plots

    #
    # aligned waveforms
    fig, ax = image_matches(shift_matches_ideal, waveform_names,
            title="Matches For Full Aligned Spectrum PCA")

    for fileformat in imageformats:
        fig.savefig('shiftspec_ideal_matches.%s'%fileformat)


    # UN-aligned waveforms
    fig, ax = image_matches(unshift_matches_ideal, waveform_names,
            title="Matches For aligned Full Spectrum PCA")

    for fileformat in imageformats:
        fig.savefig('unshiftspec_ideal_matches.%s'%fileformat)


    # Original waveforms
    fig, ax = image_matches(full_matches_ideal, waveform_names,
            title="Matches For Full Spectrum PCA")

    for fileformat in imageformats:
        fig.savefig('fullspec_ideal_matches.%s'%fileformat)



#
# End definitions
#
if __name__ == "__main__":
    main()






