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

from sklearn.decomposition import PCA 

import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as pl

import lal
import lalsimulation as lalsim
import pmns_utils

import pycbc.types
import pycbc.filter
from pycbc.psd import aLIGOZeroDetHighPower

from IPython.core.debugger import Tracer; debug_here = Tracer()

# _________________ FUNCTIONS  _________________ #


def perform_pca(cpx, magnitudes, phases):
    """
    Do PCA with magnitude and phase parts of the complex waveforms in complex_catalogue
    """

    magnitude_spectra_centered, magnitude_mean, magnitude_std = \
            condition_magnitude(magnitudes)

    phase_spectra_centered, phase_mean, phase_std, phase_trend, pfits = \
            condition_phase(phases)


    mag_pca = PCA(whiten=False)
    mag_pca.fit(magnitude_spectra_centered)

    phase_pca = PCA(whiten=False)
    phase_pca.fit(phase_spectra_centered)

    pcs_magphase={}

    pcs_magphase['magnitude_pca'] = mag_pca
    pcs_magphase['phase_pca'] = phase_pca


    pcs_magphase['magnitude_mean'] = magnitude_mean
    pcs_magphase['phase_mean'] = phase_mean

    pcs_magphase['magnitude_std'] = magnitude_std
    pcs_magphase['phase_std'] = phase_std

    pcs_magphase['phase_trend'] = phase_trend
    pcs_magphase['phase_fits'] = pfits


    # XXX

    cpx_spectra_centered, cpx_mean, cpx_std = condition_cpx(cpx.real)

    cpx_pca = PCA(whiten=False)
    cpx_pca.fit(cpx_spectra_centered)

    pcs_magphase['cpx_pca'] = cpx_pca

    pcs_magphase['cpx_mean'] = cpx_mean
    pcs_magphase['cpx_std'] = cpx_std


    return pcs_magphase

def condition_cpx(cpx_spectra):
    """
    Center and scale the cpx spectrum with optional trend removal
    """

    cpx_spectra_centered = np.copy(cpx_spectra)

    # --- Centering
    cpx_mean = np.mean(cpx_spectra_centered, axis=0)
    for w in xrange(np.shape(cpx_spectra)[0]):
        cpx_spectra_centered[w,:] -= cpx_mean

    # --- Scaling
    cpx_std = np.std(cpx_spectra_centered, axis=0)

    for w in xrange(np.shape(cpx_spectra)[0]):
        #cpx_spectra_centered[w,:] /= cpx_std
        cpx_spectra_centered[w,:] /= np.linalg.norm(cpx_spectra_centered[w,:])

    return cpx_spectra_centered, cpx_mean, cpx_std

def condition_magnitude(magnitude_spectra):

    magnitude_spectra_centered = np.copy(magnitude_spectra)

#   for w in xrange(np.shape(magnitude_spectra)[0]):
#       magnitude_spectra_centered[w,:] /= \
#               np.linalg.norm(magnitude_spectra_centered[w,:])

    # --- Centering
    magnitude_mean = np.mean(magnitude_spectra, axis=0)
    for w in xrange(np.shape(magnitude_spectra)[0]):
        magnitude_spectra_centered[w,:] -= magnitude_mean

    # --- Scaling
    magnitude_std = np.std(magnitude_spectra, axis=0)

    #for w in xrange(np.shape(magnitude_spectra)[0]):
        #magnitude_spectra_centered[w,:] /= magnitude_std

    return magnitude_spectra_centered, magnitude_mean, magnitude_std


def condition_phase(phase_spectra):
    """
    Center and scale the phase spectrum with optional trend removal
    """

    phase_spectra_centered = np.copy(phase_spectra)

#   for w in xrange(np.shape(phase_spectra)[0]):
#       phase_spectra_centered[w,:] /= \
#               np.linalg.norm(phase_spectra_centered[w,:])
#
    # --- Trend removal
    x = np.arange(len(phase_spectra[0,:]))
    pfits = np.zeros(shape=np.shape(phase_spectra))
    for w in xrange(np.shape(pfits)[0]):
        y = phase_spectra[w,:]
        pfits[w,:] = poly4(x, y)

    phase_trend = np.mean(pfits,axis=0)
    #for p in xrange(np.shape(pfits)[1]):
    #    phase_spectra_centered[:,w] -= phase_trend

    # --- Centering
    phase_mean = np.mean(phase_spectra_centered, axis=0)
    for w in xrange(np.shape(phase_spectra)[0]):
        phase_spectra_centered[w,:] -= phase_mean

    # --- Scaling
    #phase_std = np.ones(shape=np.shape(phase_mean))
    #phase_std = np.std(phase_spectra, axis=1)
    phase_std = np.std(phase_spectra_centered, axis=0)
    phase_std[0] = 1.0
    for w in xrange(np.shape(phase_spectra)[0]):
        phase_spectra_centered[w,1:] /= phase_std[1:]

    return phase_spectra_centered, phase_mean, phase_std, phase_trend, pfits


def complex_to_polar(catalogue):
    """
    Convert the complex Fourier spectrum to an amplitude and phase
    """

    magnitudes = np.zeros(shape=np.shape(catalogue))
    phases = np.zeros(shape=np.shape(catalogue))
    if len(np.shape(catalogue))==1:
        return abs(catalogue), phase_of(catalogue) - phase_of(catalogue)[0]

    for c in xrange(np.shape(catalogue)[0]):
        magnitudes[c,:] = abs(catalogue[c,:])
        phases[c,:] = phase_of(catalogue[c,:]) - phase_of(catalogue[c,:])[0]

    return magnitudes, phases

def phase_of(z):
    return np.unwrap(np.angle(z)) #- np.pi / 2
#   import mpmath
#   phases = np.zeros(len(z))
#   for p, phase in enumerate(phases):
#       phases[p] = np.float(mpmath.arg(z[p]))
#   return np.unwrap(phases)# - 2*np.pi

def build_catalogues(waveform_names, fshift_center):
    """
    Build the data matrix
    """

    sample_freq=16384
    delta_t=1./sample_freq
    nTsamples=1*sample_freq
    nFsamples=nTsamples/2 + 1
    times=np.arange(0, delta_t*nTsamples, delta_t)

    # Preallocation
    aligned_cat = np.zeros(shape=(len(waveform_names), nFsamples),  dtype=complex)
    original_cat = np.zeros(shape=(len(waveform_names), nFsamples), dtype=complex)
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
        original_cat[w,:] = np.copy(original_spectrum.data)

        # Feature alignment
        aligned_spectrum = shift_vec(original_spectrum.data, original_frequencies,
                fpeaks[w], fshift_center)

        # Populate catalogue with reconstructed, normalised aligned spectrum
        aligned_cat[w,:] = unit_hrss(aligned_spectrum,
                delta=original_spectrum.delta_f, domain='frequency').data

    return (original_frequencies, aligned_cat, original_cat, fpeaks)

def condition_spectrum(waveform_timeseries, delta_t=1./16384, nsamples=16384):
    """
    Zero-pad, window and FFT a time-series to return a frequency series of
    standard length (16384 samples)
    """

    # Time-domain Window
    win=lal.CreateTukeyREAL8Window(len(waveform_timeseries),0.1)
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

def poly4(x, y):
    """
    helper func for 4th order polyfit for phase conditioning
    """

    # 4th order polyfit to phase for trend removal
    p = np.polyfit(x, y, 4)

    return  p[0]*x**4 + p[1]*x**3 + p[2]*x**2 + p[3]*x**1 + p[4]

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


def shift_vec(vector, target_freqs, fpeak, fcenter=1000.0):

    # Frequency shift
    fshift = fcenter / fpeak
    false_freqs = target_freqs * fshift

    aligned_vector = complex_interp(vector, target_freqs, false_freqs)

    return aligned_vector

def complex_interp(ordinate, abscissa_interp, abscissa):
    """
    Interpolate complex series
    """

    ordinate_interp_real = np.interp(abscissa_interp, abscissa,
            np.real(ordinate))
    ordinate_interp_imag = np.interp(abscissa_interp, abscissa,
            np.imag(ordinate))

    return ordinate_interp_real + 1j*ordinate_interp_imag

def fine_phase_spectrum(complex_spectrum, sample_frequencies, gamma=0.01):
    """
    Return the unwrapped phase spectrum of the complex Fourier spectrum in
    complex_spectrum.

    This function interpolates the complex spectrum to a frequency axis, which is
    gamma times denser than the original frequencies, prior to unwrapping.  This
    permits a much smoother unwrapping.

    Finally, the interpolated spectrum is re-interpolated back to the original
    frequencies
    """

    delta_f = np.diff(sample_frequencies)[0]
    fine_frequencies = np.arange(sample_frequencies.min(),
            sample_frequencies.max() + delta_f, delta_f * gamma)

    complex_spectrum_interp = complex_interp(complex_spectrum, fine_frequencies,
            sample_frequencies)

    phase_spectrum = phase_of(complex_spectrum)
    phase_spectrum_fine = phase_of(complex_spectrum_interp)

    return np.interp(sample_frequencies, fine_frequencies, phase_spectrum_fine)


def residual_power(vector1, vector2):
    """
    Compute the normalised squared-residuals between vectors 1 and 2:

    result = (vector1-vector2)**2 / vector2**2

    """

    #return np.sqrt(sum((vector1-vector2)**2) / sum(vector2**2))
    #return np.linalg.norm(vector1-vector2) / np.linalg.norm(vector2)
    return np.sqrt(sum((vector1-vector2)**2))# / sum(vector2**2))


def wrap_phase(phase):
    """
    Opposite of np.wrap()
    """
    return ( phase + np.pi) % (2 * np.pi ) - np.pi

# _________________ CLASSES  _________________ #

class pmnsPCA:
    """
    An object with a catalogue and principal component decomposition of
    post-merger waveforms

    """

    def __init__(self, waveform_list, fcenter=1000, low_frequency_cutoff=0):

        #
        # Build Catalogues
        #
        self.fcenter=fcenter
        self.waveform_list=waveform_list

        print "Building catalogue"
        (self.sample_frequencies, self.cat_align, self.cat_orig,
                self.fpeaks)=build_catalogues(self.waveform_list, self.fcenter)
        self.delta_f = np.diff(self.sample_frequencies)[0]

        # min freq for match calculations
        self.low_frequency_cutoff=low_frequency_cutoff

        #
        # PCA
        #

        print "Aligning"

        # Convert to magnitude/phase
        self.magnitude, self.phase = complex_to_polar(self.cat_orig)

        self.magnitude_align = np.zeros(np.shape(self.magnitude))
        for i in xrange(np.shape(self.magnitude)[0]):
            self.magnitude_align[i,:] = shift_vec(self.magnitude[i,:],
                    self.sample_frequencies, self.fpeaks[i], self.fcenter).real

        # Do PCA
        print "Performing PCA"
        self.pca = perform_pca(self.cat_align, self.magnitude_align, self.phase)



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
                fcenter=self.fcenter, fpeak=this_fpeak)
        
        # Normalise test spectrum
        freqseries_align = unit_hrss(freqseries_align, delta=self.delta_f,
                domain='frequency')

        # Complex to polar
        testwav_magnitude, testwav_phase = complex_to_polar(freqseries)

        testwav_magnitude_align = shift_vec(testwav_magnitude,
                self.sample_frequencies, this_fpeak, self.fcenter).real

#       #
#       # Center & scale test spectrum
#       #
        magnitude_cent = np.copy(testwav_magnitude_align)
        magnitude_cent -= self.pca['magnitude_mean']
        #magnitude_cent /= self.pca['magnitude_std']

        phase_cent = np.copy(testwav_phase)
        phase_cent -= self.pca['phase_mean']
        phase_cent /= self.pca['phase_std']

        cpx_cent = np.copy(freqseries_align.data.real)
        cpx_cent -= self.pca['cpx_mean']
        #cpx_cent /= self.pca['cpx_std']

        #
        # Finally, project test spectrum onto PCs
        #
        projection['magnitude_betas'] = np.concatenate(
                self.pca['magnitude_pca'].transform(magnitude_cent)
                )

        projection['phase_betas'] = np.concatenate(
                self.pca['phase_pca'].transform(phase_cent)
                )

        projection['cpx_betas'] = np.concatenate(
                self.pca['cpx_pca'].transform(cpx_cent)
                )

        return projection

    def reconstruct(self, freqseries, npcs=1, this_fpeak=None, wfnum=None):
        """
        Reconstruct the waveform in freqseries using <npcs> principal components
        from the catalogue

        Procedure:
        1) Reconstruct the centered spectra (phase and mag) from the
        beta-weighted PCs
        2) Un-center the spectra (add the mean back on)
        """
        #print "Analysing reconstruction with %d PCs"%npcs


        if this_fpeak==None:
            # Locate fpeak
            # Note: we'll assume the peak we're aligning to is >2kHz.  This
            # avoids any low frequency stuff.
            high_idx = self.sample_frequencies>=2000 
            high_freq = self.sample_frequencies[high_idx] 
            this_fpeak = high_freq[np.argmax(abs(freqseries[high_idx]))]

        # Get projection:
        projection = self.project(freqseries)
        reconstruction=dict()

        #
        # Original Waveforms
        #
        orimag = abs(freqseries)
        oriphi = phase_of(freqseries)

        orispec = orimag*np.exp(1j*oriphi)

        reconstruction['original_spectrum'] = unit_hrss(orispec,
                delta=self.delta_f, domain='frequency')

        freqseries_align = shift_vec(freqseries, self.sample_frequencies,
                fcenter=self.fcenter, fpeak=this_fpeak)
        reconstruction['original_spectrum_align'] = unit_hrss(freqseries_align,
                delta=self.delta_f, domain='frequency')

        reconstruction['sample_frequencies'] = np.copy(self.sample_frequencies)

        #
        # Magnitude and phase reconstructions
        #

        # Initialise reconstructions
        recmag = np.zeros(shape=np.shape(orimag))
        recphi = np.zeros(shape=np.shape(oriphi))

        recspec = np.zeros(shape=np.shape(oriphi))

        # Sum contributions from PCs
        #npcs=10
        for i in xrange(npcs):

            recmag += \
                    projection['magnitude_betas'][i]*\
                    self.pca['magnitude_pca'].components_[i,:]
                                
            recphi += \
                    projection['phase_betas'][i]*\
                    self.pca['phase_pca'].components_[i,:]

            recspec += \
                    projection['cpx_betas'][i]*\
                    self.pca['cpx_pca'].components_[i,:]

        #
        # De-center the reconstruction
        #
        #recmag = recmag * self.pca['magnitude_std'] 
        recmag += self.pca['magnitude_mean']

        recphi = recphi * self.pca['phase_std']
        recphi += self.pca['phase_mean']

        #recspec = recspec * self.pca['cpx_std']
        recspec += self.pca['cpx_mean']


        #
        # Move the spectrum back to where it should be
        #
        recmag = shift_vec(recmag, self.sample_frequencies,
                fcenter=this_fpeak, fpeak=self.fcenter).real

        recspec = shift_vec(recspec, self.sample_frequencies,
                fcenter=this_fpeak, fpeak=self.fcenter).real

        reconstruction['recon_mag'] = np.copy(recmag)
        reconstruction['recon_phi'] = np.copy(recphi)
 
        reconstruction['residual_magnitude'] = residual_power(recmag, orimag)
        reconstruction['residual_phase'] = residual_power(recphi, oriphi)

        #
        # Fourier spectrum reconstructions
        #

        recon_spectrum = recmag * np.exp(1j*recphi)

#       f, ax = pl.subplots(nrows=2)
#
#       ax[0].plot(recspec, label='real from pcs')
#       ax[0].plot(np.real(freqseries), label='real sig')
#
#       ax[1].plot(-1*np.imag(signal.hilbert(recspec)), 
#               label='imag from pcs (via hilbert)')
#       ax[1].plot(np.imag(orimag*np.exp(1j*oriphi)), label='imag sig')
#
#       pl.show()
#       sys.exit()
#        recon_spectrum = recspec - 1j*np.imag(signal.hilbert(recspec))

 
        # --- Unit norm reconstruction
        reconstruction['recon_spectrum'] = unit_hrss(recon_spectrum,
                delta=self.delta_f, domain='frequency')

        # --- Match calculations for mag/phase reconstructions
        recon_spectrum = np.copy(reconstruction['recon_spectrum'].data)

        recmag = abs(recon_spectrum)
        recphi = phase_of(recon_spectrum)

#        reconstruction['recon_mag'] = np.copy(recmag)
#        reconstruction['recon_phi'] = np.copy(recphi)
 
#        reconstruction['residual_magnitude'] = residual_power(recmag, orimag)
#        reconstruction['residual_phase'] = residual_power(recphi, oriphi)


        # --- Match calculations for full reconstructions
        # make psd
        flen = len(self.sample_frequencies)
        psd = aLIGOZeroDetHighPower(flen, self.delta_f,
                low_freq_cutoff=self.low_frequency_cutoff)
 
        reconstruction['match_aligo'] = \
                pycbc.filter.match(reconstruction['recon_spectrum'],
                        reconstruction['original_spectrum'], psd = psd,
                        low_frequency_cutoff = self.low_frequency_cutoff)[0]
 

        reconstruction['match_noweight'] = \
                pycbc.filter.match(reconstruction['recon_spectrum'],
                        reconstruction['original_spectrum'],
                        low_frequency_cutoff = self.low_frequency_cutoff)[0]


        # compute overlap 'by hand':
#       idx = self.sample_frequencies>=self.low_frequency_cutoff
#
#       phase_diff = oriphi[idx] - recphi[idx] 
#       ampprod = recmag[idx] * orimag[idx]
#
#       numerator = ampprod*np.exp(1j*phase_diff)
#       
#       overlap = 4*np.real(np.trapz(numerator / psd.data[idx],
#           self.sample_frequencies[idx]))
#
#       N1 = np.sqrt(4*np.real(np.trapz(recmag[idx]*recmag[idx] / psd.data[idx],
#           self.sample_frequencies[idx])))
#       N2 = np.sqrt(4*np.real(np.trapz(orimag[idx]*orimag[idx] / psd.data[idx],
#           self.sample_frequencies[idx])))
#
#       N = N1*N2
#
#       reconstruction['match_aligo'] = overlap/N

        return reconstruction



def image_matches(match_matrix, waveform_names, title=None, mismatch=False):

    if mismatch:
        match_matrix = 1-match_matrix
        text_thresh = 0.1
        clims = (0,0.2)
        bar_label = 'mismatch'
    else:
        text_thresh = 0.85
        clims = (0.75,1.0)
        bar_label = 'match'

    #fig, ax = pl.subplots(figsize=(15,8))
    #fig, ax = pl.subplots(figsize=(7,7))
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

    ylabels=[name.replace('_lessvisc','') for name in waveform_names]
    ax.set_yticklabels(ylabels)

    im.set_clim(clims)
    im.set_cmap('gnuplot2')

    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Waveform type')

    if title is not None:
        ax.set_title(title)

    #c=pl.colorbar(im, ticks=np.arange(clims[0],clims[1]+0.05,0.05),
    #        orientation='horizontal')
    #c.set_label(bar_label)

    fig.tight_layout()

    return fig, ax

def image_residuals(residuals_matrix, waveform_names, title=None):

    text_thresh = 0.1
    clims = (0.0,0.25)
    bar_label = 'Residual'

    #fig, ax = pl.subplots(figsize=(15,8))
    #fig, ax = pl.subplots(figsize=(7,7))
    fig, ax = pl.subplots()
    nwaves = np.shape(residuals_matrix)[0]
    npcs = np.shape(residuals_matrix)[1]

    im = ax.imshow(residuals_matrix, interpolation='nearest', origin='lower',
            aspect='auto')

    for x in xrange(nwaves):
        for y in xrange(npcs):
            if residuals_matrix[x,y]<text_thresh:
                ax.text(y, x, '%.2f'%(residuals_matrix[x,y]), \
                    va='center', ha='center', color='k')
            else:
                ax.text(y, x, '%.2f'%(residuals_matrix[x,y]), \
                    va='center', ha='center', color='w')

    ax.set_xticks(range(0,npcs))
    ax.set_yticks(range(0,nwaves))

    xlabels=range(1,npcs+1)
    ax.set_xticklabels(xlabels)

    ylabels=[name.replace('_lessvisc','') for name in waveform_names]
    ax.set_yticklabels(ylabels)

    im.set_clim(clims)
    im.set_cmap('gnuplot2_r')

    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Waveform type')

    if title is not None:
        ax.set_title(title)

    #c=pl.colorbar(im, ticks=np.arange(clims[0],clims[1]+0.05,0.05),
    #        orientation='horizontal')
    #c.set_label(bar_label)

    fig.tight_layout()

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
    testwav_name = 'shen_135135_lessvisc'
    testwav_waveform = pmns_utils.Waveform(testwav_name)
    testwav_waveform.reproject_waveform()

    # Standardise
    testwav_waveform_FD, fpeak = condition_spectrum(testwav_waveform.hplus.data)

    # Normalise
    testwav_waveform_FD = unit_hrss(testwav_waveform_FD.data,
            delta=testwav_waveform_FD.delta_f, domain='frequency')

    # Reconstructions 
    reconstruction = pmpca.reconstruct(testwav_waveform_FD.data, npcs=1)

#   print reconstruction['match_aligo']
#
#
#   f, ax = pl.subplots()
#
#   ax.plot(reconstruction['sample_frequencies'],
#           abs(reconstruction['original_spectrum_align']), color='r',
#           linewidth=2)
#
#   ax.plot(reconstruction['sample_frequencies'],
#           abs(reconstruction['recon_spectrum_align']), color='k')
#
#   ax.set_xlim(0,1500)
#
#   #print projection['betas_magnitude']
#
#   pl.show()
#
#
#   # -----------
#   # Match Plots
#
#   #
#   # aligned waveforms
#   fig, ax = image_matches(shift_matches_ideal, waveform_names,
#           title="Matches For Full Aligned Spectrum PCA")
#
#   for fileformat in imageformats:
#       fig.savefig('shiftspec_ideal_matches.%s'%fileformat)
#
#
#   # UN-aligned waveforms
#   fig, ax = image_matches(unshift_matches_ideal, waveform_names,
#           title="Matches For aligned Full Spectrum PCA")
#
#   for fileformat in imageformats:
#       fig.savefig('unshiftspec_ideal_matches.%s'%fileformat)
#
#
#   # Original waveforms
#   fig, ax = image_matches(full_matches_ideal, waveform_names,
#           title="Matches For Full Spectrum PCA")
#
#   for fileformat in imageformats:
#       fig.savefig('fullspec_ideal_matches.%s'%fileformat)



#
# End definitions
#
if __name__ == "__main__":
    main()






