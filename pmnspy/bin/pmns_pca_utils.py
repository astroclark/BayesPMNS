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
from scipy import signal
from scipy import optimize
from scipy.spatial.distance import euclidean as euclidean_distance

from sklearn.decomposition import PCA 
from sklearn.decomposition import TruncatedSVD

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

import time
import cwt

# _________________ FUNCTIONS  _________________ #

def perform_TFpca(aligned_maps):

    tfmap_pca = PCA()

    #
    # Reshape data
    #
    n_samples, h, w = aligned_maps.shape
    maps = aligned_maps.reshape((n_samples, h * w))
    n_samples, n_features = maps.shape

    # global centering
    global_mean = maps.mean(axis=0)
    maps_centered = maps - global_mean

    # local centering
    #maps_centered -= maps_centered.mean(axis=1).reshape(n_samples, -1)

    #
    # PCA
    #
    pca = PCA(whiten=True)
    pca.fit(maps_centered)

    return pca, global_mean

def perform_pca(magnitudes, phases):
    """
    Do PCA with magnitude and phase parts of the complex waveforms in complex_catalogue
    """


    magnitude_spectra_centered, magnitude_mean, magnitude_std = \
            condition_magnitude(magnitudes)

    phase_spectra_centered, phase_mean, phase_std, phase_trend, pfits = \
            condition_phase(phases)

    mag_pca = PCA()
    #mag_pca.fit(magnitude_spectra_centered)
    mag_pca.fit(magnitudes)

    phase_pca = PCA()
    #phase_pca.fit(phase_spectra_centered)
    phase_pca.fit(phases)

    pcs_magphase={}

    pcs_magphase['magnitude_pca'] = mag_pca
    pcs_magphase['phase_pca'] = phase_pca


    pcs_magphase['magnitude_mean'] = magnitude_mean
    pcs_magphase['phase_mean'] = phase_mean

    pcs_magphase['magnitude_std'] = magnitude_std
    pcs_magphase['phase_std'] = phase_std

    pcs_magphase['phase_trend'] = phase_trend
    pcs_magphase['phase_fits'] = pfits

    return pcs_magphase


def condition_magnitude(magnitude_spectra):

    magnitude_spectra_centered = np.copy(magnitude_spectra)

    # --- Centering
    magnitude_mean = np.mean(magnitude_spectra, axis=0)
    for w in xrange(np.shape(magnitude_spectra)[0]):
        magnitude_spectra_centered[w,:] -= magnitude_mean

    # --- Scaling
    magnitude_std = np.std(magnitude_spectra, axis=0)

    #for w in xrange(np.shape(magnitude_spectra)[0]):
        #magnitude_spectra_centered[w,:] /= magnitude_std

    return magnitude_spectra_centered, magnitude_mean, magnitude_std


def condition_phase(phase_spectra, freqs=None, fmin=1000., fmax=4000.):
    """
    Center and scale the phase spectrum with optional trend removal
    """

    phase_spectra_centered = np.copy(phase_spectra)

    # --- Phase fits

    if freqs is None:
        freqs = np.arange(0,8192+16,16)

    fitidx = (freqs>fmin) * (freqs<fmax)

    x = np.arange(len(phase_spectra[0,:]))
    pfits = np.zeros(shape=np.shape(phase_spectra))
    for w in xrange(np.shape(pfits)[0]):
        y = phase_spectra[w,:]
        pfits[w,:] = poly4(x, y, fitidx)

    phase_trend = np.mean(pfits,axis=0)

    # --- Centering
    phase_mean = np.mean(phase_spectra_centered, axis=0)
    for w in xrange(np.shape(phase_spectra)[0]):
        phase_spectra_centered[w,:] -= phase_mean

    # --- Scaling
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
    return np.unwrap(np.angle(z))# - 2*np.pi

def build_catalogues(waveform_names, fshift_center, nTsamples=16384):
    """
    Build the data matrix
    """

    sample_freq=16384

    delta_t=1./sample_freq
    nFsamples=nTsamples/2 + 1
    sample_times=np.arange(0, delta_t*nTsamples, delta_t)

    # Preallocation
    timedomain_cat = np.zeros(shape=(len(waveform_names), len(sample_times)))
    aligned_cat = np.zeros(shape=(len(waveform_names), nFsamples),  dtype=complex)
    original_cat = np.zeros(shape=(len(waveform_names), nFsamples), dtype=complex)
    fpeaks = np.zeros(len(waveform_names))

    # Preallocate for the TF maps
    example_map = example_tfmap(delta_t=delta_t)
    mapdims = np.shape(example_map['map'])
    times = example_map['times']
    frequencies = example_map['frequencies']
    scales = example_map['scales']

    original_image_cat = np.zeros(shape=(len(waveform_names), mapdims[0],
        mapdims[1]))
    align_image_cat = np.zeros(shape=(len(waveform_names), mapdims[0],
        mapdims[1]))

    for w, name in enumerate(waveform_names):

        #
        # Create waveform instance: pmns_utils
        #
        waveform = pmns_utils.Waveform(name)
        waveform.reproject_waveform()

        # Waveform conditioning
        original_spectrum, fpeaks[w], timeseries = \
                condition_spectrum(waveform.hplus.data, nsamples=nTsamples)

        # Populate time series catalogue
        timedomain_cat[w, :] = np.copy(timeseries.data)

        # Time-frequency maps
        tfmap = build_cwt(pycbc.types.TimeSeries(waveform.hplus.data,
            delta_t=delta_t))
        original_image_cat[w,:,:] = tfmap['map']

        aligned_tfmap = align_cwt(tfmap, fpeaks[w])
        align_image_cat[w,:,:] = aligned_tfmap

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

        # Populate catalogue with normalised aligned spectrum
        aligned_cat[w,:] = unit_hrss(aligned_spectrum,
                delta=original_spectrum.delta_f, domain='frequency').data

    return (sample_times, original_frequencies, timedomain_cat, aligned_cat,
            original_cat, fpeaks, original_image_cat, align_image_cat, times,
            scales, frequencies)

def example_tfmap(name='shen_135135_lessvisc', delta_t=1./16384):

        #
        # Create waveform instance: pmns_utils
        #
        waveform = pmns_utils.Waveform(name)
        waveform.reproject_waveform()

        tfmap = build_cwt(pycbc.types.TimeSeries(waveform.hplus.data,
            delta_t=delta_t))

        return tfmap


def condition_spectrum(waveform_timeseries, delta_t=1./16384, nsamples=16384):
    """
    Zero-pad, window and FFT a time-series to return a frequency series of
    standard length (16384 samples)
    """

    # Time-domain Window
    win=lal.CreateTukeyREAL8Window(len(waveform_timeseries),0.1)
    #waveform_timeseries = np.copy(ts.data)
    waveform_timeseries *= win.data.data

    # Zero-pad
    paddata = np.zeros(nsamples)
    paddata[:len(waveform_timeseries)] = np.copy(waveform_timeseries)


    # FFT
    timeseries = pycbc.types.TimeSeries(initial_array=paddata, delta_t=delta_t)
    timeseries = unit_hrss(paddata, delta_t, 'time')
    #timeseries = pycbc.filter.highpass(timeseries, 1000, filter_order=8)

    freqseries = timeseries.to_frequencyseries()

    # Locate fpeak
    high_idx = freqseries.sample_frequencies.data>=2000 
    high_freq = freqseries.sample_frequencies.data[high_idx] 
    fpeak = high_freq[np.argmax(abs(freqseries[high_idx]))]
    
    return freqseries, fpeak, timeseries

def poly4(x, y, idx=None):
    """
    helper func for 4th order polyfit for phase conditioning
    """

    # 4th order polyfit to phase for trend removal
    if idx is None:
        p = np.polyfit(x, y, 4)
    else:
        p = np.polyfit(x[idx], y[idx], 4)

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


def wrap_phase(phase):
    """
    Opposite of np.wrap()
    """
    return ( phase + np.pi) % (2 * np.pi ) - np.pi

def build_cwt(timeseries, max_scale=256, mother_freq=2, Norm=True, fmin=1000.0,
        fmax=4096, maplen=2048):


    # Make sure the timeseries peaks in the middle of the map
    paddata = np.zeros(maplen)
    peakidx = np.argmax(timeseries.data)
    paddata[0.5*maplen-peakidx:0.5*maplen] = timeseries.data[:peakidx]
    paddata[0.5*maplen:0.5*maplen+len(timeseries.data)-peakidx] = timeseries.data[peakidx:]

    timeseries = pycbc.types.TimeSeries(paddata, delta_t=timeseries.delta_t)

    sample_rate = 1./timeseries.delta_t

    # Range of wavelet scales we're interested in
    scales = 1+np.arange(max_scale)

    # Construct the 'mother wavelet'; we'll begin using a Morlet wavelet
    # (sine-Gaussian) but we should/could investigate others
    # Could also experiment with values of f0.  See cwt.Morlet for info

    mother_wavelet = cwt.Morlet(len_signal = \
            len(timeseries), scales = scales,
            sampf=sample_rate, f0=mother_freq)

    # Compute the CWT
    wavelet = cwt.cwt(timeseries.data, mother_wavelet)

    # Take the absolute values of coefficients 
    tfmap = np.abs(wavelet.coefs)


    # Reduce to useful frequencies
    freqs = sample_rate * wavelet.motherwavelet.fc \
            / wavelet.motherwavelet.scales
    tfmap[(freqs<fmin)+(freqs>fmax),:] = 0.0

    # Normalise
    tfmap /= max(map(max,abs(wavelet.coefs)))

    # Return a dictionary
    timefreq = dict()
    timefreq['map'] = tfmap
    timefreq['times'] = timeseries.sample_times.data
    timefreq['frequencies'] = freqs
    timefreq['scales'] = scales
    timefreq['mother_wavelet'] = mother_wavelet
    timefreq['image_shape'] = np.shape(tfmap)


    return timefreq



def align_cwt(timefreqmap, fpeak):
    """
    Center and Scale the time / frequency map analogously to how we handle
    spectra
    """

    outputmap = np.copy(timefreqmap['map'])

    peak_scale = timefreqmap['scales']\
            [abs(timefreqmap['frequencies']-fpeak).argmin()]

    # shift columns
    for c in xrange(np.shape(outputmap)[1]):
        outputmap[:,c] = shift_vec(outputmap[:,c], timefreqmap['scales'],
                peak_scale, 0.25*timefreqmap['scales'].max()).real

    return outputmap

def dealign_cwt(timefreqmap, fpeak):
    """
    Center and Scale the time / frequency map analogously to how we handle
    spectra
    """

    outputmap = np.copy(timefreqmap['map'])

    peak_scale = timefreqmap['scales']\
            [abs(timefreqmap['frequencies']-fpeak).argmin()]

    # shift columns
    for c in xrange(np.shape(outputmap)[1]):
        outputmap[:,c] = shift_vec(outputmap[:,c], timefreqmap['scales'],
                0.25*timefreqmap['scales'].max(), peak_scale).real

    return outputmap
 
#    sys.exit()

# _________________ CLASSES  _________________ #

class pmnsPCA:
    """
    An object with a catalogue and principal component decomposition of
    post-merger waveforms

    """

    def __init__(self, waveform_list, fcenter=1000, low_frequency_cutoff=1000,
            nTsamples=16384):

        #
        # Build Catalogues
        #
        self.fcenter=fcenter
        self.waveform_list=waveform_list

        print "Building catalogue"
        (self.sample_times, self.sample_frequencies, self.cat_timedomain,
                self.cat_align, self.cat_orig, self.fpeaks,
                self.original_image_cat, self.align_image_cat, self.map_times,
                self.map_scales, self.map_frequencies) = \
                        build_catalogues(self.waveform_list, self.fcenter,
                                nTsamples=nTsamples)
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

        # -- Do PCA
        print "Performing Spectral PCA"
        t0 = time.time()
        self.pca = perform_pca(self.magnitude_align, self.phase)
        train_time = (time.time() - t0)
        print("...done in %0.3fs" % train_time)

        print "Performing Time-Frequency PCA"
        t0 = time.time()
        self.pca['timefreq_pca'], self.pca['timefreq_mean'] = \
                perform_TFpca(self.align_image_cat)
        train_time = (time.time() - t0)
        print("...done in %0.3fs" % train_time)


    def project_freqseries(self, freqseries, this_fpeak=None):
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

        #
        # Center & scale test spectrum
        #
        magnitude_cent = np.copy(testwav_magnitude_align)
        #magnitude_cent -= self.pca['magnitude_mean']

        projection['magnitude_cent'] = magnitude_cent

        phase_cent = np.copy(testwav_phase)
        #phase_cent -= self.pca['phase_mean']
        #phase_cent /= self.pca['phase_std']

        projection['phase_cent'] = phase_cent

        #
        # Finally, project test spectrum onto PCs
        #
        projection['magnitude_betas'] = np.concatenate(
                self.pca['magnitude_pca'].transform(magnitude_cent)
                )

        projection['phase_betas'] = np.concatenate(
                self.pca['phase_pca'].transform(phase_cent)
                )

        return projection


    def reconstruct_freqseries(self, freqseries, npcs=1, this_fpeak=None, wfnum=None):
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
        fd_projection = self.project_freqseries(freqseries)
        fd_reconstruction=dict()

        #
        # Original Waveforms
        #
        orimag = abs(freqseries)
        oriphi = phase_of(freqseries) - phase_of(freqseries)[0]
        #oriphi = self.pca['phase_fits'][wfnum,:]

        phase_fit = poly4(np.arange(len(oriphi)), oriphi)

        orispec = orimag*np.exp(1j*oriphi)

        fd_reconstruction['original_spectrum'] = unit_hrss(orispec,
                delta=self.delta_f, domain='frequency')

        freqseries_align = shift_vec(freqseries, self.sample_frequencies,
                fcenter=self.fcenter, fpeak=this_fpeak)
        fd_reconstruction['original_spectrum_align'] = unit_hrss(freqseries_align,
                delta=self.delta_f, domain='frequency')

        fd_reconstruction['sample_frequencies'] = np.copy(self.sample_frequencies)

        #
        # Magnitude and phase reconstructions
        #

        # Initialise reconstructions
        recmag = np.zeros(shape=np.shape(orimag))
        recphi = np.zeros(shape=np.shape(oriphi))

        # Sum contributions from PCs
        for i in xrange(npcs):

            recmag += \
                    fd_projection['magnitude_betas'][i]*\
                    self.pca['magnitude_pca'].components_[i,:]
                                
            recphi += \
                    fd_projection['phase_betas'][i]*\
                    self.pca['phase_pca'].components_[i,:]



        #
        # De-center the reconstruction
        #

#       recmag += self.pca['magnitude_mean']
        recmag += self.pca['magnitude_pca'].mean_
#
#       recphi = recphi * self.pca['phase_std']
#       recphi += self.pca['phase_mean']
        recphi += self.pca['phase_pca'].mean_

        # --- Raw reconstruction quality
        idx = (self.sample_frequencies>self.low_frequency_cutoff) \
                * (orimag>0.01*max(orimag))

        fd_reconstruction['magnitude_euclidean_raw'] = \
                euclidean_distance(recmag[idx], fd_projection['magnitude_cent'][idx])

        fd_reconstruction['phase_euclidean_raw'] = \
                euclidean_distance(recphi[idx], fd_projection['phase_cent'][idx])



        #recphi = phase_fit

        #
        # Move the spectrum back to where it should be
        #
        recmag = shift_vec(recmag, self.sample_frequencies,
                fcenter=this_fpeak, fpeak=self.fcenter).real

        fd_reconstruction['recon_mag'] = np.copy(recmag)
        fd_reconstruction['recon_phi'] = np.copy(recphi)

        #
        # Fourier spectrum reconstructions
        #

        recon_spectrum = recmag * np.exp(1j*recphi)

 
        # --- Unit norm reconstruction
        fd_reconstruction['recon_spectrum'] = unit_hrss(recon_spectrum,
                delta=self.delta_f, domain='frequency')

        fd_reconstruction['recon_timeseries'] = \
                fd_reconstruction['recon_spectrum'].to_timeseries()


        # --- Match calculations for mag/phase reconstructions
        recon_spectrum = np.copy(fd_reconstruction['recon_spectrum'].data)


        # --- Match calculations for full reconstructions


        idx = (self.sample_frequencies>self.low_frequency_cutoff) \
                * (orimag>0.01*max(orimag))


        fd_reconstruction['magnitude_euclidean'] = \
                euclidean_distance(recmag[idx], orimag[idx])

        fd_reconstruction['phase_euclidean'] = \
                euclidean_distance(recphi[idx], oriphi[idx])


        # make psd
        flen = len(self.sample_frequencies)
        psd = aLIGOZeroDetHighPower(flen, self.delta_f,
                low_freq_cutoff=self.low_frequency_cutoff)
 
        fd_reconstruction['match_aligo'] = \
                pycbc.filter.match(fd_reconstruction['recon_spectrum'],
                        fd_reconstruction['original_spectrum'], psd = psd,
                        low_frequency_cutoff = self.low_frequency_cutoff)[0]
 

        fd_reconstruction['match_noweight'] = \
                pycbc.filter.match(fd_reconstruction['recon_spectrum'],
                        fd_reconstruction['original_spectrum'],
                        low_frequency_cutoff = self.low_frequency_cutoff)[0]


        return fd_reconstruction


    def project_tfmap(self, tfmap, this_fpeak=None):
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
            print >> sys.stderr, "require desired fpeak"
            sys.exit()

        # Dictionary to hold input and results of projection
        projection = dict()
        projection['tfmap'] = np.copy(tfmap['map'])

        # Align test spectrum
        projection['tfmap_align'] = align_cwt(tfmap, fpeak=this_fpeak)

        
        # Reshape
        h, w = projection['tfmap_align'].shape
        reshaped_map = projection['tfmap_align'].reshape((h * w))

        #
        # Finally, project test map onto PCs
        #
        projection['timefreq_betas'] = np.concatenate(
                self.pca['timefreq_pca'].transform(reshaped_map)
                )

        return projection

    def reconstruct_tfmap(self, tfmap, npcs=1 , this_fpeak=None, wfnum=None):
        """
        Reconstruct the given timefrequency map tfmap by projecting onto the
        current instance's PCs
        """

        if this_fpeak==None:
            print >> sys.stderr, "require desired fpeak"
            sys.exit()

        #
        # Compute projection of this map onto the PCs
        #
        tf_projection = self.project_tfmap(tfmap, this_fpeak=this_fpeak)

        #
        # Reconstruct the waveform
        #
        h, w = tfmap['map'].shape
        reshaped_map = tfmap['map'].reshape((h * w))

        recmap_align = dict()
        recmap_align['map'] = np.zeros(np.shape(reshaped_map))
        npcs=10
        for i in xrange(npcs):
            if np.allclose(0,\
                    self.pca['timefreq_pca'].explained_variance_ratio_[i]):
                continue
            else:
                recmap_align['map'] += tf_projection['timefreq_betas'][i]*\
                        self.pca['timefreq_pca'].components_[i,:]

        #
        # De-center and realign the reconstruction
        #

        pl.figure()
        pl.plot(recmap_align['map'] + self.pca['timefreq_mean'])
        pl.plot(tf_projection['tfmap_align'].reshape((h * w)))
        pl.show()
        sys.exit()

        # Reshape
        recmap_align['map'] += self.pca['timefreq_mean']
        recmap_align['map'] = recmap_align['map'].reshape(h,w)


        recmap_align['times'] = np.copy(tfmap['times'])
        recmap_align['frequencies'] = np.copy(tfmap['frequencies'])
        recmap_align['scales'] = np.copy(tfmap['scales'])
        recmap_align['mother_wavelet'] = tfmap['mother_wavelet']
        recmap_align['image_shape'] = tfmap['image_shape']

        pl.figure()
        pl.contourf(recmap_align['map'])
        pl.figure()
        pl.contourf(tf_projection['tfmap_align'])
        pl.show()
        sys.exit()

        # realign
        recmap = recmap_align.copy()
        recmap['map'] = dealign_cwt(recmap, this_fpeak)


        #
        # Populate the output dictionary
        #
        tf_reconstruction=dict()
        tf_reconstruction['orig_map'] = recmap
        tf_reconstruction['align_map'] = recmap_align

        return tf_reconstruction


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

def image_euclidean(euclidean_matrix, waveform_names, title=None, clims=None):

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



#
# End definitions
#
if __name__ == "__main__":
    main()






