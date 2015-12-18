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
pmns_pca.py

Utilities for PCA of post-merger waveforms
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

import lal
import lalsimulation as lalsim
from pmns_utils import pmns_waveform

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


    #
    # PCA
    #
    #pca = PCA(whiten=True)
    pca = PCA(whiten=False)
    pca.fit(maps_centered)

    return pca, global_mean

def perform_pca(magnitudes, phases):
    """
    Do PCA with magnitude and phase parts of the complex waveforms in complex_catalogue
    """

    mag_pca = PCA()
    mag_pca.fit(magnitudes)

    phase_pca = PCA()
    phase_pca.fit(phases)

    pcs_magphase={}

    pcs_magphase['magnitude_pca'] = mag_pca
    pcs_magphase['phase_pca'] = phase_pca


    return pcs_magphase


def complex_to_polar(catalogue):
    """
    Convert the complex Fourier spectrum to an amplitude and phase
    """

    magnitudes = np.zeros(shape=np.shape(catalogue))
    phases = np.zeros(shape=np.shape(catalogue))
    if len(np.shape(catalogue))==1:
        return abs(catalogue), phase_of(catalogue) #- phase_of(catalogue)[0]

    for c in xrange(np.shape(catalogue)[0]):
        magnitudes[c,:] = abs(catalogue[c,:])
        phases[c,:] = phase_of(catalogue[c,:]) #- phase_of(catalogue[c,:])[0]

    return magnitudes, phases

def phase_of(z):
    return np.unwrap(np.angle(z))# - 2*np.pi

def build_catalogues(waveform_data, fshift_center, nTsamples=16384,
        delta_t=1./16384):
    """
    Build the data matrix.  waveform_data is the pmns_waveform_data.WaveData()
    class which contains the list of dictionaries of waveform data
    """

    nFsamples=nTsamples/2 + 1
    sample_times=np.arange(0, delta_t*nTsamples, delta_t)

    # Preallocation
    timedomain_cat = np.zeros(shape=(waveform_data.nwaves, len(sample_times)))
    original_cat = np.zeros(shape=(waveform_data.nwaves, nFsamples), dtype=complex)
    fpeaks = np.zeros(waveform_data.nwaves)

    # Preallocate for the TF maps
    example_waveform=pmns_waveform.Waveform(eos=waveform_data.waves[0]['eos'],
            mass=waveform_data.waves[0]['mass'],
            viscosity=waveform_data.waves[0]['viscosity'])
    example_waveform.reproject_waveform()
    example_map = example_tfmap(waveform=example_waveform, delta_t=delta_t)

    mapdims = np.shape(example_map['map'])
    times = example_map['times']
    frequencies = example_map['frequencies']
    scales = example_map['scales']

    original_image_cat = np.zeros(shape=(waveform_data.nwaves, mapdims[0],
        mapdims[1]))
    align_image_cat = np.zeros(shape=(waveform_data.nwaves, mapdims[0],
        mapdims[1]))

    for w, wave in enumerate(waveform_data.waves):

        print "Building waveform for %s, %s, %s (%d of %d)"%(wave['eos'], wave['mass'],
                wave['viscosity'], w+1, waveform_data.nwaves)

        #
        # Create waveform instance: pmns_utils
        #
        waveform = pmns_waveform.Waveform(eos=wave['eos'], mass=wave['mass'],
                viscosity=wave['viscosity'])
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

    return (sample_times, original_frequencies, timedomain_cat,
            original_cat, fpeaks, original_image_cat, align_image_cat, times,
            scales, frequencies)

def example_tfmap(waveform, delta_t=1./16384):

        tfmap = build_cwt(pycbc.types.TimeSeries(waveform.hplus.data,
            delta_t=delta_t))

        return tfmap


def condition_spectrum(waveform_timeseries, delta_t=1./16384, nsamples=16384):
    """
    Zero-pad, window and FFT a time-series to return a frequency series of
    standard length (16384 samples)
    """

#   # Time-domain Window
#   win=lal.CreateTukeyREAL8Window(len(waveform_timeseries),0.1)
#   win.data.data[len(win.data.data):] = 1.0
#   waveform_timeseries *= win.data.data

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
    #scales = np.logspace(0,np.log10(max_scale),max_scale)

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
    timefreq['analysed_data'] = timeseries
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

    def __init__(self, waveform_data, fcenter=1000, low_frequency_cutoff=1000,
            nTsamples=16384):

        #
        # Build Catalogues
        #
        self.fcenter=fcenter
        self.waveform_data=waveform_data

        print "Building catalogue"
        (self.sample_times, self.sample_frequencies, self.cat_timedomain,
                self.cat_orig, self.fpeaks,
                self.original_image_cat, self.align_image_cat, self.map_times,
                self.map_scales, self.map_frequencies) = \
                        build_catalogues(self.waveform_data, self.fcenter,
                                nTsamples=nTsamples)

        self.delta_f = np.diff(self.sample_frequencies)[0]
        self.delta_t = 1./16384

        # min freq for match calculations
        self.low_frequency_cutoff=low_frequency_cutoff

        #
        # PCA
        #

        print "Aligning"

        # Convert to magnitude/phase
        self.magnitude, self.phase = complex_to_polar(self.cat_orig)

        self.magnitude_align = np.zeros(np.shape(self.magnitude))
        self.phase_align = np.zeros(np.shape(self.phase))
        for i in xrange(np.shape(self.magnitude)[0]):

            self.magnitude_align[i,:] = shift_vec(self.magnitude[i,:],
                    self.sample_frequencies, self.fpeaks[i], self.fcenter).real


            self.phase_align[i,:] = shift_vec(self.phase[i,:],
                    self.sample_frequencies, self.fpeaks[i], self.fcenter).real


        # -- Do PCA
        print "Performing Spectral PCA"
        t0 = time.time()
        self.pca = perform_pca(self.magnitude_align, self.phase_align)
        #self.pca = perform_pca(self.magnitude_align, self.phase)
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
            high_spec = freqseries[high_idx] 
            this_fpeak = high_freq[np.argmax(abs(high_spec))]

        # Dictionary to hold input and results of projection
        projection = dict()
        projection['freqseries'] = np.copy(freqseries)

        # Align test spectrum

        # Complex to polar
        testwav_magnitude, testwav_phase = complex_to_polar(freqseries)

        testwav_magnitude_align = shift_vec(testwav_magnitude,
                self.sample_frequencies, this_fpeak, self.fcenter).real
        testwav_phase_align = shift_vec(testwav_phase,
                self.sample_frequencies, this_fpeak, self.fcenter).real

        #
        # Center & scale test spectrum
        #
        magnitude_cent = np.copy(testwav_magnitude_align)
        projection['magnitude_cent'] = magnitude_cent

        phase_cent = np.copy(testwav_phase_align)
        #phase_cent = np.copy(testwav_phase)
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
            high_spec = freqseries[high_idx] 
            this_fpeak = high_freq[np.argmax(abs(high_spec))]

        # Get projection:
        fd_projection = self.project_freqseries(freqseries)
        fd_reconstruction=dict()

        fd_reconstruction['fd_projection'] = fd_projection

        #
        # Original Waveforms
        #
        orimag = abs(freqseries)
        oriphi = phase_of(freqseries)# - phase_of(freqseries)[0]

        orispec = orimag*np.exp(1j*oriphi)


        fd_reconstruction['original_spectrum'] = unit_hrss(orispec,
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

        recmag += self.pca['magnitude_pca'].mean_
        recphi += self.pca['phase_pca'].mean_

        # --- Raw reconstruction quality
        idx = (self.sample_frequencies>self.low_frequency_cutoff) \
                * (orimag>0.01*max(orimag))

        fd_reconstruction['magnitude_euclidean_raw'] = \
                euclidean_distance(recmag[idx], fd_projection['magnitude_cent'][idx])

        fd_reconstruction['phase_euclidean_raw'] = \
                euclidean_distance(recphi[idx], fd_projection['phase_cent'][idx])


        #
        # Move the spectrum back to where it should be
        #
        recmag = shift_vec(recmag, self.sample_frequencies,
                fcenter=this_fpeak, fpeak=self.fcenter).real
        recphi = shift_vec(recphi, self.sample_frequencies,
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
        reshaped_map = projection['tfmap_align'].reshape((h * w)) \
                - self.pca['timefreq_mean']

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

        recmap_align = dict()
        recmap_align['map'] = np.zeros(h*w)

        for i in xrange(npcs):
            recmap_align['map'] += tf_projection['timefreq_betas'][i]*\
                    self.pca['timefreq_pca'].components_[i,:]

        #
        # De-center and realign the reconstruction
        #
 
        # Reshape
        recmap_align['map'] += self.pca['timefreq_mean']
        recmap_align['map'][recmap_align['map']<0] = 0.0
        recmap_align['map'] = recmap_align['map'].reshape(h,w)

        recmap_align['times'] = np.copy(tfmap['times'])
        recmap_align['frequencies'] = np.copy(tfmap['frequencies'])
        recmap_align['scales'] = np.copy(tfmap['scales'])
        recmap_align['mother_wavelet'] = tfmap['mother_wavelet']
        recmap_align['image_shape'] = tfmap['image_shape']

        # realign
        recmap = recmap_align.copy()
        recmap['map'] = dealign_cwt(recmap, this_fpeak)

        #
        # Populate the output dictionary
        #
        tf_reconstruction=dict()
        tf_reconstruction['orig_map'] = recmap.copy()
        tf_reconstruction['align_map'] = recmap_align.copy()

        tf_reconstruction['tfmap_euclidean_raw'] = euclidean_distance(
                recmap_align['map'].reshape(h*w),
                tf_projection['tfmap_align'].reshape(h*w)
                )

        tf_reconstruction['tfmap_euclidean'] = euclidean_distance(
                recmap['map'].reshape(h*w), tfmap['map'].reshape(h*w))


        return tf_reconstruction




# *******************************************************************************
def main():

    print 'nothing here'



#
# End definitions
#
if __name__ == "__main__":
    main()






