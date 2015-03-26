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

    #for w in xrange(np.shape(complex_catalogue)[1]):
    #    phases[:,w] = signal.detrend(phases[:,w])

    mean_mag   = np.zeros(np.shape(complex_catalogue)[0])
    mean_phase = np.zeros(np.shape(complex_catalogue)[0])
    for s in xrange(np.shape(complex_catalogue)[0]):
        mean_mag[s]   = np.mean(magnitudes[s,:])
        mean_phase[s] = np.mean(phases[s,:])


    for w in xrange(np.shape(complex_catalogue)[1]):
        magnitudes[:,w] -= mean_mag
        phases[:,w] -= mean_phase

    pcs_magphase = {}

    pcs_magphase['magnitude_scores'], pcs_magphase['magnitude_pcs'], \
            pcs_magphase['magnitude_betas'], pcs_magphase['magnitude_eigenvalues'] = \
            pca_by_svd(magnitudes)

    pcs_magphase['phase_scores'], pcs_magphase['phase_pcs'], \
            pcs_magphase['phase_betas'], pcs_magphase['phase_eigenvalues'] = \
            pca_by_svd(phases)

    pcs_magphase['magnitude_eigenergy'] = \
            eigenergy(pcs_magphase['magnitude_eigenvalues'])
    pcs_magphase['phase_eigenergy'] = \
            eigenergy(pcs_magphase['phase_eigenvalues'])

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

def dotmatch(vector1, vector2):
    """
    Compute the normalised dot product between vector1 and vector 2:

    result = dot(vector1/norm(vector1), vector2/norm(vector2))

    so that result=1 <=> vector1==vector2
    """

    #vector1_normed = vector1 / np.linalg.norm(vector1)
    #vector2_normed = vector2 / np.linalg.norm(vector2)
    vector1_normed = vector1 / np.sqrt(np.inner(vector1,vector1))
    vector2_normed = vector2 / np.sqrt(np.inner(vector2,vector2))

    #return np.dot(vector1_normed,vector2_normed)
    return np.inner(vector1_normed,vector2_normed).real

# _________________ CLASSES  _________________ #

class pmnsPCA:
    """
    An object with a catalogue and principal component decomposition of
    post-merger waveforms

    """

    def __init__(self, waveform_list, fcenter=1000, low_frequency_cutoff=1000):

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

        projection['freqseries_align'] = np.copy(freqseries_align.data)

        # Complex to polar
        magnitude_align, phase_align = complex_to_polar(freqseries_align.data)

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
        reconstruction['original_spectrum'] = unit_hrss(freqseries,
                delta=self.delta_f, domain='frequency')
        reconstruction['original_spectrum_align'] = \
                unit_hrss(projection['freqseries_align'], delta=self.delta_f,
                        domain='frequency')
        reconstruction['sample_frequencies'] = np.copy(self.sample_frequencies)

        #
        # Magnitude and phase reconstructions
        #

        # Initialise reconstructions
        reconstruction['recon_magnitude_align'] = np.copy(self.pca['mean_mag'])
        reconstruction['recon_phase_align'] = np.copy(self.pca['mean_phase'])

        # Sum contributions from PCs
        for i in xrange(npcs):

            # Sanity Check: if the eigenvalue for the current PC is close to
            # zero, ignore that PC
            #if np.isclose(0, self.pca['magnitude_eigenvalues'][i]):
            if abs(self.pca['magnitude_eigenvalues'][i])<1e-5:
                reconstruction['recon_magnitude_align'] += 0.0
            else:
                reconstruction['recon_magnitude_align'] += \
                        projection['betas_magnitude'][i]*self.pca['magnitude_pcs'][:,i]

            #if np.isclose(0, self.pca['phase_eigenvalues'][i]):
            if abs(self.pca['phase_eigenvalues'][i])<1e-5:
                reconstruction['recon_phase_align'] += 0.0
            else:
                reconstruction['recon_phase_align'] += \
                        projection['betas_phase'][i]*self.pca['phase_pcs'][:,i]

        # --- Match calculations for mag/phase reconstructions
        reconstruction['dotmatch_magnitude'] = dotmatch(
                reconstruction['recon_magnitude_align'],
                abs(reconstruction['original_spectrum_align']))
        
        reconstruction['dotmatch_phase'] = dotmatch(
                reconstruction['recon_phase_align'],
                np.unwrap(np.angle(reconstruction['original_spectrum_align'])))

        #
        # Fourier spectrum reconstructions
        #

        recmag = reconstruction['recon_magnitude_align']
        orimag = abs(reconstruction['original_spectrum_align'])

        recphi = reconstruction['recon_phase_align']
        oriphi = np.unwrap(np.angle(reconstruction['original_spectrum_align']))

        # XXX

        pl.figure()
        pl.plot(recmag, label='reconstructed')
        pl.plot(orimag, label='original')
        pl.xlabel('frequency index')
        pl.ylabel('magnitude')
        pl.title('magnitude spectra: (rec|original)=%.2f'%dotmatch(recmag,orimag))
        pl.xlim(0,1500)
        pl.xlim(0,1500)


        pl.figure()
        pl.plot(recphi, label='reconstructed')
        pl.plot(oriphi, label='original')
        pl.xlabel('frequency index')
        pl.ylabel('phase')
        pl.title('phase spectra: (rec|original)=%.2f'%dotmatch(recphi,oriphi))
        pl.xlim(0,1500)
        pl.xlim(0,1500)

    
        reccplx = recmag*np.exp(1j*recphi)
        oricplx = orimag*np.exp(1j*oriphi)

        f, ax = pl.subplots(nrows=2)
        ax[0].plot(reccplx.real, label='reconstructed')
        ax[0].plot(oricplx.real, label='original')
        ax[0].set_xlabel('frequency index')
        ax[0].set_ylabel('Re[H(f)]')
        ax[1].plot(reccplx.imag, label='reconstructed')
        ax[1].plot(oricplx.imag, label='original')
        ax[1].set_xlabel('frequency index')
        ax[1].set_ylabel('Im[H(f)]')

        reconstruction['recon_spectrum_align'] = \
                unit_hrss(recmag*np.exp(1j*recphi),
                delta=self.delta_f, domain='frequency')

        ov = \
                pycbc.filter.overlap(reconstruction['recon_spectrum_align'],
                        reconstruction['original_spectrum_align'],
                        low_frequency_cutoff = self.low_frequency_cutoff)

        ma = \
                pycbc.filter.match(reconstruction['recon_spectrum_align'],
                        reconstruction['original_spectrum_align'],
                        low_frequency_cutoff = self.low_frequency_cutoff)[0]

        ax[0].set_title('H(f): match=%.3f, overlap=%.3f'%(ma,ov))
        ax[0].set_xlim(0,1500)
        ax[1].set_xlim(0,1500)

        pl.show()
        sys.exit()

        # XXX

        reconstruction['recon_spectrum_align'] = \
                unit_hrss(recmag*np.exp(1j*recphi),
                delta=self.delta_f, domain='frequency')

        recon_spectrum = unshift_vec(reconstruction['recon_spectrum_align'].data,
                self.sample_frequencies, fpeak=this_fpeak)

        reconstruction['recon_spectrum'] = unit_hrss(recon_spectrum,
                delta=self.delta_f, domain='frequency')

        # --- Match calculations for full reconstructions

        # make psd
        flen = len(self.sample_frequencies)
        psd = aLIGOZeroDetHighPower(flen, self.delta_f,
                low_freq_cutoff=self.low_frequency_cutoff)

        reconstruction['match_aligo_align'] = \
               pycbc.filter.match(reconstruction['recon_spectrum_align'],
                       reconstruction['original_spectrum_align'], psd = psd,
                       low_frequency_cutoff = self.low_frequency_cutoff)[0]

        reconstruction['match_aligo'] = \
                pycbc.filter.match(reconstruction['recon_spectrum'],
                        reconstruction['original_spectrum'], psd = psd,
                        low_frequency_cutoff = self.low_frequency_cutoff)[0]

        reconstruction['match_noweight_align'] = \
                pycbc.filter.overlap(reconstruction['recon_spectrum_align'],
                        reconstruction['original_spectrum_align'],
                        low_frequency_cutoff = self.low_frequency_cutoff)

#               [dotmatch(reconstruction['recon_spectrum_align'],
#                       reconstruction['original_spectrum_align'])]


        reconstruction['match_noweight'] = \
                pycbc.filter.match(reconstruction['recon_spectrum'],
                        reconstruction['original_spectrum'],
                        low_frequency_cutoff = self.low_frequency_cutoff)[0]


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

    c=pl.colorbar(im, ticks=np.arange(clims[0],clims[1]+0.05,0.05),
            orientation='horizontal')
    c.set_label(bar_label)

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






