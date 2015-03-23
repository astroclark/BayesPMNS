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

def pca_by_svd(catalogue):
    """
    Perform principle component analysis via singular value decomposition
    """

    U, S, Vt = scipy.linalg.svd(catalogue, full_matrices=True)

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
    #return U, V, S**2 

def pca_magphase(catalogue, freqs, flow=1000):
    """
    Do PCA with magnitude and phase parts of the complex waveforms in catalogue
    """

    # 'highpass'
    catalogue[freqs<flow] = 0.0

    magnitudes, phases = complex_to_polar(catalogue)

    for w in xrange(np.shape(catalogue)[1]):
        phases[:,w] = signal.detrend(phases[:,w])

    mean_mag   = np.zeros(np.shape(catalogue)[0])
    mean_phase = np.zeros(np.shape(catalogue)[0])
    #std_mag = np.zeros(np.shape(catalogue)[0])
    #std_phase = np.zeros(np.shape(catalogue)[0])
    for s in xrange(np.shape(catalogue)[0]):
        mean_mag[s]   = np.mean(magnitudes[s,:])
        mean_phase[s] = np.mean(phases[s,:])
        #std_mag[s] = np.std(magnitudes[s,:])
        #std_phase[s] = np.std(phases[s,:])


    for w in xrange(np.shape(catalogue)[1]):
        #phases[:,w] = signal.detrend(phases[:,w])
        magnitudes[:,w] -= mean_mag
        #magnitudes[:,w] /= std_mag
        phases[:,w] -= mean_phase
        #phases[:,w] /= std_phase

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
    #pcs_magphase['std_mag'] = std_mag
    #pcs_magphase['std_phase'] = std_phase

    return pcs_magphase


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


def build_hf_component(reconstruction, fpeak, delta_f, fcenter=4096):

    # Re-position the spectrum
    reconstruction_FD = pycbc.types.FrequencySeries(reconstruction,
            delta_f=delta_f)

    specdata = np.zeros(len(reconstruction), dtype=complex)

    # Find where the peak should be
    peakidx = np.argmin(abs(reconstruction_FD.sample_frequencies-fpeak))

    # Start populating the output spectrum at this location
    false_freqs = reconstruction_FD.sample_frequencies.data - (fcenter-fpeak)

    specdata[:sum(false_freqs>=0)] = reconstruction[false_freqs>=0]

    return specdata

def stitch_waveform(low_pcs, nlowpcs, high_pcs, nhighpcs, fpeak,
        low_sigmas=1, high_sigmas=1, waveform_num=0, delta_f=1.0, fcenter=4096):
    """
    Shift the high component back to fpeak and sum the low and high components 

    Note that nlowpcs is a tuple whose first element is the number of magnitude
    components and second element is the number of phase components
    """

    high_component = reconstruct_signal_ampphase(high_pcs, nhighpcs[0],
            nhighpcs[1], waveform_num)
    low_component  = reconstruct_signal_ampphase(low_pcs, nlowpcs[0],
            nlowpcs[1], waveform_num)

    high_component_shifted = build_hf_component(high_component, fpeak, delta_f,
            fcenter)

    low_component = pycbc.types.FrequencySeries(low_component, delta_f=delta_f)
    low_component.data *= low_sigmas / pycbc.filter.sigma(low_component)

    high_component_shifted = pycbc.types.FrequencySeries(high_component_shifted, delta_f=delta_f)
    high_component_shifted.data *= high_sigmas / \
        pycbc.filter.sigma(high_component_shifted)

    synth = pycbc.types.FrequencySeries(low_component.data+high_component_shifted.data,
            delta_f=delta_f) 
    synth.data /= pycbc.filter.sigma(synth)

    return synth

def stitch_catalogue(low_pcs, nlowpcs, high_pcs, nhighpcs, fpeaks, low_sigmas,
        high_sigmas, delta_f=1.0, fhigh=8192, nFsamples=8193):
    """
    Synthesise the catalogue from the separate low and high PCs
    Note nlowpcs is a a tuple with the number of magnitude and phase pcs to use
    """

    synth_cat = np.zeros(shape=(nFsamples, len(fpeaks)), dtype=complex)

    for w in xrange(len(fpeaks)):
        synth_cat[:,w] = stitch_waveform(low_pcs, nlowpcs, high_pcs, nhighpcs,
                fpeaks[w], low_sigmas[w], high_sigmas[w], waveform_num=w)

    return synth_cat


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
    delta_t=1./16384
    nTsamples=16384
    nFsamples=nTsamples/2 + 1
    times=np.arange(0, delta_t*nTsamples, delta_t)

    fcenter = 4096.0  # center high-frequency peaks here
    #fcenter = 2000.0  # center high-frequency peaks here
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
    shifted_cat = np.zeros(shape=(nFsamples, len(waveform_names)), dtype=complex)
    original_cat = np.zeros(shape=(nFsamples, len(waveform_names)), dtype=complex)
    fpeaks = np.zeros(len(waveform_names))

    low_sigmas=np.zeros(len(waveform_names))
    high_sigmas=np.zeros(len(waveform_names))


    for w, name in enumerate(waveform_names):

        # Create waveform instance
        waveform = pmns_utils.Waveform(name)
        waveform.reproject_waveform()
        waveform.compute_characteristics()
        fpeaks[w] = np.copy(waveform.fpeak)

        # Window
        peakidx = np.argmax(abs(waveform.hplus.data))
        win=lal.CreateTukeyREAL8Window(len(waveform.hplus),0.25)
        waveform.hplus.data *= win.data.data

        # Zero-pad and peak-align
        rawdata = np.zeros(nTsamples)
        #rawdata[:len(waveform.hplus)] = taper(np.copy(waveform.hplus.data))
        rawdata[:len(waveform.hplus)] = np.copy(waveform.hplus.data)

#       alignidx = 0#.5*len(rawdata)
#       rightside=np.copy(waveform.hplus.data[peakidx:])
#       leftside=np.copy(waveform.hplus.data[:peakidx])
#       rawdata[alignidx:alignidx+len(rightside)]=np.copy(rightside)
#       #rawdata[alignidx-len(leftside):alignidx]=np.copy(leftside)
#       rawdata[len(rawdata)-len(leftside):len(rawdata)]=np.copy(leftside)

#       pl.figure()
#       pl.plot(rawdata)
#       pl.show()
#       sys.exit()
 
        # High-pass at 1 kHz
        #rawdata = signal.filtfilt(butter_b_high, butter_a_high, rawdata)

        del waveform

        original_signal = pycbc.types.TimeSeries(np.copy(rawdata),
                delta_t=delta_t) 
        del rawdata
        original_signal.data /= pycbc.filter.sigma(original_signal)
        original_signal_spectrum = original_signal.to_frequencyseries()

        # Frequency shift:
        tmpdata = np.copy(original_signal_spectrum.data)

        #fshift = fcenter / fpeaks[w]
        fshift = fshift_center / fpeaks[w]
        false_freqs_shift = original_signal_spectrum.sample_frequencies.data * fshift
        original_freqs = original_signal_spectrum.sample_frequencies.data

#       shiftedspec_real = np.interp(original_freqs, false_freqs,
#               np.real(tmpdata))
#       shiftedspec_imag = np.interp(original_freqs, false_freqs,
#               np.imag(tmpdata))
#
#       shift_fd = pycbc.types.FrequencySeries(shiftedspec_real +
#               1j*shiftedspec_imag, delta_f=original_signal_spectrum.delta_f) 

        shiftedspec_mag = np.interp(original_freqs, false_freqs_shift,
                abs(tmpdata))
        shiftedspec_phase = np.interp(original_freqs, false_freqs_shift,
                np.unwrap(np.angle(tmpdata)))

        shift_fd = pycbc.types.FrequencySeries(shiftedspec_mag *
                np.exp(1j*shiftedspec_phase),
                delta_f=original_signal_spectrum.delta_f) 

        shifted_cat[:,w] = shift_fd.data / pycbc.filter.sigma(shift_fd)

        # No shift
        original_cat[:,w] = np.copy(original_signal_spectrum.data)

        #
        # Extract low frequency data
        #

        # TD filter:
        lowdata = pycbc.types.TimeSeries(
                signal.filtfilt(butter_b_low, butter_a_low,
                    np.copy(original_signal.data)),
                delta_t=delta_t).to_frequencyseries() 
        low_sigmas[w] = pycbc.filter.sigma(lowdata, low_frequency_cutoff=1000)

        low_cat[:,w] = lowdata.data / low_sigmas[w]

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
        highdata_fd = pycbc.types.FrequencySeries(np.copy(highdata), 
                delta_f=original_signal_spectrum.delta_f)
        highdata_td = highdata_fd.to_timeseries()
        highdata_td.data = signal.filtfilt(butter_b_band, butter_a_band,
                highdata_td.data)

        # Unit-normalise
        high_sigmas[w] = pycbc.filter.sigma(highdata_td)

        # Populate catalogue
        high_cat[:,w] = highdata_td.to_frequencyseries().data / \
                high_sigmas[w]

        freqaxis=np.copy(original_signal_spectrum.sample_frequencies)

        del original_signal
        del original_signal_spectrum

    return (freqaxis, low_cat, high_cat, shifted_cat, original_cat, fpeaks,
            low_sigmas, high_sigmas)

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

def unshifted_rec_cat(pcs, npcs, fpeaks, freqaxis, fcenter):

    """
    Reconstruct the catalogue using npcs and the shifted waveforms
    """
    
    rec_cat = np.zeros(shape=(np.shape(pcs['magnitude_scores'])), dtype=complex)

    for w in xrange(len(fpeaks)):

        rec_cat[:, w] = unshift_waveform(pcs, [npcs[0], npcs[1]], fpeaks[w],
                freqaxis, waveform_num=w, fcenter=fcenter)

    return rec_cat

def unshift_waveform(shifted_pcs, npcs, fpeak, target_freqs, waveform_num=0,
        fcenter=3000., delta_f=1.0):
    """
    Reconstruct the shifted waveform and shift it back to the original peak
    frequency.  npcs is a tuple with the number of [mag, phase] PCs to use
    """

    reconstruction = reconstruct_signal_ampphase(shifted_pcs, npcs[0], npcs[1], waveform_num)


    #fshift = fcenter / fpeak

    fshift = fpeak / fcenter
    false_freqs = target_freqs * fshift

    shiftedspec_real = np.interp(target_freqs, false_freqs,
            np.real(reconstruction))

    shiftedspec_imag = np.interp(target_freqs, false_freqs,
            np.imag(reconstruction))

    shifted_reconstruction = shiftedspec_real + 1j*shiftedspec_imag

    return shifted_reconstruction


def unshifted_matches(catalogue, pcs, fpeaks, freqaxis, delta_f=1.0, flow=1000,
        fhigh=8192, fcenter=3000):
    """
    Compute the matches between the waveforms in the catalogue and the
    shifted spectrum reconstructions, where we use the training data as the test and
    consider full, high and low catalogues seperately
    """
    
    nwaveforms=np.shape(catalogue)[1]

    # Amplitude & Phase-maximised match
    matches = np.zeros(shape=(np.shape(catalogue)[1], np.shape(catalogue)[1]))
    # NOTE: columns = waveforms, rows=Npcs

    # Loop over the number of pcs to use
    for n in xrange(nwaveforms):

        rec_cat = unshifted_rec_cat(pcs, [n+1, n+1], fpeaks, freqaxis,
                fcenter=fcenter)

        # Loop over waveforms
        for w in xrange(nwaveforms):

            matches[w,n] = comp_match(rec_cat[:,w], catalogue[:,w],
                    delta_f=delta_f, flow=flow)

    return matches

def stitched_matches(catalogue, low_pcs, low_sigmas, high_pcs, high_sigmas,
        fpeaks, delta_f=1.0, flow=1000, fhigh=8192):
    """
    Compute the matches between the waveforms in the catalogue and the
    shifted spectrum reconstructions, where we use the training data as the test and
    consider full, high and low catalogues seperately
    """
    
    nwaveforms=np.shape(catalogue)[1]

    # Amplitude & Phase-maximised match
    matches = np.zeros(shape=(np.shape(catalogue)[1], np.shape(catalogue)[1]))
    # NOTE: columns = waveforms, rows=Npcs

    # Loop over the number of pcs to use
    for n in xrange(nwaveforms):

        rec_cat = stitch_catalogue(low_pcs, [n+1,n+1], high_pcs,
                [n+1, n+1], fpeaks, low_sigmas, high_sigmas)

        # Loop over waveforms
        for w in xrange(nwaveforms):

            matches[w,n] = comp_match(rec_cat[:,w], catalogue[:,w],
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


def unshifted_template(magBetas, phaseBetas, pcs, fpeak, target_freqs,
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

    shiftedspec_real = np.interp(target_freqs, false_freqs,
            np.real(reconstruction))

    shiftedspec_imag = np.interp(target_freqs, false_freqs,
            np.imag(reconstruction))

    shifted_reconstruction = shiftedspec_real + 1j*shiftedspec_imag

    return shifted_reconstruction

def unshift_vec(vector, target_freqs, fpeak, fcenter=1000.0):

    # Frequency shift
    fshift = fpeak / fcenter
    false_freqs = target_freqs * fshift

    unshiftedspec_real = np.interp(target_freqs, false_freqs, np.real(vector))
    unshiftedspec_imag = np.interp(target_freqs, false_freqs, np.imag(vector))

    unshifted_vector = unshiftedspec_real + 1j*unshiftedspec_imag

    return unshifted_vector

def shift_vec(vector, target_freqs, fpeak, fcenter=1000.0):

    # Frequency shift
    fshift = fcenter / fpeak
    false_freqs = target_freqs * fshift

    shiftedspec_real = np.interp(target_freqs, false_freqs, np.real(vector))
    shiftedspec_imag = np.interp(target_freqs, false_freqs, np.imag(vector))

    shifted_vector = shiftedspec_real + 1j*shiftedspec_imag

    return shifted_vector


def match(vary_args, *fixed_args):

    # variable params
    magbetas = vary_args[0:len(vary_args)/2]
    phasebetas = vary_args[len(vary_args)/2:-1]

    #if sum(abs(magbetas)>10) or sum(abs(phasebetas)>10): return -np.inf
    #else:

    # fixed params
    test_waveform, shifted_pcs, fpeak, target_freqs, fcenter, delta_f = fixed_args

    # Generate template
    tmplt = unshifted_template(magbetas, phasebetas, shifted_pcs, fpeak,
            target_freqs, fcenter, delta_f=1.0)

    try:
        match = comp_match(tmplt, test_waveform, delta_f=delta_f, flow=1000)

    except ZeroDivisionError:
        match = -np.inf

    return np.log(match)


import emcee
def emcee_maximise(ndim, fixed_params):

    # Inititalize sampler
    nwalkers=100
    nsamp=500

    # Starting points for walkers
    p0 = [np.random.rand(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, match, args=fixed_params)

    # Burn-in
    nburnin=10
    pos, prob, state = sampler.run_mcmc(p0, nburnin)
    sampler.reset()

    # Draw samples
    sampler.run_mcmc(pos, nsamp)

    return sampler

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
    npcs = len(waveform_names)

    #
    # Build Catalogues
    #
    print "building catalogues"
    fshift_center = 1000
    (freqaxis, low_cat, high_cat, shift_cat, original_cat, fpeaks, low_sigmas,
            high_sigmas) = build_catalogues(waveform_names, fshift_center)
    delta_f = np.diff(freqaxis)[0]

    # Convert to magnitude/phase
    full_mag, full_phase = complex_to_polar(original_cat)
    shift_mag, shift_phase = complex_to_polar(shift_cat)
    low_mag, low_phase = complex_to_polar(low_cat)
    high_mag, high_phase = complex_to_polar(high_cat)


    #
    # PCA
    #
    print "Performing PCA"
    high_pca = pca_magphase(high_cat, freqaxis, flow=10)
    low_pca  = pca_magphase(low_cat, freqaxis, flow=10)
    shift_pca = pca_magphase(shift_cat, freqaxis, flow=10)
    full_pca = pca_magphase(original_cat, freqaxis, flow=10)



    #
    # Compute idealised minimal matches
    #
    print "Computing all matches"
    full_matches_ideal = idealised_matches(original_cat, full_pca, delta_f=delta_f, flow=1000)
    shift_matches_ideal = idealised_matches(shift_cat, shift_pca,
            delta_f=delta_f, flow=10) # careful with this one's flow!
    low_matches_ideal = idealised_matches(low_cat, low_pca, delta_f=delta_f, flow=1000)
    high_matches_ideal = idealised_matches(high_cat, high_pca, delta_f=delta_f, flow=1000)

    unshift_matches_ideal = unshifted_matches(original_cat, shift_pca, fpeaks,
            freqaxis, fcenter=fshift_center)

    stitched_matches_ideal = stitched_matches(original_cat, low_pca, low_sigmas,
            high_pca, high_sigmas, fpeaks)

    # print some diagnostics
    nshift=0
    nstitch=0
    for i in xrange(npcs):
        for j in xrange(npcs):
            if unshift_matches_ideal[i,j] < 0.9: nshift+=1
            if stitched_matches_ideal[i,j] < 0.9: nstitch+=1
    print "Number of UNSHIFT < 0.9: ", nshift
    print "Number of STITCH < 0.9: ", nstitch

    #sys.exit()

    # ******** #
    # Plotting #
    # ******** #
    imageformats=['png','eps','pdf']


    #
    # Plot Catalogues
    #

    # Magnitude
    f, ax = pl.subplots(nrows=4,figsize=(7,15))
    ax[0].plot(freqaxis, full_mag, label='full spectrum')
    ax[0].set_xlim(1000, 5000)
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('|H(f)|')
    ax[0].minorticks_on()
    ax[0].set_title('Full Spectrum (magnitude)')

    ax[1].plot(freqaxis, shift_mag, label='shifted spectrum')
    ax[1].set_xlim(0, fshift_center+1000)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('|H(f)|')
    ax[1].minorticks_on()
    ax[1].set_title('Shifted Spectrum (magnitude)')

    ax[2].plot(freqaxis, low_mag, label='low-frequency components')
    ax[2].set_xlim(1000, 2500)
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('|H(f)|')
    ax[2].minorticks_on()
    ax[2].set_title('Low Frequencies (magnitude)')

    ax[3].plot(freqaxis, high_mag, label='high-frequency components')
    ax[3].set_xlim(3500, 4500)
    ax[3].set_xlabel('Frequency [Hz]')
    ax[3].set_ylabel('|H(f)|')
    ax[3].minorticks_on()
    ax[3].set_title('High Frequencies, aligned to 4096 Hz (magnitude)')

    
    f.tight_layout()
    for fileformat in imageformats:
        f.savefig('catalogue_magnitude_overlay.%s'%fileformat)

    # Phase
    f, ax = pl.subplots(nrows=4,figsize=(7,15))
    a=0
    ax[a].plot(freqaxis, full_phase, label='full spectrum')
    ax[a].set_xlim(1000, 5000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('Full Spectrum (phase)')

    a+=1
    ax[a].plot(freqaxis, shift_phase, label='shifted spectrum')
    ax[a].set_xlim(0, fshift_center+1000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('Shifted Spectrum (phase)')

    a+=1
    ax[a].plot(freqaxis, low_phase, label='low-frequency components')
    ax[a].set_xlim(1000, 2500)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('Low Frequencies (phase)')

    a+=1
    ax[a].plot(freqaxis, high_phase, label='high-frequency components')
    ax[a].set_xlim(3500, 4500)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('High Frequencies, aligned to 4096 Hz (phase)')

    f.tight_layout()
    for fileformat in imageformats:
        f.savefig('catalogue_phase_overlay.%s'%fileformat)

    #pl.show()

    #
    # Plot Magnitude PCs
    #
    f, ax = pl.subplots(nrows=4,figsize=(7,15))
    a=0
    ax[a].plot(freqaxis, abs(full_pca['magnitude_scores']), label='full spectrum')
    ax[a].set_xlim(1000, 5000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('|H(f)|')
    ax[a].minorticks_on()
    ax[a].set_title('Full Spectrum Principle Components (magnitude)')

    a+=1
    ax[a].plot(freqaxis, abs(shift_pca['magnitude_scores']), label='shifted spectrum')
    ax[a].set_xlim(0, fshift_center+1000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('|H(f)|')
    ax[a].minorticks_on()
    ax[a].set_title('Shifted Spectrum Principle Components (magnitude)')

    a+=1
    ax[a].plot(freqaxis, abs(low_pca['magnitude_scores']), label='low-frequency components')
    ax[a].set_xlim(1000, 2500)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('|H(f)|')
    ax[a].minorticks_on()
    ax[a].set_title('Low-frequency Principle Components (magnitude)')

    a+=1
    ax[a].plot(freqaxis, abs(high_pca['magnitude_scores']), label='high-frequency components')
    ax[a].set_xlim(3500, 4500)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('|H(f)|')
    ax[a].minorticks_on()
    ax[a].set_title('High-frequency Principle Components (magnitude)')
    f.tight_layout()
    for fileformat in imageformats:
        f.savefig('pcs_magnitude_overlay.%s'%fileformat)
    #pl.show()

    #
    # Plot Phase PCs
    #
    f, ax = pl.subplots(nrows=4,figsize=(7,15))
    a=0
    ax[a].plot(freqaxis, full_pca['phase_scores'], label='full spectrum')
    ax[a].set_xlim(1000, 5000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('Phase PCs')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('Full Spectrum Principle Components (phase)')

    a+=1
    ax[a].plot(freqaxis, shift_pca['phase_scores'], label='shifted spectrum')
    ax[a].set_xlim(0, fshift_center+1000)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('Phase PCs')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('Shifted Spectrum Principle Components (phase)')

    a+=1
    ax[a].plot(freqaxis, low_pca['phase_scores'], label='low-frequency components')
    ax[a].set_xlim(1000, 2500)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('Low-frequency Principle Components (phase)')

    a+=1
    ax[a].plot(freqaxis, high_pca['phase_scores'], label='high-frequency components')
    ax[a].set_xlim(3500, 4500)
    ax[a].set_xlabel('Frequency [Hz]')
    ax[a].set_ylabel('arg[H(f)]')
    ax[a].minorticks_on()
    ax[a].set_title('High-frequency Principle Components (phase)')
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
            label='shifted spectrum', color='g')
    ax.plot(npcs_axis, low_pca['magnitude_eigenergy'],
         label='low frequencies', color='r')
    ax.plot(npcs_axis, high_pca['magnitude_eigenergy'],
         label='high frequencies', color='c')

    ax.plot(npcs_axis, full_pca['phase_eigenergy'], linestyle='--', color='b')
    ax.plot(npcs_axis, shift_pca['phase_eigenergy'], linestyle='--', color='g')
    ax.plot(npcs_axis, low_pca['phase_eigenergy'], linestyle='--', color='r')
    ax.plot(npcs_axis, high_pca['phase_eigenergy'], linestyle='--', color='c')

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
    # Shifted waveforms
    fig, ax = image_matches(shift_matches_ideal, waveform_names,
            title="Matches For Full Aligned Spectrum PCA")

    for fileformat in imageformats:
        fig.savefig('shiftspec_ideal_matches.%s'%fileformat)


    # UN-shifted waveforms
    fig, ax = image_matches(unshift_matches_ideal, waveform_names,
            title="Matches For Shifted Full Spectrum PCA")

    for fileformat in imageformats:
        fig.savefig('unshiftspec_ideal_matches.%s'%fileformat)


    # Stitched waveforms
    fig, ax = image_matches(stitched_matches_ideal, waveform_names,
            title="Matches For Stitched Spectrum PCA")

    for fileformat in imageformats:
        fig.savefig('stitchedspec_ideal_matches.%s'%fileformat)

    # Original waveforms
    fig, ax = image_matches(full_matches_ideal, waveform_names,
            title="Matches For Full Spectrum PCA")

    for fileformat in imageformats:
        fig.savefig('fullspec_ideal_matches.%s'%fileformat)

    # Low-frequency parts
    fig, ax = image_matches(low_matches_ideal, waveform_names,
            title="Matches For Low Frequency PCA")

    for fileformat in imageformats:
        fig.savefig('lowfreq_ideal_matches.%s'%fileformat)

    # High-frequency parts
    fig, ax = image_matches(high_matches_ideal, waveform_names,
            title="Matches For High Frequency PCA")

    for fileformat in imageformats:
        fig.savefig('highfreq_ideal_matches.%s'%fileformat)


#
# End definitions
#
if __name__ == "__main__":
    main()






