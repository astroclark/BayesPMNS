#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <clark@physics.umass.edu>
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

from scipy import signal

import pmns_utils

import pycbc.noise
import pycbc.types
import pycbc.filter

import lal
import lalsimulation as lalsim

# ---------
# FUNC DEFS

# ----------
# CLASS DEFS
class DetData:
    """
    The data stream (signal+noise) for a given detector & event
    """
    #def __init__(self, det_site="H1", noise_curve='aLIGO', epoch=1087670331.0,
    def __init__(self, det_site="H1", noise_curve='aLIGO', epoch=0.0,
            duration=10.0, f_low=10.0, delta_t=1./16384, waveform=None,
            ext_params=None, seed=0, signal_only=False, taper=False,
            optimal_injection=False, target_snr=None):

        # dictionary of detector locations
        det_sites = {"H1": lal.CachedDetectors[lal.LHO_4K_DETECTOR], 
                "L1": lal.CachedDetectors[lal.LLO_4K_DETECTOR], 
                "V1": lal.CachedDetectors[lal.VIRGO_DETECTOR]}

        if det_site not in ["H1", "L1", "V1"]:
            print >> sys.stderr, "error, observatory obs_site %s not "\
                    "recognised"%det_site

        # Preliminaries
        self.seed = seed
        self.taper = taper
        self.optimal_injection=optimal_injection
        self.noise_curve = noise_curve
        self.det_site = det_sites[det_site]
        self.epoch = lal.LIGOTimeGPS(epoch)
        self.duration = duration
        self.f_low = f_low
        self.delta_t = delta_t
        #self.delta_f = 1.0 / self.duration
        self.delta_f = 1.0 / (int(self.duration/self.delta_t)*self.delta_t)
        self.tlen = int(np.ceil(self.duration / self.delta_t))
        self.flen = int(np.ceil(self.tlen/2 + 1))
        self.target_snr = target_snr

        # --- Make noise
        self.make_noise(signal_only)

        # --- Make signal
        if waveform is not None:
            self.ext_params = ext_params
            self.make_signal(waveform)
            self.waveform_name = waveform.waveform_name
        else:
            self.waveform_name = None

        # --- Add signal to noise
        if waveform is not None:
            self.add_signal_to_noise()

        # --- Make frequency domain data
        self.make_fdomain()

    def make_fdomain(self):

        self.fd_signal = self.td_signal.to_frequencyseries()
        self.fd_noise = self.td_noise.to_frequencyseries()
        self.fd_response = self.td_response.to_frequencyseries()

        # don't forget the psd
        self.delta_f = self.fd_response.delta_f
        self.assign_noise_curve()

    # ---------------
    # DetData Methods
    # ---------------

    def make_signal(self, waveform):
        """
        Generate the signal seeen in this detector
        """

        #print >> sys.stdout, "generating signal..."

        # --- Set up timing

        # index of the absolute maximum peak
        #idx = np.concatenate(np.argwhere(abs(waveform.hplus.data.data)>0))[0]
        idx = np.argmax(abs(waveform.hplus.data))

        # Epoch = GPS start of time series.  Want the peak time of the waveform
        # to be aligned to the geocenter, so set the epoch to the geocentric
        # peak time minus the time to the waveform peak.  In other words:
        # (waveform epoch) = (geocentric peak time) - (# of seconds to peak)

        hplus_epoch  = self.ext_params.geocent_peak_time - idx*waveform.hplus.delta_t
        hcross_epoch = self.ext_params.geocent_peak_time - idx*waveform.hcross.delta_t

        # XXX: create regular lal timeseries objects for this bit (may replace
        # with pycbc injection routines later)

        hplus = lal.CreateREAL8TimeSeries('hplus', hplus_epoch, 0,
                waveform.hplus.delta_t, lal.StrainUnit,
                int(waveform.hplus.duration / waveform.hplus.delta_t))
        hplus.data.data = np.array(waveform.hplus.data)

        hcross = lal.CreateREAL8TimeSeries('hcross', hcross_epoch, 0,
                waveform.hcross.delta_t, lal.StrainUnit,
                int(waveform.hcross.duration / waveform.hcross.delta_t))
        hcross.data.data = np.array(waveform.hcross.data)


        if self.taper is True:

            print >> sys.stderr, "Warning: tapering out inspiral (not a realistic strategy)"
            delay = 0.0e-3
            idx = np.argmax(hplus.data.data) + \
                    np.ceil(delay/self.delta_t)
            hplus.data.data[0:idx]=0.0
            hcross.data.data[0:idx]=0.0
            lalsim.SimInspiralREAL8WaveTaper(hplus.data,
                    lalsim.SIM_INSPIRAL_TAPER_START)
            lalsim.SimInspiralREAL8WaveTaper(hcross.data,
                    lalsim.SIM_INSPIRAL_TAPER_START)


        # Scale for distance (waveforms extracted at 20 Mpc)
        hplus.data.data *= 20.0 / self.ext_params.distance
        hcross.data.data *= 20.0 / self.ext_params.distance

        tmp = lalsim.SimDetectorStrainREAL8TimeSeries(hplus, hcross,
                self.ext_params.ra, self.ext_params.dec,
                self.ext_params.polarization, self.det_site) 

        # Pad the end so we have the same length signal and noise (useful for
        # snr and psds)
        sigdata = np.zeros(len(self.td_noise))
        sigdata[:len(tmp.data.data)] = np.copy(tmp.data.data)

        # Project waveform onto these extrinsic parameters
        self.td_signal = \
                pycbc.types.timeseries.TimeSeries(initial_array=sigdata,
                        delta_t=tmp.deltaT, epoch=tmp.epoch)

        del tmp

        # Remove extraneous data
        #self.td_signal = self.td_signal.trim_zeros()


    def make_noise(self, signal_only):
        """
        Generate Gaussian noise coloured to psd for det
        """

        #print >> sys.stdout, "generating noise..."

        if signal_only:

            # the noise is just a time series of zeros
            
            self.td_noise = pycbc.types.timeseries.TimeSeries(
                    initial_array=np.zeros(self.duration/self.delta_t),
                        delta_t=self.delta_t, epoch=self.epoch)

        else:
            # Generate noise 
            self.assign_noise_curve()

            # Generate time-domain noise
            # XXX: minimum duration seems to be 1 second.  I'll hack around this by
            # reducing the 1 second to the desired duration
            tmplen=max(self.duration,1.0)/self.delta_t
            self.td_noise = pycbc.noise.noise_from_psd(int(tmplen), self.delta_t,
                    self.psd, seed=self.seed)

            self.td_noise = \
                    pycbc.types.TimeSeries(self.td_noise.data[:self.duration/self.delta_t],
                            delta_t=self.delta_t)

            # XXX not sure if this is a good idea...
            self.td_noise.start_time = float(self.epoch)

            self.fd_noise = self.td_noise.to_frequencyseries()

    def assign_noise_curve(self):

        ligo3_curves=['base', 'highNN', 'highSPOT', 'highST',
                'highSei', 'highloss', 'highmass', 'highpow', 'highsqz',
                'lowNN', 'lowSPOT', 'lowST', 'lowSei']

        if self.noise_curve=='aLIGO': 
            from pycbc.psd import aLIGOZeroDetHighPower
            self.psd = aLIGOZeroDetHighPower(self.flen, self.delta_f, self.f_low) 
        elif self.noise_curve=='adVirgo':
            from pycbc.psd import AdvVirgo
            self.psd = AdvVirgo(self.flen, self.delta_f, self.f_low) 
        elif self.noise_curve=='Green' or self.noise_curve=='Red':

            # Load from ascii
            pmnspy_path=os.getenv('PMNSPY_PREFIX')

            psd_path=pmnspy_path+'/ligo3/PSD/%sPSD.txt'%self.noise_curve

            psd_data=np.loadtxt(psd_path)

            target_freqs = np.arange(0.0, self.flen*self.delta_f,
                    self.delta_f)
            target_psd = np.zeros(len(target_freqs))

            # Interpolate existing psd data to target frequencies
            existing = \
                    np.concatenate(np.argwhere(
                        (target_freqs<=psd_data[-1,0]) *
                        (target_freqs>=psd_data[0,0])
                        ))

            target_psd[existing] = \
                    np.interp(target_freqs[existing], psd_data[:,0],
                            psd_data[:,1])

            # Extrapolate to higher frequencies assuming f^2 for QN
            fit_idx = np.concatenate(np.argwhere((psd_data[:,0]>2000)*\
                    (psd_data[:,0]<=psd_data[-1,0])))

            p = np.polyfit(x=psd_data[fit_idx,0], \
                    y=psd_data[fit_idx,1], deg=2)

            target_psd[existing[-1]+1:] = \
                    p[0]*target_freqs[existing[-1]+1:]**2 +  \
                    p[1]*target_freqs[existing[-1]+1:] + \
                    p[2]

            # After all that, reset everything below f_low to zero (this saves
            # significant time in noise generation if we only care about high
            # frequencies)
            target_psd[target_freqs<self.f_low] = 0.0

            # Create psd as standard frequency series object
            self.psd = pycbc.types.FrequencySeries(
                    initial_array=target_psd, delta_f=np.diff(target_freqs)[0])

        elif self.noise_curve in ligo3_curves:

            # Load from ascii
            pmnspy_path=os.getenv('PMNSPY_PREFIX')

            psd_path=pmnspy_path+'/ligo3/PSD/BlueBird_%s-PSD_20140904.txt'%self.noise_curve

            psd_data=np.loadtxt(psd_path)

            target_freqs = np.arange(0.0, self.flen*self.delta_f,
                    self.delta_f)
            target_psd = np.zeros(len(target_freqs))

            # Interpolate existing psd data to target frequencies
            existing = \
                    np.concatenate(np.argwhere(
                        (target_freqs<=psd_data[-1,0]) *
                        (target_freqs>=psd_data[0,0])
                        ))

            target_psd[existing] = \
                    np.interp(target_freqs[existing], psd_data[:,0],
                            psd_data[:,1])

            # Extrapolate to higher frequencies assuming f^2 for QN
            fit_idx = np.concatenate(np.argwhere((psd_data[:,0]>2000)*\
                    (psd_data[:,0]<=psd_data[-1,0])))

            p = np.polyfit(x=psd_data[fit_idx,0], \
                    y=psd_data[fit_idx,1], deg=2)

            target_psd[existing[-1]+1:] = \
                    p[0]*target_freqs[existing[-1]+1:]**2 +  \
                    p[1]*target_freqs[existing[-1]+1:] + \
                    p[2]

            # After all that, reset everything below f_low to zero (this saves
            # significant time in noise generation if we only care about high
            # frequencies)
            target_psd[target_freqs<self.f_low] = 0.0

            # Create psd as standard frequency series object
            self.psd = pycbc.types.FrequencySeries(
                    initial_array=target_psd, delta_f=np.diff(target_freqs)[0])

        else:
            print >> sys.stderr, "error: noise curve (%s) not"\
                " supported"%self.noise_curve
            sys.exit(-1)


    def add_signal_to_noise(self):
        """
        Sum the noise and the signal to get the 'measured' strain in the
        detector
        """

        # noise
        noise = lal.CreateREAL8TimeSeries('blah', self.epoch, 0,
                self.td_noise.delta_t, lal.StrainUnit, 
                int(self.td_noise.duration / self.td_noise.delta_t))
        noise.data.data = self.td_noise.data

        # signal
        signal = lal.CreateREAL8TimeSeries('blah',
                self.ext_params.geocent_peak_time, 0, self.td_signal.delta_t,
                lal.StrainUnit, int(self.td_signal.duration /
                    self.td_signal.delta_t))
        signal.data.data = self.td_signal.data

        win = lal.CreateTukeyREAL8Window(len(signal.data.data),0.1)
        signal.data.data[len(signal.data.data):] = 1.0
        signal.data.data *= win.data.data

        # --- Scale to a target snr
        print '---'
        if self.target_snr is not None:

            tmp_sig = pycbc.types.TimeSeries(signal.data.data,
                    delta_t=self.td_signal.delta_t)

            current_snr = pycbc.filter.sigma(tmp_sig, psd=self.psd,
                    low_frequency_cutoff=self.f_low,
                    high_frequency_cutoff=0.5/self.delta_t)

            signal.data.data *= self.target_snr / current_snr
        # ----

        # sum
        noise_plus_signal = lal.AddREAL8TimeSeries(noise, signal)

        self.td_response = \
                pycbc.types.timeseries.TimeSeries(\
                initial_array=np.copy(noise_plus_signal.data.data),
                        delta_t=noise_plus_signal.deltaT,
                        epoch=noise_plus_signal.epoch)

        # Finally, zero-pad the signal vector to have the same length as the actual data
        # vector
        no_noise = lal.CreateREAL8TimeSeries('blah', self.epoch, 0,
                self.td_noise.delta_t, lal.StrainUnit, 
                int(self.td_noise.duration / self.td_noise.delta_t))

        no_noise.data.data = np.zeros(\
                int(self.td_noise.duration / self.td_noise.delta_t))

        signal = lal.AddREAL8TimeSeries(no_noise, signal)

        self.td_signal = \
                pycbc.types.timeseries.TimeSeries(initial_array=np.copy(signal.data.data),
                        delta_t=signal.deltaT, epoch=noise_plus_signal.epoch)

        del noise, signal, noise_plus_signal
        
 
class ExtParams:
    """
    A structure to store extrinsic parameters of a signal
    """
    
    def __init__(self, distance, ra, dec, polarization, inclination, phase,
            geocent_peak_time):

        self.distance = distance
        self.ra  = ra
        self.dec = dec
        self.polarization = polarization
        self.inclination = inclination
        self.phase = phase
        self.geocent_peak_time = lal.LIGOTimeGPS(geocent_peak_time)


def main():
    """
    Demonstrate construction of multiple det data streams with a signal
    injection
    """

    #
    # Generate waveform
    #

    print 'generating waveoform...'
    waveform = pmns_utils.Waveform('shen_135135_lessvisc')

    # Pick some extrinsic parameters
    ext_params = ExtParams(distance=1, ra=0.0, dec=0.0, polarization=0.0,
            inclination=0.0, phase=0.0, geocent_peak_time=0.0+5.0)

    # Construct the time series for these params
    waveform.make_wf_timeseries(theta=ext_params.inclination,
            phi=ext_params.phase)

    #
    # Generate IFO data
    #
    det1_data = DetData(waveform=waveform, ext_params=ext_params)

    from scipy import signal
    import pylab as pl

    pl.figure()
    pl.plot(det1_data.td_response.sample_times,det1_data.td_response.data)
    pl.plot(det1_data.td_signal.sample_times,det1_data.td_signal.data)

    pl.figure()
    f,p = signal.welch(det1_data.td_response.data, fs=1./det1_data.delta_t,
            nperseg=512)
    pl.loglog(f,np.sqrt(p))

    f,p = signal.welch(det1_data.td_signal.data, fs=1./det1_data.delta_t,
            nperseg=512)
    pl.loglog(f,np.sqrt(p))
    pl.ylim(1e-25,1e-21)
    pl.show()



if __name__ == "__main__":

        main()

