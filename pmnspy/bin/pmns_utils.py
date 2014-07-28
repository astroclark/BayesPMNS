#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2013-2014 James Clark <clark@physics.umass.edu>
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

import lal
import lalsimulation as lalsim

from scipy import optimize,stats,signal

__author__ = "James Clark <james.clark@ligo.org>"

class Waveform:
    """
    A class to store and compute useful attributes of each waveform
    """

    def __init__(self, waveform_name, data_path=None):

        # waveform labels and directory setup
        self.waveform_name  = waveform_name
        self.get_data_path(data_path)
        self.set_tex_label(waveform_name)
        self.set_R16(waveform_name)
        self.allowed_radii()

    def compute_characteristics(self,flow=1500,fupp=4096):
        """
        Computes SNRs, peak and characteristic frequencies and hrss.  Assumes
        ZDHP aLIGO.
        """
        # FIXME: noise curve to be optional

        # check for existence of waveforms
        if not hasattr(self, 'hplus'):
            self.make_wf_timeseries()

        # Compute xoptimal_snr measures for an optimally oriented source 
        self.snr_plus, self.hrss_plus, self.fchar, self.fpeak, self.Egw = \
                optimal_snr(self.hplus)
        self.snr_cross, self.hrss_cross, _, _ , _ = optimal_snr(self.hcross)

        # Correct hrss for sky average
        #sYlm2p2 = lal.SpinWeightedSphericalHarmonic(0.0, 0.0, -2, 2, 2)

        #self.hrss_plus /= np.real(sYlm2p2)
        #self.hrss_cross /= np.real(sYlm2p2)
        self.hrss = np.sqrt(self.hrss_plus**2 + self.hrss_cross**2)

        if not hasattr(self, 'PSD_plus'):
            self.make_wf_freqseries()

        # no more attributes if prompt collapse
        if self.waveform_name != 'sfho_1616': 


            # Get FWHM of peak
            idx_inband = (self.freq_axis>flow) * (self.freq_axis<fupp) 
            self.fwhm  = find_fwhm(self.freq_axis[idx_inband],
                    self.PSD_plus[idx_inband])

            # Get fitted values
            self.amp_fit, self.fpeak_fit, self.fwhm_fit = fit_gaussian_peak(self)

            self.gauss_fit = gauss_curve(self.freq_axis, self.amp_fit,
                    self.fpeak_fit, 0.5*self.fwhm_fit)

            # compute xoptimal_snr measures at 2-sigma interval around peak
            flow = self.fpeak-self.fwhm
            fupp = self.fpeak+self.fwhm

            self.snr_plus_peak, self.hrss_plus_peak, self.fchar_peak, \
                    self.fpeak_peak, self.Egw_peak = \
                    optimal_snr(self.hplus,freqmin=flow,freqmax=fupp)

            self.snr_cross_peak, self.hrss_cross_peak, _, _ , _ = \
                    optimal_snr(self.hcross,freqmin=flow,freqmax=fupp)

            #self.hrss_plus_peak /= np.real(sYlm2p2)
            #self.hrss_cross_peak /= np.real(sYlm2p2)
            self.hrss_peak = np.sqrt(self.hrss_plus_peak**2 +
                    self.hrss_cross_peak**2)

            # expected frequency bias
            self.delta_fpeak = abs(self.fpeak - self.fpeak_fit)

            # Expected R16 measurement
            self.set_R16_fpeak()


    def make_wf_timeseries(self,theta=0,phi=0):
        """
        makes the waveform (from just the 2,2 modes for now)
        """
        #FIXME: data length, sample freq should be variable

        # retrieve data
        mode2p2_data, mode2n2_data = get_waveform_data(self.waveform_path,
                self.waveform_name)

        # make waveforms
        self.hplus, self.hcross, self.time_axis = \
                make_signal(mode2p2_data, mode2n2_data, theta=theta, phi=phi)

    def make_wf_freqseries(self):
        """
        make complex Fourier spectra and PSDs for polarisations
        """

        # check for existence of waveforms
        if not hasattr(self, 'hplus'): self.make_wf_timeseries()

        # make freq series
        self.Hplus, self.freq_axis = lal_fft(self.hplus)
        self.Hcross, _             = lal_fft(self.hcross)

        # assign PSD
        self.PSD_plus  = 2*abs(self.Hplus.data.data)**2
        self.PSD_cross = 2*abs(self.Hcross.data.data)**2


    def get_data_path(self, data_path):
        """
        Attempts to retrieve the location of the waveform data from environment
        variable NINJA_ASCII
        """
        if data_path is None:
            # try to get it from the environment
            try:
                self.waveform_path = os.environ['NINJA_ASCII']
            except KeyError:
                print >> sys.stderr, "NINJA_ASCII environment variable not" \
                        " set, please specify waveform location in Waveform()" \
                        " instantiation"
        else:
            self.waveform_path = data_path

    def set_tex_label(self,waveform_name):
        """
        Dictionary of labels to use for each waveform name
        """
        labels={
                'apr_135135_lessvisc'          :r'$\textrm{APR}$',
                'shen_135135_lessvisc'         :r'$\textrm{Shen}$',
                'dd2_135135_lessvisc'          :r'$\textrm{DD2}$',
                'dd2_165165_lessvisc'          :r'$\textrm{DD2}_{3.3M_{\odot}}$',
                'nl3_135135_lessvisc'          :r'$\textrm{NL3}$',
                'nl3_1919_lessvisc'            :r'$\textrm{NL3}_{3.8M_{\odot}}$',
                'tm1_135135_lessvisc'          :r'$\textrm{TM1}$',
                'tma_135135_lessvisc'          :r'$\textrm{TMa}$',
                'sfhx_135135_lessvisc'         :r'$\textrm{SFHx}$',
                'sfho_135135_lessvisc'         :r'$\textrm{SFHo}$',
                'sfho_1616'                    :r'$\textrm{SFHo}_{3.2M_{\odot}}$',
                'hybrid_apr_135135_lessvisc_0_0p5'          :r'$\textrm{hlAPR}^{\dagger}$',
                'hybrid_apr_135135_lessvisc_0_0p5_0p018'    :r'$\textrm{hsAPR}^{\dagger}$',
                'hybrid_apr_135135_lessvisc_0p05_0p5'       :r'$\textrm{hlAPR}^{*}$',
                'hybrid_apr_135135_lessvisc_0p05_0p5_0p018' :r'$\textrm{hsAPR}^{*}$',
                'hybrid_dd2_135135_lessvisc_0_0p5'          :r'$\textrm{hlDD2}^{\dagger}$',
                'hybrid_dd2_135135_lessvisc_0_0p5_0p028'    :r'$\textrm{hsDD2}^{\dagger}$',
                'hybrid_dd2_135135_lessvisc_0p05_0p5'       :r'$\textrm{hlDD2}^{*}$',
                'hybrid_dd2_135135_lessvisc_0p05_0p5_0p028' :r'$\textrm{hsDD2}^{*}$',
                'SG_highfreq' :  r'$\textrm{SG_{3.5}^{250}}$', 
                'SG_lowfreq'  :  r'$\textrm{SG_{2.1}^{250}}$'
                }
                #'hybrid_dd2_135135_0df_0p5'    :r'$\textrm{DD2}^*_0$',
                #'hybrid_dd2_135135_0p05df_0p5' :r'$\textrm{DD2}^*_{0.05}$',
                #'hybrid_dd2_135135_lessvisc_0p05df_0p5_0p4':r'$_{0.4}\textrm{DD2}^*_{0.05}$',
                #'hybrid_dd2_135135_0p1df_0p5'  :r'$\textrm{DD2}^*_{0.1}$'

        self.tex_label=labels[waveform_name]

    def set_R16(self,waveform_name):
        """
        Dictionary of R16 values for each waveform
        """
        R16={
                'apr_135135'                   : 11.25,
                'apr_135135_lessvisc'          : 11.25,
                'shen_135135'                  : 14.53,
                'shen_135135_lessvisc'         : 14.53,
                'dd2_135135'                   : 13.26,
                'dd2_135135_lessvisc'          : 13.26,
                'dd2_165165'                   : 13.26,
                'dd2_165165_lessvisc'          : 13.26,
                'nl3_135135'                   : 14.8,
                'nl3_135135_lessvisc'          : 14.8,
                'nl3_1919'                     : 11.25,
                'nl3_1919_lessvisc'            : 11.25,
                'tm1_135135_lessvisc'          : 14.36,
                'tma_135135_lessvisc'          : 13.73,
                'sfhx_135135_lessvisc'         : 11.98,
                'sfho_135135_lessvisc'         : 11.76,
                'sfho_1616'                    : 11.76,
                'hybrid_apr_135135_lessvisc_0_0p2'       :11.25,
                'hybrid_apr_135135_lessvisc_0_0p5'       :11.25,
                'hybrid_apr_135135_lessvisc_0_0p5_0p018' :11.25,
                'hybrid_apr_135135_lessvisc_0p05_0p2'    :11.25,
                'hybrid_apr_135135_lessvisc_0p05_0p5'    :11.25,
                'hybrid_apr_135135_lessvisc_0p05_0p5_0p018' :11.25,
                'hybrid_apr_135135_lessvisc_0p1_0p2'     :11.25,
                'hybrid_apr_135135_lessvisc_0p1_0p5'     :11.25,
                'hybrid_dd2_135135_0_0p2'                :13.26,
                'hybrid_dd2_135135_0_0p5'                :13.26,
                'hybrid_dd2_135135_0p05_0p2'             :13.26,
                'hybrid_dd2_135135_0p05_0p5'             :13.26,
                'hybrid_dd2_135135_0p1_0p2'              :13.26,
                'hybrid_dd2_135135_0p1_0p5'              :13.26,
                'hybrid_dd2_135135_lessvisc_0_0p2'       :13.26,
                'hybrid_dd2_135135_lessvisc_0_0p5'       :13.26,
                'hybrid_dd2_135135_lessvisc_0_0p5_0p028' :13.26,
                'hybrid_dd2_135135_lessvisc_0p05_0p2'    :13.26,
                'hybrid_dd2_135135_lessvisc_0p05_0p5_0p4':13.26,
                'hybrid_dd2_135135_lessvisc_0p05_0p5'    :13.26,
                'hybrid_dd2_135135_lessvisc_0p05_0p5_0p028' :13.26,
                'hybrid_dd2_135135_lessvisc_0p1_0p2'     :13.26,
                'hybrid_dd2_135135_lessvisc_0p1_0p5'     :13.26,
                'SG_highfreq' :np.nan,
                'SG_lowfreq'  :np.nan
                }

        self.R16=R16[waveform_name]
 
    def set_R16_fpeak(self):
        """
        Set the expected value for R16 measured from fpeak
        """

        # Assign expected radius measurement
        if self.waveform_name in self.allowed_radii:
            if not hasattr(self,'fpeak'): self.compute_characteristics()
            self.R16_fpeak = fpeak_to_R16(self.fpeak)[0]
            # Expected error (magnitude) in meters
            self.deltaR16_fpeak = 1e3*abs(self.R16_fpeak - self.R16)
        else:
            self.R16_fpeak = np.nan
            self.deltaR16_fpeak = np.nan

    def allowed_radii(self):

        self.allowed_radii=[
                'apr_135135',
                'apr_135135_lessvisc',
                'dd2_135135',
                'dd2_135135_lessvisc',
                'nl3_135135',
                'nl3_135135_lessvisc',
                'sfho_135135_lessvisc',
                'sfhx_135135_lessvisc',
                'shen_135135',
                'shen_135135_lessvisc',
                'tm1_135135_lessvisc',
                'tma_135135_lessvisc',
                'hybrid_apr_135135_lessvisc_0_0p2',
                'hybrid_apr_135135_lessvisc_0_0p5',
                'hybrid_apr_135135_lessvisc_0_0p5_0p018',
                'hybrid_apr_135135_lessvisc_0p05_0p2',
                'hybrid_apr_135135_lessvisc_0p05_0p5',
                'hybrid_apr_135135_lessvisc_0p05_0p5_0p018',
                'hybrid_apr_135135_lessvisc_0p1_0p2',
                'hybrid_apr_135135_lessvisc_0p1_0p5',
                'hybrid_dd2_135135_0_0p2'           ,
                'hybrid_dd2_135135_0_0p5'           ,
                'hybrid_dd2_135135_0p05_0p2'        ,
                'hybrid_dd2_135135_0p05_0p5'        ,
                'hybrid_dd2_135135_0p1_0p2'         ,
                'hybrid_dd2_135135_0p1_0p5'         ,
                'hybrid_dd2_135135_lessvisc_0_0p2'  ,
                'hybrid_dd2_135135_lessvisc_0_0p5'  ,
                'hybrid_dd2_135135_lessvisc_0_0p5_0p028' ,
                'hybrid_dd2_135135_lessvisc_0p05_0p2'    ,
                'hybrid_dd2_135135_lessvisc_0p05_0p5_0p4',
                'hybrid_dd2_135135_lessvisc_0p05_0p5'    ,
                'hybrid_dd2_135135_lessvisc_0p05_0p5_0p028',
                'hybrid_dd2_135135_lessvisc_0p1_0p2'     ,
                'hybrid_dd2_135135_lessvisc_0p1_0p5'
                ]



#
# general use functions
#

def get_waveform_data(waveform_path, waveform_name):
    """
    Retrieve the l=m=2 spherical harmonics from the ascii file stored in the
    ninja data directory
    """
    mode2p2_data = np.loadtxt("{waveform_path}/"\
            "{waveform_name}_l2m2.asc".format(waveform_path=waveform_path,
                waveform_name=waveform_name))

    mode2n2_data = np.loadtxt("{waveform_path}/"\
            "{waveform_name}_l2m-2.asc".format(waveform_path=waveform_path,
                waveform_name=waveform_name))

    return mode2p2_data, mode2n2_data

def make_signal(mode2p2,mode2n2,skyav=False,theta=0.0,phi=0.0):
    """
    Return the plus and cross polarisations for a source at the location defined
    by the current entry in the sim_inspiral table.

    Note that the current usage of this function is to create optimally oriented
    signals but it's easy to generalise in this form.
    """

    # Scaling for physical units (out of geometric units)
    extraction_distance_MPC = 20.0
    extraction_distance_SI = lal.PC_SI * 1e6 * extraction_distance_MPC
    geom_fac = lal.MRSUN_SI / extraction_distance_SI

    # Get plus,cross from mode arrays (keeps things readable)
    mode2p2_plus=geom_fac*mode2p2[:,1]
    mode2p2_cross=geom_fac*mode2p2[:,2]

    mode2n2_plus=geom_fac*mode2n2[:,1]
    mode2n2_cross=geom_fac*mode2n2[:,2]

    # compute spherical harmonics

    # --- In this situation, I want the sky-averaged sYlmn, which are 1 by
    # makeion
    if skyav:
        sYlm2p2 = 1.0
        sYlm2n2 = 1.0
    else:
        # Assume we're talking about a face on system
        sYlm2p2 = lal.SpinWeightedSphericalHarmonic(theta, phi, -2, 2, 2)
        sYlm2n2 = lal.SpinWeightedSphericalHarmonic(theta, phi, -2, 2, -2)

    # Orient Waveforms
    hplus2p2  = mode2p2_plus*np.real(sYlm2p2) + mode2p2_cross*np.imag(sYlm2p2)
    hcross2p2 = mode2p2_cross*np.real(sYlm2p2) - mode2p2_plus*np.imag(sYlm2p2)

    hplus2n2  = mode2n2_plus*np.real(sYlm2n2) + mode2n2_cross*np.imag(sYlm2n2)
    hcross2n2 = mode2n2_cross*np.real(sYlm2n2) - mode2n2_plus*np.imag(sYlm2n2)

    # Sum modes and add to the sky

    hplus=lal.CreateREAL8TimeSeries('strain', lal.LIGOTimeGPS(), 0.0,
            1./16384, lal.StrainUnit, len(hplus2p2))
            #1./16384, lal.StrainUnit, 5*16384)
    hcross=lal.CreateREAL8TimeSeries('strain', lal.LIGOTimeGPS(), 0.0,
            1./16384, lal.StrainUnit, len(hcross2p2))
            #1./16384, lal.StrainUnit, 5*16384)

    hplus.data.data  = hplus2p2+hplus2n2
    hcross.data.data = hcross2p2+hcross2n2

    # Correction for quadrupole underestimate
    hplus.data.data  *= 1.4
    hcross.data.data *= 1.4

    time=np.arange(0,len(hplus2p2)/16384.,1./16384)
    # Zero the time axis at the peak of the plus polarisation
    tPeak=time[abs(hplus.data.data)==max(abs(hplus.data.data))]
    time-=tPeak

    return hplus,hcross,time

def lal_fft(timeseries,seglen=5,fs=16384):

    N = int(np.floor(seglen * fs))

    if N!=len(timeseries.data.data): lal.ResizeREAL8TimeSeries(timeseries, 0, N)

    window=lal.CreateRectangularREAL8Window(timeseries.data.length)
    timeseries.data.data*=window.data.data

    freqseries = lal.CreateCOMPLEX16FrequencySeries("h(f)", timeseries.epoch,
            timeseries.f0, 1./seglen, lal.HertzUnit, int(N/2 + 1)) 

    fftplan = lal.CreateForwardREAL8FFTPlan(N, 0)
    lal.REAL8TimeFreqFFT(freqseries, timeseries, fftplan)

    norm=np.sqrt(window.sumofsquares / window.data.length)
    freqseries.data.data/=norm

    freqs=np.linspace(0,fs/2.,int(N/2 + 1))

    return freqseries,freqs

#   def lal_fft(timeseries,seglen=5,fs=16384):
#
#       # reduce to significant parts of signal
#       idx = np.argwhere(abs(timeseries.data.data)>0)[0]
#       N=len(timeseries.data.data[idx:])
#       tmp = lal.CreateREAL8TimeSeries('strain', lal.LIGOTimeGPS(), 0.0,
#               1./16384, lal.StrainUnit, N)
#       tmp.data.data=timeseries.data.data[idx:]
#
#       #window=lal.CreateTukeyREAL8Window(N,0.1)
#       #window.data.data[0:N/2]=np.ones(N/2)
#       #tmp.data.data*=window.data.data
#
#       window=lal.CreateRectangularREAL8Window(N)
#       tmp.data.data*=window.data.data
#
#       freqseries = lal.CreateCOMPLEX16FrequencySeries("h(f)", tmp.epoch,
#               tmp.f0, 1./seglen, lal.HertzUnit, int(N/2 + 1)) 
#
#       fftplan = lal.CreateForwardREAL8FFTPlan(N, 0)
#       lal.REAL8TimeFreqFFT(freqseries, tmp, fftplan)
#
#       norm=np.sqrt(window.sumofsquares / window.data.length)
#       freqseries.data.data/=norm
#
#       freqs=np.linspace(0,fs/2.,int(N/2 + 1))
#
#       return freqseries,freqs

def optimal_snr(tSeries,freqmin=1500,freqmax=4096):
    """
    Compute optimal snr, characteristic frequency and peak time (see
    xoptimalsnr.m)
    """

    # Compute fft
    fSpec,freq=lal_fft(tSeries)

    p_FD = 2*abs(fSpec.data.data)**2

    p_FD = p_FD[(freq>=freqmin)*(freq<freqmax)]
    freq = freq[(freq>=freqmin)*(freq<freqmax)]

    # Generate PSD
    if 1:
        psd=np.zeros(len(freq))
        for i in range(len(freq)):
            psd[i]=lalsim.SimNoisePSDaLIGOZeroDetHighPower(freq[i])
    else:
        # Load PSD
        psd_file = '/Users/jclark/Projects/HMNS/spectrum_quantile_plot/H-H1_spectrum_quantiles.txt'
        psd_data = np.loadtxt(psd_file)
        psd_freq = psd_data[:,0]
        psd = psd_data[:,1]/np.log(2)
        psd = np.interp(freq,psd_freq,psd)

    # SNR^2 versus frequency.
    rho2f = 2.0*p_FD/psd;

    # Get peak frequency
    fPeak = freq[np.argmax(p_FD)]

    # Characteristic frequency.
    fChar = np.trapz(freq*rho2f,x=freq)/np.trapz(rho2f,x=freq);

    # SNR 
    rho = np.sqrt(np.trapz(rho2f,x=freq))

    # hrss
    hrss = np.trapz(p_FD,x=freq)**0.5

    # energy (assume elliptical polarisation)
    D = 20.0 * 1e6 * lal.PC_SI
    Egw = 8.0 * lal.PI**2 * lal.C_SI**3 * D*D * \
            np.trapz(freq*freq*p_FD, x=freq)
    Egw /= 5*lal.G_SI

    Egw /= lal.MSUN_SI * lal.C_SI * lal.C_SI

    # return Egw in solar masses

    return rho,hrss,fChar,fPeak,Egw

def find_fwhm(Fsearch,Psearch,peak_idx=None):
    """
    Compute the full width at half-maximum around the highest peak in Psearch
    """
    # XXX: this only returns the FWHM to the resolution of the PSD.  This could
    # be changed to use interpolation.

    # Find FWHM
    if peak_idx is None:
        peak_idx=np.argmax(Psearch)

    # Closest values to half maximum
    delta=abs(Psearch-0.5*Psearch[peak_idx])

    # Frequencies above and below peak
    Fmax=Fsearch[peak_idx]
    idx_upp=np.argwhere(Fsearch>Fmax)
    idx_low=np.argwhere(Fsearch<Fmax)

    # Frequencies nearest the half maxima on either side of the peak
    Flow=min(Fsearch[idx_low][np.argmin(delta[idx_low])])
    Fupp=max(Fsearch[idx_upp][np.argmin(delta[idx_upp])])
    
    return Fupp-Flow

def fit_gaussian_peak(waveform):
    """
    Fit a Gaussian to the peak of the PSD in the waveform class
    """

    # ensure we have all the data and characteristics required
    if not hasattr(waveform,'PSD_plus'): waveform.make_wf_freqseries()
    if not hasattr(waveform,'fpeak'): waveform.compute_characteristics()

    # initial guess at params
    p0_gauss = [max(waveform.PSD_plus), waveform.fpeak, 0.5*waveform.fwhm]

    # select out data to fit
    flow = waveform.fpeak-0.5*waveform.fwhm
    fupp = waveform.fpeak+0.5*waveform.fwhm
    psd_fit_data = waveform.PSD_plus[(waveform.freq_axis>flow) * \
            (waveform.freq_axis<fupp)]
    freq_fit_data = waveform.freq_axis[(waveform.freq_axis>flow) * \
            (waveform.freq_axis<fupp)]

    try:
        coeff_gauss, var_matrix_gauss = optimize.curve_fit(gauss_curve,
                freq_fit_data, psd_fit_data, p0=p0_gauss)
    except RuntimeError:
        coeff_gauss = zeros(shape=3)
        var_matrix_gauss = zeros(shape=(3,3))

    amp_fit, fpeak_fit, sigma_fit = coeff_gauss
    fwhm_fit = 2.*sigma_fit

    return amp_fit, fpeak_fit, fwhm_fit

def gauss_curve(x, A, mu, sigma):
    """
    A general Gaussian
    """
    return A*stats.norm.pdf(x, loc=mu ,scale=sigma) 


def fpeak_to_R16(fpeak,jitterwidth=None):
    """
    Compute the TOV radius of a 1.6 Msun NS from equation 3 in
    http://arxiv.org/abs/1204.1888v2
    """

    fbreak = 2.8

    alow  = -0.2823
    ahigh = -0.4667
    blow  = 6.284
    bhigh = 8.713

    R16=np.zeros(np.size(fpeak))

    R16[np.argwhere(fpeak/1000<fbreak)] = (fpeak/1000.-blow)/alow
    R16[np.argwhere(fpeak/1000>fbreak)] = (fpeak/1000.-bhigh)/ahigh

    if jitterwidth is not None:
        R16_jitter = -0.5*jitterwidth+jitterwidth*np.random.rand(np.size(R16))
        R16+=R16_jitter

    return R16

def tukeywin(window_length, alpha=0.5):
    """
    The Tukey window, also known as the tapered cosine window, can be
    regarded as a cosine lobe of width \alpha * N / 2 that is convolved with a
    rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes
    rectangular, and at \alpha = 0 it becomes a Hann window.
 
    We use the same reference as MATLAB to provide the same results in case
    users compare a MATLAB output to this function output
 
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html

    """
    # Special cases
    if alpha <= 0:
        return np.ones(window_length) #rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)
 
    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)
 
    # first condition 0 <= x < alpha/2
    first_condition = x<alpha/2
    w[first_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[first_condition] - alpha/2) ))
 
    # second condition already taken care of
 
    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x>=(1 - alpha/2)
    w[third_condition] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[third_condition] - 1 + alpha/2))) 
 
    return w

def CL_vs_PhysDist():
    """
    This data extracted from galaxy catalogue paper figure
    """

    data = [
            [1.0482250775014266,5.1537246559616445],
            [1.1161621026830237,5.153724655961663],
            [1.188502227437019,5.153724655961655],
            [1.2655308231907436,5.153724655961642],
            [1.347551756717855,5.153724655961663],
            [1.4348885888492353,5.1988715928605025],
            [1.5278858508741038,5.336699231206334],
            [1.62691040366658,5.336699231206316],
            [1.7323528848992849,5.375629058929381],
            [1.8446292500529868,5.430608372592409],
            [1.9641824133013523,5.494130741118138],
            [2.091483994743933,5.623413251903505],
            [2.227036180879864,5.623413251903505],
            [2.371373705661656,5.623413251903499],
            [2.525065959944033,5.639786014822432],
            [2.690730055697052,5.8521576545913],
            [2.8036724410946903,7.542672522627046],
            [2.985382618917961,7.901822552837815],
            [3.1812470913259574,8.850538527730347],
            [3.246113018266731,10.175968585295147],
            [3.312301567549344,12.894057731855783],
            [3.4819151350201136,14.729290178073322],
            [3.6421987112451184,17.27323721912989],
            [3.8884160517676887,18.73817422860381],
            [4.044065507078823,22.30924374408989],
            [4.216965034285822,28.56328023746043],
            [4.490272805558641,32.27292909687538],
            [4.781294059687205,33.418726726279516],
            [5.091176833821791,34.90834769036025],
            [5.421143571106645,36.3585086116857],
            [5.772495943043023,37.540058883154906],
            [6.146619984396772,37.8689120491105],
            [6.544991560906876,40.37017258596558],
            [6.969182191364353,45.74604315678953],
            [7.382132197641912,52.62742502540522],
            [7.819551030103402,62.438799656536254],
            [8.32634775405431,68.92612104349699],
            [8.865990726903478,76.97747057123702],
            [9.440608762859233,92.18207178777489],
            [10.026200001866448,109.49491212781605],
            [10.703984081132965,132.19411484660287],
            [11.397725198066265,154.66567432570935],
            [12.1364287078503,187.3817422860383],
            [12.923008689990395,217.96353277791158],
            [13.760568089815644,256.8005559439765],
            [14.652410959153606,314.3943606921614],
            [15.60205549034139,366.77011326441533],
            [16.526535398566878,440.4933880954301],
            [17.492711874398413,521.8341147110958],
            [18.350012466511885,657.9332246575663],
            [19.437319062202402,787.8882911698034],
            [20.589052624097437,947.6908045141553],
            [21.809030689946418,1116.81129600575],
            [23.222506382305163,1291.54966501488],
            [24.72759153504261,1484.9682622544606],
            [26.330223493439085,1675.8780942343196],
            [28.118354307459995,1839.9872550553773],
            [29.96562750571397,2225.3718873784937],
            [31.788697034299673,2679.3552711303446],
            [33.86865730958635,3150.043116798985],
            [36.105699024829526,3786.891204911054],
            [37.86010535689841,4343.889687584815]]


    return np.array(data)

def CL_vs_HorzDist():

    data=[
           [1.0482250775014286,2.5724890531004134],
           [1.1161621026830242,2.71858824273294],
           [1.1885022274370214,2.8233035600221092],
           [1.2655308231907416,2.957737284386305],
           [1.3475517567178565,3.10759377219193],
           [1.4348885888492358,3.2745491628777486],
           [1.5278858508741016,3.42051035167133],
           [1.6269104036665767,3.6200295963013707],
           [1.732352884899287,3.797916870129048],
           [1.8446292500529873,3.9787575409327993],
           [1.9641824133013497,4.1320124001153555],
           [2.091483994743934,4.27871275069429],
           [2.227036180879864,4.482446879289324],
           [2.371373705661655,4.60128133333346],
           [2.5250659599440355,4.778514732339206],
           [2.6887192376492473,4.948168092155151],
           [2.86297912750975,5.123844717380768],
           [3.0485330598232525,5.367820440776203],
           [3.246113018266732,5.607088020462724],
           [3.456498427467163,5.891176307140282],
           [3.680519227720025,6.243879965653657],
           [3.9190591489848123,6.484383423088379],
           [4.173059197073132,6.8526505541916976],
           [4.443521365781806,7.1998464155792155],
           [4.731512589614808,7.697747057123746],
           [5.038168952687044,8.206187091305475],
           [5.364700170413394,9.058800743644513],
           [5.712394361662688,10.146426911075636],
           [6.082623130202107,11.26594629900075],
           [6.476846975477492,12.364362155264358],
           [6.896621054074424,13.569872201511744],
           [7.3436013145881285,14.635381255577812],
           [7.819551030103399,15.922827933410963],
           [8.326347754054309,17.679694142400283],
           [8.865990726903478,19.234943239861558],
           [9.440608762859233,21.049041445120153],
           [10.052468647742682,22.967361763386293],
           [10.703984081132965,25.650209056800406],
           [11.397725198066265,28.48035868435799],
           [12.1364287078503,31.166416393504004],
           [12.923008689990402,35.180037357462645],
           [13.760568089815651,38.986037025490795],
           [14.652410959153606,44.35776920824754],
           [15.602055490341384,50.79349312944742],
           [16.625672479242972,57.366454402426704],
           [17.68997718225404,65.52607301436079],
           [18.83649089489799,74.77172047313216],
           [20.05731186524713,86.4706426905291],
           [21.373228552614016,100.24950827598643],
           [22.74145156407282,118.02382812915613],
           [24.21535883908015,138.891851614525],
           [25.784792235153187,162.5019533829791],
           [27.455942942172474,191.23424103512446],
           [29.235403410236195,222.4447817359433],
           [31.130193355931485,267.93552711303556],
           [33.147787454110215,311.66416393503874],
           [35.29614482441081,368.90895022418346],
           [37.583740428844415,439.21459954003336],
           [39.92994401682097,509.83712613116535],
           [41.08056080177272,596.0085984796393],
           [43.51473706164981,689.2612104349681],
           [46.33499184373776,799.4250960446699],
           [49.33803152981389,968.5256148599307],
           [51.17889550210738,1100.6941712522057],
           [54.21143571106643,1222.140713536755],
           [57.72495943043012,1384.8863713938633],
           [61.14537831239096,1562.9295196511955]]

    return np.array(data)


def main():
    """
    Create a waveform class for the user-specified waveform and print
    interesting attributes to stdout
    """
    waveform = Waveform(sys.argv[1])

    # compute characteristics
    waveform.compute_characteristics()

    if waveform.waveform_name!='sfho_1616':
        outstr="""
    {waveform} summary
    *******************
    fchar: {fchar}
    fpeak: {fpeak}
    fwhm: {fwhm}
    fpeak_fit: {fpeak_fit}
    fwhm_fit: {fwhm_fit}
    R16: {R16}
    R16_fit: {R16_fit}
    Brodband content:
    SNR: {snr}
    hrss: {hrss}
    Egw: {Egw}
    Narrowband content:
    SNR: {snr_peak}
    hrss: {hrss_peak}
    Egw: {Egw_peak}
        """.format(waveform=waveform.waveform_name,
                snr=waveform.snr_plus,
                hrss=waveform.hrss,
                Egw=waveform.Egw,
                snr_peak=waveform.snr_plus_peak,
                hrss_peak=waveform.hrss_peak,
                Egw_peak=waveform.Egw_peak,
                fchar=waveform.fchar,
                fpeak=waveform.fpeak,
                fwhm=waveform.fwhm,
                fpeak_fit=waveform.fpeak_fit,
                fwhm_fit=waveform.fwhm_fit,
                R16=waveform.R16_fpeak,
                R16_fit=waveform.R16_fpeak
                )
    else:
        outstr="""
    {waveform} summary
    *******************
    fchar: {fchar}
    fpeak: {fpeak}
    Brodband content:
    SNR: {snr}
    hrss: {hrss}
    Egw: {Egw}
        """.format(waveform=waveform.waveform_name,
                snr=waveform.snr_plus,
                hrss=waveform.hrss,
                Egw=waveform.Egw,
                fchar=waveform.fchar,
                fpeak=waveform.fpeak
                )

    print >> sys.stdout, outstr


    import pylab as pl
    pl.figure(figsize=(10,4))
    pl.subplot(121)
    pl.plot(waveform.time_axis,waveform.hplus.data.data)
    pl.xlim(-0.01,0.02)
    pl.minorticks_on()

    pl.subplot(122)
    if not hasattr(waveform, 'PSD_plus'): waveform.make_wf_freqseries()
    pl.plot(waveform.freq_axis, waveform.PSD_plus)

    if waveform.waveform_name != 'sfho_1616':
        pl.plot(waveform.freq_axis, waveform.gauss_fit,'r')
        pl.axvline(waveform.fpeak,color='r')
        pl.axvline(waveform.fpeak-0.5*waveform.fwhm,color='r',linestyle='--')
        pl.axvline(waveform.fpeak+0.5*waveform.fwhm,color='r',linestyle='--')
        pl.xlim(waveform.fpeak-500,waveform.fpeak+500)
    
    pl.show()


#
# End definitions
#
if __name__ == "__main__":
    main()

