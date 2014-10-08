#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <james.clark@ligo.org>
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
Chirplet_demo.py

Generates plots and sanity checks for new chirplet waveform in LALSimulation

See LIGO-XXXXXX-XX for details.
"""

from __future__ import division
import os,sys
import numpy as np
#np.seterr(all="raise", under="ignore")
import matplotlib
from matplotlib import pyplot as pl

import lalsimulation as lalsim

import pycbc
import pycbc.filter

#
# Demo chirp parameters
#

delta_t = 1.0/16384
delta_f = 0.1
phi0 = 0.0

# Use the same parameters as figure 1 in arXiv:1005.2876
Q = 50.0
centre_frequency = 350.0
chirp_rate = -5000.0
hrss = 1.0

# -----------------------------------------
# Plus-polarised time-domain waveform
#

print " -----------------------------------------------"
print ""
print "Generating plus-polarised TD Chirplet waveform"
alpha = 0.0
hp_tmp, hc_tmp = lalsim.SimBurstChirplet(Q, centre_frequency, chirp_rate, hrss, alpha,
        phi0, delta_t)
print "Check hrss:  desired=%f, actual=%f"%(hrss,
        lalsim.MeasureHrss(hp_tmp,hc_tmp))

# Put data in pycbc TimeSeries object for further manipulation
times = np.arange(0, hp_tmp.data.length * delta_t, delta_t) + hp_tmp.epoch
hp = pycbc.types.TimeSeries(initial_array=hp_tmp.data.data, delta_t=delta_t,
        epoch=hp_tmp.epoch)
hc = pycbc.types.TimeSeries(initial_array=hc_tmp.data.data, delta_t=delta_t,
        epoch=hc_tmp.epoch)

#
# Make plots
#


f,ax = pl.subplots(nrows=1, ncols=2, figsize=(10,4))

# Time domain
ax[0].plot(hp.sample_times, hp, label='Plus')
ax[0].plot(hc.sample_times, hc, label='Cross')
ax[0].set_title('Measured hrss=%f'%lalsim.MeasureHrss(hp_tmp,hc_tmp))
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Strain')
ax[0].legend(loc='lower left')

# Freq domain
ax[1].loglog(hp.to_frequencyseries().sample_frequencies,
        abs(hp.to_frequencyseries())**2, label='Plus')
ax[1].loglog(hc.to_frequencyseries().sample_frequencies,
        abs(hc.to_frequencyseries())**2, label='Cross')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].legend(loc='lower left')

f.tight_layout()

f.savefig('LinearPol.eps')

# -----------------------------------------
# Circularly polarised time-domain waveform

print " -----------------------------------------------"
print ""
print "Generating circ-polarised TD Chirplet waveform"
alpha = np.pi/4
hp_tmp, hc_tmp = lalsim.SimBurstChirplet(Q, centre_frequency, chirp_rate, hrss, alpha,
        phi0, delta_t)
print "Check hrss:  desired=%f, actual=%f"%(hrss,
        lalsim.MeasureHrss(hp_tmp,hc_tmp))

# Put data in pycbc TimeSeries object for further manipulation
times = np.arange(0, hp_tmp.data.length * delta_t, delta_t) + hp_tmp.epoch
hpc = pycbc.types.TimeSeries(initial_array=hp_tmp.data.data, delta_t=delta_t,
        epoch=hp_tmp.epoch)
hcc = pycbc.types.TimeSeries(initial_array=hc_tmp.data.data, delta_t=delta_t,
        epoch=hc_tmp.epoch)

print "Check hrss:  desired=%f, actual=%f"%(hrss,
        np.sqrt(np.vdot(hpc,hpc)*delta_t+np.vdot(hcc,hcc)*delta_t))

#
# Make plots
#
f,ax = pl.subplots(nrows=1, ncols=2, figsize=(10,4))

# Time domain
ax[0].plot(hpc.sample_times, hpc, label='Plus')
ax[0].plot(hcc.sample_times, hcc, label='Cross')
ax[0].set_title('Measured hrss=%f'%lalsim.MeasureHrss(hp_tmp,hc_tmp))
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Strain')
ax[0].legend(loc='lower left')

# Freq domain
ax[1].loglog(hpc.to_frequencyseries().sample_frequencies,
        abs(hpc.to_frequencyseries())**2, label='Plus')
ax[1].loglog(hcc.to_frequencyseries().sample_frequencies,
        abs(hcc.to_frequencyseries())**2, label='Cross')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].legend(loc='lower left')

f.tight_layout()

f.savefig('CircPol.eps')


# -----------------------------------------
# Plus-polarised frequency-domain waveform
#
print " -----------------------------------------------"
print ""
print "Generating plus-polarised FD Chirplet waveform"
hpf_tmp, hcf_tmp = lalsim.SimBurstChirpletF(Q, centre_frequency, chirp_rate,
        hrss, alpha, phi0, delta_f, delta_t)

hpf = pycbc.types.FrequencySeries(initial_array=hpf_tmp.data.data,
        delta_f=delta_f, epoch=hpf_tmp.epoch)
hcf = pycbc.types.FrequencySeries(initial_array=hcf_tmp.data.data,
        delta_f=delta_f, epoch=hcf_tmp.epoch)


print "Check hrss:  desired=%f, actual=%f"%(hrss,
        np.sqrt(np.vdot(hpf,hpf)*delta_f + np.vdot(hcf,hcf)*delta_f))

f,ax = pl.subplots(nrows=1, ncols=2, figsize=(10,4))

# F-domain
ax[0].plot(hpf.sample_frequencies, np.real(hpf), label='ChirpletF')
ax[0].plot(hp.to_frequencyseries().sample_frequencies,
        np.real(hp.to_frequencyseries()), label='FFT(Chirplet)')
ax[0].set_title('Measured hrss=%f'%np.sqrt(np.vdot(hpf,hpf)*delta_f + np.vdot(hcf,hcf)*delta_f))
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel('Real Part')
ax[0].legend(loc='lower right')
ax[0].set_xlim(0,1000)

xlims=ax[0].get_xlim()

# FFT'd T-domain
ax[1].plot(hpf.sample_frequencies, np.imag(hpf), label='ChirpletF')
ax[1].plot(hc.to_frequencyseries().sample_frequencies,
        np.imag(hp.to_frequencyseries()), label='FFT(Chirplet)')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Imaginary Part')
ax[1].legend(loc='lower right')
ax[1].set_xlim(xlims)
ax[1].set_xlim(0,1000)

f.tight_layout()

f.savefig('FdomainLin.eps')


pl.show()
