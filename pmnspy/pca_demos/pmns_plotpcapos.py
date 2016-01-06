#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
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
import glob
import numpy as np
from matplotlib import pyplot as pl

from sklearn.neighbors.kde import KernelDensity

from pmns_utils import pmns_waveform as pwave

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def deltafpeak_to_deltaR16(fpeak, delta_fpeak, mass_string):
    """
    Compute 
    R16 = a*fpeak**2 + b*fpeak + c

    where (a,b,c) are functions of the total mass of the system
    """

    fpeak_kHz = fpeak/1000.0
    delta_fpeak_kHz = delta_fpeak/1000.0

    # Parse mass string; convert to Mtot
    if len(mass_string)==4: den=10.0
    else: den=100.0

    m1=float(mass_string[:int(len(mass_string)/2)]) / den
    m2=float(mass_string[int(len(mass_string)/2):]) / den
    mtotal = m1+m2

    if mtotal==2.4:
        a=1.143
        b=-8.79
        c=27.7
        deltaR_systematic=.236
    elif mtotal==2.7:
        a=1.099
        b=-8.57
        c=28.1
        deltaR_systematic=.175
    elif mtotal==3.0:
        a=-0.463
        b=0.580
        c=18.5
        deltaR_systematic=.254
    else:
        # No relationship
        return np.nan, np.nan

    deltaR16_statistical = a*fpeak_kHz**2 + b*fpeak_kHz + c

    deltaR16_statistical = abs(2.0*a*fpeak_kHz+b)*delta_fpeak_kHz

    return deltaR16_statistical, deltaR_systematic


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pre-allocation and identify results

ifo = sys.argv[1]
results_files = glob.glob("maxLfpeak_%s*SNR*npz"%ifo)

fpeak_modes = np.zeros(len(results_files))
fpeak_std = np.zeros(len(results_files))
fpeak_interquartiles = np.zeros(len(results_files))
fpeak_targets = np.zeros(len(results_files))
fpeak_fisher = np.zeros(len(results_files))
deltaR16_statistical = np.zeros(len(results_files))
deltaR16_bias = np.zeros(len(results_files))
deltaR16_systematic = np.zeros(len(results_files))

for r, result in enumerate(results_files):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load Results

    data = np.load(result)

    fpeak_maxL=data['fpeak_maxL']

    sigma=data['sigma']
    SNR=data['SNR']
    eos=data['eos']
    mass=data['mass']
    sigma_fpeak=data['sigma_fpeak']
    target_fpeak=data['target_fpeak']


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Radius Calculations 

    deltaR16_statistical[r], deltaR16_systematic[r] = \
            deltafpeak_to_deltaR16(target_fpeak, np.std(fpeak_maxL), str(mass))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Show distribution of max-likelihood fpeaks

    fpeak_grid = np.arange(1000, 4000, 0.1)
    fpeak_pdf = kde_sklearn(fpeak_maxL, fpeak_grid, bandwidth=50)
    fpeak_mode = fpeak_grid[np.argmax(fpeak_pdf)]

    deltaR16_bias[r], _ = deltafpeak_to_deltaR16(target_fpeak,
            target_fpeak-fpeak_mode, str(mass))


    f, ax = pl.subplots()
    #bins = np.arange(1000, 4000, 25)
    bins = 25
    ax.hist(fpeak_maxL, bins=bins, normed=True, histtype='stepfilled', alpha=0.5)
    ax.set_xlabel('Max-likelihood f$_{\mathrm{peak}}$')
    ax.set_ylabel('Normalised count')

    ax.axvline(target_fpeak, label='Target Value=%.2f'%target_fpeak, color='r')
    ax.axvline(target_fpeak+sigma_fpeak, label='Fisher Uncertainty=%.2f'%sigma_fpeak, color='r',
            linestyle='--')
    ax.axvline(target_fpeak-sigma_fpeak, color='r', linestyle='--')

    ax.axvline(fpeak_mode, color='g',
            label='mode=%.2f'%(fpeak_mode))

    upp = np.percentile(fpeak_maxL, 75)
    low = np.percentile(fpeak_maxL, 25)

    #upp = np.mean(fpeak_maxL)+np.std(fpeak_maxL)
    #low = np.mean(fpeak_maxL)-np.std(fpeak_maxL)

    ax.axvline(upp, color='g', label='Interquartile width=%.2f'%(upp-low),
            linestyle='--', linewidth=2)
    ax.axvline(low, color='g', linestyle='--', linewidth=2)

    ax.plot(fpeak_grid, fpeak_pdf, color='k', linewidth=2, label='KDE')

    ax.legend(loc='upper left')
    ax.minorticks_on()

    ax.set_xlim(500, 4000)

    print '---'
    #print "Empirical std / Fisher = %.2f"%(np.std(fpeak_maxL) / sigma_fpeak)
    #print "Empirical std / Fisher = %.2f"%(0.5*(upp-low) / sigma_fpeak)

    print "Interquartile Range: %.2f"%(upp-low)
    print "Empirical std: %.2f"%np.std(fpeak_maxL)
    print "Fisher: %.2f"%sigma_fpeak

    #print "Systematic: %.2f"%(target_fpeak - np.median(fpeak_maxL))
    print "Systematic: %.2f"%(target_fpeak - fpeak_mode)

    print "Statistical Radius Error: %.2f"%deltaR16_statistical[r]
    print "Radius Bias: %.2f"%deltaR16_bias[r]
    print "Systematic Radius Error: %.2f"%deltaR16_systematic[r]


    ax.set_title("Systematic: %.2f\nInterquartile Width: %.2f"%(target_fpeak -
        fpeak_mode, upp-low))


    pl.tight_layout()

    pl.savefig(sys.argv[1].replace('npz','png'))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Append and save results

    fpeak_modes[r] = fpeak_mode
    fpeak_std[r] = np.std(fpeak_maxL)
    fpeak_interquartiles[r] = upp-low
    fpeak_targets[r] = target_fpeak
    fpeak_fisher[r]  = sigma_fpeak

outfile="maxLfpeak_%s"%ifo
np.savez(outfile, fpeak_modes=fpeak_modes, fpeak_std=fpeak_std,
        fpeak_interquartiles=fpeak_interquartiles, fpeak_targets=fpeak_targets,
        fpeak_fisher=fpeak_fisher)





