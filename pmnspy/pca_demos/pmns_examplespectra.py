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
pmns_pca_loomatches.py

Script to produce matches from PCA of post-merger waveforms.  This version
computes the matches when the test waveform is removed from the training data
(leave-one-out strategy)
"""

from __future__ import division
import os,sys
import cPickle as pickle
import numpy as np

from matplotlib import pyplot as pl

import pycbc.types

from pmns_utils import pmns_waveform as pwave
from pmns_utils import pmns_waveform_data as pdata
from pmns_utils import pmns_pca as ppca

pl.rcParams.update({'axes.labelsize': 18})
pl.rcParams.update({'xtick.labelsize':18})
pl.rcParams.update({'ytick.labelsize':18})
pl.rcParams.update({'legend.fontsize':18})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue

# XXX: Hardcoding
nTsamples=16384
low_frequency_cutoff=1000
fcenter=2710
deltaF=1.0

eos="all"
mass="135135"
viscosity="lessvisc"


# XXX: should probably fix this at the module level..
if eos=="all": eos=None
if mass=="all": mass=None
if viscosity=="all": viscosity=None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instantiate
#

#
# Create the list of dictionaries which comprises our catalogue
#
waveform_data = pdata.WaveData(eos=eos,viscosity=viscosity, mass=mass)

#
# Create PMNS PCA instance for the full catalogue
#
pmpca = ppca.pmnsPCA(waveform_data, low_frequency_cutoff=low_frequency_cutoff,
        fcenter=fcenter, nTsamples=nTsamples)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting
#

f1, ax1 = pl.subplots()
ax1.semilogy(pmpca.sample_frequencies, pmpca.magnitude.T)
ax1.set_xlim(999, 4096)
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('|H(f)|')
ax1.set_ylim(1e-5, 1e-1)
ax1.minorticks_on()

f2, ax2 = pl.subplots()
ax2.semilogy(pmpca.sample_frequencies, pmpca.magnitude_align.T, color='0.8')
ax2.semilogy(pmpca.sample_frequencies, pmpca.pca['magnitude_mean'], color='r',
        linewidth=2, linestyle='--', label='mean')
ax2.set_xlim(999, 4096)
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel(r'|H$_{\mathrm{align}}$(f)|')
ax2.set_ylim(1e-5, 1e-1)
ax2.legend()
ax2.minorticks_on()

f1.tight_layout()
f2.tight_layout()
pl.show()

f1.savefig('all_spectra.eps')
f2.savefig('mean_all_spectra.eps')

