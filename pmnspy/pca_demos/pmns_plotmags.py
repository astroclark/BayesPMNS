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
plotmags.py

make some nice plots showing the original, aligned and then centered magnitude
spectra
"""

from __future__ import division
import os,sys
import numpy as np

from matplotlib import pyplot as pl

import pmns_utils as pu
import pmns_pca_utils as ppca

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA
#
# Produces: 1) Explained variance calculation
#           2) Matches for waveforms reconstructed from their own PCs

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

#
# Create PMNS PCA instance for this catalogue
#

nTsamples=16384

pmpca = ppca.pmnsPCA(waveform_names, low_frequency_cutoff=1000, fcenter=2710,
        nTsamples=nTsamples)


#
# Plotting
#
f, ax = pl.subplots(ncols=3, figsize=(15,5))#, sharey=True)

for i in xrange(3):
    ax[0].plot(pmpca.sample_frequencies, pmpca.magnitude[i,:],
            label=waveform_names[i].replace('_135135_lessvisc',''))
    ax[0].set_title('Original Spectra')

    ax[1].plot(pmpca.sample_frequencies, pmpca.magnitude_align[i,:],
            label=waveform_names[i].replace('_135135_lessvisc',''))
    ax[1].set_title('Aligned Spectra')

    ax[2].plot(pmpca.sample_frequencies,
            pmpca.magnitude_align[i,:]-pmpca.pca['magnitude_mean'],
            label=waveform_names[i].replace('_135135_lessvisc',''))

    ax[2].set_title('Aligned & Centered Spectra')

ax[0].legend()

ax[0].set_xlim(999,4096)
ax[0].minorticks_on()
ax[0].grid()
ax[1].set_xlim(999,4096)
ax[1].minorticks_on()
ax[1].grid()
ax[2].set_xlim(999,4096)
ax[2].minorticks_on()
ax[2].grid()

ax[0].set_ylabel('|H(f)|')
ax[0].set_xlabel('Frequency [Hz]')
ax[1].set_xlabel('Frequency [Hz]')
ax[2].set_xlabel('Frequency [Hz]')

#f.subplots_adjust(wspace=0)

pl.show()



