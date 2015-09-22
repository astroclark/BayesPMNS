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
import cPickle as pickle

from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pmns_utils import pmns_waveform_data as pdata
from pmns_utils import pmns_pca as ppca

pl.rcParams.update({'axes.labelsize': 18})
pl.rcParams.update({'xtick.labelsize':18})
pl.rcParams.update({'ytick.labelsize':18})
pl.rcParams.update({'legend.fontsize':18})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA
#

#
# Create the list of dictionaries which comprises our catalogue
#
#waveform_data = pdata.WaveData(viscosity='lessvisc')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Results
#
pickle_file = sys.argv[1]
waveform_data, pmpca, magnitude_euclidean, phase_euclidean, matches, \
        delta_fpeak, delta_R16 = pickle.load(open(pickle_file, "rb"))


#
# Get indices and labels of example waveforms
#
eos_examples = ['apr', 'tm1', 'dd2']
mass_examples = ['135135', '135135', '135135']
viscosity = 'lessvisc'
labels=[ eos.upper() for eos in eos_examples ]

#
# Create PMNS PCA instance for this catalogue
#
# XXX: could just load this...

#pmpca = ppca.pmnsPCA(waveform_data, low_frequency_cutoff=1000, fcenter=2710,
#        nTsamples=16384)


#
# Plotting
#
f1, ax1 = pl.subplots()#igsize=(8,3))
f2, ax2 = pl.subplots()#igsize=(8,3))
f3, ax3 = pl.subplots()#igsize=(8,3))
f4, ax4 = pl.subplots()#igsize=(8,3))
f5, ax5 = pl.subplots()#igsize=(8,3))

for i in xrange(3):

    example_idx = [j for j in xrange(waveform_data.nwaves) if
            waveform_data.waves[j]['eos']==eos_examples[i] and
            waveform_data.waves[j]['mass']==mass_examples[i] and
            waveform_data.waves[j]['viscosity']==viscosity
            ][0]

    ax1.plot(pmpca.sample_frequencies, pmpca.magnitude[example_idx,:],
            label=labels[i])

    ax2.plot(pmpca.sample_frequencies, pmpca.magnitude_align[example_idx,:],
            label=labels[i])

    ax3.plot(pmpca.sample_frequencies,
            pmpca.magnitude_align[example_idx,:]-pmpca.pca['magnitude_mean'],
            label=labels[i])


ax4.plot(pmpca.sample_frequencies, pmpca.pca['magnitude_mean'], color='k')
ax5.plot(pmpca.sample_frequencies, pmpca.pca['magnitude_pca'].components_[0,:], color='k')

ax1.legend()

ax1.set_xlim(999,4096)
ax1.minorticks_on()
ax1.grid()

ax2.set_xlim(999,4096)
ax2.minorticks_on()
ax2.grid()

ax3.set_xlim(999,4096)
ax3.minorticks_on()
ax3.grid()

ax4.set_xlim(999,4096)
ax4.minorticks_on()
ax4.grid()

ax5.set_xlim(999,4096)
ax5.minorticks_on()
ax5.grid()

ax1.set_ylabel('|H(f)|')
ax1.set_xlabel('Frequency [Hz]')

ax2.set_ylabel('|H(f)|$_{\\rm align}$')
ax2.set_xlabel('Frequency [Hz]')

ax3.set_ylabel('|H(f)|$_{\\rm cent}$')
ax3.set_xlabel('Frequency [Hz]')

ax4.set_ylabel('|H(f)|$_{\\rm mean}$')
ax4.set_xlabel('Frequency [Hz]')

ax5.set_ylabel('A$_{\\rm PC}$(f)')
ax5.set_xlabel('Frequency [Hz]')

#f.subplots_adjust(wspace=0)
f1.tight_layout()
f2.tight_layout()
f3.tight_layout()
f4.tight_layout()
f5.tight_layout()

pl.show()

f1.savefig('three_spectra.eps')
f2.savefig('three_spectra_align.eps')
f3.savefig('three_spectra_cent.eps')
f4.savefig('mean_spectrum_alleos_allmass_lessvisc.eps')
f5.savefig('first_magnitude_pc_alleos_allmass_lessvisc.eps')

#
# Explained variance
#
f, ax = pl.subplots()#figsize=(5,4))
ax.step(np.arange(1,waveform_data.nwaves+1)-0.5,
                 np.cumsum(pmpca.pca['phase_pca'].explained_variance_ratio_),
                 label='Phase spectra', color='k', linestyle='-')
ax.minorticks_on()
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.legend(loc='lower right')
ax.minorticks_on()
ylims=ax.get_ylim()
ax.set_ylim(0.98,1)
ax.set_xlim(1,10)#0.5*waveform_data.nwaves+1)
ax.grid()
f.tight_layout()


f, ax = pl.subplots()#figsize=(5,4))
ax.step(np.arange(1,waveform_data.nwaves+1)-0.5,
                np.cumsum(pmpca.pca['magnitude_pca'].explained_variance_ratio_),
                label='Magnitude spectra', color='grey')

ax.minorticks_on()
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.legend(loc='lower right')
ax.minorticks_on()
ylims=ax.get_ylim()
#ax.set_ylim(0.5,1)
ax.set_xlim(1,0.5*waveform_data.nwaves+1)
ax.grid()
f.tight_layout()

#   axins = inset_axes(ax, width="50%", height="50%", loc=1) 
#   axins.step(np.arange(1,waveform_data.nwaves+1)-0.5,
#                    np.cumsum(pmpca.pca['phase_pca'].explained_variance_ratio_),
#                    label='Phase spectra', color='k', linestyle='--')
#   axins.step(np.arange(1,waveform_data.nwaves+1)-0.5,
#                   np.cumsum(pmpca.pca['magnitude_pca'].explained_variance_ratio_),
#                   label='Magnitude spectra', color='grey')
#
#   axins.minorticks_on()
#   axins.set_ylim(0.5,1)
#   axins.set_xlim(1,20)
#   axins.grid()

f.tight_layout()

f.savefig('explained_variance_alleos_allmass_lessvisc.eps')

pl.show()


