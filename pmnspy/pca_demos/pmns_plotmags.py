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

#fig_width_pt = 223.0  # Get this from LaTeX using \showthe\columnwidth
#inches_per_pt = 1.0/72.27               # Convert pt to inch
#golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
#fig_width = fig_width_pt*inches_per_pt  # width in inches
#fig_height = fig_width*golden_mean      # height in inches
#fig_size =  [fig_width,fig_height]
#   params = {'backend': 'ps',
#             'axes.labelsize': 10,
#             'text.fontsize': 10,
#             'legend.fontsize': 10,
#             'xtick.labelsize': 8,
#             'ytick.labelsize': 8,
#             'text.usetex': True,
#             'figure.figsize': fig_size}
#params = {'figure.figsize': fig_size}
#pl.rcParams.update(params)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct the full waveform catalogue and perform PCA
#
# Produces: 1) Explained variance calculation
#           2) Matches for waveforms reconstructed from their own PCs

waveform_names=['apr_135135_lessvisc',
                'tm1_135135_lessvisc' ,
                'dd2_135135_lessvisc' ,
                'dd2_165165_lessvisc' ,
                'shen_135135_lessvisc',
                'nl3_135135_lessvisc' ,
                'nl3_1919_lessvisc'   ,
                'tm1_135135_lessvisc' ,
                'tma_135135_lessvisc' ,
                'sfhx_135135_lessvisc',
                'sfho_135135_lessvisc']#,

#
# Create PMNS PCA instance for this catalogue
#

labels=sys.argv[1].split(',')

nTsamples=16384

pmpca = ppca.pmnsPCA(waveform_names, low_frequency_cutoff=1000, fcenter=2710,
        nTsamples=nTsamples)


#
# Plotting
#
#f, ax = pl.subplots(ncols=3, figsize=(15,5))#, sharey=True)
f1, ax1 = pl.subplots(figsize=(8,3))#, sharey=True)
f2, ax2 = pl.subplots(figsize=(8,3))#, sharey=True)
f3, ax3 = pl.subplots(figsize=(8,3))#, sharey=True)
f4, ax4 = pl.subplots(figsize=(8,3))#, sharey=True)
f5, ax5 = pl.subplots(figsize=(8,3))#, sharey=True)

for i in xrange(3):
    ax1.plot(pmpca.sample_frequencies, pmpca.magnitude[i,:],
            label=labels[i])
            #label=waveform_names[i].replace('_135135_lessvisc',''))
#    ax[0].set_title('Original Spectra')

    ax2.plot(pmpca.sample_frequencies, pmpca.magnitude_align[i,:],
            label=labels[i])
            #label=waveform_names[i].replace('_135135_lessvisc',''))
#    ax[1].set_title('Aligned Spectra')

    ax3.plot(pmpca.sample_frequencies,
            pmpca.magnitude_align[i,:]-pmpca.pca['magnitude_mean'],
            label=labels[i])
            #label=waveform_names[i].replace('_135135_lessvisc',''))

#    ax[2].set_title('Aligned & Centered Spectra')


ax4.plot(pmpca.sample_frequencies, pmpca.pca['magnitude_mean'], color='k')
ax5.plot(pmpca.sample_frequencies, pmpca.pca['magnitude_pca'].components_[0,:], color='k')

ax1.legend()
#ax2.legend()
#ax3.legend()

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


f, ax = pl.subplots()#figsize=(5,4))
ax.bar(np.arange(1,len(waveform_names)+1)-0.5,
                np.cumsum(pmpca.pca['phase_pca'].explained_variance_ratio_),
                        label='Phase spectra', color='k')
ax.bar(np.arange(1,len(waveform_names)+1)-0.5,
                np.cumsum(pmpca.pca['magnitude_pca'].explained_variance_ratio_),
                        label='Magnitude spectra', color='grey')

ax.minorticks_on()
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.legend(loc='lower right')
ylims=ax.get_ylim()
ax.set_ylim(0.5,1)
ax.set_xlim(0,len(waveform_names)+0.5)
#ax.grid()
f.tight_layout()




pl.show()



