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
import numpy as np

from matplotlib import pyplot as pl
from itertools import cycle

from pmns_utils import pmns_waveform as pwave


def detector_names(instrument):
    """
    Parse the detector name from the pickle file name and return string for plot
    labels
    """

    labels=dict()
    labels['aLIGO'] = 'aLIGO'
    labels['A+'] = 'A+'
    labels['A++'] = 'A++'
    labels['CE1'] = 'CE'
    labels['CE2_narrow'] = 'LIGO CE2'
    labels['ET-D'] = 'ET-D'
    labels['Voyager'] = 'LV'

    return labels[instrument]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set noise curves and example waveform

instruments=['aLIGO', 'A+', 'Voyager', 'CE1', 'ET-D']
fmax=4096
fmin=10
delta_f=0.5

# --- Example signal
eos="tm1"
mass="135135"
viscosity="lessvisc"

waveform = pwave.Waveform(eos=eos, mass=mass,
        viscosity=viscosity, distance=50)
waveform.reproject_waveform()
waveform.compute_characteristics()
Hplus = waveform.hplus.to_frequencyseries()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build curves and plot on the fly

f, ax = pl.subplots()

ax.semilogy(Hplus.sample_frequencies,
        2*np.sqrt(Hplus.sample_frequencies)*abs(Hplus.data), 
        label='BNS @ 50 Mpc', color='k')

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

for instrument in instruments:

    psd = pwave.make_noise_curve(fmax=fmax, delta_f=delta_f,
            noise_curve=instrument)

    det_label = detector_names(instrument)


    ax.semilogy(psd.sample_frequencies, np.sqrt(psd), label=det_label,
            linestyle=next(linecycler))


ax.grid()
ax.legend()
ax.minorticks_on()
ax.set_xlim(999,4096)
ax.set_ylim(1e-25, 1e-20)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('2|H$_+$($f$)|$\sqrt{f}$ & $\sqrt{S(f)}$')
pl.show()


