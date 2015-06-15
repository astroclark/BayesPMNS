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

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as pl

def shift_vec(vector, target_freqs, fpeak, fcenter=1000.0):

    # Frequency shift
    fshift = fcenter / fpeak
    false_freqs = target_freqs * fshift

    aligned_vector = complex_interp(vector, target_freqs, false_freqs)

    return aligned_vector

def complex_interp(ordinate, abscissa_interp, abscissa):
    """
    Interpolate complex series
    """

    ordinate_interp_real = np.interp(abscissa_interp, abscissa,
            np.real(ordinate))
    ordinate_interp_imag = np.interp(abscissa_interp, abscissa,
            np.imag(ordinate))

    return ordinate_interp_real + 1j*ordinate_interp_imag

def reconstruct_spectrum(mean_spectrum, coefficients, principal_components,
        sample_frequencies, target_fpeak, fcenter=2710.0, npcs=1):

    # sum contributions from mean and PC corrections
    reconstructed_spectrum = np.copy(mean_spectrum)
    for i in xrange(npcs):
        reconstructed_spectrum += \
                coefficients[i]*principal_components[i,:]

    # XXX: note the order of fcenter and fpeak!
    reconstruted_spectrum = shift_vec(reconstructed_spectrum,
            sample_frequencies, fcenter=target_fpeak, fpeak=fcenter)

    return reconstruted_spectrum



# Load and extract the data
data = sio.loadmat('./postmergerpca.mat')

sample_frequencies = np.concatenate(data['sample_frequencies'])
fcenter=np.concatenate(data['fcenter'])
mean_magnitude_spectrum = np.concatenate(data['magnitude_spectrum_global_mean'])
magnitude_pcs = data['magnitude_principal_components']
magnitude_coefficients = data['magnitude_coeffficients']
fpeaks = np.concatenate(data['fpeaks'])

# Reconstruct the first magnitude spectrum in the catalogue with e.g.,:
reconstructed_magnitude = reconstruct_spectrum(mean_magnitude_spectrum,
        magnitude_coefficients[0,:], magnitude_pcs, sample_frequencies,
        fpeaks[0], npcs=len(magnitude_coefficients[0,:]))




f, ax = pl.subplots()
ax.plot(sample_frequencies, reconstructed_magnitude)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('|H(f)|')
pl.show()










