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
import numpy as np
from matplotlib import pyplot as pl

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load Results
data = np.load(sys.argv[1])

fpeak_maxL=data['fpeak_maxL']
sigma=data['sigma']
SNR=data['SNR']
eos=data['eos']
mass=data['mass']
sigma_fpeak=data['sigma_fpeak']
target_fpeak=data['target_fpeak']


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Show distribution of max-likelihood fpeaks

f, ax = pl.subplots()
#bins = np.arange(1000, 4000, 25)
bins = 10
ax.hist(fpeak_maxL, bins=bins, normed=True, histtype='stepfilled', alpha=0.5)
ax.set_xlabel('Max-likelihood f$_{\mathrm{peak}}$')
ax.set_ylabel('Normalised count')

ax.axvline(target_fpeak, label='Target Value', color='r')
ax.axvline(target_fpeak+sigma_fpeak, label='Fisher', color='r', linestyle='--')
ax.axvline(target_fpeak-sigma_fpeak, color='r', linestyle='--')

ax.axvline(np.median(fpeak_maxL), color='g', label='mean')

#upp = np.percentile(fpeak_maxL, 75)
#low = np.percentile(fpeak_maxL, 25)

upp = np.mean(fpeak_maxL)+np.std(fpeak_maxL)
low = np.mean(fpeak_maxL)-np.std(fpeak_maxL)

ax.axvline(upp, color='g', label='1$\sigma$', linestyle='--')
ax.axvline(low, color='g', linestyle='--')

ax.legend(loc='upper left')
ax.minorticks_on()

ax.set_xlim(1000, 4000)

#print "Empirical std / Fisher = %.2f"%(np.std(fpeak_maxL) / sigma_fpeak)
print "Empirical std / Fisher = %.2f"%(0.5*(upp-low) / sigma_fpeak)

print "Empirical std: %.2f"%np.std(fpeak_maxL)
print "Fisher: %.2f"%sigma_fpeak


pl.show()
