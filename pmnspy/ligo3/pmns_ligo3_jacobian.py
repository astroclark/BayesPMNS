#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <clark@physics.umass.edu>
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
#np.seterr(all="raise", under="ignore")
import matplotlib
#matplotlib.use("Agg")

ligo3_curves=['base', 'highNN', 'highSPOT', 'highST',
        'highSei', 'highloss', 'highmass', 'highpow', 'highsqz',
        'lowNN', 'lowSPOT', 'lowST', 'lowSei']

quantum_curves = ['mass', 'pow', 'sqz', 'loss']
nonquantum_curves = ['NN', 'SPOT', 'ST', 'Sei']

waveform_name = sys.argv[1]

# Extract all the FOM values for this waveform
foms={}
for curve in ligo3_curves:
    data = np.loadtxt('%s-%s_135135-detectability.txt'%(curve,waveform_name),
            dtype=str)
    foms[curve]=float(data[4])


# Compute jacobian values for quantum curves
jacobian={}
for curve in quantum_curves:

        S1 = foms['base']
        S2 = foms['high'+curve]
        dx = 0.2
        jacobian[curve] = np.log(S2/S1) / dx

for curve in nonquantum_curves:

        S1 = foms['low'+curve]
        S2 = foms['high'+curve]
        dx = 0.2
        jacobian[curve] = np.log(S2/S1) / dx

# Write to file
f = open ('%s-jacobian.txt'%waveform_name,'w')
f.writelines('# Noise-Curve Jacobian\n')
for val in jacobian:
    f.writelines('%s\t%.2e\n'%(val, jacobian[val]))
f.close()



