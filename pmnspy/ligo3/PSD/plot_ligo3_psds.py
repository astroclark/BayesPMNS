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
import matplotlib
from matplotlib import pyplot as pl
import glob

psd_path=sys.argv[1]
psds=glob.glob(psd_path+'/*.txt')

fig,ax=pl.subplots()
for psd in psds:
    data=np.loadtxt(psd)
    
    if 'ZERO_DET' not in psd:
        data[:,1]=np.sqrt(data[:,1])

    ax.loglog(data[:,0], data[:,1], label=psd.split('/')[-1].replace('.txt',''))

ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('$\sqrt{S_h(f)}$ (Hz$^{-1/2}$)')
ax.legend()
ax.set_xlim(3,4e3)
fig.savefig('all_psds.png')

