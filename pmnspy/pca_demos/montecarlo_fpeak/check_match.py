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

ifo = sys.argv[1]
results_files = glob.glob("maxLfpeak_%s*SNR*npz"%ifo)

rerunfile=open("%s_resume.sh"%ifo, 'w')

for r, result in enumerate(results_files):

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load Results

    data = np.load(result)

    try:
        matches = data['matches']
    except KeyError:

        print '---'
        print result
        cmdline = "pmns_pca_recovery.py %s %s %s"%(
                ifo, str(data['eos']), str(data['mass'])
                )

        print cmdline
        rerunfile.writelines(cmdline+'\n')

rerunfile.close()








