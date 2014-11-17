#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <james.clark@ligo.org>
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
spharms_from_quadrupoles.py

Read mass-quadrupole time-derivative data from an ascii file and construct
(sky-averaged) spherical harmonic representation for the GW signal from each
mode.  Generates 1 output file for each mode, in NINJA1 format, as well as an
ini file for input to the LAL NINJA codes.

USAGE: spharms_from_quadrupoles.py <input_file> <waveform_label>

Input file must contain mass quadrupole data in geometerized units with the
following columns:

    times, Ixx, Ixy, Ixz, Iyy, Iyz, Izz 

"""

import sys
from optparse import OptionParser

import numpy as np

import lal
import lalsimulation as lalsim


def construct_Hlm(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, l=2, m=2):
    """
    Construct the expansion parameters Hlm from T1000553.  Returns the expansion
    parameters for l=2, m=m 
    """

    if l!=2:
        print "l!=2 not supported"
        sys.exit()
    if abs(m)>2:
        print "Only l=2 supported, |m| must be <=2"
        sys.exit()
    if abs(m)!=2:
        print "Actually, only supporting |m|=2 for now, bye!"
        sys.exit()

    #H2n2 = np.sqrt(4.0*lal.PI/5.0) * lal.G_SI / lal.C_SI**4 * (Ixx - Iyy + 2*1j*Ixy)
    #H2p2 = np.sqrt(4.0*lal.PI/5.0) * lal.G_SI / lal.C_SI**4 * (Ixx - Iyy - 2*1j*Ixy)
    if m==-2:
        Hlm = np.sqrt(4.0*lal.PI/5.0) * (Ixx - Iyy + 2*1j*Ixy)

    if m==2:
        Hlm = np.sqrt(4.0*lal.PI/5.0) * (Ixx - Iyy - 2*1j*Ixy)

    return Hlm



####################################################
# INPUT
sample_rate = 16384

#
# Load data
#
parser=OptionParser()
opts,args = parser.parse_args()
simdatafile=args[0]
waveformlabel=args[1]
extract_dist=20.0

#
# Load quadrupole data
#
times, Ixx, Ixy, Ixz, Iyy, Iyz, Izz = \
        np.loadtxt(simdatafile, unpack=True)
# ensure first time stamp is zero
times -= times[0]

######################################################################
#                                                                    #
# Construct expansion parameters and write to ascii in NINJA1 format #
#                                                                    #
######################################################################

# putting the data in laltimeseries allows us to use lalsim tapering functions
hplus_sim=lal.CreateREAL8TimeSeries('hplus', lal.LIGOTimeGPS(), 0.0,
        1./sample_rate, lal.StrainUnit, len(times))

hcross_sim=lal.CreateREAL8TimeSeries('hcross', lal.LIGOTimeGPS(), 0.0,
        1./sample_rate, lal.StrainUnit, len(times))

#
# Construct an ini file for further processing with NINJA-type tools
#
inifile = open(waveformlabel+".ini", 'w')

# XXX: hard-coding the mass-ratio and mass-scale here.  We can add these as
# arguments later if desired.  Mass ratio is irrelevant for general matter
# waveforms, but is needed by the existing ninja codes
headerstr="""mass-ratio = 1.0
mass-scale = 1
simulation-details = {0}\n
""".format(waveformlabel)
inifile.writelines(headerstr)

# Loop over harmonics
for m in [-2,-1,0,1,2]:
#for m in [-2,2]:

    filename=waveformlabel+"_l2m%d.asc"%m

    if abs(m)==2:
        # Construct expansion parameters
        Hlm = construct_Hlm(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, l=2, m=m)

        # Populate time series
        hplus_sim.data.data  = extract_dist*Hlm.real
        hcross_sim.data.data = extract_dist*-1*Hlm.imag

        # --- Apply Tapering Window
        lalsim.SimInspiralREAL8WaveTaper(hplus_sim.data,
                lalsim.SIM_INSPIRAL_TAPER_START)
        lalsim.SimInspiralREAL8WaveTaper(hcross_sim.data,
                lalsim.SIM_INSPIRAL_TAPER_START)
    else:
        hplus_sim.data.data = np.zeros(len(times))
        hcross_sim.data.data = np.zeros(len(times))

    # --- Write data to file
    f = open(filename,'w')
    for j in range(hplus_sim.data.length):
        f.write("%.16f %.16f %.16f\n"%(times[j], hplus_sim.data.data[j],
            hcross_sim.data.data[j]))
    f.close()

    # --- append to ini file
    inifile.writelines("2,{0} = {1}\n".format(m, filename))

inifile.close()




