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
mode.  Generates 1 output file for each mode, in NINJA format, as well as an
ini file for input to the LAL NINJA codes.

USAGE: spharms_from_quadrupoles.py <input_file> <waveform_label>

Input file must contain mass quadrupole data in geometerized units with the
following columns:

    times, Ixx, Ixy, Ixz, Iyy, Iyz, Izz 

See http://arxiv.org/abs/0709.0093 for details

"""

import sys
from optparse import OptionParser

import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp

import lal
import lalsimulation as lalsim

def parser():

    # --- Command line input
    parser = OptionParser()
    parser.add_option("-w", "--waveform-name", default="mattersim", type=str)
    parser.add_option("-D", "--extraction-distance", type=float, default=None)
    parser.add_option("-f", "--sample-rate", type=float, default=16384)
    parser.add_option("-s", "--ninja-units", default=False, action="store_true")
    parser.add_option("-q", "--mass-ratio", default=1.0, type=float)
    parser.add_option("-g", "--nr-group", default="unspecified", type=str)
    parser.add_option("-d", "--simulation-details", default=None)

    (opts,args) = parser.parse_args()

    if opts.extraction_distance is None:
        print >> sys.stderr, "Error: no extraction distance defined.  "\
                "For SN, this is probably 0.01 (10 kpc). For BNS 20 (Mpc)"
        sys.exit()

    return opts, args

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

    if m==-2:
        Hlm = np.sqrt(4*lal.PI/5) * (Ixx - Iyy + 2*1j*Ixy)
    elif m==-1:
        Hlm = np.sqrt(16*lal.PI/5) * (Ixx + 1j*Iyz)
    elif m==0:
        Hlm = np.sqrt(32*lal.PI/15) * (Izz - 0.5*(Ixx + Iyy))
    elif m==1:
        Hlm = np.sqrt(16*lal.PI/5) * (-1*Ixx + 1j*Iyz)
    elif m==2:
        Hlm = np.sqrt(4*lal.PI/5) * (Ixx - Iyy - 2*1j*Ixy)

    return Hlm

def interpolate(x_old, y_old, x_new):
    """
    Convenience funtion to avoid repeated code
    """
    interpolator = interp.interp1d(x_old, y_old)
    return interpolator(x_new)

####################################################
# INPUT

#
# Parse input
#
opts, args = parser()

simdatafile=args[0]
waveformlabel=opts.waveform_name
if opts.simulation_details is None:
    simulation_details = opts.waveform_name
else:
    simulation_details = opts.simulation_details

extract_dist=opts.extraction_distance

sample_rate = opts.sample_rate * lal.MTSUN_SI

#
# Load quadrupole data
#
times, Ixx, Ixy, Ixz, Iyy, Iyz, Izz = \
        np.loadtxt(simdatafile, unpack=True)

# ensure first time stamp is zero
times -= times[0]

###########################
#                         #
# Scaling And Resampling  #
#                         #
###########################
if opts.ninja_units:

    # don't do anything
    Ixx *= 1.0
    Ixy *= 1.0
    Ixz *= 1.0
    Iyy *= 1.0
    Iyz *= 1.0
    Izz *= 1.0

else:
    # Convert to NINJA units
    #massMpc = lal.MRSUN_SI / ( extract_dist * lal.PC_SI * 1.0e6)
    massMpc = lal.MRSUN_SI / ( extract_dist )#* lal.PC_SI * 1.0e6)

    # Also 
    #fac = lal.MRSUN_SI # * lal.C_SI**4 / lal.G_SI

    Ixx /= massMpc 
    Ixy /= massMpc 
    Ixz /= massMpc 
    Iyy /= massMpc 
    Iyz /= massMpc 
    Izz /= massMpc 
    
    times /= lal.MTSUN_SI


######################################################################
#                                                                    #
# Construct expansion parameters and write to ascii in NINJA format  #
#                                                                    #
######################################################################

target_times = np.arange(0, times[-1], 1./sample_rate)

# putting the data in laltimeseries allows us to use lalsim tapering functions
hplus_sim=lal.CreateREAL8TimeSeries('hplus', lal.LIGOTimeGPS(), 0.0,
        1./sample_rate, lal.StrainUnit, len(target_times))

hcross_sim=lal.CreateREAL8TimeSeries('hcross', lal.LIGOTimeGPS(), 0.0,
        1./sample_rate, lal.StrainUnit, len(target_times))

#
# Construct an ini file for further processing with NINJA-type tools
#
inifile = open(waveformlabel+".ini", 'w')

# XXX: hard-coding the mass-ratio and mass-scale here.  We can add these as
# arguments later if desired.  Mass ratio is irrelevant for general matter
# waveforms, but is needed by the existing ninja codes
headerstr="""mass-ratio = {0}
mass-scale = 1
simulation-details = {1}
nr-group = {2}\n
""".format(opts.mass_ratio, 
        simulation_details,
        opts.nr_group)
inifile.writelines(headerstr)

# Loop over harmonics
print >> sys.stdout, "Building modes ..."
for m in [-2,-1,0,1,2]:

    filename=waveformlabel+"_l2m%d.asc"%m

    # Construct expansion parameters
    Hlm = construct_Hlm(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, l=2, m=m)

    #
    # Resample to uniform spacing at 16384 kHz
    #
    Hlm_real = interpolate(times, Hlm.real, target_times)
    Hlm_imag = interpolate(times, Hlm.imag, target_times)

    # Populate time series
    hplus_sim.data.data  = Hlm_real
    hcross_sim.data.data = -1*Hlm_imag

    # --- Apply Tapering Window (important for merger sims)
    lalsim.SimInspiralREAL8WaveTaper(hplus_sim.data,
            lalsim.SIM_INSPIRAL_TAPER_START)
    lalsim.SimInspiralREAL8WaveTaper(hcross_sim.data,
            lalsim.SIM_INSPIRAL_TAPER_START)

    # --- Write data to file
    f = open(filename,'w')
    for j in range(hplus_sim.data.length):
        f.write("%.16f %.16f %.16f\n"%(target_times[j], hplus_sim.data.data[j],
            hcross_sim.data.data[j]))
    f.close()

    # --- append to ini file
    inifile.writelines("2,{0} = {1}\n".format(m, filename))

inifile.close()




