#!/usr/bin/env python

import sys
from optparse import OptionParser

import numpy as np
from scipy import stats
from scipy import signal
from scipy.signal import cspline1d, cspline1d_eval

from lmfit import minimize, Parameters, Parameter, report_fit

import lal
import lalsimulation as lalsim

# define a factor for G/c^4. 
quadrupole_factor = 1 # geometric units!

def re_Hlm(l,m,Idd):
    """
    Compute real part of Hlm from quadrupole moments
    """
    if l != 2:
        print "l != 2 not implemented! Good Bye!"
        sys.exit()
    if m < -2 or m > 2:
        print "m must be in [-2,2]! Good Bye!"
        sys.exit()
    if m == 0:
        ret = quadrupole_factor * np.sqrt(32*lal.LAL_PI/15.0) * \
              (Idd[2,2] - 0.5e0*(Idd[0,0] + Idd[1,1]))
    if m == 1:
        ret =  - quadrupole_factor * np.sqrt(16*lal.LAL_PI/5) * \
              Idd[0,2]
    if m == 2:
        ret = quadrupole_factor * np.sqrt(4*lal.LAL_PI/5) * \
              (Idd[0,0] - Idd[1,1])
    if m == -1:
        ret =  quadrupole_factor * np.sqrt(16*lal.LAL_PI/5) * \
              Idd[0,2]
    if m == -2:
        ret = quadrupole_factor * np.sqrt(4.0*lal.LAL_PI/5) * \
               (Idd[0,0] - Idd[1,1])
    return ret

def im_Hlm(l,m,Idd):
    """
    Compute imaginary part of Hlm from quadrupole moments
    """
    if l != 2:
        print "l != 2 not implemented! Good Bye!"
        sys.exit()
    if m < -2 or m > 2:
        print "m must be in [-2,2]! Good Bye!"
        sys.exit()
    if m == 0:
        ret = 0.0e0
    if m == 1:
        ret =  quadrupole_factor * np.sqrt(16*lal.LAL_PI/5) * \
              Idd[1,2]
    if m == 2:
        ret = quadrupole_factor * np.sqrt(4*lal.LAL_PI/5) * \
              -2 * Idd[0,1]
    if m == -1:
        ret = +quadrupole_factor * np.sqrt(16*lal.LAL_PI/5) * \
              Idd[1,2]
    if m == -2:
        ret =  quadrupole_factor * np.sqrt(4.0*lal.LAL_PI/5) * \
              +2 * Idd[0,1]
    return ret    


####################################################
# MAIN
####################################################


####################################################
# INPUT
# parameters
theta = 00.0
phi = 00.0
dist = 20
sample_rate = 16384
duration = 5

#
# Load data
#

parser=OptionParser()
opts,args = parser.parse_args()
sim_data=np.genfromtxt(args[0])

####################################################

#
# Set up time stamps
#

# Compute factor to get to SI units
sim_time=np.arange(0,duration,1.0/sample_rate)
SI=0
if SI:
    extraction_distance_SI = lal.LAL_PC_SI * 1e6 * dist
    geom_fac = lal.LAL_MRSUN_SI / extraction_distance_SI
else:
    geom_fac=1.0
    sim_time/=lal.LAL_MTSUN_SI

#
# Allocate storage for simulation data
#
Idotdot = np.zeros((3,3),float)

hplus_sim=lal.CreateREAL8TimeSeries('hplus', lal.LIGOTimeGPS(), 0.0,
        1./sample_rate, lal.lalStrainUnit, int(duration*sample_rate))

hcross_sim=lal.CreateREAL8TimeSeries('hcross', lal.LIGOTimeGPS(), 0.0,
        1./sample_rate, lal.lalStrainUnit, int(duration*sample_rate))

# Get index middle of series
startidx=np.floor(0.5*duration*sample_rate)

# Loop over harmonics
for m in [-2,-1,0,1,2]:

    if len(args)==2:
        filename=args[1]+"_l2m%d.asc"%m
    elif len(args)==1:
        filename="MPI_BNSmerger_1-35_1-35_l2m%d.asc"%m
    else:
        print >> sys.stderr, "error, incorrect number of input arguments"

    # Ensure initialisation to zero
    hplus_sim.data.data=np.zeros(int(duration*sample_rate))
    hcross_sim.data.data=np.zeros(int(duration*sample_rate))

    # Spin-weighted spherical harmonic term
    if SI:
        # Then compute the spherical harmonic term explicitly
        theta *= lal.LAL_PI_180
        phi   *= lal.LAL_PI_180
        sYlm = lal.SpinWeightedSphericalHarmonic(theta,phi,-2,2,m)
    else:
        # otherwise, the spherical harmonic term is computed @ line 191 of
        # NRWaveInject.c
        sYlm = 1.0

    for i in range(len(sim_data)):

        Idotdot[0,0] = sim_data[i,1]
        Idotdot[0,1] = sim_data[i,2]
        Idotdot[0,2] = sim_data[i,3]

        Idotdot[1,0] = sim_data[i,1]
        Idotdot[1,1] = sim_data[i,4]
        Idotdot[1,2] = sim_data[i,5]

        Idotdot[2,0] = sim_data[i,3]
        Idotdot[2,1] = sim_data[i,5]
        Idotdot[2,2] = sim_data[i,6]

        Idotdot *= geom_fac

        Hlm  = complex(re_Hlm(2,m,Idotdot),im_Hlm(2,m,Idotdot))
        h = sYlm * Hlm

        hplus_sim.data.data[i+startidx]  = h.real
        hcross_sim.data.data[i+startidx] = -1*h.imag
        #if abs(m)==2:
        #    hplus_sim.data.data[i+startidx]  = h.real
        #    hcross_sim.data.data[i+startidx] = -1*h.imag
        #else:
        #    hplus_sim.data.data[i+startidx] = 0.0
        #    hcross_sim.data.data[i+startidx] = 0.0

    # --- Apply Tapering Window
    lalsim.SimInspiralREAL8WaveTaper(hplus_sim.data,
            lalsim.LAL_SIM_INSPIRAL_TAPER_START)
    lalsim.SimInspiralREAL8WaveTaper(hcross_sim.data,
            lalsim.LAL_SIM_INSPIRAL_TAPER_START)

    #hcross_sim.data.data = np.imag(signal.hilbert(hplus_sim.data.data))

    # --- Write to file
    f = open(filename,'w')
    if SI:
        for j in range(hplus_sim.data.length):
            f.write("%.16e %.16e %.16e\n"%(sim_time[j],hplus_sim.data.data[j],hcross_sim.data.data[j]))
    else:
        for j in range(hplus_sim.data.length):
            f.write("%.16f %.16f %.16f\n"%(sim_time[j],hplus_sim.data.data[j],hcross_sim.data.data[j]))
    f.close()






