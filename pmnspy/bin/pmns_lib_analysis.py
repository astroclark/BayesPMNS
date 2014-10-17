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
np.seterr(all="raise", under="ignore")
from optparse import OptionParser
import ConfigParser
import random
import string
import time
import glob
import subprocess
import cPickle as pickle

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

import lal
import pycbc.filter
import pmns_utils
import pmns_simsig as simsig

from pylal import antenna, Fr
from glue import lal as gluelal

from sklearn.neighbors.kde import KernelDensity

def parser():

    # --- Command line input
    parser = OptionParser()
    #parser.add_option("-c", "--config-file", default="test.ini", type=str)
    parser.add_option("-w", "--waveform-name", default="dd2_135135", type=str)
    parser.add_option("-D", "--fixed-distance", type=float, default=None)
    parser.add_option("-S", "--init-seed", type=int, default=101)
    parser.add_option("-N", "--ninject", type=int, default=None)
    parser.add_option("-o", "--output-dir", type=str, default=None)

    #parser.add_option("-Dmin", "--min-distance")
    #parser.add_option("-Dmax", "--max-distance")

    (opts,args) = parser.parse_args()

    # --- Read config file
    cp = ConfigParser.ConfigParser()
    cp.read(args[0])

    return opts, args, cp


def write_frame(det_data, ifo, seed, epoch, datalen, outdir):
    """
    Write a frame 
    """

    # Construct name
    site=ifo.strip('1')

    frame_name = '{site}-{ifo}_{wf_name}_{seed}-{epoch}-{datalen}.gwf'.format(
            site=site, ifo=ifo,
            wf_name=det_data.waveform_name,  seed=seed,
            epoch=str(int(epoch)),
            datalen=str(int(datalen)))

    channel_list = [
            {'name':'%s:STRAIN'%ifo, 
                'data':np.array(det_data.td_response.data),
                'start':epoch,
                'dx':1.0/16384,
                'kind':'SIM'}, 

            {'name':'%s:SIGNAL'%ifo, 
                'data':np.array(det_data.td_signal.data),
                'start':epoch,
                'dx':1.0/16384,
                'kind':'SIM'}, 

            {'name':'%s:NOISE'%ifo, 
                'data':np.array(det_data.td_noise.data),
                'start':epoch,
                'dx':1.0/16384,
                'kind':'SIM'}, 
        ]

    print 'writing frame %s...'%frame_name

    frame_out_path = '%s/%s'%(os.path.abspath(outdir), frame_name)
    Fr.frputvect(frame_out_path, channel_list)

    #
    # Generate a cache file
    #

    # setup url
    path, filename = os.path.split(frame_out_path.strip())
    url = "file://localhost%s" % os.path.abspath(os.path.join(path, filename))

    # create cache entry
    c=gluelal.CacheEntry.from_T050017(url)

    # write to file
    cache_file = frame_out_path.replace('gwf','lcf')
    f=open(cache_file,'w')
    f.writelines('%s\n'%str(c))
    f.close()

    return frame_out_path,cache_file

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def compute_confints(x_axis,pdf,alpha):
    """
    Confidence Intervals From KDE
    """

    # Make sure PDF is correctly normalised
    pdf /= np.trapz(pdf,x_axis)

    # --- initialisation
    peak = freq_axis[np.argmax(pdf)]

    # Initialisation
    area=0.

    i=0 
    j=0 

    x_axis_left=x_axis[(x_axis<peak)][::-1]
    x_axis_right=x_axis[x_axis>peak]

    while area <= alpha:

        x_axis_current=x_axis[(x_axis>=x_axis_left[i])*(x_axis<=x_axis_right[j])]
        pdf_current=pdf[(x_axis>=x_axis_left[i])*(x_axis<=x_axis_right[j])]

        area=np.trapz(pdf_current,x_axis_current)

        if i<len(x_axis_left)-1: i+=1
        if j<len(x_axis_right)-1: j+=1

    low_edge, upp_edge = x_axis_left[i], x_axis_right[j]

    return peak,low_edge,upp_edge,area

# --- End Defs

#####################################################################
# Input
USAGE='''%prog [options] [args]
Run external analysis programs on internally generated post-merger injections
with Gaussian noise.
'''

opts, args, cp = parser()

# Data configuration
datalen=cp.getfloat('analysis', 'datalength')
flow=cp.getfloat('analysis','flow')
srate=cp.getfloat('analysis','srate')

seed=opts.init_seed
#seed+=random.randint(0,lal.GPSTimeNow())


epoch=0.0
trigtime=0.5*datalen+epoch

if opts.output_dir is None:
    outdir=cp.get('program', 'output-dir')

######################################################################
# Data Generation
print >> sys.stdout, '''
-------------------------------------
Beginning pmns_wrap_analysis.py:

1) Generating Data
'''

#
# Generate Signal Data
#
ts0=time.time()

print >> sys.stdout, "generating waveform..."
waveform = pmns_utils.Waveform('%s_lessvisc'%opts.waveform_name)
waveform.compute_characteristics()

te=time.time()

print >> sys.stdout, "...waveform construction took: %f sec"%(te-ts0)


# ------------------------------------------------------------------------------------
# --- Set up timing for injection: inject in center of segment
# Need to be careful.  The latest, longest template must lie within the data
# segment.   note also that the noise length needs to be a power of 2.

max_sig=100e-3
cbc_delta_t=10e-3 # std of gaussian with cbc timing error
time_prior_width=3*cbc_delta_t
max_sig_len=int(np.ceil(max_sig*16384))

# Sanity check: data segment must contain full extent of max signal
# duration, when the signal is injected in the middle of the segment
if max_sig > 0.5*datalen:
    print >> sys.stderr, "templates will lie outside data segment: extend data length"
    sys.exit()

# ------------------------------------------------------------------------------------
# Begin Loop over injections (if required)
#seed+=random.randint(0,lal.GPSTimeNow())

if opts.ninject is not None:
    # Get distance from command line if present
    ninject=opts.ninject
else:
    # Read from config file
    ninject = cp.getint('injections', 'ninject')

for i in xrange( ninject ):

    # -----------------------------
    # --- Generate injection params
    #

    # Distance
    if opts.fixed_distance is not None:
        # Get distance from command line if present
        distance=opts.fixed_distance
    else:
        # Read from config file
        if cp.get('injections', 'dist-distr')=='fixed':
            distance = cp.getfloat('injections', 'fixed-dist')

    # Sky angles
    inj_ra  = -1.0*np.pi + 2.0*np.pi*np.random.random()
    inj_dec = -0.5*np.pi + np.arccos(-1.0 + 2.0*np.random.random())
    inj_pol = 2.0*np.pi*np.random.random()
    inj_inc = 0.5*(-1.0*np.pi + 2.0*np.pi*np.random.random())
    inj_phase = 2.0*np.pi*random.random()

    # Antenna response
    det1_fp, det1_fc, det1_fav, det1_qval = antenna.response(
            epoch, inj_ra, inj_dec, inj_inc, inj_pol, 
            'radians', cp.get('analysis', 'ifo1'))

    if cp.getboolean('injections', 'inj-overhead'):
        # set the injection distance to that which yields an effective distance
        # equal to the targeted fixed-dist
        inj_distance = det1_qval*distance
    else:
        inj_distance = np.copy(distance)

    # --- End injection params

    # -----------------------------------------------
    # --- Project waveform onto these extrinsic params
    # Extrinsic parameters
    ext_params = simsig.ExtParams(distance=inj_distance, ra=inj_ra, dec=inj_dec,
            polarization=inj_pol, inclination=inj_inc, phase=inj_phase,
            geocent_peak_time=trigtime)

    # Construct the time series for these params
    waveform.make_wf_timeseries(theta=ext_params.inclination,
            phi=ext_params.phase)

    # -----------------
    #
    # Generate IFO data
    #

    ts=time.time()
    print >> sys.stdout, "generating detector responses & noise..."

    det1_data = simsig.DetData(det_site="H1",
            noise_curve=cp.get('analysis','noise-curve'), waveform=waveform,
            ext_params=ext_params, duration=datalen, seed=seed, epoch=epoch,
            f_low=flow, taper=cp.getboolean('analysis','taper_inspiral'))

    # Compute optimal SNR for injection
    det1_optSNR=pycbc.filter.sigma(det1_data.td_signal, psd=det1_data.psd,
            low_frequency_cutoff=flow, high_frequency_cutoff=0.5*srate)


    te=time.time()

    print >> sys.stdout, "...data generation took %f sec"%(te-ts)
    print >> sys.stdout, "Total elapsed time: %f sec"%(te-ts0)


    # ####################################################################
    # Write to Frame 
    # For now, write separate detectors to separate frames.  Think this is probably
    # more manageable / maintainable than multi-channel frames

    det1_frame, det1_frame_cache = write_frame(det1_data, 'H1', seed,  epoch, datalen, outdir)

    ######################################################################
    # Run Executable (e.g., LALInference)

    # Retrieve LIB executable (i.e., the *shell script* used to call LIB)
    lalinf_exec=cp.get('program','lalinf_exec')

    print >> sys.stdout, '''
    -------------------------------------

    2) Running LIB Analysis using:
    {0}
    '''.format(lalinf_exec)

    lib_outfile="LIB-PMNS_waveform-{0}_seed-{1}_distance-{2}".format(
            opts.waveform_name,
            seed, distance)

    # EXECUTE!

    ts=time.time()
    subprocess.call([lalinf_exec, "--progress", 
        "--trigtime", str(trigtime), "--seglen", cp.get('analysis','seglen'), 
        "--psdstart", str(0), "--psdlength", cp.get('analysis','psdlength'), 
        "--ifo", "[%s]"%cp.get('analysis','ifo1'), 
        "--channel", "[%s]"%cp.get('analysis','ifo1-channel'), 
        "--cache", "[%s]"%det1_frame_cache, 
        "--flow", "[%f]"%flow,  "--srate", str(srate), 
        "--fmin", cp.get('priors','fmin'), "--fmax", cp.get('priors','fmax'), 
        "--qmin", cp.get('priors','qmin'), "--qmax", cp.get('priors','qmax'), 
        "--loghrssmin", cp.get('priors','loghrssmin'), 
        "--loghrssmax", cp.get('priors','loghrssmax'), 
        "--dt", str(time_prior_width), 
        "--approx", "SineGaussianF",
        "--nlive", cp.get('analysis','nlive'), "--nmcmc", cp.get('analysis','nmcmc'), 
        "--padding", cp.get('analysis','padding'), 
        "--randomseed", str(seed),  
        "--outfile", lib_outfile+'.dat',
        "--ra", str(inj_ra), "--dec", str(inj_dec)])
    te=time.time()

    print >> sys.stdout, "LIB analysis complete!"
    print >> sys.stdout, "...LIB analysis took %f sec"%(te-ts)
    print >> sys.stdout, "Total elapsed time: %f sec"%(te-ts0)

    ######################################################################
    # Post-Process Inference Output
    print >> sys.stdout, """
    -------------------------------------

    3) Beginning post-processing

    """

    #
    # Produce posterior samples
    #
    nest2pos_exec=cp.get('program','nest2pos')
    print >> sys.stdout, """
    producing posterior samples using:
    {0}""".format(nest2pos_exec)

    ts=time.time()
    subprocess.call([nest2pos_exec, "--Nlive", cp.get('analysis','nlive'),
        "--pos", lib_outfile+'_posterior_samples.dat',
        "--headers", lib_outfile+'.dat_params.txt', lib_outfile+'.dat'])

    te=time.time()
    print >> sys.stdout, "...posterior samples produced in %f sec"%(te-ts)
    print >> sys.stdout, "Total elapsed time: %f sec"%(te-ts0)


    #
    # Read LALInference results into this code
    #
    logB, logZ, logZnoise, logLmax = np.loadtxt(lib_outfile+'.dat_B.txt')
    netSNR = np.sqrt(2.0*(logLmax - logZnoise))

    lib_params = np.loadtxt(lib_outfile+'.dat_params.txt', dtype=str)

    # XXX: for now, we only care about frequency
    freq_samps  = np.loadtxt(lib_outfile+'_posterior_samples.dat', skiprows=1,
            usecols=np.argwhere(lib_params=='frequency')[0])

    print >> sys.stdout, """
    Performing kernel density estimation and computing confidence intervals for
    frequency posterior...
    """

    ts=time.time()
    freq_axis = np.arange(cp.getfloat('priors','fmin'), cp.getfloat('priors','fmax'), 0.1)

    freq_pdf  = kde_sklearn(x=freq_samps, x_grid=freq_axis,
            bandwidth=cp.getfloat('postproc','kde-bandwidth'),
                algorithm='kd_tree') 

    # Get max-likelihood & confidence intervals
    freq_maxL, freq_low, freq_upp, freq_area = \
            compute_confints(freq_axis, freq_pdf, 0.68)
    te=time.time()

    print >> sys.stdout, """
    ...KDE and confint calculations completed in %f sec"""%(te-ts)
    print >> sys.stdout, "Total elapsed time: %f sec"%(te-ts0)

    # Save Inference results to pickle
    pickle.dump((logB, logZ, logZnoise, logLmax, netSNR, det1_optSNR, freq_samps,
        freq_axis, freq_pdf, freq_maxL, freq_low, freq_upp, freq_area),
        open("%s.pickle"%lib_outfile, 'wb'))


    ######################################################################
    # Housekeeping
    print >> sys.stdout, '''
    -------------------------------------

    4) Cleaning Up
    '''

    ts=time.time()
    # delete frame(s) data
    os.remove(det1_frame)
    os.remove(det1_frame_cache)

    # delete lalinference output ascii
    os.remove(lib_outfile+'.dat')
    os.remove(lib_outfile+'.dat_params.txt')
    os.remove(lib_outfile+'.dat_B.txt')
    os.remove(lib_outfile+'_posterior_samples.dat')
    os.remove(lib_outfile+'_posterior_samples.dat_B.txt')


    ######################################################################
    # Finalise

    # update seed
    seed+=random.randint(0,lal.GPSTimeNow())

    te=time.time()
    print >> sys.stdout, "tmp file deletion completed in %f"%(te-ts)

print >> sys.stdout, "FINISHED; total elapsed time: %f"%(te-ts0)





