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
from optparse import OptionParser
import ConfigParser
import random
import string
import time
import glob

import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as pl

import lal
import pmns_utils
import pmns_simsig as simsig

from pylal import Fr
from glue import lal as gluelal

def parser():

    # --- Command line input
    parser = OptionParser()
    parser.add_option("-c", "--config-file", default="test.ini", type=str)
    parser.add_option("-w", "--waveform-name", default="dd2_135135", type=str)
    parser.add_option("-D", "--fixed-distance", type=float, default=5.0)
    parser.add_option("-S", "--init-seed", type=int, default=101)
    parser.add_option("-L", "--data-length", type=int, default=None)
    parser.add_option("-N", "--num-runs", type=int, default=1)
    parser.add_option("-o", "--output-dir", type=str, default=None)

    #parser.add_option("-Dmin", "--min-distance")
    #parser.add_option("-Dmax", "--max-distance")

    (opts,args) = parser.parse_args()

    # --- Read config file
    cp = ConfigParser.ConfigParser()
    cp.read(opts.config_file)

    # XXX: retrieve the executable we'll run
    print >> sys.stdout, " %s"%cp.get('analysis', 'executable')

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

    return cache_file

#####################################################################
# Input
USAGE='''%prog [options] [args]
Run external analysis programs on internally generated post-merger injections
with Gaussian noise.
'''

opts, args, cp = parser()

######################################################################
# Generate Injection Population

#
# --- extract data from parsers
#
seed=opts.init_seed
#seed+=random.randint(0,lal.GPSTimeNow())
distance=opts.fixed_distance

epoch=0.0

if opts.data_length is None:
    datalen=cp.getfloat('analysis', 'data-length')
else:
    datalen=opts.data_length

if opts.output_dir is None:
    outdir=cp.get('analysis', 'output-dir')

print >> sys.stdout, "using datalen:", datalen
print >> sys.stdout, "writing to:", outdir

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

print 'generating waveform...'
waveform = pmns_utils.Waveform('%s_lessvisc'%opts.waveform_name)
waveform.compute_characteristics()

# XXX: LOOP OVER INJECTIONS WILL BEGIN HERE

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

# Extrinsic parameters
ext_params = simsig.ExtParams(distance=distance, ra=0.0, dec=0.0,
        polarization=0.0, inclination=0.0, phase=0.0,
        geocent_peak_time=0.5*datalen+epoch)

# Construct the time series for these params
waveform.make_wf_timeseries(theta=ext_params.inclination,
        phi=ext_params.phase)

#
# Generate IFO data
#

print >> sys.stdout, "generating detector responses & noise..."

det1_data = simsig.DetData(det_site="H1", noise_curve='aLIGO', waveform=waveform,
        ext_params=ext_params, duration=datalen, seed=seed, epoch=epoch,
        f_low=10.0)

# ####################################################################
# Write to Frame 
# For now, write separate detectors to separate frames.  Think this is probably
# more manageable / maintainable than multi-channel frames

det1_frame_cache = write_frame(det1_data, 'H1', seed,  epoch, datalen, outdir)

######################################################################
# Run Executable (e.g., LALInference)
print >> sys.stdout, '''
-------------------------------------

2) Running Analysis
{0}
'''.format(cp.get('analysis', 'executable'))


######################################################################
# Housekeeping
print >> sys.stdout, '''
-------------------------------------

3) Cleaning Up
'''

# delete frame(s)

# standardise executable output

# update seed
seed+=random.randint(0,lal.GPSTimeNow())

######################################################################
# Plot data
if 0:
    pl.figure()
    pl.semilogy(det1_data.td_response.to_frequencyseries().sample_frequencies,
            abs(det1_data.td_response.to_frequencyseries()))
    pl.semilogy(det1_data.td_signal.to_frequencyseries().sample_frequencies,
            abs(det1_data.td_signal.to_frequencyseries()),'r')
    pl.xlim(1000,5000)
    pl.ylim(1e-25,1e-19)

    pl.figure()
    pl.plot(det1_data.td_response.sample_times, det1_data.td_response.data)
    pl.plot(det1_data.td_signal.sample_times, det1_data.td_signal.data,'r')

    pl.show()




