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
import glob
import numpy as np
from itertools import combinations
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

import lalsimulation as lalsim
from pylal import bayespputils as bppu
import pmns_utils
import triangle

import pycbc.types
import pycbc.filter
from pycbc.psd import aLIGOZeroDetHighPower

from sklearn.neighbors.kde import KernelDensity

def stacc(pos, param, truth):
    """
    Compute the standard accuracy statistic
    (see bayespputils.py:422)
    """
    if truth is None:
        return None
    else:
        return np.sqrt(np.mean(pos[param].samples - truth)**2)

def acc(pos, param, truth):
    """
    Compute the accuracy of the posterior measurement
    """
    if truth is None:
        return [None]*3
    else:
        delta_maxP=abs(truth - pos.maxP[1][param])
        delta_mean=abs(truth - pos.means[param])
        delta_median=abs(truth - pos.medians[param])
        return [delta_maxP, delta_mean, delta_median]

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def get_extent(posterior,name,truevals):
    """
    Get the extent of the axes for this parameter
    """
    #parnames = truevals.keys()
    #parnames=filter(lambda x: x in posterior.names, parnames)

    extents=[]

    # If a parameter has a 'true' value, make sure it's visible on the plot by
    # adjusting the extent of the panel to the max/min sample +/-
    # 0.5*standardaccuarcy

    if truevals[name]:
        if truevals[name]>=posterior[name].samples.max():
            upper=truevals[name] + 0.5*stacc(posterior,name,truevals[name])
            lower=posterior[name].samples.min()
        elif truevals[name]<=posterior[name].samples.min():
            lower=truevals[name] - 0.5*stacc(posterior,name,truevals[name])
            upper=posterior[name].samples.max()
        else:
            lower=posterior[name].samples.min()
            upper=posterior[name].samples.max()
    else:
        lower=posterior[name].samples.min()
        upper=posterior[name].samples.max()

    return (lower,upper)

def plot_corner(posterior,percentiles,parvals=None):
    """
    Local version of a corner plot to allow bespoke truth values

    posterior: posterior object
    percentiles: percentiles to draw on 1D histograms
    parvals: dictionary of parameters with true values.  These parameters are
    used to select which params in posterior to use for the triangle plot, as
    well as to draw the target values

    """
    if parvals==None:
        print >> sys.stderr, "need param names and values"
    parnames = parvals.keys()

    parnames=filter(lambda x: x in posterior.names, parnames)
    truths=[parvals[p] for p in parnames]

    data = np.hstack([posterior[p].samples for p in parnames])
    extents = [get_extent(posterior,name,parvals) for name in parnames]

    trifig=triangle.corner(data, labels=parnames, truths=truths,
            quantiles=percentiles, truth_color='r', extents=extents)

    return trifig

def oneD_bin_params():
    """
    Return the list of parameters we care about and a dictionary for greedy
    binning
    """
    polParams=['psi','polarisation','polarization']
    skyParams=['ra','rightascension','declination','dec']
    timeParams=['time']
    ellParams=['alpha']
    burstParams=['frequency','loghrss','quality','hrss','duration','bandwidth']

    #oneDMenu = polParams + ellParams + burstParams
    oneDMenu = burstParams

    twoDMenu=[]
    for b1,b2 in combinations(burstParams,2):
        twoDMenu.append([b1,b2])


    binSizes = {'time':1e-4, 'ra':0.05, 'dec':0.05, 'polarisation':0.04,
            'rightascension':0.05, 'declination':0.05, 'loghrss':0.01,
            'frequency':0.5, 'quality':0.05, 'phase':0.1, 'phi_orb':0.1,
            'psi':0.04, 'polarization':0.04, 'alpha':0.01, 'duration':0.0001,
            'bandwidth':0.5, 'hrss':1e-23}

    return oneDMenu, twoDMenu, binSizes

def plot_oneDposterior(posterior, param, cl_intervals, 
        parvals=None, plotkde=False):
    """
    Plots a 1D histogram of the distribution of posterior samples for a given
    parameter
    """
    pos_samps = posterior[param].samples


    fig, ax = pl.subplots(figsize=(6,4))#, dpi=200)

    if plotkde==False:

        # Plot histogram
        histbinswidth = 3.5*posterior[param].stdev / len(pos_samps)**(1./3)
        histbins = np.arange(pos_samps.min(), pos_samps.max(), histbinswidth)
        (n, bins, patches) = ax.hist(pos_samps, histbins, normed='true',
                histtype='step', facecolor='grey', color='k')
        ax.set_ylim(0, 1.05*max(n))
    else:
        bw = 1.06*np.std(pos_samps)* len(pos_samps)**(-1./5)
        x_grid = np.linspace(0.9*min(pos_samps),1.1*max(pos_samps),1000)
        pdf = kde_sklearn(x=np.concatenate(pos_samps), x_grid=x_grid, bandwidth=bw)

        ax.plot(x_grid,pdf,color='grey')

    # Show injected value
    if parvals[param] is not None:
        ax.axvline(parvals[param], color='r', label='Target %s'%param)

    # Show the median and 90% confidence interval
    try:
        ax.axvline(cl_intervals[0], color='k', linestyle='--')
        ax.axvline(cl_intervals[1], color='k', linestyle='--', label=r'$\alpha=0.9$')
        ax.axvline(posterior[param].median, color='k', linestyle='-',
                label=r'median')
    except RuntimeError:
        pass

    # set axis limits
    ax.set_xlim(get_extent(posterior,param,parvals))

    if param in ['frequency', 'bandwidth']:
        ax.set_xlabel(param+' [Hz]')
    elif param in ['duration']:
        ax.set_xlabel(param+' [s]')
    else:
        ax.set_xlabel(param)

    ax.set_ylabel('Probability Density')
    ax.minorticks_on()
    ax.legend()
    pl.tight_layout()

    return fig

def single_injection_results(outdir, posterior_file, bsn_file, snr_file,
        waveform):

    """
    Characterise the results, including producing plots, for a single injection
    """

    # Mostly taken from cbcBayesBurstPostProc.py

    headerfile=snr_file.replace('_snr','_params')

    # Output dir for this injection (top-level directory + event number and gps time
    # taken from posterior samples filename)
    currentdir=os.path.join(outdir,posterior_file.split('/')[-1].split('_')[-1].replace('.dat',''))
    if not os.path.isdir(currentdir):
        os.makedirs(currentdir)

    # Create PE parser object
    peparser = bppu.PEOutputParser('common')
    resultsObj = peparser.parse(open(posterior_file,'r'), info=[headerfile, None])

    # Read Bayes and SNR files and copy them into this directory in case it's
    # useful later
    bfile=open(bsn_file, 'r')
    BSN=bfile.read()
    bfile.close()
    BSN=float(BSN.split()[0])


    snrstring=""
    snrf=open(snr_file,'r')
    snrs=snrf.readlines()
    snrf.close()
    for snr in snrs:
        if snr=="\n":
            continue
        snrstring=snrstring +" "+str(snr[0:-1])+" ,"
    snrstring=snrstring[0:-1]

    # Just return the final (usually network) snr
    SNR=float(snrstring.split()[-1])

    # Archive files to this directory
    shutil.copyfile(posterior_file, os.path.join(currentdir,posterior_file.split('/')[-1]))
    shutil.copyfile(bsn_file, os.path.join(currentdir,bsn_file.split('/')[-1]))
    shutil.copyfile(snr_file, os.path.join(currentdir,snr_file.split('/')[-1]))

    # Create Posterior Sample object (shouldn't matter whether we go with Posterior or BurstPosterior)
    pos = bppu.Posterior(resultsObj)

    # Add in derived parameters
    pos = add_derived_params(pos)

    oneDMenu, twoDMenu, binSizes = oneD_bin_params()

    # Get true values:
    truevals={}
    for item in oneDMenu:
        if item=='frequency':
            truevals[item]=waveform.fpeak
        else:
            truevals[item]=None

    # Make sure we have quality AND bandwidth samples

    # TODO:
    #   3) Loop over injections
    #   4) webpage for injection population
    #   5) plot injected / recovered waveforms

    # ----------------------------------------------------------

    # --- Corner plot
    corner_fig = plot_corner(pos, [0.05, 0.5, 0.95], parvals=truevals)
    corner_fig.savefig(os.path.join(currentdir,'corner.png'))

    # --- 1D Posterior results (see cbcBayesBurstPostProc.py:733)

    # Dictionaries to contain confidence intervals and standard accuracies
    cl_intervals_allparams={}
    staccs_allparams={}
    accs_allparams={}
    for par_name in oneDMenu:
        print >> sys.stdout, "Producing 1D posteriors for %s"%par_name

        # Get bin params
        try: 
            pos[par_name.lower()]
        except KeyError:
            print "No posterior samples for %s, skipping binning."%par_name
            continue
        try: 
            par_bin=binSizes[par_name]
        except KeyError:
            print "Bin size is not set for %s, skipping binning."%par_name
            continue
        binParams = {par_name:par_bin}

        # --- Statistics from this posterior
        toppoints, injection_cl, reses, injection_area, cl_intervals = \
                bppu.greedy_bin_one_param(pos, binParams, [0.67, 0.9, 0.95])

        # add conf intervals dictionary
        cl_intervals_allparams[par_name]=cl_intervals[1]

        # add standard accuracy to dictionary
        staccs_allparams[par_name]=stacc(pos,par_name,truevals[par_name])
        accs_allparams[par_name]=acc(pos,par_name,truevals[par_name])


        # --- Plotting
        fig = plot_oneDposterior(pos, par_name, cl_intervals[1],
                truevals, plotkde=False)
        
        figname=par_name+'.png'
        oneDplotPath=os.path.join(currentdir,figname)
        fig.savefig(oneDplotPath)

    pl.close('all')
    return currentdir, BSN, SNR, pos, cl_intervals_allparams, \
            staccs_allparams, accs_allparams, pos

def add_derived_params(posterior):
    """
    Derive posterior samples for quality, bandwidth and duration

    See LALSimBurst.c:800
    """

    if 'frequency' not in posterior.names:
        print >> sys.stderr, "no frequency parameter"
        sys.exit()

    if ('bandwidth' not in posterior.names) and ('quality' in posterior.names):

        bandwidth_samps = posterior['frequency'].samples /\
                posterior['quality'].samples  
        bandwidthPDF = bppu.PosteriorOneDPDF(name='bandwidth',
                posterior_samples=bandwidth_samps)
        posterior.append(bandwidthPDF)

    if ('quality' not in posterior.names) and ('bandwidth' in posterior.names):
        quality_samps = posterior['frequency'].samples /\
                posterior['bandwidth'].samples
        qualityPDF = bppu.PosteriorOneDPDF(name='quality',
                posterior_samples=quality_samps)
        posterior.append(qualityPDF)

    # By now we have bandwidth and quality
    if ('duration' not in posterior.names):
        duration_samps =  1.0/(np.sqrt(2)*np.pi*posterior['bandwidth'].samples)
        durationPDF = bppu.PosteriorOneDPDF(name='duration',
                posterior_samples=duration_samps)
        posterior.append(durationPDF)


    return posterior

# -----------------------------------------------------------------------------------------------------------

# **** MAIN **** #
posterior_file = sys.argv[1]
bsn_file = sys.argv[2]
snr_file = sys.argv[3]

waveform=pmns_utils.Waveform("shen_135135_lessvisc")
waveform.compute_characteristics()
waveform.reproject_waveform()

oneDMenu, twoDMenu, binSizes = oneD_bin_params()

# Get true values:
truevals={}
for item in oneDMenu:
    if item=='frequency':
        truevals[item]=waveform.fpeak
    else:
        truevals[item]=None

# Analyse

outputdirectory="./"
injdir, BSN, SNR, thisposterior, cl_intervals, staccs, accs, poss \
        = single_injection_results(outputdirectory, posterior_file, bsn_file,
                snr_file, waveform)

# plot
fig = plot_oneDposterior(poss, 'frequency', cl_intervals['frequency'],
        truevals, plotkde=True)

pl.show()





