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

import matplotlib
from matplotlib import pyplot as pl

from pylal import bayespputils as bppu
import pmns_utils
import triangle

from sklearn.neighbors.kde import KernelDensity

fig_width_pt = 246  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (2.236-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]


matplotlib.rcParams.update(                                                                                       
        {'axes.labelsize': 12,
        'text.fontsize':   12,                                                                                     
        'legend.fontsize': 12,
        'xtick.labelsize': 12,                                                                                     
        'ytick.labelsize': 12,                                                                                     
        'text.usetex': True,                                                                                      
        'figure.figsize': fig_size,                                                                               
        'font.family': "serif",                                                                                   
        'font.serif': ["Times"]
        })  

matplotlib.rcParams.update(                                                                                       
        {'savefig1.dpi': 200,                                                                                     
        'xtick.major.size':8,
        'xtick.minor.size':4,
        'ytick.major.size':8,                                                                                     
        'ytick.minor.size':4
        })  
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"] 

def write_results_page(outdir,injection_dirs,posteriors):
    f = open(os.path.join(outdir, "population_summary.html"), 'w')
    htmlstr="""
    <html>
    <h1>Results Summary for {outdir}</h1>
    <hr>
    <p>
        <ul>
            <li></li>
        </ul>
    </p>

    <h2>Summary Plots</h2>
    <table>
        <tr>
            <td><h3>SNRs</h3></td>
            <td><h3>Bayes Factors</h3></td>
        </tr>
        <tr>
            <td><img src="logB_PDF.png"></td>
            <td><img src="snr_PDF.png"></h3></td>
        </tr>
    </table>

    <h2>Per Injection Results</h2>
    """.format(outdir=outdir)

    for posterior, injection_dir in zip(posteriors, injection_dirs):
        injection=injection_dir.split('/')[-1]


        htmlstr+="""
        <hr>
        <h2>{injection}</h2>
        
        <h3>Summary Statistics</h3>
            <table>
                <tr align="center">
                    <td></td>
                </tr>
                <tr>
                    <td><a href="{injection}">All results</a></td>
                </tr>
            </table>
            <img width=800px src="{injection}/corner.png">
        """.format(injection=injection)

    f.write(htmlstr)
    f.close()


def sort_enginefiles(filelist):
    """
    elSort the list of files based on their event number
    """
    eventnums=[]
    for filename in filelist:
        eventnums.append(int(filename[filename.find('lalinferencenest'):].split('-')[1]))
    return list(np.array(filelist)[np.argsort(eventnums)])

def sort_possampfiles(filelist):
    """
    Sort the list of files based on their event number
    """
    eventnums=[]
    for filename in filelist:
        eventnums.append(int(filename[filename.find('posterior_V1H1L1'):].split('-')[1].split('.')[0]))
    return list(np.array(filelist)[np.argsort(eventnums)])

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

def stacc(pos, param, truth):
    """
    Compute the standard accuracy statistic
    (see bayespputils.py:422)
    """
    return np.sqrt(np.mean(pos[param].samples - truth)**2)

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

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

    # set axis limits
    ax.set_xlim(get_extent(posterior,param,parvals))

    # Show injected value
    if parvals[param] is not None:
        ax.axvline(parvals[param], color='r', label='Target %s'%param)

    # Show the median and 90% confidence interval
    ax.axvline(cl_intervals[0], color='k', linestyle='--')
    ax.axvline(cl_intervals[1], color='k', linestyle='--', label=r'$\alpha=0.9$')
    ax.axvline(posterior[param].median, color='k', linestyle='-',
            label=r'median')

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

def get_extent(posterior,name,parvals):
    """
    Get the extent of the axes for this parameter
    """
    #parnames = parvals.keys()
    #parnames=filter(lambda x: x in posterior.names, parnames)

    extents=[]

    # If a parameter has a 'true' value, make sure it's visible on the plot by
    # adjusting the extent of the panel to the max/min sample +/-
    # 0.5*standardaccuarcy

    if parvals[name]:
        if parvals[name]>=posterior[name].samples.max():
            upper=parvals[name] + 0.5*stacc(posterior,name,parvals[name])
            lower=posterior[name].samples.min()
        else:
            lower=parvals[name] - 0.5*stacc(posterior,name,parvals[name])
            upper=posterior[name].samples.max()
    else:
        lower=posterior[name].samples.min()
        upper=posterior[name].samples.max()

    return (lower,upper)

def add_derived_params(posterior):
    """
    Derive posterior samples for quality, bandwidth and duration

    See LALSimBurst.c:800
    """

    if 'frequency' not in posterior.names:
        print >> sys.stderr, "no frequency parameter"
        sys.exit()

    if ('quality' not in posterior.names) and ('bandwidth' in posterior.names):
        quality_samps = posterior['frequency'].samples /\
                posterior['bandwidth'].samples
        qualityPDF = bppu.PosteriorOneDPDF(name='quality',
                posterior_samples=quality_samps)
        posterior.append(qualityPDF)

    if ('duration' not in posterior.names) and ('bandwidth' in posterior.names):
        duration_samps =  1.0/(np.sqrt(2)*np.pi*posterior['bandwidth'].samples)
        durationPDF = bppu.PosteriorOneDPDF(name='duration',
                posterior_samples=duration_samps)
        posterior.append(durationPDF)

    if ('duration' not in posterior.names) and ('quality' in posterior.names):
        duration_samps =  1.0/(np.sqrt(2)*np.pi*posterior['quality'].samples)
        durationPDF = bppu.PosteriorOneDPDF(name='duration',
                posterior_samples=duration_samps)
        posterior.append(durationPDF)

    if ('bandwidth' not in posterior.names) and ('duration' in posterior.names):
        bandwidth_samps =  1.0/(np.sqrt(2)*np.pi*posterior['duration'].samples)
        bandwidthPDF = bppu.PosteriorOneDPDF(name='bandwidth',
                posterior_samples=bandwidth_samps)
        posterior.append(bandwidthPDF)

    if ('bandwidth' not in posterior.names) and ('quality' in posterior.names):
        bandwidth_samps = posterior['frequency'].samples /\
                posterior['quality'].samples  
        bandwidthPDF = bppu.PosteriorOneDPDF(name='bandwidth',
                posterior_samples=bandwidth_samps)
        posterior.append(bandwidthPDF)

    return posterior

# =========================================================================
#
# BPPU tools
#
def single_injection_results(outdir, posterior_file, bsn_file, snr_file):

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

    # Read Bayes and SNR files
    bfile=open(bsn_file, 'r')
    BSN=bfile.read()
    bfile.close()

    snrstring=""
    snrf=open(snr_file,'r')
    snrs=snrf.readlines()
    snrf.close()
    for snr in snrs:
        if snr=="\n":
            continue
        snrstring=snrstring +" "+str(snr[0:-1])+" ,"
    snrstring=snrstring[0:-1]

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
    corner_fig = plot_corner(pos, [0.1, 0.5, 0.9], parvals=truevals)
    corner_fig.savefig(os.path.join(currentdir,'corner.png'))

    # --- 1D Posterior results (see cbcBayesBurstPostProc.py:733)
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

        # --- Plotting
        fig = plot_oneDposterior(pos, par_name, cl_intervals[1],
                truevals, plotkde=False)
        
        figname=par_name+'.png'
        oneDplotPath=os.path.join(currentdir,figname)
        fig.savefig(oneDplotPath)


    pl.close('all')
    # Things to return:
    # results directories
    # cl_intervals
    # stacc
    # median, mean, maxL
    # snr, Bayes factor
    return currentdir, pos

# End
# =========================================================================

# ********************************************************************************
# MAIN SCRIPT

# -------------------------------
# Load results

resultsdir=sys.argv[1]
waveform_name='shen_135135'#sys.argv[2]
Nlive=512#sys.argv[3]

outputdirectory=resultsdir+'_allout'
if not os.path.isdir(outputdirectory):
    os.makedirs(outputdirectory) 
else:
    print >> sys.stdout, \
            "warning: %s exists, results will be overwritten"%outputdirectory
currentdir=os.path.join(outputdirectory,'summaryfigs')

# --- Construct the injected waveform
waveform = pmns_utils.Waveform('%s_lessvisc'%waveform_name)
waveform.compute_characteristics()


# --- Identify files
ifospattern="V1H1L1"
engineglobpattern="lalinferencenest"
sampglobpattern="posterior"

BSNfiles = glob.glob('%s/posterior_samples/%s_%s*.dat_B.txt'%(
    resultsdir, sampglobpattern, ifospattern ) )

snrfiles  = glob.glob('%s/engine/%s*-%s-*_snr.txt'%(
    resultsdir, engineglobpattern, ifospattern ) )

sampfiles = glob.glob('%s/posterior_samples/%s_%s*.dat'%(
    resultsdir, sampglobpattern, ifospattern ) )


# Sort the different file lists in ascending event number
# XXX: DANGER this may be prone to breaking with changes to filenames
snrfiles=sort_enginefiles(snrfiles)
BSNfiles=sort_possampfiles(BSNfiles)
sampfiles=sort_possampfiles(sampfiles)

if len(snrfiles)==0:
    print >> sys.stderr, "Failed to find SNR files; check glob patterns"
    sys.exit()
if len(BSNfiles)==0:
    print >> sys.stderr, "Failed to find Bayes factor files; check glob patterns"
    sys.exit()
if len(sampfiles)==0:
    print >> sys.stderr, "Failed to find posterior samples; check glob patterns"
    sys.exit()

# Set up list of injection directories and posterior objects to point to in html
injection_dirs = []
allposteriors = []
for fileidx, files in enumerate(zip(sampfiles,BSNfiles,snrfiles)):

    posterior_file = files[0]
    bsn_file = files[1]
    snr_file = files[2]

    injdir, thisposterior = single_injection_results(outputdirectory,
            posterior_file, bsn_file, snr_file)

    injection_dirs.append(injdir)
    allposteriors.append(thisposterior)


# -------------------------------------------------------
# Construct summary plots and statistics

# -------------------------------------------------------
# Write HTML summary
    write_results_page(outputdirectory, injection_dirs, allposteriors)

    sys.exit()

# --- Preallocation
logBs   = np.zeros(len(datafiles))
netSNRs = np.zeros(len(datafiles))

freq_pdfs = []
freq_maxL = np.zeros(len(datafiles))
freq_low  = np.zeros(len(datafiles))
freq_upp  = np.zeros(len(datafiles))
freq_area = np.zeros(len(datafiles))

# --- Load each file
for d, datafile in enumerate(datafiles):
    print >> sys.stdout, "Loading %d of %d (%s)"%(d, len(datafiles), datafile)



if 0:
    # ----------------------------------------------------------
    # -- 2D Posterior results (see cbcBayesBurstPostProc.py:733)

    for par1_name,par2_name in twoDMenu:
        print >> sys.stdout, "Producing 2D posteriors for %s-%s"%(par1_name, par2_name)

        # Get bin params
        par1_name=par1_name.lower()
        par2_name=par2_name.lower()
        try: 
            pos[par1_name.lower()]
        except KeyError:
            print "No posterior samples for %s, skipping binning."%par1_name
            continue
        try: 
            pos[par2_name.lower()]
        except KeyError:
            print "No posterior samples for %s, skipping binning."%par2_name
            continue

        # Bin sizes
        try: 
            par1_bin=binSizes[par1_name]
        except KeyError:
            print "Bin size is not set for %s, skipping %s/%s binning."%(\
                par1_name, par1_name, par2_name)
            continue
        try: 
            par2_bin=binSizes[par2_name]
        except KeyError:
            print "Bin size is not set for %s, skipping %s/%s binning."%(\
                par2_name, par1_name, par2_name)
            continue

        #print "Binning %s-%s to determine confidence levels ..."%(par1_name,par2_name)
        #Form greedy binning input structure
        twoDParams={par1_name:par1_bin,par2_name:par2_bin}

        #Greedy bin the posterior samples
        toppoints, injection_cl, reses, injection_area = \
            bppu.greedy_bin_two_param(pos, twoDParams, [0.67, 0.9, 0.95])


        # --- Plotting
        greedy2ContourPlot = bppu.plot_two_param_kde_greedy_levels({'Result':pos},
                twoDParams, [0.67,0.9,0.95], {'Result':'k'})

        greedy2contourpath = os.path.join(currentdir,
                '%s-%s_greedy2contour.png'%(par1_name, par2_name))
        greedy2ContourPlot.savefig(greedy2contourpath)
        
        greedy2HistFig = bppu.plot_two_param_greedy_bins_hist(pos, twoDParams,
                [0.67, 0.9, 0.95])
        greedy2histpath = os.path.join(currentdir,'% s-%s_greedy2.png'%(par1_name,
            par2_name))
        greedy2HistFig.savefig(greedy2histpath)

        sys.exit()


