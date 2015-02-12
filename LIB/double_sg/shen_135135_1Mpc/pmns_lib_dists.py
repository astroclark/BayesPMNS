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


def sort_enginefiles(filelist):
    """
    Sort the list of files based on their event number
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
            'bandwidth':0.5}

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

def plot_oneDposterior(posterior, cl_intervals, truth=None, plot1DParams=None,
        plotkde=False):
    """
    Plots a 1D histogram of the distribution of posterior samples for a given
    parameter
    """
    param = plot1DParams.keys()[0].lower()
    pos_samps = posterior[param].samples

    histbins = plot1DParams.values()[0]

    fig, ax = pl.subplots(figsize=(6,4))#, dpi=200)

    if plotkde==False:
        # Plot histogram
        (n, bins, patches) = ax.hist(pos_samps, histbins, normed='true',
                histtype='stepfilled', facecolor='grey')
        ax.set_ylim(0, 1.05*max(n))
    else:

        bw = 1.06*np.std(pos_samps)* len(pos_samps)**(-1./5)
        x_grid = np.linspace(0.9*min(pos_samps),1.1*max(pos_samps),1000)

        pdf = kde_sklearn(x=np.concatenate(pos_samps), x_grid=x_grid, bandwidth=bw)

        ax.plot(x_grid,pdf,color='grey')

    ax.set_xlim(0.95*min(cl_intervals),1.05*max(cl_intervals))

    # Show injected value
    if truth is not None:
        ax.axvline(truth, color='r', label='Target %s'%param)

    # Show the median and 90% confidence interval
    ax.axvline(cl_intervals[0], color='k', linestyle='--')
    ax.axvline(cl_intervals[1], color='k', linestyle='--', label=r'$\alpha=0.9$')
    ax.axvline(posterior[param].median, color='k', linestyle='-',
            label=r'median')

    if param in ['frequency', 'bandwidth']:
        ax.set_xlabel(param+' [Hz]')
    else:
        ax.set_xlabel(param)

    ax.set_ylabel('Probability Density')
    ax.minorticks_on()
    ax.legend()

    pl.tight_layout()

    return fig

def plot_corner(posterior,levels,parvals=None):
    """
    Local version of a corner plot to allow bespoke truth values
    """
    if parvals==None:
        print >> sys.stderr, "need param names and values"
    parnames = parvals.keys()

    parnames=filter(lambda x: x in posterior.names, parnames)
    truths=[parvals[p] for p in parnames]

    data = np.hstack([posterior[p].samples for p in parnames])

    #extents=[(0,1)]*len(parnames)
    #extents[parnames=='frequency'] = 
    #trifig=triangle.corner(data, labels=parnames, truths=truths,
    #        quantiles=levels, truth_color='r', extents=extents)
    trifig=triangle.corner(data, labels=parnames, truths=truths,
            quantiles=levels, truth_color='r')

    return trifig


# -------------------------------
# Load results

resultsdir=sys.argv[1]
waveform_name='shen_135135'#sys.argv[2]
Nlive=512#sys.argv[3]

outdir=resultsdir+'_allout'
currentdir=os.path.join(outdir,'summaryfigs')
if not os.path.isdir(currentdir):
    os.makedirs(currentdir) 
else:
    print >> sys.stdout, \
            "warning: %s exists, results will be overwritten"%outdir

# --- Construct the injected waveform
waveform = pmns_utils.Waveform('%s_lessvisc'%waveform_name)
waveform.compute_characteristics()


# --- Identify files
ifospattern="V1H1L1"
engineglobpattern="lalinferencenest"
sampglobpattern="posterior"

Bfiles = glob.glob('%s/posterior_samples/%s_%s*.dat_B.txt'%(
    resultsdir, sampglobpattern, ifospattern ) )

snrfiles  = glob.glob('%s/engine/%s*-%s-*_snr.txt'%(
    resultsdir, engineglobpattern, ifospattern ) )

sampfiles = glob.glob('%s/posterior_samples/%s_%s*.dat'%(
    resultsdir, sampglobpattern, ifospattern ) )


# Sort the different file lists in ascending event number
# XXX: DANGER this may be prone to breaking with changes to filenames
snrfiles=sort_enginefiles(snrfiles)
Bfiles=sort_possampfiles(Bfiles)
sampfiles=sort_possampfiles(sampfiles)

if len(snrfiles)==0:
    print >> sys.stderr, "Failed to find SNR files; check glob patterns"
    sys.exit()
if len(Bfiles)==0:
    print >> sys.stderr, "Failed to find Bayes factor files; check glob patterns"
    sys.exit()
if len(sampfiles)==0:
    print >> sys.stderr, "Failed to find posterior samples; check glob patterns"
    sys.exit()

# =========================================================================
#
# BPPU tools
#
# Taken from cbcBayesBurstPostProc.py, move into a function?

currentfile=sampfiles[0]
headerfile=snrfiles[0].replace('_snr','_params')

# Output dir for this injection (top-level directory + event number and gps time
# taken from posterior samples filename)
currentdir=os.path.join(outdir,currentfile.split('/')[-1].split('_')[-1].replace('.dat',''))
if not os.path.isdir(currentdir):
    os.makedirs(currentdir)

# Create PE parser object
peparser = bppu.PEOutputParser('common')
resultsObj = peparser.parse(open(currentfile,'r'), info=[headerfile, None])

# Read Bayes and SNR files
bfile=open(Bfiles[0], 'r')
BSN=bfile.read()
bfile.close()

snrfactor=snrfiles[0]
snrstring=""
snrfile=open(snrfactor,'r')
snrs=snrfile.readlines()
snrfile.close()
for snr in snrs:
    if snr=="\n":
        continue
    snrstring=snrstring +" "+str(snr[0:-1])+" ,"
snrstring=snrstring[0:-1]

# Create Posterior Sample object (shouldn't matter whether we go with Posterior or BurstPosterior)
pos = bppu.Posterior(resultsObj)

oneDMenu, twoDMenu, binSizes = oneD_bin_params()

# Get true values:
truevals={}
for item in oneDMenu:
    if item=='frequency':
        truevals[item]=waveform.fpeak
    else:
        truevals[item]=None

# TODO:
#   2) confidence intervals marked on 1D posteriors: nicer figures
#   3) Loop over injections
#   4) webpage for injection population
#   5) plot injected / recovered waveforms

# ----------------------------------------------------------
# -- 1D Posterior results (see cbcBayesBurstPostProc.py:733)
for par_name in ['frequency']:#oneDMenu:
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
    oneDPDFParams={par_name:50}
    fig = plot_oneDposterior(pos, cl_intervals[1], truth=truevals[par_name],
            plot1DParams=oneDPDFParams, plotkde=True)
    
    figname=par_name+'.png'
    oneDplotPath=os.path.join(currentdir,figname)
    fig.savefig(oneDplotPath)

# --- Corner plot
#corner_fig = plot_corner(pos, [0.67, 0.9, 0.95], parnames=oneDMenu, truevals)
corner_fig = plot_corner(pos, [0.1, 0.5, 0.9], parvals=truevals)

pl.show()

# End
# =========================================================================

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


