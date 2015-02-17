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
from matplotlib import pyplot as pl

from pylal import bayespputils as bppu
import pmns_utils
import triangle

import pycbc.types
import pycbc.filter
from pycbc.psd import aLIGOZeroDetHighPower

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

def write_results_page(outdir, injection_dirs, posteriors, all_cl_intervals,
        all_staccs, accs, all_bsn, all_snr, truevals):
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
    """.format(outdir=outdir)

    htmlstr+="""
    <h3>Parameter Estimation</h3>
    """

    # -----------------------
    # Parameters vs inj SNR
    htmlstr+="""
    <h3>Recovered Parameters vs Injected Optimal SNR</h3>
    <p>
        <ul>
            <li>Injected value shown as red-dashed line, where available</li>
            <li>Markers:  maximum a posteriori measurement</li>
            <li>Black lines: 90% confidence interval</li>
        </ul>
    </p>
    <table>
        <tr>
    """
    for param in truevals.keys():
        # Only add the plot if the trueval name is in posterior keys - hacky way
        # to just make sure we don't try to include exponentiated-away values
        # like loghrss
        if param in posteriors[0].maxP[1].keys():
            htmlstr+="""
            <td><img width=350px src="{param}vsinjSNR.png"</td>
            """.format(param=param)
    htmlstr+="""
        </tr>
    </table>
    """


    # -----------------
    # Parameters vs logB
    htmlstr+="""
    <h3>Recovered Parameters vs Bayes factor</h3>
    <p>
        <ul>
            <li>Injected value shown as red-dashed line, where available</li>
            <li>Markers:  maximum a posteriori measurement</li>
            <li>Black lines: 90% confidence interval</li>
        </ul>
    </p>
    <table>
        <tr>
    """
    for param in truevals.keys():
        # Only add the plot if the trueval name is in posterior keys - hacky way
        # to just make sure we don't try to include exponentiated-away values
        # like loghrss
        if param in posteriors[0].maxP[1].keys():
            htmlstr+="""
            <td><img width=350px src="{param}vslogB.png"</td>
            """.format(param=param)
    htmlstr+="""
        </tr>
    </table>
    """

    # -----------------
    # Parameters 1D hists
    htmlstr+="""
    <h3>Recovered Parameter Histograms</h3>
    <p>
        <ul>
            <li>Injected value shown as red-dashed line, where available</li>
        </ul>
    </p>
    <table>
        <tr>
    """
    for param in truevals.keys():
        # Only add the plot if the trueval name is in posterior keys - hacky way
        # to just make sure we don't try to include exponentiated-away values
        # like loghrss
        if param in posteriors[0].maxP[1].keys():
            htmlstr+="""
            <td><img width=350px src="{param}.png"</td>
            """.format(param=param)
    htmlstr+="""
        </tr>
    </table>
    """

    # -----------------
    # Accuracy
    htmlstr+="""
    <h3>Parameter Accuracy</h3>
    <p>
        <ul>
            <li>Accuracy = max posterior - injected / target value</li>
            <li>For posterior samples x: Standard Accuracy = sqrt[mean(x-x_true)^2]</li>
        </ul>
    </p>
    <table>
        <tr>
    """
    for param in truevals.keys():
        if truevals[param] is None: continue
        htmlstr+="""
        <td><img width=350px src="{param}accvsinjSNR.png"</td>
        <td><img width=350px src="{param}accvslogB.png"</td>
        <td><img width=350px src="{param}acc.png"</td>
        """.format(param=param)
    htmlstr+="""
        </tr>
    </table>
    """

    htmlstr+="""
    <table>
        <tr>
    """
    for param in truevals.keys():
        if truevals[param] is None: continue
        htmlstr+="""
        <td><img width=350px src="{param}staccvsinjSNR.png"</td>
        <td><img width=350px src="{param}staccvslogB.png"</td>
        <td><img width=350px src="{param}stacc.png"</td>
        """.format(param=param)

    htmlstr+="""
        </tr>
    </table>
    """


    # ---------------
    htmlstr+="""
    <hr>
    <h2>Per Injection Results</h2>
    """

    p=0
    for posterior, injection_dir in zip(posteriors, injection_dirs):
        injection=injection_dir.split('/')[-1]


        htmlstr+="""
        <h2>{injection}</h2>
        <a href="{injection}"><b>All results</b></a></td>
        <h3>Summary Statistics</h3>
        <p>
            <ul>
                <li>NetSNR (injected): {snr}</li>
                <li>BSN: {bsn}</li>
            </ul>
        </p>

        <p>
            <table border="2">
                <tr>
                    <td><b>Parameter</b></td>
                    <td><b>maxP (true-maxP)</b></td>
                    <td><b>mean (true-mean)</b></td>
                    <td><b>median (true-median)</b></td>
                    <td><b>stdev</b></td>
                    <td><b>low</b></td>
                    <td><b>high</b></td>
                    <td><b>stacc</b></td>
                </tr>
        """.format(injection=injection, snr=all_snr[p], bsn=all_bsn[p])


        oneDMenu, twoDMenu, binSizes = oneD_bin_params()
        parnames=filter(lambda x: x in posterior.names, oneDMenu)

        for par_name in parnames:

            # get confidence intervals and staccs
            (low, high) = all_cl_intervals[p][par_name]
            this_stacc = all_staccs[p][par_name]
            this_acc = all_accs[p][par_name]

            htmlstr+="""
                        <tr>
                            <td>{parname}</td>
                            <td>{maxP} ({delta_maxP})</td>
                            <td>{mean} ({delta_mean})</td>
                            <td>{median} ({delta_median})</td>
                            <td>{stdev}</td>
                            <td>{low}</td>
                            <td>{high}</td>
                            <td>{stacc}</td>
                        </tr>
                    """.format(parname=par_name, 
                            maxP=posterior.maxP[1][par_name],
                            mean=posterior.means[par_name],
                            median=posterior.medians[par_name],
                            delta_maxP=this_acc[0], delta_mean=this_acc[1],
                            delta_median=this_acc[2],
                            stdev=posterior.stdevs[par_name], 
                            low=low, high=high,
                            stacc=this_stacc)
                            
        htmlstr+="""
        </tr>
        </table>
        </p>

        <p>
            <img width=800px src="{injection}/corner.png">
        </p>
        <hr>
            """.format(injection=injection)
        p+=1

    htmlstr+="""
    </html>
    """

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
        delta_maxP=truth - pos.maxP[1][param]
        delta_mean=truth - pos.means[param]
        delta_median=truth - pos.medians[param]
        return [delta_maxP, delta_mean, delta_median]

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
            staccs_allparams, accs_allparams

def define_truth(oneDMenu,waveform):
    """
    Return a dictionary of injected values
    """
    truevals={}
    for item in oneDMenu:
        if item=='frequency':
            truevals[item]=waveform.fpeak
        else:
            truevals[item]=None

    return truevals

def plot_measurement_vs_statistic(x, y, yerrs=None, truevals=None, param=None,
        xlabel='', ylabel=''):
    """
    Plot recovered value, with error bars, vs detection stat (like injected SNR
    or recovered logB)
    """
    fig, ax = pl.subplots(figsize=(6,4))#, dpi=200)

    if yerrs is not None: yerrs=abs(yerrs)
    
    ax.errorbar(x, y, yerr=yerrs, linestyle='none', marker='^', capsize=0,
            ecolor='k', color='m')
    ax.minorticks_on()

    if truevals is not None:
        ax.axhline(truevals[param], color='r', linestyle='--')

    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel('Ranking statistic')

    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        if param in ['frequency', 'bandwidth']:
            ax.set_ylabel('%s [Hz]'%(param))
        else:
            ax.set_ylabel(param)

    fig.tight_layout()

    return fig

def make_oneDhist(samples, param=None, xlabel='', ylabel=''):
    """
    Plot histogram
    """

    fig, ax = pl.subplots(figsize=(6,4))#, dpi=200)

    histbinswidth = 3.5*np.std(samples) / len(samples)**(1./3)
    histbins = np.arange(min(samples), max(samples), histbinswidth)
    (n, bins, patches) = ax.hist(samples, histbins, normed='true',
            histtype='step', facecolor='grey', color='k')
    ax.set_ylim(0, 1.05*max(n))

    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('PDF')

    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        if param in ['frequency', 'bandwidth']:
            ax.set_xlabel('%s [Hz]'%(param))
        else:
            ax.set_xlabel(param)

    fig.tight_layout()

    return fig

# End
# =========================================================================

def reconstructed_SineGaussianF(posterior, waveform, flow=1000, fupp=4096):
    """
    return the reconstructed F-domain sine-Gaussians from the posterior samples
    in posterior, as well as the max-posterior reconstruction and the matches
    with the target waveform
    """
    wlen=16384


    # Get a zero-padded version of the target waveform
    htarget=np.zeros(wlen)
    htarget[0:len(waveform.hplus)]=waveform.hplus.data
    htarget=pycbc.types.TimeSeries(htarget, delta_t = waveform.hplus.delta_t)

    # Get frequency series of target waveform
    H_target = htarget.to_frequencyseries()

    # Make psd for matches
    flen = len(H_target)
    delta_f = np.diff(H_target.sample_frequencies)[0]
    psd = aLIGOZeroDetHighPower(flen, H_target.delta_f, low_freq_cutoff=flow)

    # -----------
    # MAP waveform
    # XXX: Time-domain is a little easier, since we don't have to figure out
    # which frequencies to populate in a pycbc object

    hp, _ = lalsim.SimBurstSineGaussian(posterior.maxP[1]['quality'],
            posterior.maxP[1]['frequency'], posterior.maxP[1]['hrss'], 0.0, 0.0,
            waveform.hplus.delta_t)

    # zero-pad
    h_MAP = np.zeros(wlen)
    h_MAP[:hp.data.length]=hp.data.data
    h_MAP_ts = pycbc.types.TimeSeries(h_MAP,waveform.hplus.delta_t)
    H_MAP = h_MAP_ts.to_frequencyseries()

    MAP_match = pycbc.filter.match(H_target, H_MAP, low_frequency_cutoff=flow,
            high_frequency_cutoff=fupp)

    # -----------

    return H_target, H_MAP, MAP_match

    # -------------------------
    # Waveforms for all samples
    # Pre-allocate:
    nsamps = len(posterior['frequency'].samples)
    reconstructions = np.zeros(shape=(nsamps, len(Hplus)))
    matches = np.zeros(nsamps)



    # All samples!
    for idx in xrange(nsamps):

        hcurrent = np.zeros(wlen)

        # let's just use hplus for now...
        hp, _ = lalsim.SimBurstSineGaussian(posterior['quality'].samples[idx],
                posterior['frequency'].samples[idx],
                posterior['hrss'].samples[idx], 0.0, 0.0,
                waveform.hplus.delta_t)

        # zero-padded version
        hcurrent[:hp.data.length] = hp.data.data

        # pycbc object
        hcurrent_ts = pycbc.types.TimeSeries(hcurrent, delta_t=hp.deltaT)

        reconstructions[idx, :] = Hcurrent.data


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
waveform.reproject_waveform(0,0)
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
all_cl_intervals = []
all_staccs = []
all_accs = []
all_BSN = []
all_SNR = []

for fileidx, files in enumerate(zip(sampfiles,BSNfiles,snrfiles)):

    posterior_file = files[0]
    bsn_file = files[1]
    snr_file = files[2]

    injdir, BSN, SNR, thisposterior, cl_intervals, staccs, accs = \
            single_injection_results(outputdirectory, posterior_file, bsn_file,
                    snr_file)

    injection_dirs.append(injdir)
    allposteriors.append(thisposterior)
    all_cl_intervals.append(cl_intervals)
    all_staccs.append(staccs)
    all_accs.append(accs)
    all_BSN.append(BSN)
    all_SNR.append(SNR)


# -------------------------------------------------------
# Construct summary plots and statistics

# 
# Parameter Estimation
#

# Get true values (XXX: repeated code!):
oneDMenu, twoDMenu, binSizes = oneD_bin_params()
truevals=define_truth(oneDMenu, waveform)

for param in truevals.keys():

    #
    # Measured values
    #

    # Measured values for this parameter
    try:
        measured_vals = [ pos.maxP[1][param] for pos in allposteriors ]
    except KeyError: continue


    # Error bars for this parameter
    try:
        error_bars = np.transpose([
            all_cl_intervals[p][param]-allposteriors[p].maxP[1][param] for p in
            xrange(len(allposteriors)) ])
    except KeyError: continue

    # param vs statistic
    fig = plot_measurement_vs_statistic(x=all_BSN, y=measured_vals, \
            yerrs=error_bars, truevals=truevals, param=param, xlabel=r'$\log B_{\rm{s,n}}$')
    fig.savefig('{outputdirectory}/{param}vslogB.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)

    fig = plot_measurement_vs_statistic(x=all_SNR, y=measured_vals, \
            yerrs=error_bars, truevals=truevals, param=param, xlabel=r'Injected SNR')
    fig.savefig('{outputdirectory}/{param}vsinjSNR.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)

    # histogrammed params
    fig = make_oneDhist(measured_vals, param=param)
    fig.savefig('{outputdirectory}/{param}.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)

    #
    # Accuracy
    #
    if truevals[param] is None: continue

    # Get the maxP estimate (zeroth)
    measured_vals = [ all_accs[p][param][0] for p in xrange(len(all_accs)) ]

    fig = plot_measurement_vs_statistic(x=all_BSN, y=measured_vals, param=param,
            ylabel=param+' accuracy', xlabel=r'$\log B_{\rm{s,n}}$')
    pl.yscale('log')
    fig.savefig('{outputdirectory}/{param}accvslogB.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)

    fig = plot_measurement_vs_statistic(x=all_SNR, y=measured_vals, param=param,
            ylabel=param+' accuracy', xlabel=r'Injected SNR')
    pl.yscale('log')
    fig.savefig('{outputdirectory}/{param}accvsinjSNR.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)

    fig = make_oneDhist(measured_vals, param=param, xlabel=param+' accuracy')
    fig.savefig('{outputdirectory}/{param}acc.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)

    #
    # STANDARD Accuracy
    #

    # Get the maxP estimate (zeroth)
    measured_vals = [ all_staccs[p][param] for p in xrange(len(all_staccs)) ]

    fig = plot_measurement_vs_statistic(x=all_BSN, y=measured_vals, param=param,
            ylabel=param+' standard accuracy', xlabel=r'$\log B_{\rm{s,n}}$')
    pl.yscale('log')
    fig.savefig('{outputdirectory}/{param}staccvslogB.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)

    fig = plot_measurement_vs_statistic(x=all_SNR, y=measured_vals, param=param,
            ylabel=param+' standard accuracy', xlabel=r'Injected SNR')
    pl.yscale('log')
    fig.savefig('{outputdirectory}/{param}staccvsinjSNR.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)

    fig = make_oneDhist(measured_vals, param=param, xlabel=param+' standard accuracy')
    fig.savefig('{outputdirectory}/{param}stacc.png'.format(param=param,
        outputdirectory=outputdirectory))
    pl.close(fig)



# -------------------------------------------------------
# Write HTML summary
write_results_page(outputdirectory, injection_dirs, allposteriors,
        all_cl_intervals, all_staccs, all_accs, all_BSN, all_SNR, truevals)


