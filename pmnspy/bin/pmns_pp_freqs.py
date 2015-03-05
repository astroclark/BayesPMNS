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
import fnmatch

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as pl

from sklearn import mixture

from pylal import bayespputils as bppu

import pmns_utils

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
        {'savefig.dpi': 200,                                                                                     
        'xtick.major.size':8,
        'xtick.minor.size':4,
        'ytick.major.size':8,                                                                                     
        'ytick.minor.size':4
        })  
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"] 

def gmm_peaks(Z, min_membership=0.01):
    """
    Fit a Gaussian mixture model to samples and return the means and covariances
    of the resulting Gaussians

    We do not allow clusters to have membership smaller than 1% and we allow up
    to 3 components
    """

    bic=np.zeros(3)
    min_bic=np.inf
    for c in xrange(3):
        print c
        g = mixture.GMM(n_components=c+1)#, n_iter=1000, covariance_type='full')
        g.fit(Z)
        bic[c] = g.bic(Z)

        # Check that clusters satisfy minimum membership requirement
        labels = g.predict(Z)
        print 'trying %d clusters'%c
        min_membership=0.01
        for label in xrange(c+1):
            membership=float(sum(labels==label))/len(labels) 
            print 'cluster %d membership: %.2f'%(\
                    label, membership)
            if membership<min_membership:
                bic[c] = np.inf
                continue
     
        if bic[c] < min_bic:
            min_bic=bic[c]
            best_g = g 

#   gmm_result={}
#   gmm_result['means'] = best_g.means_
#   gmm_result['covars'] = best_g.covars_

    return best_g

def find_posterior_files(path, pospattern="posterior*dat",
        Bpattern="posterior*dat_B.txt", SNRpattern="*snr.txt"):
    """
    Locates the posterior sample files in the postproc directory specified in
    path
    """

    posfiles = []
    Bfiles = []
    SNRfiles = []
    roots = []

    for root, dirnames, filenames in os.walk(path):

        for filename in fnmatch.filter(filenames, pospattern):
            posfiles.append(os.path.join(root, filename))

        for filename in fnmatch.filter(filenames, Bpattern):
            Bfiles.append(os.path.join(root, filename))

        for filename in fnmatch.filter(filenames, SNRpattern):
            SNRfiles.append(os.path.join(root, filename))

        roots.append(root)

    return posfiles, Bfiles, SNRfiles, roots

def get_modes(posfiles, plot=False):
    """
    Return a tuple with the means and covariances of Gaussian mixture models fit
    to the posterior modes
    """

    fmeans = np.empty(shape=(len(posfiles), 3))
    fmeans.fill(None)

    fcovars = np.empty(shape=(len(posfiles), 3))
    fcovars.fill(None)

    # Create PE parser object and construct posterior
    peparser = bppu.PEOutputParser('common')

    gmms=[]
    for p,posfile in enumerate(posfiles):

        resultsObj = peparser.parse(open(posfile,'r'), info=[None, None])
        posterior = bppu.Posterior(resultsObj)

        # Get GMM for frequency posterior
        gmmresult = gmm_peaks(posterior['frequency'].samples)

        # Populate the fpeaks array such that the highest frequency is the first
        # column
        idx = np.argsort(gmmresult.means_)[::-1]
        for i in xrange(len(idx)):
            fmeans[p, i] = gmmresult.means_[idx][i]
            fcovars[p, i] = gmmresult.covars_[idx][i]

        if plot:
            outdir = os.path.dirname(posfile)
            f, ax = plot_oneDposterior(posterior, 'frequency', histbinswidth=2.5)
            freqs = np.arange(1500, 4000, 0.1)
            ax.plot(freqs, np.exp(gmmresult.score_samples(freqs)[0]), color='r',
                    label='GMM')
            ax.legend()
            f.savefig(os.path.join(outdir, 'frequency_posterior_with_GMM.png'))

    return fmeans, fcovars

def dump_summary(outfile, summary_list):
    """
    Dump a text file with the summaries
    """

    f=open(outfile, "w")
    f.write("# name mean std median 25thPercentile 75thPercentile\n")
    for summary in summary_list:
        f.write("{name} {mean} {std} {median} {pc25} {pc75}\n".format(
                    name=summary['name'],
                    mean=summary['mean'],
                    std=summary['std'],
                    median=summary['median'],
                    pc25=summary['25thPercentile'],
                    pc75=summary['75thPercentile'],
                    ))


def measurement_summary(name, values):
    """
    Return a dictionary with summary statistics of the sample of values
    """
    
    # Use a masked array
    values = np.ma.masked_array(values, np.isnan(values))

    summary = {}
    summary['name'] = name
    summary['mean'] = np.mean(values)
    summary['median'] = np.median(values)
    summary['std'] = np.std(values)
    summary['25thPercentile'] = np.percentile(values, 25)
    summary['75thPercentile'] = np.percentile(values, 75)

    return summary

def load_Bfiles(bfiles):

    logBs=np.zeros(len(bfiles))
    for b, bfile in enumerate(bfiles):
        bfile=open(bfile, 'r')
        BSN=bfile.read()
        bfile.close()
        logBs[b]=float(BSN.split()[0])

    return logBs

def load_snrfiles(snrfiles):

    snrs_out=np.zeros(len(snrfiles))

    for s, snr_file in enumerate(snrfiles):

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
        snrs_out[s] = float(snrstring.split()[-1])


    return snrs_out

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

    return fig, ax

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

def plot_oneDposterior(posterior, param, cl_intervals=None, 
        parvals=None, plotkde=False, histbinswidth=None):
    """
    Plots a 1D histogram of the distribution of posterior samples for a given
    parameter
    """
    pos_samps = posterior[param].samples


    fig, ax = pl.subplots(figsize=(6,4))#, dpi=200)

    if plotkde==False:

        # Plot histogram
        if histbinswidth is None:
            histbinswidth = 3.5*posterior[param].stdev / len(pos_samps)**(1./3)
        histbins = np.arange(pos_samps.min(), pos_samps.max(), histbinswidth)
        (n, bins, patches) = ax.hist(pos_samps, histbins, normed='true',
                histtype='step', facecolor='grey', color='k', label='posterior samples',
                linewidth=2)
        ax.set_ylim(0, 1.05*max(n))
    else:
        bw = 1.06*np.std(pos_samps)* len(pos_samps)**(-1./5)
        x_grid = np.linspace(0.9*min(pos_samps),1.1*max(pos_samps),1000)
        pdf = kde_sklearn(x=np.concatenate(pos_samps), x_grid=x_grid, bandwidth=bw)

        ax.plot(x_grid,pdf,color='grey')

    # Show injected value
    if parvals is not None:
        ax.axvline(parvals[param], color='r', label='Target %s'%param)

    # Show the median and 90% confidence interval
    if cl_intervals is not None:
        try:
            ax.axvline(cl_intervals[0], color='k', linestyle='--')
            ax.axvline(cl_intervals[1], color='k', linestyle='--', label=r'$\alpha=0.9$')
            ax.axvline(posterior[param].median, color='k', linestyle='-',
                    label=r'median')
        except RuntimeError:
            pass

    # set axis limits
    if parvals is not None:
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

    return fig, ax


# -------------------

def main():

    print "running pmns_pp_utils.py doesn't do much (yet)"

    # Get post-proc dir
    ppdir = sys.argv[1]
    waveform_name = sys.argv[2]
    waveform = pmns_utils.Waveform(waveform_name)
    waveform.reproject_waveform()
    waveform.compute_characteristics()

    outdir=os.path.join(ppdir, "gmmfreqs")
    try:
        os.makedirs(outdir)
    except OSError:
        print 'e'

    # Identify posterior files:
    posfiles, Bfiles, snrsfiles, posparents = find_posterior_files(ppdir)
    logBs = load_Bfiles(Bfiles)
    SNRs = load_snrfiles(snrsfiles)

    # Get the frequency posterior modes
    freq_mode_means, freq_mode_covars = get_modes(posfiles, plot=True)

    # Compute the fpeak accuracy, assuming the highest freq mode is the one
    # which captures fpeak
    freq_acc = abs(waveform.fpeak - freq_mode_means[:,0])

    #
    # Summarise the results
    #
    freq_acc_summary = measurement_summary('GMMfpeakAccuracy', freq_acc)

    fmax_mu_summary = measurement_summary('GMMfmax',freq_mode_means[:,0])

    fmid_mu_summary = measurement_summary('GMMfmid',freq_mode_means[:,1])

    fmin_mu_summary = measurement_summary('GMMfmin',freq_mode_means[:,2])

    dump_summary(os.path.join(outdir, 'GMMfreqsSummary.txt'), [freq_acc_summary,
        fmax_mu_summary, fmid_mu_summary, fmin_mu_summary])

#   fmax_sigma_summary = measurement_summary('GMMfmax_sigma',
#           np.sqrt(freq_mode_covars[:,0]))
#   fmid_sigma_summary = measurement_summary('GMMfmid_sigma',
#           np.sqrt(freq_mode_covars[:,1]))
#   fmin_sigma_summary = measurement_summary('GMMfmin_sigma',
#           np.sqrt(freq_mode_covars[:,2]))
#
    #
    # Generate Web Page & Plots
    #

    # SNR vs freq acc
    fig, ax = plot_measurement_vs_statistic(x=SNRs, y=freq_acc, \
            yerrs=None, truevals=None, param=None, xlabel=r'Injected SNR',
            ylabel='GMM Frequency Accuracy [Hz]')
    ax.set_yscale('log')
    fig.savefig('{outdir}/GMMfpeakAccuracyvsinjSNR.png'.format(outdir=outdir))
    pl.close(fig)

    # logB vs freq acc
    fig, ax = plot_measurement_vs_statistic(x=logBs, y=freq_acc, \
            yerrs=None, truevals=None, param=None, xlabel=r'BSN',
            ylabel='GMM Frequency Accuracy [Hz]')
    ax.set_yscale('log')
    fig.savefig('{outdir}/GMMfpeakAccuracyvsBSN.png'.format(outdir=outdir))
    pl.close(fig)

    # SNR vs maxfreq
    fig, ax = plot_measurement_vs_statistic(x=SNRs, y=freq_mode_means[:,0], \
            yerrs=None, truevals=None, param=None, xlabel=r'Injected SNR',
            ylabel='GMM Max Frequency [Hz]')
    fig.savefig('{outdir}/GMMMaxFreqvsinjSNR.png'.format(outdir=outdir))
    pl.close(fig)

    # logB vs maxfreq
    fig, ax = plot_measurement_vs_statistic(x=logBs, y=freq_mode_means[:,0], \
            yerrs=None, truevals=None, param=None, xlabel=r'BSN',
            ylabel='GMM Max Frequency [Hz]')
    fig.savefig('{outdir}/GMMMaxFreqvsBSN.png'.format(outdir=outdir))
    pl.close(fig)

    # SNR vs midfreq
    fig, ax = plot_measurement_vs_statistic(x=SNRs, y=freq_mode_means[:,1], \
            yerrs=None, truevals=None, param=None, xlabel=r'Injected SNR',
            ylabel='GMM Mid Frequency [Hz]')
    fig.savefig('{outdir}/GMMMidFreqvsinjSNR.png'.format(outdir=outdir))
    pl.close(fig)

    # logB vs midfreq
    fig, ax = plot_measurement_vs_statistic(x=logBs, y=freq_mode_means[:,1], \
            yerrs=None, truevals=None, param=None, xlabel=r'BSN',
            ylabel='GMM Mid Frequency [Hz]')
    fig.savefig('{outdir}/GMMMidFreqvsBSN.png'.format(outdir=outdir))
    pl.close(fig)

    # SNR vs minfreq
    fig, ax = plot_measurement_vs_statistic(x=SNRs, y=freq_mode_means[:,0], \
            yerrs=None, truevals=None, param=None, xlabel=r'Injected SNR',
            ylabel='GMM Min Frequency [Hz]')
    fig.savefig('{outdir}/GMMMinFreqvsinjSNR.png'.format(outdir=outdir))
    pl.close(fig)

    # logB vs minfreq
    fig, ax = plot_measurement_vs_statistic(x=logBs, y=freq_mode_means[:,0], \
            yerrs=None, truevals=None, param=None, xlabel=r'BSN',
            ylabel='GMM Min Frequency[Hz]')
    fig.savefig('{outdir}/GMMMinFreqvsBSN.png'.format(outdir=outdir))
    pl.close(fig)


#
# End definitions
#
if __name__ == "__main__":
    main()



