#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2014-2015 James Clark <james.clark@physics.gatech.edu>
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
import matplotlib
#matplotlib.use("Agg")

from scipy import signal, optimize, special, stats

import pmns_utils
#import pmns_simsig as simsig

import lal
import lalsimulation as lalsim
import pycbc.types
import pycbc.filter

import pylab as pl

def apply_taper(TimeSeries,newlen=16384):
    """
    Smoothly taper the start of the data in TimeSeries using LAL tapering
    routines

    Also zero pads
    """

    # Create and populate lal time series
    tmp=lal.CreateREAL8TimeSeries('tmp', lal.LIGOTimeGPS(), 0.0,
            TimeSeries.delta_t, lal.StrainUnit, newlen)
            #TimeSeries.delta_t, lal.StrainUnit, len(TimeSeries))
    tmp.data.data = np.zeros(newlen)
    tmp.data.data[0:len(TimeSeries)]=TimeSeries.data

    # Taper
    lalsim.SimInspiralREAL8WaveTaper(tmp.data, lalsim.SIM_INSPIRAL_TAPER_START)

    return pycbc.types.TimeSeries(tmp.data.data, delta_t=TimeSeries.delta_t)


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

def top_peaks(PSD, npeaks=3, min_separation=500):
    """
    Identify up to the top 3 peaks using the signal.find_peaks_cwt algorithm

    Returns their indices
    """
    #peakidx = signal.find_peaks_cwt(PSD, widths)
    peakidx = detect_peaks(PSD, mpd=min_separation)

    
    PSD_peaks = np.sort(PSD[peakidx])[::-1]

    n=1
    peakidx=[]
    for peak in PSD_peaks:
        peakidx.append(np.concatenate(np.argwhere(PSD==peak))[0])
        if n==npeaks: return peakidx
        n+=1



# --------------------------------------------------------------------
# Data Generation

#
# Generate Signal Data
#

# Signal
print ''
print '--- %s ---'%sys.argv[1]
waveform = pmns_utils.Waveform('%s_lessvisc'%sys.argv[1])
waveform.reproject_waveform()
flow=1000

#
# Condition time series and get spectrum
#
h = apply_taper(waveform.hplus)
H = abs(h.to_frequencyseries())**2
freqs = h.to_frequencyseries().sample_frequencies.data

# reduce
H=H[freqs>=flow]
freqs=freqs[freqs>=flow]

# Find top 3 peaks
peakidx=top_peaks(H)

# discard the lowest frequency peak; should leave us with fpeak, f-
#peakidx.pop(np.argmin(freqs[peakidx]))

print "original f1, f2:", freqs[peakidx]

# --------------------------------------------------------------------
# Windowing experiment
#
# We'll look at the properties of the waveform as a function of the amount of
# data we retain around the inspiral peak
# 
# Seconds before peak to retain: negative will truncate waveform (then taper)
# waveform *prior* to peak, positive will truncate *after* the peak
Ndelays=10
fpeaks=[]

min_delay = -1*h.max_loc()[1] * h.delta_t

delays=np.linspace(min_delay, 0, Ndelays)

for d, delay in enumerate(delays):
    #delay=-1e-3

    #
    # Find time-domain peak and zero-out data prior to peak time+deltaT
    #
    tzero_idx = h.max_loc()[1] + delay / h.delta_t

    hnew = pycbc.types.TimeSeries(h,delta_t=h.delta_t)
    hnew.data[:tzero_idx]=0.0
    hnew = apply_taper(hnew)

    #
    # Recover peak frequencies
    #

    # Get spectrum
    Hnew = abs(hnew.to_frequencyseries())**2
    freqsnew = hnew.to_frequencyseries().sample_frequencies.data

    Hnew=Hnew[freqs>=flow]
    freqsnew=freqsnew[freqs>=flow]

    # Find top 3 peaks
    peakidxnew=top_peaks(Hnew)

    # discard the lowest frequency peak; should leave us with fpeak, f-
    #peakidxnew.pop(np.argmin(freqsnew[peakidxnew]))

    # record result
    #fpeaks[d,0] = min(freqsnew[peakidxnew])
    #fpeaks[d,1] = max(freqsnew[peakidxnew])
    fpeaks.append(np.sort(freqsnew[peakidxnew]))

    # some plots
    print "delay: %.2f ms: f1, f2="%(1e3*delay), freqsnew[peakidxnew]



    f, ax = pl.subplots(nrows=2)
    ax[0].plot(h.sample_times*1e3, h)
    ax[0].plot(hnew.sample_times*1e3, hnew, color='r')
    ax[0].set_xlim(0,30)

    ax[1].plot(freqs,H)
    ax[1].plot(freqs[peakidx],H[peakidx], 'b^')

    ax[1].plot(freqsnew,Hnew, color='r')
    ax[1].plot(freqsnew[peakidxnew],Hnew[peakidxnew], 'rv')

    ax[1].set_xlim(flow, 4096)

    #sys.exit()
    pl.savefig('time_spec-%.2f.png'%(1e3*delay))

fpeaks=np.array(fpeaks)

pl.close('all')



