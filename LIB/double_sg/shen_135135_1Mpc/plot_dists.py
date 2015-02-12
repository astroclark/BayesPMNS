#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as pl

def getfreq(params,samps):
    return samps[:,params=='frequency']
def getquality(params,samps):
    return samps[:,params=='quality']
def getbandwidth(params,samps):
    return samps[:,params=='bandwidth']

def quality(f,b):
    return f/b 
def bandwidth(f,q):
    return f/q 


bw_params = \
        np.loadtxt('lalinferencenest-0-V1H1L1-1106316981.43-0.dat_params.txt',dtype=str)
bw_samps = np.loadtxt('bw_parameterisation_pos.dat',skiprows=1)

Q_params = \
        np.loadtxt('SG_lalinferencenest-0-V1H1L1-1106316981.43-0.dat_params.txt',dtype=str)
Q_samps = np.loadtxt('Q_parameterisation_pos.dat',skiprows=1)


bw_freq = getfreq(bw_params, bw_samps)
Q_freq = getfreq(Q_params, Q_samps)

bw_bws = getbandwidth(bw_params, bw_samps)
Q_Qs = getquality(Q_params, Q_samps)

bw_Qs = quality(bw_freq,bw_bws)
Q_bws = bandwidth(Q_freq, Q_Qs)

# ---
# Compare Bandwidths
pl.figure()
pl.hist(Q_bws, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='Q parameterisation')
pl.hist(bw_bws, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='BW parameterisation')
pl.ylim(0,1)
pl.xlabel('Bandwidth [Hz]')
pl.legend(loc='upper left')
pl.savefig('bandwidth_comparison.png')

# ---
# Compare Qualities
pl.figure()
pl.hist(Q_Qs, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='Q parameterisation')
pl.hist(bw_Qs, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='BW parameterisation')
pl.ylim(0,1)
pl.xlabel('Quality')
pl.legend(loc='upper left')
pl.savefig('quality_comparison.png')

# ---
# Compare Frequencies
pl.figure()
pl.hist(Q_freq, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='Q parameterisation')
pl.hist(bw_freq, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='BW parameterisation')
pl.ylim(0,.25)
pl.xlabel('Frequency [Hz]')
pl.legend(loc='upper left')
pl.savefig('frequency_comparison.png')

pl.show()
