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

gbw_params = \
        np.loadtxt('gaussianBW_lalinferencenest-0-V1H1L1-1106316981.43-0.dat_params.txt',dtype=str)
gbw_samps = np.loadtxt('gBW_parameterisation_pos.dat',skiprows=1)

jeffbw_params = \
        np.loadtxt('jeffBW_lalinferencenest-0-V1H1L1-1106316981.43-0.dat_params.txt',dtype=str)
jeffbw_samps = np.loadtxt('jeffBW_parameterisation_pos.dat',skiprows=1)

bw_params = \
        np.loadtxt('lalinferencenest-0-V1H1L1-1106316981.43-0.dat_params.txt',dtype=str)
bw_samps = np.loadtxt('bw_parameterisation_pos.dat',skiprows=1)

Q_params = \
        np.loadtxt('SG_lalinferencenest-0-V1H1L1-1106316981.43-0.dat_params.txt',dtype=str)
Q_samps = np.loadtxt('Q_parameterisation_pos.dat',skiprows=1)


jeffbw_freq = getfreq(jeffbw_params, jeffbw_samps)
gbw_freq = getfreq(gbw_params, gbw_samps)
bw_freq = getfreq(bw_params, bw_samps)
Q_freq = getfreq(Q_params, Q_samps)

jeffbw_bws = getbandwidth(jeffbw_params, jeffbw_samps)
gbw_bws = getbandwidth(gbw_params, gbw_samps)
bw_bws = getbandwidth(bw_params, bw_samps)
Q_Qs = getquality(Q_params, Q_samps)

jeffbw_Qs = quality(jeffbw_freq,jeffbw_bws)
gbw_Qs = quality(gbw_freq,gbw_bws)
bw_Qs = quality(bw_freq,bw_bws)
Q_bws = bandwidth(Q_freq, Q_Qs)

# ---
# Compare Bandwidths
pl.figure()
pl.hist(Q_bws, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='Q ')
pl.hist(bw_bws, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='BW ')
pl.hist(gbw_bws, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='GaussPrior BW ')
pl.hist(jeffbw_bws, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='JeffPrior BW ')
pl.ylim(0,1)
pl.xlabel('Bandwidth [Hz]')
pl.legend(loc='upper left')
pl.savefig('bandwidth_comparison.png')

# ---
# Compare Qualities
pl.figure()
pl.hist(Q_Qs, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='Q ')
pl.hist(bw_Qs, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='BW ')
pl.hist(gbw_Qs, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='GaussPrior BW ')
pl.hist(jeffbw_Qs, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='JeffPrior BW ')
pl.ylim(0,1)
pl.xlabel('Quality')
pl.legend(loc='upper left')
pl.savefig('quality_comparison.png')

# ---
# Compare Frequencies
pl.figure()
pl.hist(Q_freq, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='Q ')
pl.hist(bw_freq, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='BW ')
pl.hist(gbw_freq, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='GaussPrior BW ')
pl.hist(jeffbw_freq, 100, normed=True, 
        histtype='stepfilled', alpha=0.5, label='JeffPrior BW ')
pl.ylim(0,.25)
pl.xlabel('Frequency [Hz]')
pl.legend(loc='upper left')
pl.savefig('frequency_comparison.png')

pl.show()
