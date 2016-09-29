#!/usr/bin/env python

import numpy as np
import scipy.io as sio
import cPickle as pickle

quantumGEO1 = np.loadtxt('noisequantum1.txt')
quantumGEO2 = np.loadtxt('noisequantum2.txt')

noisedata = sio.loadmat('./curves.mat')

ncurves = len(noisedata['lgnd'][0])

# Turn the data into a more useful dictionary
noisecurves=dict()
for i in xrange(ncurves):
    noisecurves[str(noisedata['lgnd'][0][i][0])] = noisedata['h'][i,:] 

noisecurves['sample_frequencies'] = noisedata['f'][0]

noisecurves['quantumGEO1'] = np.interp(noisecurves['sample_frequencies'],
        quantumGEO1[:,0], quantumGEO1[:,1])
noisecurves['quantumGEO2'] = np.interp(noisecurves['sample_frequencies'],
        quantumGEO2[:,0], quantumGEO2[:,1])

pickle.dump(noisecurves, open('ADE_noise_curves.pickle','w'))

