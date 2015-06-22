#!/usr/bin/env python

import numpy as np
import scipy.io as sio

from matplotlib import pyplot as pl

noisedata = sio.loadmat('./curves.mat')


ncurves = len(noisedata['lgnd'][0])

# Turn the data into a more useful dictionary
noisecurves=dict()
for i in xrange(ncurves):
    noisecurves[str(noisedata['lgnd'][0][i][0])] = noisedata['h'][i,:] 
