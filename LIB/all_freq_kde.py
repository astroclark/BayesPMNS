#!/usr/bin/env python

import sys
import numpy as np
import cPickle as pickle
from sklearn.neighbors.kde import KernelDensity

def kde_sklearn(x, x_grid, bw=5, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""

#    bw = 1.06*np.std(x)*len(x)**(-1./5)

    kde_skl = KernelDensity(bandwidth=bw, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

data = pickle.load(open(sys.argv[1],'rb'))
logB = np.loadtxt('logBsn.txt')[::-1]

freq_grid = np.arange(1500, 4000, 1)
freq_kde_pdf = np.zeros(shape=(len(data), len(freq_grid)))

for f in xrange(len(data)):

    print '%d of %d'%(f, len(data))

    freq_kde_pdf[f, :] = kde_sklearn(data[f], freq_grid)
