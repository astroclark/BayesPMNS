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

detstats, samples = pickle.load(open())

grid = np.arange(1500, 4000, 1)
kde_pdf = np.zeros(shape=(len(samples), len(grid)))

for f in xrange(len(samples)):

    print '%d of %d'%(f, len(samples))

    freq_kde_pdf[f, :] = kde_sklearn(samples[f], freq_grid)
