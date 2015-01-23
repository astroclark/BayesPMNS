#!/usr/bin/env python

import sys
import numpy as np
import cPickle as pickle
from optparse import OptionParser
from sklearn.neighbors.kde import KernelDensity

def parser():

    parser = OptionParser()

    parser.add_option('--grid-min', default=1500, type=float)
    parser.add_option('--grid-max', default=4000, type=float)
    parser.add_option('--grid-spacing', default=1, type=float)
    parser.add_option('--kde-bandwidth', default=5, type=float)

    (opts,args) = parser.parse_args()

    results_pickle = args[0]

    return opts, results_pickle

def kde_sklearn(x, x_grid, bw=5, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""

#    bw = 1.06*np.std(x)*len(x)**(-1./5)

    kde_skl = KernelDensity(bandwidth=bw, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

# ------------------------
# Main

opts, results_pickle = parser()

detstats, samples = pickle.load(open(results_pickle, 'r'))

grid = np.arange(opts.grid_min, opts.grid_max, opts.grid_spacing)
kde_pdf = np.zeros(shape=(len(samples), len(grid)))

for f in xrange(len(samples)):

    print '%d of %d'%(f, len(samples))

    kde_pdf[f, :] = kde_sklearn(samples[f], grid, bw=opts.kde_bandwidth)
