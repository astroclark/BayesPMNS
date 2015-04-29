#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
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

import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as pl

import triangle

from sklearn.neighbors.kde import KernelDensity
from pylal import bayespputils as bppu

from IPython.core.debugger import Tracer; debug_here = Tracer()

__author__ = "James Clark <james.clark@ligo.org>"


class PosteriorResults:

    def __init__(self, samples_file, inj_theta=None):

        self.params = ['hrss','fpeak','beta1']

        self.samples_file = samples_file
        
        if inj_theta is None:
            self.inj_theta = [None, None, None]
        else:
            self.inj_theta = inj_theta

        self.binSizes = {'hrss':5e-24, 'fpeak':1, 'beta1':0.01}
        self.levels = [0.67, 0.9, 0.95]

        self.make_posteriors()


    def make_posteriors(self):

        peparser = bppu.PEOutputParser('common')
        resultsObj = peparser.parse(open(self.samples_file, 'r'))
        self.posterior = bppu.Posterior(resultsObj)

        # Create bppu posterior instance for easy conf intervals and
        # characterisation

        self.intervals = {}
        for p, param in enumerate(self.params):

            # --- Statistics from this posterior
            toppoints, injection_cl, reses, injection_area, cl_intervals = \
                    bppu.greedy_bin_one_param(self.posterior,
                            {param:self.binSizes[param]}, self.levels)

            self.intervals[param]=cl_intervals

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
    #extents = [get_extent(posterior,name,parvals) for name in parnames]

    print percentiles
    trifig=triangle.corner(data, labels=parnames, truths=truths,
            quantiles=percentiles, truth_color='r')#, extents=extents)

    return trifig
    

def main():

    params = ['hrss', 'fpeak', 'beta1']

    #
    # Load Data
    #
    results_from_file = np.load(sys.argv[1])
    samples_file = sys.argv[2]

    inj_theta = results_from_file['inj_theta']
    parvals={'hrss':inj_theta[0], 'fpeak':inj_theta[1], 'beta1':inj_theta[2]}

    pos_result = PosteriorResults(samples_file=samples_file,
            inj_theta=inj_theta)

    # 
    # Triangle Plot
    #
    f = plot_corner(pos_result.posterior, [.05, .50, .95], parvals)
    f.savefig(samples_file.replace('.txt','_corner.png'))
    pl.show()

    #
    # Text dump of summary statistics
    #
    f = open(samples_file.replace('.txt','_summary.txt'),'w')
    f.write("# name mean median std 67_low 67_upp 90_low 90_upp 95_low 95_upp\n")
    for param in params:
        f.write("{name} {mean} {median} {std} {low67} {upp67} {low90} {upp90} {low95} {upp95}\n".format(
                name=param,
                mean=pos_result.posterior[param].mean,
                median=pos_result.posterior[param].mean,
                std=pos_result.posterior[param].stdev,
                low67=pos_result.intervals[param][0][0],
                upp67=pos_result.intervals[param][0][1],
                low90=pos_result.intervals[param][1][0],
                upp90=pos_result.intervals[param][1][1],
                low95=pos_result.intervals[param][2][0],
                upp95=pos_result.intervals[param][2][1]))
    f.close()


    return pos_result

if __name__ == "__main__":

    pos_result = main()

