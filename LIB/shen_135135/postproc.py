#!/usr/bin/env python

import os, sys
import cPickle as pickle
import numpy as np

possampfile='posterior_samples.dat'
subdir='V1H1L1'

resultsdirs=os.listdir(sys.argv[1])

allfreqdata=[]
logBsn=[]
logBci=[]
for r, resdir in enumerate(resultsdirs):

        try:
            freqdata = np.loadtxt('./%s/%s/%s'%(resdir, subdir, possampfile),
                    skiprows=1, usecols=[1])
            allfreqdata.append(freqdata)
        except:
            continue


pickle.dump(allfreqdata, open("allfreqdata.pickle", "wb"))
