#!/usr/bin/env python

import numpy as np

def quality(f,b):
    return f/b
def bandwidth(f,q):
    return f/q

bw_min=10.
bw_max=100.
f_min=1500.
f_max=4000.
q_min=40
q_max=150

bws = bw_min + (bw_max-bw_min)*np.random.random(1e5)
fs  = f_min  + (f_max-f_min)*np.random.random(1e5)
Qs  = q_min  + (q_max-q_min)*np.random.random(1e5)


bws_derived = bandwidth(fs,Qs)
qs_derived = quality(fs,bws)


print min(bws_derived)
print max(bws_derived)

f0s = [1500, 2000, 2500, 3000, 3500, 4000]

from matplotlib import pyplot as pl
bw = np.arange(10,100)
pl.figure()
for f0 in f0s:
    pl.plot(bw, quality(f0,bw), label='f=%d Hz'%f0)
pl.xlabel('bandwidth [Hz]')
pl.ylabel('quality')
pl.legend()

q = np.arange(40, 150, 0.1)
pl.figure()
for f0 in f0s:
    pl.plot(q, bandwidth(f0,q), label='f=%d Hz'%f0)
pl.ylabel('bandwidth [Hz]')
pl.xlabel('quality')
pl.legend()
pl.show()

