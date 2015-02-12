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
