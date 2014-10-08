#!/usr/bin/python/

import numpy as np
from scipy import interpolate

# Load in mat file
inFile = 'BlueBird_rawdata_20140904.txt'
data = np.loadtxt(inFile)

# Separate out data into components
f_orig = data[:,0]
resgas_orig = data[:,1]
susptherm_orig = data[:,2]
quantum_orig = data[:,3]
newtonian_orig = data[:,4]
seismic_orig = data[:,5]
basePSD_orig = data[:,6]
mirrortherm_orig = data[:,7]
quantum_highloss_orig = data[:,8]
quantum_highmass_orig = data[:,9]
quantum_highpow_orig = data[:,10]
quantum_highsqz_orig = data[:,11]

# Interpolate data - use 0.5 Hz frequency resolution
df = 0.5
f = np.arange(f_orig[0],f_orig[-1],step=df)
tck_resgas = interpolate.splrep(f_orig,resgas_orig)
resgas = interpolate.splev(f,tck_resgas)
tck_susptherm = interpolate.splrep(f_orig,susptherm_orig)
susptherm = interpolate.splev(f,tck_susptherm)
tck_quantum = interpolate.splrep(f_orig,quantum_orig)
quantum = interpolate.splev(f,tck_quantum)
tck_newtonian = interpolate.splrep(f_orig,newtonian_orig)
newtonian = interpolate.splev(f,tck_newtonian)
tck_seismic = interpolate.splrep(f_orig,seismic_orig)
seismic = interpolate.splev(f,tck_seismic)
tck_basePSD = interpolate.splrep(f_orig,basePSD_orig)
basePSD = interpolate.splev(f,tck_basePSD)
tck_mirrortherm = interpolate.splrep(f_orig,mirrortherm_orig)
mirrortherm = interpolate.splev(f,tck_mirrortherm)
tck_highloss = interpolate.splrep(f_orig,quantum_highloss_orig)
quantum_highloss = interpolate.splev(f,tck_highloss)
tck_highmass = interpolate.splrep(f_orig,quantum_highmass_orig)
quantum_highmass = interpolate.splev(f,tck_highmass)
tck_highpow = interpolate.splrep(f_orig,quantum_highpow_orig)
quantum_highpow = interpolate.splev(f,tck_highpow)
tck_highsqz = interpolate.splrep(f_orig,quantum_highsqz_orig)
quantum_highsqz = interpolate.splev(f,tck_highsqz)

# Save base curve
N = len(f)
outArr = np.zeros((N,2),float)
outArr[:,0] = f 
outArr[:,1] = basePSD
np.savetxt('BlueBird_basePSD_20140904.txt',outArr)

# Construct curves for Jacobian
lo = 0.8**2	# Change by 20% to match quantum changes
hi = 1.2**2	# Change by 20% to match quantum changes
lowNN = resgas + susptherm + quantum + seismic + mirrortherm + newtonian/lo
highNN = resgas + susptherm + quantum + seismic + mirrortherm + newtonian/hi 
lowSei = resgas + susptherm + quantum + seismic/lo + mirrortherm + newtonian
highSei = resgas + susptherm + quantum + seismic/hi + mirrortherm + newtonian
lowSPOT = resgas + susptherm + quantum + seismic + mirrortherm/lo + newtonian
highSPOT = resgas + susptherm + quantum + seismic + mirrortherm/hi + newtonian
lowST = resgas + susptherm/lo + quantum + seismic + mirrortherm + newtonian
highST = resgas + susptherm/hi + quantum + seismic + mirrortherm + newtonian
highloss = resgas + susptherm + quantum_highloss + seismic + mirrortherm +\
           newtonian
highmass = resgas + susptherm + quantum_highmass + seismic + mirrortherm +\
           newtonian
highpow  = resgas + susptherm + quantum_highpow + seismic + mirrortherm +\
           newtonian
highsqz  = resgas + susptherm + quantum_highsqz + seismic + mirrortherm +\
           newtonian

# Save PSDs to file
outArr[:,1] = lowNN
np.savetxt('BlueBird_lowNN-PSD_20140904.txt',outArr)

outArr[:,1] = highNN
np.savetxt('BlueBird_highNN-PSD_20140904.txt',outArr)

outArr[:,1] = lowSei
np.savetxt('BlueBird_lowSei-PSD_20140904.txt',outArr)

outArr[:,1] = highSei
np.savetxt('BlueBird_highSei-PSD_20140904.txt',outArr)

outArr[:,1] = lowSPOT
np.savetxt('BlueBird_lowSPOT-PSD_20140904.txt',outArr)

outArr[:,1] = highSPOT
np.savetxt('BlueBird_highSPOT-PSD_20140904.txt',outArr)

outArr[:,1] = lowST
np.savetxt('BlueBird_lowST-PSD_20140904.txt',outArr)

outArr[:,1] = highST
np.savetxt('BlueBird_highST-PSD_20140904.txt',outArr)

outArr[:,1] = highloss
np.savetxt('BlueBird_highloss-PSD_20140904.txt',outArr)

outArr[:,1] = highmass
np.savetxt('BlueBird_highmass-PSD_20140904.txt',outArr)

outArr[:,1] = highpow
np.savetxt('BlueBird_highpow-PSD_20140904.txt',outArr)

outArr[:,1] = highsqz
np.savetxt('BlueBird_highsqz-PSD_20140904.txt',outArr)
