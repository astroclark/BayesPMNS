#!/usr/bin/env python

from pycbc import pnutils
import h5py

filepath = '/home/jclark/Projects/BayesPMNS/pmnspy/geohf/tm1_135135.h5'

f = h5py.File(filepath, 'r')

params = {}

# Metadata parameters:

params['eta'] = f.attrs['eta']
params['spin1x'] = f.attrs['spin1x']
params['spin1y'] = f.attrs['spin1y']
params['spin1z'] = f.attrs['spin1z']
params['spin2x'] = f.attrs['spin2x']
params['spin2y'] = f.attrs['spin2y']
params['spin2z'] = f.attrs['spin2z']

params['coa_phase'] = f.attrs['coa_phase']

# Extrinsic parameters:

params['mtotal'] = 2.7

params['mass1'] = pnutils.mtotal_eta_to_mass1_mass2(params['mtotal'], params['eta'])[0]
params['mass2'] = pnutils.mtotal_eta_to_mass1_mass2(params['mtotal'], params['eta'])[1]

params['f_lower'] = 1000.
params['inclination'] = 0.0
params['distance'] = 100.0

f.close()


from pycbc.waveform import get_td_waveform

hp, hc = get_td_waveform(approximant='NR_hdf5',
                                 numrel_data=filepath,
                                 mass1=params['mass1'],
                                 mass2=params['mass2'],
                                 spin1z=params['spin1z'],
                                 spin2z=params['spin2z'],
                                 delta_t=1.0/16384.,
                                 f_lower=1000,
                                 inclination=params['inclination'],
                                 coa_phase=params['coa_phase'],
                                 distance=params['distance'])
hp.data *= 1.4
Hp = hp.to_frequencyseries()

from matplotlib import pyplot as pl
f, ax = pl.subplots(nrows=1, ncols=2)
ax[0].plot(hp.sample_times, hp)
ax[0].set_ylim(-2e-22, 2e-22)
ax[0].set_ylim(-1.5e-22, 1.5e-22)
ax[1].loglog(Hp.sample_frequencies, abs(Hp))
ax[1].loglog(waveform_tilde.sample_frequencies, abs(waveform_tilde))
pl.show()
