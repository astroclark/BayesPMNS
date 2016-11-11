#!/usr/bin/env python

from pycbc import pnutils
import numpy as np
import h5py
from matplotlib import pyplot as pl
from pmns_utils import pmns_waveform as pwave
import pycbc.filter


#filepath = '/home/jclark/Projects/pmnsdata/bns_apr4_4p_mb_1.42_1.59_irrot_trento.h5'
#mass1=1.42
#mass2=1.59

filepath = '/home/jclark/Projects/pmnsdata/bns_ls220_mb1.5_irrot_trento.h5'
mass1=1.546
mass2=1.546

#   filepath = '/home/jclark/Projects/pmnsdata/bns_ls220_mb1.5_spinf0.5_trento.h5'
#   filepath = '/home/jclark/Projects/pmnsdata/bns_ls220_mb1.7_irrot_trento.h5'
#   filepath = '/home/jclark/Projects/pmnsdata/bns_ls220_mb1.8_irrot_trento.h5'
#   filepath = '/home/jclark/Projects/pmnsdata/bns_sht_mb2.0_irrot_trento.h5'
#   filepath = '/home/jclark/Projects/pmnsdata/bns_sht_mb2.0_spinf0.5_trento.h5'
#   filepath = '/home/jclark/Projects/pmnsdata/bns_sht_mb2.2_irrot_trento.h5'

filepath = '/home/jclark/Projects/pmnsdata/thessaloniki_hdf5/shen_135135_thessaloniki.h5'
mass1=1.35
mass2=1.35

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

params['mass1'] = mass1
params['mass2'] = mass2

params['f_lower'] = 1000.
params['inclination'] = 0.0
params['distance'] = 50.0

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
Hp = hp.to_frequencyseries()

psd = pwave.make_noise_curve(fmax=Hp.sample_frequencies.max(),
        delta_f=Hp.delta_f, noise_curve='aLIGO')

SNR = pycbc.filter.sigma(Hp, psd=psd, low_frequency_cutoff=1000)
print SNR


f,ax=pl.subplots()
ax.plot(Hp.sample_frequencies, 2*abs(Hp)*np.sqrt(Hp.sample_frequencies))
ax.plot(psd.sample_frequencies, np.sqrt(psd))
ax.set_ylim(0,5e-23)
pl.show()



