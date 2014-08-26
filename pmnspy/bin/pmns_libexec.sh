#!/bin/bash

source ~/etc/source_lalinference_burst.sh

#######################
# Input

# --- Timing & analysis freq
trigtime=32 #${1}
psdstart=0 #${2}
seed=101 #${3}
dt=0.01 #${4}

cachefile="H-H1_dd2_135135_lessvisc_101-0-64.lcf" #${5}

# XXX: Hardcoded from here

channel="H1:STRAIN"

seglen=4

psdlength=64
srate=8192
flow=1000
padding=0.5

# --- priors
fmin=1500
fmax=4000

qmin=10
qmax=500

loghrssmin=-53.5
loghrssmax=-46.5


#######################
# Execute

which lalinference_nest
#lalinference_nest \
#    --progress  \
#    --trigtime ${trigtime} --seglen ${seglen} \
#    --psdstart ${psdstart} --psdlength ${psdlength} \
#    --ifo [H1] --cache [${cachefile}] --channel [${channel}] \
#    --flow [${flow}]  --srate ${srate} \
#    --fmin ${fmin} --fmax ${fmax} --qmin ${qmin} --qmax ${qmax} \
#    --loghrssmin ${loghrssmin} --loghrssmax ${loghrssmax} \
#    --dt ${dt} \
#    --approx DampedSinusoidF \
#    --nlive 256 --nmcmc 256 \
#    --H1-timeslide 0 \
#    --padding ${padding} \
#    --randomseed ${seed}  \
#    --outfile lalinferencenest-${trigtime}.dat 

lalinference_nest --psdlength 64  --qmin 2 --nlive 256 --srate 8192 --seglen 5.0 --cache [LALSimLIGO] --trigtime 966383951.0 --approx SineGaussianF --psdstart 966383883.5 --progress  --channel [H1:SIM-STRAIN] --loghrssmax -46.5 --timeslide [0] --fmax 1300 --padding 0.5 --outfile lalinferencenest-0-H1L1-966383951.0-1.dat --dt 0.1  --randomseed 595564537 --dataseed 1 --nmcmc 256 --flow [30] --qmax 35 --fmin 30 --loghrssmin -53.5 --ifo [H1]



