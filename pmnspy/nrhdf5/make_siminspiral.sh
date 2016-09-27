#!/bin/bash


totalmass=2.7


seed=`lalapps_tconvert now`
gpsstart=1126256640 # first CO2 frame for O1
gpsend=`python -c "print ${gpsstart} + 4096"`

#nrfile="GT0887.xml.gz"
nrfile="tm1_135135.xml.gz"


lalapps_inspinj \
    --seed ${seed} --f-lower 0.1 --gps-start-time ${gpsstart} \
    --gps-end-time ${gpsend} --waveform NR_hdf5threePointFivePN \
    --amp-order 0 \
    --ninja2-mass --nr-file ${nrfile} \
    --time-step 100 --time-interval 10 --l-distr random \
    --i-distr uniform \
    --m-distr nrwaves --disable-spin \
    --min-mtotal ${totalmass} --max-mtotal ${totalmass}\
    --taper-injection start  \
    --dchirp-distr uniform  --min-distance 2500 --max-distance 2500 --verbose #\
#       --snr-distr volume \
#       --min-snr 15 --max-snr 15 \
#       --ligo-psd aligopsd.txt \
#       --ligo-start-freq 30 \
#       --ifos H1,L1 \
#       --ninja-snr 

