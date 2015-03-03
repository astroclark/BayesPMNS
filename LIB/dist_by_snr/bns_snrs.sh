#!/bin/sh 
# inspinj.sh
# Makes a call to lalapps_inspinj with some standard naming conventions.
# James Clark, <james.clark@ligo.org>

# FRAMEPATH should contain the ninja frame and the ninja xml file created by
# fr_ninja.sh and xninja.sh, respectively.

gpsstart=`lalapps_tconvert now`
gpsend=`python -c "print ${gpsstart} + 24*60*60"`
seed=`lalapps_tconvert now`

lalapps_inspinj \
    --i-distr uniform  --seed 1551 \
    --snr-distr volume \
    --min-snr 100 --max-snr 100000 \
    --ligo-fake-psd LALAdLIGO \
    --ligo-start-freq 10 \
    --virgo-fake-psd LALAdVirgo \
    --virgo-start-freq 10 \
    --ifos H1,L1,V1 \
    --output bnsdistances.xml \
    --waveform TaylorF2threePointFivePN \
    --gps-start-time ${gpsstart} --gps-end-time ${gpsend} --time-step 60 \
    --time-interval 10 --l-distr random \
    --max-mass1 1.35 --max-mass2 1.35 \
    --min-mass1 1.35 --min-mass2 1.35 \
    --m-distr componentMass --f-lower 10 \
    --disable-spin \
    --verbose
