#!/bin/sh 
# inspinj.sh
# Makes a call to lalapps_inspinj with some standard naming conventions.
# James Clark, <james.clark@ligo.org>

# FRAMEPATH should contain the ninja frame and the ninja xml file created by
# fr_ninja.sh and xninja.sh, respectively.

gpsstart=1106316980
gpsend=1106328980

lalapps_inspinj \
    --i-distr uniform  --seed 1551 \
    --output bnsdistances.xml \
    --waveform TaylorF2threePointFivePN \
    --gps-start-time ${gpsstart} --gps-end-time ${gpsend} --time-step 30 \
    --time-interval 10 --l-distr source \
    --max-mass1 1.35 --max-mass2 1.35 \
    --min-mass1 1.35 --min-mass2 1.35 \
    --m-distr componentMass --f-lower 10 \
    --disable-spin \
    --verbose \
    --d-distr source \
    --source-file inspsrcs100Mpc.errors \
    --disable-milkyway  \
    --sourcecomplete 10000
    #--d-distr volume \
    #--min-distance 1000 --max-distance 100000 \
    #--dchirp-distr volume \
    #--min-distance 1000 --max-distance 100000 \
