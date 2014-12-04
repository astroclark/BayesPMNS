#!/bin/sh 
# inspinj.sh
# Makes a call to lalapps_inspinj with some standard naming conventions.
# James Clark, <james.clark@ligo.org>

# FRAMEPATH should contain the ninja frame and the ninja xml file created by
# fr_ninja.sh and xninja.sh, respectively.

gpsstart=1101501504
gpsend=1101506504

lalapps_inspinj \
    --i-distr uniform  --seed 101 \
    --waveform TaylorT2threePointFivePN \
    --gps-start-time ${gpsstart} --gps-end-time ${gpsend} --time-step 30 \
    --time-interval 10 --l-distr random --d-distr uniform \
    --min-distance 50000 --max-distance 50000 \
    --fixed-mass1 10 --fixed-mass2 10 \
    --m-distr fixMasses --f-lower 10 --disable-spin
