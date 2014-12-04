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
    --waveform NumRelNinja2 \
    --gps-start-time ${gpsstart} --gps-end-time ${gpsend} --time-step 30 \
    --time-interval 10 --l-distr random --d-distr uniform \
    --min-distance 5000 --max-distance 5000 \
    --min-mtotal 1 --max-mtotal 1 \
    --m-distr nrwaves --f-lower 10 \
    --real8-ninja2 \
    --nr-file "shen_135135.xml" 
