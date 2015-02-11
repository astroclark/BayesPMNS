#!/bin/sh 
# inspinj.sh
# Makes a call to lalapps_inspinj with some standard naming conventions.
# James Clark, <james.clark@ligo.org>

# FRAMEPATH should contain the ninja frame and the ninja xml file created by
# fr_ninja.sh and xninja.sh, respectively.

gpsstart=1106316980
gpsend=1106403380

lalapps_inspinj \
    --i-distr uniform  --seed 1001 \
    --waveform NumRelNinja2 \
    --gps-start-time ${gpsstart} --gps-end-time ${gpsend} --time-step 30 \
    --time-interval 10 --l-distr random --d-distr uniform \
    --min-distance 1000 --max-distance 1000 \
    --min-mtotal 1 --max-mtotal 1 \
    --m-distr nrwaves --f-lower 10 \
    --real8-ninja2 \
    --nr-file "shen_135135.xml" 
