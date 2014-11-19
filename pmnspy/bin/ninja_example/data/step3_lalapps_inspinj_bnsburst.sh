#!/bin/sh 
# inspinj.sh
# Makes a call to lalapps_inspinj with some standard naming conventions.
# James Clark, <james.clark@ligo.org>

# FRAMEPATH should contain the ninja frame and the ninja xml file created by
# fr_ninja.sh and xninja.sh, respectively.

gpsstart=1097965864
gpsend=1097966376

lalapps_inspinj \
    --i-distr uniform  --seed 101 \
    --waveform NumRelNinja2 \
    --gps-start-time ${gpsstart} --gps-end-time ${gpsend} --time-step 60 \
    --time-interval 10 --l-distr random \
    --min-mtotal 1 --max-mtotal 1 \
    --m-distr nrwaves --f-lower 10 \
    --real8-ninja2 \
    --nr-file "dd2_135135.xml"  \
    --ifos H1,L1 \
    --snr-distr uniform --min-snr 10 --max-snr 10 \
    --ninja-snr --ligo-start-freq 1000 --ligo-psd ZERO_DET_high_P_PSD.txt 
#    --min-distance 500 --max-distance 5000 --d-distr uniform
