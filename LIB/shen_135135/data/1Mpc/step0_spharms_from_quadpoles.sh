#!/bin/bash
#
# Construct NINJA mode data from second time derivatives of quadrupole moments
# (Ixxdotdot, etc)

label="shen_135135"
datafile="secderivqpoles_16384Hz_shen_135135_lessvisc.dat"
extract_dist=20

./spharms_from_quadrupoles.py \
    --waveform-name ${label} \
    --extraction-distance ${extract_dist}\
    --ninja-units \
    ${datafile}

