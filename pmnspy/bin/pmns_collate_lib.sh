#!/bin/bash

distances="5 7.5 10 12.5 15 17.5 20"
waveform=${1}

for distance in ${distances}
do
    pmns_collate_lib.py ${waveform} ${distance}
done
