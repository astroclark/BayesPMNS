#!/bin/bash

export BAYESWAVE_PREFIX=/home/jclark/src/lscsoft/bayeswave/trunk

bwb_pipe.py flow_1024-srate_8192_seglen-0.25.ini \
    --workdir flow_1024-srate_8192_seglen-0.25  \
    -I HL-INJECTIONS_1234-1126256640-4096.xml\
    --sim-data
