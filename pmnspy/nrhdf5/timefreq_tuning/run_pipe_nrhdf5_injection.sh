#!/bin/bash
#
########################################################################
#                                                                      #
# Example BayesWave pipeline run for NR injection in SIMULATED data.   #   
#                                                                      # 
# RUN TIME XX hours                                                    #  
#  - Processor: 2 GHz Intel Core i7                                    #
#  - Memory: 4 GB 1600 Mhz                                             #
#                                                                      #
########################################################################

source ~jclark/etc/source_master.sh


#
# Setup the pipeline.  
#
source ~jclark/etc/bayeswave_ldg-user-env.sh

for ini in *ini
do

    workdir=`echo ${ini} | sed 's/.ini//'`

    bwb_pipe.py ${ini} \
        --workdir ${workdir} \
        -I HL-INJECTIONS_1158947321-1126256640-4096.xml\
        --sim-data

done
