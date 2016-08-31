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

bwb_pipe.py cbc_nrhdf5_injection.ini \
    --workdir cbc_nrhdf5_injection  \
    -I HL-INJECTIONS_1156676448-1126256640-4096.xml \
    --sim-data
