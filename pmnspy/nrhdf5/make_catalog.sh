#!/bin/bash

#source ~/.local/etc/pycbc-user-env.sh


#h5file="/data/lvc_nr/GaTech/GT0887.h5"
#catfile="GT0887.xml.gz"
h5file="tm1_135135.h5"
catfile="tm1_135135.xml.gz"

python ~/src/lscsoft/pycbc/bin/pycbc_make_nr_hdf_catalog \
    --output-file ${catfile} \
    --input-files ${h5file}

