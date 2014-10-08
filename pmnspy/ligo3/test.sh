#!/bin/bash

waveforms="apr_135135 dd2_135135 shen_135135 nl3_135135"
for waveform in ${waveforms}
do
    #echo "---------- ${waveform}, aLIGO ------------"
    #pmns_ligo3.py ${waveform} 10 aLIGO

    echo "---------- ${waveform}, LIGOIII ------------"
    pmns_ligo3.py ${waveform} 10 basePSD

    echo ""
    echo ""
done
