#!/bin/bash

waveforms="shen_135135 apr_135135 nl3_135135 dd2_135135"
for waveform in ${waveforms}
do
    echo "---------- ${waveform}, aLIGO ------------"
    pmns_ligo3.py ${waveform} 10 aLIGO

    echo "---------- ${waveform}, LIGOIII ------------"
    pmns_ligo3.py ${waveform} 10 ligo3_basePSD

    echo ""
    echo ""
done
