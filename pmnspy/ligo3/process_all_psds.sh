#!/bin/bash

waveforms="apr_135135 dd2_135135 shen_135135 nl3_135135"
#noisecurves="base highNN highSPOT highST highSei highloss highmass highpow highsqz lowNN lowSPOT lowST lowSei"
noisecurves="Red Green"

for waveform in ${waveforms}
do
    for noisecurve in ${noisecurves}
    do 

        echo "---------- ${waveform}, ${noisecurve} ------------"
        pmns_ligo3.py ${waveform} 10 ${noisecurve}

    done

    echo ""
    echo ""
done
