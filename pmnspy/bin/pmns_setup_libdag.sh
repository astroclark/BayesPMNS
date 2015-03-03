waveform=${1}
injdir=/home/jclark/PMNS/data/${waveform}/5Mpc

lalinference_pipe 5Mpc-${waveform}_standardSGF.ini \
    -r 5Mpc-${waveform}_standardSGF -p 5Mpc-${waveform}_standardSGF/logs \
    -I ${injdir}/*.xml
