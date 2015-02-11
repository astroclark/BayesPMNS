#!/bin/sh
# James Clark, <james.clark@ligo.org>


gpsstart=1101501504
gpsend=1101506504

framelength=256


start=${gpsstart}

while [ ${start} -lt ${gpsend} ]
do
	end=$((${start}+${framelength}))

	lalapps_mdc_ninja \
		--verbose --injection-type NR \
		--injection-file ${1} \
		--all-ifos \
		--gps-start-time ${start} --gps-end-time ${end}  \
		--sample-rate 16384 --write-mdc-log \
		--frame-type BNSBURST_EXAMPLE --set-name BNSBURST_EXAMPLE \
		--mdc-log BNSBURST_EXAMPLE-${start}-${framelength}.log \
		--freq-low-cutoff 10 --snr-low 0 --snr-high 1e6 \
		--fr-out-dir ./frames --double-precision \
		--write-frame --verbose

	start=$((${start}+${framelength}))

done

