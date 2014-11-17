#!/bin/sh
# James Clark, <james.clark@ligo.org>


gpsstart=1097965864
gpsend=1097966376

framelength=256


start=${gpsstart}

while [ ${start} -lt ${gpsend} ]
do
	end=$((${start}+${framelength}))

	lalapps_mdc_ninja \
		--verbose --injection-type NR \
		--injection-file HL-INJECTIONS_101-1097965864-512.xml \
		--all-ifos \
		--gps-start-time ${start} --gps-end-time ${end}  \
		--sample-rate 16384 --write-mdc-log \
		--frame-type BNSBURST_EXAMPLE --set-name BNSBURST_EXAMPLE \
		--mdc-log BNSBURST_EXAMPLE-${start}-${framelength}.log \
		--freq-low-cutoff 100 --snr-low 0 --snr-high 1e6 \
		--fr-out-dir ./ --double-precision \
		--write-frame --verbose

	start=$((${start}+${framelength}))

done

