#! /bin/bash
#
# pmns_lib_pp_dag.sh
#
# Copyright (C) 2015-2016 James Clark <james.clark@ligo.org>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# ------------- #
#               #
# --- Input --- #
#               #

executable=`which pmns_lib_dists.py`
waveform=${1}
inputdir="${2}"
outputdir="${3}"

thresholds="0 1 2 3 4 5 6 7 8 9 10"

# --- End Input
if [ ! -d ${outputdir} ]
then
    mkdir -p ${outputdir}/condor_logs
else
    echo "warning ${outputdir} exists, move"
    exit
fi

# change to full path
inputdir=`readlink -f ${inputdir}`
outputdir=`readlink -f ${outputdir}`

dagseed=`python -c "import lal; import random; print random.randint(0,int(lal.GPSTimeNow()))"`
dagfile="${outputdir}/postproc_${waveform}.dag"
subfile="${outputdir}/postproc_${waveform}.sub"
shellfile="${outputdir}/postproc_${waveform}.sh"

rm -rf ${dagfile} ${subfile} ${shellfile} condor_logs

# Set up the shell file
shelltext="\
#!/bin/bash
#########################
# PMNS LIB: Shell  file #
#########################
"
echo "${shelltext}" > ${shellfile}


#
# Set up the sub file
#
# The header for the sub file
echo "writing condor submission file: ${subfile}"
subtext="\
#########################
# PMNS LIB: SUB file    #
#########################

executable = `which pmns_lib_dists.py`
universe   = vanilla 
arguments  = \$(macroarguments)
output     = condor_logs/libdists-\$(macroid)-\$(cluster)-\$(process).out
error      = condor_logs/libdists-\$(macroid)-\$(cluster)-\$(process).err
log        = condor_logs/libdists.log
getenv     = True

queue
"
echo "${subtext}" > ${subfile}


#
# Set up the DAG file
#
echo "writing dag file: ${dagfile}"

echo "on threshold ${threshold} of ${thresholds}"

for threshold in ${thresholds}
do

    # 
    # Dag writing
    #
    jobname="pmnslibdists-${threshold}"

    thisdir="${outputdir}/threshold-${threshold}"
    jobargs="${inputdir} ${waveform} ${thisdir} ${threshold}"

    mkdir -p ${thisdir}

    echo "JOB ${jobname} ${subfile}" >> ${dagfile}
    echo "VARS ${jobname} macroid=\"${jobname}\" macroarguments=\"${jobargs}\"" >> ${dagfile}
    echo "" >> ${dagfile}

    #
    # Shell writing
    #
    echo "${executable} ${jobargs}" >> ${shellfile}


done
