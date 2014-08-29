#! /bin/bash
#
# pmns_lib_dag.sh
#
# Copyright (C) 2014-2015 James Clark <james.clark@ligo.org>
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

# -------------#
#              #
# --- Input ---#
#              #

exececutable="${HOME}/bayespmns/pmnspy/bin/pmns_lib_analysis.py"
configfile="${PWD}/pmns_libwrap.ini"

ninject=10

distances="5 7.5 10 12.5 15 17.5 20"

bigname="dd2_135135-100_5-20"

# --- End Input

dagfile="${bigname}.dag"
subfile="${bigname}.sub"

#
# Set up the sub file
#
# The header for the sub file
echo "writing condor submission file: ${subfile}"
subtext="\
#########################
# PMNS LIB: SUB file #
#########################

executable = `which pmns_lib_analysis.py`
universe   = vanilla 
arguments  = \$(macroarguments)
output     = condor_logs/LIB_PMNS-\$(macroid)-\$(cluster)-\$(process).out
error      = condor_logs/LIB_PMNS-\$(macroid)-\$(cluster)-\$(process).err
log        = condor_logs/LIB_PMNS.log
getenv     = True

queue
"
echo "${subtext}" > ${subfile}

mkdir condor_logs

#
# Set up the DAG file
#
echo "writing dag file: ${dagfile}"

for dist in ${distances}
do
    echo "on distance ${dist} of ${distances}"

    for i in `seq 1 ${ninject}`
    do

        initseed=`python -c "import lal; import random; print random.randint(0,int(lal.GPSTimeNow()))"`
        jobname="${bigname}-${initseed}"

        jobargs="--fixed-distance ${dist} --init-seed ${initseed} ${configfile}"

        echo "JOB ${jobname} ${subfile}" >> ${dagfile}
        echo "VARS ${jobname} macroid=\"${jobname}\" macroarguments=\"${jobargs}\"" >> ${dagfile}
        #echo "RETRY ${jobname} 3 " >> ${dagfile}
        echo "" >> ${dagfile}
    done

done


