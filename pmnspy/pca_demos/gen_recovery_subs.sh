executable="/home/jclark308/Projects/BayesPMNS/pmnspy/pca_demos/pmns_pca_recovery.py"
eos_mass_file="/home/jclark308/Projects/BayesPMNS/pmnspy/pca_demos/eos_mass_pairs.txt"

npcs=${1}
dir_tag="montecarlo_matchbyNpc_LOO"
subfile="${dir_tag}.sub"
LOO=true


if [ "${LOO}" = true ]
then
    NPCseq=`seq 1 48`
else
    NPCseq=`seq 1 49`
fi

mkdir ${dir_tag}
pushd ${dir_tag}
mkdir "condor_logs"

echo '
executable='${executable}'
universe=vanilla
getenv=True
' > ${subfile}


while read eos_mass_pair 
do 

    for npc in ${NPCseq}
    do

        if [ "${LOO}" = true ]
        then
            args="aLIGO ${eos_mass_pair} ${npc} LOO"
        else
            args="aLIGO ${eos_mass_pair} ${npc} ALL"
        fi

        echo ${args}

        tag=`echo ${eos_mass_pair} | sed  's/ /_/g'`

        echo '

        arguments = '${args}'
        output = condor_logs/'${tag}'_'${npc}'.out
        error = condor_logs/'${tag}'_'${npc}'.err
        log = condor_logs/'${tag}'_'${npc}'.log
        queue 1
        ' >> ${subfile}

    done

done < ${eos_mass_file}


popd
