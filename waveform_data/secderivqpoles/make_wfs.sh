#!/bin/bash
source ~/etc/source_lalinference_burst.sh


for w in $1
do 
    # get rid of confusing bits
    wftag=`echo ${w} | sed "s/secderivqpoles_16384Hz_//g"`
    wftag=`echo ${wftag} | sed "s/ini//g"`
    wftag=`echo ${wftag} | sed "s/df//g"`
    wftag=`echo ${wftag} | sed "s/\.dat//g"`
    wftag=`echo ${wftag} | sed "s/tau//g"`

    # remove the . to avoid any confusion later
    wftag=`echo ${wftag} | sed "s/\./p/g"`


    # write the ini file
cat > ${wftag}.ini <<-EOF
mass-ratio = 1.0
mass-scale = 1
simulation-details = ${w}
nr-group = Bauswein

2,-2 = ${wftag}_l2m-2.asc
2,-1 = ${wftag}_l2m-1.asc
2,0 =  ${wftag}_l2m0.asc
2,1 =  ${wftag}_l2m1.asc
2,2 =  ${wftag}_l2m2.asc
EOF

    # create spharms
    if [[ ${w} == *hybrid* ]]
    then
        echo pylal_spharms_from_quadrupoles.py ${w} ${wftag}
        pylal_spharms_from_quadrupoles.py ${w} ${wftag}
    else
        echo pylal_spharms_from_quadrupoles.py secderivqpoles_16384Hz_${wftag}.dat ${wftag}
        pylal_spharms_from_quadrupoles.py secderivqpoles_16384Hz_${wftag}.dat ${wftag}
    fi


done

# move into ninja_ascii dir
#mv *ini ../ninja_ascii
#mv *asc ../ninja_ascii

