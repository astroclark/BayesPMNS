# add script directory to path
export PATH=/home/jclark/Projects/BayesPMNS/pmnspy/bin:$PATH
export PYTHONPATH=/home/jclark/Projects/BayesPMNS/pmnspy/bin:/home/jclark/Projects/BayesPMNS/pmnspy:${PYTHONPATH}
export PYTHONPATH=${PYTHONPATH}:/home/jclark/lib/python2.7/site-packages
export PYTHONPATH=${PYTHONPATH}:${HOME}/src/lscsoft/romspline
# define variable for location of waveform data
export PMNSPY_PREFIX=/home/jclark/Projects/BayesPMNS/pmnspy
# LAL/pycbc
export LALSUITE_PREFIX=${HOME}/opt/lscsoft/lalsuite-6.38_NR-taper-patch
source ${LALSUITE_PREFIX}/etc/lalsuiterc
source ${LALSUITE_PREFIX}/pylal/etc/pylal-user-env.sh
source ${LALSUITE_PREFIX}/glue/etc/glue-user-env.sh
export PYCBC_PREFIX=${HOME}/opt/lscsoft/pycbc_nr
source ${PYCBC_PREFIX}/etc/pycbc-user-env.sh
