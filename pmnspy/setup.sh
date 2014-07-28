#! /bin/sh
# Make env setup script
# James Clark, james.clark@ligo.org

# Get the location of the git repository by finding the full path of this file
# and stripping off the name
PMNSPY_PREFIX=`python -c "import os, sys; print os.path.realpath('${0}')" | sed 's|/setup.sh||g'`

# create an etc directory
test -d "${PMNSPY_PREFIX}/etc" || mkdir "${PMNSPY_PREFIX}/etc"

# define a variable to point to the NINJA ascii data.  This is useful for waveform generation scripts.
NINJA_ASCII="${PMNSPY_PREFIX}/../waveform_data/ninja_ascii"

echo "# add script directory to path" > "${PMNSPY_PREFIX}/etc/pmnspy-user-env.sh"
echo "export PATH=$PMNSPY_PREFIX/bin:\$PATH" >> "${PMNSPY_PREFIX}/etc/pmnspy-user-env.sh"
echo "export PYTHONPATH=$PMNSPY_PREFIX/bin:$PMNSPY_PREFIX/pmns_utils:\${PYTHONPATH}" >> "${PMNSPY_PREFIX}/etc/pmnspy-user-env.sh"
echo "# define variable for location of ninja ascii files" >> "${PMNSPY_PREFIX}/etc/pmnspy-user-env.sh"
echo "export NINJA_ASCII=${NINJA_ASCII}" >> "${PMNSPY_PREFIX}/etc/pmnspy-user-env.sh"

