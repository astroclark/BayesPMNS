#!/bin/sh
# James Clark <james.clark@ligo.org>

# location of NINJA-format ascii files
example_name="shen_135135"
nrpath="./"

lalapps_fr_ninja \
    --verbose --format NINJA2 \
    --double-precision \
    --nr-data-dir ${nrpath} \
    --nr-meta-file ${nrpath}/${example_name}.ini \
    --output "${example_name}.gwf"
