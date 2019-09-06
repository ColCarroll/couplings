#! /bin/bash

set -ex # fail on first error, print commands


# This script should create all images used in the static site
# Run it every once in a while and fix errors!

SRC_DIR=${SRC_DIR:-$(pwd)}

NOTEBOOKS=(
    "maximal_coupling_images.ipynb"  # images related to maximal couplings
)

for ipynb in "${NOTEBOOKS[@]}"; do
    python -m jupyter nbconvert --to notebook --execute "${SRC_DIR}"/notebooks/"${ipynb}" --stdout > /dev/null
done
