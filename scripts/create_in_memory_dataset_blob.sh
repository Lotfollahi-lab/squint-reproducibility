#!/bin/bash

CONFIG_FILE=$1  # path to the config file

# Modify to match your python environment activation command(s)
source /etc/profile.d/modules.sh
if [ "$USER" == "am84" ]; then
    module load cellgen/conda
    module load cuda-12.1.1
    # The env was created as vqniche-reproducibility before the package was renamed.
    # Use the renamed env if it exists, else the original prefix, and stop if neither activates
    # (without this check the job would silently run on whatever python is on PATH).
    conda activate squint-reproducibility 2>/dev/null \
        || conda activate /software/cellgen/team361/am84/envs/vqniche-reproducibility \
        || { echo "ERROR: could not activate the squint-reproducibility conda env" >&2; exit 1; }
elif [ "$USER" == "ls34" ]; then
    # edit
    echo "Load Python environment"
fi

python analysis/create_in_memory_dataset_blob.py \
    --config_file ${CONFIG_FILE}