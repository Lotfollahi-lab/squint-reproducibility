#!/bin/bash

CONFIG_FILE=$1  # path to the config file

# Modify to match your python environment activation command(s)
source /etc/profile.d/modules.sh
if [ "$USER" == "am84" ]; then
    module load cellgen/conda
    conda activate vqniche-reproducibility
elif [ "$USER" == "ls34" ]; then
    # edit
    echo "Load Python environment"
fi

python analysis/create_in_memory_dataset_blob.py \
    --config_file ${CONFIG_FILE}