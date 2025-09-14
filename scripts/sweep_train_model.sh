#!/bin/bash

BASE_CONFIG_FILE=$1  # path to the config file
SWEEP_CONFIG_FILE=$2  # path to the sweep config file

# Modify to match your python environment activation command(s)
source /etc/profile.d/modules.sh
if [ "$USER" == "am84" ]; then
    module load cellgen/conda
    module load cuda-12.1.1
    conda activate vqniche-reproducibility
elif [ "$USER" == "ls34" ]; then
    # edit
    echo "Load Python environment"
fi

python analysis/train_model.py \
    --base_config_file ${BASE_CONFIG_FILE} \
    --sweep_config_files ${SWEEP_CONFIG_FILE}