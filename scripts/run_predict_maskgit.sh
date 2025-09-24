#!/bin/bash
#BSUB -q gpu-cellgen-restricted # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1:block=yes' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -o predict_maskgit_vqniche_%J.out # output file
#BSUB -e predict_maskgit_vqniche_%J.err # error file
#BSUB -M 64G  # RAM memory part 2. Default: 100MB
#BSUB -R 'span[ptile=16]'  # Allocate 16 CPU cores per node
#BSUB -R 'select[mem>64G] rusage[mem=64G]' # RAM memory part 1. Default: 100MB
#BSUB -J predict_maskgit_vqniche # job name


set -eo pipefail
echo "I'm predict_maskgit script"
echo "I'm Job ID = $LSB_JOBID"
echo "I'm running on $HOSTNAME"

# activate pyenv
source /software/cellgen/team361/hk11/venvs/cellgen_venv/bin/activate

export WANDB_DIR=/lustre/scratch126/cellgen/lotfollahi/hk11/.wandb

module load cellgen/conda
# module load cuda-12.1.1
conda activate vqniche-reproducibility


# run script
echo "--- Start predict with maskgit model"

python analysis/maskgit_generation/test_model.py \
--wandb_run_dir "/nfs/team361/am84/VQNiche/logs/xhs1000-39b_1p/standalone/VQNiche/batch=[4, 8, 9, 14, 20, 21, 25, 33, 5, 12, 13, 17, 23, 26, 35, 38, 1, 7, 10, 18, 24, 27, 0, 6, 15, 16, 22, 29, 36, 37]/spatial_n_neighs_8/wandb/offline-run-20250921_163037-zcsx5q7p" \
--model_ckpt_fname "/nfs/team361/am84/VQNiche/logs/xhs1000-39b_1p/standalone/VQNiche/batch=[4, 8, 9, 14, 20, 21, 25, 33, 5, 12, 13, 17, 23, 26, 35, 38, 1, 7, 10, 18, 24, 27, 0, 6, 15, 16, 22, 29, 36, 37]/spatial_n_neighs_8/wandb/offline-run-20250921_163037-zcsx5q7p/files/checkpoints/epoch=4-train_pearson_1hop_nbr=0.74.ckpt" \
--batch_size 256 \
--output_path "/lustre/scratch126/cellgen/lotfollahi/hk11/maskgit_vqniche_out" \
--depth 6 \
--cond_drop_prob 0.10 \
--learning_rate 0.0001 \
--weight_decay 0.001 \
--cond_scale 3 \
--patients "train" \
--experiment_name "lr_0.0001_wd_0.001_depth_6_cond_drop_prob_0.1_20250923_1923_maskgit" \
--run_id "cond_scale_3_train_patients" \
--compute_metrics

echo "--- Finished predict with maskgit model"