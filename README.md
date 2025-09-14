# vqniche-reproducibility

## Software on Farm22

Conda Environment:
```
/software/cellgen/team361/am84/envs/vqniche-reproducibility
```

Modules:
```
module load cellgen/conda
module load cuda-12.1.1
```


## Datasets

List of datasets: https://docs.google.com/spreadsheets/d/1bfdBZ1MZEVKz2-4Ge89zwe5jVT4qnXAndStmXmJehTo/edit?gid=1836124407#gid=1836124407

`/lustre/scratch126/cellgen/lotfollahi/DATASETS/` houses all the datasets across experiments. A dataset is defined as a collection of AnnData objects that may be from the same or different datasets, species, tissues, gene panels, and batches. 
- `silver` -- contains preprocessed AnnData files
- `gold` -- contains the processed dataset object

## In-Memory Dataset-Blob

Execute the following to create a Pytorch Geometric In-Memory Dataset from the processed AnnData (`silver` to `gold`):
```
python analysis/create_in_memory_dataset_blob.py --config_file config/create_in_memory_dataset/[DATASET-NAME]
```
For an example of a config file, see `config/create_in_memory_dataset_blob/sss2-1b_1p.yaml`.
Currently, the following options for `DATASET-NAME` are tested:
- `sss2-1b_1p`
- `xhs1000-39b_1p`
- `mmb0-4b_1p`
- `xhk1020-CV1-CV2-5b_1p`
- `xhs1021-15b_1p`

On Sanger's `farm22`, the recommended usage is to use the wrapper script  that builds a `bjob` with appropriate choices for cores, RAM, queue, etc. as follows:
```
./scripts/wrapper_create_in_memory_dataset_blob.sh [DATASET-NAME] [CORES] [QUEUE]
```


## Train-Val-Test Pipeline

### Stand-alone Run

To train one instance of the model on a specific set of configuration parameters, use the following:
```
python analysis/train_model.py --base_config_file </path/to/config/file>
```

Hyperparameters and other experiment configurations can be adjusted via the config file. See `config/train_model/` for examples of config file. 

Config files should be named in the following format `[DATASET-NAME]_[SPLIT-NAME]_[MODEL-NAME].yaml`.

For a simple toy example, use:
`xhs1000-39b_1p-batch11_random-split_vqniche_graphsage.yaml`.

Take note of the path to the WandB Run Directory. It contains the model checkpoints, train logs, a copy of the user-specified config used for this run, etc.

### Sweep

To run a collection of models over a set of parameters, define a sweep config file and execute the training:
```
python analysis/train_model.py --base_config_file </path/to/config/file> --sweep_config_files <path/to/sweep/config/file>
```

For example, to ablate the backbone GNN (GraphSAGE vs GATv2 vs GIN), define `SWEEP_NAME` config file such as `config/sweeps/backbone_gnn.yaml`.

Use the following wrapper script to send a farm job:
```
./scripts/wrapper_sweep_train_model.sh [DATASET-NAME] [SPLIT-NAME] [MODEL-NAME] [SWEEP-NAME] [CORES] [QUEUE]
```

Multiple sweep config files is not tested.

## Predict Pipeline

To test an instance of a previously trained model, use the WandB Run Directory as follows:
```
python analysis/test_model.py --wandb_run_dir [WANDB_RUN_DIR]
```