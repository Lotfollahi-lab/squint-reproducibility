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

`/lustre/scratch126/cellgen/team361/DATASETS/` houses all the datasets across experiments. A dataset is defined as a collection of AnnData objects that may be from the same or different datasets, species, tissues, gene panels, and batches. 
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

On Sanger's `farm22`, the recommended usage is to use the wrapper script as follows:
```
./scripts/wrapper_create_in_memory_dataset_blob.sh [DATASET-NAME]
```

The other datasets in the list above should work by appropriately adding/modifying config files and the wrapper script for job requirements.


## Training

To train one model on one section of data, for example, Batch 11 of `xhs1000_39b_1p`, execute the following:

```
python analysis/train_model.py --base_config_file config/train_model/xhs1000-39b-batch11_1p_vqniche_graphsage.yaml
```

Hyperparameters and other experiment configurations can be adjusted via the config file. Take note of the path to the WandB Run Directory. It will look something like this:
```
/nfs/team361/am84/VQNiche/logs/xhs1000-39b_1p/standalone/VQNiche/batch=[11]/spatial_n_neighs_8/wandb/offline-run-20250827_085244-f04x8bzi
```

## Testing

To test an instance of a previously trained model, use the WandB Run Directory as follows:
```
python analysis/test_model.py --wandb_run_dir [WANDB_RUN_DIR]
```