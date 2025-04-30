# vqniche-reproducibility

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


## Results

### Pearson Scores
- No Cross Entropy Loss
- Num Epochs: 20

| Model | `sss2-1b_1p` | `xhs1000-39b-batch11_1p` | `xhs1000-39b-batch1_1p` |
|-------|-------------------|--------------|--------------|
| MLP | 0.8055 | 0.8681 | 0.8229 |
| GraphSAGE | 0.7563 | 0.8215 | 0.8101 |
| VQNiche-MLP | | | |
| VQNiche-GraphSAGE | | | |
