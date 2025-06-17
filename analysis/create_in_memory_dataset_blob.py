"""
This script creates an InMemoryDatasetBlob for use in training GNN-style models. We use this to convert the preprocessed (silver) AnnData batches into a collection of PyG Data objects with various features, labels, and edge indices. This collection of Data objects is then saved to disk in the "gold" directory as a single PyTorch Geometric Dataset.

Usage:
>>> python analysis/create_in_memory_dataset_blob.py --config_file config/create_in_memory_dataset_blob/sss2-1b_1p.yaml
"""
from vqniche.dataset.in_memory_dataset_blob import InMemoryDatasetBlob
from vqniche.utils.parse_datasetblob_configs import parse_datasetblob_arguments, collect_datasetblob_configs


def main(config: dict):
    print(f"Experiment: {config['experiment']['description']}")
    
    # configure parameters for creating the InMemoryDatasetBlob
    dataset_name = config['dataset']['name']
    feature_names = config['dataset']['feature_names']
    label_names = config['dataset']['label_names']
    graph_kwargs = config['dataset']['graph_kwargs']
    data_directory_path = config['dataset']['data_directory_path']
    pre_transform = config['dataset']['pre_transform']
    pre_filter = config['dataset']['pre_filter']
    overwrite = config['dataset']['overwrite']

    software_paths = config['software_paths']

    # initialize InMemoryDatasetBlob
    dataset_blob = InMemoryDatasetBlob(
                    name=dataset_name,
                    feature_names=feature_names,
                    label_names=label_names,
                    graph_kwargs=graph_kwargs,
                    data_directory_path=data_directory_path,
                    pre_transform=pre_transform,
                    pre_filter=pre_filter,
                    overwrite=overwrite,
                    software_paths=software_paths
                )
    
    for data_batch in dataset_blob:
        print(f"Batch: {data_batch.adata_batch_id}")
        print(f"Data: {data_batch}")
        print("")
    
    print(f"Processed data saved at {dataset_blob.processed_dir}")


if __name__ == '__main__':
    args = parse_datasetblob_arguments()
    config = collect_datasetblob_configs(args)
    
    main(config)