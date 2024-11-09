"""
This script creates an InMemoryDatasetBlob for use in training GNN-style models. We use this to convert the preprocessed (silver) AnnData batches into a collection of PyG Data objects with various features, labels, and edge indices. This collection of Data objects is then saved to disk in the "gold" directory as a single PyTorch Geometric Dataset.

Usage:
>>> python analysis/create_in_memory_dataset_blob.py --config config/in_memory_dataset_blob/in_memory_dataset_blob_sss2-1b_1p.yaml
"""

from vqniche.dataloaders.in_memory_dataset_blob import InMemoryDatasetBlob
from vqniche.utils.config_parsers import parse_arguments, collect_configs


def main(config: dict):
    print(f"Experiment: {config['experiment']['description']}")
    
    # Configure parameters for creating the InMemoryDatasetBlob
    dataset_name = config['data']['dataset_name']
    feature_names = config['data']['feature_names']
    label_names = config['data']['label_names']
    graph_kwargs = config['data']['graph_kwargs']
    data_directory_path = config['data']['data_directory_path']
    pre_transform = config['data']['pre_transform']
    pre_filter = config['data']['pre_filter']
    overwrite = config['data']['overwrite']

    dataset = InMemoryDatasetBlob(
                    name=dataset_name,
                    feature_names=feature_names,
                    label_names=label_names,
                    graph_kwargs=graph_kwargs,
                    data_directory_path=data_directory_path,
                    pre_transform=pre_transform,
                    pre_filter=pre_filter,
                    overwrite=overwrite
                )

    print(f"Processed data saved at {dataset.processed_dir}")


if __name__ == '__main__':
    args = parse_arguments()
    config = collect_configs(args)
    
    main(config)