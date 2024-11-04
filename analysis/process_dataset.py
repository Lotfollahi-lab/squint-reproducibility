"""
This script processes the dataset for the experiment. It creates a CustomInMemoryDataset object and saves the processed data. Use this to process the raw data so that all the various required features, labels, and edge indices are precomputed and can be accessed quickly during training.

Usage:
>>> python analysis/process_dataset.py --config config/process_dataset/process_dataset_sss2-1b_1p.yaml
"""

from vqniche.dataloaders.custom_in_memory_dataset import CustomInMemoryDataset
from vqniche.utils.config_parsers import parse_arguments, collect_configs


def main(config: dict):
    # Get parameters for the Experiment
    experiment_name = config['experiment']['name']
    overwrite = config['experiment']['overwrite']
    print(f"Processing: {experiment_name}")
    
    # Get parameters for Data
    label_names = config['data']['label_names']
    graph_kwargs = config['data']['graph_kwargs']
    data_directory_path = config['data']['data_directory_path']
    pre_transform = config['data']['pre_transform']
    pre_filter = config['data']['pre_filter']

    dataset = CustomInMemoryDataset(name=experiment_name,
                                    label_names=label_names,
                                    graph_kwargs=graph_kwargs,
                                    data_directory_path=data_directory_path,
                                    pre_transform=pre_transform,
                                    pre_filter=pre_filter,
                                    overwrite=overwrite)

    print(f"Processed data saved at {dataset.processed_dir}")


if __name__ == '__main__':
    args = parse_arguments()
    config = collect_configs(args)
    
    main(config)