"""
This script creates an OnDiskDatasetBlob for use in training GNN-style models. We use this to convert the preprocessed (silver) AnnData batches into a collection of PyG Data objects with various features, labels, and edge indices. This collection of Data objects is then saved to disk in the "gold" directory as a single PyTorch Geometric Dataset.

Usage:
>>> python analysis/create_on_disk_dataset_blob.py --config_file config/create_on_disk_dataset_blob/sss2-1b_1p.yaml

"""
import sys
from pathlib import Path

# Make the local `vqniche/src` visible when running this script directly
# (developer convenience). Prefer `pip install -e /path/to/vqniche` for
# permanent installs in environments.
_repo_src = "/lustre/scratch126/cellgen/lotfollahi/am84/riley_jung/vqniche/src"
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from vqniche.dataset.on_disk_dataset_blob import OnDiskDatasetBlob
from vqniche.utils.parse_datasetblob_configs import parse_datasetblob_arguments, collect_datasetblob_configs


def main(config: dict):
    print(f"Experiment: {config['experiment']['description']}")
    
    # configure parameters for creating the OnDiskDatasetBlob
    dataset_name = config['dataset']['name']
    feature_names = config['dataset']['feature_names']
    label_names = config['dataset']['label_names']
    graph_kwargs = config['dataset']['graph_kwargs']
    data_directory_path = config['dataset']['data_directory_path']
    pre_transform = config['dataset']['pre_transform']
    pre_filter = config['dataset']['pre_filter']
    overwrite = config['dataset']['overwrite']
    num_graphs_to_load = config['dataset'].get('num_graphs_to_load', 0)

    software_paths = config['software_paths']

    import tracemalloc
    import psutil
    import os
    tracemalloc.start()
    
    process = psutil.Process(os.getpid())
    
    # initialize OnDiskDatasetBlob
    dataset_blob = OnDiskDatasetBlob(
                    name=dataset_name,
                    feature_names=feature_names,
                    label_names=label_names,
                    graph_kwargs=graph_kwargs,
                    data_directory_path="/lustre/scratch126/cellgen/lotfollahi/rj5/DATASETS",
                    pre_transform=pre_transform,
                    pre_filter=pre_filter,
                    overwrite=overwrite,
                    software_paths=software_paths, 
                    num_graphs_to_load=num_graphs_to_load,
                )
    
    # Memory during creation
    current, peak = tracemalloc.get_traced_memory()
    rss = process.memory_info().rss / 1024**2
    print(f"Memory after dataset creation: tracemalloc current {current / 1024**2:.2f} MB, peak {peak / 1024**2:.2f} MB; RSS {rss:.2f} MB")
    
    tracemalloc.reset_peak()
    
    for data_batch in dataset_blob:
        print(f"Batch: {data_batch.adata_batch_id}")
        print(f"Data: {data_batch}")
        print("")
    
    # Memory during iteration
    current, peak = tracemalloc.get_traced_memory()
    rss = process.memory_info().rss / 1024**2
    print(f"Memory during iteration: tracemalloc current {current / 1024**2:.2f} MB, peak {peak / 1024**2:.2f} MB; RSS {rss:.2f} MB")
        
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    output_path = "/lustre/scratch126/cellgen/lotfollahi/rj5/logs/memory/on_disk_memory_report.txt"

    with open(output_path, "w") as f:
        f.write("[ Top 10 memory allocations ]\n")
        for stat in top_stats[:10]:
            f.write(str(stat) + "\n")

    print(f"Memory report saved to {output_path}")

if __name__ == '__main__':
    args = parse_datasetblob_arguments()
    config = collect_datasetblob_configs(args)
    
    main(config)
    