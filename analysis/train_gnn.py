"""
This script trains one Graph Neural Network (GNN) model on one batch of AnnData.

We use PyTorch Lightning to support distributed data parallel (DDP) training across multiple GPUs.

It is required that the data has previously been processed and stored as a PyTorch Geometric dataset in the "gold" directory.  and PyG dataset is already processed and saved in the gold data directory.

Usage:
>>> python analysis/train_gnn.py --config_file config/train_models/sss2-1b_1p_graphsage.yaml
"""
import os
import tracemalloc
import wandb
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
import torch_geometric.transforms as T

from vqniche.utils.config_parsers import parse_arguments, collect_configs
from vqniche.preprocessors.graph_constructors import set_edge_index_name
from vqniche.dataloaders.transforms import SetExperimentDataKeys, init_data_transforms
from vqniche.dataloaders.in_memory_dataset_blob import InMemoryDatasetBlob
from vqniche.dataloaders.in_memory_datamodule import InMemoryDataModule
from vqniche.models.graphsage import GraphSAGE
from vqniche.models.vqgraph import VQGraph


def main(config: dict):

    # --------------------- Configure Backend ---------------------
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    num_cores = int(os.environ.get("LSB_DJOB_NUMPROC", 1))
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Number of CPU cores: {num_cores}")
    print(f"Number of GPU devices: {num_gpus}")
    
    # --------------------- Determinism Settings ---------------------
    # Set seed for reproducibility
    seed = config['experiment']['seed']
    pl.seed_everything(seed)
    
    # Set backend deterministic as true
    torch.backends.cudnn.deterministic = True
    
    print(f"Seed: {seed}")

    # --------------------- Define Experiment Parameters ---------------------
    experiment_name = config['experiment']['name']
    dataset_name = config['dataset']['name']
    batch_idx = config['datamodule']['batch_idx']
    model_name = config['model']['name']

    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Batch(es): {batch_idx}")
    print(f"Model: {model_name}")

    # --------------------- Wandb and Logger ---------------------
    # set logging directory
    log_dir = Path(config['logging']['log_dir']) / dataset_name / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # configure PyTorch Lightning Logger parameters
    log_model = config['logging']['log_model']
    offline = config['logging']['offline']
    wandb_tags = [
                    experiment_name,
                    dataset_name,
                    model_name,
                    f"batch{batch_idx}"
                ]
    
    # initialize wandb logger
    if config['logging']['mode'] == 'disabled':
        wandb_logger = None
    elif config['logging']['mode'] == 'enabled':
        wandb_logger = WandbLogger(
                            project="VQNiche",
                            save_dir=log_dir,
                            log_model=log_model,
                            offline=offline,
                            tags=wandb_tags,
                        )
    
    # --------------------- Set Data Keys ---------------------
    # decide which node features to use in this experiment
    feature_name = config['dataset']['feature_name']

    # decide which node labels to use in this experiment
    label_name = config['dataset']['label_name']

    # decide which edge index to use in this experiment
    graph_params = config['dataset']['graph_params']
    spatial_key = graph_params['spatial_key']
    delaunay = graph_params['delaunay']
    n_neighs = graph_params['n_neighs']
    radius = graph_params['radius']

    assert delaunay or n_neighs or radius, "Specify at least one of delaunay, n_neighs, or radius."

    edge_index_name = set_edge_index_name(
                        spatial_key=spatial_key,
                        delaunay=delaunay,
                        n_neighs=n_neighs,
                        radius=radius,
                    )

    print(f"Feature Name: {feature_name}")
    print(f"Label Name: {label_name}")
    print(f"Graph Name: {edge_index_name}")

    # --------------------- Initialize Dataset Transforms ---------------------
    ExperimentDataKeys = SetExperimentDataKeys(
                            feature_name=feature_name,
                            label_name=label_name,
                            edge_index_name=edge_index_name
                        )
    
    # e.g. normalize features , train-val-test split, etc.
    data_transform_names = config['dataset']['transform_names']
    transform_params = config['dataset']['transform_params']
    DataTransforms = init_data_transforms(
                        data_transform_names=data_transform_names,
                        **transform_params
                    )

    # initialize a composed transform
    transforms = T.Compose([ExperimentDataKeys] + DataTransforms)

    tracemalloc.start()
    # --------------------- Initialize Dataset Blob ---------------------
    # set root data directory
    data_directory_path = config['dataset']['data_directory_path']
    
    # initialize pytorch geometric dataset blob stored at:
    # data_directory_path / 'gold' / 'in-memory-PyG-dataset-blob' / dataset_name / 'dataset_blob.pt'
    dataset_blob = InMemoryDatasetBlob(
                        name=dataset_name,
                        data_directory_path=data_directory_path,
                        transform=transforms
                    )

    # --------------------- Load Data (one batch) ---------------------
    # load PyG data object corresponding to batch_idx (e.g. AnnData batch0)
    # TODO: ensure that the batch_idx is valid from the dataset_blob
    data_batch = dataset_blob[batch_idx]
    print(f"Data Batch: {data_batch}")
    print(data_batch.train_mask)
    print(data_batch.val_mask)
    print(data_batch.test_mask)
    # assert data_batch['batch'] == f"batch{batch_idx+1}", "Batch ID mismatch."
    # print(f"Batch ID: {data_batch['batch']}")

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage after loading data_batch is {current / 10**6}MB")
    print(f"Peak memory usage during data loading was {peak / 10**6}MB.")
    # --------------------- Initialize Lightning DataModule ---------------------
    # set parameters for data loader and sampler for training, validation, and testing
    loader_name = config['datamodule']['loader_name']
    loader_params = config['datamodule']['loader_params']

    sampler_name = config['datamodule']['sampler_name']
    sampler_params = config['datamodule']['sampler_params']

    inference_params = config['datamodule']['inference_params']

    datamodule_batch = InMemoryDataModule(
                            data=data_batch,
                            loader_name=loader_name,
                            loader_params=loader_params,
                            sampler_name=sampler_name,
                            sampler_params=sampler_params,
                            **inference_params,
                        )

    # --------------------- Initialize Model ---------------------
    # get model, optimizer, loss, and task parameters
    model_params = config['model']['model_params']
    optimizer_params = config['model']['optimizer_params']
    loss_params = config['model']['loss_params']
    task_params = config['model']['task_params']
    inference_params = config['model']['inference_params']

    # initialize model 
    if model_name == 'GraphSAGE':
        Model = GraphSAGE
    elif model_name == 'VQGraph':
        Model = VQGraph
    else:
        raise ValueError(f"Model {model_name} not found.")
    model = Model(
                name=model_name,                
                in_channels=data_batch.num_features,
                out_channels=data_batch.num_classes,
                **model_params,
                **optimizer_params,
                **loss_params,
                **task_params,
                **inference_params,
            )

    # log model architecture
    if wandb_logger is not None:
        wandb_logger.watch(model)

    # --------------------- Initialize Trainer ---------------------
    # configure model checkpointing
    ckpt_log_dir = Path(wandb.run.dir) / 'checkpoints'
    ckpt_log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_params = config['trainer']['checkpoint_params']
    checkpoint = pl.callbacks.ModelCheckpoint(
                        dirpath=ckpt_log_dir,
                        filename='{epoch}-{val_acc:.2f}',
                        **checkpoint_params
                        )
    
    # initialize trainer
    max_epochs = config['trainer']['max_epochs']
    enable_checkpointing = config['trainer']['enable_checkpointing']
    trainer = pl.Trainer(
                    accelerator="auto",
                    devices="auto",
                    deterministic=True,
                    logger=wandb_logger,
                    callbacks=[checkpoint],
                    strategy="ddp_find_unused_parameters_true",
                    max_epochs=max_epochs,
                    enable_checkpointing=enable_checkpointing,
                    num_sanity_val_steps=0,
                    enable_progress_bar=False,
                )
    
    # --------------------- Train and Test Model ---------------------
    # train model
    print("Training Model...")
    trainer.fit(
        model=model,
        datamodule=datamodule_batch
    )

    # test model
    print("Testing Model...")
    test_acc = trainer.test(
                    ckpt_path=config['trainer']['ckpt_path'],
                    datamodule=datamodule_batch,      
                )[0]['test_acc']
    print(f"Test Accuracy: {test_acc}")


if __name__ == '__main__':
    args = parse_arguments()
    config = collect_configs(args)
    
    main(config)