"""
This script trains one Graph Neural Network (GNN) model on one batch of AnnData.

We use PyTorch Lightning to support distributed data parallel (DDP) training across multiple GPUs.

It is required that the data has previously been processed and stored as a PyTorch Geometric dataset in the "gold" directory.  and PyG dataset is already processed and saved in the gold data directory.

Usage:
>>> python analysis/train_gnn.py --config config/train_gnn/train_gnn_sss2-1b_1p.yaml
"""

import random
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.data.lightning import LightningNodeData
import torch_geometric.transforms as T

from vqniche.utils.config_parsers import parse_arguments, collect_configs
from vqniche.dataloaders.transforms import SetExperimentDataKeys, init_data_transforms
from vqniche.dataloaders.in_memory_dataset_blob import InMemoryDatasetBlob
from vqniche.models.graphsage import GraphSAGE


def main(config: dict):
    # --------------------- Wandb and Logger ---------------------
    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="VQNiche", 
                               log_model=True,
                        )
    # Log hyperparameters
    wandb_logger.log_hyperparams(config)
    
    # --------------------- Experiment Settings ---------------------
    print(f"Experiment: {config['experiment']['description']}")
    
    seed = config['experiment']['seed']
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set backend deterministic as true
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')
    
    num_devices = torch.cuda.device_count()
    print(f"Number of devices: {num_devices}")
    
    # --------------------- Dataset Parameters ---------------------
    # NOTE: currently supports only one feature, label, and edge index
    # decide which node features to use in this experiment
    feature_name = config['data']['feature_name']
    
    # decide which node labels to use in this experiment
    label_name = config['data']['label_name']

    # decide which edge index to use in this experiment
    graph_kwargs = config['data']['graph_kwargs']
    spatial_key = graph_kwargs['spatial_key']
    delaunay = graph_kwargs['delaunay']
    radii = graph_kwargs['radii']
    assert delaunay or len(radii) >= 1, "Either `delaunay` or `radii` must be provided."
    if delaunay:
        edge_index_name = f"{spatial_key}_delaunay"
    else:
        radius = radii[0] # only supports using the first radius
        delaunay_radius_union = graph_kwargs['delaunay_radius_union']
        if delaunay_radius_union:
            edge_index_name = f"{spatial_key}_delaunay_radius_{radius}"
        else:
            edge_index_name = f"{spatial_key}_radius_{radius}"

    DataKeyTransform = SetExperimentDataKeys(       
                            feature_name=feature_name,
                            label_name=label_name,
                            edge_index_name=edge_index_name
                        )
    
    print(f"Feature Name: {feature_name}")
    print(f"Label Name: {label_name}")
    print(f"Graph Name: {edge_index_name}")
    
    # --------------------- Initialize In-Memory Dataset Blob ---------------------
    dataset_name = config['dataset']['name']
    print(f"Initializing Dataset: {dataset_name}")
    
    # set root data directory
    data_directory_path = config['data']['data_directory_path']

    # e.g. normalize features , train-val-test split, etc.
    transform_names = config['data']['transform_names']
    transform_kwargs = config['data']['transform_kwargs']
    data_transforms = init_data_transforms(
                        transform_names=transform_names,
                        **transform_kwargs
                    )

    # initialize a composed transform
    transform = T.Compose([DataKeyTransform, data_transforms])
    
    # initialize pytorch geometric dataset blob stored at:
    # data_directory_path / 'gold' / 'in-memory-PyG-dataset-blob' / dataset_name / 'dataset_blob.pt'
    dataset_blob = InMemoryDatasetBlob(
                        name=dataset_name,
                        data_directory_path=data_directory_path,
                        transform=transform
                    )
    
    # --------------------- Data and Loader ---------------------
    # NOTE: currently trains one model on one batch of data
    # Load PyG data object corresponding to batch_idx (e.g. AnnData batch0)
    batch_idx = config['data']['batch_idx']
    data_batch = dataset_blob[batch_idx]
    assert data_batch['metadata_batch_id'] == f"batch{batch_idx}", "Batch ID mismatch."
    print(f"Batch ID: {data_batch['metadata_batch_id']}")
     
    # set data loader (currently uses a string, .e.g. 'neighbor')
    # TODO: Upgrade to support string or custom Callable data loader
    loader = config['data']['loader']
    loader_kwargs = config['data']['loader_kwargs']

    # initialize lightning node data module 
    datamodule_batch = LightningNodeData(
                            data=data_batch,
                            loader=loader,
                            shuffle=False,
                            **loader_kwargs
                        )
    
    # --------------------- Initialize Model ---------------------    
    model_name = config['model']['name']
    print(f"Model: {model_name}")

    # get model, optimizer, loss, and task parameters
    model_params = config['model']['model_params']
    optimizer_params = config['model']['optimizer_params']
    loss_params = config['model']['loss_params']
    task_params = config['model']['task_params']

    # initialize model 
    if model_name == 'GraphSAGE':
        Model = GraphSAGE
    else:
        raise ValueError(f"Model {model_name} not found.")
    model = Model(
                name=model_name,                
                in_channels=data_batch.num_features,
                out_channels=data_batch.num_classes,
                **model_params,
                **optimizer_params,
                **loss_params,
                **task_params
            )

    # Log model architecture
    wandb_logger.watch(model)

    # --------------------- Initialize Trainer ---------------------
    # set strategy to Distributed Data Parallel (DDP)
    accelerator = config['trainer']['accelerator']
    strategy = pl.strategies.DDPStrategy(accelerator=accelerator)

    # configure model checkpointing
    monitor = config['trainer']['monitor']
    save_top_k = config['trainer']['save_top_k']
    mode = config['trainer']['mode']
    save_last = config['trainer']['save_last']
    checkpoint = pl.callbacks.ModelCheckpoint(
                    monitor=monitor,
                    save_top_k=save_top_k,
                    mode=mode,
                    save_last=save_last
                )
    
    # initialize trainer
    max_epochs = config['trainer']['max_epochs']
    trainer = pl.Trainer(
                    devices=num_devices,
                    strategy=strategy,
                    max_epochs=max_epochs,
                    callbacks=[checkpoint]
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
    ckpt_path = config['trainer']['ckpt_path']
    trainer.test(
        ckpt_path=ckpt_path,
        datamodule=datamodule_batch
    )


if __name__ == '__main__':
    args = parse_arguments()
    config = collect_configs(args)
    
    main(config)    