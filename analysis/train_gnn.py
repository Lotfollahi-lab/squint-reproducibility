"""
This script trains a Graph Neural Network (GNN) model on a PyTorch Geometric dataset and uses PyTorch Lightning to train the model. Ideally, the script should be run on a multi-GPU-enabled machine and PyG dataset is already processed and saved in the gold data directory.

Usage:
>>> python analysis/train_gnn.py --config config/train_gnn/train_gnn_sss2-1b_1p.yaml
"""

import random
import numpy as np

import torch
import pytorch_lightning as pl
from torch_geometric.data.lightning import LightningNodeData

from vqniche.utils.config_parsers import parse_arguments, collect_configs
from vqniche.dataloaders.transforms import prepare_transforms
from vqniche.dataloaders.custom_in_memory_dataset import CustomInMemoryDataset
from vqniche.models.graphsage import GraphSAGE


def main(config: dict):
    # Get parameters for the Experiment
    experiment_name = config['experiment']['name']
    seed = config['experiment']['seed']
    print(f"Experiment: {experiment_name}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set backend deterministic as true
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')
    
    # Get parameters for Data
    # set root data directory
    data_directory_path = config['data']['data_directory_path']

    # set feature and label keys to use from the PyG Data object
    feature_name = config['data']['feature_name']
    feature_key = f"x_{feature_name}"
    label_name = config['data']['label_name']
    label_key = f"y_{label_name}"

    # set edge index key to use from the PyG Data object
    graph_kwargs = config['data']['graph_kwargs']
    if graph_kwargs['delaunay']:
        edge_index_name = f"{graph_kwargs['spatial_key']}-delaunay"
    else:
        edge_index_name = f"{graph_kwargs['spatial_key']}-radius-{graph_kwargs['radius']}"
    edge_index_key = f"edge_index_{edge_index_name}"
    
    # prepare transforms 
    # (train-val-test split, normalization by read depth, etc.)
    transform_list = config['data']['transform_list']
    transform_kwargs = config['data']['transform_kwargs']
    transform = prepare_transforms(transform_list=transform_list,
                                   feature_key=feature_key,
                                   label_key=label_key,
                                   edge_index_key=edge_index_key,
                                   **transform_kwargs)
    
    # load pytorch geometric data
    dataset = CustomInMemoryDataset(name=experiment_name,
                                    label_names=[label_key],
                                    graph_kwargs=graph_kwargs,
                                    data_directory_path=data_directory_path,
                                    transform=transform)
    data = dataset[0]
    
    # prepare loader
    # TODO: improve to allow custom loaders
    loader = config['data']['loader']
    loader_kwargs = config['data']['loader_kwargs']

    # prepare lightning data module 
    datamodule = LightningNodeData(
            data=data,
            loader=loader,
            shuffle=False,
            **loader_kwargs
            )
    
    # initialize model
    model_name = config['model']['name']
    in_channels = data.num_features
    out_channels = data.num_classes
    hidden_channels = config['model']['hidden_channels']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    loss_names = config['model']['loss_names']
    loss_kwargs = config['model']['loss_kwargs']
    task = config['model']['task']
    optimizer_name = config['model']['optimizer_name']
    lr = config['model']['lr']
    weight_decay = config['model']['weight_decay']
    if model_name == 'GraphSAGE':
        Model = GraphSAGE
    else:
        raise ValueError(f"Model {model_name} not found.")
    
    model = Model(in_channels=in_channels,
                    out_channels=out_channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    lr=lr,
                    weight_decay=weight_decay,                    
                    optimizer_name=optimizer_name,
                    loss_names=loss_names,
                    loss_kwargs=loss_kwargs,
                    task=task,
                    )

    accelerator = config['train']['accelerator']
    max_epochs = config['train']['max_epochs']
    monitor = config['train']['monitor']
    save_top_k = config['train']['save_top_k']
    mode = config['train']['mode']
    ckpt_path = config['train']['ckpt_path']
    
    devices = torch.cuda.device_count()
    strategy = pl.strategies.DDPStrategy(accelerator=accelerator)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor=monitor,
                                                save_top_k=save_top_k,
                                                mode=mode,
                                                save_last=True)
    trainer = pl.Trainer(devices=devices,
                            strategy=strategy,
                            max_epochs=max_epochs,
                            callbacks=[checkpoint]
                            )
    
    trainer.fit(model, datamodule)
    trainer.test(ckpt_path=ckpt_path, datamodule=datamodule)


if __name__ == '__main__':
    args = parse_arguments()
    config = collect_configs(args)
    
    main(config)