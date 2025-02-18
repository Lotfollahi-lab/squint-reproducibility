"""
This script trains one Graph Neural Network (GNN) model on one batch of AnnData.

We use PyTorch Lightning to support distributed data parallel (DDP) training across multiple GPUs.

It is required that the data has previously been processed and stored as a PyTorch Geometric dataset in the "gold" directory.  and PyG dataset is already processed and saved in the gold data directory.

Usage:
>>> python analysis/train_gnn.py --config_file config/train_model/sss2-1b_1p_vq_graphsage.yaml
"""
import os

import torch
import pytorch_lightning as pl

from vqniche.utils.config_parsers import parse_arguments, collect_configs
from vqniche.utils.initialize import initialize_data_and_model, initialize_logger


def train(
        config,
        model,
        datamodule_batch
    ):
    # initialize trainer
    trainer = pl.Trainer(
                    accelerator="auto",
                    devices="auto",
                    deterministic=True,
                    logger=logger,
                    callbacks=checkpoints,
                    strategy="ddp",
                    max_epochs=config['trainer']['max_epochs'],
                    enable_checkpointing=True,
                    num_sanity_val_steps=0,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                )
    
    
    print("Training Model...")
    trainer.fit(
        model=model,
        datamodule=datamodule_batch
    )

    print("Validating Model...")
    trainer.validate(
                    ckpt_path="best",
                    datamodule=datamodule_batch,
                )[0]['val_acc']


if __name__ == '__main__':

    # --------------------- Configure Backend ---------------------
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    num_cores = int(os.environ.get("LSB_DJOB_NUMPROC", 1))
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Number of CPU cores: {num_cores}")
    print(f"Number of GPU devices: {num_gpus}")
    
    # --------------------- Parse Arguments ---------------------
    args = parse_arguments()
    config = collect_configs(args)
    
    # --------------------- Initialize ---------------------
    _, \
    datamodule_batch, \
    model, \
    param_strings = initialize_data_and_model(
                        config
                    )
    
    logger, \
    checkpoints = initialize_logger(
                    config,
                    param_strings
                )

    
    train(
        config,
        model,
        datamodule_batch
    )