"""
This script provides a pipeline for a single training run of a Model on a single batch of data. It requires that the data has previously been processed and stored as a PyTorch Geometric dataset in the "gold" directory. The input to this file is a YAML config file containing all the relevant parameters for the training run. 

The pipeline consists of the following steps:
1. Parse the arguments from the config file.
2. Initialize the DatasetBlob (see vqniche.dataset.in_memory_dataset_blob.py).
3. Load the data corresponding to the batch index specified in the config file.
4. Initialize the DataModule (see vqniche.dataloaders.in_memory_datamodule.py).
5. Initialize the Model (see vqniche.models.vqgraph.py).
6. Initialize the Logger (if enabled).
7. Initialize the Checkpoints (if enabled).
8. Initialize the Trainer (see vqniche.utils.initialize.py).
9. Train the Model.
10. Validate the Model.

The pipeline employs PyTorch Lightning to support distributed data parallel (DDP) training across multiple GPUs. 

Example Usage:
>>> python analysis/train_model.py --config_file config/train_model/sss2-1b_1p_vq_graphsage.yaml
"""
import os
from typing import Dict

import torch
import pytorch_lightning as pl

from vqniche.utils.config_parsers import parse_arguments, collect_configs
from vqniche.initializers.initialize import *


def train(config: Dict):
    """
    Train a model on a single batch of data using the provided configuration.
    
    Parameters
    ----------
    - config: Dict
        A dictionary containing the configuration parameters for the training run.
    
    Returns
    -------
    - None
    """
    # --------------------- Determinism Settings ---------------------
    pl.seed_everything(config['experiment']['seed'])

    # --------------------- Dataset ---------------------
    dataset_blob = initialize_dataset_blob(config)

    # --------------------- Databatch ---------------------
    data_batch = initialize_databatch(
                    config=config,
                    dataset_blob=dataset_blob,
                )
    
    # --------------------- Dataloader ---------------------
    datamodule_batch = initialize_datamodule(
                            config=config,
                            data=data_batch,
                        )

    # --------------------- Model ---------------------
    model = initialize_model(
                config=config,
                in_channels=data_batch.num_features,
                out_channels=data_batch.num_classes,
            )
    
    # --------------------- Logger ---------------------
    if config['logging']['enabled']:
        logger = initialize_logger(config)
    else:
        logger = False

    # --------------------- Checkpoints ---------------------
    enable_checkpointing = config['trainer']['enable_checkpointing']
    if enable_checkpointing:
        try:
            ckpt_log_dir = Path(logger.experiment.dir) / 'checkpoints'
        except:
            ckpt_log_dir = Path.cwd() / 'checkpoints'
        ckpt_log_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_params = config['trainer']['checkpoint_params']
        checkpoints = [
                        pl.callbacks.ModelCheckpoint(
                            dirpath=ckpt_log_dir,
                            monitor='val_acc',
                            filename='{epoch}-{val_acc:.2f}',
                            **checkpoint_params
                            )
                        ]
    else:
        checkpoints = False
    
    # --------------------- Trainer ---------------------
    trainer = pl.Trainer(
                    accelerator="auto",
                    devices="auto",
                    deterministic=True,
                    logger=logger,
                    callbacks=checkpoints,
                    strategy="ddp",
                    max_epochs=config['trainer']['max_epochs'],
                    enable_checkpointing=enable_checkpointing,
                    num_sanity_val_steps=0,
                    enable_progress_bar=False,
                    enable_model_summary=True,
                )
    
    # --------------------- Train Model ---------------------
    
    print("Training Model...")
    trainer.fit(
        model=model,
        datamodule=datamodule_batch
    )

    # --------------------- Validate Model ---------------------
    if enable_checkpointing:
        print("Validating Model using the best checkpoint...")
        trainer.validate(
            ckpt_path="best",
            datamodule=datamodule_batch,
            verbose=True,
        )
    else:
        print("Validating Model using the last checkpoint...")
        trainer.validate(
            model=model,
            ckpt_path=None,
            datamodule=datamodule_batch,
            verbose=True,
        )


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
    
    # --------------------- Train ---------------------
    train(config)