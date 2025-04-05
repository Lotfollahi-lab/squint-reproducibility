"""

"""
import os
import scanpy as sc
from pathlib import Path
from typing import Dict

import torch
import pytorch_lightning as pl

from vqniche.utils.parse_test_configs import *
from vqniche.initializers.initialize import *


def test(config: Dict):
    """
    Test a model on a single batch of data using the provided configuration.
    
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

    # --------------------- Load Adata ---------------------
    f = Path(config['dataset']['root_data_dir']) / 'silver' / config['dataset']['dataset_name'] / config['dataset']['adata_fname'][0]
    adata = sc.read_h5ad(f)

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
    Model = set_model_class(config['model']['model_name'])
    model = Model.load_from_checkpoint(config['model']['model_ckpt'])
    
    # --------------------- Trainer ---------------------
    # strategy = "ddp_find_unused_parameters_true"
    strategy = "ddp"
    
    trainer = pl.Trainer(
                    accelerator="auto",
                    devices="auto",
                    deterministic=True,
                    logger=False,
                    callbacks=False,
                    strategy=strategy,
                    max_epochs=config['trainer']['max_epochs'],
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                    enable_progress_bar=False,
                    enable_model_summary=True,
                )
    
    # --------------------- Accuracy ---------------------
    # compute model accuracy on validation and test sets
    print("Computing Model Validation Accuracy...")
    trainer.validate(
        model=model,
        datamodule=datamodule_batch,
        verbose=True,
    )
    print("Computing Model Test Accuracy...")
    trainer.test(
        model=model,
        datamodule=datamodule_batch,
        verbose=True,
    )

    # --------------------- Compute Attribute Metrics ---------------------


if __name__ == '__main__':

    # --------------------- Configure Backend ---------------------
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    num_cores = int(os.environ.get("LSB_DJOB_NUMPROC", 1))
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Number of CPU cores: {num_cores}")
    print(f"Number of GPU devices: {num_gpus}")
    
    # --------------------- Parse Arguments ---------------------
    args = parse_test_arguments()
    config = collect_configs(args)
    
    # --------------------- Test ---------------------
    test(config)