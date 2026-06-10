"""
This script provides a pipeline for a single training run of a Model on a single batch of data. It requires that the data has previously been processed and stored as a PyTorch Geometric dataset in the "gold" directory. The input to this file is a YAML config file containing all the relevant parameters for the training run. 

The pipeline consists of the following steps:
1. Parse the arguments from the config file.
2. Initialize the DatasetBlob (see vqniche.dataset.in_memory_dataset_blob.py).
3. Load the data corresponding to the batches specified in the config file.
4. Initialize the DataModule (see vqniche.dataloaders.in_memory_datamodule.py).
5. Initialize the Model (see vqniche.models.vqgraph.py).
6. Initialize the Logger (if enabled).
7. Initialize the Checkpoints (if enabled).
8. Initialize the Trainer (see vqniche.utils.initialize.py).
9. Train the Model.
10. Validate the Model.

The pipeline employs PyTorch Lightning to support distributed data parallel (DDP) training across multiple GPUs. With wandb, the script supports a standalone run via a base config file and a sweep run via a base config file and a list of sweep config files.

Example Usage for a standalone run:
>>> python analysis/train_model.py --base_config_file config/train_model/xhs1000-39b_1p-batch11_random-split_vqniche_graphsage.yaml

Example Usage for a sweep run:
>>> python analysis/train_model.py --base_config_file config/train_model/xhs1000-39b_1p-batch11_random-split_vqniche_graphsage.yaml --sweep_config_files config/sweep/backbone_gnn.yaml
"""
import re
import os
import wandb
from typing import Dict
from pathlib import Path
import pandas as pd

import torch
import pytorch_lightning as pl

from vqniche.utils.parse_train_configs import parse_train_arguments, collect_train_configs, update_config
from vqniche.initializers.initialize import *
from vqniche.utils.parse_test_configs import find_best_checkpoint


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
    if 'conditioning_params' in config['model']['encoder_params']:
        config['model']['encoder_params']['conditioning_params']['condition_dim'] = data_batch.encoder_condition_dim
    if 'spatial_prior_params' in config['model']['encoder_params']:
        config['model']['encoder_params']['spatial_prior_params']['spatial_prior_feature_dim'] = data_batch.spatial_prior_feature_dim
    if 'conditioning_params' in config['model']['attribute_decoder_params']:
        config['model']['attribute_decoder_params']['conditioning_params']['condition_dim'] = data_batch.attr_decoder_condition_dim
    if 'conditioning_params' in config['model']['adjacency_decoder_params']:
        config['model']['adjacency_decoder_params']['conditioning_params']['condition_dim'] = data_batch.adj_decoder_condition_dim

    model = initialize_model(
                config=config,
                in_channels=data_batch.num_features,
                out_channels=data_batch.num_classes,
            )
    
    # --------------------- Logger ---------------------
    logger = initialize_logger(config)

    # --------------------- Checkpoints ---------------------
    enable_checkpointing = config['trainer']['enable_checkpointing']
    if enable_checkpointing:
        ckpt_log_dir = Path(logger.experiment.dir) / 'checkpoints'
        ckpt_log_dir.mkdir(parents=True, exist_ok=True)
        
        monitor = config['trainer']['monitor']
        if monitor == 'val_pearson_cell_wise':
            filename = '{epoch}-{val_pearson_cell_wise:.2f}'
        elif monitor == 'val_pearson_1hop_nbr':
            filename = '{epoch}-{val_pearson_1hop_nbr:.2f}'
        elif monitor == 'val_pearson_gene_wise':
            filename = '{epoch}-{val_pearson_gene_wise:.2f}'
        elif monitor == 'val_pearson_gene_wise_1hop_nbr':
            filename = '{epoch}-{val_pearson_gene_wise_1hop_nbr:.2f}'
        elif monitor == 'train_pearson_cell_wise':
            filename = '{epoch}-{train_pearson_cell_wise:.2f}'
        elif monitor == 'train_pearson_1hop_nbr':
            filename = '{epoch}-{train_pearson_1hop_nbr:.2f}'
        elif monitor == 'train_pearson_gene_wise':
            filename = '{epoch}-{train_pearson_gene_wise:.2f}'
        elif monitor == 'train_pearson_gene_wise_1hop_nbr':
            filename = '{epoch}-{train_pearson_gene_wise_1hop_nbr:.2f}'
        else:
            monitor = 'val_pearson_cell_wise'
            filename = '{epoch}-{val_pearson_cell_wise:.2f}'
            
        checkpoint_params = config['trainer']['checkpoint_params']
        checkpoints = [
                        pl.callbacks.ModelCheckpoint(
                            dirpath=ckpt_log_dir,
                            monitor=monitor,
                            filename=filename,
                            **checkpoint_params
                            )
                        ]
    else:
        checkpoints = False
    
    # --------------------- Trainer ---------------------
    strategy = "ddp_find_unused_parameters_true"
    # strategy = "ddp"
    
    trainer = pl.Trainer(
                    accelerator="auto",
                    devices="auto",
                    deterministic=True,
                    # Mirror run_squint.py: honour an opt-in
                    # `trainer.precision` cfg key (e.g. "bf16-mixed")
                    # if provided, otherwise fall back to "32-true"
                    # for legacy behaviour. Legacy callers that don't
                    # set the key are unaffected.
                    precision=config['trainer'].get('precision', '32-true'),
                    logger=logger,
                    callbacks=checkpoints,
                    strategy=strategy,
                    max_epochs=config['trainer']['max_epochs'],
                    # Mirror run_squint.py: validate every N train epochs
                    # (default N=1 if the cfg doesn't carry the key, to
                    # preserve legacy behaviour for any caller that
                    # uses this trainer without the new setting).
                    check_val_every_n_epoch=int(
                        config['trainer'].get('check_val_every_n_epoch', 1)
                    ),
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
    if 'validate' in config['experiment'] and config['experiment']['validate']:
        if enable_checkpointing:
            print("Validating Model using the best checkpoint...")
            val_results = trainer.validate(
                                ckpt_path="best",
                                datamodule=datamodule_batch,
                                verbose=True,
                            )
        else:
            print("Validating Model using the last checkpoint...")
            val_results = trainer.validate(
                                model=model,
                                ckpt_path=None,
                                datamodule=datamodule_batch,
                                verbose=True,
                            )
        print("Validation Results:")
            
    # --------------------- Test Model ---------------------
    if 'test' in config['experiment'] and config['experiment']['test']:
        if enable_checkpointing:
            print("Testing Model using the best checkpoint...")
            test_results = trainer.test(
                                ckpt_path="best",
                                datamodule=datamodule_batch,
                                verbose=True,
                            )
        else:
            print("Testing Model using the last checkpoint...")
            test_results = trainer.test(
                                model=model,
                                ckpt_path=None,
                                datamodule=datamodule_batch,
                                verbose=True,
                            )
        best_ckpt_fname = find_best_checkpoint(
                                wandb_run_dir=Path(logger.experiment.dir).parent,
                                mode=config['trainer']['checkpoint_params']['mode'],
                                metric_name=config['trainer']['monitor'],
                            ).stem
        best_epoch = int(re.search(r'epoch=(\d+)', best_ckpt_fname).group(1))
        for result in test_results:
            result['epoch'] = best_epoch
        
        results_dir = Path(logger.experiment.dir) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        test_metrics_fname = results_dir / 'test_metrics.csv'
        test_metrics_df = pd.DataFrame(test_results)
        test_metrics_df.to_csv(
            path_or_buf=test_metrics_fname,
            sep=',',
            index=False
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
    args = parse_train_arguments()
    base_config, sweep_config = collect_train_configs(args)
    
    # -------------- Initiate WandB Sweep/Run ------------------
    if base_config['experiment']['mode'] == 'sweep':
        # registers sweep with specified hyperparameter grid
        if 'sweep_id' not in base_config['experiment']:
            sweep_id = wandb.sweep(
                project="VQNiche",
                sweep=sweep_config,
            )
        else:
            sweep_id = base_config['experiment']['sweep_id']

        # sets directory for sweep runs
        sweep_dir = set_wandb_experiment_dir(
                            config=base_config,
                            experiment_mode='sweep',
                            sweep_name=sweep_config['name'],
                        )

        # defines training function for an individual run of the sweep
        def single_sweep_run_train_wrapper():
            # initializes a run for the current config from the sweep
            sweep_run = wandb.init(
                        dir=str(sweep_dir),
                        project="VQNiche",
                        mode="offline" if base_config['logging']['offline'] else "online",
                        group=f"{base_config['dataset']['dataset_name']}:batch={base_config['dataset']['adata_batch_idx']}",
                        job_type="train",
                    )
            # updates base config with run config
            config = update_config(
                base_config,
                dict(sweep_run.config)
            )
            # trains the model with the full config
            train(config)
            # shuts down the run
            sweep_run.finish()

        # calls wandb agent to train the sweep
        wandb.agent(
            sweep_id=sweep_id,
            function=single_sweep_run_train_wrapper,
            count=sweep_config['run_cap']
        )
        
        # shuts down the sweep
        wandb.teardown()
        
    elif base_config['experiment']['mode'] == 'standalone':
        # sets directory for standalone run
        standalone_dir = set_wandb_experiment_dir(
                            config=base_config,
                            experiment_mode='standalone',
                        )
        # initializes a run for the standalone config
        standalone_run = wandb.init(
                            dir=str(standalone_dir),
                            project="VQNiche",
                            mode="offline" if base_config['logging']['offline'] else "online",
                            group=f"{base_config['dataset']['dataset_name']}:batch={base_config['dataset']['adata_batch_idx']}",
                            job_type="train",
                            monitor_gym=True,
                        )

        # trains the model with the full config
        train(base_config)

        # shuts down the run
        standalone_run.finish()
    else:
        raise ValueError(f"Invalid experiment mode: {base_config['experiment']['mode']}")