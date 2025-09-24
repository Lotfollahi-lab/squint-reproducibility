from typing import Dict

import os
import pickle
from pathlib import Path
import random

import pandas as pd
from tabulate import tabulate

import torch
import pytorch_lightning as pl

from vqniche.utils.parse_test_configs import *
from vqniche.initializers.initialize import *
from vqniche import metrics
from vqniche.utils.type_conversions import *
from vqniche.plotting import *
from vqniche.utils.loss_utils import aggregate_1hop_neighbor_features

from multi_timepoint_datamodule import create_multi_timepoint_datamodule
from muse_maskgit_pytorch import MaskGitTransformer, MaskGit, MaskGitRunner

from datetime import datetime
import uuid
import glob
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import MPIEnvironment


time_points_dict = {
    't1': [4, 8, 9, 14, 20, 21, 25, 33],      # Non-lesional Baseline
    't2': [5, 12, 13, 17, 23, 26, 35, 38],    # Day 3
    't3': [1, 7, 10, 18, 24, 27, 31, 34],     # Day 14
    't4': [0, 6, 15, 16, 22, 29, 36, 37],     # Week 12
}

train_patients = {
    'BK27': [0, 1, 9, 13],
    'BK21': [2, 5, 10, 14, 16],
    'BK20': [3, 7, 8, 15, 17],
    'BK18': [4, 6, 11, 12, 18],
    'BK30': [20, 26, 27, 32, 36],
    'BK25': [23, 24, 25, 28, 29],
}

test_patients = {
    'BK39': [19, 21, 22, 34, 38],
    'BK24': [30, 31, 33, 35, 37],
}


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for training.

    Returns:
    ----------
    - argparse.Namespace: The command line arguments.
    """
    
    # --------------------- VQNiche Arguments ---------------------
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    parser.add_argument('--wandb_run_dir',
                        type=str,
                        help='Path to the wandb run directory')
    parser.add_argument('--model_ckpt_fname',
                        type=str,
                        default=None,
                        help='Optional model checkpoint file name')
    parser.add_argument('--metric_name',
                        type=str,
                        default=None,
                        help='Metric name to find the best checkpoint')
    parser.add_argument('--mode',
                        type=str,
                        default=None,
                        help='Mode to find the best checkpoint')
    parser.add_argument('--compute_metrics',
                        action='store_true',
                        help='Compute metrics')
    parser.add_argument('--plot_figures',
                        action='store_true',
                        help='Plot figures')
    parser.add_argument('--override',
                        nargs='+',
                        help='Override parameters in the config file')

    # --------------------- MaskGit Arguments ---------------------
    parser.add_argument('--num_nodes',
                        type=int,
                        default=1,
                        help='Number of nodes for MaskGit')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help='Learning rate for MaskGit')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.001,
                        help='Weight decay for MaskGit')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size for MaskGit')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='Number of epochs for MaskGit')
    parser.add_argument('--output_path',
                        type=str,
                        default='maskgit_vqniche_out',
                        help='Output path for MaskGit')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='checkpoints',
                        help='Checkpoint directory for MaskGit')
    parser.add_argument('--results_dir',
                        type=str,
                        default='results',
                        help='Results directory for MaskGit')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Log directory for MaskGit')
    parser.add_argument('--cond_scale',
                        type=float,
                        default=3.,
                        help='Cond scale for MaskGit')
    parser.add_argument('--experiment_name',
                        type=str,
                        default=None,
                        help='Experiment name')
    parser.add_argument('--run_id',
                        type=str,
                        default=None,
                        help='Run ID')
    parser.add_argument('--add_run_id',
                        action='store_true',
                        help='Add run ID to the experiment name')
    parser.add_argument('--depth',
                        type=int,
                        default=6,
                        help='Depth for MaskGitTransformer')
    parser.add_argument('--cond_drop_prob',
                        type=float,
                        default=0.10,
                        help='Cond drop probability for MaskGit')
    parser.add_argument('--patients',
                        type=str,
                        choices=['train', 'test', 'both'],
                        default='both',
                        help='Patients to train and test on')
    return parser.parse_args()


def train(config: Dict, args: argparse.Namespace):
    """
    Train a model on train patients of data distributed across multiple timepoints.
    
    Parameters
    ----------
    - config: Dict
        A dictionary containing the configuration parameters for the training run of VAE.
    - args: argparse.Namespace
        The command line arguments.
    
    Returns
    -------
    - None
    """
    
    # --------------------- Experiment Name ---------------------
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = f'lr_{args.learning_rate}_wd_{args.weight_decay}_depth_{args.depth}_cond_drop_prob_{args.cond_drop_prob}'
    if args.run_id:
        experiment_name = f'{experiment_name}_{args.run_id}'
    if args.add_run_id:
        run_id = datetime.now().strftime('%Y%m%d_%H%M_maskgit')
        experiment_name = f'{experiment_name}_{run_id}'
    
    # --------------------- Determinism Settings ---------------------
    pl.seed_everything(config['experiment']['seed'])
    # Use the same seed for random (used in dataloader)
    random.seed(config['experiment']['seed'])

    # --------------------- Dataloader ---------------------
    datamodule_batch = create_multi_timepoint_datamodule(
        config=config,
        time_points_dict=time_points_dict,
        train_patients=train_patients if args.patients in ['train', 'both'] else test_patients,
        test_patients=test_patients if args.patients in ['test', 'both'] else train_patients,
        batch_size=args.batch_size,
    )

    # --------------------- VAE Model ---------------------
    vae_Model = set_model_class(config['model']['model_name'])
    vae = vae_Model.load_from_checkpoint(
        config['model']['model_ckpt_fname'],
    )
    
    # --------------------- Transformer and MaskGit ---------------------
    transformer = MaskGitTransformer(
        num_tokens = config['model']['encoder_params']['vq_params']['codebook_size'],
        num_timepoints = 3,       # number of timepoints
        dim = 512,                # model dimension
        depth = args.depth,       # depth
        dim_head = 64,            # attention head dimension
        heads = 8,                # attention heads,
        ff_mult = 4,              # feedforward expansion factor
        add_mask_id = True,       # add mask id
        add_pad_id = True,        # add pad id
    )
    
    # pass the trained VAE and the base transformer to MaskGit
    maskgit = MaskGit(
        vae = vae,
        transformer = transformer,
        cond_drop_prob = args.cond_drop_prob,
    )
    
    maskgit_runner = MaskGitRunner(
        base = maskgit,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
    )
    
    # --------------------- Train ---------------------
    
    log_path = Path(args.output_path) / args.log_dir / experiment_name
    log_path.mkdir(parents=True, exist_ok=True)

    monitor_metric = 'train/loss'
    mode = 'min'

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{args.output_path}/{args.checkpoint_dir}/{experiment_name}',
        save_top_k=-1,
        every_n_epochs=2,
        verbose=True,
        monitor=monitor_metric,
        mode=mode,
    )

    if torch.cuda.device_count() > 1:
        mpi_environment = MPIEnvironment()
        
        # multi gpu training with group logging
        wandb_logger = WandbLogger(
            project='maskgit-vqniche',
            name=f'{experiment_name}_{mpi_environment.world_size()}_{str(uuid.uuid4())[:6]}',
            save_dir=log_path,
            log_model='all',
        )
    else:
        wandb_logger = WandbLogger(
            project='maskgit-vqniche',
            name=f'{experiment_name}',
            save_dir=log_path,
            log_model='all',
        )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode=mode,
    )

    is_cuda = torch.cuda.is_available()
    accelerator = 'gpu' if is_cuda else 'cpu'
    num_cuda_devices = torch.cuda.device_count()
    if num_cuda_devices > 1:
        print(f"Using {torch.cuda.device_count()} GPU(s).")
        strategy = DDPStrategy(cluster_environment=mpi_environment, find_unused_parameters=True)
    elif is_cuda:
        cuda_device_name = torch.cuda.get_device_name()
        print(f'Using {cuda_device_name} for training.')
        strategy = 'auto'
    else:
        print('Using device {}.'.format(accelerator))
        strategy = 'auto'

    print("Training Model...")

    maskgit_trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=10),
            checkpoint_callback,
            early_stop_callback,
        ],
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=-1 if is_cuda else 0,
        strategy=strategy,
        gradient_clip_algorithm='norm',
        num_nodes=args.num_nodes,
        # limit_val_batches=0,      # never run the val loop
        # num_sanity_val_steps=0,   # skip the initial sanity val checks
        # check_val_every_n_epoch=None,  # (PL>=2.0) don't schedule epoch-end val
    )
    
    # Look for checkpoint files
    ckpts = glob.glob(f"{args.output_path}/{args.checkpoint_dir}/{experiment_name}/*.ckpt")
    # Finally, kick of the training process.
    if ckpts:
        print(f"Found {len(ckpts)} checkpoint files in {args.output_path}/{args.checkpoint_dir}/{experiment_name}")
        # Load the latest checkpoint
        latest_ckpt = max(ckpts, key=os.path.getctime)
        print(f"Loading checkpoint from {latest_ckpt}")
        maskgit_trainer.fit(maskgit_runner, datamodule_batch, ckpt_path=latest_ckpt)
    else:
        maskgit_trainer.fit(maskgit_runner, datamodule_batch)


if __name__ == '__main__':

    # --------------------- Configure Backend ---------------------
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    num_cores = int(os.environ.get("LSB_DJOB_NUMPROC", 1))
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Number of CPU cores: {num_cores}")
    print(f"Number of GPU devices: {num_gpus}")
    
    # --------------------- Parse Arguments ---------------------
    args = get_args()
    
    # --------------------- Parse VAE Eval Arguments ---------------------
    config = collect_test_configs(args)
    
    # --------------------- Run Training Pipeline---------------------
    train(config, args)