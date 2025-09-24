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

from train_model import get_args

from datetime import datetime
import glob


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

codebook_metrics = []
# codebook_metrics = []

attr_impute_metrics = ['pearson_cell_wise', 'pearson_1hop_nbr', 'pearson_gene_wise', 'pearson_gene_wise_1hop_nbr']
# attr_impute_metrics = []

# graph_impute_metrics = ["mmd_degree", "num_edges", "max_degree"]
graph_impute_metrics = []

# sup_global_spatial_conserve_metrics = ['cas']
sup_global_spatial_conserve_metrics = []

# unsup_global_spatial_conserve_metrics = ['mlami']
unsup_global_spatial_conserve_metrics = []

# unsup_local_spatial_conserve_metrics = ['gcs']
unsup_local_spatial_conserve_metrics = []

# sup_niche_cohere_metrics = ['casw']
sup_niche_cohere_metrics = []

# unsup_niche_cohere_metrics = ['nasw']
unsup_niche_cohere_metrics = []

METRICS_LIST = codebook_metrics + attr_impute_metrics + graph_impute_metrics + unsup_global_spatial_conserve_metrics + unsup_local_spatial_conserve_metrics + unsup_niche_cohere_metrics + sup_global_spatial_conserve_metrics + sup_niche_cohere_metrics


def collate_predict_outputs(
        data_cache: List[Dict],
        model: str,
        results_dir: Path,
    ) -> Dict:
    """
    Collate a list of dicts (one per batch from trainer.predict) into a single dict with concatenated tensors or lists.
    Also, convert the collated dict to an AnnData object and save to a pickle file.
    
    Parameters
    ----------
    - data_cache: List[Dict]
        A list of dicts, one per batch from trainer.predict
    - model: str
        The model used for prediction
    - results_dir: Path
        The directory to save the predict data dict and adata
    """
    # --------------------- Collate Data ---------------------
    collated_dict = {}
    for key in data_cache[0][model]:
        collated_dict[key] = torch.cat([d[model][key] for d in data_cache], dim=0)
    
    predict_dict_fname = results_dir / f'predict_data_dict_{model}.pkl'
    print(f"Saving to {predict_dict_fname}...")
    
    with open(predict_dict_fname, 'wb') as f:
        pickle.dump(collated_dict, f)
    
    adata_fname = results_dir / f'predict_adata_{model}.pkl'
    print(f"Converting inference data to AnnData and saving to {adata_fname}...")
    adata = inference_data_dict_to_adata(
                inference_data=collated_dict,
            )
    with open(adata_fname, 'wb') as f:
        pickle.dump(adata, f)
    return adata


def test(config: Dict, args: argparse.Namespace):
    """
    Test a model on a single batch of data using the provided configuration.
    
    Parameters
    ----------
    - config: Dict
        A dictionary containing the configuration parameters for the training run.
    - args: argparse.Namespace
        The command line arguments.
    
    Returns
    -------
    - None
    """
    
    # --------------------- Experiment Name ---------------------
    if args.experiment_name:
        test_experiment_name = args.experiment_name
    else:
        test_experiment_name = f'lr_{args.learning_rate}_wd_{args.weight_decay}_depth_{args.depth}_cond_drop_prob_{args.cond_drop_prob}'
    train_experiment_name = test_experiment_name
    if args.run_id:
        test_experiment_name = f'{test_experiment_name}_{args.run_id}'
    if args.add_run_id:
        run_id = datetime.now().strftime('%Y%m%d_%H%M_maskgit')
        test_experiment_name = f'{test_experiment_name}_{run_id}'
    
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

    # --------------------- Model ---------------------
    vae_Model = set_model_class(config['model']['model_name'])
    vae = vae_Model.load_from_checkpoint(
        config['model']['model_ckpt_fname'],
    )

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

    # Look for checkpoint files
    ckpts = glob.glob(f"{args.output_path}/{args.checkpoint_dir}/{train_experiment_name}/*.ckpt")
    assert len(ckpts) > 0, f"No checkpoint files found in {args.output_path}/{args.checkpoint_dir}/{train_experiment_name}"
    
    print(f"Found {len(ckpts)} checkpoint files in {args.output_path}/{args.checkpoint_dir}/{train_experiment_name}")
    # Load the latest checkpoint
    latest_ckpt = max(ckpts, key=os.path.getctime)
    assert latest_ckpt is not None, f"Checkpoint files found in {args.output_path}/{args.checkpoint_dir}/{train_experiment_name} do not comply with the expected naming convention"
    
    print(f"Loading checkpoint from {latest_ckpt}")
    model = MaskGitRunner.load_from_checkpoint(
                latest_ckpt,
                base=maskgit,
                batch_size=args.batch_size,
                cond_scale=args.cond_scale,
            )
    model.eval()
    
    # --------------------- Trainer ---------------------
    strategy = "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
                    accelerator="auto",
                    devices="auto",
                    deterministic=True,
                    logger=False,
                    strategy=strategy,
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                    enable_progress_bar=True,
                    enable_model_summary=True,
                )

    # --------------------- Results ---------------------
    # create a results directory for the model run
    results_dir = Path(args.output_path) / args.results_dir / test_experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- Predict ---------------------
    print("Predicting Model...")
    predict_data_cache = trainer.predict(
        model=model,
        datamodule=datamodule_batch,
    )
    
    adata_vae = collate_predict_outputs(
        data_cache=predict_data_cache,
        model='vae',
        results_dir=results_dir,
    )
    adata_maskgit = collate_predict_outputs(
        data_cache=predict_data_cache,
        model='maskgit',
        results_dir=results_dir,
    )

    # --------------------- Metrics (if --compute_metrics flag is set) ---------------------
    if config['experiment']['compute_metrics']:
        print('Computing metrics...')
        for adata, model in [(adata_vae, 'vae'), (adata_maskgit, 'maskgit')]:
            metrics_values = metrics.compute_benchmarking_metrics(
                            adata=adata,
                            metrics=METRICS_LIST,
                            cell_type_key='cell_types',
                            spatial_key='spatial',
                            latent_key='H_adj',
                            seed=0
                        )
            
            df = pd.DataFrame([
                    {'metric': metric, 'score': score} 
                    for metric, score in metrics_values.items()
                ])
            df = df.round(4)
            
            # print the metrics to the console and save to a text file
            print(tabulate(df, headers='keys', tablefmt='grid'))
            with open(results_dir / f'metrics_{model}.txt', 'w') as f:
                f.write(tabulate(df, headers='keys', tablefmt='grid'))

            # save the metrics to a CSV file
            df.to_csv(results_dir / f'metrics_{model}.csv', index=False)


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
    
    # --------------------- Test Pipeline---------------------
    test(config, args)