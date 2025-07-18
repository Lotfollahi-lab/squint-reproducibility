"""
This script provides a pipeline for a single testing run of a Model on a single batch of data. It requires that the data has previously been processed and stored as a PyTorch Geometric dataset in the "gold" directory. The input to this file is a YAML config file containing all the relevant parameters for the testing run. 

The pipeline consists of the following steps:
1. Parse the arguments from the config file.
2. Initialize the DatasetBlob (see vqniche.dataset.in_memory_dataset_blob.py).
3. Load the data corresponding to the batches specified in the config file.
4. Initialize the DataModule (see vqniche.dataloaders.in_memory_datamodule.py).
5. Load the Model from the checkpoint file specified in the config file (see vqniche.models.vqgraph.py).
6. Initialize the Trainer (see vqniche.utils.initialize.py).
7. Test the Model.
8. Compute the Pearson correlation between the original and reconstructed cell-gene matrices.
9. Compute the Codebook Utilization.
10. Compute the Graph Metrics: MMD for Node Degree Distribution.

Example Usage:
>>> python analysis/test_model.py --wandb_run_dir /path/to/wandb/run

"""
from typing import Dict

import os
import pickle
from pathlib import Path
import pandas as pd
from tabulate import tabulate

import torch
import pytorch_lightning as pl

from vqniche.utils.parse_test_configs import *
from vqniche.initializers.initialize import *
from vqniche import metrics
from vqniche.utils.type_conversions import *

attr_impute_metrics = ['pearson_cell_wise', 'pearson_1hop_nbr']
graph_impute_metrics = ["mmd_degree", "mmd_eigenvalues", "num_edges", "max_degree"]
sup_global_spatial_conserve_metrics = ['cas']
unsup_global_spatial_conserve_metrics = ['mlami']
sup_local_spatial_conserve_metrics = ['clisis']
unsup_local_spatial_conserve_metrics = ['gcs']
sup_niche_cohere_metrics = ['cnmi', 'cari', 'casw', 'clisi']
unsup_niche_cohere_metrics = ['nasw']
METRICS_LIST = attr_impute_metrics + graph_impute_metrics + unsup_global_spatial_conserve_metrics + unsup_local_spatial_conserve_metrics + unsup_niche_cohere_metrics + sup_global_spatial_conserve_metrics + sup_local_spatial_conserve_metrics + sup_niche_cohere_metrics


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

    # --------------------- Dataset ---------------------
    dataset_blob = initialize_dataset_blob(config)
    with open(Path(dataset_blob.processed_dir) / 'label_categories.pkl', 'rb') as f:
        label_categories = pickle.load(f)

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
    model = Model.load_from_checkpoint(
                config['model']['model_ckpt_fname'],
            )
    model.eval()
    
    # --------------------- Trainer ---------------------
    strategy = "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
                    accelerator="auto",
                    devices="auto",
                    deterministic=True,
                    logger=False,
                    # callbacks=False,
                    strategy=strategy,
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                    enable_progress_bar=False,
                    enable_model_summary=True,
                )
    
    inference_data = model.collect_inference_data(
                    datamodule_batch.infer_dataloader()
                )
    adata = inference_data_dict_to_adata(
                inference_data=inference_data,
                label_categories_dict=label_categories,
            )
    
    # --------------------- Infer ---------------------
    # TODO: add validation and test steps
    # compute loss term values and train-time metrics for the entire tissue section using the model in evaluation mode
    print("Validating Model using the best checkpoint...")
    trainer.validate(
        model=model,
        ckpt_path=None,
        datamodule=datamodule_batch,
        verbose=True,
    )

    # --------------------- Compute All Metrics ---------------------
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
    print(tabulate(df, headers='keys', tablefmt='grid'))

    # # --------------------- Plot ---------------------
    # results_dir = Path(config['experiment']['wandb_run_dir']) / 'results'
    # results_dir.mkdir(parents=True, exist_ok=True)
    
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # sns.heatmap(
    #     inference_data['X'].numpy(),
    #     cmap='viridis',
    #     ax=ax[0]
    # )
    # sns.heatmap(
    #     inference_data['X_hat'].numpy(),
    #     cmap='viridis',
    #     ax=ax[1]
    # )
    # title = f"{config['dataset']['dataset_name']}, batch {config['dataset']['adata_batch_idx']}\n" \
    #         f"{config['model']['model_name']}, h={config['model']['encoder_params']['hidden_channels']}, " \
    #         f"k={config['model']['encoder_params']['codebook_params']['codebook_size']}\n" \
    #         f"Mean Pearson 1-hop NBR: {attribute_imputation_metrics['pearson_1hop_nbr'].mean():.3f}"
    # plt.suptitle(title)
    # plt.savefig(
    #     results_dir / 'X_X_hat.png',
    #     dpi=300,
    #     bbox_inches='tight'
    # )


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
    config = collect_test_configs(args)
    
    # --------------------- Test ---------------------
    test(config)