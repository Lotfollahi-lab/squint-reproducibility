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
import os
import scanpy as sc
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import pytorch_lightning as pl

from vqniche.utils.parse_test_configs import *
from vqniche.initializers.initialize import *
from vqniche import metrics
from vqniche.utils.type_conversions import *


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
    model = Model.load_from_checkpoint(config['model']['model_ckpt_fname'])
    
    # --------------------- Trainer ---------------------
    strategy = "ddp_find_unused_parameters_true"
    # strategy = "ddp"
    
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
    
    # --------------------- Validation and Test ---------------------
    # compute model accuracy on validation and test sets
    # print("Computing Model Validation Accuracy...")
    # trainer.validate(
    #     model=model,
    #     datamodule=datamodule_batch,
    #     verbose=True,
    # )
    # print("Computing Model Test Accuracy...")
    # trainer.test(
    #     model=model,
    #     datamodule=datamodule_batch,
    #     verbose=True,
    # )

    # --------------------- Inference on the full dataset ---------------------
    print("Collecting Inference Data...")
    inference_data = model.collect_inference_data(
                        datamodule_batch.infer_dataloader()
                    )
    
    # compute Pearson correlation between the original and reconstructed cell-gene matrices
    attribute_imputation_metrics = model.compute_attribute_imputation_metrics(
                                        data_dict=inference_data
                                    )
    print(attribute_imputation_metrics)

    # compute graph imputation metrics such as MMD between the degree distribution and eigenvalue distribution of the original and reconstructed graphs
    graph_imputation_metrics = model.compute_graph_imputation_metrics(
                                    data_dict=inference_data
                                )
    print(graph_imputation_metrics)

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
    
    # # ------------------ Codebook Utilization ------------------
    # codebook_utilization = 1.0 * len(set(Indices)) / model.encoder.codebook.shape[0]
    # print(f"Codebook Utilization: {codebook_utilization}")
    
    # # Create a DataFrame to store Indices and Labels
    # df = pd.DataFrame({
    #     'Indices': Indices.squeeze().numpy(),
    #     'Labels_Cell_Type': torch.argmax(Labels_cell_type, dim=1).squeeze().numpy(),
    #     'Labels_Niche_Type': torch.argmax(Labels_niche_type, dim=1).squeeze().numpy()
    # })

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # sns.histplot(
    #     data=df,
    #     x='Indices',
    #     hue='Labels_Cell_Type',
    #     ax=ax[0]
    # )
    # sns.histplot(
    #     data=df,
    #     x='Indices',
    #     hue='Labels_Niche_Type',
    #     ax=ax[1]
    # )
    # title = f"{config['dataset']['dataset_name']}, batch {config['dataset']['adata_batch_idx']}\n" \
    #         f"{config['model']['model_name']}, h={config['model']['encoder_params']['hidden_channels']}, " \
    #         f"k={config['model']['encoder_params']['codebook_params']['codebook_size']}\n" \
    #         f"Codebook Utilization: {codebook_utilization}"
    # plt.suptitle(title)
    
    # results_dir = Path(config['experiment']['wandb_run_dir']) / 'results'
    # results_dir.mkdir(parents=True, exist_ok=True)
    # plt.savefig(
    #     results_dir / 'codebook_utilization.png',
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