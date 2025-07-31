"""
This script provides a pipeline for a single testing run of a Model on a single batch of data. This assumes that the data has already been processed and stored as a PyTorch Geometric dataset in the "gold" directory and the model has been trained using the train_model.py script. The input to this file is the path to the wandb run directory of the trained model.

The pipeline consists of the following steps:
1. Load the user-specified config file used for training the model from the wandb run directory.
2. Initialize the DatasetBlob (see vqniche.dataset.in_memory_dataset_blob.py).
3. Load the data corresponding to the batch specified in the config file.
4. Initialize the DataModule (see vqniche.dataloaders.in_memory_datamodule.py).
5. Load the Model from the checkpoint. The checkpoint can be specified via a command line argument or automatically determined from the wandb run directory.
6. Initialize the Trainer (see vqniche.utils.initialize.py).
7. Run the Validation step(s) of the model.
8. Run the Inference step of the model and collect all the data needed for the downstream analysis.
9. Convert the inference data to an AnnData object.
10. Compute the metrics for the model if the --compute_metrics flag is set.
11. Plot the UMAP of the original and imputed attributes colored by the cell types and niche types if the --plot_figures flag is set.
12. Plot the loss and metrics as a function of epoch if the --plot_figures flag is set.

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
from vqniche.plotting import *
from vqniche.utils.loss_utils import aggregate_1hop_neighbor_features


codebook_metrics = ['codebook_utilization']
codebook_metrics = []
attr_impute_metrics = ['pearson_cell_wise', 'pearson_1hop_nbr']
graph_impute_metrics = ["mmd_degree", "num_edges", "max_degree"]
sup_global_spatial_conserve_metrics = ['cas']
unsup_global_spatial_conserve_metrics = ['mlami']
unsup_local_spatial_conserve_metrics = ['gcs']
sup_niche_cohere_metrics = ['casw']
sup_niche_cohere_metrics = []
unsup_niche_cohere_metrics = ['nasw']
unsup_niche_cohere_metrics = []
METRICS_LIST = codebook_metrics + attr_impute_metrics + graph_impute_metrics + unsup_global_spatial_conserve_metrics + unsup_local_spatial_conserve_metrics + unsup_niche_cohere_metrics + sup_global_spatial_conserve_metrics + sup_niche_cohere_metrics


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
                    strategy=strategy,
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                    enable_progress_bar=False,
                    enable_model_summary=True,
                )

    # --------------------- Results ---------------------
    # create a results directory for the model run with the name of the model checkpoint file
    results_dir = Path(config['experiment']['wandb_run_dir']) / 'results' / config['model']['model_ckpt_fname'].stem
    results_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- Validate and Test ---------------------
    # TODO: replace with trainer.test() that computes metrics for nodes in the test set
    # compute loss term values for nodes in the validation set and metrics that were computed during training for the entire tissue section 
    # print("Validating Model...")
    # trainer.validate(
    #     model=model,
    #     ckpt_path=None,
    #     datamodule=datamodule_batch,
    #     verbose=True,
    # )

    # --------------------- Infer ---------------------
    infer_dict_fname = results_dir / 'inference_data_dict.pkl'
    print(f"Computing inference data and saving to {infer_dict_fname}...")
    inference_data = model.collect_inference_data(
                    datamodule_batch.infer_dataloader()
                )
    with open(infer_dict_fname, 'wb') as f:
        pickle.dump(inference_data, f)
    
    # --------------------- Convert Inference Data to AnnData ---------------------
    adata_fname = results_dir / 'inference_adata.pkl'
    print(f"Converting inference data to AnnData and saving to {adata_fname}...")
    adata = inference_data_dict_to_adata(
                inference_data=inference_data,
                label_categories_dict=label_categories,
            )
    if not adata_fname.exists():
        with open(adata_fname, 'wb') as f:
            pickle.dump(adata, f)

    # --------------------- Metrics (if --compute_metrics flag is set) ---------------------
    if config['experiment']['compute_metrics']:

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
        with open(results_dir / 'metrics.txt', 'w') as f:
            f.write(tabulate(df, headers='keys', tablefmt='grid'))

        # save the metrics to a CSV file
        df.to_csv(results_dir / 'metrics.csv', index=False)

    # --------------------- Draw Plots (if --plot_figures flag is set) ---------------------
    if config['experiment']['plot_figures']:

        fig_dir = results_dir / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # -------- Draw UMAP Plots of Original and Imputed Attributes ---------
        # compute UMAP embeddings for the original and imputed 1-hop neighbor attributes
        embedding_keys = ['X', 'X_hat', 'X_nbr', 'X_hat_nbr']
        adata.uns['X_nbr'] = aggregate_1hop_neighbor_features(
            X=adata.uns['X'],
            edge_index=adata.uns['edge_index'],
            return_mean=False,
        )
        adata.uns['X_hat_nbr'] = aggregate_1hop_neighbor_features(
            X=adata.uns['X_hat'],
            edge_index=adata.uns['edge_index'],
            return_mean=False,
        )
        adata = compute_umap(
                adata=adata,
                embedding_keys=embedding_keys,
            )
        
        # plot UMAP embeddings for the original and imputed attributes colored by the cell types
        save_fname = fig_dir / 'UMAP_X_X_hat_cell_types.png'
        plot_umap_attribute_imputation(
                adata=adata,
                embedding_keys=embedding_keys,
                label_key='cell_types',
                save_fname=save_fname,
            )
        
        # plot UMAP embeddings for the original and imputed attributes colored by the niche types
        save_fname = fig_dir / 'UMAP_X_X_hat_niche_types.png'
        plot_umap_attribute_imputation(
                adata=adata,
                embedding_keys=embedding_keys,
                label_key='niche_types',
                save_fname=save_fname,
            )
        
        # -------- Draw Loss and Metrics vs Epoch ---------
        # read the on_train_epoch_end_logs.csv file from the wandb run directory
        df_fname = Path(config['experiment']['wandb_run_dir']) / 'files' / 'on_train_epoch_end_logs.csv'
        df_loss, df_metrics = read_on_train_epoch_end_logs(
                                    file_path=df_fname,
                                )
        
        # plot the loss and metrics as a function of epoch
        save_fname = fig_dir / 'loss_vs_epoch.png'
        plot_logged_values_vs_epoch(
            df=df_loss,
            value_col="Value",
            name_col="Loss Term",
            mode_col="Mode",
            title="Losses vs Epoch",
            save_fname=save_fname,
        )

        # plot the metrics as a function of epoch
        save_fname = fig_dir / 'metrics_vs_epoch.png'
        plot_logged_values_vs_epoch(
            df=df_metrics,
            value_col="Value",
            name_col="Metric",
            mode_col="Mode",
            title="Metrics vs Epoch",
            save_fname=save_fname,
        )

        # -------- Draw Code Assignment on X-Y Coordinates ---------
        save_fname = fig_dir / 'code_assignment_on_xy_coordinates.png'
        plot_code_assignment_on_xy_coordinates(
                adata=adata,
                save_fname=save_fname,
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
    args = parse_test_arguments()
    config = collect_test_configs(args)
    
    # --------------------- Test Pipeline---------------------
    test(config)