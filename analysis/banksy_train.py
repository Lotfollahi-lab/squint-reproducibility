import sys
sys.path.append("/lustre/scratch126/cellgen/lotfollahi/am84/VQNiche/baselines/banksy")

import os
import time
from datetime import datetime

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from banksy.initialize_banksy import initialize_banksy
from banksy_utils.load_data import load_adata, display_adata
from banksy_utils.filter_utils import filter_cells, filter_hvg, normalize_total, print_max_min
from banksy_utils.umap_pca import pca_umap
from banksy.embed_banksy import generate_banksy_matrix

model_name = "banksy"
latent_key = f"{model_name}_latent"
lambda_list = [0.2]  # list of lambda parameters

sc.set_figure_params(figsize=(6, 6))
now = datetime.now()
current_timestamp = now.strftime("%d%m%Y_%H%M%S")

dataset = "adata_batch11"
data_folder_path = "/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver/xhs1000-39b_1p/"
benchmarking_folder_path = "./banksy_benchmarking"
figure_folder_path = f"./banksy_benchmarking"


metric_cols_single_sample = [
    "cas", "mlami", # global spatial consistency
    "clisis", "gcs", # local spatial consistency
    "nasw", "cnmi", # niche coherence
]
metric_col_weights_single_sample = [ # separate for each category (later multiplied with category_col_weights)
    (1/8), (1/8), # global spatial consistency
    (1/8), (1/8), # local spatial consistency
    (1/4), (1/4), # niche clustering performance
]
metric_col_titles_single_sample = [
    "CAS", # "Cell Type Affinity Similarity",
    "MLAMI", # "Maximum Leiden Adjusted Mutual Info",
    "CLISIS", # "Cell Type Local Inverse Simpson's Index Similarity",
    "GCS", # "Graph Connectivity Similarity",
    "NASW", # "Niche Average Silhouette Width",
    "CNMI", # "Cell Type Normalized Mutual Info",
]

category_cols_single_sample = [
    "Global Spatial Consistency Score",
    "Local Spatial Consistency Score",
    "Niche Coherence Score"]
category_col_weights_single_sample = [
    0.25,
    0.25,
    0.5]
category_col_titles_single_sample = [
    "Global Spatial Consistency Score",
    "Local Spatial Consistency Score",
    "Niche Coherence Score"]


def train_banksy_models(dataset,
                        cell_type_key,
                        niche_type_key=None,
                        adata_new=None,
                        n_start_run=1,
                        n_end_run=8,
                        n_neighbor_list=[4, 4, 8, 8, 12, 12, 16, 16],
                        gp_inference=False):
    # Configure figure folder path
    dataset_figure_folder_path = f"{figure_folder_path}/{dataset}/single_sample_method_benchmarking/" \
                                 f"{model_name}/{current_timestamp}"
    os.makedirs(dataset_figure_folder_path, exist_ok=True)

    # Create new adata to store results from training runs in storage-efficient way
    if adata_new is None:
        adata_original = sc.read_h5ad(data_folder_path + f"{dataset}.h5ad")
        adata_new = sc.AnnData(sp.csr_matrix(
            (adata_original.shape[0], adata_original.shape[1]),
            dtype=np.float32))
        adata_new.var_names = adata_original.var_names
        adata_new.obs_names = adata_original.obs_names
        adata_new.obs["cell_type"] = adata_original.obs[cell_type_key].values
        if niche_type_key in adata_original.obs.columns:
            adata_new.obs["niche_type"] = adata_original.obs[niche_type_key].values
        adata_new.obsm["spatial"] = adata_original.obsm["spatial"]
        del(adata_original)

    model_seeds = list(range(10))
    for run_number, n_neighbors in zip(np.arange(n_start_run, n_end_run+1), n_neighbor_list):
        # n_neighbors is here used for k_geom parameter in banksy method as well as the latent neighbor graph construction used for
        # UMAP generation and clustering 

        # Load data
        adata = sc.read_h5ad(data_folder_path + f"{dataset}.h5ad")

        start_time = time.time()

        # Set default model hyperparams
        max_m = 1 # use both mean and AFT
        nbr_weight_decay = "scaled_gaussian" # can also choose "reciprocal", "uniform" or "ranked"
        lambda_list = [0.8]
        pca_dims = [20]
        
        # Define spatial coordinates
        adata.obs["spatial_x"] = adata.obsm['spatial'][:, 0]
        adata.obs["spatial_y"] = adata.obsm['spatial'][:, 1]

        banksy_dict = initialize_banksy(
            adata,
            ("spatial_x", "spatial_y", "spatial"),
            n_neighbors,
            nbr_weight_decay=nbr_weight_decay,
            max_m=max_m,
            plt_edge_hist=False,
            plt_nbr_weights=False,
            plt_agf_angles=False, # takes long time to plot
            plt_theta=False)

        banksy_dict, banksy_matrix = generate_banksy_matrix(
            adata,
            banksy_dict,
            lambda_list,
            max_m)

        pca_umap(
            banksy_dict,
            pca_dims = pca_dims,
            add_umap = True,
            plt_remaining_var = False)

        adata.obsm[latent_key] = banksy_dict[nbr_weight_decay][lambda_list[0]]["adata"].obsm["reduced_pc_20"]

        # Measure time for model training
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Duration of model training in run {run_number}: "
              f"{int(hours)} hours, {int(minutes)} minutes and {int(seconds)} seconds.")
        adata_new.uns[f"{model_name}_model_training_duration_run{run_number}"] = (
            elapsed_time)

        # Store latent representation
        adata_new.obsm[latent_key + f"_run{run_number}"] = adata.obsm[latent_key]

        # Store intermediate adata to disk
        if gp_inference:
            adata_new.write(f"{benchmarking_folder_path}/{dataset}_{model_name}_gpinference.h5ad")
        else:
            adata_new.write(f"{benchmarking_folder_path}/{dataset}_{model_name}.h5ad")  

    # Store final adata to disk
    if gp_inference:
        adata_new.write(f"{benchmarking_folder_path}/{dataset}_{model_name}_gpinference.h5ad")
    else:
        adata_new.write(f"{benchmarking_folder_path}/{dataset}_{model_name}.h5ad")
        

def analyze_banksy_models(dataset,
                          model_name):
    datasets = [dataset]
    models = [model_name]

    summary_df = pd.DataFrame()
    for dataset in datasets:
        dataset_df = pd.DataFrame()
        for model in models:
            try:
                benchmark_df = pd.read_csv(f"{benchmarking_folder_path}/{dataset}_{model}_metrics.csv")
                #adata = sc.read_h5ad(f"../../artifacts/single_sample_method_benchmarking/{dataset}_{model}.h5ad")
                #training_durations = []
                #for run_number in [1, 2, 3, 4, 5, 6, 7, 8]:
                #    training_durations.append(adata.uns[f"{model.split('_')[0]}_model_training_duration_run{run_number}"])
                #benchmark_df["run_time"] = training_durations
                #benchmark_df = benchmark_df[["dataset", "run_number", "run_time", "gcs", "mlami", "cas", "clisis", "nasw", "cnmi", "cari", "casw", "clisi"]]
                #benchmark_df.to_csv(f"{benchmarking_folder_path}/{dataset}_{model}_metrics.csv", index=False)
                benchmark_df["model"] = model
                dataset_df = pd.concat([dataset_df, benchmark_df], ignore_index=True)
            except FileNotFoundError:
                print(f"Did not find file {benchmarking_folder_path}/{dataset}_{model}_metrics.csv. Continuing...")
                missing_run_data = {
                    "dataset": [dataset] * 8,
                    "model": [model] * 8,
                    "run_number": [1, 2, 3, 4, 5, 6, 7, 8],
                    "run_time": [np.nan] * 8
                }
                missing_run_df = pd.DataFrame(missing_run_data)
                dataset_df = pd.concat([dataset_df, missing_run_df], ignore_index=True)
                
        # Apply min-max scaling to metric columns
        for i in range(len(metric_cols_single_sample)):
            min_val = dataset_df[metric_cols_single_sample[i]].min()
            max_val = dataset_df[metric_cols_single_sample[i]].max()
            dataset_df[metric_cols_single_sample[i] + "_scaled"] = ((
                dataset_df[metric_cols_single_sample[i]] - min_val) / (max_val - min_val))

        summary_df = pd.concat([summary_df, dataset_df], ignore_index=True)
        continue
        
    cat_0_scaled_metric_cols = [metric_col + "_scaled" for metric_col in metric_cols_single_sample[0:2]]
    cat_1_scaled_metric_cols = [metric_col + "_scaled" for metric_col in metric_cols_single_sample[2:4]]
    cat_2_scaled_metric_cols = [metric_col + "_scaled" for metric_col in metric_cols_single_sample[4:6]]
        
    summary_df[category_cols_single_sample[0]] = np.average(summary_df[cat_0_scaled_metric_cols],
                                                            weights=metric_col_weights_single_sample[0:2],
                                                            axis=1)
    summary_df[category_cols_single_sample[1]] = np.average(summary_df[cat_1_scaled_metric_cols],
                                                            weights=metric_col_weights_single_sample[2:4],
                                                            axis=1)
    summary_df[category_cols_single_sample[2]] = np.average(summary_df[cat_2_scaled_metric_cols],
                                                            weights=metric_col_weights_single_sample[4:6],
                                                            axis=1)
    summary_df["Overall Score"] = np.average(summary_df[category_cols_single_sample[:3]],
                                            weights=category_col_weights_single_sample[:3],
                                            axis=1)
    
    print(summary_df)
    
    # # Reformat for plot
    # # summary_df.replace({"nichecompass_gatv2conv": "NicheCompass",
    # #                     #"nichecompass_gcnconv": "NicheCompass Light",
    # #                     "staci": "STACI",
    # #                     "deeplinc": "DeepLinc",
    # #                     "expimap": "expiMap",
    # #                     "graphst": "GraphST",
    # #                     "cellcharter": "CellCharter",
    # #                     "banksy": "BANKSY"},
    # #                 inplace=True)

    # # Filter for just second run
    # summary_df = summary_df[summary_df["run_number"] == 2]

    # # Plot over all loss weights combinations
    # # Prepare metrics table plot
    # group_cols = ["dataset", "model"]
    # aggregate_df = summary_df.groupby(group_cols).mean("Overall Score").sort_values("Overall Score", ascending=False)[
    #     metric_cols_single_sample + ["Overall Score"]].reset_index()

    # unrolled_df = pd.melt(aggregate_df, 
    # id_vars=group_cols,
    # value_vars=metric_cols_single_sample + ["Overall Score"],
    # var_name="score_type", 
    # value_name="score")

    # # Create spatial indicator column
    # def is_spatially_aware_model(row):
    #     if row["model"] in ["NicheCompass", "STACI", "DeepLinc", "GraphST", "SageNet"]:
    #         return True
    #     return False
    # unrolled_df["spatially_aware"] = unrolled_df.apply(lambda row: is_spatially_aware_model(row), axis=1)
    # unrolled_df = unrolled_df[["dataset", "spatially_aware", "model", "score_type", "score"]]

    # # Order datasets
    # unrolled_df["dataset"] = pd.Categorical(unrolled_df["dataset"], categories=datasets, ordered=True)
    # unrolled_df = unrolled_df.sort_values(by="dataset")

    # unrolled_df["model"] = unrolled_df["model"].replace("NicheCompass GATv2", "NicheCompass")
    # #unrolled_df["model"] = unrolled_df["model"].replace("NicheCompass GCN", "NicheCompass Light")

    # # Plot table
    # plot_simple_metrics_table(
    #     df=unrolled_df,
    #     model_col="model",
    #     model_col_width=1.6,
    #     group_col="dataset",
    #     metric_cols=metric_cols_single_sample, # metric_cols_single_sample, category_cols_single_sample
    #     metric_col_weights=metric_col_weights_single_sample, # metric_col_weights_single_sample, category_col_weights_single_sample
    #     metric_col_titles=[col.replace(" ", "\n") for col in metric_col_titles_single_sample], # category_col_titles_single_sample
    #     metric_col_width=0.7, # 0.8,
    #     aggregate_col_width=1.2,
    #     plot_width=8, # 8.5, # 32,
    #     plot_height=7,# 8,
    #     show=True,
    #     save_dir=benchmarking_folder_path,
    #     save_name=f"benchmarking_metrics_slideseqv2_mouse_hippocampus_run2.svg")