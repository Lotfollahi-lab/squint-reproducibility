# SQUINT reproducibility

Code to reproduce the experiments of

> **Learning Discrete Cell and Niche Codes from Spatial Transcriptomics Using Dual Residual Vector Quantization**\
> Sebastian Birk, Arpit Merchant, Amirhossein Vahidi, Fabian J. Theis, Mo Lotfollahi.\
> Proceedings of the Machine Learning in Computational Biology (MLCB) 2026, PMLR.

The model lives in [Lotfollahi-lab/squint](https://github.com/Lotfollahi-lab/squint) (Python
package `squint`). The configuration of every SQUINT model reported in the paper, including the
ablations and the region-holdout model, is defined in the `VARIANTS` registry of squint
`examples/run_squint.py` rather than in separate YAML files, and the models are trained with the
LSF launchers in squint `examples/`; the stage-2 settings are the options of its stage-2
launchers. Section 3 names the variant and options behind every table column and figure, and each
run writes its complete configuration to `user_specified_config.yaml` (stage 2:
`stage2_config.json`). This repository holds everything around the model: wrappers and
launchers for the baselines, data-preparation helpers, the corrected-iLISI pipeline, scoring,
statistics and the figure panels. Clone the two repositories side by side: several scripts here
look for `squint/examples/` in a sibling folder.

## Code version

- The tag **`mlcb2026-camera-ready`** marks the camera-ready state of both repositories and
  contains every script referenced below.
- At the tag, the stage-1 model code of squint (`src/squint/` outside `stage2/`) is unchanged
  since commit **`4d561b0`** (28 June 2026) and `src/squint/stage2/` since **`9802adf`**
  (30 June 2026), apart from the package rename.
- **iLISI.** squint commit **`89d9c82`** corrected `compute_ilisi` in
  `examples/compute_inference_metrics.py`, which every SQUINT run and every baseline runner here
  uses: cells are permuted before the neighbour search, so exact ties in SQUINT's quantized
  embeddings no longer pick same-section neighbours, and iLISI requires `scib-metrics` instead
  of falling back to an unscaled statistic. The reported runs predate it. Their iLISI values in
  Tables 1 and 3, Table S1 and Fig. S3b,d were recomputed from the stored embeddings with
  `analysis/benchmarking/ilisi_recompute/` (Section 4; Supplement App. C); NMI, ARI, MMD and
  runtimes are those of the original runs. Runs made at the tag compute the corrected statistic
  directly (one permutation, where the recompute averages three), except in baseline
  environments without `scib-metrics`, where iLISI is now skipped.
- **Read depth for imputation.** The SQUINT (MC) and SQUINT (No MC) values of Table 2 were
  decoded with `--read-depth-mode neighbor` (`READ_DEPTH_MODE=neighbor` in
  `submit_stage2_mc_sweep.sh`): a held-out cell's library size is the mean of its 16 nearest
  observed cells, as for GeST, CIFM and the k-NN floor. The default of both the decoder and the
  launcher is `true`, which uses the held-out cell's own library size and so reads part of the
  target. Always set it explicitly; the decoder logs its choice (`[decode] [read-depth]`).
- **Trained checkpoints are not distributed.** Each run writes its own to `<run>/checkpoints/`.

## Before you start: adapt the cluster paths

The scripts were written for the authors' LSF cluster and default to paths under
`/nfs/team361/...` and `/lustre/...`. Set the following before running anything:

```bash
export REPO=/path/to/squint-reproducibility   # this repository (read by its launchers)
export ART=$REPO/artifacts                    # all run outputs (git-ignored); keep them here, see below
export ARTIFACTS_ROOT=$ART LOG_ROOT=$ART/logs WANDB_BASE=$ART/wandb_cache
export VENV_ROOT=/path/to/venvs               # one virtual environment per method (Section 1)
export VENV_PATH=$VENV_ROOT/squint            # environment of the SQUINT launchers
export LSF_GROUP=<your LSF group>             # queues via LSF_QUEUE or --queue
DATA=/path/to/DATASETS                        # silver/ (per-section h5ads) and gold/ (dataset blobs)
export SILVER_ROOT=$DATA/silver
# used only in this README's commands:
SQUINT=/path/to/squint                        # clone of Lotfollahi-lab/squint, next to this repository
H="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1"
```

`$H` is the common prefix of the paper's variants other than the Mouse Brain default
(`s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p`) and its ablations. Keep `$ART` at
`$REPO/artifacts` (a symbolic link works): `submit_label_metrics.sh` reads the runs from
`$REPO/artifacts` whatever `$ART` is set to (`submit_recompute_ilisi.sh` follows `$ART`).

Hard-coded paths that have no environment variable and must be edited:

| File | Line(s) | Set to |
|---|---|---|
| squint `examples/run_squint.py` | `DATA_ROOT` (l. 173), `REPRO_REPO` (l. 179), `SQUINT_PKG` (l. 180), `ARTIFACTS_DIR` (l. 184) | `$DATA`, `$REPO`, `$SQUINT`, `$ART` |
| squint `examples/run_squint.py` | CosMx and Xenium data roots: `data_directory_path` (ll. 314, 362, 620), `root_data_dir` (ll. 4359, 4437, 4645) | `$DATA` |
| squint `examples/run_squint.py` | `"offline"` (l. 907); training logs to Weights & Biases online by default | `True` to log locally only |
| `analysis/benchmarking/cell_type_identification/run_pca_leiden.py` | `_import_metric_helpers` (ll. 137–141); every baseline runner imports the metric code through it | `$SQUINT/examples` |
| `analysis/benchmarking/niche_identification/compute_label_metric.py` | ll. 115–116 | `$SQUINT/examples` |
| `analysis/benchmarking/gene_expr_imputation/_holdout_utils.py` | `DEFAULT_ARTIFACTS_ROOT` (ll. 67–69); `submit_imputation_recon.sh` does not pass `--artifacts-root` to the Table 2 runners | `$ART` |

Other Python scripts take the paths as options (`--artifacts-root`, `--silver-dir`, `--out-dir`,
...); the commands below pass them.

**Without LSF.** Most launchers accept `DRY_RUN=1` and then print their `bsub` commands instead
of submitting them; the command after the `bsub` options (a runner script, or
`bash -lc '<job>'`) runs unchanged on a GPU machine. The exceptions:

- SQUINT seeds: see "Several seeds" in the squint README. The sweep wrappers
  `submit_s57_crossbatch_ablations_multiseed.sh` and `submit_s65_emadecay_multiseed.sh` print one
  `bash submit_multi_seed.sh <variant> <seeds>` call per variant; run each variant as described
  there.
- `submit_label_metrics.sh` prints only a job table; Section 3.6 gives the direct command.
- `submit_recompute_ilisi.sh` prints only its first job; Section 4 gives the direct commands.

The paper's models were trained on a single NVIDIA H200 GPU, except scGPT and scGPT-spatial,
which ran on an A100 because the torch build of their environment has no H200 kernels
(Supplement App. C). The Table 1 baseline wrappers submit every baseline with one node class
(`UNIFORM_RESOURCE=gpu_high_memory`: 8 cores, 384 GB, one GPU). SQUINT seeds request 24 cores
(22 DataLoader workers), 128 GB and one GPU (`submit_multi_seed.sh`).

## 1. Environments

Every environment is a virtual environment under `$VENV_ROOT`, named as the launchers expect
(the `METHODS` table in `analysis/benchmarking/submit_all_benchmarks.sh` and `method_spec()` in
`analysis/benchmarking/gene_expr_imputation/submit_imputation_recon.sh`).

```bash
cd $SQUINT && UV_PROJECT_ENVIRONMENT=$VENV_ROOT/squint uv sync --python 3.10   # squint: torch 2.2.0+cu121 and PyG pins from pyproject.toml
cd $REPO/analysis/benchmarking/gene_expr_imputation
VENV=$VENV_ROOT/cifm bash install_cifm.sh    # CIFM: the same torch/PyG pins plus e3nn; fetches the ynyou/CIFM checkpoint
```

| Environment | Methods | Recipe |
|---|---|---|
| `squint` | SQUINT, stage 2, GeST, k-NN floor, Vanilla VQ-VAE, PCA-Leiden, neighbourhood-expression PCA; all analysis and plotting scripts | above |
| `cifm` | CIFM | `install_cifm.sh` (above); `verify_cifm.py` checks it |
| `cellcharter` | CellCharter; scVI (Tables 1 and 2) | not packaged |
| `banksy` | BANKSY; Harmony | not packaged |
| `graphst` | GraphST (with PASTE) | not packaged |
| `novae` | Novae | not packaged |
| `nichecompass` | NicheCompass (Tables 1 and 2) | not packaged |
| `geneformer`, `nicheformer` | Geneformer, Nicheformer; `add_ensembl_ids.py` (Section 2; `geneformer` has `mygene`) | install notes in the runner docstrings |
| `scgpt-spatial` | scGPT, scGPT-spatial | install notes in the runner docstrings |
| `tissuejepa` | UCE | `uce-model` package (runner docstring) |

Only the `squint` and `cifm` environments have recipes here; the others were built from the
methods' released packages. `cell_type_identification/install_scgpt_h200.sh` builds an
H200-compatible scGPT environment (`scgpt-h200`), which the reported runs did not use.
`envs/rsc_rapids_25.10.yaml` is an optional RAPIDS (rapids-singlecell) conda environment for GPU
UMAP and Leiden.

Pretrained weights and auxiliary files are git-ignored and must be placed at these defaults (or
pointed to with the variables in brackets):

| Method | Default location under `analysis/benchmarking/` |
|---|---|
| Geneformer | `geneformer/geneformer-v2-104M/Geneformer-V2-104M` (`GENEFORMER_MODEL_DIR`) |
| Nicheformer | `nicheformer/data` (`NICHEFORMER_MODEL_DIR`) |
| scGPT | `scGPT/scGPT_human` (`SCGPT_MODEL_DIR`) |
| scGPT-spatial | `scGPT-spatial/checkpoints/scGPT_spatial_v1` (`SCGPT_SPATIAL_MODEL_DIR`) |
| UCE | `uce_model/33l_8ep_1024t_1280.torch` (`UCE_MODEL_LOC`, `UCE_MODEL_FILES_DIR`) |
| NicheCompass | `nichecompass/human_mouse_gene_orthologs.csv`, `nichecompass/metabolite_enzyme_sensor_gps` (`NICHECOMPASS_ORTHOLOGS_CSV`, `NICHECOMPASS_MEBOCOST_DIR`; Table 2: `NC_ORTHOLOGS`, `NC_MEBOCOST`) |
| CIFM | `cifm` (`CIFM_REPO`); fetched by `install_cifm.sh` through `download_cifm.py` |

### Upgrading an environment from `vqniche`

The Python package used to be called `vqniche`. To update an existing environment and checkout,
pull both repositories, delete the leftover `src/vqniche` folder in the squint clone (git leaves
its ignored `__pycache__` folders behind) and replace the package:

```bash
pip uninstall -y vqniche          # or: uv pip uninstall vqniche
pip install -e /path/to/squint    # or: uv pip install -e /path/to/squint
python -c "import squint; print(squint.__file__, squint.__version__)"
```

The earlier pipeline in `scripts/` activates a conda environment named `squint-reproducibility`
and falls back to its old name, `vqniche-reproducibility`, until it is renamed (`conda rename -p
<envs>/vqniche-reproducibility <envs>/squint-reproducibility`, with full paths). Re-register the
notebooks' Jupyter kernel (`python -m ipykernel install --user --name squint-reproducibility`) and
remove the old one (`jupyter kernelspec remove -y vqniche-reproducibility`). Runs of
`analysis/train_model.py` now write to `.../Squint/` instead of `.../VQNiche/`, and its W&B project
is `squint` (previously `VQNiche`).

## 2. Data

Every method reads the same per-section `.h5ad` files (raw counts, `obsm["spatial"]`,
`obs["cell_id"]`, `uns["batch"]`; see "Input data" in the squint README), one folder per dataset
under `$DATA/silver/`:

| Dataset (paper) | Folder under `$DATA/silver/` | Source | Content |
|---|---|---|---|
| Mouse Brain (Tables 1–3, S1, S2; Figs. 2, S2–S5) | `mmb0-1b_smb1-1b_1p_coord_aligned/`: `harmonised_starmap_plus_mouse_cns_batch15_shared_genes.h5ad`, `harmonised_merfish_mouse_brain_239-batch_batch82_shared_genes.h5ad` | STARmap PLUS: [Zenodo 8327576](https://zenodo.org/records/8327576); MERFISH: [CELLxGENE](https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09) | one section each (`uns["batch"]` `batch15` and `batch82`), 86,822 cells, 431 shared genes; `cell_type` (26 and 23 types); niches `Sub_molecular_tissue_region` (STARmap, 63) and `ccf_region_name` (MERFISH, 167) |
| CosMx NSCLC (Tables 1, S2, S3) | `chl59-2b_1p/` | [Bruker Spatial Biology](https://brukerspatialbiology.com/products/cosmx-spatial-molecular-imager/ffpe-dataset/nsclc-ffpe-dataset/) | two sections of one donor, 199,672 cells; `cell_type` (10), `niche` (12) |
| CosMx NSCLC cohort (Figs. 3, S6) | `chl59-8b_1p/` | as above | eight samples of five donors; `Lung13` and `Lung5_Rep3` are the queries |
| Xenium Eczema (Tables 1, S2, S3) | `xhs1000-3b_1p/` | not public | three sections of different donors, 53,655 cells; `new_annotation` (21), `niche_type` (12) |

The preprocessing from these sources to the silver files is not packaged in this repository. The
CosMx files carry a `var["ensembl_id"]` column, which Geneformer and Nicheformer read. Keep the
folder names: `submit_all_benchmarks.sh` resolves `$SILVER_ROOT/<dataset>` (for the Mouse Brain
`<dataset>_coord_aligned`), and the dataset tag names the output folder `$ART/<dataset>/` that the
scoring scripts read. The `_coord_aligned` suffix names the prepared Mouse Brain files, whose two
sections share one coordinate frame. SQUINT does not depend on it: its k-NN graph is built within
each section and never joins two sections.

**Xenium human skin (Table 1, Tables S2 and S3).** These data are not publicly available: they
come from the atopic-eczema cohort of Steele et al. (2026, bioRxiv) and were used with the data
owners' permission. Only the scripts are provided:

```bash
cd $REPO/analysis/data_preparation
# sections 11, 19 and 32 of the 39-section cohort, with uns["batch"] stamped and unique cell names
python prep_xhs_3b.py --src-dir <39-section silver folder> --out-dir $DATA/silver/xhs1000-3b_1p
# var["ensembl_id"] for Geneformer and Nicheformer (needs internet and mygene, which the
# squint environment lacks: run it in the geneformer environment, or pip install mygene)
$VENV_ROOT/geneformer/bin/python add_ensembl_ids.py --silver-dir $DATA/silver/xhs1000-3b_1p
```

SQUINT reads a dataset blob built once per dataset (written to `$DATA/gold/`):

```bash
cd $SQUINT && source $VENV_PATH/bin/activate
for d in mmb0-1b_smb1-1b_1p chl59-2b_1p chl59-8b_1p xhs1000-3b_1p; do
  python examples/run_squint.py --build-blob --build-blob-dataset $d
done
```

## 3. Experiments

| Paper element | Experiment | Section |
|---|---|---|
| Table 1, §4.2–4.3, Figs. 2b, S3 | Niche and cell-type identification and integration: SQUINT and 11 baselines, 3 datasets (seeds 0–4) | 3.1 |
| Table 2, §4.4, Fig. S5 | Reconstruction and imputation, Mouse Brain with held-out regions (seeds 0–4) | 3.2 |
| §4.4, Supplement App. C | CIFM, zero-shot | 3.3 |
| Table 3, Table S1, §4.6 | Ablations (seeds 0–4) | 3.4 |
| Table S2, §4.6 | Codebook specialisation | 3.5 |
| Table S3, §4.3 | Label-conditioned iLISI (CiLISI) | 3.6 |
| Fig. 3, Fig. S6, §4.5 | Section-level query-to-reference mapping on the CosMx NSCLC cohort | 3.7 |
| Fig. S4, §3.3 | Codebook usage and EMA-decay sweep | 3.8 |

SQUINT runs write `$ART/<dataset>/<variant>/<YYYYMMDD_HHMMSS>_seed<N>/`, and a multi-seed sweep
adds `$ART/<dataset>/<variant>__multiseed/<timestamp>/metrics/per_seed_*.csv` (see "Outputs" in
the squint README). The Table 1 baseline runners write
`$ART/<dataset>/baseline-<method>/<timestamp>/` with the same `metrics/per_seed_*.csv` files. The
Table 2 runners write folders ending in `+region-holdout`, mostly
`baseline-<method>+region-holdout`; the exceptions are GeST (`gest-imputed+region-holdout`) and
the Vanilla VQ-VAE (`vanilla-vq-cell+region-holdout`, `vanilla-vq-nbr+region-holdout`). The
scoring scripts read the newest timestamp of each folder. Run the SQUINT launchers from the squint
clone with its environment active, and the Python helpers here in the `squint` environment
(except `add_ensembl_ids.py`, Section 2).

### 3.1 Identification and integration benchmark (Table 1; Figs. 2b, S3)

SQUINT: three datasets, one configuration (see "Key model options" in the squint README), each run
trained on all sections of its dataset.

```bash
cd $SQUINT && source $VENV_PATH/bin/activate
bash examples/submit_multi_seed.sh "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p" 0,1,2,3,4   # Mouse Brain
bash examples/submit_multi_seed.sh "$H+chl59-2b_1p" 0,1,2,3,4                                    # CosMx NSCLC
bash examples/submit_multi_seed.sh "$H+xhs1000-3b_1p" 0,1,2,3,4                                  # Xenium Eczema
```

Baselines: each runner embeds the data, clusters with Leiden at the resolution that gives
30 clusters (Novae: its own domain assignment at 30 domains, `assign_domains(level=30)`) and
repeats this for seeds 0–4. Novae and the single-cell foundation models are applied zero-shot.

```bash
cd $REPO/analysis/benchmarking
for d in mmb0-1b_smb1-1b_1p chl59-2b_1p xhs1000-3b_1p; do
  bash submit_niche_id_baselines.sh $d -m nichecompass,cellcharter,banksy,graphst,novae
  bash submit_cell_type_baselines.sh $d -m scvi,nicheformer,uce,geneformer
  bash submit_cell_type_baselines.sh $d -m scgpt,scgpt-spatial --queue <A100 queue>   # or GPU_MODEL=<A100 model token>
done
```

`run_banksy.py` applies Harmony to the BANKSY embedding before clustering, and `run_graphst.py`
aligns the sections with PASTE before training GraphST (on the GPU, `--paste-use-gpu` in the
`METHODS` table of `submit_all_benchmarks.sh`). The per-dataset block of the same script sets the
gene-ID handling of the foundation models (mouse-to-human orthologs on the Mouse Brain, the
Ensembl column on the human data) and the Xenium label columns. GraphST ran out of
memory on CosMx NSCLC (n/a in Table 1). PCA-Leiden, PCA-Harmony and neighbourhood-expression PCA
are wired in but not reported.

### 3.2 Reconstruction and imputation (Table 2; Fig. S5)

Stage 1 is the paper's Mouse Brain model with one 25% × 25% box held out in each section
(STARmap upper left, MERFISH lower right; 6,612 cells). The held-out cells are not used as
reconstruction targets and are tagged `data_split = test`; the baselines use the same boxes
(`DEFAULT_HOLDOUT_REGIONS` in `gene_expr_imputation/_holdout_utils.py`).

```bash
cd $SQUINT && source $VENV_PATH/bin/activate
bash examples/submit_multi_seed.sh "$H+region-holdout+mmb0-1b_smb1-1b_1p" 0,1,2,3,4

# stage 2 on each seed's frozen codes; S1_SEEDS lists the five seed run folders of that sweep
S1=$ART/mmb0-1b_smb1-1b_1p/$H+region-holdout+mmb0-1b_smb1-1b_1p
S1_SEEDS="<TS>_seed0 <TS>_seed1 <TS>_seed2 <TS>_seed3 <TS>_seed4"
STAGE1_VARIANT_DIR=$S1 SEEDS="$S1_SEEDS" VARIANTS=decode-temp05-nonoise \
    bash examples/submit_stage2_ablation.sh
# decode the in-painted codes: single decode (No MC) and S = 1000 samples (MC), leak-free depth
STAGE1_VARIANT_DIR=$S1 SEEDS="$S1_SEEDS" ONLY_NICHE_FROM_CELL=1 NICHE_FROM_CELL_KS="0 1000" \
    READ_DEPTH_MODE=neighbor bash examples/submit_stage2_mc_sweep.sh
```

The first launcher trains the prior (`run_stage2.py --holdout-split test`; MaskGIT decoding with
20 steps, temperature 0.5 and no confidence noise) into
`$ART/mmb0-1b_smb1-1b_1p/stage2-ablation/decode-temp05-nonoise/<seed>/` and scores it with the
default, leaky read depth; those metrics are not the reported ones. The second re-decodes the
saved codes into `.../stage2-ablation/decode-temp05-nonoise-nichecell/` (SQUINT (No MC)) and
`.../decode-temp05-nonoise-mc1000-nichecell/` (SQUINT (MC)). The `-nichecell` arms build the
neighbourhood level by averaging the cell-level prediction over the 16-NN graph, as for the
baselines. Both launchers default to the authors' seed folders, so always pass `SEEDS`.
`analysis/ablations/rank_stage2_ablations.py` ranks the arms of `submit_stage2_ablation.sh`;
only `decode-temp05-nonoise` is reported.

Baselines:

```bash
cd $REPO/analysis/benchmarking
SILVER_DIR=$DATA/silver/mmb0-1b_smb1-1b_1p_coord_aligned \
METHODS="scvi scvi-nbr nichecompass vanilla-cell vanilla-nbr gest knn" \
    bash gene_expr_imputation/submit_imputation_recon.sh
```

Reconstruction: scVI on per-cell counts (`scvi`) and on 1-hop neighbourhood means (`scvi-nbr`),
NicheCompass, and the single-codebook Vanilla VQ-VAE (`vanilla-cell`, `vanilla-nbr`). Imputation:
GeST (`gest`; our reimplementation in `gene_expr_imputation/gest/`, with tests in
`gest/tests/`, as no code is released) and the spatial k-NN floor (`knn`). Each runner applies
the shared holdout and prints `[holdout] data_split: train=…, test=…`; check that it reports
`test=6612`. Point `SILVER_DIR` at the Mouse Brain files that SQUINT reads: the default
(`DEFAULT_SILVER_DIR` in `_holdout_utils.py`, ll. 64–66) is another folder on the authors'
cluster. `mlp` and `gestarch` are wired in but not reported.

### 3.3 CIFM (§4.4, Supplement App. C)

The released checkpoint is applied zero-shot (no training code is released); 413 of the 431
mouse genes map to human orthologs. CIFM's own 20 µm radius graph is the reported setting and a
k-NN graph the sensitivity check. The checkpoint is deterministic, so one seed suffices.

```bash
cd $REPO/analysis/benchmarking
MB=$DATA/silver/mmb0-1b_smb1-1b_1p_coord_aligned
SILVER_DIR=$MB METHODS=cifm SEEDS=0 bash gene_expr_imputation/submit_imputation_recon.sh         # radius graph
SILVER_DIR=$MB METHODS=cifm SEEDS=0 CIFM_FLAGS="--graph knn --variant-tag <new folder name>" \
    bash gene_expr_imputation/submit_imputation_recon.sh                                          # k-NN graph
```

The supplement states that CIFM's hard dropout gate was disabled. `run_cifm.py` defaults to the
gate (`--output gate`), and the setting of the reported run is not recorded; `--output marginal`
(add it to `CIFM_FLAGS`) is the ungated expectation. `run_cifm.py` checks its wrapper against the
released inference path (`verify_native_equivalence`, tolerance 1e-4).
`reproduce_cifm_fig2.py` reproduces the authors' Spearman correlation on their demo data;
`cifm_mse_definition.py`, `cifm_mse_remaining.py` and `cifm_target_sum_sweep.py` document that
their reported MSE is not reproduced; the other `cifm_*.py` scripts are diagnostics.

### 3.4 Ablations (Table 3, Table S1)

Each ablation is one `s57_v*` variant of the Mouse Brain model, trained with seeds 0–4 like the
default (`s57_v19`). One launcher submits all 33 (it includes `s57_v19`; `s57_v26` to `s57_v29`
are not reported, and its header still says 29); `submit_s57_residual_vq_ablation_multiseed.sh`
submits only `s57_v30` to `s57_v32`.

```bash
cd $SQUINT && source $VENV_PATH/bin/activate
bash examples/submit_s57_crossbatch_ablations_multiseed.sh 0,1,2,3,4
```

Every variant name below ends in `+mmb0-1b_smb1-1b_1p`:

| Table 3 column | Variant |
|---|---|
| Default (Adjacency w/, Contrastive Cross, Decoder cov. w/, Coupling FiLM, VQ (codes), Residual VQ (30, 90)) | `s57_v19_reference-filmscale` |
| Adjacency w/o | `s57_v1_filmscale+no-adj` |
| Contrastive Within / None | `s57_v20_contrast-within-batch+film` / `s57_v21_contrast-none+film` |
| Decoder cov. w/o | `s57_v2_filmscale+no-dec-cov` |
| Coupling Decpl / Trunk / X-st / Aff | `s57_v25_coupling-decoupled` / `s57_v22_coupling-coupled` / `s57_v23_coupling-crossstitch` / `s57_v24_coupling-affine` |
| Discretization VQ (Leiden) / VAE (Leiden) | `s57_v19` latent / `s57_v33_continuous-ref-filmscale` latent, both Leiden-clustered (below) |
| Residual VQ 2700 / 120 / 30 | `s57_v30_vq1level-k2700` / `s57_v31_vq1level-k120` / `s57_v32_vq1level-k30` |

| Table S1 block | Columns: variant |
|---|---|
| GNN depth | 1L: `s57_v19`; 2L: `s57_v3_filmscale+gnn-l2` |
| Neighbours | 16: `s57_v19`; 8: `s57_v4_filmscale+knn8`; 16/8: `s57_v5_filmscale+knn16-sampler8`; 24: `s57_v6_filmscale+knn24` |
| Cell cb. K(1) | (30,30): `s57_v8`; (10,30): `s57_v13`; (90,30): `s57_v14`; (300,30): `s57_v15` |
| Cell cb. K(2) | (30,90): `s57_v19`; (30,10): `s57_v7`; (30,30): `s57_v8`; (30,300): `s57_v9` |
| Niche cb. K(1) | (30,30): `s57_v11`; (10,30): `s57_v16`; (90,30): `s57_v17`; (300,30): `s57_v18` |
| Niche cb. K(2) | (30,90): `s57_v19`; (30,10): `s57_v10`; (30,30): `s57_v11`; (30,300): `s57_v12` |

The codebook variants are named `s57_v<N>_filmscale+rvq-{cell,niche}-<K(1)>-<K(2)>`.

The two Leiden columns of the Discretization block cluster the latent before quantization of the
default model and of `s57_v33` (the same model with a continuous bottleneck) to 30 clusters:

```bash
cd $REPO/analysis/discretization_ablation
M=$ART/mmb0-1b_smb1-1b_1p
bash submit_compare_discrete_vs_continuous.sh --out-dir $M/discretization \
    --discrete-runs "$M/s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p__multiseed" \
    --continuous-runs "$M/s57_v33_continuous-ref-filmscale+mmb0-1b_smb1-1b_1p__multiseed"
```

This writes `discretization_per_seed.csv` into `$M/discretization/`; Section 4 aggregates it.

### 3.5 Codebook specialisation (Table S2)

Level-1 codes against both annotations, the annotations against each other and the two codebooks
against each other, on the cells that carry both labels:

```bash
cd $REPO/analysis/coupling_diagnostics
python compute_codebook_overlap.py --artifacts-root $ART --dataset mmb0-1b_smb1-1b_1p \
    --variant "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p" --out $ART/codebook_overlap/mmb0-1b_smb1-1b_1p
python compute_codebook_overlap.py --artifacts-root $ART --dataset chl59-2b_1p \
    --variant "$H+chl59-2b_1p" --out $ART/codebook_overlap/chl59-2b_1p
python compute_codebook_overlap.py --artifacts-root $ART --dataset xhs1000-3b_1p \
    --variant "$H+xhs1000-3b_1p" --out $ART/codebook_overlap/xhs1000-3b_1p \
    --cell-label-keys new_annotation --niche-label-keys niche_type
```

Each writes `codebook_overlap_per_seed.csv` and `codebook_overlap_summary.csv`. The scVI
reference row is the niche NMI/ARI of the Table 1 scVI runs
(`$ART/<dataset>/baseline-scvi/<timestamp>/metrics/per_seed_niche_identification.csv`). The
paired t-tests on the gaps and the Welch tests against scVI have no packaged script.

### 3.6 Label-conditioned iLISI (Table S3)

iLISI computed within each annotated cell type or niche (k = 90, capped at the group size, cells
permuted; groups with fewer than 10 cells or a single section skipped), for SQUINT's quantized
and continuous embeddings and for every baseline embedding:

```bash
cd $REPO/analysis/benchmarking/niche_identification
VENV=$VENV_PATH METRICS=cilisi DATASETS="chl59-2b_1p xhs1000-3b_1p" bash submit_label_metrics.sh                   # cell-type groups
VENV=$VENV_PATH METRICS=cilisi DATASETS="chl59-2b_1p xhs1000-3b_1p" LABEL_KIND=niche bash submit_label_metrics.sh  # niche groups
python summarize_label_metrics.py --dir $ART/label_conditioned_metrics
python label_batch_feasibility.py --adata <a Mouse Brain predicted_adata.h5ad>   # no annotated group spans both sections
```

The launcher submits one job per metric, dataset and method: SQUINT (five seed runs, scored on
`cell_emb,neighborhood_emb,cell_latent,neighborhood_latent`) and every `baseline-*` folder (scored
on the `emb_key` of its `metrics/batch_integration_metrics.csv`). Without LSF, run each job
directly, e.g. SQUINT on CosMx NSCLC, conditioned on cell types:

```bash
cd $REPO
python analysis/benchmarking/niche_identification/compute_label_metric.py --metric cilisi \
    --method SQUINT --cell-type-key cell_type --batch-key adata_batch_id --drop-label-nan \
    --latent-keys cell_emb,neighborhood_emb,cell_latent,neighborhood_latent \
    --adata <seed0 run>/predicted_adata.h5ad ... --adata <seed4 run>/predicted_adata.h5ad \
    --out $REPO/artifacts/label_conditioned_metrics/lm_cilisi_chl59-2b_1p_SQUINT.csv
```

For niche groups, pass the niche column (`niche`; Xenium Eczema: `niche_type`, cell types
`new_annotation`) and add `_niche` to the file name; `--drop-label-nan` is used on CosMx NSCLC
only. Novae's stored outputs hold no embedding and are excluded. The Welch tests of Table S3 have
no packaged script.

### 3.7 Section-level query-to-reference mapping on CosMx NSCLC (Fig. 3, Fig. S6)

One SQUINT model on the eight CosMx NSCLC samples, with `Lung13` and `Lung5_Rep3` held out of
training (`adata_batch_id` 2 and 3) and given codes afterwards; one seed, all plots. Sections are
compared by the distributions of their codes; no labels are transferred to query cells (the label
file below is used only for the composition panels).

```bash
cd $SQUINT && source $VENV_PATH/bin/activate
bash examples/submit_single_run.sh "<variant>"   # a +chl59-8b_1p variant; without LSF: python examples/run_squint.py --all --variant "<variant>"

cd $REPO
RUN=$ART/chl59-8b_1p/<variant>/<run folder>
python analysis/benchmarking/plots/plot_code_distributions.py --adata $RUN/predicted_adata.h5ad \
    --query-sections Lung5_Rep3,Lung13 \
    --label-adata $DATA/bronze/chl59-8b_1p/nanostring_cosmx_human_nsclc_reference_query_mapping.h5ad   # Fig. 3, Fig. S6e,f
python $SQUINT/examples/plot_chl59_ground_truth.py --silver-dir $DATA/silver/chl59-8b_1p \
    --out-dir $ART/chl59-8b_1p/figures                                                                 # Fig. S6a,b
python $SQUINT/examples/plot_code_indices_spatial.py --predicted-adata $RUN/predicted_adata.h5ad    # Fig. S6c,d,g
```

The label file (cell types and niches of all eight samples, queries included) is not packaged.

### 3.8 Codebook usage and EMA decay (Fig. S4)

```bash
cd $SQUINT && source $VENV_PATH/bin/activate
VENV_PYTHON=$VENV_PATH/bin/python bash examples/submit_s65_emadecay_multiseed.sh 0,1,2,3,4   # β = 0.5, 0.7, 0.9, 0.95, 0.99 (0.8 is s57_v19)

cd $REPO/analysis/codebook_usage
python report_codebook_usage.py --artifacts-root $ART --variant "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p" \
    --all-seeds --codebook-sizes-cell 30,90 --codebook-sizes-niche 30,90            # Fig. S4a,b
python plot_ema_decay_sweep.py --artifacts-root $ART                                # Fig. S4c (ema_decay_sweep.csv)
```

## 4. Scoring, statistics and figures

Run these in the `squint` environment. Unless `--out-dir` is given, the benchmark plotting scripts
write to `$ART/benchmarking/figures/`.

**Table 1.** One per-seed CSV per dataset and block (method, seed, NMI, ARI, MMD, runtime in
seconds, iLISI), plus `_summary.csv`, `_wide.csv` and the panels:

```bash
cd $REPO/analysis/benchmarking/plots
N="python plot_niche_identification_benchmark.py --artifacts-root $ART"
C="python plot_cell_type_identification_benchmark.py --artifacts-root $ART"
V=s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p__multiseed
$N --dataset-tag mmb0-1b_smb1-1b_1p --squint-variant $V --out-name niche_identification_benchmark_mmb0-1b_smb1
$C --dataset-tag mmb0-1b_smb1-1b_1p --squint-variant $V --out-name cell_type_identification_benchmark_mmb0-1b_smb1
$N --dataset-tag chl59-2b_1p --squint-variant "$H+chl59-2b_1p__multiseed" --out-name niche_identification_benchmark_chl59
$C --dataset-tag chl59-2b_1p --squint-variant "$H+chl59-2b_1p__multiseed" --out-name cell_type_identification_benchmark_chl59
$N --dataset-tag xhs1000-3b_1p --squint-variant "$H+xhs1000-3b_1p__multiseed" --niche-label-key niche_type \
    --out-name niche_identification_benchmark_xhs1000
$C --dataset-tag xhs1000-3b_1p --squint-variant "$H+xhs1000-3b_1p__multiseed" --cell-label-key new_annotation \
    --out-name cell_type_identification_benchmark_xhs1000
```

**Corrected iLISI (Tables 1, 3, S1; Fig. S3b,d).** `recompute_ilisi.py` rescores, in the
`squint` environment, the stored embeddings behind the affected iLISI values: SQUINT's quantized
embeddings and the four baseline values above 1 that came from the fallback statistic, with
sanity recomputations of unaffected continuous embeddings (pynndescent graph, k = 90, cells
permuted, mean over three permutations, `scib-metrics`). NMI, ARI, MMD and runtime are kept as
stored. Run it after Sections 3.1 and 3.4:

```bash
cd $REPO
W=analysis/benchmarking/ilisi_recompute/submit_recompute_ilisi.sh
VENV=$VENV_PATH SELFTEST=1 bash $W            # synthetic checks of the estimator
VENV=$VENV_PATH ONLY=chl59.squint bash $W     # NSCLC control first: already random order, must reproduce 0.948 / 0.967
VENV=$VENV_PATH bash $W                       # every other row, then a dependent merge job
python analysis/benchmarking/ilisi_recompute/recompute_ilisi.py figures --artifacts-root $ART   # Fig. S3b,d
python analysis/benchmarking/compute_table1_significance.py \
    --tables-dir $ART/ilisi_recompute/merged/<timestamp>/t1_tables                               # Table 1 stars
```

Without LSF, call the script's subcommands directly (the launcher adds `--knn both`; three
permutations are the default):

```bash
cd $REPO
export JAX_PLATFORMS=cpu                             # as the launcher does
R="python analysis/benchmarking/ilisi_recompute/recompute_ilisi.py"
$R selftest
$R manifest --artifacts-root $ART                    # writes $ART/ilisi_recompute/manifest.csv (--discretization-csv if needed)
$R plan --artifacts-root $ART                        # lists the job ids
$R compute --artifacts-root $ART --knn both --job <job id>   # once per job id, or --job ALL
$R merge --artifacts-root $ART
```

The merge writes `$ART/ilisi_recompute/merged/<timestamp>/`: `summary.csv` (five-seed means and
Welch tests against the reference row, for tables `T1`, `T3` and `S1`), `per_seed.csv`,
`t1_tables/` (the Table 1 CSVs above with corrected iLISI) and `ilisi_diff_report.txt`, which
compares every value with the printed tables. The Discretization rows need
`discretization_per_seed.csv` (Section 3.4; pass `DISC_CSV=` if it is not found).
`compute_table1_significance.py` gives the Welch tests of NMI, ARI, MMD and iLISI against SQUINT.

**Table 2.**

```bash
cd $REPO/analysis/benchmarking/plots
for b in cell niche; do
  python plot_reconstruction_benchmark.py --artifacts-root $ART --metric panel --branch $b \
      --squint-path "$H+region-holdout+mmb0-1b_smb1-1b_1p__multiseed"
  python plot_imputation_benchmark.py --artifacts-root $ART --metric panel --branch $b \
      --squint-imputed-path stage2-ablation/decode-temp05-nonoise-mc1000-nichecell \
      --squint-imputed-label "SQUINT (MC)" \
      --extra-variant stage2-ablation/decode-temp05-nonoise-nichecell "SQUINT (No MC)" \
      --extra-variant baseline-knn-spatial+region-holdout "k-NN (spatial)"
done
python ../compute_table2_significance.py --tables-dir $ART/benchmarking/figures
```

This writes `{reconstruction,imputation}_benchmark_panel_{cell,niche}.csv` (one row per metric,
panel, split and seed) and the panels. Pass these flags: the defaults point at earlier runs (a
wider-decoder SQUINT model and another stage-2 architecture). `compute_table2_significance.py`
lists which panel code each Table 2 column is (e.g. ρ cell = `pe_cw_log`).

**Tables 3 and S1.**

```bash
cd $REPO
python analysis/ablations/run_ablation_report.py --artifacts-root $ART \
    --discretization-summary $ART/mmb0-1b_smb1-1b_1p/discretization/discretization_per_seed.csv
```

This writes one figure per axis and `ablations_all_metrics_{long,wide}.csv` (five-seed means and
significance against each axis default) to `$ART/mmb0-1b_smb1-1b_1p/ablations/`. The axes are
defined in `AXES` in `analysis/ablations/plots/plot_ablations.py`. The iLISI rows of Tables 3 and
S1 and their stars come from the corrected-iLISI `summary.csv` (tables `T3`, `S1`).

**Figures.**

- Fig. 1 and Fig. S1 were drawn by hand; Fig. 2a is the Allen Reference Atlas.
- Fig. 2b and Fig. S3a,c,e,f: `python $SQUINT/examples/plot_code_indices_spatial.py
  --predicted-adata <Mouse Brain s57_v19 seed run>/predicted_adata.h5ad` (one map per branch and
  level). `analysis/analyze_run.py --artifacts-root $ART --variant
  s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p --all-seeds --only code_index` runs it for every
  seed.
- Fig. S2: `python analysis/data_preparation/plot_dataset_overview.py --dataset-tag
  mmb0-1b_smb1-1b_1p --silver-dir $DATA/silver/mmb0-1b_smb1-1b_1p_coord_aligned --out-dir <dir>`.
- Fig. S3b,d: `recompute_ilisi.py figures` (above). Fig. S3g: `python
  analysis/benchmarking/plots/plot_runtime_benchmark.py --artifacts-root $ART --squint-variant
  s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p__multiseed` (the default points at an earlier
  model).
- Fig. S4: Section 3.8. Fig. 3 and Fig. S6: Section 3.7.
- Fig. S5: `python $SQUINT/examples/plot_svg_reconstruction.py --predicted-adata <Mouse Brain
  run>/predicted_adata.h5ad`; it shows the genes ranked highest by Moran's I (spatially variable
  genes).

The LaTeX tables were transcribed from these CSVs by hand; there is no table generator.

## 5. Expected results

Mean over five runs (seeds 0–4). MMD in 10⁻³, runtime (RT) in minutes. Bold: best per column
within each block, as printed in the paper (iLISI is not ranked).

**Table 1, Mouse Brain (STARmap + MERFISH).**

| Block | Method | NMI ↑ | ARI ↑ | MMD ↓ | iLISI ↑ | RT ↓ |
|---|---|---|---|---|---|---|
| Niche | SQUINT | **0.702** | **0.410** | **1.05** | 0.897 | **16** |
| | NicheCompass | 0.694 | 0.373 | 8.90 | 0.005 | 50 |
| | CellCharter | 0.643 | 0.326 | 28.23 | 0.017 | 21 |
| | BANKSY | 0.602 | 0.306 | 1.36 | 0.769 | 19 |
| | GraphST | 0.517 | 0.211 | 46.83 | 0.380 | 42 |
| | Novae | 0.506 | 0.185 | 4.89 | 0.801 | 41 |
| Cell type | SQUINT | 0.494 | 0.204 | **2.27** | 0.810 | **16** |
| | Nicheformer | **0.529** | **0.298** | 36.32 | 0.015 | 27 |
| | scVI | 0.501 | 0.237 | 17.67 | 0.289 | 21 |
| | UCE | 0.474 | 0.276 | 30.00 | 0.039 | 70 |
| | Geneformer | 0.383 | 0.196 | 43.50 | 0.099 | 19 |
| | scGPT | 0.243 | 0.103 | 19.17 | 0.368 | 25 |
| | scGPT-spatial | 0.222 | 0.117 | 22.77 | 0.393 | 32 |

**Table 1, CosMx NSCLC.**

| Block | Method | NMI ↑ | ARI ↑ | MMD ↓ | iLISI ↑ | RT ↓ |
|---|---|---|---|---|---|---|
| Niche | SQUINT | 0.397 | 0.172 | **0.52** | 0.948 | **33** |
| | NicheCompass | **0.423** | 0.246 | 2.02 | 0.418 | 210 |
| | CellCharter | 0.414 | **0.247** | 12.45 | 0.521 | 82 |
| | BANKSY | 0.228 | 0.114 | 0.78 | 0.762 | 202 |
| | GraphST | n/a | n/a | n/a | n/a | n/a |
| | Novae | 0.124 | 0.108 | 18.56 | 0.704 | 141 |
| Cell type | SQUINT | **0.604** | 0.339 | **0.61** | 0.967 | **33** |
| | Nicheformer | 0.407 | 0.222 | 7.64 | 0.791 | 92 |
| | scVI | 0.594 | **0.378** | 5.86 | 0.809 | 90 |
| | UCE | 0.336 | 0.312 | 6.21 | 0.737 | 211 |
| | Geneformer | 0.503 | 0.340 | 8.43 | 0.752 | 78 |
| | scGPT | 0.487 | 0.301 | 6.32 | 0.774 | 102 |
| | scGPT-spatial | 0.496 | 0.337 | 7.69 | 0.755 | 49 |

**Table 1, Xenium Eczema.**

| Block | Method | NMI ↑ | ARI ↑ | MMD ↓ | iLISI ↑ | RT ↓ |
|---|---|---|---|---|---|---|
| Niche | SQUINT | 0.514 | 0.207 | **3.76** | 0.542 | 25 |
| | NicheCompass | **0.580** | 0.278 | 8.27 | 0.042 | 54 |
| | CellCharter | 0.545 | 0.258 | 33.50 | 0.133 | **15** |
| | BANKSY | 0.480 | 0.217 | 6.01 | 0.305 | 16 |
| | GraphST | 0.372 | 0.184 | 17.08 | 0.371 | 23 |
| | Novae | 0.358 | **0.301** | 14.30 | 0.278 | 25 |
| Cell type | SQUINT | 0.593 | 0.322 | **8.93** | 0.702 | 25 |
| | Nicheformer | 0.393 | 0.158 | 18.48 | 0.541 | 18 |
| | scVI | **0.662** | **0.426** | 18.47 | 0.487 | 12 |
| | UCE | 0.599 | 0.376 | 11.23 | 0.451 | 43 |
| | Geneformer | 0.553 | 0.302 | 17.05 | 0.481 | 13 |
| | scGPT | 0.546 | 0.301 | 19.21 | 0.501 | 8 |
| | scGPT-spatial | 0.559 | 0.349 | 16.42 | 0.463 | **6** |

**Table 2, cell level.** Pearson ρ (cell-wise, gene-wise, 50 HVGs) and gene-wise Spearman ρs on
log1p counts, RMSE on raw counts, expressed-gene detection (AUROC, AP); held-out cells only.

| Block | Method | ρ cell ↑ | ρ gene ↑ | ρ HVG ↑ | ρs ↑ | RMSE ↓ | AUROC ↑ | AP ↑ |
|---|---|---|---|---|---|---|---|---|
| Reconstruction | SQUINT | 0.634 | 0.362 | 0.702 | 0.275 | 0.792 | 0.862 | 0.500 |
| | scVI | 0.648 | 0.373 | 0.744 | 0.267 | **0.612** | 0.867 | 0.506 |
| | NicheCompass | **0.688** | **0.418** | **0.773** | **0.284** | 0.692 | **0.897** | **0.572** |
| | Vanilla VQ-VAE | 0.494 | 0.268 | 0.563 | 0.221 | 0.874 | 0.829 | 0.370 |
| Imputation | SQUINT (MC) | **0.468** | **0.194** | **0.385** | **0.177** | 1.164 | **0.807** | **0.336** |
| | SQUINT (No MC) | 0.432 | 0.178 | 0.354 | 0.171 | 1.241 | 0.794 | 0.320 |
| | GeST | 0.385 | 0.138 | 0.290 | 0.125 | **1.077** | 0.773 | 0.263 |
| | k-NN (spatial) | 0.371 | 0.126 | 0.300 | 0.109 | 1.087 | 0.732 | 0.250 |

**Table 2, neighbourhood level** (1-hop means).

| Block | Method | ρ cell ↑ | ρ gene ↑ | ρ HVG ↑ | ρs ↑ | RMSE ↓ | AUROC ↑ | AP ↑ |
|---|---|---|---|---|---|---|---|---|
| Reconstruction | SQUINT | **0.896** | **0.615** | 0.896 | **0.576** | 0.303 | **0.850** | **0.886** |
| | scVI | 0.893 | 0.589 | **0.907** | 0.520 | **0.230** | 0.832 | 0.847 |
| | NicheCompass | 0.888 | 0.516 | 0.900 | 0.466 | 0.275 | 0.810 | 0.829 |
| | Vanilla VQ-VAE | 0.714 | 0.412 | 0.644 | 0.410 | 0.486 | 0.795 | 0.794 |
| Imputation | SQUINT (MC) | **0.804** | **0.495** | **0.715** | **0.497** | **0.533** | **0.837** | **0.854** |
| | SQUINT (No MC) | 0.761 | 0.477 | 0.684 | 0.491 | 0.604 | 0.830 | 0.848 |
| | GeST | 0.682 | 0.359 | 0.547 | 0.349 | 0.572 | 0.785 | 0.787 |
| | k-NN (spatial) | 0.674 | 0.337 | 0.592 | 0.310 | 0.578 | 0.736 | 0.746 |

**Table 3** (columns of the paper as rows; variants in Section 3.4).

| Table 3 column | Niche NMI ↑ | Niche ARI ↑ | Niche MMD ↓ | Niche iLISI ↑ | Cell NMI ↑ | Cell ARI ↑ | Cell MMD ↓ | Cell iLISI ↑ |
|---|---|---|---|---|---|---|---|---|
| Default | 0.702 | 0.410 | 1.05 | 0.897 | 0.494 | 0.204 | 2.27 | 0.810 |
| Adjacency w/o | 0.350 | 0.149 | 4.50 | 0.488 | 0.493 | 0.207 | 2.08 | 0.819 |
| Contrastive Within | 0.704 | 0.415 | 1.51 | 0.838 | 0.497 | 0.216 | 2.79 | 0.532 |
| Contrastive None | 0.702 | 0.413 | 1.04 | 0.878 | 0.424 | 0.165 | 2.21 | 0.742 |
| Decoder cov. w/o | 0.702 | 0.415 | 9.96 | 0.124 | 0.525 | 0.252 | 46.30 | 0.050 |
| Coupling Decpl | 0.705 | 0.419 | 0.98 | 0.936 | 0.493 | 0.202 | 2.21 | 0.806 |
| Coupling Trunk | 0.691 | 0.403 | 0.99 | 0.947 | 0.486 | 0.208 | 0.96 | 0.923 |
| Coupling X-st | 0.702 | 0.413 | 0.89 | 0.937 | 0.496 | 0.202 | 2.35 | 0.783 |
| Coupling Aff | 0.692 | 0.407 | 1.21 | 0.941 | 0.489 | 0.218 | 1.11 | 0.903 |
| Discretization VQ (Leiden) | 0.715 | 0.396 | 19.79 | 0.036 | 0.509 | 0.243 | 3.12 | 0.387 |
| Discretization VAE (Leiden) | 0.717 | 0.391 | 16.45 | 0.052 | 0.514 | 0.245 | 9.15 | 0.319 |
| Residual VQ 2700 | 0.704 | 0.300 | 1.06 | 0.573 | 0.387 | 0.006 | 2.54 | 0.559 |
| Residual VQ 120 | 0.682 | 0.339 | 1.13 | 0.829 | 0.445 | 0.080 | 1.79 | 0.909 |
| Residual VQ 30 | 0.707 | 0.422 | 1.07 | 0.942 | 0.479 | 0.199 | 1.79 | 0.966 |

Other outputs to compare against:

- **Table S1:** as printed in the supplement; `ablations_all_metrics_wide.csv` holds it with the
  default first.
- **Table S2:** matching minus non-matching codebook, NMI, positive in all five seeds: cell types
  +0.275 / +0.333 / +0.243, niches +0.319 / +0.106 / +0.055 (Mouse Brain / CosMx NSCLC / Xenium
  Eczema).
- **Table S3:** SQUINT CiLISI 0.961 (cell types) and 0.930 (niches) on CosMx NSCLC, 0.657 and
  0.503 on Xenium Eczema, the highest of all methods.
- **Fig. 3:** cell-code similarity (1 − Jensen–Shannon distance) of `Lung5_Rep3` to the other two
  Lung5 replicates 0.96 / 0.93; `Lung13` at most 0.52 to any reference sample.
- **CIFM:** held-out cell-wise Pearson 0.081 (radius graph) and 0.078 (k-NN graph), against 0.371
  for the k-NN floor; non-held-out cells 0.078.
- **Fig. S4c:** active-code fraction and normalised perplexity are flat for β from 0.5 to 0.95;
  only β = 0.99 starves the fine cell codebook.

Re-runs are not guaranteed to be bit-identical across hardware or library versions; compare at the
level of the seed-to-seed variation (standard deviations in the `_summary.csv` files and in
Tables S2 and S3).

## Repository layout

| Path | Contents |
|---|---|
| `analysis/benchmarking/submit_*.sh`, `_run_one_benchmark.sh` | Table 1 baseline launchers; method registry in `submit_all_benchmarks.sh` |
| `analysis/benchmarking/{cell_type,niche}_identification/` | Table 1 baseline runners; label-conditioned metrics (Table S3) |
| `analysis/benchmarking/gene_expr_imputation/` | Table 2 and CIFM runners, shared holdout and metrics (`_holdout_utils.py`), GeST reimplementation (`gest/`) |
| `analysis/benchmarking/ilisi_recompute/` | Corrected iLISI |
| `analysis/benchmarking/plots/` | Table 1 and 2 CSVs and panels, runtime panel, section-level query-to-reference code distributions |
| `analysis/benchmarking/compute_table{1,2}_significance.py` | Table 1 and 2 significance marks |
| `analysis/ablations/`, `analysis/discretization_ablation/` | Tables 3 and S1 (`run_ablation_report.py`, `plots/`, discretization comparison); `rank_stage2_ablations.py` (Section 3.2); `ablations_list.md` and `backbone_gnn.ipynb` belong to the earlier pipeline |
| `analysis/coupling_diagnostics/` | Table S2 (`compute_codebook_overlap.py`); `diagnose_coupling.py` is exploratory |
| `analysis/codebook_usage/` | Fig. S4 |
| `analysis/data_preparation/` | Xenium sub-selection, Ensembl IDs, Fig. S2; the other scripts concern datasets and checks not reported in the paper |
| `analysis/analyze_run.py`, `analysis/submit_analyze_run.sh` | Per-run plots through the squint plotting scripts |
| `analysis/predict_model.py` | `collate_predict_outputs`, imported by squint's inference |
| `envs/` | Optional RAPIDS environment |
| `config/`, `scripts/`, `test/`, `analysis/{train_model,test_model,create_in_memory_dataset_blob,banksy_train}.py`, `analysis/{notebooks,tasks,statistics,plots}/`, `analysis/benchmarking/recompute_niche_metrics.py` | Earlier config-driven single-codebook pipeline (its YAML configurations are in `config/`) and exploratory analyses; not used for the paper, whose variants are defined in squint's `VARIANTS` registry (Section 3) |

## Citation

```bibtex
@inproceedings{birk2026squint,
  title     = {Learning Discrete Cell and Niche Codes from Spatial Transcriptomics Using Dual Residual Vector Quantization},
  author    = {Birk, Sebastian and Merchant, Arpit and Vahidi, Amirhossein and Theis, Fabian J. and Lotfollahi, Mo},
  booktitle = {Proceedings of the Machine Learning in Computational Biology (MLCB) 2026},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026},
  note      = {Volume and pages to be assigned}
}
```

The single-codebook precursor (Merchant et al., ICLR 2026 MLGenX workshop) is cited in the
[squint README](https://github.com/Lotfollahi-lab/squint#citation).

## Licence

MIT (see `LICENSE`). The baselines are run from their authors' released packages and checkpoints
under their own terms; none of them is redistributed here (their model folders are git-ignored).
`analysis/benchmarking/gene_expr_imputation/gest/` is our reimplementation of GeST, whose code
is not public.
