"""
NicheCompass baseline for the SQUINT cell-identification benchmark.

Sibling of `run_graphst.py` / `run_novae.py` -- uses the NicheCompass
package (https://github.com/Lotfollahi-lab/nichecompass) to learn a
gene-program-based latent on a per-batch spatial graph, then clusters
with Leiden for niche identification.

Pipeline (matches `nichecompass_benchmarking.ipynb`):
  1. Load + concat silver h5ads.
  2. Per-batch spatial kNN graph -> block-diag concat.
  3. ONE-TIME (deterministic given the inputs):
       - Pull OmniPath / NicheNet / MEBOCOST gene programs.
       - `add_gps_from_gp_dict_to_adata` to attach the binary GP masks.
       - `layers['counts'] = adata.X` for the NB likelihood.
  4. For each seed:
       - Set torch / numpy seeds.
       - `model = NicheCompass(adata, ...)`
       - `model.train(n_epochs=..., lambda_*=...)`
         -> `obsm['nichecompass_latent']`
       - Leiden binary search to hit `--n-clusters` on the latent.
       - UMAP layout.
       - NMI/ARI vs cell_type / niche labels.
       - iLISI / MMD on `nichecompass_latent`.

Output layout mirrors `run_scvi.py` so compare_variants.py picks this
baseline up identically.

Approximate runtime: ~20-40 min per seed on a single GPU (training
dominates). Five seeds = ~3 h.

Notes:
- The GP dictionary download (OmniPath / NicheNet / MEBOCOST) can take
  a few minutes on first run; the notebook runs it inline. Pass
  --skip-gp to load a precomputed `combined_gp_dict.pkl` instead (see
  --gp-cache-path).
- `--mebocost-dir` must point to the local
  `metabolite_enzyme_sensor_gps` folder used by
  `extract_gp_dict_from_mebocost_ms_interactions`.
- `--gene-orthologs-csv` must point to the human↔mouse gene ortholog
  mapping CSV used by NicheCompass' OmniPath / NicheNet extractors.

Usage:
    python analysis/benchmarking/niche_identification/run_nichecompass.py \\
        --mebocost-dir /nfs/.../metabolite_enzyme_sensor_gps \\
        --gene-orthologs-csv /nfs/.../human_mouse_gene_orthologs.csv
    # smoke test:
    python analysis/benchmarking/niche_identification/run_nichecompass.py \\
        --mebocost-dir /nfs/.../metabolite_enzyme_sensor_gps \\
        --gene-orthologs-csv /nfs/.../human_mouse_gene_orthologs.csv \\
        --seeds 0
"""

import argparse
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings(
    "ignore",
    message=r".*Importing read_text from `anndata` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*legacy Dask DataFrame implementation is deprecated.*",
    category=FutureWarning,
)

import anndata as ad
import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# Reuse the shared helpers.
# This script lives in `analysis/benchmarking/niche_identification/`.
# The shared helpers (`_load_concat`, `_compute_*`, runtime tracking,
# Leiden bisect, …) live in `analysis/benchmarking/cell_type_identification/`
# (the sibling folder). Add both dirs to sys.path: this folder for the
# niche-method-specific helpers, the sibling for the shared code.
_THIS_DIR = Path(__file__).resolve().parent
_SIBLING_CELL_TYPE_DIR = _THIS_DIR.parent / "cell_type_identification"
for _p in (_THIS_DIR, _SIBLING_CELL_TYPE_DIR):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from run_pca_leiden import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_DATASET_TAG,
    DEFAULT_SILVER_DIR,
    DEFAULT_CELL_LABEL_KEYS,
    DEFAULT_NICHE_LABEL_KEYS,
    _aggregate_batch_int,
    _aggregate_niche,
    _compute_batch_integration,
    _compute_niche_identification,
    _import_metric_helpers,
    _load_concat,
    _plot_umap,
    _sanitize_for_h5ad,
    _record_seed_runtime,
    _write_per_seed_outputs,
    _write_runtime_csvs,
)
from run_neigh_expr_pca import _spatial_knn_per_batch  # noqa: E402
from run_scvi import _leiden_binary_search_on_latent  # noqa: E402


DEFAULT_VARIANT_TAG = "baseline-nichecompass"

# Notebook-level NicheCompass key constants (kept in one place so they
# can be tweaked from the CLI later if needed).
NC_COUNTS_KEY = "counts"
NC_ADJ_KEY = "spatial_connectivities"
NC_GP_NAMES_KEY = "nichecompass_gp_names"
NC_ACTIVE_GP_NAMES_KEY = "nichecompass_active_gp_names"
NC_GP_TARGETS_MASK_KEY = "nichecompass_gp_targets"
NC_GP_TARGETS_CATEGORIES_MASK_KEY = "nichecompass_gp_targets_categories"
NC_GP_SOURCES_MASK_KEY = "nichecompass_gp_sources"
NC_GP_SOURCES_CATEGORIES_MASK_KEY = "nichecompass_gp_sources_categories"
NC_LATENT_KEY = "nichecompass_latent"


# ---------------------------------------------------------------------------
# Gene program dictionary assembly (one-time, deterministic).
# ---------------------------------------------------------------------------

def _build_combined_gp_dict(
        species: str,
        gene_orthologs_csv: Path,
        mebocost_dir: Path,
        cache_path: Optional[Path],
        skip_gp: bool,
    ) -> Dict:
    """Pull OmniPath + NicheNet + MEBOCOST GPs and combine. If
    `cache_path` exists and `skip_gp` is True, load from disk instead;
    otherwise build from scratch and (when `cache_path` is given) save
    for re-use across runs.
    """
    from nichecompass.utils import (
        extract_gp_dict_from_mebocost_ms_interactions,
        extract_gp_dict_from_nichenet_lrt_interactions,
        extract_gp_dict_from_omnipath_lr_interactions,
        filter_and_combine_gp_dict_gps_v2,
    )

    if skip_gp and cache_path is not None and Path(cache_path).is_file():
        print(f"  loading combined GP dict from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            combined = pickle.load(f)
        print(f"  cached GP dict has {len(combined)} programs.")
        return combined

    print("  extract_gp_dict_from_omnipath_lr_interactions(...)")
    omnipath = extract_gp_dict_from_omnipath_lr_interactions(
        species=species,
        load_from_disk=False, save_to_disk=False,
        gene_orthologs_mapping_file_path=str(gene_orthologs_csv),
        plot_gp_gene_count_distributions=False,
    )
    print("  extract_gp_dict_from_nichenet_lrt_interactions(...)")
    nichenet = extract_gp_dict_from_nichenet_lrt_interactions(
        species=species, version="v2",
        keep_target_genes_ratio=1.0,
        max_n_target_genes_per_gp=250,
        load_from_disk=False, save_to_disk=False,
        gene_orthologs_mapping_file_path=str(gene_orthologs_csv),
        plot_gp_gene_count_distributions=False,
    )
    print("  extract_gp_dict_from_mebocost_ms_interactions(...)")
    mebocost = extract_gp_dict_from_mebocost_ms_interactions(
        dir_path=str(mebocost_dir),
        species=species,
        plot_gp_gene_count_distributions=False,
    )
    combined = filter_and_combine_gp_dict_gps_v2(
        [omnipath, nichenet, mebocost], verbose=True,
    )
    print(f"  combined GP dict has {len(combined)} programs.")
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(combined, f)
        print(f"  saved combined GP dict -> {cache_path}")
    return combined


# ---------------------------------------------------------------------------
# Per-seed NicheCompass training
# ---------------------------------------------------------------------------

def _train_nichecompass_and_get_latent(
        adata: ad.AnnData,
        cat_covariates_keys: List[str],
        cat_covariates_embeds_injection: List[str],
        cat_covariates_embeds_nums: List[int],
        cat_covariates_no_edges: List[bool],
        conv_layer_encoder: str,
        active_gp_thresh_ratio: float,
        n_epochs: int,
        n_epochs_all_gps: int,
        lr: float,
        lambda_edge_recon: float,
        lambda_gene_expr_recon: float,
        lambda_l1_masked: float,
        lambda_l1_addon: float,
        edge_batch_size: int,
        n_sampled_neighbors: int,
        use_cuda_if_available: bool,
        seed: int,
    ) -> np.ndarray:
    """Initialise + train one NicheCompass model for THIS seed.
    Returns `obsm[NC_LATENT_KEY]` as a NumPy array.
    """
    import random
    import torch
    from nichecompass.models import NicheCompass

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    print(f"  NicheCompass(...)  (seed={seed})")
    model = NicheCompass(
        adata,
        counts_key=NC_COUNTS_KEY,
        adj_key=NC_ADJ_KEY,
        cat_covariates_embeds_injection=cat_covariates_embeds_injection,
        cat_covariates_keys=cat_covariates_keys,
        cat_covariates_no_edges=cat_covariates_no_edges,
        cat_covariates_embeds_nums=cat_covariates_embeds_nums,
        gp_names_key=NC_GP_NAMES_KEY,
        active_gp_names_key=NC_ACTIVE_GP_NAMES_KEY,
        gp_targets_mask_key=NC_GP_TARGETS_MASK_KEY,
        gp_targets_categories_mask_key=NC_GP_TARGETS_CATEGORIES_MASK_KEY,
        gp_sources_mask_key=NC_GP_SOURCES_MASK_KEY,
        gp_sources_categories_mask_key=NC_GP_SOURCES_CATEGORIES_MASK_KEY,
        latent_key=NC_LATENT_KEY,
        conv_layer_encoder=conv_layer_encoder,
        active_gp_thresh_ratio=active_gp_thresh_ratio,
    )
    print(f"  model.train(n_epochs={n_epochs}, lr={lr}, ...)")
    model.train(
        n_epochs=n_epochs,
        n_epochs_all_gps=n_epochs_all_gps,
        lr=lr,
        lambda_edge_recon=lambda_edge_recon,
        lambda_gene_expr_recon=lambda_gene_expr_recon,
        lambda_l1_masked=lambda_l1_masked,
        edge_batch_size=edge_batch_size,
        n_sampled_neighbors=n_sampled_neighbors,
        use_cuda_if_available=use_cuda_if_available,
        verbose=False,
    )
    if NC_LATENT_KEY not in adata.obsm:
        raise RuntimeError(
            f"NicheCompass.train() did not populate obsm[{NC_LATENT_KEY!r}]."
        )
    latent = np.asarray(adata.obsm[NC_LATENT_KEY])
    print(f"  {NC_LATENT_KEY} shape: {latent.shape}")
    return latent


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--silver-dir", type=str, default=DEFAULT_SILVER_DIR)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--variant-tag", type=str, default=DEFAULT_VARIANT_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    # NicheCompass GP-dictionary inputs.
    p.add_argument("--species", type=str, default="mouse")
    p.add_argument("--gene-orthologs-csv", type=Path, required=True,
                   help="Path to human↔mouse gene ortholog CSV (used by "
                        "OmniPath / NicheNet extractors).")
    p.add_argument("--mebocost-dir", type=Path, required=True,
                   help="Path to the metabolite_enzyme_sensor_gps "
                        "folder used by extract_gp_dict_from_mebocost_ms_interactions.")
    p.add_argument("--gp-cache-path", type=Path, default=None,
                   help="Optional cache path for the combined GP "
                        "dictionary pickle. When set, the combined dict "
                        "is saved here on first run, and re-used on "
                        "later runs if --skip-gp is passed.")
    p.add_argument("--skip-gp", action="store_true",
                   help="Load the combined GP dict from --gp-cache-path "
                        "instead of pulling from OmniPath/NicheNet/MEBOCOST. "
                        "Cache path must exist.")
    # NicheCompass training hyperparams (notebook defaults).
    p.add_argument("--n-spatial-neighs", type=int, default=10)
    p.add_argument("--cat-covariates-key", type=str, default="adata_batch_id",
                   help="The notebook uses 'batch'. Mapped to the "
                        "user-supplied --batch-key by default.")
    p.add_argument("--conv-layer-encoder", type=str, default="gatv2conv")
    p.add_argument("--active-gp-thresh-ratio", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=400)
    p.add_argument("--n-epochs-all-gps", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lambda-edge-recon", type=float, default=500_000.0)
    p.add_argument("--lambda-gene-expr-recon", type=float, default=300.0)
    p.add_argument("--lambda-l1-masked", type=float, default=0.0)
    p.add_argument("--lambda-l1-addon", type=float, default=30.0)
    p.add_argument("--edge-batch-size", type=int, default=4096)
    p.add_argument("--n-sampled-neighbors", type=int, default=4)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--min-genes-per-gp", type=int, default=2)
    p.add_argument("--min-source-genes-per-gp", type=int, default=1)
    p.add_argument("--min-target-genes-per-gp", type=int, default=1)
    # Clustering / metric knobs.
    p.add_argument("--n-clusters", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--cell-label-keys", type=str,
                   default=",".join(DEFAULT_CELL_LABEL_KEYS))
    p.add_argument("--niche-label-keys", type=str,
                   default=",".join(DEFAULT_NICHE_LABEL_KEYS))
    p.add_argument("--batch-key", type=str, default="adata_batch_id")
    p.add_argument("--ilisi-n-neighbors", type=int, default=90)
    p.add_argument("--mmd-n-sub", type=int, default=2000)
    p.add_argument("--mmd-n-sigma", type=int, default=1000)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    cell_keys = [k.strip() for k in args.cell_label_keys.split(",") if k.strip()]
    niche_keys = [k.strip() for k in args.niche_label_keys.split(",") if k.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds parsed to empty list.")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = (
            args.artifacts_root / args.dataset_tag / args.variant_tag / ts
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {args.out_dir}")
    print(f"Seeds   : {seeds}")

    # 1. Load + spatial graph + GP dict + masks (one-time).
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from obs.")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
    # Raw counts on layers['counts'] for the NB likelihood.
    adata.layers[NC_COUNTS_KEY] = adata.X.copy()
    adata = _spatial_knn_per_batch(
        adata, n_neighs=args.n_spatial_neighs, batch_key=args.batch_key,
        include_self_loop=True,
    )

    print("\n=== Building combined GP dictionary ===")
    combined_gp = _build_combined_gp_dict(
        species=args.species,
        gene_orthologs_csv=args.gene_orthologs_csv,
        mebocost_dir=args.mebocost_dir,
        cache_path=args.gp_cache_path,
        skip_gp=args.skip_gp,
    )

    print("\n=== Attaching GP masks to AnnData ===")
    from nichecompass.utils import add_gps_from_gp_dict_to_adata
    add_gps_from_gp_dict_to_adata(
        gp_dict=combined_gp, adata=adata,
        gp_targets_mask_key=NC_GP_TARGETS_MASK_KEY,
        gp_targets_categories_mask_key=NC_GP_TARGETS_CATEGORIES_MASK_KEY,
        gp_sources_mask_key=NC_GP_SOURCES_MASK_KEY,
        gp_sources_categories_mask_key=NC_GP_SOURCES_CATEGORIES_MASK_KEY,
        gp_names_key=NC_GP_NAMES_KEY,
        min_genes_per_gp=args.min_genes_per_gp,
        min_source_genes_per_gp=args.min_source_genes_per_gp,
        min_target_genes_per_gp=args.min_target_genes_per_gp,
        max_genes_per_gp=None,
        max_source_genes_per_gp=None,
        max_target_genes_per_gp=None,
    )

    # 2. Per-seed loop: train NicheCompass + Leiden + metrics.
    compute_nmi_ari, compute_ilisi, compute_mmd_comparable = (
        _import_metric_helpers()
    )
    per_seed_niche: List[pd.DataFrame] = []
    per_seed_batch: List[pd.DataFrame] = []
    seed_summary: List[Dict] = []
    runtime_tracker: List[Dict] = []
    seed0_state: Optional[Dict] = None

    cat_covariates_keys = [args.batch_key]
    cat_covariates_embeds_injection = ["gene_expr_decoder"]
    cat_covariates_embeds_nums = [2]
    cat_covariates_no_edges = [True]

    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        seed_t0 = time.time()

        _train_nichecompass_and_get_latent(
            adata=adata,
            cat_covariates_keys=cat_covariates_keys,
            cat_covariates_embeds_injection=cat_covariates_embeds_injection,
            cat_covariates_embeds_nums=cat_covariates_embeds_nums,
            cat_covariates_no_edges=cat_covariates_no_edges,
            conv_layer_encoder=args.conv_layer_encoder,
            active_gp_thresh_ratio=args.active_gp_thresh_ratio,
            n_epochs=args.n_epochs,
            n_epochs_all_gps=args.n_epochs_all_gps,
            lr=args.lr,
            lambda_edge_recon=args.lambda_edge_recon,
            lambda_gene_expr_recon=args.lambda_gene_expr_recon,
            lambda_l1_masked=args.lambda_l1_masked,
            lambda_l1_addon=args.lambda_l1_addon,
            edge_batch_size=args.edge_batch_size,
            n_sampled_neighbors=args.n_sampled_neighbors,
            use_cuda_if_available=not args.no_cuda,
            seed=seed,
        )

        leiden_key, n_found, resolution = _leiden_binary_search_on_latent(
            adata, n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors,
            latent_key=NC_LATENT_KEY, seed=seed,
        )
        sc.tl.umap(adata, random_state=seed)

        print("\n  -- Niche identification --")
        niche_df = _compute_niche_identification(
            adata=adata, leiden_key=leiden_key,
            cell_label_keys=cell_keys, niche_label_keys=niche_keys,
            compute_nmi_ari=compute_nmi_ari,
        )
        if not niche_df.empty:
            niche_df.insert(0, "seed", seed)
        per_seed_niche.append(niche_df)

        print(f"\n  -- Batch integration on {NC_LATENT_KEY} --")
        adata.obsm["X_pca"] = adata.obsm[NC_LATENT_KEY]
        bint_df = _compute_batch_integration(
            adata=adata, batch_key=args.batch_key,
            compute_ilisi=compute_ilisi,
            compute_mmd_comparable=compute_mmd_comparable,
            ilisi_n_neighbors=args.ilisi_n_neighbors,
            mmd_n_sub=args.mmd_n_sub, mmd_n_sigma=args.mmd_n_sigma,
            seed=seed,
        )
        del adata.obsm["X_pca"]
        if not bint_df.empty:
            bint_df["emb_key"] = NC_LATENT_KEY
            bint_df.insert(0, "seed", seed)
        per_seed_batch.append(bint_df)

        seed_summary.append({
            "seed": seed,
            "leiden_n_clusters": int(n_found),
            "leiden_resolution": float(resolution),
        })

        seed_dir = _write_per_seed_outputs(
            seed=seed, run_dir=args.out_dir, adata=adata,
            leiden_key=leiden_key, niche_df=niche_df,
            batch_df=bint_df,
            cell_keys=cell_keys, niche_keys=niche_keys,
            batch_key=args.batch_key, dpi=args.dpi,
        )
        print(f"  -> wrote per-seed outputs to {seed_dir}")

        seed_seconds = time.time() - seed_t0
        _record_seed_runtime(
            runtime_tracker, seed=seed, seconds=seed_seconds,
            run_dir=args.out_dir, method="NicheCompass-Leiden",
        )
        print(f"  runtime (seed {seed}): {seed_seconds:.1f}s")

        if s_idx == 0:
            seed0_state = {"leiden_key": leiden_key,
                           "n_found": n_found, "resolution": resolution}

    # 3. Aggregate.
    long_niche = (
        pd.concat([df for df in per_seed_niche if not df.empty], ignore_index=True)
        if per_seed_niche else pd.DataFrame()
    )
    long_batch = (
        pd.concat([df for df in per_seed_batch if not df.empty], ignore_index=True)
        if per_seed_batch else pd.DataFrame()
    )

    if not long_niche.empty:
        out = metrics_dir / "per_seed_niche_identification.csv"
        long_niche.to_csv(out, index=False); print(f"\n  -> {out}")
        agg = _aggregate_niche([long_niche.drop(columns=["seed"])])
        out = metrics_dir / "niche_identification_metrics.csv"
        agg.to_csv(out, index=False)
        print(f"  -> {out}  (mean across {len(seeds)} seeds)")
    if not long_batch.empty:
        out = metrics_dir / "per_seed_batch_integration.csv"
        long_batch.to_csv(out, index=False); print(f"  -> {out}")
        agg = _aggregate_batch_int([long_batch.drop(columns=["seed"])])
        out = metrics_dir / "batch_integration_metrics.csv"
        agg.to_csv(out, index=False)
        print(f"  -> {out}  (mean across {len(seeds)} seeds)")

    _write_runtime_csvs(runtime_tracker, args.out_dir)

    # 4. Console summary.
    print()
    print("=" * 78)
    print(f"SUMMARY (mean ± std across {len(seeds)} seeds)")
    print("=" * 78)
    if not long_niche.empty:
        head = long_niche[long_niche["split"] == "all"]
        for label in ("cell_type", "niche"):
            sub = head[head["label_key"] == label]
            if sub.empty: continue
            for col in ("NMI", "ARI"):
                vals = sub[col].astype(float)
                std = vals.std(ddof=1) if len(vals) > 1 else 0.0
                print(f"  leiden vs {label:<10s} {col} = "
                      f"{vals.mean():.4f} ± {std:.4f}  "
                      f"(min={vals.min():.4f}, max={vals.max():.4f})")
    if not long_batch.empty:
        for metric in ("iLISI", "MMD"):
            sub = long_batch[long_batch["metric"] == metric]
            if sub.empty: continue
            vals = sub["score"].astype(float)
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(f"  {NC_LATENT_KEY} {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 5. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["nichecompass_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["nichecompass_resolution"] = float(seed0_state["resolution"])
        adata.uns["nichecompass_seeds"] = seeds
        _sanitize_for_h5ad(adata).write_h5ad(args.out_dir / "predicted_adata.h5ad")
        print(f"\n  -> {args.out_dir / 'predicted_adata.h5ad'}  (seed[0])")

        umap_dir = args.out_dir / "umap_plots"
        umap_dir.mkdir(parents=True, exist_ok=True)
        plot_keys = [(seed0_state["leiden_key"], "tab20")]
        plot_keys += [(k, "tab20") for k in cell_keys if k in adata.obs.columns]
        plot_keys += [(k, "tab10") for k in niche_keys if k in adata.obs.columns]
        plot_keys += [(args.batch_key, "Set2")]
        print("\n=== UMAP plots (seed[0]) ===")
        for key, cmap in plot_keys:
            out_path = umap_dir / key.replace("/", "_")
            _plot_umap(adata, color_key=key, out_path=out_path,
                       cmap_name=cmap, dpi=args.dpi)
            print(f"  -> {out_path}.{{png,svg}}")

    # 6. Stub config.
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "NicheCompass + Leiden baseline "
                                      "(multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag": args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "NicheCompass-Leiden"},
        "nichecompass": {
            "species": args.species,
            "n_spatial_neighs": int(args.n_spatial_neighs),
            "conv_layer_encoder": args.conv_layer_encoder,
            "active_gp_thresh_ratio": float(args.active_gp_thresh_ratio),
            "n_epochs": int(args.n_epochs),
            "n_epochs_all_gps": int(args.n_epochs_all_gps),
            "lr": float(args.lr),
            "lambda_edge_recon": float(args.lambda_edge_recon),
            "lambda_gene_expr_recon": float(args.lambda_gene_expr_recon),
            "lambda_l1_masked": float(args.lambda_l1_masked),
            "lambda_l1_addon": float(args.lambda_l1_addon),
            "edge_batch_size": int(args.edge_batch_size),
            "n_sampled_neighbors": int(args.n_sampled_neighbors),
            "n_clusters_target": int(args.n_clusters),
            "seeds": seeds,
            "seed_summary": seed_summary,
        },
    }
    with open(args.out_dir / "user_specified_config.yaml", "w") as f:
        yaml.safe_dump(stub_cfg, f, sort_keys=False)
    print(f"\n  -> {args.out_dir / 'user_specified_config.yaml'}")

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  Run dir : {args.out_dir}")
    print(f"  Variant : {args.variant_tag}")
    print(f"  Seeds   : {len(seeds)}  ({seeds})")
    print("=" * 78)


if __name__ == "__main__":
    main()
