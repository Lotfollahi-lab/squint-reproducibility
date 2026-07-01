"""
NicheCompass baseline for the gene-expression imputation benchmark
with the SQUINT region-holdout split.

NicheCompass is a graph autoencoder with two NB decoders:
  - cell-level (target gene NB head): predicts each cell's own
    expression at its node — this is the X_hat for cell-level Pearson.
  - nbr-level (source gene NB head): predicts the source-gene
    contribution aggregated from the spatial neighborhood — this is
    the X_hat_nbr for nbr-level Pearson.

Trained on the non-held-out cells (`data_split == "train"`); the
held-out cells are dropped from the train graph so they don't leak
into either head's loss. At inference, we run the trained model on
the FULL adata and pull both decoder outputs.

This script REUSES the GP-dictionary setup + training helpers from
`niche_identification/run_nichecompass.py` so any improvements to GP
extraction propagate automatically. It only adds:
  1. The held-out-region split (`data_split` column).
  2. The recon-extraction step (`get_omics_decoder_outputs`).
  3. The Pearson computation pipeline.

Output:
  <out_dir>/predicted_adata.h5ad             (X_hat + X_hat_nbr layers + data_split)
  <out_dir>/metrics/per_seed_pearson_reconstruction.csv
  <out_dir>/metrics/pearson_reconstruction_metrics.csv
  <out_dir>/user_specified_config.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from _holdout_utils import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_DATASET_TAG,
    DEFAULT_HOLDOUT_REGIONS,
    DEFAULT_SILVER_DIR,
    apply_holdout_regions,
    build_pearson_dataframe,
    compute_X_nbr,
    load_silver_concat,
    spatial_knn_per_batch,
    write_pearson_outputs, write_predicted_adata,
)

# Re-use NicheCompass setup from the existing benchmarking script.
_NICHE_DIR = (Path(__file__).resolve().parent.parent
              / "niche_identification")
if str(_NICHE_DIR) not in sys.path:
    sys.path.insert(0, str(_NICHE_DIR))
from run_nichecompass import (  # noqa: E402
    NC_COUNTS_KEY,
    NC_ADJ_KEY,
    NC_LATENT_KEY,
    NC_GP_NAMES_KEY,
    NC_ACTIVE_GP_NAMES_KEY,
    NC_GP_TARGETS_MASK_KEY,
    NC_GP_TARGETS_CATEGORIES_MASK_KEY,
    NC_GP_SOURCES_MASK_KEY,
    NC_GP_SOURCES_CATEGORIES_MASK_KEY,
    _build_combined_gp_dict,
)


DEFAULT_VARIANT_TAG = "baseline-nichecompass+region-holdout"


# ---------------------------------------------------------------------------
# Train + extract reconstructions
# ---------------------------------------------------------------------------

def _build_and_train_nichecompass(
        adata: ad.AnnData,
        seed: int,
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
    ):
    """Construct + train ONE NicheCompass model. Returns the trained
    model object so the caller can call `get_omics_decoder_outputs(...)`
    for reconstruction extraction."""
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
    t0 = time.time()
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
    print(f"  training took {time.time() - t0:.1f}s")
    return model


def _attach_recon_layers(
        adata_full: ad.AnnData,
        model,
        node_batch_size: int,
    ) -> ad.AnnData:
    """Run the trained NicheCompass model on `adata_full` (train + held-out)
    and attach `X_hat` (cell-level recon) + `X_hat_nbr` (nbr-level recon)
    as layers. Mutates `adata_full` in place; returns it.

    Assumes NicheCompass's omics decoder outputs:
      - `target_rna_nb_means` -> cell-level NB mean (cell branch)
      - `source_rna_nb_means` -> nbr-level NB mean (niche branch)

    Both outputs are gene-subset-restricted (only genes that appear as
    target / source in any active GP). Non-predicted genes are filled
    with the per-cell library × empirical-mean to avoid trivially-zero
    Pearson rows; the gene_subset is recorded in the saved adata.uns
    for downstream filtering.
    """
    print("  pulling omics decoder outputs (target + source)...")
    out = model.get_omics_decoder_outputs(
        adata=adata_full, node_batch_size=int(node_batch_size),
    )

    n_genes = adata_full.n_vars
    n_obs   = adata_full.n_obs

    def _resolve_recon(key: str, gp_mask_key: str) -> np.ndarray:
        """Convert the (n_cells, n_subset_genes) decoder output to a
        full (n_cells, n_genes) array by scattering into the columns
        flagged by `adata.varm[gp_mask_key]`. Non-predicted columns are
        filled with the per-gene cohort mean of `adata.X` so they
        contribute a constant baseline (not zero), keeping the
        cell-wise Pearson well-defined for those genes too.

        If the decoder output already has n_genes columns (e.g. an
        all-genes GP setup), the scatter step is skipped.
        """
        pred = np.asarray(out[key], dtype=np.float32)
        if pred.shape == (n_obs, n_genes):
            return pred
        if pred.ndim != 2 or pred.shape[0] != n_obs:
            raise RuntimeError(
                f"NicheCompass {key} has unexpected shape {pred.shape}; "
                f"expected (n_obs={n_obs}, n_genes_or_subset)."
            )
        # Subset → full-genes scatter.
        if gp_mask_key not in adata_full.varm:
            raise RuntimeError(
                f"varm[{gp_mask_key!r}] missing — cannot align "
                f"{key} ({pred.shape[1]} subset genes) back to "
                f"{n_genes} full genes."
            )
        gene_mask_2d = np.asarray(adata_full.varm[gp_mask_key])
        # `varm` masks may be (n_genes, n_gps); collapse along GPs to
        # find genes that appear in ANY GP.
        if gene_mask_2d.ndim == 2:
            gene_in_any_gp = gene_mask_2d.any(axis=1)
        else:
            gene_in_any_gp = gene_mask_2d.astype(bool)
        cols = np.where(gene_in_any_gp)[0]
        if cols.size != pred.shape[1]:
            raise RuntimeError(
                f"{gp_mask_key} flags {cols.size} genes but "
                f"{key} returned {pred.shape[1]} columns — these should "
                "match. Inspect NicheCompass GP masks."
            )
        full = np.zeros((n_obs, n_genes), dtype=np.float32)
        # Non-predicted columns: fill with empirical per-gene mean of X
        # so cell-wise Pearson over the FULL gene vector isn't dominated
        # by zeros. (Predicted columns get the model's NB means.)
        X_dense = adata_full.X
        if hasattr(X_dense, "toarray"):
            X_dense = X_dense.toarray()
        empirical_mean = np.asarray(X_dense, dtype=np.float32).mean(axis=0)
        full[:] = empirical_mean[None, :]
        full[:, cols] = pred
        return full

    X_hat     = _resolve_recon("target_rna_nb_means", NC_GP_TARGETS_MASK_KEY)
    X_hat_nbr = _resolve_recon("source_rna_nb_means", NC_GP_SOURCES_MASK_KEY)

    adata_full.layers["X_hat"]     = X_hat
    adata_full.layers["X_hat_nbr"] = X_hat_nbr
    print(f"  X_hat     shape={X_hat.shape}      "
          f"(min={X_hat.min():.3f}, mean={X_hat.mean():.3f})")
    print(f"  X_hat_nbr shape={X_hat_nbr.shape}  "
          f"(min={X_hat_nbr.min():.3f}, mean={X_hat_nbr.mean():.3f})")
    return adata_full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    p.add_argument("--batch-key", type=str, default="adata_batch_id")
    p.add_argument("--use-default-holdout-regions", action="store_true", default=True)
    p.add_argument("--no-default-holdout-regions",
                   dest="use_default_holdout_regions", action="store_false")
    # --- NicheCompass GP-dict inputs (mirror sibling script) ----------
    p.add_argument("--species", type=str, default="mouse")
    p.add_argument("--gene-orthologs-csv", type=Path, required=True,
                   help="Path to human↔mouse gene ortholog CSV (used by "
                        "OmniPath / NicheNet extractors).")
    p.add_argument("--mebocost-dir", type=Path, required=True,
                   help="Path to the metabolite_enzyme_sensor_gps folder.")
    p.add_argument("--gp-cache-path", type=Path, default=None,
                   help="Optional cache path for the combined GP dict pickle.")
    p.add_argument("--skip-gp", action="store_true",
                   help="Load combined GP dict from --gp-cache-path "
                        "instead of pulling from OmniPath/NicheNet/MEBOCOST.")
    p.add_argument("--min-genes-per-gp", type=int, default=2)
    p.add_argument("--min-source-genes-per-gp", type=int, default=1)
    p.add_argument("--min-target-genes-per-gp", type=int, default=1)
    # --- NicheCompass training hparams (notebook defaults) ----------
    p.add_argument("--n-spatial-neighs", type=int, default=16,
                   help="Spatial kNN graph size (model graph + X_nbr target). "
                        "Default 16 (matches SQUINT's native niche graph; was 10).")
    p.add_argument("--conv-layer-encoder", type=str, default="gatv2conv")
    p.add_argument("--active-gp-thresh-ratio", type=float, default=0.01)
    p.add_argument("--n-epochs", type=int, default=400)
    p.add_argument("--n-epochs-all-gps", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lambda-edge-recon", type=float, default=500000.0)
    p.add_argument("--lambda-gene-expr-recon", type=float, default=300.0)
    p.add_argument("--lambda-l1-masked", type=float, default=0.0)
    p.add_argument("--lambda-l1-addon", type=float, default=30.0)
    p.add_argument("--edge-batch-size", type=int, default=256)
    p.add_argument("--n-sampled-neighbors", type=int, default=4)
    p.add_argument("--node-batch-size", type=int, default=64,
                   help="Batch size for the inference call.")
    p.add_argument("--no-cuda", action="store_true")
    # --- Repro ------------------------------------------------------
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds parsed to empty list.")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = (
            args.artifacts_root / args.dataset_tag / args.variant_tag / ts
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {args.out_dir}")
    print(f"Seeds   : {seeds}")

    # ---- 1. Load silver -------------------------------------------------
    print("\n=== Loading silver ===")
    adata = load_silver_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"Concatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from obs.")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
    # Raw counts on layers['counts'] for the NB likelihood.
    adata.layers[NC_COUNTS_KEY] = adata.X.copy()

    # ---- 2. Held-out region split (mark cells; we'll subset later) -----
    print("\n=== Held-out region split ===")
    regions = DEFAULT_HOLDOUT_REGIONS if args.use_default_holdout_regions else None
    if regions is None:
        raise SystemExit("Custom regions not configured; pass --use-default-holdout-regions.")
    apply_holdout_regions(adata, batch_key=args.batch_key, regions=regions)

    # ---- 3. Spatial graph + X_nbr (computed on FULL adata so the
    #         nbr-level target is defined for both train and test cells).
    print("\n=== Spatial graph + X_nbr target ===")
    spatial_knn_per_batch(adata, n_neighs=args.n_spatial_neighs,
                          batch_key=args.batch_key)
    compute_X_nbr(adata, normalize="mean")

    # ---- 4. Build GP dictionary (one-time) -----------------------------
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

    # ---- 5. Per-seed training + recon + Pearson ------------------------
    cat_covariates_keys = [args.batch_key]
    cat_covariates_embeds_injection = ["gene_expr_decoder"]
    cat_covariates_embeds_nums = [2]
    cat_covariates_no_edges = [True]

    per_seed_frames: List[pd.DataFrame] = []
    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)

        # Subset adata to TRAIN cells; rebuild the spatial graph on this
        # subset so the held-out cells don't contribute to either head's
        # loss (their neighbours-of-neighbours leak is unavoidable but
        # their direct loss is gone).
        train_mask = (adata.obs["data_split"].to_numpy() == "train")
        train_adata = adata[train_mask].copy()
        # Re-build spatial graph on train_adata only — the existing one
        # was per the FULL adata; subsetting drops half the rows and
        # leaves the original graph mismatched.
        spatial_knn_per_batch(train_adata, n_neighs=args.n_spatial_neighs,
                              batch_key=args.batch_key)

        model = _build_and_train_nichecompass(
            adata=train_adata,
            seed=seed,
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
        )

        # Inference on the FULL adata. Pull both decoder outputs,
        # scatter back to full-gene shape, and put as layers.
        adata_s = adata.copy()
        _attach_recon_layers(
            adata_s, model, node_batch_size=args.node_batch_size,
        )

        df = build_pearson_dataframe(adata_s, seed=seed, log1p=True, n_hvg=50)
        per_seed_frames.append(df)
        write_predicted_adata(adata_s, args.out_dir, seed)   # per-seed h5ad -> rescore-able

        for branch in ("cell", "niche"):
            for split in ("all", "train", "test"):
                row = df[
                    (df["split"]       == split) &
                    (df["branch"]      == branch) &
                    (df["axis"]        == "cell_wise") &
                    (df["transform"]   == "log1p") &
                    (df["gene_subset"] == "all")
                ]
                if not row.empty:
                    v = float(row["pearson_mean"].iloc[0])
                    print(f"  Pearson {branch:<5s} cell_wise log1p all "
                          f"(split={split:<5s}) = {v:.4f}")

    per_seed = (
        pd.concat(per_seed_frames, ignore_index=True)
        if per_seed_frames else pd.DataFrame()
    )

    # ---- 6. Write outputs ----------------------------------------------
    print("\n=== Writing outputs ===")
    write_pearson_outputs(args.out_dir, per_seed)

    # ---- 7. Console summary --------------------------------------------
    print()
    print("=" * 78)
    print(f"SUMMARY (mean ± std across {len(seeds)} seeds, "
          "cell_wise × log1p × all)")
    print("=" * 78)
    canonical = per_seed[
        (per_seed["axis"]        == "cell_wise") &
        (per_seed["transform"]   == "log1p") &
        (per_seed["gene_subset"] == "all")
    ]
    for branch in ("cell", "niche"):
        for split in ("all", "train", "test"):
            sub = canonical[
                (canonical["branch"] == branch) &
                (canonical["split"]  == split)
            ]
            if sub.empty: continue
            vals = sub["pearson_mean"].astype(float)
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            print(f"  {branch:<5s} Pearson (split={split:<5s}) = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")
    print(f"  (full {len(per_seed)}-row variant table written to "
          f"per_seed_pearson_reconstruction.csv)")

    # ---- 8. Stub config -------------------------------------------------
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "NicheCompass baseline (region-holdout)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "NicheCompass"},
        "nichecompass": {
            "species":              args.species,
            "n_spatial_neighs":     args.n_spatial_neighs,
            "conv_layer_encoder":   args.conv_layer_encoder,
            "n_epochs":             args.n_epochs,
            "n_epochs_all_gps":     args.n_epochs_all_gps,
            "lr":                   args.lr,
            "lambda_edge_recon":    args.lambda_edge_recon,
            "lambda_gene_expr_recon": args.lambda_gene_expr_recon,
            "lambda_l1_masked":     args.lambda_l1_masked,
            "lambda_l1_addon":      args.lambda_l1_addon,
            "edge_batch_size":      args.edge_batch_size,
            "n_sampled_neighbors":  args.n_sampled_neighbors,
            "active_gp_thresh_ratio": args.active_gp_thresh_ratio,
            "min_genes_per_gp":     args.min_genes_per_gp,
            "min_source_genes_per_gp": args.min_source_genes_per_gp,
            "min_target_genes_per_gp": args.min_target_genes_per_gp,
            "seeds":                seeds,
            "batch_key":            args.batch_key,
            "holdout_regions":      DEFAULT_HOLDOUT_REGIONS,
        },
    }
    with open(args.out_dir / "user_specified_config.yaml", "w") as f:
        yaml.safe_dump(stub_cfg, f, sort_keys=False)
    print(f"\n  -> {args.out_dir / 'user_specified_config.yaml'}")

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  Run dir : {args.out_dir}")
    print(f"  Variant : {args.variant_tag}")
    print(f"  Seeds   : {len(seeds)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
