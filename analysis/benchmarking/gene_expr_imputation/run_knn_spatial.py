"""
Spatial kNN-regression baseline for the gene-expression IMPUTATION benchmark
(SQUINT region-holdout split).

This is the canonical NON-PARAMETRIC spatial in-painting floor: a held-out
cell's expression is predicted as the (distance-weighted) mean of its k nearest
OBSERVED (train) cells' raw counts, within the same section. Held-out cells'
own expression is never seen; train cells are predicted leave-one-out (self
excluded). It's the natural simple comparator for GeST / SQUINT-stage-2 on this
task (GeST itself benchmarks against a GP and an MLP over spatial coordinates;
Hao et al., MLCB 2025) — if the fancy machinery's RMSE edge is really just
spatial smoothing, this baseline lands right next to it.

Deterministic (given k + weighting + coords), so it is single-seed by default;
pass --seeds to replicate the (identical) rows for schema parity.

Mirrors run_scvi.py exactly except for the model: the shared `_holdout_utils`
handles silver loading, the held-out split, the neighborhood layers, the full
metric panel (Pearson / Spearman / RMSE / AUROC), and the CSVs.

Output (per the harness):
  <out_dir>/predicted_adata_seed{N}.h5ad
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
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from _holdout_utils import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_DATASET_TAG,
    DEFAULT_HOLDOUT_REGIONS,
    DEFAULT_SILVER_DIR,
    add_neighborhood_layers,
    apply_holdout_regions,
    build_pearson_dataframe,
    load_silver_concat,
    write_pearson_outputs, write_predicted_adata,
)


DEFAULT_VARIANT_TAG = "baseline-knn-spatial+region-holdout"


# ---------------------------------------------------------------------------
# Spatial kNN over OBSERVED (train) cells
# ---------------------------------------------------------------------------

def _knn_train_neighbors(
        coords: np.ndarray,
        batch: np.ndarray,
        is_train: np.ndarray,
        k: int,
        exclude_self: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """For every cell, the global indices + distances of its `k` nearest cells
    that are in the TRAIN split, restricted to the SAME section (no cross-batch
    neighbors) and (optionally) excluding the query cell itself.

    Returns (nbr_idx, nbr_dist) each shape (n_obs, k). Missing slots (section
    with < k train neighbors, or an isolated cell) are filled with -1 / inf.
    """
    from scipy.spatial import cKDTree

    n = coords.shape[0]
    nbr_idx = np.full((n, k), -1, dtype=np.int64)
    nbr_dist = np.full((n, k), np.inf, dtype=np.float64)
    for b in pd.unique(batch):
        sec = np.where(batch == b)[0]                     # global idx in section
        train_glob = sec[is_train[sec]]                   # global idx of train cells here
        if train_glob.size == 0:
            print(f"  [knn] section {b!r}: no train cells — skipped "
                  f"({sec.size} cells left unpredicted)")
            continue
        tree = cKDTree(coords[train_glob])
        # Query k+1 so a train query cell can drop its own (distance-0) match.
        kq = min(k + 1, train_glob.size)
        dist, loc = tree.query(coords[sec], k=kq)
        if kq == 1:                                       # cKDTree squeezes k==1
            dist = dist[:, None]
            loc = loc[:, None]
        glob = train_glob[loc]                            # (n_sec, kq) global idx
        if exclude_self:
            dist = dist.copy()
            dist[glob == sec[:, None]] = np.inf           # push self to the end
        order = np.argsort(dist, axis=1)[:, :k]           # k smallest per row
        rows = np.arange(sec.size)[:, None]
        sel_glob = glob[rows, order]
        sel_dist = dist[rows, order]
        valid = np.isfinite(sel_dist)
        sel_glob = np.where(valid, sel_glob, -1)
        # pad to width k if the section had < k train neighbors
        if sel_glob.shape[1] < k:
            padw = k - sel_glob.shape[1]
            sel_glob = np.pad(sel_glob, ((0, 0), (0, padw)), constant_values=-1)
            sel_dist = np.pad(sel_dist, ((0, 0), (0, padw)), constant_values=np.inf)
        nbr_idx[sec] = sel_glob
        nbr_dist[sec] = sel_dist
    return nbr_idx, nbr_dist


def _knn_predict(
        X: np.ndarray,
        nbr_idx: np.ndarray,
        nbr_dist: np.ndarray,
        weighting: str,
        eps: float = 1e-8,
    ) -> np.ndarray:
    """X_hat[i] = weighted mean of X over cell i's valid neighbors. Weighting:
    'uniform' | 'distance' (1/d) | 'gaussian' (RBF, bw = median neighbor dist).
    Cells with no valid neighbor get all-zeros (rare; isolated section)."""
    X = X.astype(np.float32)
    n, g = X.shape
    k = nbr_idx.shape[1]
    valid = nbr_idx >= 0                                  # (n, k)
    if weighting == "uniform":
        w = valid.astype(np.float64)
    elif weighting == "distance":
        w = np.where(valid, 1.0 / (nbr_dist + eps), 0.0)
    elif weighting == "gaussian":
        d = nbr_dist[valid & np.isfinite(nbr_dist)]
        bw = float(np.median(d)) if d.size else 1.0
        w = np.where(valid, np.exp(-(nbr_dist ** 2) / (2.0 * bw ** 2 + eps)), 0.0)
    else:
        raise ValueError(f"unknown weighting={weighting!r}")
    wsum = w.sum(axis=1, keepdims=True)
    wsum = np.where(wsum > 0, wsum, 1.0)
    w = (w / wsum).astype(np.float32)
    idx_safe = np.where(valid, nbr_idx, 0)                # -1 -> 0 (weight is 0)
    X_hat = np.zeros((n, g), dtype=np.float32)
    for kk in range(k):                                   # k small (~16); vectorised adds
        X_hat += w[:, kk:kk + 1] * X[idx_safe[:, kk]]
    return X_hat


def predict_knn(
        adata_full: ad.AnnData,
        k: int,
        weighting: str,
        batch_key: str,
        spatial_key: str = "spatial",
    ) -> ad.AnnData:
    """Fill `adata_full.layers['X_hat']` with the spatial-kNN prediction
    (weighted mean of each cell's k nearest TRAIN neighbors' raw counts).
    Returns the same adata."""
    if spatial_key not in adata_full.obsm:
        raise SystemExit(f"obsm[{spatial_key!r}] missing.")
    coords = np.asarray(adata_full.obsm[spatial_key], dtype=np.float64)
    batch = adata_full.obs[batch_key].to_numpy()
    is_train = (adata_full.obs["data_split"].to_numpy() == "train")
    n_train = int(is_train.sum())
    print(f"  train cells: {n_train}   held-out cells: {adata_full.n_obs - n_train}")
    print(f"  spatial kNN (k={k}, weighting={weighting}) over train neighbors "
          f"per section...")
    t0 = time.time()
    nbr_idx, nbr_dist = _knn_train_neighbors(coords, batch, is_train, k)
    X = adata_full.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_hat = _knn_predict(np.asarray(X, dtype=np.float32), nbr_idx, nbr_dist, weighting)
    if X_hat.shape != (adata_full.n_obs, adata_full.n_vars):
        raise RuntimeError(f"X_hat shape {X_hat.shape} != "
                           f"({adata_full.n_obs}, {adata_full.n_vars})")
    adata_full.layers["X_hat"] = X_hat
    print(f"  X_hat populated (shape={X_hat.shape}, min={X_hat.min():.3f}, "
          f"mean={X_hat.mean():.3f}, max={X_hat.max():.3f}) in {time.time() - t0:.1f}s")
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
    p.add_argument("--nbr-neighs", type=int, default=16,
                   help="Spatial kNN neighbors for the niche-branch aggregation "
                        "(X_nbr / X_hat_nbr), matching SQUINT's native niche graph.")
    p.add_argument("--k", type=int, default=16,
                   help="Number of nearest OBSERVED (train) neighbors used for the "
                        "cell-level prediction. Default 16 (matches the niche graph "
                        "degree so the two levels use a comparable neighborhood).")
    p.add_argument("--weighting", type=str, default="distance",
                   choices=["uniform", "distance", "gaussian"],
                   help="Neighbor weighting for the mean. 'distance' = 1/d (default), "
                        "'gaussian' = RBF, 'uniform' = plain mean.")
    p.add_argument("--use-default-holdout-regions", action="store_true", default=True)
    p.add_argument("--no-default-holdout-regions",
                   dest="use_default_holdout_regions", action="store_false")
    p.add_argument("--seeds", type=str, default="0",
                   help="Comma-separated seeds. kNN is DETERMINISTIC, so extra seeds "
                        "produce identical rows (kept only for schema parity); default "
                        "is a single seed.")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds parsed to empty list.")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = args.artifacts_root / args.dataset_tag / args.variant_tag / ts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {args.out_dir}")
    print(f"Seeds   : {seeds}  (deterministic)")

    # ---- 1. Load silver -------------------------------------------------
    print("\n=== Loading silver ===")
    adata = load_silver_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"Concatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from obs.")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    # ---- 2. Held-out region split ---------------------------------------
    print("\n=== Held-out region split ===")
    regions = DEFAULT_HOLDOUT_REGIONS if args.use_default_holdout_regions else None
    if regions is None:
        raise SystemExit("Custom regions not configured; pass --use-default-holdout-regions.")
    apply_holdout_regions(adata, batch_key=args.batch_key, regions=regions)

    # ---- 3. kNN prediction (deterministic) + neighborhood layers --------
    # Compute once; the per-seed loop just re-labels the (identical) rows.
    print("\n=== Spatial kNN prediction ===")
    predict_knn(adata, k=args.k, weighting=args.weighting, batch_key=args.batch_key)
    add_neighborhood_layers(adata, batch_key=args.batch_key, n_neighs=args.nbr_neighs)

    per_seed_frames: List[pd.DataFrame] = []
    for s_idx, seed in enumerate(seeds):
        print(f"\n--- seed {seed} ({s_idx + 1}/{len(seeds)}) ---")
        df = build_pearson_dataframe(adata, seed=seed, log1p=True, n_hvg=50)
        per_seed_frames.append(df)
        write_predicted_adata(adata, args.out_dir, seed)
        for split in ("all", "train", "test"):
            row = df[(df["split"] == split) & (df["branch"] == "cell") &
                     (df["axis"] == "cell_wise") & (df["transform"] == "log1p") &
                     (df["gene_subset"] == "all")]
            if not row.empty:
                print(f"  Pearson cell_wise log1p all (split={split:<5s}) = "
                      f"{float(row['pearson_mean'].iloc[0]):.4f}")

    per_seed = (pd.concat(per_seed_frames, ignore_index=True)
                if per_seed_frames else pd.DataFrame())

    # ---- 4. Write CSVs --------------------------------------------------
    print("\n=== Writing outputs ===")
    write_pearson_outputs(args.out_dir, per_seed)

    # ---- 5. Console summary --------------------------------------------
    print("\n" + "=" * 78)
    print(f"SUMMARY (cell_wise × log1p × all, {len(seeds)} seed(s); kNN deterministic)")
    print("=" * 78)
    canonical = per_seed[(per_seed["branch"] == "cell") &
                         (per_seed["axis"] == "cell_wise") &
                         (per_seed["transform"] == "log1p") &
                         (per_seed["gene_subset"] == "all")]
    for split in ("all", "train", "test"):
        sub = canonical[canonical["split"] == split]
        if not sub.empty:
            v = sub["pearson_mean"].astype(float)
            print(f"  cell Pearson (split={split:<5s}) = {v.mean():.4f}")

    # ---- 6. Stub config -------------------------------------------------
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "Spatial kNN-regression imputation baseline "
                                      "(region-holdout)."},
        "dataset": {"dataset_name": args.dataset_tag, "dataset_tag": args.dataset_tag,
                    "root_data_dir": str(Path(args.silver_dir).parent.parent)},
        "model": {"model_name": "knn-spatial"},
        "knn": {"k": args.k, "weighting": args.weighting,
                "nbr_neighs": args.nbr_neighs, "batch_key": args.batch_key,
                "seeds": seeds, "holdout_regions": DEFAULT_HOLDOUT_REGIONS},
    }
    with open(args.out_dir / "user_specified_config.yaml", "w") as f:
        yaml.safe_dump(stub_cfg, f, sort_keys=False)
    print(f"\n  -> {args.out_dir / 'user_specified_config.yaml'}")

    print("\n" + "=" * 78)
    print(f"DONE\n  Run dir : {args.out_dir}\n  Variant : {args.variant_tag}\n"
          f"  Seeds   : {len(seeds)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
