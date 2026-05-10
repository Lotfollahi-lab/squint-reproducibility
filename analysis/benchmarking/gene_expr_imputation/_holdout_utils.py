"""
Shared utilities for the gene-expression imputation (Pearson) benchmark.

Used by every method script in this directory:

  - apply_holdout_regions(...)
        Mark cells inside per-batch spatial rectangles as
        `data_split = "test"`; everyone else gets `"train"`.
        Mirrors the geometry of the existing SQUINT
        `_patch_holdout_regions` so the train/test split is identical
        across SQUINT and the new baselines.

  - spatial_knn_per_batch(...)
        Block-diagonal squidpy spatial kNN — same recipe as
        `run_neigh_expr_pca._spatial_knn_per_batch`.

  - compute_X_nbr(...)
        1-hop neighborhood mean of `adata.X` via the spatial graph,
        stored in `adata.layers["X_nbr"]`. Target for nbr-level
        reconstruction.

  - compute_pearson_split(...)
        Cell-wise Pearson on log1p(target) vs log1p(pred) over genes,
        per-cell mean (and median).

  - build_pearson_dataframe(...)
        Builds the per-seed Pearson DataFrame in the SAME schema as
        SQUINT's `compute_inference_metrics.compute_pearson_metrics`
        so the plotting script can union them.

Output convention used by every method script:
  <out_dir>/predicted_adata.h5ad
        Contains `obs["data_split"]` ("train" / "test"), and one or
        both of `layers["X_hat"]` (cell-level pred) and
        `layers["X_hat_nbr"]` (nbr-level pred), `layers["X_nbr"]`
        (nbr-level target).
  <out_dir>/metrics/per_seed_pearson_reconstruction.csv
        Long format: seed, branch, split, axis, transform, gene_subset,
        pearson_mean, pearson_median, n_cells, n_genes
  <out_dir>/metrics/pearson_reconstruction_metrics.csv
        Mean across seeds (same columns minus `seed`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Shared defaults — keep in lockstep with the niche / cell-type benchmarks.
# Override at the CLI per script if you point at a different dataset.
# ---------------------------------------------------------------------------

DEFAULT_SILVER_DIR = (
    "/nfs/team361/sb75/DATASETS/silver/mmb0-1b_smb1-1b_1p"
)
DEFAULT_ARTIFACTS_ROOT = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts"
)
DEFAULT_DATASET_TAG = "mmb0-1b_smb1-1b_1p"


# ---------------------------------------------------------------------------
# Default holdout regions — matches SQUINT's `_patch_holdout_regions`.
# ---------------------------------------------------------------------------

DEFAULT_HOLDOUT_REGIONS: Dict[int, Dict[str, float]] = {
    # STARmap+ batch 15: a patch in the upper-left quadrant.
    15: {"x_min_pct": 0.10, "x_max_pct": 0.35,
         "y_min_pct": 0.55, "y_max_pct": 0.80},
    # MERFISH batch 82: a patch in the lower-right quadrant
    # (different anatomical region than batch 15's hold-out).
    82: {"x_min_pct": 0.55, "x_max_pct": 0.80,
         "y_min_pct": 0.20, "y_max_pct": 0.45},
}


def _resolve_region_to_bbox(
        coords_xy: np.ndarray,
        region: Dict[str, float],
    ) -> Tuple[float, float, float, float]:
    """Resolve a region spec (mix of absolute and percentile keys) to
    absolute (x_min, x_max, y_min, y_max). Percentile keys are
    interpreted in the section's own xy range — the same convention as
    SQUINT's `_patch_holdout_regions`."""
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]
    x_min_data, x_max_data = float(np.min(x)), float(np.max(x))
    y_min_data, y_max_data = float(np.min(y)), float(np.max(y))
    x_range = x_max_data - x_min_data
    y_range = y_max_data - y_min_data

    def _resolve(side: str, axis: str) -> float:
        # `side` in {"min", "max"}; `axis` in {"x", "y"}.
        abs_key = f"{axis}_{side}"
        pct_key = f"{axis}_{side}_pct"
        if abs_key in region:
            return float(region[abs_key])
        if pct_key in region:
            base = x_min_data if axis == "x" else y_min_data
            ran  = x_range    if axis == "x" else y_range
            return base + ran * float(region[pct_key])
        # Fallback: the section's own min/max (the rectangle is
        # silently the entire section in that axis).
        if side == "min":
            return x_min_data if axis == "x" else y_min_data
        return x_max_data if axis == "x" else y_max_data

    return (
        _resolve("min", "x"), _resolve("max", "x"),
        _resolve("min", "y"), _resolve("max", "y"),
    )


def apply_holdout_regions(
        adata: ad.AnnData,
        batch_key: str,
        regions: Optional[Dict[int, Dict[str, float]]] = None,
        spatial_key: str = "spatial",
    ) -> ad.AnnData:
    """Tag cells inside each batch's spatial rectangle as
    `data_split == "test"`; everyone else gets `"train"`. Operates in
    place. Returns the same `adata` for chaining.

    Identical mask geometry to SQUINT's `_patch_holdout_regions`
    (default) so `data_split` aligns 1:1 between SQUINT and the new
    baselines.
    """
    if regions is None:
        regions = DEFAULT_HOLDOUT_REGIONS
    if spatial_key not in adata.obsm:
        raise SystemExit(
            f"obsm[{spatial_key!r}] missing — silver h5ad must include "
            "spatial coordinates."
        )
    if batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={batch_key!r} missing from obs.")

    test_mask = np.zeros(adata.n_obs, dtype=bool)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    batch_arr = adata.obs[batch_key].to_numpy()

    n_batches_with_region = 0
    for batch_id, region in regions.items():
        m = batch_arr == batch_id
        if not m.any():
            print(f"  [holdout] batch {batch_id} not present in adata; skipping")
            continue
        x_lo, x_hi, y_lo, y_hi = _resolve_region_to_bbox(coords[m], region)
        in_box = (
            (coords[:, 0] >= x_lo) & (coords[:, 0] <= x_hi)
            & (coords[:, 1] >= y_lo) & (coords[:, 1] <= y_hi)
            & m
        )
        n_in = int(in_box.sum())
        n_total_batch = int(m.sum())
        pct = 100.0 * n_in / max(n_total_batch, 1)
        print(f"  [holdout] batch {batch_id}: held out {n_in}/{n_total_batch} "
              f"({pct:.1f}%) cells (bbox x=[{x_lo:.1f},{x_hi:.1f}] "
              f"y=[{y_lo:.1f},{y_hi:.1f}])")
        test_mask |= in_box
        n_batches_with_region += 1

    if n_batches_with_region == 0:
        raise SystemExit(
            "No batches matched the holdout regions dict; check your "
            f"--batch-key and the regions {list(regions.keys())} vs the "
            f"batches present {list(np.unique(batch_arr))}."
        )

    split = np.full(adata.n_obs, "train", dtype=object)
    split[test_mask] = "test"
    adata.obs["data_split"] = pd.Categorical(split, categories=["train", "test"])
    n_test = int(test_mask.sum())
    print(f"  [holdout] data_split: train={adata.n_obs - n_test}, test={n_test}")
    return adata


# ---------------------------------------------------------------------------
# Spatial kNN graph + X_nbr (neighborhood-mean target for nbr-level Pearson)
# ---------------------------------------------------------------------------

def spatial_knn_per_batch(
        adata: ad.AnnData,
        n_neighs: int,
        batch_key: str,
        include_self_loop: bool = True,
        spatial_key: str = "spatial",
    ) -> ad.AnnData:
    """Compute squidpy spatial kNN per batch and concat block-diagonally
    (no cross-batch edges). Stores combined sparse CSR in
    `adata.obsp["spatial_connectivities"]`. Identical recipe to
    `run_neigh_expr_pca._spatial_knn_per_batch` so the resulting graph
    matches the existing baselines exactly."""
    import squidpy as sq

    if spatial_key not in adata.obsm:
        raise SystemExit(f"obsm[{spatial_key!r}] missing")
    if batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={batch_key!r} missing from obs")

    first_idx = adata.obs.reset_index().groupby(batch_key).head(1).index
    batches = adata.obs.iloc[first_idx][batch_key].tolist()
    print(f"  spatial kNN (n_neighs={n_neighs}, "
          f"include_self_loop={include_self_loop}) over {len(batches)} "
          f"batch(es): {batches}")
    pieces = []
    for b in batches:
        sub = adata[adata.obs[batch_key] == b].copy()
        sq.gr.spatial_neighbors(
            sub, coord_type="generic", spatial_key=spatial_key,
            n_neighs=n_neighs, set_diag=include_self_loop,
        )
        pieces.append(sub.obsp["spatial_connectivities"])
    adata.obsp["spatial_connectivities"] = sp.block_diag(pieces, format="csr")
    return adata


def compute_X_nbr(
        adata: ad.AnnData,
        normalize: str = "mean",
    ) -> ad.AnnData:
    """Compute the 1-hop neighborhood-aggregated expression target for
    nbr-level Pearson, stored at `adata.layers["X_nbr"]`.

    `normalize="mean"` (default): each cell's nbr expression is the
    MEAN over its neighbors (including self if the spatial graph has
    self-loops). This matches SQUINT's nbr-target convention.

    `normalize="sum"` matches `run_neigh_expr_pca` (which uses
    `connectivities.T @ X` raw sum + normalize_total). Use "mean" here
    for direct comparability with NB-style decoders that predict per-
    cell-mean expression rather than CPM-normalised expression.
    """
    if "spatial_connectivities" not in adata.obsp:
        raise SystemExit(
            "obsp['spatial_connectivities'] missing — "
            "run spatial_knn_per_batch() first."
        )
    A = adata.obsp["spatial_connectivities"].astype(np.float32)
    X = adata.X
    # Aggregate: for each cell i, sum/mean over its neighbors. Use the
    # forward direction (A @ X), where A[i, j] != 0 iff j is a neighbor
    # of i — so row i collects neighbour expression.
    nbr_sum = A @ X
    if normalize == "mean":
        # Row sums of A == number of neighbors of each cell.
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        # Avoid div-by-zero on isolated cells.
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        if sp.issparse(nbr_sum):
            nbr_mean = nbr_sum.multiply(1.0 / row_sums[:, None])
            nbr_mean = nbr_mean.tocsr()
        else:
            nbr_mean = nbr_sum / row_sums[:, None]
        adata.layers["X_nbr"] = nbr_mean
    elif normalize == "sum":
        adata.layers["X_nbr"] = nbr_sum
    else:
        raise ValueError(f"unknown normalize={normalize!r}")
    nnz = (
        adata.layers["X_nbr"].nnz
        if sp.issparse(adata.layers["X_nbr"]) else
        np.count_nonzero(adata.layers["X_nbr"])
    )
    print(f"  computed X_nbr ({normalize}-aggregated, density="
          f"{nnz / (adata.n_obs * adata.n_vars):.3f}); stored at "
          "adata.layers['X_nbr']")
    return adata


# ---------------------------------------------------------------------------
# Pearson computation — full SQUINT-compatible variant set
# ---------------------------------------------------------------------------

def _to_dense_2d(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {X.shape}")
    return X


def _pearson_pairwise(
        a: np.ndarray,
        b: np.ndarray,
        axis: int,
    ) -> np.ndarray:
    """Pairwise Pearson between rows (axis=1) or columns (axis=0) of `a`
    and `b`. Returns a 1D array. NaN where either side has zero
    variance. Identical implementation to SQUINT's
    `compute_inference_metrics._pearson_pairwise` so the numbers are
    bit-for-bit comparable to the SQUINT artifact CSV.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a_mean = a.mean(axis=axis, keepdims=True)
    b_mean = b.mean(axis=axis, keepdims=True)
    a_c = a - a_mean
    b_c = b - b_mean
    num = (a_c * b_c).sum(axis=axis)
    den = np.sqrt((a_c ** 2).sum(axis=axis) * (b_c ** 2).sum(axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _select_hvg_indices(
        target_log1p: np.ndarray,
        n_hvg: int,
    ) -> np.ndarray:
    """Top-`n_hvg` genes by per-gene variance on the log1p target. Same
    selection rule as SQUINT (variance on the log1p TARGET, not pred —
    so the HVG identity is fixed across methods)."""
    n_hvg_eff = min(int(n_hvg), int(target_log1p.shape[1]))
    if n_hvg_eff <= 0:
        return np.array([], dtype=int)
    gene_var = target_log1p.var(axis=0)
    hvg_idx = np.argpartition(-gene_var, n_hvg_eff - 1)[:n_hvg_eff]
    hvg_idx.sort()
    return hvg_idx


def _branch_pearson_rows(
        branch: str,
        target_full: np.ndarray,
        pred_full: np.ndarray,
        cell_mask: np.ndarray,
        split_label: str,
        log1p: bool,
        n_hvg: int,
    ) -> List[dict]:
    """Emit the 6 Pearson variant rows for ONE branch × ONE split:

       gene_wise × raw       × all
       gene_wise × raw       × hvg{N}
       gene_wise × log1p     × all
       gene_wise × log1p     × hvg{N}
       cell_wise × raw       × all
       cell_wise × log1p     × all

    `cell_wise × hvg{N}` is intentionally skipped — Pearson per cell
    over only N genes is statistically noisy. Same convention as SQUINT.

    `log1p=False` flips the loop to skip log1p variants entirely (3
    rows: gene_wise × raw × {all, hvg}, cell_wise × raw × all).
    """
    target = target_full[cell_mask]
    pred   = pred_full[cell_mask]
    if target.size == 0:
        return []
    n_cells, n_genes = target.shape

    # HVG indices computed once per branch × split on the log1p target.
    target_log_for_hvg = np.log1p(np.clip(target, 0, None))
    hvg_idx = _select_hvg_indices(target_log_for_hvg, n_hvg)

    transforms: List[str] = []
    if log1p:
        transforms.append("log1p")
    transforms.append("raw")

    rows: List[dict] = []
    for transform in transforms:
        if transform == "log1p":
            t_full = np.log1p(np.clip(target, 0, None))
            p_full = np.log1p(np.clip(pred,   0, None))
        else:
            t_full, p_full = target, pred

        for axis_name, axis in (("gene_wise", 0), ("cell_wise", 1)):
            gene_subsets = ["all"]
            if axis_name == "gene_wise" and hvg_idx.size > 0:
                gene_subsets.append(f"hvg{hvg_idx.size}")

            for gene_subset in gene_subsets:
                if gene_subset == "all":
                    t_sub, p_sub = t_full, p_full
                    n_genes_sub = n_genes
                else:
                    t_sub = t_full[:, hvg_idx]
                    p_sub = p_full[:, hvg_idx]
                    n_genes_sub = int(hvg_idx.size)

                vec = _pearson_pairwise(p_sub, t_sub, axis=axis)
                vec = vec[np.isfinite(vec)]
                if vec.size == 0:
                    continue
                rows.append({
                    "split":          split_label,
                    "branch":         branch,
                    "axis":           axis_name,
                    "transform":      transform,
                    "gene_subset":    gene_subset,
                    "pearson_mean":   float(vec.mean()),
                    "pearson_median": float(np.median(vec)),
                    "n_cells":        int(n_cells),
                    "n_genes":        n_genes_sub,
                })
    return rows


def build_pearson_dataframe(
        adata: ad.AnnData,
        seed: int,
        log1p: bool = True,
        n_hvg: int = 50,
        verbose: bool = False,
    ) -> pd.DataFrame:
    """Build per-seed Pearson DataFrame matching SQUINT's
    `compute_pearson_metrics` schema EXACTLY.

    For each branch present in `adata.layers` (cell: X_hat vs X;
    niche: X_hat_nbr vs X_nbr), and for each split (all / train /
    test), emits the 6 variant rows defined in `_branch_pearson_rows`:

        gene_wise × raw       × all
        gene_wise × raw       × hvg{N}
        gene_wise × log1p     × all
        gene_wise × log1p     × hvg{N}
        cell_wise × raw       × all
        cell_wise × log1p     × all

    so 1 branch × 3 splits × 6 variants = 18 rows per seed per branch.

    Output columns: seed, branch, split, axis, transform, gene_subset,
    pearson_mean, pearson_median, n_cells, n_genes.

    Set `log1p=False` to skip log1p variants (only 3 rows per branch×split).
    `n_hvg` defaults to 50 — same as SQUINT's training-time metric set.
    """
    if "data_split" not in adata.obs.columns:
        raise SystemExit("data_split column missing — call apply_holdout_regions first.")

    split_col = adata.obs["data_split"].astype("object").to_numpy()

    branches: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if "X_hat" in adata.layers:
        branches.append((
            "cell",
            _to_dense_2d(adata.X),
            _to_dense_2d(adata.layers["X_hat"]),
        ))
    if "X_hat_nbr" in adata.layers:
        if "X_nbr" not in adata.layers:
            raise SystemExit(
                "X_hat_nbr is present but X_nbr (the target) is missing — "
                "call compute_X_nbr() before scoring."
            )
        branches.append((
            "niche",
            _to_dense_2d(adata.layers["X_nbr"]),
            _to_dense_2d(adata.layers["X_hat_nbr"]),
        ))

    if not branches:
        raise SystemExit(
            "No reconstruction layers found — at least one of "
            "layers['X_hat'] / layers['X_hat_nbr'] must be present."
        )

    n_obs = adata.n_obs
    rows: List[dict] = []
    for branch, target, pred in branches:
        for split_label in ("all", "train", "test"):
            if split_label == "all":
                mask = np.ones(n_obs, dtype=bool)
            else:
                mask = split_col == split_label
            if mask.sum() == 0:
                continue
            split_rows = _branch_pearson_rows(
                branch=branch,
                target_full=target,
                pred_full=pred,
                cell_mask=mask,
                split_label=split_label,
                log1p=log1p,
                n_hvg=n_hvg,
            )
            for r in split_rows:
                r["seed"] = int(seed)
                rows.append(r)
                if verbose:
                    print(
                        f"  [{r['split']:<5s}] {r['branch']:<5s} "
                        f"{r['axis']:<9s} {r['transform']:<5s} "
                        f"{r['gene_subset']:<7s} "
                        f"mean={r['pearson_mean']:.4f}  "
                        f"median={r['pearson_median']:.4f}  "
                        f"(n_cells={r['n_cells']}, n_genes={r['n_genes']})"
                    )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Re-order columns: seed first, then SQUINT-canonical order.
    cols = ["seed", "branch", "split", "axis", "transform", "gene_subset",
            "pearson_mean", "pearson_median", "n_cells", "n_genes"]
    return df[cols]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def aggregate_pearson_across_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean of pearson_{mean,median} across seeds, grouped by
    (branch, split, axis, transform, gene_subset)."""
    if per_seed.empty:
        return per_seed
    groupcols = ["branch", "split", "axis", "transform", "gene_subset"]
    agg = (
        per_seed
        .groupby(groupcols, sort=False)
        .agg(
            pearson_mean=("pearson_mean", "mean"),
            pearson_median=("pearson_median", "mean"),
            n_cells=("n_cells", "mean"),
            n_genes=("n_genes", "first"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return agg


def write_pearson_outputs(
        out_dir: Path,
        per_seed: pd.DataFrame,
    ) -> None:
    """Write `<out_dir>/metrics/per_seed_pearson_reconstruction.csv` and
    `pearson_reconstruction_metrics.csv` (mean across seeds)."""
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_seed_csv = metrics_dir / "per_seed_pearson_reconstruction.csv"
    per_seed.to_csv(per_seed_csv, index=False)
    print(f"  -> {per_seed_csv}")
    if not per_seed.empty:
        agg = aggregate_pearson_across_seeds(per_seed)
        mean_csv = metrics_dir / "pearson_reconstruction_metrics.csv"
        agg.to_csv(mean_csv, index=False)
        print(f"  -> {mean_csv}")


# ---------------------------------------------------------------------------
# Silver loader (thin wrapper around the existing helper from the
# cell_type_identification side, so we keep the same concat ordering /
# index_unique convention).
# ---------------------------------------------------------------------------

def load_silver_concat(silver_dir: Path, batch_key: str) -> ad.AnnData:
    """Load + concat all silver h5ads in `silver_dir`. Defers to
    `run_pca_leiden._load_concat` so the index-unique convention,
    `obs[batch_key]` population, and ordering match every other
    benchmark script in the repo."""
    import sys
    sib = (Path(__file__).resolve().parent.parent / "cell_type_identification")
    if str(sib) not in sys.path:
        sys.path.insert(0, str(sib))
    from run_pca_leiden import _load_concat  # type: ignore[import-not-found]
    return _load_concat(silver_dir, batch_key=batch_key)
