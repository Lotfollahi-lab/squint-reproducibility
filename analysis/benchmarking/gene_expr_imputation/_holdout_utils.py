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
  <out_dir>/predicted_adata_seed{N}.h5ad
        ONE per seed (via write_predicted_adata) so EVERY seed can be
        re-scored later (rescore_imputation.py), not just seed 0.
        Contains `obs["data_split"]` ("train" / "test"), and one or
        both of `layers["X_hat"]` (cell-level pred) and
        `layers["X_hat_nbr"]` (nbr-level pred), `layers["X_nbr"]`
        (nbr-level target). (Legacy runs may have a single
        `predicted_adata.h5ad` = seed 0 only; still written for compat.)
  <out_dir>/metrics/per_seed_pearson_reconstruction.csv
        Long format: seed, branch, split, axis, transform, gene_subset,
        pearson_mean, pearson_median, n_cells, n_genes
  <out_dir>/metrics/pearson_reconstruction_metrics.csv
        Mean across seeds (same columns minus `seed`).
"""
from __future__ import annotations

import sys
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
# Neighborhood aggregation of a cell-level prediction, so methods WITHOUT a
# native neighborhood branch (GeST, scVI, vanilla-VQ-cell, SQUINT-imputed) can
# still be scored at the neighborhood level: X_hat_nbr = spatial-graph mean of
# X_hat, compared against X_nbr (the same aggregation of the TRUE X). Default
# n_neighs=16 to MATCH SQUINT's native niche graph (its `+knn16+` spatial graph),
# so the X_nbr targets — and hence the niche-level Pearson — are graph-consistent
# across all methods and both figures (2026-07-01; was 10).
# ---------------------------------------------------------------------------

def _neighbor_mean(A, M, normalize: str = "mean") -> np.ndarray:
    """Spatial-neighborhood aggregation `A @ M` (dense), optionally divided by
    each cell's neighbor count (`normalize='mean'` — matches compute_X_nbr)."""
    M = _to_dense_2d(M)
    nbr = np.asarray(A @ M, dtype=np.float32)
    if normalize == "mean":
        rs = np.asarray(A.sum(axis=1)).ravel()
        rs = np.where(rs > 0, rs, 1.0)
        nbr = nbr / rs[:, None]
    return nbr


def add_neighborhood_layers(
        adata: ad.AnnData,
        *,
        batch_key: str = "adata_batch_id",
        n_neighs: int = 16,
        normalize: str = "mean",
    ) -> ad.AnnData:
    """Ensure the niche-branch layers exist so `build_pearson_dataframe` scores
    the 'niche' branch for ANY method. Builds the per-section spatial kNN graph
    if absent, sets `layers['X_nbr']` (target = nbr-agg of TRUE X) and — when the
    method only produced a cell-level `layers['X_hat']` and has no native
    `layers['X_hat_nbr']` — sets `layers['X_hat_nbr']` = nbr-agg of X_hat. A
    native X_hat_nbr (SQUINT / NicheCompass) is left untouched. No-op (warns)
    if obsm['spatial'] is missing."""
    if "X_hat_nbr" in adata.layers and "X_nbr" in adata.layers:
        return adata
    if "spatial_connectivities" not in adata.obsp:
        if "spatial" not in adata.obsm:
            print("  [nbr] obsm['spatial'] missing — cannot build neighborhood "
                  "layers; niche branch will be skipped.", file=sys.stderr)
            return adata
        spatial_knn_per_batch(adata, n_neighs=n_neighs, batch_key=batch_key)
    if "X_nbr" not in adata.layers:
        compute_X_nbr(adata, normalize=normalize)
    if "X_hat" in adata.layers and "X_hat_nbr" not in adata.layers:
        A = adata.obsp["spatial_connectivities"].astype(np.float32)
        adata.layers["X_hat_nbr"] = _neighbor_mean(A, adata.layers["X_hat"], normalize)
        print(f"  [nbr] X_hat_nbr = neighborhood-aggregated X_hat "
              f"({normalize}, n_neighs={n_neighs}); "
              f"shape={adata.layers['X_hat_nbr'].shape}")
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


# ---------------------------------------------------------------------------
# Extra metrics beyond Pearson (reviewer panel): rank correlation (Spearman),
# magnitude error (MSE/RMSE), zero/nonzero recovery (AUROC/AUPRC), and a
# marker-gene subset. Each row of the long CSV keeps `pearson_mean` (so the
# existing plots are untouched) and gains `spearman_*` / `mse_*` / `rmse_mean`;
# zero/nonzero rows are emitted separately with axis="entrywise".
# ---------------------------------------------------------------------------

def _rankdata_axis(x: np.ndarray, axis: int) -> np.ndarray:
    """Average-rank `x` along `axis` (ties -> mean rank — matters for the many
    tied zeros in sparse SRT). Uses scipy; falls back to per-slice if the
    installed scipy lacks the `axis` kwarg."""
    from scipy.stats import rankdata
    try:
        return rankdata(x, axis=axis).astype(float)
    except TypeError:                       # scipy < 1.10 has no axis= kwarg
        return np.apply_along_axis(rankdata, axis, x).astype(float)


def _spearman_pairwise(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    """Spearman == Pearson on average-ranks. Rank-based -> invariant to the
    log1p transform (raw and log1p rows get identical values)."""
    return _pearson_pairwise(_rankdata_axis(a, axis), _rankdata_axis(b, axis), axis)


def _mse_pairwise(pred: np.ndarray, target: np.ndarray, axis: int) -> np.ndarray:
    """Per-vector mean squared error along `axis` (per-gene for axis=0,
    per-cell for axis=1). mean(over vectors) == overall MSE; the median is a
    robust complement."""
    return ((pred - target) ** 2).mean(axis=axis)


def _zero_nonzero_scores(pred_raw: np.ndarray, target_raw: np.ndarray):
    """Pooled AUROC / AUPRC for recovering nonzero entries (y = target>0,
    score = predicted magnitude) over all (cell, gene) entries. Returns
    (nan, nan) if degenerate (all-zero / all-nonzero) or sklearn missing."""
    y = (target_raw.ravel() > 0).astype(np.int8)
    if y.min() == y.max():                  # no positives or no negatives
        return float("nan"), float("nan")
    s = pred_raw.ravel().astype(float)
    if not np.isfinite(s).all():
        ok = np.isfinite(s)
        y, s = y[ok], s[ok]
        if y.size == 0 or y.min() == y.max():
            return float("nan"), float("nan")
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception:                       # noqa: BLE001  (sklearn missing / degenerate)
        return float("nan"), float("nan")


def _select_marker_indices(
        target_log1p: np.ndarray,
        labels: np.ndarray,
        n_per_label: int = 10,
        max_total: int = 100,
    ) -> np.ndarray:
    """Union of the top `n_per_label` one-vs-rest marker genes per label, scored
    by (in-label mean - out-of-label mean) on the log1p TARGET (numpy-only,
    scanpy-free). Markers are derived from the truth so the gene identity is
    fixed across methods. Returns [] if <2 usable labels."""
    labels = np.asarray(labels)
    uniq = [u for u in pd.unique(labels) if u == u and str(u) != "nan"]
    if len(uniq) < 2 or target_log1p.shape[1] == 0:
        return np.array([], dtype=int)
    n_genes = target_log1p.shape[1]
    n_per_label = min(int(n_per_label), n_genes)
    grand = target_log1p.mean(axis=0)
    picked: set = set()
    for u in uniq:
        m = labels == u
        if m.sum() == 0:
            continue
        score = target_log1p[m].mean(axis=0) - grand          # log-FC-like vs rest
        top = np.argpartition(-score, n_per_label - 1)[:n_per_label]
        picked.update(int(i) for i in top)
    idx = np.array(sorted(picked), dtype=int)
    if idx.size > max_total:                                  # cap by global score
        order = np.argsort(-(target_log1p[:, idx].var(axis=0)))
        idx = np.sort(idx[order[:max_total]])
    return idx


def _finite_mean_median(vec: np.ndarray) -> Tuple[float, float]:
    """(mean, median) over finite entries; (nan, nan) if none."""
    v = vec[np.isfinite(vec)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(np.median(v))


def _branch_pearson_rows(
        branch: str,
        target_full: np.ndarray,
        pred_full: np.ndarray,
        cell_mask: np.ndarray,
        split_label: str,
        log1p: bool,
        n_hvg: int,
        marker_idx: Optional[np.ndarray] = None,
    ) -> List[dict]:
    """Emit the per-(axis, transform, gene_subset) metric rows for ONE branch ×
    ONE split. Each correlation row carries Pearson AND the reviewer-panel
    metrics (Spearman rank-correlation, MSE/RMSE) so the magnitude / rank views
    sit next to the linear correlation. Gene subsets: all, hvg{N}, and (when
    `marker_idx` is given) markers — markers/hvg are gene_wise only (per-cell
    correlation over a small gene set is statistically noisy; same convention
    as SQUINT). `log1p=False` skips the log1p transform.

    Two extra rows per branch × split (axis="entrywise", transform="counts")
    carry the zero/nonzero recovery scores (AUROC/AUPRC on raw counts) for
    gene_subset in {all, markers}; their correlation columns are NaN.
    """
    target = target_full[cell_mask]
    pred   = pred_full[cell_mask]
    if target.size == 0:
        return []
    n_cells, n_genes = target.shape

    # HVG indices computed once per branch × split on the log1p target.
    target_log_for_hvg = np.log1p(np.clip(target, 0, None))
    hvg_idx = _select_hvg_indices(target_log_for_hvg, n_hvg)
    mk_idx = (marker_idx if marker_idx is not None and marker_idx.size > 0
              else None)

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
            if axis_name == "gene_wise":
                if hvg_idx.size > 0:
                    gene_subsets.append(f"hvg{hvg_idx.size}")
                if mk_idx is not None:
                    gene_subsets.append("markers")

            for gene_subset in gene_subsets:
                if gene_subset == "all":
                    t_sub, p_sub = t_full, p_full
                    n_genes_sub = n_genes
                elif gene_subset == "markers":
                    t_sub, p_sub = t_full[:, mk_idx], p_full[:, mk_idx]
                    n_genes_sub = int(mk_idx.size)
                else:
                    t_sub, p_sub = t_full[:, hvg_idx], p_full[:, hvg_idx]
                    n_genes_sub = int(hvg_idx.size)

                pvec = _pearson_pairwise(p_sub, t_sub, axis=axis)
                if pvec[np.isfinite(pvec)].size == 0:
                    continue
                p_mean, p_med = _finite_mean_median(pvec)
                s_mean, s_med = _finite_mean_median(
                    _spearman_pairwise(p_sub, t_sub, axis=axis))
                m_mean, m_med = _finite_mean_median(
                    _mse_pairwise(p_sub, t_sub, axis=axis))
                rows.append({
                    "split":          split_label,
                    "branch":         branch,
                    "axis":           axis_name,
                    "transform":      transform,
                    "gene_subset":    gene_subset,
                    "pearson_mean":   p_mean,
                    "pearson_median": p_med,
                    "spearman_mean":  s_mean,
                    "spearman_median": s_med,
                    "mse_mean":       m_mean,
                    "mse_median":     m_med,
                    "rmse_mean":      float(np.sqrt(m_mean)) if m_mean == m_mean else float("nan"),
                    "n_cells":        int(n_cells),
                    "n_genes":        n_genes_sub,
                })

    # ---- zero/nonzero recovery (raw counts; transform-independent) ----------
    zsubsets = [("all", None)]
    if mk_idx is not None:
        zsubsets.append(("markers", mk_idx))
    for gs, idx in zsubsets:
        t_z = target if idx is None else target[:, idx]
        p_z = pred   if idx is None else pred[:, idx]
        auroc, auprc = _zero_nonzero_scores(p_z, t_z)
        if auroc != auroc and auprc != auprc:        # both NaN -> skip
            continue
        rows.append({
            "split":       split_label,
            "branch":      branch,
            "axis":        "entrywise",
            "transform":   "counts",
            "gene_subset": gs,
            "auroc_zero":  auroc,
            "auprc_zero":  auprc,
            "n_cells":     int(n_cells),
            "n_genes":     int(t_z.shape[1]),
        })
    return rows


# Cell-type label columns to auto-detect for the marker-gene subset (first
# present wins). Mirrors the niche/cell-type benchmark label preference.
_MARKER_LABEL_KEYS = ("cell_type", "cell_types", "new_annotation", "annotation",
                      "celltype", "CellType")


def build_pearson_dataframe(
        adata: ad.AnnData,
        seed: int,
        log1p: bool = True,
        n_hvg: int = 50,
        verbose: bool = False,
        cell_type_key: Optional[str] = None,
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

    # --- marker-gene subset: top one-vs-rest DE genes per cell type, derived
    #     from the TRUTH (expression + labels) on the TRAIN cells only (so the
    #     held-out region never informs the gene selection). Gene-level, so the
    #     same indices apply to both branches. Skipped if no label column.
    marker_idx: Optional[np.ndarray] = None
    lab_key = cell_type_key or next(
        (k for k in _MARKER_LABEL_KEYS if k in adata.obs.columns), None)
    if lab_key is not None and lab_key in adata.obs.columns:
        labels_all = adata.obs[lab_key].to_numpy()
        train_mask = split_col != "test"
        if train_mask.sum() == 0:
            train_mask = np.ones(n_obs, dtype=bool)
        Xcell = _to_dense_2d(adata.X)
        tl = np.log1p(np.clip(Xcell[train_mask], 0, None))
        marker_idx = _select_marker_indices(tl, labels_all[train_mask])
        if verbose:
            print(f"  markers: {0 if marker_idx is None else marker_idx.size} genes "
                  f"from label '{lab_key}' (train cells)")

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
                marker_idx=marker_idx,
            )
            for r in split_rows:
                r["seed"] = int(seed)
                rows.append(r)
                if verbose:
                    if r["axis"] == "entrywise":   # zero/nonzero recovery row
                        print(
                            f"  [{r['split']:<5s}] {r['branch']:<5s} "
                            f"zero/nonzero {r['gene_subset']:<7s} "
                            f"AUROC={r.get('auroc_zero', float('nan')):.4f}  "
                            f"AUPRC={r.get('auprc_zero', float('nan')):.4f}"
                        )
                    else:
                        print(
                            f"  [{r['split']:<5s}] {r['branch']:<5s} "
                            f"{r['axis']:<9s} {r['transform']:<5s} "
                            f"{r['gene_subset']:<8s} "
                            f"r={r.get('pearson_mean', float('nan')):.4f}  "
                            f"rho={r.get('spearman_mean', float('nan')):.4f}  "
                            f"mse={r.get('mse_mean', float('nan')):.4f}  "
                            f"(n_cells={r['n_cells']}, n_genes={r['n_genes']})"
                        )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Re-order columns: seed first, then the SQUINT-canonical Pearson columns
    # (so existing readers/plots are byte-compatible), then the reviewer-panel
    # metrics, then any zero/nonzero columns that were emitted.
    lead = ["seed", "branch", "split", "axis", "transform", "gene_subset",
            "pearson_mean", "pearson_median"]
    panel = ["spearman_mean", "spearman_median", "mse_mean", "mse_median",
             "rmse_mean", "auroc_zero", "auprc_zero"]
    tail = ["n_cells", "n_genes"]
    ordered = lead + [c for c in panel if c in df.columns] + tail
    # keep any unexpected extra columns at the end rather than dropping them
    ordered += [c for c in df.columns if c not in ordered]
    return df[ordered]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def aggregate_pearson_across_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean across seeds of every metric column, grouped by
    (branch, split, axis, transform, gene_subset). New panel columns
    (spearman/mse/rmse/auroc_zero/auprc_zero) are averaged when present;
    NaNs (e.g. Pearson cols on the entrywise zero/nonzero rows) are skipped."""
    if per_seed.empty:
        return per_seed
    groupcols = ["branch", "split", "axis", "transform", "gene_subset"]
    metric_cols = [c for c in ("pearson_mean", "pearson_median",
                               "spearman_mean", "spearman_median",
                               "mse_mean", "mse_median", "rmse_mean",
                               "auroc_zero", "auprc_zero", "n_cells")
                   if c in per_seed.columns]
    agg_spec = {c: (c, "mean") for c in metric_cols}
    agg_spec["n_genes"] = ("n_genes", "first")
    agg_spec["n_seeds"] = ("seed", "nunique")
    return (per_seed.groupby(groupcols, sort=False).agg(**agg_spec).reset_index())


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


def write_predicted_adata(adata: ad.AnnData, out_dir: Path, seed: int) -> Path:
    """Write ``<out_dir>/predicted_adata_seed{seed}.h5ad`` (one per seed) so
    EVERY seed can be re-scored later (rescore_imputation.py), not just seed 0.
    Sanitises object-dtype obs/var for h5ad using the cell-type-id helper (same
    as the old seed-0 snapshot path). Mutates ``adata`` in place (sanitise)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sib = Path(__file__).resolve().parent.parent / "cell_type_identification"
        if str(sib) not in sys.path:
            sys.path.insert(0, str(sib))
        from run_pca_leiden import _sanitize_for_h5ad  # type: ignore
        _sanitize_for_h5ad(adata)
    except Exception as exc:  # noqa: BLE001
        print(f"  (h5ad sanitizer unavailable: {exc})")
    out = out_dir / f"predicted_adata_seed{seed}.h5ad"
    adata.write_h5ad(out)
    print(f"  -> {out}")
    return out


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
