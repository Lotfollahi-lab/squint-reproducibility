"""
PCA + Leiden baseline for the SQUINT cell-identification benchmark.

Pipeline (matches what compute_inference_metrics.py does for a SQUINT
variant, but the "code" here is the Leiden cluster id, not a VQ index):
  1. Load every silver `.h5ad` in --silver-dir, concatenate.
  2. log1p-CPM normalisation, PCA (n_comps=50).  ONE-TIME (PCA with
     arpack solver is deterministic on the same data).
  3. For each seed in `--seeds`:
       - kNN graph (random_state=seed)
       - Leiden binary search to hit `--n-clusters` (default 30,
         matching the SQUINT cell codebook size)
       - UMAP layout
       - NMI/ARI vs cell_type and niche labels (SAME helper as
         compute_inference_metrics.py — same CSV schema)
       - iLISI / MMD on the PCA embedding vs `obs[batch_key]` (SAME
         helpers, RNG-controlled sub-sampling).
  4. Aggregate across seeds:
       - Mean of NMI / ARI / iLISI / MMD per (split, code_key, label_key)
         and per (emb_key, metric) -> top-level
         `metrics/{niche_identification,batch_integration}_metrics.csv`.
         compare_variants.py reads these and compares the BASELINE MEAN
         against SQUINT variants directly.
       - Long-format per-seed tables -> `metrics/per_seed_*.csv`. Use
         these for variance / error-bar analysis.
  5. UMAP plots from seed[0] only at top level (PNG + SVG, hybrid
     raster scatter + vector text), coloured by Leiden / cell_type /
     niche / batch. Per-seed UMAPs not saved (would 5x the plot count
     for marginal value — UMAP layout varies but cluster identity
     across runs is captured by the metrics).

Default output location:
  <ARTIFACTS>/<dataset>/<variant>/<TS>/
      predicted_adata.h5ad                            (seed[0])
      metrics/
          niche_identification_metrics.csv             (MEAN across seeds)
          batch_integration_metrics.csv                (MEAN across seeds)
          per_seed_niche_identification.csv            (long, with `seed`)
          per_seed_batch_integration.csv               (long, with `seed`)
      umap_plots/{leiden,cell_type,niche,batch}.{png,svg}   (seed[0])
      user_specified_config.yaml                       (records seeds + n_seeds)

`<variant>` defaults to "baseline-pca-leiden" so this drops into
compare_variants.py heatmaps alongside the SQUINT variants. Override
--variant-tag to register the same baseline against a different
hyperparameter setting (e.g. --variant-tag baseline-pca-leiden-k15
when scanning n-clusters).

Usage:
    python analysis/benchmarking/cell_type_identification/run_pca_leiden.py
    # 5-seed default; override:
    python analysis/benchmarking/cell_type_identification/run_pca_leiden.py \\
        --seeds 0,1,2,3,4
    # custom dataset / cluster count:
    python analysis/benchmarking/cell_type_identification/run_pca_leiden.py \\
        --silver-dir /nfs/.../silver/chl59-8b_1p \\
        --dataset-tag chl59-8b_1p \\
        --n-clusters 30
"""

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Same upstream-warning suppression as the sibling SQUINT scripts.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

# Editable text in SVG, matching plot_holdout_regions.py / chl59 ground-truth.
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42


# ---------------------------------------------------------------------------
# Paths + defaults (mirror the SQUINT artifacts layout so this baseline's
# outputs are picked up by compare_variants.py with zero changes).
# ---------------------------------------------------------------------------

DEFAULT_ARTIFACTS_ROOT = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts"
)
DEFAULT_DATASET_TAG    = "mmb0-1b_smb1-1b_1p"
DEFAULT_SILVER_DIR     = (
    "/nfs/team361/sb75/DATASETS/silver/mmb0-1b_smb1-1b_1p"
)
DEFAULT_VARIANT_TAG    = "baseline-pca-leiden"

# Same defaults as compute_inference_metrics.py so the comparison is
# apples-to-apples.
DEFAULT_CELL_LABEL_KEYS  = ["cell_type", "cell_types"]
DEFAULT_NICHE_LABEL_KEYS = ["niche", "Sub_molecular_tissue_region", "ccf_region_name"]


# ---------------------------------------------------------------------------
# Metric helpers (imported from compute_inference_metrics.py to guarantee
# the numbers are computed identically — same iLISI kNN size, same MMD
# bandwidth heuristic, same NMI/ARI sklearn flavour, etc.).
# ---------------------------------------------------------------------------

def _import_metric_helpers() -> Tuple[callable, callable, callable]:
    """Return (compute_nmi_ari, compute_ilisi, compute_mmd_comparable).
    Adds the squint examples/ dir to sys.path so the imports succeed
    when this script is run from the reproducibility repo."""
    squint_examples = Path(
        "/Users/sebastian.birk/workspace/squint_claude_project/squint/examples"
    )
    # Also try the team NFS install path (where it lives on the cluster).
    cluster_examples = Path("/nfs/team361/sb75/squint/examples")
    for p in (squint_examples, cluster_examples):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from compute_inference_metrics import (  # noqa: E402
        compute_nmi_ari,
        compute_ilisi,
        compute_mmd_comparable,
    )
    return compute_nmi_ari, compute_ilisi, compute_mmd_comparable


# ---------------------------------------------------------------------------
# Data loading + preprocessing
# ---------------------------------------------------------------------------

def _load_concat(silver_dir: Path, batch_key: str) -> ad.AnnData:
    """Read every .h5ad under `silver_dir`, attach a per-cell `obs[batch_key]`
    derived from `uns['batch']` (matches the convention SQUINT uses), and
    concatenate. The dataset blob's `process_anndata_batch` reindexes every
    section to a canonical gene panel; here we expect the silver files to
    already share columns (run examples/harmonize_*.py first if not).

    Held-out sections — held out of *both training and metrics* — are
    skipped at load time via the `SQUINT_EXCLUDE_BATCHES` env var
    (comma-separated list of tokens, e.g. `Lung13,Lung5_Rep3`). Each
    token is matched as a case-sensitive substring against either the
    file basename (e.g. `Lung13+SMI+Flat+data.tar.h5ad`) OR
    `str(uns['batch'])`. Matching files are dropped from the concat
    entirely, so every downstream method sees only the training set
    and per-seed metrics are computed only over training-set cells.
    Set to empty / unset to load everything.
    """
    # ---- Parse holdout tokens from env (CSV) -------------------------
    import os as _os
    raw_excl = _os.environ.get("SQUINT_EXCLUDE_BATCHES", "")
    excluded = [t.strip() for t in raw_excl.split(",") if t.strip()]
    if excluded:
        print(f"  [_load_concat] SQUINT_EXCLUDE_BATCHES={excluded} — "
              "files matching any of these tokens (against filename or "
              "uns['batch']) will be SKIPPED from train + metrics.")

    files = sorted(Path(silver_dir).glob("*.h5ad"))
    if not files:
        raise SystemExit(f"No .h5ad files under {silver_dir}.")
    print(f"Loading {len(files)} silver file(s):")
    pieces = []
    n_skipped_holdout = 0
    for f in files:
        a = ad.read_h5ad(f)
        # Per-cell batch label, broadcast from uns['batch'] (same convention
        # as InMemoryDatasetBlob.process_anndata_batch).
        bid = a.uns.get("batch", None)
        if bid is None:
            print(f"  skip {f.name}: missing uns['batch']")
            continue

        # Skip if this section matches the holdout list. We check both
        # the filename (most user-friendly — matches what they see on
        # disk) AND the uns['batch'] value (most robust — matches what
        # the in-memory adata reports). Case-sensitive substring match.
        if excluded:
            haystacks = [f.name, str(bid)]
            matched_token = next(
                (tok for tok in excluded
                 if any(tok in hay for hay in haystacks)),
                None,
            )
            if matched_token is not None:
                print(f"  skip {f.name:60s}  HELD OUT "
                      f"(matched {matched_token!r}; batch={bid!r})")
                n_skipped_holdout += 1
                continue

        # Normalise to int when possible (some files store 'batchN' string).
        if isinstance(bid, str) and bid.startswith("batch"):
            try:    bid = int(bid[5:])
            except: pass
        elif isinstance(bid, str):
            try:    bid = int(bid)
            except: pass
        a.obs[batch_key] = bid
        a.obs[batch_key] = a.obs[batch_key].astype("category")
        print(f"  {f.name:60s}  n_obs={a.n_obs:>7d}  batch={bid!r}")
        pieces.append(a)

    if excluded:
        print(f"  [_load_concat] held out {n_skipped_holdout} section(s); "
              f"using {len(pieces)} for train + metrics.")

    if not pieces:
        raise SystemExit(
            "No usable .h5ad files (all missing uns['batch'], or all "
            "matched the holdout list)."
        )
    if len(pieces) == 1:
        return pieces[0]
    return ad.concat(pieces, axis=0, join="outer", index_unique="-",
                     uns_merge="first")


def _preprocess(adata: ad.AnnData, n_pcs: int) -> ad.AnnData:
    """Standard log1p-CPM + PCA. Operates in place on a copy so the
    caller can keep the raw counts on `adata.layers["counts"]`."""
    a = adata.copy()
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    # Scale per-gene to unit variance (zero-mean) before PCA. This is
    # the standard scanpy / scIB convention and matches what most
    # benchmark methods feed into PCA for cell-type clustering.
    sc.pp.scale(a, max_value=10.0)
    sc.tl.pca(a, n_comps=n_pcs, zero_center=True, svd_solver="arpack",
              random_state=0)
    return a


# ---------------------------------------------------------------------------
# Leiden with resolution tuned for n_clusters
# ---------------------------------------------------------------------------

def _leiden_n_clusters(
        adata: ad.AnnData,
        n_clusters: int,
        n_neighbors: int = 15,
        max_iters: int = 25,
        rng_seed: int = 0,
    ) -> Tuple[str, int, float]:
    """Binary search on Leiden resolution to land at exactly `n_clusters`
    (or as close as possible). Returns (obs_key, achieved_n_clusters,
    final_resolution).

    The standard scanpy Leiden takes a `resolution` knob with no closed-
    form mapping to cluster count; for benchmark reproducibility we
    bisect resolution in [0.05, 10.0] until cluster count matches.
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca",
                    random_state=rng_seed)

    obs_key = "leiden"
    lo, hi = 0.05, 10.0
    best_key = None
    best_diff = None
    best_n = None
    best_res = None
    print(f"Bisecting Leiden resolution to hit n_clusters = {n_clusters} "
          f"(initial range {lo}-{hi}, max {max_iters} iters):")
    for it in range(max_iters):
        mid = 0.5 * (lo + hi)
        sc.tl.leiden(adata, resolution=mid, key_added=obs_key,
                     random_state=rng_seed)
        n_found = int(adata.obs[obs_key].astype(str).nunique())
        diff = abs(n_found - n_clusters)
        print(f"  iter {it+1:>2d}  resolution={mid:.4f}  -> n_clusters={n_found}")
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_n = n_found
            best_res = mid
            best_key = obs_key
        if n_found == n_clusters:
            return obs_key, n_found, mid
        if n_found < n_clusters:
            lo = mid
        else:
            hi = mid

    # Re-run at the best resolution we found (in case the loop ended on
    # a non-best iteration).
    sc.tl.leiden(adata, resolution=best_res, key_added=obs_key,
                 random_state=rng_seed)
    print(f"  ! exact match not reached; using closest "
          f"(n={best_n}, resolution={best_res:.4f}).")
    return best_key, best_n, best_res


# ---------------------------------------------------------------------------
# UMAP plots (hybrid raster scatter + vector text, same convention as the
# other plot_*.py scripts in squint/examples).
# ---------------------------------------------------------------------------

def _spot_size(n_cells: int) -> float:
    """Same density-bucket auto-pick as plot_chl59_ground_truth.py."""
    if n_cells >= 50_000: return 0.5
    if n_cells >= 20_000: return 1.0
    if n_cells >=  5_000: return 2.0
    return 4.0


def _save_dual(fig, out_path: Path, **kw) -> None:
    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".png"), **kw)
    fig.savefig(out_path.with_suffix(".svg"), **kw)


def _categorical_palette(values: pd.Series, cmap_name: str) -> Dict[str, str]:
    cats = sorted({str(v) for v in values if str(v) not in {"nan", "None", ""}})
    if not cats:
        return {}
    cmap = mpl.colormaps.get_cmap(cmap_name)
    return {c: mpl.colors.to_hex(cmap(i % cmap.N)) for i, c in enumerate(cats)}


def _plot_umap(
        adata: ad.AnnData,
        color_key: str,
        out_path: Path,
        cmap_name: str = "tab20",
        legend_markersize: float = 9.0,
        spot_size: Optional[float] = None,
        dpi: int = 300,
    ) -> None:
    """Single-panel UMAP coloured by `obs[color_key]`. Hybrid SVG:
    rasterised scatter (one <image>) + vector title and legend (<text>)
    so titles remain editable in Illustrator/Inkscape."""
    if "X_umap" not in adata.obsm:
        raise RuntimeError("Compute UMAP before plotting.")
    if color_key not in adata.obs.columns:
        print(f"  skip UMAP({color_key}): column missing")
        return
    xy = np.asarray(adata.obsm["X_umap"], dtype=float)
    labels = adata.obs[color_key].astype(str)
    palette = _categorical_palette(labels, cmap_name=cmap_name)
    s = spot_size if spot_size is not None else _spot_size(adata.n_obs)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    # Cells with NaN-equivalent label go to grey.
    mask_known = labels.isin(palette.keys()).to_numpy()
    if (~mask_known).any():
        ax.scatter(xy[~mask_known, 0], xy[~mask_known, 1],
                   c="#dddddd", s=s, linewidths=0, marker="o",
                   rasterized=True)
    for cat, color in palette.items():
        m = (labels == cat).to_numpy()
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1],
                   c=color, s=s, linewidths=0, marker="o",
                   rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"PCA + Leiden — coloured by {color_key}", fontsize=12)
    if palette:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markeredgewidth=0,
                   markersize=legend_markersize, label=cat)
            for cat, color in palette.items()
        ]
        ax.legend(
            handles=handles,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize=9, handletextpad=0.4,
            borderaxespad=0.0,
        )
    fig.tight_layout()
    _save_dual(fig, out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metric runners (delegate to the SQUINT helpers for parity)
# ---------------------------------------------------------------------------

def _compute_niche_identification(
        adata: ad.AnnData,
        leiden_key: str,
        cell_label_keys: List[str],
        niche_label_keys: List[str],
        compute_nmi_ari: callable,
    ) -> pd.DataFrame:
    """Build the (codes, code_name, label_key) triples that
    `compute_nmi_ari` expects, then return the assembled DataFrame in
    the same schema as compute_inference_metrics.py writes.

    `code_key` is set to "leiden" for all rows (this baseline doesn't
    have a separate cell vs niche codebook). Compare_variants.py will
    pick up these as a separate column in the heatmap, alongside the
    SQUINT cell_code_indices / neighborhood_code_indices columns.

    `leiden_key` may hold any cluster identifier — integer-string
    labels like Leiden's ('0', '1', ...), or alphanumeric labels like
    Novae's `assign_domains` output ('D162', 'D45', ...). We always
    pass cluster IDs through pd.Categorical to produce 0..N-1 integer
    codes regardless of the input format. The actual integer values
    don't carry semantics — `compute_nmi_ari` only cares about the
    partition (which cells share a code).
    """
    codes = pd.Categorical(
        adata.obs[leiden_key].astype(str)
    ).codes.astype(np.int64)
    code_label_pairs = [
        (codes, "leiden", lab)
        for lab in cell_label_keys + niche_label_keys
    ]
    rows = compute_nmi_ari(
        adata=adata, code_label_pairs=code_label_pairs,
        cell_mask=None, split_label="all",
    )
    if "data_split" in adata.obs.columns:
        # Per-split breakdown (mirrors compute_inference_metrics.py).
        split_col = adata.obs["data_split"].astype("object")
        for split_name in ("train", "test"):
            mask = (split_col == split_name).to_numpy()
            if mask.any():
                more = compute_nmi_ari(
                    adata=adata, code_label_pairs=code_label_pairs,
                    cell_mask=mask, split_label=split_name,
                )
                if not more.empty:
                    rows = pd.concat([rows, more], ignore_index=True)
    if rows.empty:
        return rows
    # Aggregate niche labels into one weighted "niche" row per (split,
    # code_key) -- same convention as compute_inference_metrics.
    niche_set = set(niche_label_keys)
    agg_rows: List[dict] = []
    for (split, code_key), grp in rows[
        rows["label_key"].isin(niche_set)
    ].groupby(["split", "code_key"], sort=False):
        n_total = int(grp["n_cells"].sum())
        if n_total <= 0:
            continue
        w_nmi = float((grp["NMI"] * grp["n_cells"]).sum() / n_total)
        w_ari = float((grp["ARI"] * grp["n_cells"]).sum() / n_total)
        agg_rows.append({
            "split": split, "code_key": code_key, "label_key": "niche",
            "NMI": w_nmi, "ARI": w_ari, "n_cells": n_total,
            "n_true_clusters": pd.NA, "n_pred_clusters": pd.NA,
        })
    if agg_rows:
        rows = pd.concat([rows, pd.DataFrame(agg_rows)], ignore_index=True)
    return rows


def _compute_batch_integration(
        adata: ad.AnnData,
        batch_key: str,
        compute_ilisi: callable,
        compute_mmd_comparable: callable,
        ilisi_n_neighbors: int,
        mmd_n_sub: int,
        mmd_n_sigma: int,
        seed: int,
    ) -> pd.DataFrame:
    """iLISI + MMD on the PCA embedding vs `obs[batch_key]`. Same
    helpers + same defaults as compute_inference_metrics.py. Returns
    a DataFrame with columns (emb_key, metric, score)."""
    if batch_key not in adata.obs.columns:
        print(f"  batch key {batch_key!r} missing; skipping batch integration")
        return pd.DataFrame(columns=["emb_key", "metric", "score"])
    emb = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    batch = adata.obs[batch_key].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    rows = []
    ilisi = compute_ilisi(emb, batch, n_neighbors=ilisi_n_neighbors)
    if ilisi is not None:
        rows.append({"emb_key": "X_pca", "metric": "iLISI", "score": ilisi})
        print(f"    iLISI = {ilisi:.4f}")
    mmd = compute_mmd_comparable(emb, batch, n_sub=mmd_n_sub,
                                  n_sigma=mmd_n_sigma, rng=rng)
    if mmd is not None:
        rows.append({"emb_key": "X_pca", "metric": "MMD", "score": mmd})
        print(f"    MMD   = {mmd:.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def _aggregate_niche(per_seed_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Take a list of per-seed niche-identification DataFrames and
    return ONE DataFrame whose values are means across seeds (NMI, ARI)
    and medians (n_cells, n_true_clusters, n_pred_clusters — these can
    differ slightly because the Leiden binary search lands at slightly
    different cluster counts per seed).

    Output schema matches the per-seed schema, so compare_variants.py
    consumes it identically. Empty input returns an empty DataFrame.
    """
    if not per_seed_dfs:
        return pd.DataFrame()
    long_df = pd.concat(per_seed_dfs, ignore_index=True)
    if long_df.empty:
        return long_df
    group_cols = ["split", "code_key", "label_key"]
    agg = long_df.groupby(group_cols, as_index=False, sort=False).agg(
        NMI=("NMI", "mean"),
        ARI=("ARI", "mean"),
        n_cells=("n_cells", lambda s: int(s.median()) if s.notna().any() else pd.NA),
        n_true_clusters=("n_true_clusters", lambda s: int(s.median()) if s.notna().any() else pd.NA),
        n_pred_clusters=("n_pred_clusters", lambda s: int(s.median()) if s.notna().any() else pd.NA),
    )
    return agg


def _aggregate_batch_int(per_seed_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Mean of `score` across seeds per (emb_key, metric)."""
    if not per_seed_dfs:
        return pd.DataFrame()
    long_df = pd.concat(per_seed_dfs, ignore_index=True)
    if long_df.empty:
        return long_df
    return long_df.groupby(["emb_key", "metric"], as_index=False, sort=False).agg(
        score=("score", "mean"),
    )


# ---------------------------------------------------------------------------
# Per-seed runtime tracking (shared by every benchmark runner)
# ---------------------------------------------------------------------------

def _record_seed_runtime(
        tracker: List[Dict],
        seed: int,
        local_seconds: float,
        run_dir: Path,
        method: Optional[str] = None,
        shared_setup_seconds: float = 0.0,
    ) -> None:
    """Append THIS seed's wall-clock runtime to the tracker, and also
    drop a `runtime.csv` into `<run_dir>/seeds/seed_<N>/` for per-seed
    inspection.

    Runtime methodology (apples-to-apples vs SQUINT and across baselines):
      - `local_seconds`: per-seed clustering cost — typically Leiden
        binary search (sc.pp.neighbors + iterated sc.tl.leiden). For
        seed-dependent methods (scVI, NicheCompass, GraphST) this also
        includes the per-seed model fit. Caller times this with
        `time.time()` around the in-loop code.
      - `shared_setup_seconds`: one-shot cost of producing the
        embedding the per-seed clusterer reads — e.g. BANKSY+Harmony,
        PCA, FM extraction (scGPT / Geneformer / etc.), neighbor-expr
        PCA. Zero for methods where everything is per-seed (scVI).
      - `runtime_seconds` (= local + shared) is what gets reported in
        the per_seed_runtimes.csv. This is the "time to obtain
        clusters from raw data for one seed" — directly comparable to
        SQUINT's per-seed (train + predict) numbers.

    EXCLUDED from both timers: UMAP (visualization only — it isn't a
    preprocessing step for Leiden in any of our runners), NMI/ARI
    computation, iLISI/MMD/ASW computation, per-seed plot writes.
    Those are benchmark scaffolding, not method cost.

    The single-row per-seed CSV has columns: seed, method,
    runtime_seconds, local_seconds, shared_setup_seconds.
    """
    runtime_seconds = float(local_seconds) + float(shared_setup_seconds)
    row = {
        "seed": int(seed),
        "method": method,
        "runtime_seconds":      runtime_seconds,
        "local_seconds":        float(local_seconds),
        "shared_setup_seconds": float(shared_setup_seconds),
    }
    tracker.append(row)
    seed_dir = Path(run_dir) / "seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(seed_dir / "runtime.csv", index=False)


# String written into `runtime_summary.csv`'s `runtime_includes` column
# so the methodology travels with the CSV. Cross-method runtime
# comparisons live or die on getting this right — making it explicit
# avoids silently comparing a "metrics-included" baseline against a
# "metrics-excluded" SQUINT number.
_RUNTIME_INCLUDES_NOTE = (
    "model fit (per-seed where seed-dependent) + clustering "
    "(typically Leiden binary search). Excludes metric computation "
    "(NMI/ARI/iLISI/MMD), UMAP (visualization), and plot writes. "
    "runtime_seconds = local_seconds + shared_setup_seconds where "
    "shared_setup_seconds amortises one-shot embedding compute "
    "(e.g. BANKSY+Harmony, PCA, FM extraction) by adding it back to "
    "each seed's local cost — so each row represents 'time to obtain "
    "clusters from raw data for one seed', directly comparable to "
    "SQUINT's per-seed (train + predict) numbers."
)


def _write_runtime_csvs(
        tracker: List[Dict],
        run_dir: Path,
    ) -> None:
    """Write `metrics/per_seed_runtimes.csv` (long format with one row
    per seed) and `metrics/runtime_summary.csv` (mean / std / min / max
    / total across all seeds). No-op if `tracker` is empty.

    Both CSVs reflect the new "model + clustering only" runtime
    methodology — see `_record_seed_runtime` and the
    `runtime_includes` column of `runtime_summary.csv`.
    """
    if not tracker:
        return
    metrics_dir = Path(run_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.DataFrame(tracker)
    out = metrics_dir / "per_seed_runtimes.csv"
    long_df.to_csv(out, index=False)
    print(f"  -> {out}")

    secs = long_df["runtime_seconds"].astype(float)
    # Shared setup is identical across seeds for a given method; pull
    # it from any row (defaults to 0.0 if the column is missing for
    # legacy callers).
    shared_secs = (
        float(long_df["shared_setup_seconds"].iloc[0])
        if "shared_setup_seconds" in long_df.columns else 0.0
    )
    summary = pd.DataFrame([{
        "method":               long_df["method"].iloc[0]
                                if "method" in long_df.columns
                                and long_df["method"].notna().any()
                                else None,
        "n_seeds":              int(len(long_df)),
        "mean_seconds":         float(secs.mean()),
        "std_seconds":          float(secs.std(ddof=1)) if len(secs) > 1 else 0.0,
        "min_seconds":          float(secs.min()),
        "max_seconds":          float(secs.max()),
        "total_seconds":        float(secs.sum()),
        "shared_setup_seconds": shared_secs,
        "runtime_includes":     _RUNTIME_INCLUDES_NOTE,
    }])
    out = metrics_dir / "runtime_summary.csv"
    summary.to_csv(out, index=False)
    print(f"  -> {out}  (mean={secs.mean():.1f}s ± "
          f"{secs.std(ddof=1) if len(secs) > 1 else 0.0:.1f}s, "
          f"total={secs.sum():.1f}s; shared_setup={shared_secs:.1f}s)")


# ---------------------------------------------------------------------------
# h5ad-write compatibility helper (shared by every benchmark runner)
# ---------------------------------------------------------------------------

def _sanitize_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    """Strip pandas Arrow-backed string dtypes from `adata.obs` /
    `adata.var` so the H5AD writer can serialize them.

    Newer pandas (>=2.x) sometimes produces
    `pandas.arrays.ArrowStringArray` for string columns/indexes when
    PyArrow is installed (notably after `ad.concat`). Older anndata
    versions don't have a registered writer for that dtype and crash
    with::

        IORegistryError: No method registered for writing
        <class 'pandas.arrays.ArrowStringArray'> into <h5py.Group>
        Error raised while writing key '_index' of <h5py.Group> to /obs

    This helper:
      * forces `obs.index` and `var.index` to a plain NumPy object
        array (going through `np.asarray(..., dtype=object)` —
        `Index.astype(str)` alone is NOT enough on pandas 2.x with
        PyArrow because it returns ANOTHER ArrowStringArray).
      * casts any column with `string` / `string[pyarrow]` dtype to
        `object` via the same NumPy round-trip.
      * leaves numeric, bool, categorical, datetime columns untouched.

    Mutates `adata` in place AND returns it (chainable).
    """
    import numpy as np
    import pandas as pd

    def _is_arrow_string_dtype(dtype) -> bool:
        # pandas.StringDtype (numpy or pyarrow backed) reports as
        # `string` / `string[python]` / `string[pyarrow]`. Also catch
        # the bare `pandas.arrays.ArrowStringArray` case which
        # historically reported just as `string`.
        s = str(dtype)
        return s.startswith("string") or "pyarrow" in s.lower()

    def _force_object_index(idx) -> pd.Index:
        # Going through np.asarray(..., dtype=object) is the only
        # reliable way to escape PyArrow backing on newer pandas;
        # `Index.astype(str)` and `Index.astype("object")` both keep
        # the index as ArrowStringArray on some pandas/Arrow combos.
        arr = np.asarray(list(map(str, idx)), dtype=object)
        return pd.Index(arr, name=idx.name)

    def _force_object_series(s: pd.Series) -> pd.Series:
        arr = np.asarray(list(map(str, s.to_list())), dtype=object)
        return pd.Series(arr, index=s.index, name=s.name)

    for df_name in ("obs", "var"):
        df = getattr(adata, df_name)
        # Index — always rewrite if dtype looks like a string/Arrow type,
        # OR if its underlying values array is from pandas.arrays.
        idx = df.index
        idx_dtype_str = str(getattr(idx, "dtype", "object"))
        if (_is_arrow_string_dtype(idx_dtype_str)
                or "Arrow" in type(getattr(idx, "values", idx)).__name__):
            df.index = _force_object_index(idx)
        # Columns — same logic per column.
        for col in df.columns:
            s = df[col]
            if _is_arrow_string_dtype(s.dtype) or "Arrow" in type(s.array).__name__:
                df[col] = _force_object_series(s)
    return adata


# ---------------------------------------------------------------------------
# Per-seed output writer (shared by every benchmark runner)
# ---------------------------------------------------------------------------

def _write_per_seed_outputs(
        seed: int,
        run_dir: Path,
        adata: ad.AnnData,
        leiden_key: str,
        niche_df: pd.DataFrame,
        batch_df: pd.DataFrame,
        cell_keys: List[str],
        niche_keys: List[str],
        batch_key: str,
        dpi: int,
        spot_size: Optional[float] = None,
    ) -> Path:
    """Write THIS seed's metric CSVs and UMAP plots into
    `<run_dir>/seeds/seed_<N>/`.

    Layout produced::

        <run_dir>/seeds/seed_<N>/
            metrics/
                niche_identification_metrics.csv
                batch_integration_metrics.csv
            umap_plots/
                <leiden_key>.{png,svg}
                <each cell_label_key present>.{png,svg}
                <each niche_label_key present>.{png,svg}
                <batch_key>.{png,svg}

    The `seed` column is dropped from the per-seed CSVs (the file
    location already records which seed it came from). The aggregated
    top-level `metrics/*.csv` (mean across seeds) is still produced
    separately by the caller — this helper only writes the per-seed
    files.

    Assumes `adata.obs[leiden_key]` and `adata.obsm["X_umap"]` are
    already populated for THIS seed (caller's responsibility).
    """
    seed_dir = run_dir / "seeds" / f"seed_{seed}"
    metrics_dir = seed_dir / "metrics"
    umap_dir    = seed_dir / "umap_plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    umap_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed metric CSVs (drop the redundant `seed` column).
    if niche_df is not None and not niche_df.empty:
        out = niche_df.drop(columns=["seed"], errors="ignore")
        out.to_csv(metrics_dir / "niche_identification_metrics.csv",
                   index=False)
    if batch_df is not None and not batch_df.empty:
        out = batch_df.drop(columns=["seed"], errors="ignore")
        out.to_csv(metrics_dir / "batch_integration_metrics.csv",
                   index=False)

    # Per-seed UMAPs (one figure per obs key — same set as the top-
    # level umap_plots/ dir).
    plot_keys = [(leiden_key, "tab20")]
    plot_keys += [(k, "tab20") for k in cell_keys  if k in adata.obs.columns]
    plot_keys += [(k, "tab10") for k in niche_keys if k in adata.obs.columns]
    plot_keys += [(batch_key, "Set2")]
    for key, cmap in plot_keys:
        out_path = umap_dir / key.replace("/", "_")
        _plot_umap(adata, color_key=key, out_path=out_path,
                   cmap_name=cmap, spot_size=spot_size, dpi=dpi)

    return seed_dir


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
    p.add_argument("--dataset-tag",  type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--variant-tag",  type=str, default=DEFAULT_VARIANT_TAG)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override run dir. Default: "
                        "<ARTIFACTS>/<dataset>/<variant>/<TS>/")
    p.add_argument("--n-clusters", type=int, default=30,
                   help="Target Leiden cluster count (matches the SQUINT "
                        "cell codebook size).")
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="kNN size for the Leiden + UMAP graph.")
    p.add_argument("--cell-label-keys", type=str,
                   default=",".join(DEFAULT_CELL_LABEL_KEYS))
    p.add_argument("--niche-label-keys", type=str,
                   default=",".join(DEFAULT_NICHE_LABEL_KEYS))
    p.add_argument("--batch-key", type=str, default="adata_batch_id")
    p.add_argument("--ilisi-n-neighbors", type=int, default=90)
    p.add_argument("--mmd-n-sub",   type=int, default=2000)
    p.add_argument("--mmd-n-sigma", type=int, default=1000)
    p.add_argument(
        "--seeds", type=str, default="0,1,2,3,4",
        help="Comma-separated random seeds; one full pipeline run per "
             "seed (kNN graph + Leiden + UMAP + iLISI/MMD subsampling "
             "all use the seed). PCA itself is deterministic (arpack) "
             "so it's done once and shared across seeds. Default: "
             "'0,1,2,3,4' (5 seeds for variance estimation).",
    )
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    cell_keys  = [k.strip() for k in args.cell_label_keys.split(",")  if k.strip()]
    niche_keys = [k.strip() for k in args.niche_label_keys.split(",") if k.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds parsed to empty list; pass at least one integer.")

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

    # 1. Load + preprocess + PCA (one-time — PCA with arpack solver is
    #    deterministic on the same input, so we share it across seeds).
    #    Timed separately as `shared_setup_seconds` so the per-seed
    #    runtime can fold it back in for an apples-to-apples comparison
    #    with seed-dependent methods (scVI, SQUINT) that re-pay this
    #    cost on every seed.
    _shared_t0 = time.time()
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    adata = _preprocess(adata, n_pcs=args.n_pcs)
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + preprocess + PCA): {shared_setup_seconds:.1f}s")

    # 2. Per-seed loop: rebuild kNN graph + Leiden + UMAP + metrics.
    #    Seed-controlled randomness:
    #      - sc.pp.neighbors(random_state=seed)  # kNN graph
    #      - sc.tl.leiden(random_state=seed)     # cluster init
    #      - sc.tl.umap(random_state=seed)       # layout init
    #      - compute_mmd_comparable(rng=seed)    # subsample
    #      - compute_ilisi (pynndescent has internal randomness; we
    #        cannot pass a seed to it directly, but the underlying kNN
    #        is recomputed each call so the variability is captured).
    compute_nmi_ari, compute_ilisi, compute_mmd_comparable = (
        _import_metric_helpers()
    )
    per_seed_niche: List[pd.DataFrame] = []
    per_seed_batch: List[pd.DataFrame] = []
    seed_summary: List[Dict] = []
    runtime_tracker: List[Dict] = []

    # We keep seed[0]'s leiden + UMAP as the canonical adata snapshot
    # written at the top level (predicted_adata.h5ad + umap_plots/).
    seed0_state: Optional[Dict] = None

    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        # TIMED block — Leiden binary search (kNN graph + iterated
        # leiden). For PCA-Leiden the embedding is shared (PCA is
        # deterministic), so there's no per-seed model fit here.
        # Everything below the `seed_seconds = ...` line runs UNTIMED
        # (benchmark scaffolding: metrics, UMAP-for-viz, plot writes).
        seed_t0 = time.time()
        leiden_key, n_found, resolution = _leiden_n_clusters(
            adata, n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors, rng_seed=seed,
        )
        seed_seconds = time.time() - seed_t0
        print(f"Leiden settled at {n_found} clusters (resolution={resolution:.4f}).")
        # Record runtime BEFORE any of the untimed scaffolding so a
        # later failure in metrics / plots doesn't strand the timing.
        _record_seed_runtime(
            runtime_tracker, seed=seed,
            local_seconds=seed_seconds,
            shared_setup_seconds=shared_setup_seconds,
            run_dir=args.out_dir, method="PCA-Leiden",
        )
        print(f"  runtime (seed {seed}): local={seed_seconds:.1f}s, "
              f"shared={shared_setup_seconds:.1f}s, "
              f"total={seed_seconds + shared_setup_seconds:.1f}s")

        # ---- UNTIMED below: metric computation + visualization ----------
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

        print("\n  -- Batch integration on X_pca --")
        bint_df = _compute_batch_integration(
            adata=adata, batch_key=args.batch_key,
            compute_ilisi=compute_ilisi,
            compute_mmd_comparable=compute_mmd_comparable,
            ilisi_n_neighbors=args.ilisi_n_neighbors,
            mmd_n_sub=args.mmd_n_sub, mmd_n_sigma=args.mmd_n_sigma,
            seed=seed,
        )
        if not bint_df.empty:
            bint_df.insert(0, "seed", seed)
        per_seed_batch.append(bint_df)

        seed_summary.append({
            "seed": seed,
            "leiden_n_clusters": int(n_found),
            "leiden_resolution": float(resolution),
        })

        # Per-seed outputs: metric CSVs + UMAP plots into
        # `<run_dir>/seeds/seed_<N>/` so every seed is fully
        # inspectable on disk (not only via the long-format
        # per_seed_*.csv aggregate at the top level).
        seed_dir = _write_per_seed_outputs(
            seed=seed, run_dir=args.out_dir, adata=adata,
            leiden_key=leiden_key, niche_df=niche_df,
            batch_df=bint_df,
            cell_keys=cell_keys, niche_keys=niche_keys,
            batch_key=args.batch_key, dpi=args.dpi,
        )
        print(f"  -> wrote per-seed outputs to {seed_dir}")

        # Snapshot seed[0] for the top-level AnnData / UMAP outputs.
        if s_idx == 0:
            seed0_state = {
                "leiden_key": leiden_key,
                "n_found": n_found,
                "resolution": resolution,
            }

    # 3. Write per-seed long-format CSVs and aggregated mean CSVs.
    if per_seed_niche:
        long_niche = pd.concat(
            [df for df in per_seed_niche if not df.empty],
            ignore_index=True,
        )
    else:
        long_niche = pd.DataFrame()
    if per_seed_batch:
        long_batch = pd.concat(
            [df for df in per_seed_batch if not df.empty],
            ignore_index=True,
        )
    else:
        long_batch = pd.DataFrame()

    if not long_niche.empty:
        out = metrics_dir / "per_seed_niche_identification.csv"
        long_niche.to_csv(out, index=False)
        print(f"\n  -> {out}")
    if not long_batch.empty:
        out = metrics_dir / "per_seed_batch_integration.csv"
        long_batch.to_csv(out, index=False)
        print(f"  -> {out}")

    # Aggregated MEANS — same schema as compute_inference_metrics.py
    # writes for SQUINT runs, so compare_variants.py reads these
    # without any custom logic. Drop the `seed` column before
    # aggregation.
    if not long_niche.empty:
        long_niche_no_seed = long_niche.drop(columns=["seed"])
        agg_niche = _aggregate_niche([long_niche_no_seed])
        out = metrics_dir / "niche_identification_metrics.csv"
        agg_niche.to_csv(out, index=False)
        print(f"  -> {out}  (mean across {len(seeds)} seeds)")
    if not long_batch.empty:
        long_batch_no_seed = long_batch.drop(columns=["seed"])
        agg_batch = _aggregate_batch_int([long_batch_no_seed])
        out = metrics_dir / "batch_integration_metrics.csv"
        agg_batch.to_csv(out, index=False)
        print(f"  -> {out}  (mean across {len(seeds)} seeds)")

    # Runtime CSVs (per-seed long + summary).
    _write_runtime_csvs(runtime_tracker, args.out_dir)

    # 4. Summary printout: mean ± std for the headline metrics.
    print()
    print("=" * 78)
    print("SUMMARY (mean ± std across {} seeds)".format(len(seeds)))
    print("=" * 78)
    if not long_niche.empty:
        # Headline rows: split=all, label=cell_type, niche
        head = long_niche[long_niche["split"] == "all"].copy()
        for label in ("cell_type", "niche"):
            sub = head[head["label_key"] == label]
            if sub.empty:
                continue
            for col in ("NMI", "ARI"):
                vals = sub[col].astype(float)
                print(
                    f"  leiden vs {label:<10s} {col} = "
                    f"{vals.mean():.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}  "
                    f"(min={vals.min():.4f}, max={vals.max():.4f})"
                )
    if not long_batch.empty:
        for metric in ("iLISI", "MMD"):
            sub = long_batch[long_batch["metric"] == metric]
            if sub.empty:
                continue
            vals = sub["score"].astype(float)
            print(
                f"  X_pca {metric:<5s} = "
                f"{vals.mean():.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}  "
                f"(min={vals.min():.4f}, max={vals.max():.4f})"
            )

    # 5. Top-level AnnData snapshot (seed[0]) + UMAP plots.
    if seed0_state is not None:
        adata.uns["pca_leiden_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["pca_leiden_resolution"] = float(seed0_state["resolution"])
        adata.uns["pca_leiden_seeds"]      = seeds
        _sanitize_for_h5ad(adata).write_h5ad(args.out_dir / "predicted_adata.h5ad")
        print(f"\n  -> {args.out_dir / 'predicted_adata.h5ad'}  "
              f"(seed[0] snapshot)")

        umap_dir = args.out_dir / "umap_plots"
        umap_dir.mkdir(parents=True, exist_ok=True)
        plot_keys = [(seed0_state["leiden_key"], "tab20")]
        plot_keys += [(k, "tab20") for k in cell_keys  if k in adata.obs.columns]
        plot_keys += [(k, "tab10") for k in niche_keys if k in adata.obs.columns]
        plot_keys += [(args.batch_key, "Set2")]
        print("\n=== UMAP plots (seed[0]) ===")
        for key, cmap in plot_keys:
            out_path = umap_dir / key.replace("/", "_")
            _plot_umap(adata, color_key=key, out_path=out_path,
                       cmap_name=cmap, dpi=args.dpi)
            print(f"  -> {out_path}.{{png,svg}}")

    # 6. Stub config — record EVERY seed and the per-seed Leiden
    #    resolution for traceability.
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "PCA + Leiden baseline (multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "PCA-Leiden"},
        "pca_leiden": {
            "n_clusters_target": int(args.n_clusters),
            "n_pcs":              int(args.n_pcs),
            "n_neighbors":        int(args.n_neighbors),
            "seeds":              seeds,
            "seed_summary":       seed_summary,
        },
    }
    with open(args.out_dir / "user_specified_config.yaml", "w") as f:
        yaml.safe_dump(stub_cfg, f, sort_keys=False)
    print(f"\n  -> {args.out_dir / 'user_specified_config.yaml'}")

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  Run dir : {args.out_dir}")
    print(f"  Variant : {args.variant_tag}  (picked up by compare_variants.py)")
    print(f"  Seeds   : {len(seeds)}  ({seeds})")
    print("=" * 78)


if __name__ == "__main__":
    main()
