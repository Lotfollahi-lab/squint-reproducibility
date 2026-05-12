"""
Neighborhood Gene Expression + PCA + Leiden baseline for the SQUINT
cell-identification benchmark.

Niche-aware extension of `run_pca_leiden.py`: instead of running PCA on
each cell's own expression, we first aggregate expression across each
cell's spatial neighborhood (sum of `spatial_connectivities.T @ X`),
then log1p-CPM normalise the aggregated counts and run PCA on those.
This is the simplest possible "neighborhood PCA" baseline — same
philosophy as Banksy (mean of neighbor expression) and CellCharter
(neighbor-aggregated scVI latent), but with no learned embedding.

Pipeline (matches `neigh_expr_pca_benchmarking.ipynb`):
  1. Load every silver `.h5ad` in --silver-dir, concatenate.
  2. Per-batch spatial kNN graph -> block-diagonal concat into
     `obsp['spatial_connectivities']` (same recipe as
     `nichecompass_benchmarking.ipynb`).
  3. `layers['X_neigh'] = spatial_connectivities.T @ X`  (sum of
     neighbor expression).
  4. log1p-CPM normalise the X_neigh layer.
  5. PCA (n_comps=20) on the X_neigh layer -> `obsm['X_pca']`. ONE-TIME
     (PCA with arpack solver is deterministic on the same input).
  6. For each seed in `--seeds`:
       - kNN graph + Leiden binary search to hit `--n-clusters`
       - UMAP layout
       - NMI/ARI vs cell_type and niche labels
       - iLISI / MMD on X_pca vs `obs[batch_key]`.
  7. Aggregate across seeds (mean) -> top-level metrics CSVs.

Output layout mirrors `run_pca_leiden.py` so compare_variants.py picks
this baseline up identically.

Usage:
    python analysis/benchmarking/niche_identification/run_neigh_expr_pca.py
    # custom n-clusters / pca dim:
    python analysis/benchmarking/niche_identification/run_neigh_expr_pca.py \\
        --n-clusters 30 --n-pcs 20
"""

import argparse
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
import scipy.sparse as sp

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
    _leiden_n_clusters,
    _load_concat,
    _plot_umap,
    _sanitize_for_h5ad,
    _record_seed_runtime,
    _write_per_seed_outputs,
    _write_runtime_csvs,
)


DEFAULT_VARIANT_TAG = "baseline-neigh-expr-pca"


# ---------------------------------------------------------------------------
# Per-batch spatial kNN graph (squidpy) + block-diagonal concat.
# Same recipe as the NicheCompass benchmarking notebook.
# ---------------------------------------------------------------------------

def _spatial_knn_per_batch(
        adata: ad.AnnData,
        n_neighs: int,
        batch_key: str,
        include_self_loop: bool = True,
    ) -> ad.AnnData:
    """Compute squidpy spatial kNN graphs separately per batch, then
    concat block-diagonally so cross-batch edges don't appear. Stores
    the combined sparse CSR in `adata.obsp['spatial_connectivities']`.
    """
    import squidpy as sq

    if "spatial" not in adata.obsm:
        raise SystemExit(
            "obsm['spatial'] missing -- silver h5ad must include "
            "spatial coordinates."
        )
    if batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={batch_key!r} missing from obs.")
    # Preserve first-seen batch order (matches NicheCompass nb behaviour).
    first_idx = adata.obs.reset_index().groupby(batch_key).head(1).index
    batches = adata.obs.iloc[first_idx][batch_key].tolist()
    print(f"  computing per-batch spatial kNN (n_neighs={n_neighs}, "
          f"include_self_loop={include_self_loop}) over {len(batches)} "
          f"batch(es): {batches}")
    pieces = []
    for b in batches:
        sub = adata[adata.obs[batch_key] == b].copy()
        sq.gr.spatial_neighbors(
            sub, coord_type="generic", spatial_key="spatial",
            n_neighs=n_neighs, set_diag=include_self_loop,
        )
        pieces.append(sub.obsp["spatial_connectivities"])
    adata.obsp["spatial_connectivities"] = sp.block_diag(pieces, format="csr")
    return adata


# ---------------------------------------------------------------------------
# Neighbor-aggregate expression + log1p-CPM + PCA
# ---------------------------------------------------------------------------

def _neigh_expr_pca(
        adata: ad.AnnData,
        n_pcs: int,
    ) -> ad.AnnData:
    """X_neigh = spatial_connectivities.T @ X (sum-aggregated expression
    per neighborhood). log1p-CPM normalised, then PCA on it. Returns a
    NEW AnnData where `obsm['X_pca']` is the neighborhood PCA.

    Matches `neigh_expr_pca_benchmarking.ipynb` exactly:
        - target_sum=1e4
        - sc.pp.pca(adata, layer='X_neigh', n_comps=n_pcs)
    """
    a = adata.copy()
    if "spatial_connectivities" not in a.obsp:
        raise SystemExit(
            "obsp['spatial_connectivities'] missing -- did you run "
            "_spatial_knn_per_batch first?"
        )
    print("  building neighborhood-aggregated expression layer "
          "(X_neigh = spatial_connectivities.T @ X)")
    a.layers["counts"] = a.X.copy()
    # `.T @ X` so cell i picks up the SUM of expression across cells
    # that have i as a neighbor. With include_self_loop=True the cell
    # itself is included.
    a.layers["X_neigh"] = a.obsp["spatial_connectivities"].T @ a.X
    sc.pp.normalize_total(a, target_sum=1e4, layer="X_neigh")
    sc.pp.log1p(a, layer="X_neigh")
    sc.pp.pca(
        a, layer="X_neigh", n_comps=n_pcs,
        zero_center=True, svd_solver="arpack", random_state=0,
    )
    return a


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
    # Spatial graph + PCA hyperparams (match the notebook).
    p.add_argument("--n-spatial-neighs", type=int, default=10,
                   help="Spatial kNN size (squidpy) for the neighbor-aggregation. "
                        "Default 10 (matches nichecompass / banksy notebooks).")
    p.add_argument("--n-pcs", type=int, default=20,
                   help="PCA dimensionality. Default 20 (matches the "
                        "neigh_expr_pca notebook).")
    p.add_argument("--n-clusters", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="Expression-graph kNN size (downstream Leiden / UMAP).")
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

    # 1. Load + spatial graph + neighborhood PCA (one-time shared setup).
    #    Timed as `shared_setup_seconds` for apples-to-apples runtime
    #    comparison (see `_record_seed_runtime` docstring).
    _shared_t0 = time.time()
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    adata = _spatial_knn_per_batch(
        adata, n_neighs=args.n_spatial_neighs, batch_key=args.batch_key,
        include_self_loop=True,
    )
    adata = _neigh_expr_pca(adata, n_pcs=args.n_pcs)
    print(f"X_neigh PCA: shape={adata.obsm['X_pca'].shape}")
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + spatial graph + neigh-PCA): "
          f"{shared_setup_seconds:.1f}s")

    # 2. Per-seed Leiden + metrics.
    compute_nmi_ari, compute_ilisi, compute_mmd_comparable = (
        _import_metric_helpers()
    )
    per_seed_niche: List[pd.DataFrame] = []
    per_seed_batch: List[pd.DataFrame] = []
    seed_summary: List[Dict] = []
    runtime_tracker: List[Dict] = []
    seed0_state: Optional[Dict] = None

    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        # TIMED block: Leiden binary search on the shared neigh-PCA
        # latent. Below `seed_seconds = ...` runs UNTIMED.
        seed_t0 = time.time()
        leiden_key, n_found, resolution = _leiden_n_clusters(
            adata, n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors, rng_seed=seed,
        )
        seed_seconds = time.time() - seed_t0
        print(f"Leiden settled at {n_found} clusters "
              f"(resolution={resolution:.4f}).")
        _record_seed_runtime(
            runtime_tracker, seed=seed,
            local_seconds=seed_seconds,
            shared_setup_seconds=shared_setup_seconds,
            run_dir=args.out_dir, method="NeighExprPCA-Leiden",
        )
        print(f"  runtime (seed {seed}): local={seed_seconds:.1f}s, "
              f"shared={shared_setup_seconds:.1f}s, "
              f"total={seed_seconds + shared_setup_seconds:.1f}s")

        # ---- UNTIMED below: metrics + visualization ---------------------
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

        print("\n  -- Batch integration on X_pca (neigh-expr) --")
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

        seed_dir = _write_per_seed_outputs(
            seed=seed, run_dir=args.out_dir, adata=adata,
            leiden_key=leiden_key, niche_df=niche_df,
            batch_df=bint_df,
            cell_keys=cell_keys, niche_keys=niche_keys,
            batch_key=args.batch_key, dpi=args.dpi,
        )
        print(f"  -> wrote per-seed outputs to {seed_dir}")

        if s_idx == 0:
            seed0_state = {"leiden_key": leiden_key,
                           "n_found": n_found, "resolution": resolution}

    # 3. Aggregate across seeds.
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
            print(f"  X_pca {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 5. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["neigh_expr_pca_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["neigh_expr_pca_resolution"] = float(seed0_state["resolution"])
        adata.uns["neigh_expr_pca_seeds"] = seeds
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
                       "description": "Neighborhood Expression + PCA + Leiden "
                                      "baseline (multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag": args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "NeighExprPCA-Leiden"},
        "neigh_expr_pca": {
            "n_spatial_neighs": int(args.n_spatial_neighs),
            "n_pcs": int(args.n_pcs),
            "n_clusters_target": int(args.n_clusters),
            "n_neighbors": int(args.n_neighbors),
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
