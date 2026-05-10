"""
CellCharter (scVI + neighbor-aggregated latent + Leiden) baseline for
the SQUINT cell-identification benchmark.

Sibling of `run_scvi.py` and `run_neigh_expr_pca.py` -- this version
trains scVI per seed (so variance reflects scVI training noise too),
then concatenates each cell's scVI latent with sums-of-neighbor scVI
latents at multiple hops via `cc.gr.aggregate_neighbors`, giving the
"X_cellcharter" niche-aware embedding.

Pipeline (matches `cellcharter_benchmarking.ipynb`):
  1. Load + concat silver h5ads.
  2. `layers['counts'] = X.copy()`, log1p-CPM (target_sum=1e6).
  3. Per-batch spatial kNN graph -> block-diag concat.
  4. For each seed:
       - `scvi.settings.seed = seed` + `seed_everything(seed)`
       - `SCVI.setup_anndata(layer='counts', batch_key=batch_key)`
       - Train scVI -> `obsm['X_scVI']`
       - `cc.gr.aggregate_neighbors(adata, n_layers=3,
              use_rep='X_scVI', out_key='X_cellcharter',
              sample_key=batch_key)`
       - Leiden binary search to hit `--n-clusters` on X_cellcharter
       - UMAP layout
       - NMI/ARI vs cell_type / niche labels
       - iLISI / MMD on X_cellcharter

Output layout mirrors `run_scvi.py` so compare_variants.py picks this
baseline up identically.

Approximate runtime: ~25-40 min per seed on a single GPU (scVI
training dominates). Five seeds = ~3 h.

Usage:
    python analysis/benchmarking/niche_identification/run_cellcharter.py
    # smoke test:
    python analysis/benchmarking/niche_identification/run_cellcharter.py --seeds 0
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


DEFAULT_VARIANT_TAG = "baseline-cellcharter"


def _train_scvi_and_aggregate(
        adata: ad.AnnData,
        batch_key: str,
        n_layers_aggregate: int,
        scvi_max_epochs: Optional[int],
        seed: int,
        accelerator: str,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Train scVI for THIS seed, then run cc.gr.aggregate_neighbors to
    build the multi-hop neighbor-aggregated latent.

    Returns (scvi_latent, cellcharter_latent), both NumPy arrays of
    shape (n_obs, n_dim). The caller is expected to have already
    populated `adata.obsp['spatial_connectivities']` (per-batch block-
    diag) — `aggregate_neighbors` reads from there.
    """
    import scvi
    import cellcharter as cc
    from lightning.pytorch import seed_everything

    seed_everything(int(seed))
    scvi.settings.seed = int(seed)

    print(f"  setup_anndata(layer='counts', batch_key={batch_key!r})")
    scvi.model.SCVI.setup_anndata(
        adata, layer="counts", batch_key=batch_key,
    )
    print(f"  training scVI (seed={seed})...")
    model = scvi.model.SCVI(adata)
    model.train(
        max_epochs=scvi_max_epochs,
        accelerator=accelerator,
        early_stopping=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
    )
    scvi_latent = np.asarray(
        model.get_latent_representation(adata), dtype=np.float32,
    )
    adata.obsm["X_scVI"] = scvi_latent
    print(f"  scVI latent shape: {scvi_latent.shape}")

    print(f"  cc.gr.aggregate_neighbors(n_layers={n_layers_aggregate}, "
          f"use_rep='X_scVI', sample_key={batch_key!r})")
    cc.gr.aggregate_neighbors(
        adata, n_layers=int(n_layers_aggregate),
        use_rep="X_scVI", out_key="X_cellcharter",
        sample_key=batch_key,
    )
    cc_latent = np.asarray(adata.obsm["X_cellcharter"])
    print(f"  X_cellcharter shape: {cc_latent.shape}")
    return scvi_latent, cc_latent


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
    # CellCharter / scVI hyperparams (match the notebook).
    p.add_argument("--n-spatial-neighs", type=int, default=10)
    p.add_argument("--n-cc-layers", type=int, default=3,
                   help="Number of neighbor-aggregation hops "
                        "(cc.gr.aggregate_neighbors n_layers). Default 3.")
    p.add_argument("--scvi-max-epochs", type=int, default=None)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--target-sum", type=float, default=1e6,
                   help="normalize_total target_sum. Notebook uses 1e6.")
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

    # 1. Load + log1p-CPM + spatial graph (one-time).
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from obs.")
    adata.layers["counts"] = adata.X.copy()
    print(f"  normalize_total(target_sum={args.target_sum:g}) + log1p")
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
    adata = _spatial_knn_per_batch(
        adata, n_neighs=args.n_spatial_neighs, batch_key=args.batch_key,
        include_self_loop=True,
    )

    # 2. Per-seed loop: scVI + aggregate_neighbors + Leiden + metrics.
    compute_nmi_ari, compute_ilisi, compute_mmd_comparable = (
        _import_metric_helpers()
    )
    per_seed_niche: List[pd.DataFrame] = []
    per_seed_batch: List[pd.DataFrame] = []
    seed_summary: List[Dict] = []
    runtime_tracker: List[Dict] = []
    seed0_state: Optional[Dict] = None
    LATENT_KEY = "X_cellcharter"

    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        seed_t0 = time.time()

        scvi_latent, cc_latent = _train_scvi_and_aggregate(
            adata=adata, batch_key=args.batch_key,
            n_layers_aggregate=args.n_cc_layers,
            scvi_max_epochs=args.scvi_max_epochs,
            seed=seed, accelerator=args.accelerator,
        )

        leiden_key, n_found, resolution = _leiden_binary_search_on_latent(
            adata, n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors,
            latent_key=LATENT_KEY, seed=seed,
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

        print("\n  -- Batch integration on X_cellcharter --")
        adata.obsm["X_pca"] = adata.obsm[LATENT_KEY]
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
            bint_df["emb_key"] = LATENT_KEY
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
            run_dir=args.out_dir, method="CellCharter-Leiden",
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
            print(f"  X_cellcharter {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 5. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["cellcharter_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["cellcharter_resolution"] = float(seed0_state["resolution"])
        adata.uns["cellcharter_seeds"] = seeds
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
                       "description": "CellCharter (scVI + neighbor-"
                                      "aggregated) + Leiden baseline "
                                      "(multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag": args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "CellCharter-Leiden"},
        "cellcharter": {
            "n_spatial_neighs": int(args.n_spatial_neighs),
            "n_cc_layers": int(args.n_cc_layers),
            "scvi_max_epochs": args.scvi_max_epochs,
            "target_sum": float(args.target_sum),
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
