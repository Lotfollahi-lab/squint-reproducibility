"""
scVI baseline (unsupervised) for the SQUINT cell-type-identification
benchmark.

Sibling of `run_pca_leiden.py` / `run_harmony.py` / `run_scgpt.py`.
scVI learns a batch-corrected latent purely from raw counts, with no
cell-type label supervision. Standard scvi-tools workflow.

Reference workflow (canonical scvi-tools tutorial):
    https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html

Per seed:
  1. `scvi.settings.seed = seed`
  2. `SCVI.setup_anndata(adata, batch_key=…)`  (no labels_key)
  3. Train scVI with default params.
  4. Latent: `vae.get_latent_representation()` -> `adata.obsm['X_scvi']`
  5. Leiden binary search to hit `--n-clusters` on the scVI latent.
  6. UMAP layout.
  7. NMI / ARI on (leiden vs cell_type / niche) — same helper as
     compute_inference_metrics.py.
  8. iLISI / MMD on `X_scvi` vs `obs[batch_key]`.

Output layout mirrors `run_pca_leiden.py` / `run_harmony.py` so
compare_variants.py picks all baselines up identically.

Approximate runtime: ~20-30 min per seed on a single GPU for ~100k-cell
datasets. Five seeds = ~2 hours.

Usage:
    python analysis/benchmarking/cell_type_identification/run_scvi.py
    # smoke-test with a single seed:
    python analysis/benchmarking/cell_type_identification/run_scvi.py --seeds 0
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

# Reuse the shared helpers from the PCA+Leiden script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
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
DEFAULT_VARIANT_TAG = "baseline-scvi"


def _hvg_subset(adata: ad.AnnData, n_top: int, batch_key: str) -> ad.AnnData:
    """HVG selection on raw counts. Uses scanpy's `seurat_v3` flavor,
    which expects raw integer counts and ranks genes by per-batch
    variance — i.e. the canonical scvi-tools tutorial recipe.

    Returns a NEW AnnData restricted to the top-N HVG columns.
    """
    if n_top is None or n_top <= 0 or n_top >= adata.n_vars:
        print(f"  HVG: skipping (n_top={n_top}, n_vars={adata.n_vars}).")
        return adata
    print(f"  HVG: selecting top {n_top} highly-variable genes "
          f"(seurat_v3, batch_key={batch_key!r}).")
    a = adata.copy()
    sc.pp.highly_variable_genes(
        a,
        n_top_genes=int(n_top),
        flavor="seurat_v3",
        batch_key=batch_key if batch_key in a.obs.columns else None,
        subset=True,
    )
    return a


def _train_scvi_and_get_latent(
        adata: ad.AnnData,
        batch_key: str,
        scvi_max_epochs: Optional[int],
        seed: int,
        accelerator: str,
    ) -> np.ndarray:
    """One scVI training run. Returns the latent as a NumPy array of
    shape (n_obs, n_latent)."""
    import scvi
    scvi.settings.seed = int(seed)

    print(f"  setup_anndata(batch_key={batch_key!r})  [unsupervised; no labels_key]")
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)

    print(f"  training scVI (seed={seed})...")
    vae = scvi.model.SCVI(adata)
    vae.train(
        max_epochs=scvi_max_epochs,
        accelerator=accelerator,
        check_val_every_n_epoch=1,
    )
    latent = np.asarray(vae.get_latent_representation())
    print(f"  latent shape: {latent.shape}")
    return latent


def _leiden_binary_search_on_latent(
        adata: ad.AnnData,
        n_clusters: int,
        n_neighbors: int,
        latent_key: str,
        seed: int,
        max_iters: int = 25,
    ) -> tuple:
    """Bisect Leiden resolution to hit `n_clusters` on the given latent
    representation. Builds the kNN graph from `use_rep=latent_key`
    (NOT 'X_pca' as the shared helper does, since here the scVI /
    Harmony latent IS the embedding we want to cluster).

    Returns (leiden_obs_key, n_found, resolution).
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=latent_key,
                    random_state=seed)
    leiden_key = "leiden"
    lo, hi = 0.05, 10.0
    best_n = best_res = best_diff = None
    n_found = 0; resolution = 1.0
    print(f"  Bisecting Leiden resolution to hit {n_clusters} clusters:")
    for it in range(max_iters):
        mid = 0.5 * (lo + hi)
        sc.tl.leiden(adata, resolution=mid, key_added=leiden_key,
                     random_state=seed)
        n_found = int(adata.obs[leiden_key].astype(str).nunique())
        diff = abs(n_found - n_clusters)
        print(f"    iter {it+1:>2d}  resolution={mid:.4f}  -> "
              f"n_clusters={n_found}")
        if best_diff is None or diff < best_diff:
            best_diff = diff; best_n = n_found; best_res = mid
        if n_found == n_clusters:
            return leiden_key, n_found, mid
        if n_found < n_clusters:
            lo = mid
        else:
            hi = mid
    sc.tl.leiden(adata, resolution=best_res, key_added=leiden_key,
                 random_state=seed)
    print(f"  ! exact match not reached; using closest "
          f"(n={best_n}, resolution={best_res:.4f}).")
    return leiden_key, best_n, best_res


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--silver-dir", type=str, default=DEFAULT_SILVER_DIR)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag",  type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--variant-tag",  type=str, default=DEFAULT_VARIANT_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    # scVI hyperparams — defaults match the tutorial.
    p.add_argument("--n-hvg", type=int, default=2000,
                   help="HVG count (seurat_v3, raw counts). Pass 0 to "
                        "disable; default 2000 per scvi-tools tutorial.")
    p.add_argument("--scvi-max-epochs", type=int, default=None,
                   help="Override scVI max_epochs. Default None lets "
                        "scvi-tools auto-pick.")
    p.add_argument("--accelerator", type=str, default="auto")
    # Clustering / metric knobs.
    p.add_argument("--n-clusters", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--cell-label-keys", type=str,
                   default=",".join(DEFAULT_CELL_LABEL_KEYS))
    p.add_argument("--niche-label-keys", type=str,
                   default=",".join(DEFAULT_NICHE_LABEL_KEYS))
    p.add_argument("--batch-key", type=str, default="adata_batch_id")
    p.add_argument("--ilisi-n-neighbors", type=int, default=90)
    p.add_argument("--mmd-n-sub",   type=int, default=2000)
    p.add_argument("--mmd-n-sigma", type=int, default=1000)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    cell_keys  = [k.strip() for k in args.cell_label_keys.split(",")  if k.strip()]
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

    # 1. Load + HVG (one-time).
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(
            f"--batch-key={args.batch_key!r} missing from obs."
        )
    print("\n=== HVG selection ===")
    adata = _hvg_subset(adata, n_top=args.n_hvg, batch_key=args.batch_key)
    print(f"After HVG: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    # 2. Per-seed loop.
    compute_nmi_ari, compute_ilisi, compute_mmd_comparable = (
        _import_metric_helpers()
    )
    per_seed_niche: List[pd.DataFrame] = []
    per_seed_batch: List[pd.DataFrame] = []
    seed_summary: List[Dict] = []
    runtime_tracker: List[Dict] = []
    seed0_state: Optional[Dict] = None
    LATENT_KEY = "X_scvi"

    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        seed_t0 = time.time()

        latent = _train_scvi_and_get_latent(
            adata=adata, batch_key=args.batch_key,
            scvi_max_epochs=args.scvi_max_epochs,
            seed=seed, accelerator=args.accelerator,
        )
        adata.obsm[LATENT_KEY] = latent

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

        print("\n  -- Batch integration on X_scvi --")
        # Alias for the batch-int helper (which reads obsm['X_pca']).
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

        # Per-seed outputs: metric CSVs + UMAP plots into
        # `<run_dir>/seeds/seed_<N>/`.
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
            run_dir=args.out_dir, method="scVI",
        )
        print(f"  runtime (seed {seed}): {seed_seconds:.1f}s")

        if s_idx == 0:
            seed0_state = {"leiden_key": leiden_key,
                           "n_found": n_found, "resolution": resolution}

    # 3. Per-seed long + aggregated mean CSVs.
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
            print(f"  X_scvi {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 5. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["scvi_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["scvi_resolution"] = float(seed0_state["resolution"])
        adata.uns["scvi_seeds"]      = seeds
        _sanitize_for_h5ad(adata).write_h5ad(args.out_dir / "predicted_adata.h5ad")
        print(f"\n  -> {args.out_dir / 'predicted_adata.h5ad'}  (seed[0])")

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

    # 6. Stub config.
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "scVI baseline (unsupervised, multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "scVI"},
        "scvi": {
            "n_hvg":              int(args.n_hvg),
            "scvi_max_epochs":    args.scvi_max_epochs,
            "batch_key":          args.batch_key,
            "n_clusters_target":  int(args.n_clusters),
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
    print(f"  Variant : {args.variant_tag}")
    print(f"  Seeds   : {len(seeds)}  ({seeds})")
    print("=" * 78)


if __name__ == "__main__":
    main()
