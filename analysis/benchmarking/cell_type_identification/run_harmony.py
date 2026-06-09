"""
Harmony batch-correction baseline for the SQUINT cell-identification
benchmark (5 seeds).

Sibling of `run_pca_leiden.py` / `run_scvi.py`. Runs:
  1. log1p-CPM normalisation, scale, PCA (n_comps=50)  ONE-TIME.
  2. Per seed:
       - Harmony correction on the PCA embedding via
         `scanpy.external.pp.harmony_integrate` (uses harmonypy under
         the hood). Output lands in `adata.obsm['X_pca_harmony']`.
       - Leiden binary search on the corrected embedding to hit
         `--n-clusters` (default 30).
       - UMAP layout.
       - NMI / ARI on (leiden vs cell_type / niche) — same helper as
         compute_inference_metrics.py.
       - iLISI / MMD on `X_pca_harmony` vs `obs[batch_key]` — same
         helpers, RNG-seeded subsampling.
  3. Aggregate across seeds → `metrics/{niche,batchint}_metrics.csv`
     (mean) + `metrics/per_seed_*.csv` (long format).

Approximate runtime: ~3-8 minutes per seed (Harmony is fast — kmeans +
linear correction, no NN training). Five seeds = ~30 minutes total.
Way cheaper than scVI — feasible to run interactively.

Reference workflow:
    https://scanpy-tutorials.readthedocs.io/en/latest/integrating-data-using-ingest.html
    (and the Harmony paper, Korsunsky et al. 2019).

Usage:
    python analysis/benchmarking/cell_type_identification/run_harmony.py
    # different dataset:
    python analysis/benchmarking/cell_type_identification/run_harmony.py \\
        --silver-dir /nfs/.../silver/chl59-8b_1p \\
        --dataset-tag chl59-8b_1p
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

# Reuse shared helpers (data loading, metric runners, aggregators,
# plotter) from the PCA+Leiden script.
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
    _preprocess,
    _compute_umap_if_needed,
    _record_seed_runtime,
    _write_per_seed_outputs,
    _write_runtime_csvs,
)
# The Leiden-on-arbitrary-latent helper lives in run_scvi.py.
from run_scvi import _leiden_binary_search_on_latent  # noqa: E402


DEFAULT_VARIANT_TAG = "baseline-harmony"


def _run_harmony(
        adata: ad.AnnData,
        batch_key: str,
        seed: int,
        max_iter_harmony: int = 10,
    ) -> str:
    """Run Harmony correction on `adata.obsm['X_pca']` and write the
    corrected embedding to `adata.obsm['X_pca_harmony']`. Returns the
    obsm key holding the corrected embedding.

    `random_state=seed` controls Harmony's internal kmeans init (the
    main source of cross-run variance for this method).

    Implementation note
    -------------------
    We call `harmonypy.run_harmony` directly rather than going through
    `scanpy.external.pp.harmony_integrate`. The scanpy wrapper assumes
    `harmony_out.Z_corr` is a 2-D NumPy array of shape (n_pcs, n_cells)
    and writes `Z_corr.T` straight into obsm. Recent harmonypy versions
    add a PyTorch / GPU backend whose result object exposes Z_corr in
    a different layout (sometimes as a 1-D torch tensor in some buggy
    intermediate releases) — that mismatches the scanpy assumption and
    fails with a confusing ValueError about obsm shape. Calling
    harmonypy directly lets us defensively cope with both layouts and
    convert torch tensors to NumPy.
    """
    try:
        import harmonypy as hm
    except ImportError as exc:
        raise SystemExit(
            "harmonypy is not installed. Install via "
            "`pip install harmonypy`. Original error: " + str(exc)
        )

    print(f"  Running Harmony (seed={seed}, batch_key={batch_key!r})...")
    pca = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    ho = hm.run_harmony(
        data_mat=pca,
        meta_data=adata.obs,
        vars_use=[batch_key],
        max_iter_harmony=int(max_iter_harmony),
        random_state=int(seed),
    )

    n_obs = adata.n_obs
    n_pcs = pca.shape[1]

    def _to_numpy(x):
        # harmonypy's PyTorch backend may return torch tensors on GPU.
        if hasattr(x, "cpu") and hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    # Pull the corrected embedding from the result object. Different
    # harmonypy versions / backends store it under different attribute
    # names; try each, and accept whichever yields a 2-D matrix whose
    # rows are cells.
    embedding = None
    for attr in ("Z_corr", "result", "harmonized", "Z", "Zhat"):
        if not hasattr(ho, attr):
            continue
        val = getattr(ho, attr)
        if callable(val):
            try:
                val = val()
            except TypeError:
                continue
        try:
            arr = _to_numpy(val)
        except Exception:
            continue
        if arr.ndim != 2:
            continue
        # Match orientation by checking which dim equals n_obs.
        if arr.shape[0] == n_obs and arr.shape[1] == n_pcs:
            embedding = arr
            break
        if arr.shape[1] == n_obs and arr.shape[0] == n_pcs:
            embedding = arr.T
            break
    if embedding is None:
        # Diagnostic dump for the user.
        avail = sorted(a for a in dir(ho) if not a.startswith("_"))
        zc = getattr(ho, "Z_corr", None)
        zc_shape = (
            tuple(_to_numpy(zc).shape) if zc is not None else None
        )
        raise RuntimeError(
            "Could not extract harmony-corrected embedding from the "
            "harmonypy result object. This usually means harmonypy's "
            "backend (PyTorch / CUDA) returned Z_corr in an unexpected "
            "layout. Diagnostics:\n"
            f"  result-object attrs: {avail}\n"
            f"  ho.Z_corr shape:     {zc_shape}\n"
            f"  expected (n_obs={n_obs}, n_pcs={n_pcs}) or its transpose.\n"
            "Workarounds: pin `pip install \"harmonypy==0.0.9\"` (the "
            "last NumPy-only release) or report at "
            "https://github.com/slowkow/harmonypy/issues."
        )

    adata.obsm["X_pca_harmony"] = embedding.astype(np.float32)
    print(f"  X_pca_harmony shape: {adata.obsm['X_pca_harmony'].shape}")
    return "X_pca_harmony"


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
    # Preprocessing / Harmony hyperparams.
    p.add_argument("--n-pcs", type=int, default=50,
                   help="PCA components fed to Harmony.")
    p.add_argument("--max-iter-harmony", type=int, default=10,
                   help="Harmony max correction iterations (harmonypy "
                        "default 10).")
    # Clustering / metric knobs (defaults match run_pca_leiden).
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

    # 1. Load + preprocess + PCA (one-time; PCA with arpack is deterministic).
    #    Timed as `shared_setup_seconds` so the per-seed runtime can fold
    #    it back in for an apples-to-apples comparison (see
    #    `_record_seed_runtime` docstring).
    _shared_t0 = time.time()
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(
            f"--batch-key={args.batch_key!r} missing from obs."
        )
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
    adata = _preprocess(adata, n_pcs=args.n_pcs)
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + preprocess + PCA): {shared_setup_seconds:.1f}s")

    # 2. Per-seed loop: Harmony on X_pca, then Leiden + UMAP + metrics.
    compute_nmi_ari, compute_ilisi, compute_mmd_comparable = (
        _import_metric_helpers()
    )
    per_seed_niche: List[pd.DataFrame] = []
    per_seed_batch: List[pd.DataFrame] = []
    seed_summary: List[Dict] = []
    runtime_tracker: List[Dict] = []
    seed0_state: Optional[Dict] = None
    LATENT_KEY = "X_pca_harmony"

    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        # TIMED block: Harmony correction (per-seed, has stochastic
        # init) + Leiden binary search. Everything below the
        # `seed_seconds = ...` line runs UNTIMED (benchmark
        # scaffolding: UMAP-for-viz, NMI/ARI, iLISI/MMD, plot writes).
        seed_t0 = time.time()
        _run_harmony(
            adata=adata, batch_key=args.batch_key,
            seed=seed, max_iter_harmony=args.max_iter_harmony,
        )
        leiden_key, n_found, resolution = _leiden_binary_search_on_latent(
            adata, n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors,
            latent_key=LATENT_KEY, seed=seed,
        )
        seed_seconds = time.time() - seed_t0
        _record_seed_runtime(
            runtime_tracker, seed=seed,
            local_seconds=seed_seconds,
            shared_setup_seconds=shared_setup_seconds,
            run_dir=args.out_dir, method="Harmony",
        )
        print(f"  runtime (seed {seed}): local={seed_seconds:.1f}s, "
              f"shared={shared_setup_seconds:.1f}s, "
              f"total={seed_seconds + shared_setup_seconds:.1f}s")

        # ---- UNTIMED below: metrics + visualization ---------------------
        _compute_umap_if_needed(adata, random_state=seed)

        print("\n  -- Niche identification --")
        niche_df = _compute_niche_identification(
            adata=adata, leiden_key=leiden_key,
            cell_label_keys=cell_keys, niche_label_keys=niche_keys,
            compute_nmi_ari=compute_nmi_ari,
        )
        if not niche_df.empty:
            niche_df.insert(0, "seed", seed)
        per_seed_niche.append(niche_df)

        print("\n  -- Batch integration on X_pca_harmony --")
        # Alias for the batch-int helper (which reads obsm['X_pca']).
        # We need the HARMONY-CORRECTED embedding here, not the raw PCA.
        adata.obsm["__pca_backup"] = adata.obsm["X_pca"]
        adata.obsm["X_pca"] = adata.obsm[LATENT_KEY]
        bint_df = _compute_batch_integration(
            adata=adata, batch_key=args.batch_key,
            compute_ilisi=compute_ilisi,
            compute_mmd_comparable=compute_mmd_comparable,
            ilisi_n_neighbors=args.ilisi_n_neighbors,
            mmd_n_sub=args.mmd_n_sub, mmd_n_sigma=args.mmd_n_sigma,
            seed=seed,
        )
        # Restore PCA so subsequent seeds re-run Harmony from the same
        # uncorrected starting point.
        adata.obsm["X_pca"] = adata.obsm["__pca_backup"]
        del adata.obsm["__pca_backup"]
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
            print(f"  X_pca_harmony {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 5. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["harmony_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["harmony_resolution"] = float(seed0_state["resolution"])
        adata.uns["harmony_seeds"]      = seeds
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
                       "description": "PCA + Harmony baseline (multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "PCA-Harmony"},
        "harmony": {
            "n_pcs":              int(args.n_pcs),
            "max_iter_harmony":   int(args.max_iter_harmony),
            "batch_key":          args.batch_key,
            "n_clusters_target":  int(args.n_clusters),
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
    print(f"  Variant : {args.variant_tag}")
    print(f"  Seeds   : {len(seeds)}  ({seeds})")
    print("=" * 78)


if __name__ == "__main__":
    main()
