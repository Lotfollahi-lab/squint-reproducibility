"""
Banksy + Harmony + Leiden baseline for the SQUINT cell-identification
benchmark.

Sibling of `run_neigh_expr_pca.py`, but uses the Banksy package
(`banksy.initialize_banksy` + `generate_banksy_matrix` +
`banksy_utils.umap_pca.pca_umap`) to build a niche-aware embedding,
then runs Harmony on top to remove the residual batch effect, and
clusters with Leiden.

Pipeline (matches `banksy_benchmarking.ipynb`):
  1. Load + concat silver h5ads.
  2. Per-batch spatial kNN graph -> block-diag concat (same as the
     NicheCompass / NEPL recipes).
  3. Build Banksy matrix:
        coord_keys = ('spatial_x', 'spatial_y', 'spatial')
        max_m=1, lambda_list=[1.0], pca_dims=[20]
     -> latent at `obsm['banksy_latent']` (the `reduced_pc_20` slot).
  4. Harmonypy on the Banksy latent vs `obs[batch_key]` -> latent at
     `obsm['banksy_latent_harmony']`. ONE-TIME (Banksy + Harmony are
     deterministic given the same input).
  5. For each seed:
       - kNN graph + Leiden binary search to hit `--n-clusters`
       - UMAP layout
       - NMI/ARI vs cell_type / niche labels
       - iLISI / MMD on the Harmony-corrected latent.

Usage:
    python analysis/benchmarking/niche_identification/run_banksy.py
    # smoke test:
    python analysis/benchmarking/niche_identification/run_banksy.py --seeds 0
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
    _compute_umap_if_needed,
    _record_seed_runtime,
    _write_per_seed_outputs,
    _write_runtime_csvs,
)
from run_neigh_expr_pca import _spatial_knn_per_batch  # noqa: E402
from run_scvi import _leiden_binary_search_on_latent  # noqa: E402


DEFAULT_VARIANT_TAG = "baseline-banksy"


# ---------------------------------------------------------------------------
# Banksy embedding + Harmony correction (one-time)
# ---------------------------------------------------------------------------

def _build_banksy_latent(
        adata: ad.AnnData,
        n_neighs: int,
        max_m: int,
        nbr_weight_decay: str,
        lambda_val: float,
        pca_dim: int,
    ) -> np.ndarray:
    """Return the Banksy reduced-PC matrix as a NumPy array.

    Mirrors the notebook recipe:
        banksy_dict = initialize_banksy(adata, coord_keys, n_neighs,
                          nbr_weight_decay=..., max_m=...)
        banksy_dict, _ = generate_banksy_matrix(adata, banksy_dict,
                                                lambda_list=[lambda_val],
                                                max_m=...)
        pca_umap(banksy_dict, pca_dims=[pca_dim], add_umap=True)
        latent = banksy_dict[nbr_weight_decay][lambda_val]['adata'].obsm[
            f'reduced_pc_{pca_dim}'
        ]

    Caller's responsibility to make sure `obsm['spatial']` is populated
    and `obs['spatial_x']` / `obs['spatial_y']` are set.
    """
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy_utils.umap_pca import pca_umap

    coord_keys = ("spatial_x", "spatial_y", "spatial")
    print(f"  initialize_banksy(n_neighs={n_neighs}, max_m={max_m}, "
          f"nbr_weight_decay={nbr_weight_decay!r})")
    banksy_dict = initialize_banksy(
        adata, coord_keys, n_neighs,
        nbr_weight_decay=nbr_weight_decay,
        max_m=max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    print(f"  generate_banksy_matrix(lambda_list=[{lambda_val}], "
          f"max_m={max_m})")
    banksy_dict, _ = generate_banksy_matrix(
        adata, banksy_dict, [lambda_val], max_m,
    )
    print(f"  pca_umap(pca_dims=[{pca_dim}], add_umap=True)")
    pca_umap(
        banksy_dict, pca_dims=[pca_dim], add_umap=True,
        plt_remaining_var=False,
    )
    pc_key = f"reduced_pc_{pca_dim}"
    latent = (
        banksy_dict[nbr_weight_decay][lambda_val]["adata"].obsm[pc_key]
    )
    return np.asarray(latent, dtype=np.float64)


def _harmony_correct(
        latent: np.ndarray,
        obs: pd.DataFrame,
        batch_key: str,
    ) -> np.ndarray:
    """harmonypy.run_harmony with defensive shape extraction. Returns a
    (n_cells, n_dim) NumPy array regardless of which harmonypy version
    is installed (some return Z_corr, some return harmonized, etc.).
    """
    import harmonypy
    print(f"  harmonypy.run_harmony(batch_key={batch_key!r}, "
          f"input shape={latent.shape})")
    res = harmonypy.run_harmony(latent, obs, batch_key)
    Z = None
    for attr in ("Z_corr", "harmonized", "Z", "Zhat"):
        if hasattr(res, attr):
            cand = getattr(res, attr)
            if cand is not None:
                Z = cand
                break
    if Z is None and callable(getattr(res, "result", None)):
        Z = res.result()
    if Z is None:
        raise RuntimeError(
            "Could not extract corrected latent from harmonypy result "
            f"(attrs: {dir(res)})."
        )
    # Some backends return torch tensors.
    try:
        Z = Z.detach().cpu().numpy()
    except Exception:
        pass
    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape[0] != latent.shape[0]:
        # harmonypy historically returned (n_dim, n_cells) -> transpose.
        if Z.shape[1] == latent.shape[0] and Z.shape[0] == latent.shape[1]:
            Z = Z.T
        else:
            raise RuntimeError(
                f"Harmony output shape {Z.shape} incompatible with "
                f"input shape {latent.shape}."
            )
    return Z


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
    # Banksy hyperparams.
    p.add_argument("--n-spatial-neighs", type=int, default=10)
    p.add_argument("--banksy-max-m", type=int, default=1,
                   help="Use both mean (m=0) and AFT (m=1).")
    p.add_argument("--banksy-nbr-weight-decay", type=str,
                   default="scaled_gaussian")
    p.add_argument("--banksy-lambda", type=float, default=1.0)
    p.add_argument("--banksy-pca-dim", type=int, default=20)
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

    # 1. Load + spatial graph + Banksy + Harmony (one-time shared
    #    setup). Timed as `shared_setup_seconds` for apples-to-apples
    #    runtime comparison (see `_record_seed_runtime` docstring).
    _shared_t0 = time.time()
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if "spatial" not in adata.obsm:
        raise SystemExit("obsm['spatial'] missing.")
    # Banksy reads spatial_x / spatial_y from obs.
    adata.obs["spatial_x"] = np.asarray(adata.obsm["spatial"])[:, 0]
    adata.obs["spatial_y"] = np.asarray(adata.obsm["spatial"])[:, 1]
    adata = _spatial_knn_per_batch(
        adata, n_neighs=args.n_spatial_neighs, batch_key=args.batch_key,
        include_self_loop=True,
    )

    print("\n=== Banksy embedding ===")
    banksy_latent = _build_banksy_latent(
        adata, n_neighs=args.n_spatial_neighs,
        max_m=args.banksy_max_m,
        nbr_weight_decay=args.banksy_nbr_weight_decay,
        lambda_val=float(args.banksy_lambda),
        pca_dim=int(args.banksy_pca_dim),
    )
    adata.obsm["banksy_latent"] = banksy_latent
    print(f"  banksy_latent shape: {banksy_latent.shape}")

    print("\n=== Harmony batch correction ===")
    LATENT_KEY = "banksy_latent_harmony"
    adata.obsm[LATENT_KEY] = _harmony_correct(
        banksy_latent, adata.obs, batch_key=args.batch_key,
    )
    print(f"  {LATENT_KEY} shape: {adata.obsm[LATENT_KEY].shape}")
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + spatial graph + Banksy + Harmony): "
          f"{shared_setup_seconds:.1f}s")

    # 2. Per-seed Leiden + metrics on the Harmony-corrected latent.
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
        # TIMED block: Leiden binary search on the shared
        # Banksy+Harmony latent. Below `seed_seconds = ...` runs UNTIMED.
        seed_t0 = time.time()
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
            run_dir=args.out_dir, method="Banksy-Harmony-Leiden",
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

        print("\n  -- Batch integration on banksy_latent_harmony --")
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
            print(f"  banksy_latent_harmony {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 5. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["banksy_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["banksy_resolution"] = float(seed0_state["resolution"])
        adata.uns["banksy_seeds"] = seeds
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
                       "description": "Banksy + Harmony + Leiden baseline "
                                      "(multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag": args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "Banksy-Harmony-Leiden"},
        "banksy": {
            "n_spatial_neighs": int(args.n_spatial_neighs),
            "max_m": int(args.banksy_max_m),
            "nbr_weight_decay": args.banksy_nbr_weight_decay,
            "lambda": float(args.banksy_lambda),
            "pca_dim": int(args.banksy_pca_dim),
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
