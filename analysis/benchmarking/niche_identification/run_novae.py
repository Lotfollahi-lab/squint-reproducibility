"""
Novae (zero-shot) baseline for the SQUINT cell-identification benchmark.

Sibling of the other spatial-method baselines, but uses the pretrained
`MICS-Lab/novae-mouse-0` foundation model from the Novae package via
its `from_pretrained` API. Pipeline:

  1. Load + concat silver h5ads.
  2. `novae.spatial_neighbors(adata, n_neighs=11, delaunay=False)` to
     populate the spatial graph that Novae's MPNN encoder needs.
     Then overwrite with our per-batch block-diag squidpy kNN graph
     (matches the notebook's "construct_neighbor_graph" call).
  3. `model = novae.Novae.from_pretrained('MICS-Lab/novae-mouse-0')`
     ONE-TIME (the pretrained checkpoint is fixed).
  4. For each seed:
       - Set torch / numpy seeds.
       - `model.compute_representations(adata, zero_shot=True)`
         (the only documented stochastic step; per the notebook,
         "this is not reproducible" so per-seed variance comes from
         this call).
       - `model.assign_domains(adata, level=<n_clusters>)` -> domains
         at `obs[f'novae_domains_{level}']` (used as the cluster key
         for niche identification metrics).
       - Per-batch split + `model.batch_effect_correction(adatas,
         obs_key=domain_key)` to produce `obsm['novae_latent_corrected']`.
         Concat back.
       - UMAP layout on the corrected latent.
       - NMI/ARI vs cell_type / niche labels using `novae_domains_<L>`
         as the cluster column (matches the notebook).
       - iLISI / MMD on `novae_latent_corrected` vs `obs[batch_key]`.

Output layout mirrors `run_scvi.py` so compare_variants.py picks this
baseline up identically.

Note: Novae's `_correct._domains_counts_per_slide` and
`batch_effect_correction` are monkey-patched here to fix
duplicate-index handling and drop NA / zero-only domains, matching
the patches used in the original benchmarking notebook.

Usage:
    python analysis/benchmarking/niche_identification/run_novae.py
    # smoke test:
    python analysis/benchmarking/niche_identification/run_novae.py --seeds 0
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


DEFAULT_VARIANT_TAG = "baseline-novae"


# ---------------------------------------------------------------------------
# Novae monkey-patches (match the notebook).
# ---------------------------------------------------------------------------

def _apply_novae_monkey_patches() -> None:
    """Patch `novae.utils.correct._domains_counts_per_slide` (drop
    duplicate slide IDs) and `batch_effect_correction` (drop NA / zero
    domains) so the multi-batch correction works on real-world data
    where a domain may exist in only one batch.

    This duplicates the notebook's patches verbatim. Idempotent.
    """
    import novae.utils as _utils  # noqa: F401
    import novae.utils.correct as _correct
    from novae._constants import Keys

    # 1. duplicate-index defence.
    if not getattr(_correct._domains_counts_per_slide, "_squint_patched", False):
        _orig_dcps = _correct._domains_counts_per_slide

        def _fixed_dcps(*args, **kwargs):
            result = _orig_dcps(*args, **kwargs)
            result = result.loc[~result.index.duplicated(keep="first")]
            return result
        _fixed_dcps._squint_patched = True
        _correct._domains_counts_per_slide = _fixed_dcps

    # 2. drop NA / zero-only domains in batch_effect_correction.
    if not getattr(_correct.batch_effect_correction, "_squint_patched", False):
        def _patched_batch_effect_correction(adatas, obs_key):
            for adata in adatas:
                assert obs_key in adata.obs
                assert Keys.REPR in adata.obsm

            adata_indices, slides_obs_indices = _correct._slides_indices(adatas)
            domains_counts_per_slide = _correct._domains_counts_per_slide(
                adatas, obs_key
            )

            domains = domains_counts_per_slide.columns[:-1]
            valid_mask = domains_counts_per_slide[domains].notna().any(axis=0)
            domains = domains[valid_mask]
            valid_mask2 = (domains_counts_per_slide[domains] > 0).any(axis=0)
            domains = domains[valid_mask2]

            ref_slide_ids = domains_counts_per_slide[domains].idxmax(axis=0)

            def _centroid_reference(domain, slide_id, obs_key):
                adata_ref_index = (
                    domains_counts_per_slide[Keys.ADATA_INDEX].loc[slide_id]
                )
                if isinstance(adata_ref_index, pd.Series):
                    adata_ref_index = int(adata_ref_index.iloc[0])
                else:
                    adata_ref_index = int(adata_ref_index)
                adata_ref = adatas[adata_ref_index]
                where = (
                    (adata_ref.obs[Keys.SLIDE_ID] == slide_id)
                    & (adata_ref.obs[obs_key] == domain)
                )
                return adata_ref.obsm[Keys.REPR][where].mean(0)

            centroids_reference = pd.DataFrame({
                domain: _centroid_reference(domain, slide_id, obs_key)
                for domain, slide_id in ref_slide_ids.items()
            })

            for adata in adatas:
                adata.obsm[Keys.REPR_CORRECTED] = adata.obsm[Keys.REPR].copy()

            for adata_index, obs_indices in zip(adata_indices, slides_obs_indices):
                adata = adatas[adata_index]
                for domain in domains:
                    if (adata.obs[Keys.SLIDE_ID].iloc[obs_indices[0]]
                            == ref_slide_ids.loc[domain]):
                        continue
                    indices_domain = obs_indices[
                        adata.obs.iloc[obs_indices][obs_key] == domain
                    ]
                    if len(indices_domain) == 0:
                        continue
                    centroid_reference = centroids_reference[domain].values
                    centroid = adata.obsm[Keys.REPR][indices_domain].mean(0)
                    adata.obsm[Keys.REPR_CORRECTED][indices_domain] += (
                        centroid_reference - centroid
                    )
        _patched_batch_effect_correction._squint_patched = True
        _correct.batch_effect_correction = _patched_batch_effect_correction
        _utils.batch_effect_correction = _patched_batch_effect_correction


# ---------------------------------------------------------------------------
# Per-seed Novae embedding + domain assignment + batch correction
# ---------------------------------------------------------------------------

def _novae_embed_assign_correct(
        model,
        adata: ad.AnnData,
        domain_level: int,
        batch_key: str,
        seed: int,
    ) -> tuple[str, str]:
    """Run THIS seed's pass through the Novae model:
        compute_representations(zero_shot=True) ->
        assign_domains(level=domain_level) ->
        per-batch split + batch_effect_correction.

    Returns (latent_key, domain_key) where:
        - latent_key  = 'novae_latent_corrected' (the batch-corrected latent)
        - domain_key  = f'novae_domains_{domain_level}' (the cluster col)
    Both are populated in `adata` (in place; concat back from the per-
    batch split).
    """
    import random
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    print(f"  model.compute_representations(zero_shot=True)  (seed={seed})")
    model.compute_representations(adata, zero_shot=True)

    print(f"  model.assign_domains(level={domain_level})")
    model.assign_domains(adata, level=domain_level)

    domain_key = f"novae_domains_{domain_level}"
    if domain_key not in adata.obs.columns:
        # Older Novae versions used a different naming convention; try
        # to detect.
        cands = [c for c in adata.obs.columns if "novae_domains" in c]
        if cands:
            domain_key = cands[0]
            print(f"  (auto-detected domain_key={domain_key!r})")
        else:
            raise RuntimeError(
                "model.assign_domains did not populate any 'novae_domains_*' "
                "obs column."
            )

    print(f"  per-batch split + model.batch_effect_correction("
          f"obs_key={domain_key!r})")
    batches = adata.obs[batch_key].unique().tolist()
    adatas = [adata[adata.obs[batch_key] == b].copy() for b in batches]
    model.batch_effect_correction(adatas, obs_key=domain_key)
    merged = ad.concat(adatas)
    # Bring latent_corrected + domain back into the original adata,
    # keyed by obs_names.
    for obsm_key in ("novae_latent_corrected",):
        if obsm_key in merged.obsm:
            adata.obsm[obsm_key] = (
                merged.obsm[obsm_key]
                if list(merged.obs_names) == list(adata.obs_names)
                else _reindex_obsm(merged, adata, obsm_key)
            )
    if domain_key in merged.obs.columns:
        if list(merged.obs_names) == list(adata.obs_names):
            adata.obs[domain_key] = merged.obs[domain_key].values
        else:
            adata.obs[domain_key] = (
                merged.obs[domain_key].reindex(adata.obs_names).values
            )
    return "novae_latent_corrected", domain_key


def _reindex_obsm(src: ad.AnnData, dst: ad.AnnData, key: str) -> np.ndarray:
    """Re-order `src.obsm[key]` to match `dst.obs_names`. Fills missing
    rows with zeros (shouldn't happen in practice — kept defensive)."""
    n = dst.n_obs
    d = src.obsm[key].shape[1]
    out = np.zeros((n, d), dtype=src.obsm[key].dtype)
    src_pos = pd.Index(src.obs_names).get_indexer(dst.obs_names)
    valid = src_pos >= 0
    out[valid] = src.obsm[key][src_pos[valid]]
    return out


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
    # Novae hyperparams.
    p.add_argument("--novae-checkpoint", type=str,
                   default="MICS-Lab/novae-mouse-0")
    p.add_argument("--novae-spatial-n-neighs", type=int, default=11,
                   help="Spatial graph used by Novae's MPNN encoder. "
                        "Default 11 (notebook).")
    p.add_argument("--n-spatial-neighs", type=int, default=10,
                   help="Squidpy spatial kNN size for the post-Novae "
                        "block-diag concat. Default 10.")
    p.add_argument("--domain-level", type=int, default=30,
                   help="Novae assign_domains level (= target cluster "
                        "count). Default 30 (matches the SQUINT cell "
                        "codebook size).")
    # Clustering / metric knobs.
    p.add_argument("--n-clusters", type=int, default=30,
                   help="Reported alongside domain-level for parity "
                        "with sibling scripts; not used for binary "
                        "search (Novae picks the count).")
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

    # Pin the script's cwd to a stable local path BEFORE any heavy
    # subprocesses run. Without this, `model.compute_representations`
    # (and the torch DDP launcher under it) can cd into a tempdir that
    # gets cleaned up at exit, leaving cwd as a stale handle for the
    # next seed. Subsequent calls to sc.pp.neighbors trigger
    # pynndescent → numba compilation; if numba hits ANY error
    # (e.g. ConstantInferenceError) and tries to format it via
    # `os.path.relpath(self.filename)`, the missing cwd surfaces as a
    # confusing FileNotFoundError that hides the real numba error.
    # Pinning cwd to /tmp/$USER (already created by the runner shell
    # script) avoids this entirely. We also pin NUMBA_CACHE_DIR to
    # local disk so the JIT cache survives across seeds.
    import os
    _cwd_anchor = f"/tmp/{os.environ.get('USER', 'user')}"
    os.makedirs(_cwd_anchor, exist_ok=True)
    os.chdir(_cwd_anchor)
    os.environ.setdefault("NUMBA_CACHE_DIR",
                          os.path.join(_cwd_anchor, ".numba_cache"))
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    print(f"cwd     : {os.getcwd()}  (pinned to local /tmp to "
          "avoid stale-cwd FileNotFoundError under numba)")

    # 1. Load + Novae spatial graph + per-batch squidpy kNN +
    #    pretrained model load (one-time shared setup). Timed as
    #    `shared_setup_seconds` for apples-to-apples runtime comparison
    #    (see `_record_seed_runtime` docstring). Per-seed Novae
    #    forward/assign is timed inside the loop below.
    import novae

    _apply_novae_monkey_patches()

    _shared_t0 = time.time()
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from obs.")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    print(f"\n  novae.spatial_neighbors(n_neighs={args.novae_spatial_n_neighs}, "
          f"delaunay=False)")
    novae.spatial_neighbors(
        adata, n_neighs=args.novae_spatial_n_neighs,
        radius=None, delaunay=False,
    )
    # Overwrite with the symmetric per-batch block-diag squidpy graph
    # (matches the notebook).
    adata = _spatial_knn_per_batch(
        adata, n_neighs=args.n_spatial_neighs, batch_key=args.batch_key,
        include_self_loop=True,
    )

    print(f"\n=== Loading Novae model: {args.novae_checkpoint} ===")
    model = novae.Novae.from_pretrained(args.novae_checkpoint)
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + spatial graph + Novae model): "
          f"{shared_setup_seconds:.1f}s")

    # 2. Per-seed loop.
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
        seed_t0 = time.time()

        # Defensively re-pin cwd at the start of each seed iteration
        # in case `compute_representations` or its DDP launcher cd'd
        # into a tempdir that's now gone. See the cwd-anchor block at
        # the top of main() for the rationale.
        try:
            os.chdir(_cwd_anchor)
        except OSError:
            os.makedirs(_cwd_anchor, exist_ok=True)
            os.chdir(_cwd_anchor)

        # TIMED block: per-seed Novae forward + assign_domains. The
        # domains ARE the clusters (Novae does its own clustering, no
        # Leiden needed). Below `seed_seconds = ...` runs UNTIMED.
        latent_key, domain_key = _novae_embed_assign_correct(
            model=model, adata=adata,
            domain_level=int(args.domain_level),
            batch_key=args.batch_key, seed=seed,
        )
        seed_seconds = time.time() - seed_t0
        n_found = int(adata.obs[domain_key].astype(str).nunique())
        print(f"Novae assign_domains: {n_found} domains "
              f"(level={args.domain_level}).")
        _record_seed_runtime(
            runtime_tracker, seed=seed,
            local_seconds=seed_seconds,
            shared_setup_seconds=shared_setup_seconds,
            run_dir=args.out_dir, method="Novae",
        )
        print(f"  runtime (seed {seed}): local={seed_seconds:.1f}s, "
              f"shared={shared_setup_seconds:.1f}s, "
              f"total={seed_seconds + shared_setup_seconds:.1f}s")

        # ---- UNTIMED below: visualization (neighbors+UMAP are for the
        # notebook's post-correction UMAP only, not for clustering) +
        # metrics + plot writes ------------------------------------------
        sc.pp.neighbors(adata, n_neighbors=args.n_neighbors,
                        use_rep=latent_key, random_state=seed)
        _compute_umap_if_needed(adata, random_state=seed)

        print("\n  -- Niche identification (using Novae domains) --")
        # Novae assigns its own clusters; we feed them in the same role
        # the Leiden binary-search uses for the other baselines.
        niche_df = _compute_niche_identification(
            adata=adata, leiden_key=domain_key,
            cell_label_keys=cell_keys, niche_label_keys=niche_keys,
            compute_nmi_ari=compute_nmi_ari,
        )
        if not niche_df.empty:
            niche_df.insert(0, "seed", seed)
        per_seed_niche.append(niche_df)

        print(f"\n  -- Batch integration on {latent_key} --")
        adata.obsm["X_pca"] = adata.obsm[latent_key]
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
            bint_df["emb_key"] = latent_key
            bint_df.insert(0, "seed", seed)
        per_seed_batch.append(bint_df)

        seed_summary.append({
            "seed": seed,
            "novae_domains": int(n_found),
            "domain_level": int(args.domain_level),
        })

        seed_dir = _write_per_seed_outputs(
            seed=seed, run_dir=args.out_dir, adata=adata,
            leiden_key=domain_key, niche_df=niche_df,
            batch_df=bint_df,
            cell_keys=cell_keys, niche_keys=niche_keys,
            batch_key=args.batch_key, dpi=args.dpi,
        )
        print(f"  -> wrote per-seed outputs to {seed_dir}")

        if s_idx == 0:
            seed0_state = {"leiden_key": domain_key,
                           "n_found": n_found,
                           "domain_level": int(args.domain_level),
                           "latent_key": latent_key}

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
                print(f"  novae vs {label:<10s} {col} = "
                      f"{vals.mean():.4f} ± {std:.4f}  "
                      f"(min={vals.min():.4f}, max={vals.max():.4f})")
    if not long_batch.empty:
        for metric in ("iLISI", "MMD"):
            sub = long_batch[long_batch["metric"] == metric]
            if sub.empty: continue
            vals = sub["score"].astype(float)
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(f"  novae_latent_corrected {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 5. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["novae_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["novae_domain_level"] = int(seed0_state["domain_level"])
        adata.uns["novae_seeds"] = seeds
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
                       "description": "Novae (zero-shot) + batch correction "
                                      "baseline (multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag": args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "Novae"},
        "novae": {
            "checkpoint": args.novae_checkpoint,
            "novae_spatial_n_neighs": int(args.novae_spatial_n_neighs),
            "n_spatial_neighs": int(args.n_spatial_neighs),
            "domain_level": int(args.domain_level),
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
