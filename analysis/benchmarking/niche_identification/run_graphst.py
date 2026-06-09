"""
GraphST baseline for the SQUINT cell-identification benchmark.

GraphST is a graph contrastive method that learns a self-supervised
embedding from spatial transcriptomics data and a per-section spatial
kNN graph. For multi-section integration the authors recommend running
PASTE pairwise alignment first to put sections in a common coordinate
frame; we follow that recipe here.

Pipeline (matches `graphst_benchmarking.ipynb`):
  1. Load + concat silver h5ads.
  2. PASTE alignment (one-time, deterministic):
       - Center-and-rescale each section's spatial coords.
       - `pst.pairwise_align(adata_a, adata_b, alpha=0.05, numItermax=200000)`
       - `pst.stack_slices_pairwise([adata_a, adata_b], [pi])` to land
         sections in a common frame.
  3. Per-batch spatial kNN graph -> block-diag concat.
  4. For each seed:
       - Set torch / numpy seeds.
       - `model = GraphST.GraphST(adata, device=device)`
       - `adata = model.train()` -> `obsm['emb']`
       - Leiden binary search to hit `--n-clusters` on `emb`.
       - UMAP layout.
       - NMI/ARI vs cell_type / niche labels.
       - iLISI / MMD on `emb`.

Output layout mirrors `run_scvi.py`.

Approximate runtime: ~10-20 min per seed on a single GPU (training
dominates). PASTE alignment adds ~5-15 min one-time.

Note on environments: the GraphST notebook used a separate GraphST env
plus a NicheCompass env for downstream metrics. Here we run everything
in one env that has GraphST + paste + scib_metrics installed (caller's
responsibility).

Usage:
    python analysis/benchmarking/niche_identification/run_graphst.py
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


DEFAULT_VARIANT_TAG = "baseline-graphst"


# ---------------------------------------------------------------------------
# PASTE alignment (one-time)
# ---------------------------------------------------------------------------

def _apply_rotation(coords: np.ndarray, angle_deg: float,
                    flip: Optional[str] = None) -> np.ndarray:
    """Rotate `coords` (N,2) by `angle_deg`, then optionally flip an
    axis. Replicates `apply_rotation` from `graphst_benchmarking.ipynb`.
    """
    x, y = coords[:, 0], coords[:, 1]
    theta = np.radians(angle_deg)
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    if flip == "x":
        x_rot = -x_rot
    elif flip == "y":
        y_rot = -y_rot
    return np.column_stack([x_rot, y_rot])


# Per-dataset manual pre-alignment recipes. Each entry maps a
# dataset-tag to a list of (batch-id-substring, dict-of-transforms)
# tuples. Transforms are applied BEFORE the centre+normalise step in
# `_paste_align_pairwise`, putting slices into approximate alignment
# so PASTE's OT solver starts from a near-optimal initialisation and
# converges in <1h instead of running the full numItermax=200_000
# without convergence (the failure mode that took 5+h on mmb-smb
# previously).
#
# To extend to a new dataset, add a new key matching --dataset-tag
# and figure out the per-batch rotations / flips by visually
# inspecting an overlay of the two slices' spatial coords.
PASTE_PREALIGN_RECIPES: Dict[str, List[tuple]] = {
    "mmb0-1b_smb1-1b_1p": [
        # MERFISH: flip y-axis (raw coords have inverted y), then
        # center+normalise (auto), then rotate 180° + flip x.
        ("merfish",   {"flip_pre_norm": "y", "rotate_post_norm": 180,
                       "flip_post_norm": "x"}),
        # STARmap PLUS: just center+normalise + rotate 270°.
        ("starmap",   {"rotate_post_norm": 270}),
    ],
}


def _paste_align_pairwise(
        adata: ad.AnnData,
        batch_key: str,
        paste_alpha: float = 0.05,
        paste_num_iter_max: int = 200_000,
        use_gpu: bool = False,
        dataset_tag: Optional[str] = None,
        prealign_enabled: bool = True,
    ) -> ad.AnnData:
    """Center / rescale each section's spatial coords, optionally
    apply dataset-specific manual pre-alignment, run pairwise PASTE
    alignment, then stack slices into a common frame and concat back
    into one AnnData. Returns the realigned AnnData.

    Reproduces the notebook's recipe (per-batch flip + centre +
    normalise + rotate, then `pst.pairwise_align`, then
    `stack_slices_pairwise`). Operates on >= 2 batches; concatenation
    order is the per-batch first-row order in `obs[batch_key]`.

    The manual pre-alignment is critical for convergence speed:
    without it the OT solver starts from arbitrary relative
    orientations and can fail to converge within `numItermax`
    iterations (~5 hours wasted on mmb-smb in the run that motivated
    this addition). With the recipe, PASTE typically converges in
    well under an hour.
    """
    import paste as pst
    import ot

    if batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={batch_key!r} missing from obs.")
    first_idx = adata.obs.reset_index().groupby(batch_key).head(1).index
    batches = adata.obs.iloc[first_idx][batch_key].tolist()
    if len(batches) < 2:
        print(f"  only {len(batches)} batch(es); skipping PASTE alignment")
        return adata
    print(f"  PASTE: aligning {len(batches)} batch(es): {batches}")

    # Resolve which prealignment recipe (if any) applies to this dataset.
    recipe = (
        PASTE_PREALIGN_RECIPES.get(dataset_tag, [])
        if (prealign_enabled and dataset_tag is not None) else []
    )
    if recipe:
        print(f"  PASTE prealign recipe for dataset_tag={dataset_tag!r}: "
              f"{len(recipe)} per-batch transform(s).")
    else:
        print(f"  PASTE prealign: NONE (dataset_tag={dataset_tag!r} not "
              "in PASTE_PREALIGN_RECIPES, or --no-paste-prealign passed). "
              "Without manual rotation/flip the OT solver may need 5h+ "
              "and still hit numItermax. If PASTE doesn't converge, add "
              "a recipe entry for this dataset.")

    # Split the concatenated AnnData into per-batch pieces.
    pieces: List[ad.AnnData] = []
    for b in batches:
        pieces.append(adata[adata.obs[batch_key] == b].copy())

    def _match_recipe(batch_id) -> dict:
        """Find the per-batch transform entry matching `batch_id` by
        case-insensitive substring."""
        bid = str(batch_id).lower()
        for substr, transforms in recipe:
            if substr.lower() in bid:
                return transforms
        return {}

    # Per-piece prealign + centre + normalise.
    for i, (piece, batch_id) in enumerate(zip(pieces, batches)):
        if "spatial" not in piece.obsm:
            raise SystemExit("obsm['spatial'] missing on a section.")
        coords = np.asarray(piece.obsm["spatial"], dtype=np.float64)
        transforms = _match_recipe(batch_id)
        if transforms:
            print(f"    piece[{i}] (batch={batch_id}): "
                  f"applying {transforms}")
        # Pre-norm flip (e.g. MERFISH y-axis).
        if transforms.get("flip_pre_norm") == "y":
            coords[:, 1] = -coords[:, 1]
        elif transforms.get("flip_pre_norm") == "x":
            coords[:, 0] = -coords[:, 0]
        # Centre + normalise to unit diagonal.
        coords -= coords.mean(axis=0)
        diag = np.sqrt(
            (coords[:, 0].max() - coords[:, 0].min()) ** 2
            + (coords[:, 1].max() - coords[:, 1].min()) ** 2
        )
        if diag > 0:
            coords /= diag
        # Post-norm rotation + flip.
        rot = transforms.get("rotate_post_norm")
        flip = transforms.get("flip_post_norm")
        if rot is not None or flip is not None:
            coords = _apply_rotation(coords, float(rot or 0.0), flip=flip)
        piece.obsm["spatial"] = coords

    # PASTE requires BOTH `use_gpu=True` AND `backend=ot.backend.TorchBackend()`
    # — passing `use_gpu=True` alone falls back to CPU with the message
    # "We currently only have gpu support for Pytorch, please set
    # backend = ot.backend.TorchBackend(). Reverting to selected backend cpu."
    # because the default `ot.backend.NumpyBackend()` has no GPU support.
    # See PASTE source `pairwise_align` lines 60-70.
    if use_gpu:
        backend = ot.backend.TorchBackend()
    else:
        backend = ot.backend.NumpyBackend()

    # Run pairwise alignment between consecutive slices.
    pis = []
    for i in range(len(pieces) - 1):
        print(f"  pst.pairwise_align(pieces[{i}], pieces[{i + 1}], "
              f"alpha={paste_alpha}, numItermax={paste_num_iter_max}, "
              f"use_gpu={use_gpu}, backend={type(backend).__name__})")
        t0 = time.time()
        pi = pst.pairwise_align(
            pieces[i], pieces[i + 1],
            alpha=paste_alpha,
            numItermax=paste_num_iter_max,
            use_gpu=use_gpu,
            backend=backend,
        )
        print(f"    done ({time.time() - t0:.1f}s)")
        pis.append(pi)
    aligned = pst.stack_slices_pairwise(pieces, pis)
    out = ad.concat(aligned, join="inner")
    return out


# ---------------------------------------------------------------------------
# GraphST training (per seed)
# ---------------------------------------------------------------------------

def _train_graphst_and_get_latent(
        adata: ad.AnnData,
        seed: int,
        device: str,
    ) -> np.ndarray:
    """Train GraphST for THIS seed on the (PASTE-aligned, neighbor-
    graph-attached) AnnData. Returns the embedding `obsm['emb']` as a
    NumPy array.

    GraphST seeds via torch / numpy / random globally — we set them
    here and call `GraphST.GraphST(...).train()`.
    """
    import random
    import torch
    from GraphST import GraphST

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    print(f"  GraphST.GraphST(device={device!r})")
    model = GraphST.GraphST(adata, device=device)
    print(f"  training GraphST (seed={seed})...")
    a_out = model.train()
    if "emb" not in a_out.obsm:
        raise RuntimeError(
            "GraphST.train() did not populate obsm['emb']."
        )
    emb = np.asarray(a_out.obsm["emb"])
    # Push the result back into the caller's adata (in place).
    adata.obsm["emb"] = emb
    print(f"  emb shape: {emb.shape}")
    return emb


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
    # GraphST + PASTE hyperparams.
    p.add_argument("--n-spatial-neighs", type=int, default=10)
    p.add_argument("--paste-alpha", type=float, default=0.05)
    p.add_argument("--paste-num-iter-max", type=int, default=50_000,
                   help="POT solver iteration cap inside PASTE. "
                        "Default 50000 (was 200000). Empirically the "
                        "OT solver hits this cap on mmb-smb regardless "
                        "of the value — POT then returns the last "
                        "iterate, which downstream PASTE consumes "
                        "as-is. The numItermax cap therefore controls "
                        "wall-time more than alignment quality once "
                        "the manual prealign recipe is in place. "
                        "Increase if you see poor alignment in the "
                        "saved spatial UMAPs; decrease for faster "
                        "(slightly less optimal) runs.")
    p.add_argument("--paste-use-gpu", action="store_true", default=True,
                   help="Use POT's torch backend on CUDA for PASTE's "
                        "OT solver. Default ON — gives ~5-20x speedup "
                        "over CPU on mmb-smb. Requires a CUDA-enabled "
                        "torch in this venv (otherwise POT falls back "
                        "to CPU silently). Pass --no-paste-use-gpu to "
                        "force CPU.")
    p.add_argument("--no-paste-use-gpu", dest="paste_use_gpu",
                   action="store_false",
                   help="Force CPU for PASTE's OT solver. Useful for "
                        "diagnosing GPU/CUDA issues or when the venv's "
                        "torch wasn't built with CUDA.")
    p.add_argument("--skip-paste", action="store_true",
                   help="Skip PASTE alignment (use raw spatial coords). "
                        "Use when sections are already aligned or when "
                        "PASTE is too expensive.")
    p.add_argument("--no-paste-prealign", action="store_true",
                   help="Disable the dataset-specific manual "
                        "pre-alignment (rotations / flips) before "
                        "PASTE. Pre-alignment puts slices in "
                        "approximate frame so PASTE's OT solver "
                        "converges in <1h vs hitting numItermax "
                        "without convergence (~5h wasted). Default: "
                        "ON when --dataset-tag matches a recipe in "
                        "PASTE_PREALIGN_RECIPES.")
    p.add_argument("--paste-cache-h5ad", type=Path, default=None,
                   help="If set, write the PASTE-aligned AnnData here "
                        "after alignment, and reload from it on the "
                        "next run instead of recomputing PASTE. Saves "
                        "the ~1h alignment cost on every retry. "
                        "Recommended path: <variant_dir>/"
                        "paste_aligned.h5ad")
    p.add_argument("--device", type=str, default="auto",
                   help="GraphST device. 'auto' picks cuda:0 if "
                        "available else cpu.")
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

    if args.device == "auto":
        try:
            import torch
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

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
    print(f"Device  : {args.device}")

    # Pin cwd to a stable local path BEFORE PASTE / GraphST run. The
    # PASTE optimal-transport solver (and torch DDP under GraphST)
    # often cd into a tempdir which is removed at exit, leaving cwd
    # as a stale handle. The next call into squidpy / scanpy.pp.
    # neighbors triggers numba JIT compilation of `_occur_count` etc.;
    # if numba hits any error it tries to format it via
    # `os.path.relpath(self.filename)` which calls os.getcwd() and
    # surfaces the missing cwd as a confusing FileNotFoundError that
    # hides the real numba error. Pinning cwd to /tmp/$USER (already
    # used as TMPDIR) avoids this. NUMBA_CACHE_DIR also pinned so the
    # JIT cache survives across PASTE / training stages.
    import os
    _cwd_anchor = f"/tmp/{os.environ.get('USER', 'user')}"
    os.makedirs(_cwd_anchor, exist_ok=True)
    os.chdir(_cwd_anchor)
    os.environ.setdefault("NUMBA_CACHE_DIR",
                          os.path.join(_cwd_anchor, ".numba_cache"))
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    print(f"cwd     : {os.getcwd()}  (pinned to local /tmp to "
          "avoid stale-cwd FileNotFoundError under numba)")

    # 1. Load + PASTE align (one-time, cacheable) + spatial kNN.
    #    Shared setup timed as `shared_setup_seconds` for apples-to-
    #    apples runtime comparison (see `_record_seed_runtime`
    #    docstring). Per-seed GraphST training is timed inside the loop.
    _shared_t0 = time.time()
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if not args.skip_paste:
        # Reload from cache if --paste-cache-h5ad exists; otherwise run
        # PASTE and (optionally) save for next time.
        if args.paste_cache_h5ad is not None and Path(args.paste_cache_h5ad).is_file():
            print(f"\n=== Loading cached PASTE-aligned AnnData from "
                  f"{args.paste_cache_h5ad} ===")
            adata = ad.read_h5ad(args.paste_cache_h5ad)
            print(f"  n_obs={adata.n_obs}, n_vars={adata.n_vars}  "
                  "(skipping PASTE — cache hit)")
        else:
            print("\n=== PASTE alignment ===")
            adata = _paste_align_pairwise(
                adata, batch_key=args.batch_key,
                paste_alpha=args.paste_alpha,
                paste_num_iter_max=args.paste_num_iter_max,
                use_gpu=args.paste_use_gpu,
                dataset_tag=args.dataset_tag,
                prealign_enabled=not args.no_paste_prealign,
            )
            print(f"After PASTE: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
            if args.paste_cache_h5ad is not None:
                Path(args.paste_cache_h5ad).parent.mkdir(
                    parents=True, exist_ok=True,
                )
                adata.write_h5ad(args.paste_cache_h5ad)
                print(f"  -> cached PASTE result to "
                      f"{args.paste_cache_h5ad}")
    adata = _spatial_knn_per_batch(
        adata, n_neighs=args.n_spatial_neighs, batch_key=args.batch_key,
        include_self_loop=True,
    )
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + PASTE + spatial graph): "
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
    LATENT_KEY = "emb"

    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        seed_t0 = time.time()

        # Defensively re-pin cwd at each seed start (see top-of-main()
        # block for rationale).
        try:
            os.chdir(_cwd_anchor)
        except OSError:
            os.makedirs(_cwd_anchor, exist_ok=True)
            os.chdir(_cwd_anchor)

        # TIMED block: per-seed GraphST training + Leiden binary search.
        # Below `seed_seconds = ...` runs UNTIMED.
        _train_graphst_and_get_latent(
            adata=adata, seed=seed, device=args.device,
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
            run_dir=args.out_dir, method="GraphST-Leiden",
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

        print("\n  -- Batch integration on emb (GraphST) --")
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
            print(f"  emb {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 5. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["graphst_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["graphst_resolution"] = float(seed0_state["resolution"])
        adata.uns["graphst_seeds"] = seeds
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
                       "description": "GraphST + Leiden baseline (multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag": args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "GraphST-Leiden"},
        "graphst": {
            "n_spatial_neighs": int(args.n_spatial_neighs),
            "paste_alpha": float(args.paste_alpha),
            "paste_num_iter_max": int(args.paste_num_iter_max),
            "skip_paste": bool(args.skip_paste),
            "device": args.device,
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
