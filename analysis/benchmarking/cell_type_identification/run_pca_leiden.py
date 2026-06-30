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
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
# Niche / cell-type label keys searched in obs by NMI / ARI helpers.
# These are PREFERENCE LISTS — `compute_nmi_ari` skips any key that's
# absent from `adata.obs`, so adding spatch's `annotation` /
# `spatial_cluster` here is harmless for mmb / chl59 (whose silver
# files don't carry those columns) and rescues the spatch tissue
# subsets, whose silver files use this naming convention instead of
# `cell_type` / `niche`.
DEFAULT_CELL_LABEL_KEYS  = ["cell_type", "cell_types", "annotation"]
DEFAULT_NICHE_LABEL_KEYS = ["niche", "Sub_molecular_tissue_region",
                            "ccf_region_name", "spatial_cluster"]


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

# ---------------------------------------------------------------------------
# Leiden timing singleton + optional rapids-singlecell GPU dispatch
# ---------------------------------------------------------------------------
#
# The two shared Leiden helpers (`_leiden_n_clusters` here and
# `_leiden_binary_search_on_latent` in run_scvi.py) both update this
# singleton on every call. `_record_seed_runtime` reads it to emit a
# `leiden_seconds` column in the per-seed runtime CSV — so the cost of
# Leiden binary-search is broken out from the broader "model fit +
# clustering" `local_seconds` envelope. Callers don't need to thread
# anything through; the most-recent Leiden timing is grabbed
# automatically when they call `_record_seed_runtime`.
#
# Caveat: this is a single mutable cell. If two runners both call
# Leiden between consecutive `_record_seed_runtime` calls, only the
# last one's timing is captured. In practice every runner is
# "Leiden -> record" in lockstep, so this is safe.
_LAST_LEIDEN_SECONDS: List[float] = [0.0]
# Parallel singleton for the UMAP phase. The rapids pipeline bundles
# `pp.neighbors + tl.leiden + tl.umap` in one GPU pass; UMAP is for
# visualisation only and must NOT be counted in the per-seed runtime.
# Rapids helpers (subprocess / inline) set this when they run
# `rsc.tl.umap`; the CPU scanpy path leaves it 0 because UMAP is
# called by the runner OUTSIDE the leiden helper there.
# `_record_seed_runtime` subtracts this from `local_seconds` so the
# recorded clustering-only runtime is consistent across backends.
_LAST_UMAP_SECONDS:   List[float] = [0.0]


def _record_last_leiden_seconds(elapsed: float) -> None:
    """Update the shared Leiden-timing singleton. Called by both
    Leiden helpers."""
    _LAST_LEIDEN_SECONDS[0] = float(elapsed)


def _get_last_leiden_seconds() -> float:
    """Return the elapsed seconds of the most recent Leiden binary
    search. Used by `_record_seed_runtime` to emit the
    `leiden_seconds` column."""
    return _LAST_LEIDEN_SECONDS[0]


def _record_last_umap_seconds(elapsed: float) -> None:
    """Update the shared UMAP-timing singleton. Rapids helpers set
    this when they run `rsc.tl.umap` as part of the GPU pipeline."""
    _LAST_UMAP_SECONDS[0] = float(elapsed)


def _get_last_umap_seconds() -> float:
    """Return the elapsed seconds of the most recent UMAP phase
    inside the leiden helper. Zero when CPU scanpy is used (UMAP
    isn't part of the helper there)."""
    return _LAST_UMAP_SECONDS[0]


def _reset_last_phase_timings() -> None:
    """Zero ALL phase timers at the top of a leiden-helper call.
    Defensive: prevents a previous seed's timings from leaking into
    the current call when an inner helper exits early."""
    _LAST_LEIDEN_SECONDS[0]           = 0.0
    _LAST_UMAP_SECONDS[0]             = 0.0
    _LAST_NEIGHBORS_SECONDS[0]        = 0.0
    _LAST_LEIDEN_ONE_ITER_SECONDS[0]  = 0.0


# Fine-grained phase timers added to support multiple runtime
# reporting flavours (cluster-total / model-only / model+neighbors+
# one-leiden). All four are populated by every leiden helper —
# whether CPU scanpy, inline rapids, or subprocess rapids — so
# `_record_seed_runtime` can compute three different runtime metrics
# without per-runner changes.
_LAST_NEIGHBORS_SECONDS:       List[float] = [0.0]
_LAST_LEIDEN_ONE_ITER_SECONDS: List[float] = [0.0]


def _record_last_neighbors_seconds(elapsed: float) -> None:
    """Update the neighbors-phase timing (pp.neighbors only)."""
    _LAST_NEIGHBORS_SECONDS[0] = float(elapsed)


def _get_last_neighbors_seconds() -> float:
    return _LAST_NEIGHBORS_SECONDS[0]


def _record_last_leiden_one_iter_seconds(elapsed: float) -> None:
    """Mean cost of ONE Leiden iteration during the binary search.
    Used to compute the 'model + neighbors + ONE leiden' runtime
    flavour (excludes the binary-search overhead, which is a
    benchmarking artifact for landing on a target cluster count)."""
    _LAST_LEIDEN_ONE_ITER_SECONDS[0] = float(elapsed)


def _get_last_leiden_one_iter_seconds() -> float:
    return _LAST_LEIDEN_ONE_ITER_SECONDS[0]


# Env var: when set to "rapids" (or "gpu"), Leiden binary-search uses
# rapids-singlecell (`rsc.pp.neighbors` + `rsc.tl.leiden`) directly on
# the parent's adata — the API is identical to scanpy so no extra
# plumbing is needed. The default ("scanpy" / unset) uses CPU scanpy.
#
# REQUIRES rapids-singlecell to be importable in whatever venv is
# running the baseline (we don't shell out to a separate env). Easiest
# way to satisfy that:
#   * Run the baselines from the rapids-singlecell conda env directly
#     (`conda activate /nfs/team361/sb75/ENVS/rapids-singlecell`), OR
#   * `pip install rapids-singlecell` into each baseline's venv
#     (check cupy / cuML wheel compatibility with the venv's torch).
#
# The wrapper scripts set this env var when `--rapids-leiden` is passed.
_LEIDEN_BACKEND_ENV_VAR = "SQUINT_LEIDEN_BACKEND"


def _leiden_backend() -> str:
    """Return the configured Leiden backend: 'scanpy' (default) or
    'rapids'. Reads `SQUINT_LEIDEN_BACKEND`. Anything unrecognised
    raises so a typo doesn't silently fall back to scanpy."""
    raw = os.environ.get(_LEIDEN_BACKEND_ENV_VAR, "").strip().lower()
    if raw in ("", "scanpy", "cpu"):
        return "scanpy"
    if raw in ("rapids", "rapids-singlecell", "gpu"):
        return "rapids"
    raise SystemExit(
        f"{_LEIDEN_BACKEND_ENV_VAR}={raw!r} is not recognised. "
        f"Valid: 'scanpy' (default) | 'rapids'."
    )


# Subprocess mode: when this env var is set to a bash command (typically
# `source /etc/profile.d/modules.sh && module load cellgen/conda &&
# conda activate /nfs/.../rapids-singlecell`), Leiden runs in a
# subprocess inside that env via `_leiden_rapids_worker.py`. Use this
# when rapids-singlecell isn't installed in the baseline's own venv
# but exists as a separate conda env on the cluster.
#
# The two rapids env vars compose:
#   SQUINT_LEIDEN_RAPIDS_ENV_SETUP=<cmd>  ALONE       -> subprocess
#   SQUINT_LEIDEN_BACKEND=rapids          ALONE       -> inline
#   BOTH                                              -> subprocess (env-setup wins)
#   NEITHER                                           -> CPU scanpy
_LEIDEN_RAPIDS_ENV_SETUP_VAR = "SQUINT_LEIDEN_RAPIDS_ENV_SETUP"


def _rapids_leiden_setup_cmd() -> Optional[str]:
    """Return the bash command that activates the rapids-singlecell
    env for the Leiden subprocess, or None when the env-var is unset
    / empty."""
    cmd = os.environ.get(_LEIDEN_RAPIDS_ENV_SETUP_VAR, "")
    return cmd.strip() if cmd and cmd.strip() else None


def _run_leiden_binary_search_rapids_subprocess(
        adata: ad.AnnData,
        n_clusters: int,
        n_neighbors: int,
        max_iters: int,
        rng_seed: int,
        use_rep: str,
        env_setup_cmd: str,
    ) -> Tuple[str, int, float]:
    """Leiden binary-search in a subprocess inside a separate
    rapids-singlecell env. Use this when rapids isn't installed in
    the baseline's own venv but exists as a sibling conda env.

    Writes the embedding to a tempdir, spawns
    `bash -lc "<env_setup_cmd> && python _leiden_rapids_worker.py ..."`,
    reads back the chosen cluster assignments + (n_found, resolution).

    Post-condition (mirrors the CPU + inline-rapids paths):
      * `adata.obs["leiden"]` populated with the best cluster assignment.
      * `adata.uns["neighbors"]` + `adata.obsp["connectivities"]` +
        `adata.obsp["distances"]` populated with the GPU-built kNN
        graph (rapids-singlecell's pp.neighbors output, serialised
        through the tempdir and loaded back here). This avoids the
        CPU `sc.pp.neighbors` recomputation that an earlier version
        of this helper did — for large datasets where neighbors is
        the new bottleneck after rapids eliminates the Leiden-
        iterations cost, this saves another O(N log N) build.
        If the worker's graph artifacts aren't on disk (older
        workers, partial failure), the parent falls back to CPU
        `sc.pp.neighbors` so the downstream UMAP step still works.
    """
    if use_rep not in adata.obsm:
        raise RuntimeError(
            f"_run_leiden_binary_search_rapids_subprocess: "
            f"adata.obsm[{use_rep!r}] missing. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    # Stage the embedding on the parent under the canonical name
    # `X_emb` BEFORE serialising. The worker uses `obsm['X_emb']`
    # internally, so this guarantees parent and worker share the
    # same name — any downstream code that reads
    # `uns['neighbors']['params']['use_rep']` (e.g. sc.tl.umap)
    # finds the embedding on the parent's adata under the same key
    # the worker recorded. No-op when use_rep is already 'X_emb'
    # (e.g. when a previous Leiden call left the alias in place).
    if use_rep != "X_emb":
        adata.obsm["X_emb"] = adata.obsm[use_rep]
    canonical_rep = "X_emb"
    worker_py = Path(__file__).resolve().parent / "_leiden_rapids_worker.py"
    if not worker_py.is_file():
        raise RuntimeError(
            f"_leiden_rapids_worker.py not found at {worker_py}. "
            f"Add it alongside run_pca_leiden.py."
        )
    work = Path(tempfile.mkdtemp(prefix="leiden_rapids_"))
    try:
        np.save(work / "embedding.npy",
                np.asarray(adata.obsm[canonical_rep], dtype=np.float32))
        (work / "config.json").write_text(json.dumps({
            "n_neighbors": int(n_neighbors),
            "max_iters":   int(max_iters),
            "rng_seed":    int(rng_seed),
            "n_clusters":  int(n_clusters),
        }))
        # `bash -lc` so login-shell init (sourced by the user's
        # `source /etc/profile.d/modules.sh`) is honoured. `set -e`
        # makes a failed `conda activate` surface as a non-zero exit
        # before reaching python.
        cmd = (
            f"set -e; {env_setup_cmd} && "
            f"python {worker_py} --workdir {work}"
        )
        print(f"  [Leiden][rapids-subprocess] env: {env_setup_cmd!r}")
        proc = subprocess.run(
            ["bash", "-lc", cmd], capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail_out = "\n".join(proc.stdout.splitlines()[-40:])
            tail_err = "\n".join(proc.stderr.splitlines()[-40:])
            raise RuntimeError(
                "rapids Leiden subprocess failed.\n"
                f"--- stdout (tail) ---\n{tail_out}\n"
                f"--- stderr (tail) ---\n{tail_err}"
            )
        for line in proc.stdout.splitlines():
            print(f"  [Leiden][rapids-subprocess] {line}")
        clusters = np.load(work / "clusters.npy", allow_pickle=True)
        result = json.loads((work / "result.json").read_text())
        adata.obs["leiden"] = pd.Categorical(
            np.asarray(clusters).astype(str)
        )
        # Pull the worker's per-phase timings into the parent's
        # singletons. All four are populated so `_record_seed_runtime`
        # can compute the three runtime flavours (cluster total /
        # model-only / model + neighbors + one-leiden).
        _record_last_neighbors_seconds(
            float(result.get("neighbors_seconds", 0.0))
        )
        _record_last_leiden_seconds(
            float(result.get("leiden_seconds", 0.0))
        )
        _record_last_leiden_one_iter_seconds(
            float(result.get("leiden_one_iter_seconds", 0.0))
        )
        _record_last_umap_seconds(
            float(result.get("umap_seconds", 0.0))
        )

        # Paste the rapids-GPU-built kNN graph onto the parent's main
        # adata. The worker dumped:
        #   * connectivities.npz, distances.npz — scipy sparse CSRs.
        #   * neighbors_uns.json — the uns['neighbors'] metadata dict
        #     (params, connectivities_key, distances_key).
        # Loading these directly avoids the CPU sc.pp.neighbors
        # recomputation that this helper used to do.
        conn_path = work / "connectivities.npz"
        dist_path = work / "distances.npz"
        # `uns_path` (neighbors_uns.json) is informational only — we
        # don't trust its `use_rep` field. The worker's minimal AnnData
        # stored the embedding at obsm['X_emb'], so the JSON would
        # claim use_rep='X_emb'; the parent's real adata has it at
        # obsm[<our use_rep>] (e.g. 'X_pca'), which is what scanpy's
        # `sc.tl.umap` will try to look up. We instead reconstruct the
        # uns dict from the parameters the parent actually knows.
        if conn_path.is_file() and dist_path.is_file():
            from scipy.sparse import load_npz
            adata.obsp["connectivities"] = load_npz(conn_path)
            adata.obsp["distances"]      = load_npz(dist_path)
            # Build the uns dict on the parent side using the REAL
            # adata's parameters. `method='umap'` is the canonical
            # value scanpy looks for to confirm the graph is
            # UMAP-compatible — without it `sc.tl.umap` warns
            # ".obsp['connectivities'] have not been computed using
            # umap" and falls through to recomputing from scratch
            # (which then fails because use_rep doesn't match).
            adata.uns["neighbors"] = {
                "connectivities_key": "connectivities",
                "distances_key":      "distances",
                "params": {
                    "n_neighbors":  int(n_neighbors),
                    "method":       "umap",
                    "random_state": int(rng_seed),
                    # Standardised on `X_emb` so the parent and the
                    # subprocess worker share the same key. We aliased
                    # `obsm[use_rep] -> obsm['X_emb']` above, so this
                    # lookup will resolve correctly downstream.
                    "use_rep":      canonical_rep,
                },
            }
            # Also load the GPU-computed UMAP coords if the worker
            # dumped them. The runner's downstream `sc.tl.umap` call
            # — when routed through `_compute_umap_if_needed` — will
            # see `obsm['X_umap']` already populated and skip the CPU
            # recomputation.
            umap_path = work / "umap.npy"
            umap_loaded_shape = None
            if umap_path.is_file():
                adata.obsm["X_umap"] = np.load(umap_path)
                umap_loaded_shape = adata.obsm["X_umap"].shape
            print(
                f"  [Leiden][rapids-subprocess] loaded GPU pipeline "
                f"onto parent adata: kNN graph "
                f"(connectivities.nnz={adata.obsp['connectivities'].nnz}, "
                f"distances.nnz={adata.obsp['distances'].nnz}), "
                f"X_umap={umap_loaded_shape}, "
                f"use_rep={canonical_rep!r} (original was {use_rep!r}). "
                f"Downstream UMAP + re-clustering reuses the rapids "
                f"outputs directly (no CPU recomputation)."
            )
        else:
            # Worker didn't dump the graph (older worker, or partial
            # write). Fall back to CPU recomputation so downstream
            # UMAP still works.
            missing = [
                p.name for p in (conn_path, dist_path)
                if not p.is_file()
            ]
            print(
                f"  [Leiden][rapids-subprocess] WARN: graph artifacts "
                f"missing in tempdir ({missing}); falling back to CPU "
                f"sc.pp.neighbors on parent adata. "
                f"Upgrade _leiden_rapids_worker.py to the latest "
                f"version to skip this recomputation."
            )
            sc.pp.neighbors(
                adata, n_neighbors=n_neighbors, use_rep=canonical_rep,
                random_state=rng_seed,
            )
        return "leiden", int(result["n_found"]), float(result["resolution"])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_leiden_binary_search_rapids(
        adata: ad.AnnData,
        n_clusters: int,
        n_neighbors: int,
        max_iters: int,
        rng_seed: int,
        use_rep: str,
    ) -> Tuple[str, int, float]:
    """Inline rapids-singlecell Leiden binary-search. Calls
    `rsc.pp.neighbors` + `rsc.tl.leiden` DIRECTLY on the parent's
    adata — same API as scanpy, so the post-condition matches the
    CPU path exactly (`adata.obs['leiden']` and `adata.uns['neighbors']`
    both populated for downstream UMAP).

    Requires `import rapids_singlecell` to succeed in the active venv.
    Errors with a clear hint if it doesn't.
    """
    try:
        import rapids_singlecell as rsc
    except ImportError as exc:
        raise SystemExit(
            f"SQUINT_LEIDEN_BACKEND=rapids was requested but "
            f"`import rapids_singlecell` failed in the active venv.\n"
            f"Two ways to fix:\n"
            f"  (a) run this baseline from the rapids-singlecell conda "
            f"env directly (typically: `source /etc/profile.d/modules.sh "
            f"&& module load cellgen/conda && conda activate /nfs/team361/"
            f"sb75/ENVS/rapids-singlecell`), OR\n"
            f"  (b) `pip install rapids-singlecell` into this venv "
            f"(check cupy / cuML wheel compatibility first).\n"
            f"Or unset SQUINT_LEIDEN_BACKEND to fall back to CPU scanpy.\n"
            f"Original ImportError: {exc!r}"
        ) from exc
    if use_rep not in adata.obsm:
        raise RuntimeError(
            f"_run_leiden_binary_search_rapids: adata.obsm[{use_rep!r}] "
            f"missing. Available obsm keys: {list(adata.obsm.keys())}"
        )

    # Stage the embedding under the canonical `X_emb` name so the
    # inline and subprocess paths produce identical adata state
    # (obsm['X_emb'] + uns['neighbors'].params.use_rep='X_emb').
    if use_rep != "X_emb":
        adata.obsm["X_emb"] = adata.obsm[use_rep]
    canonical_rep = "X_emb"
    print(f"  [Leiden][rapids] using rapids-singlecell "
          f"v{getattr(rsc, '__version__', '?')} inline (rsc.pp.neighbors "
          f"+ rsc.tl.leiden) on use_rep={canonical_rep!r} "
          f"(original was {use_rep!r})")
    # rapids-singlecell mirrors scanpy's API; the call has the SAME
    # side-effect (populates adata.uns['neighbors']) so downstream
    # `sc.tl.umap(adata)` works without any extra plumbing.
    _t_n = time.time()
    rsc.pp.neighbors(
        adata, n_neighbors=n_neighbors, use_rep=canonical_rep,
        random_state=rng_seed,
    )
    _record_last_neighbors_seconds(time.time() - _t_n)

    # Also run UMAP on the same GPU kNN graph so the parent's adata
    # ends up with obsm['X_umap'] populated. Downstream
    # `_compute_umap_if_needed` will detect it and skip the CPU
    # recomputation. UMAP time is recorded into _LAST_UMAP_SECONDS so
    # the runtime tracker can report / subtract it correctly.
    print(f"  [Leiden][rapids] running rsc.tl.umap inline on GPU graph "
          f"(random_state={rng_seed})")
    _umap_t0 = time.time()
    rsc.tl.umap(adata, random_state=rng_seed)
    _record_last_umap_seconds(time.time() - _umap_t0)
    obs_key = "leiden"
    lo, hi = 0.05, 10.0
    best_diff = None
    best_n = None
    best_res = None
    print(f"  [Leiden][rapids] Bisecting Leiden resolution to hit "
          f"n_clusters = {n_clusters} (initial range {lo}-{hi}, "
          f"max {max_iters} iters):")
    _iter_times: List[float] = []
    _t_l = time.time()
    for it in range(max_iters):
        mid = 0.5 * (lo + hi)
        _t_iter = time.time()
        rsc.tl.leiden(
            adata, resolution=mid, key_added=obs_key,
            random_state=rng_seed,
        )
        _iter_times.append(time.time() - _t_iter)
        n_found = int(adata.obs[obs_key].astype(str).nunique())
        diff = abs(n_found - n_clusters)
        print(f"    iter {it+1:>2d}  resolution={mid:.4f}  -> "
              f"n_clusters={n_found}")
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_n = n_found
            best_res = mid
        if n_found == n_clusters:
            _record_last_leiden_seconds(time.time() - _t_l)
            if _iter_times:
                _record_last_leiden_one_iter_seconds(
                    sum(_iter_times) / len(_iter_times)
                )
            return obs_key, n_found, mid
        if n_found < n_clusters:
            lo = mid
        else:
            hi = mid
    # Re-run at the best resolution we found (in case the loop ended
    # on a non-best iteration).
    _t_iter = time.time()
    rsc.tl.leiden(
        adata, resolution=best_res, key_added=obs_key,
        random_state=rng_seed,
    )
    _iter_times.append(time.time() - _t_iter)
    _record_last_leiden_seconds(time.time() - _t_l)
    if _iter_times:
        _record_last_leiden_one_iter_seconds(
            sum(_iter_times) / len(_iter_times)
        )
    print(f"  [Leiden][rapids] ! exact match not reached; using closest "
          f"(n={best_n}, resolution={best_res:.4f}).")
    return obs_key, best_n, best_res


def _compute_umap_if_needed(
        adata: ad.AnnData,
        random_state: int,
        force: bool = False,
    ) -> None:
    """`sc.tl.umap(adata, random_state=...)` with a fast-path skip
    when `obsm['X_umap']` is already populated (e.g. by the rapids
    subprocess worker or the inline rapids path).

    Use this in every runner INSTEAD of calling `sc.tl.umap` directly.
    Drop-in replacement — same side effect (writes obsm['X_umap'])
    when rapids hasn't already done so.

    `force=True` always recomputes (legacy escape hatch). Default
    behaviour is to short-circuit if `obsm['X_umap']` exists, since the
    rapids pipeline now bundles `pp.neighbors + tl.umap + tl.leiden`
    in one GPU pass and populates `obsm['X_umap']` directly on the
    parent's adata.
    """
    if not force and "X_umap" in adata.obsm:
        print(f"  [umap] obsm['X_umap'] already populated "
              f"(shape={adata.obsm['X_umap'].shape}) — skipping CPU "
              f"recomputation (rapids did it).")
        return
    sc.tl.umap(adata, random_state=random_state)


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

    Timing + GPU
    ------------
    Wall-clock elapsed seconds are recorded into the module-level
    `_LAST_LEIDEN_SECONDS` singleton on EVERY call (CPU and GPU paths
    alike). `_record_seed_runtime` reads it.

    When `SQUINT_LEIDEN_RAPIDS_ENV_SETUP` is set, the binary search
    runs in a subprocess inside the rapids-singlecell env for
    GPU-accelerated neighbors + Leiden. Otherwise (default) the
    binary search runs on CPU via scanpy.
    """
    _reset_last_phase_timings()
    t0 = time.time()
    try:
        rapids_env_setup = _rapids_leiden_setup_cmd()
        if rapids_env_setup is not None:
            # Subprocess path: rapids lives in a separate conda env.
            return _run_leiden_binary_search_rapids_subprocess(
                adata=adata, n_clusters=n_clusters,
                n_neighbors=n_neighbors, max_iters=max_iters,
                rng_seed=rng_seed, use_rep="X_pca",
                env_setup_cmd=rapids_env_setup,
            )
        if _leiden_backend() == "rapids":
            # Inline path: rapids importable in this venv.
            return _run_leiden_binary_search_rapids(
                adata=adata, n_clusters=n_clusters,
                n_neighbors=n_neighbors, max_iters=max_iters,
                rng_seed=rng_seed, use_rep="X_pca",
            )
        # ---- CPU scanpy path (original implementation) ----
        _t_n = time.time()
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca",
                        random_state=rng_seed)
        _record_last_neighbors_seconds(time.time() - _t_n)

        obs_key = "leiden"
        lo, hi = 0.05, 10.0
        best_key = None
        best_diff = None
        best_n = None
        best_res = None
        print(f"Bisecting Leiden resolution to hit n_clusters = {n_clusters} "
              f"(initial range {lo}-{hi}, max {max_iters} iters):")
        _iter_times: List[float] = []
        _t_l = time.time()
        for it in range(max_iters):
            mid = 0.5 * (lo + hi)
            _t_iter = time.time()
            sc.tl.leiden(adata, resolution=mid, key_added=obs_key,
                         random_state=rng_seed)
            _iter_times.append(time.time() - _t_iter)
            n_found = int(adata.obs[obs_key].astype(str).nunique())
            diff = abs(n_found - n_clusters)
            print(f"  iter {it+1:>2d}  resolution={mid:.4f}  -> n_clusters={n_found}")
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_n = n_found
                best_res = mid
                best_key = obs_key
            if n_found == n_clusters:
                _record_last_leiden_seconds(time.time() - _t_l)
                if _iter_times:
                    _record_last_leiden_one_iter_seconds(
                        sum(_iter_times) / len(_iter_times)
                    )
                return obs_key, n_found, mid
            if n_found < n_clusters:
                lo = mid
            else:
                hi = mid

        # Re-run at the best resolution we found (in case the loop ended on
        # a non-best iteration).
        _t_iter = time.time()
        sc.tl.leiden(adata, resolution=best_res, key_added=obs_key,
                     random_state=rng_seed)
        _iter_times.append(time.time() - _t_iter)
        _record_last_leiden_seconds(time.time() - _t_l)
        if _iter_times:
            _record_last_leiden_one_iter_seconds(
                sum(_iter_times) / len(_iter_times)
            )
        print(f"  ! exact match not reached; using closest "
              f"(n={best_n}, resolution={best_res:.4f}).")
        return best_key, best_n, best_res
    finally:
        # Each helper now sets _LAST_NEIGHBORS_SECONDS,
        # _LAST_LEIDEN_SECONDS (binary search only),
        # _LAST_LEIDEN_ONE_ITER_SECONDS, _LAST_UMAP_SECONDS directly.
        # The finally block doesn't time anything — if an inner helper
        # raised before setting timings, the singletons stay at the
        # 0.0 from _reset_last_phase_timings() so we won't attribute
        # stale values.
        pass


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
        leiden_seconds: Optional[float] = None,
        umap_seconds: Optional[float] = None,
        neighbors_seconds: Optional[float] = None,
        leiden_one_iter_seconds: Optional[float] = None,
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
      - `leiden_seconds`: ISOLATED cost of the Leiden binary-search
        phase ONLY (sc.pp.neighbors + iterated sc.tl.leiden, or the
        rapids-singlecell equivalent). Captured automatically from the
        most recent `_leiden_n_clusters` /
        `_leiden_binary_search_on_latent` call's wall-clock — callers
        do NOT have to thread it through manually. Pass an explicit
        value to override the auto-detection (e.g. Novae, which uses
        `model.assign_domains` instead of Leiden — set to 0.0 there).
      - `umap_seconds`: cost of the UMAP phase (rapids modes only —
        the rapids pipeline runs `pp.neighbors + tl.leiden + tl.umap`
        in one GPU pass). EXCLUDED from `runtime_seconds` because
        UMAP is visualisation, not clustering cost. The caller's
        `local_seconds` (timed around the helper call) is REDUCED by
        this amount so the recorded clustering runtime is consistent
        between rapids and scanpy modes.
      - `runtime_seconds` (= local + shared) is what gets reported in
        the per_seed_runtimes.csv. This is the "time to obtain
        clusters from raw data for one seed" — directly comparable to
        SQUINT's per-seed (train + predict) numbers.

    EXCLUDED from `runtime_seconds`: UMAP (visualization only — it
    isn't a preprocessing step for Leiden in any of our runners),
    NMI/ARI computation, iLISI/MMD/ASW computation, per-seed plot
    writes. Those are benchmark scaffolding, not method cost.

    The single-row per-seed CSV has columns: seed, method,
    runtime_seconds, local_seconds, shared_setup_seconds, leiden_seconds.
    """
    if leiden_seconds is None:
        leiden_seconds = _get_last_leiden_seconds()
    if umap_seconds is None:
        umap_seconds = _get_last_umap_seconds()
    if neighbors_seconds is None:
        neighbors_seconds = _get_last_neighbors_seconds()
    if leiden_one_iter_seconds is None:
        leiden_one_iter_seconds = _get_last_leiden_one_iter_seconds()
    leiden_seconds          = float(leiden_seconds)
    umap_seconds            = float(umap_seconds)
    neighbors_seconds       = float(neighbors_seconds)
    leiden_one_iter_seconds = float(leiden_one_iter_seconds)

    # The rapids leiden helper bundles UMAP into its work, so the
    # caller's `local_seconds` (timed around the helper call) may
    # include UMAP. UMAP is visualisation, not clustering — strip it
    # here so the recorded runtimes are clustering-only. On the CPU
    # scanpy path `umap_seconds` is 0 (UMAP runs OUTSIDE the helper)
    # so this is a no-op.
    local_seconds_clustering = max(
        0.0, float(local_seconds) - umap_seconds
    )
    # Decompose the clustering wall time:
    #     local_seconds_clustering  ≈  M + N + L_full
    #   where
    #     M       = model fit + inference (seed-dependent training,
    #               zero for shared-embedding methods)
    #     N       = neighbors_seconds (pp.neighbors only)
    #     L_full  = leiden_seconds  (binary search loop only)
    # Then derive the three runtime flavours the user requested:
    model_only_local = max(
        0.0, local_seconds_clustering - neighbors_seconds - leiden_seconds
    )
    one_leiden_local = max(
        0.0,
        local_seconds_clustering - leiden_seconds + leiden_one_iter_seconds,
    )

    # (1) total runtime to obtain clusters — the existing headline:
    runtime_seconds                  = local_seconds_clustering   + float(shared_setup_seconds)
    # (2) training + inference only (no neighbors, no leiden):
    runtime_model_only_seconds       = model_only_local           + float(shared_setup_seconds)
    # (3) training + inference + neighbors + ONE leiden iteration
    #     (the binary-search overhead is a benchmark artifact —
    #     this number reflects the cost of "fit the embedding and
    #     run Leiden once at a single resolution"):
    runtime_one_leiden_seconds       = one_leiden_local           + float(shared_setup_seconds)

    row = {
        "seed": int(seed),
        "method": method,
        # Three runtime flavours (see decomposition above).
        "runtime_seconds":             runtime_seconds,
        "runtime_model_only_seconds":  runtime_model_only_seconds,
        "runtime_one_leiden_seconds":  runtime_one_leiden_seconds,
        # Underlying components.
        "local_seconds":               local_seconds_clustering,
        "shared_setup_seconds":        float(shared_setup_seconds),
        "neighbors_seconds":           neighbors_seconds,
        "leiden_seconds":              leiden_seconds,
        "leiden_one_iter_seconds":     leiden_one_iter_seconds,
        # Per-seed UMAP cost. NOT included in any runtime metric.
        "umap_seconds":                umap_seconds,
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
    "Three per-seed runtime flavours, all excluding metric "
    "computation (NMI/ARI/iLISI/MMD), UMAP (visualisation; reported "
    "as umap_seconds and SUBTRACTED from local_seconds in rapids "
    "modes), and plot writes. "
    "(1) runtime_seconds = model fit + inference + pp.neighbors + "
    "Leiden binary search + shared_setup. This is the 'time to "
    "obtain clusters from raw data' headline, directly comparable to "
    "SQUINT's per-seed (train + predict) numbers. "
    "(2) runtime_model_only_seconds = model fit + inference + "
    "shared_setup (no neighbors, no Leiden) — useful to isolate the "
    "embedding-production cost from the clustering overhead. "
    "(3) runtime_one_leiden_seconds = model fit + inference + "
    "pp.neighbors + ONE Leiden iteration + shared_setup. The Leiden "
    "binary-search loop is a benchmark artifact for landing on a "
    "target cluster count; this flavour reports the cost of 'fit "
    "embedding and Leiden once at a fixed resolution', which is "
    "what users would do in practice. "
    "shared_setup_seconds amortises one-shot embedding compute "
    "(e.g. BANKSY+Harmony, PCA, FM extraction) by adding it back to "
    "each seed's local cost."
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
    # Leiden seconds are per-seed (each seed runs its own binary search).
    leiden_secs = (
        long_df["leiden_seconds"].astype(float)
        if "leiden_seconds" in long_df.columns
        else pd.Series([0.0] * len(long_df))
    )
    # UMAP seconds are per-seed and NOT included in the headline
    # runtime metric (visualisation cost, not clustering). Reported
    # for transparency so users can compare GPU UMAP vs CPU UMAP cost.
    umap_secs = (
        long_df["umap_seconds"].astype(float)
        if "umap_seconds" in long_df.columns
        else pd.Series([0.0] * len(long_df))
    )
    # Neighbors-only and one-iteration timings (added to support the
    # three runtime flavours).
    neighbors_secs = (
        long_df["neighbors_seconds"].astype(float)
        if "neighbors_seconds" in long_df.columns
        else pd.Series([0.0] * len(long_df))
    )
    leiden_one_iter_secs = (
        long_df["leiden_one_iter_seconds"].astype(float)
        if "leiden_one_iter_seconds" in long_df.columns
        else pd.Series([0.0] * len(long_df))
    )
    # Companion runtime-flavour columns mirroring `runtime_seconds`.
    rt_model_only = (
        long_df["runtime_model_only_seconds"].astype(float)
        if "runtime_model_only_seconds" in long_df.columns
        else pd.Series([0.0] * len(long_df))
    )
    rt_one_leiden = (
        long_df["runtime_one_leiden_seconds"].astype(float)
        if "runtime_one_leiden_seconds" in long_df.columns
        else pd.Series([0.0] * len(long_df))
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
        # Per-seed Leiden cost (mean / total across seeds). Useful to
        # quantify the Leiden bottleneck vs the model-fit bottleneck
        # — e.g. for foundation models (scGPT / UCE / NicheFormer)
        # the FM extraction dominates `shared_setup_seconds`, while
        # for shared-embedding methods (BANKSY / neigh-expr-pca)
        # Leiden IS the per-seed cost.
        "mean_leiden_seconds":            float(leiden_secs.mean()),
        "total_leiden_seconds":           float(leiden_secs.sum()),
        # Per-seed neighbors cost (pp.neighbors only) — useful to
        # gauge how much of the clustering wall time is the kNN build
        # vs the Leiden binary-search.
        "mean_neighbors_seconds":         float(neighbors_secs.mean()),
        "total_neighbors_seconds":        float(neighbors_secs.sum()),
        # Mean cost of ONE Leiden iteration — drives the
        # `runtime_one_leiden_seconds` flavour (= model + neighbors +
        # one Leiden iter, excluding the binary-search overhead).
        "mean_leiden_one_iter_seconds":   float(leiden_one_iter_secs.mean()),
        # Companion runtime-flavour means (parallel to mean_seconds /
        # total_seconds for the existing cluster-total metric).
        "mean_runtime_model_only_seconds":  float(rt_model_only.mean()),
        "total_runtime_model_only_seconds": float(rt_model_only.sum()),
        "mean_runtime_one_leiden_seconds":  float(rt_one_leiden.mean()),
        "total_runtime_one_leiden_seconds": float(rt_one_leiden.sum()),
        # Per-seed UMAP cost (rapids modes only). NOT included in
        # mean_seconds / total_seconds — visualisation only.
        "mean_umap_seconds":              float(umap_secs.mean()),
        "total_umap_seconds":             float(umap_secs.sum()),
        # Which Leiden backend was active for this run? Recorded so
        # the runtime CSVs are self-describing.
        "leiden_backend":       (
            "rapids-singlecell-subprocess"
            if _rapids_leiden_setup_cmd() is not None
            else (
                "rapids-singlecell-inline"
                if _leiden_backend() == "rapids"
                else "scanpy-cpu"
            )
        ),
        "runtime_includes":     _RUNTIME_INCLUDES_NOTE,
    }])
    out = metrics_dir / "runtime_summary.csv"
    summary.to_csv(out, index=False)
    print(f"  -> {out}")
    print(f"     runtime [cluster total]:          mean={secs.mean():.1f}s ± "
          f"{secs.std(ddof=1) if len(secs) > 1 else 0.0:.1f}s, "
          f"total={secs.sum():.1f}s")
    print(f"     runtime [model only]:             "
          f"mean={float(rt_model_only.mean()):.1f}s")
    print(f"     runtime [model + neigh + 1 leiden]: "
          f"mean={float(rt_one_leiden.mean()):.1f}s")
    print(f"     phase breakdown — shared_setup={shared_secs:.1f}s, "
          f"mean_neighbors={float(neighbors_secs.mean()):.2f}s, "
          f"mean_leiden_full={float(leiden_secs.mean()):.2f}s, "
          f"mean_leiden_one_iter={float(leiden_one_iter_secs.mean()):.3f}s, "
          f"mean_umap={float(umap_secs.mean()):.2f}s (UMAP excluded)")


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
        # Index — ALWAYS rewrite to a plain object array. Detection by dtype is
        # unreliable: on some pandas/anndata combos (e.g. novae's env, with the
        # obs_names prep_xhs_3b.py produced) an index reports dtype 'object' yet
        # is still ArrowStringArray-backed, slips past the check, and crashes
        # the H5AD writer on key '_index'. obs_names/var_names are always
        # strings, so forcing them to object is cheap and safe — do it
        # unconditionally rather than rely on dtype sniffing.
        df.index = _force_object_index(df.index)
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
