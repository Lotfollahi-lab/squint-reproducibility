"""
Nicheformer baseline for the SQUINT cell-identification benchmark.

Wraps the local `_nicheformer_embedding.py` preprocessor (taken from
the user's prior project — itself mirrors the official tokenization
notebook in theislab/nicheformer:
``notebooks/tokenization/merfish_mouse_brain.ipynb``) and runs the
standard 5-seed downstream pipeline on the resulting embedding.

Why "embed once, vary downstream" applies here too
--------------------------------------------------
Nicheformer in zero-shot mode is forward-pass-only. Like scGPT, the
latent is approximately deterministic (modulo CUDA-attention non-
determinism, ~1e-4 per dim, negligible for cluster identity). So we
embed ONCE up front and the seed loop only varies the kNN graph +
Leiden + UMAP + iLISI/MMD subsampling.

Pre-requisites
--------------
1. `pip install nicheformer` (and PyTorch + the model deps from the
   theislab/nicheformer repo).
2. Three artefacts from the Nicheformer release:
     - the pretrained checkpoint (`.ckpt`)
     - the gene reference `model.h5ad` (20,310 canonical human
       Ensembl IDs)
     - the technology-specific mean (`.npy`, length 20,310)
   By default the script reads them from
   `/nfs/team361/sb75/squint-reproducibility/analysis/benchmarking/
   nicheformer/` (override via `--model-dir`). The conventional
   filenames it expects there are:
     - `nicheformer.ckpt`
     - `model.h5ad`
     - `<technology>_mean.npy` (e.g. `merfish_mean.npy`)
   Pass `--pretrained-model-path` / `--model-h5ad-path` /
   `--technology-mean-path` to override any single file.
3. For mouse data, two options to map mouse genes to the human
   Ensembl reference Nicheformer expects:
   (a) `--auto-map-symbols`: runtime mapping via the SHARED helper
       `add_human_ortholog_ensembl_ids` in `_nicheformer_embedding.py`.
       This is the same 4-step pipeline used by run_geneformer.py /
       run_scgpt.py / run_scgpt_spatial.py (mygene mouse-DB -> Ensembl
       REST homology -> mygene human-DB direct -> NCBI HomoloGene).
       Typically achieves ~90% coverage on a 500-gene MERFISH panel.
       Slow first call (~1-3 min); needs internet. Recommended when
       you don't already have a Biomart export.
   (b) `--gene-mapper-path <mart_export.csv>`: a static Biomart export
       with `Gene stable ID` and `Human gene stable ID` columns.
       Deterministic and reproducible if you have one curated; useful
       for paper-grade reproducibility.
   Or `--gene-col <var_col>` if you've already populated an obs col
   with human Ensembl IDs upstream.

Usage:
    # Recommended: pick up the model dir + conventional filenames + runtime
    # ortholog mapping. Just point at the silver dir.
    python analysis/benchmarking/cell_type_identification/run_nicheformer.py \\
        --auto-map-symbols

    # Custom model directory:
    python analysis/benchmarking/cell_type_identification/run_nicheformer.py \\
        --model-dir /path/to/nicheformer_release \\
        --auto-map-symbols

    # With a static Biomart export (deterministic, reproducible):
    python analysis/benchmarking/cell_type_identification/run_nicheformer.py \\
        --gene-mapper-path /path/to/mart_export.csv

    # Per-file overrides (e.g. when filenames don't match the convention):
    python analysis/benchmarking/cell_type_identification/run_nicheformer.py \\
        --pretrained-model-path /path/to/some_other_name.ckpt \\
        --technology-mean-path /path/to/custom_mean.npy \\
        --auto-map-symbols
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

# Reuse shared helpers + the preprocessor module living next to this file.
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
from run_scvi import _leiden_binary_search_on_latent  # noqa: E402
from _nicheformer_embedding import (  # noqa: E402
    NICHEFORMER_CONTEXT_LENGTH,
    add_human_ortholog_ensembl_ids,
    compute_nicheformer_embedding,
)


DEFAULT_VARIANT_TAG = "baseline-nicheformer"
DEFAULT_LATENT_KEY  = "X_nicheformer"

# Default Nicheformer artefact directory on the cluster. Holds the
# `.ckpt`, `model.h5ad`, and per-technology `<tech>_mean.npy` files.
# Pass `--model-dir` to override; pass `--pretrained-model-path` /
# `--model-h5ad-path` / `--technology-mean-path` to override individual
# files when the filenames don't match the conventional layout below.
DEFAULT_MODEL_DIR = Path(
    "/nfs/team361/sb75/squint-reproducibility/analysis/benchmarking/nicheformer"
)


def _resolve_nicheformer_paths(
        model_dir: Path,
        technology: str,
        pretrained_model_path: Optional[Path],
        model_h5ad_path: Optional[Path],
        technology_mean_path: Optional[Path],
    ) -> tuple:
    """Resolve the three Nicheformer artefact paths.

    Priority per file: explicit `--<file>-path` arg if given, else
    the first existing candidate under `model_dir`.

    Search layout (each file is tried in order; first hit wins):
      .ckpt:
        - <model_dir>/nicheformer.ckpt
        - <model_dir>/data/nicheformer.ckpt
      model.h5ad:
        - <model_dir>/model.h5ad
        - <model_dir>/model_means/model.h5ad        (release layout)
        - <model_dir>/data/model_means/model.h5ad
      <tech>_mean*.npy:
        - <model_dir>/<technology>_mean.npy
        - <model_dir>/means/<technology>_mean.npy
        - <model_dir>/model_means/<technology>_mean.npy        (release)
        - <model_dir>/model_means/<technology>_mean_script.npy (release)
        - <model_dir>/data/model_means/<technology>_mean.npy
        - <model_dir>/data/model_means/<technology>_mean_script.npy

    The `model_means/` + `_mean_script.npy` candidates match the
    on-disk layout of the official Nicheformer HuggingFace release
    (which stores `model.h5ad` and `<tech>_mean_script.npy` together
    under `<release>/data/model_means/`). The bare and `means/`
    candidates are kept for back-compat with hand-staged layouts.

    Raises SystemExit with a clear message if a path can't be resolved.
    """
    def _resolve_one(
            explicit: Optional[Path],
            candidates: List[Path],
            label: str,
            override_flag: str,
        ) -> Path:
        if explicit is not None:
            if not Path(explicit).is_file():
                raise SystemExit(
                    f"{override_flag}={explicit!r} does not exist."
                )
            return Path(explicit)
        for c in candidates:
            if c.is_file():
                return c
        searched = "\n  ".join(str(c) for c in candidates)
        raise SystemExit(
            f"Could not find {label} under --model-dir={model_dir}.\n"
            f"Searched:\n  {searched}\n"
            f"Pass {override_flag}=<path> to override."
        )

    ckpt = _resolve_one(
        explicit=pretrained_model_path,
        candidates=[
            model_dir / "nicheformer.ckpt",
            model_dir / "data" / "nicheformer.ckpt",
        ],
        label="pretrained .ckpt",
        override_flag="--pretrained-model-path",
    )
    h5ad = _resolve_one(
        explicit=model_h5ad_path,
        candidates=[
            model_dir / "model.h5ad",
            model_dir / "model_means" / "model.h5ad",
            model_dir / "data" / "model_means" / "model.h5ad",
        ],
        label="model.h5ad gene reference",
        override_flag="--model-h5ad-path",
    )
    mean_candidates = [
        model_dir / f"{technology}_mean.npy",
        model_dir / "means" / f"{technology}_mean.npy",
        model_dir / "model_means" / f"{technology}_mean.npy",
        model_dir / "model_means" / f"{technology}_mean_script.npy",
        model_dir / "data" / "model_means" / f"{technology}_mean.npy",
        model_dir / "data" / "model_means" / f"{technology}_mean_script.npy",
    ]
    mean = _resolve_one(
        explicit=technology_mean_path,
        candidates=mean_candidates,
        label=f"technology-mean .npy for technology={technology!r}",
        override_flag="--technology-mean-path",
    )
    return ckpt, h5ad, mean


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
    # Nicheformer artefact paths. The common case is to point at the
    # cluster directory holding all three files via `--model-dir`; the
    # per-file overrides exist for non-conventional filenames.
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help="Directory containing nicheformer.ckpt, "
                        "model.h5ad, and <technology>_mean.npy. Default: "
                        f"{DEFAULT_MODEL_DIR}")
    p.add_argument("--pretrained-model-path", type=Path, default=None,
                   help="Override: explicit .ckpt path. Default: "
                        "<model_dir>/nicheformer.ckpt.")
    p.add_argument("--model-h5ad-path", type=Path, default=None,
                   help="Override: explicit gene-reference .h5ad path. "
                        "Default: <model_dir>/model.h5ad.")
    p.add_argument("--technology-mean-path", type=Path, default=None,
                   help="Override: explicit technology-mean .npy. "
                        "Default: <model_dir>/<technology>_mean.npy "
                        "(falls back to <model_dir>/means/<technology>_mean.npy).")
    # Gene-id alignment route (one of the two below should be set).
    p.add_argument("--gene-mapper-path", type=Path, default=None,
                   help="Biomart export CSV with mouse->human ortholog "
                        "Ensembl IDs (cols 'Gene stable ID' + 'Human "
                        "gene stable ID'). Deterministic / reproducible "
                        "if you have a curated Biomart export; otherwise "
                        "use --auto-map-symbols (which now uses a 4-step "
                        "mapping: mygene mouse-DB lookup -> Ensembl REST "
                        "homology -> direct mygene human-DB lookup -> "
                        "NCBI HomoloGene fallback, typically achieving "
                        "~90% coverage).")
    p.add_argument("--gene-col", type=str, default=None,
                   help="adata.var column already holding human "
                        "Ensembl IDs. Use if you've pre-mapped via "
                        "the shared helper "
                        "(add_human_ortholog_ensembl_ids in "
                        "_nicheformer_embedding.py).")
    p.add_argument("--auto-map-symbols", action="store_true",
                   help="Derive human Ensembl IDs at runtime via the "
                        "shared 4-step mapping helper "
                        "(add_human_ortholog_ensembl_ids: mygene "
                        "mouse-DB -> Ensembl REST -> mygene human-DB "
                        "-> NCBI HomoloGene). Slow first call (~1-3 min "
                        "for ~500 genes); needs internet. Same code "
                        "path used by run_geneformer.py / run_scgpt.py "
                        "/ run_scgpt_spatial.py with their "
                        "--map-via-human-orthologs flag — improvements "
                        "to that helper benefit Nicheformer "
                        "automatically.")
    # Token vocabulary knobs.
    p.add_argument("--technology", type=str, default="merfish",
                   help="One of merfish/cosmx/visium/10x_*; see "
                        "TECHNOLOGY_TOKENS in _nicheformer_embedding.")
    p.add_argument("--species", type=str, default="mouse")
    p.add_argument("--modality", type=str, default="spatial")
    # Inference knobs.
    p.add_argument("--nicheformer-batch-size", type=int, default=32)
    p.add_argument("--max-seq-len", type=int,
                   default=NICHEFORMER_CONTEXT_LENGTH,
                   help=f"Token sequence length per cell. "
                        f"Default={NICHEFORMER_CONTEXT_LENGTH} "
                        "(model context length).")
    p.add_argument("--n-latent", type=int, default=None,
                   help="Optional PCA projection of the 512-dim "
                        "raw embedding. Default None = keep all 512.")
    p.add_argument("--device", type=str, default=None,
                   help="Device string for torch (default: auto).")
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

    if args.gene_col is None and args.gene_mapper_path is None and not args.auto_map_symbols:
        raise SystemExit(
            "For mouse data you must specify ONE of:\n"
            "  --gene-mapper-path <mart_export.csv>     (recommended, "
            "Biomart download)\n"
            "  --gene-col <var-column with human Ensembl>   (pre-mapped)\n"
            "  --auto-map-symbols                         (SLOW; "
            "uses mygene + Ensembl REST at runtime)"
        )

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = (
            args.artifacts_root / args.dataset_tag / args.variant_tag / ts
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the three Nicheformer artefact paths from `--model-dir`
    # (with per-file overrides). Aborts loudly if any are missing.
    args.pretrained_model_path, args.model_h5ad_path, args.technology_mean_path = (
        _resolve_nicheformer_paths(
            model_dir=args.model_dir,
            technology=args.technology,
            pretrained_model_path=args.pretrained_model_path,
            model_h5ad_path=args.model_h5ad_path,
            technology_mean_path=args.technology_mean_path,
        )
    )

    print(f"Run dir : {args.out_dir}")
    print(f"Seeds   : {seeds}")
    print(f"Model dir : {args.model_dir}")
    print(f"  ckpt   : {args.pretrained_model_path}")
    print(f"  h5ad   : {args.model_h5ad_path}")
    print(f"  mean   : {args.technology_mean_path}")
    print(f"Tech    : {args.technology}  Species: {args.species}  "
          f"Modality: {args.modality}")

    # 1. Load + concat + Nicheformer embedding extraction (one-time
    #    shared setup). Deterministic inference, so we share the latent
    #    across seeds. Timed as `shared_setup_seconds` for apples-to-
    #    apples runtime comparison (see `_record_seed_runtime` docstring).
    _shared_t0 = time.time()
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(
            f"--batch-key={args.batch_key!r} missing from obs."
        )
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    # 2. (Optional) auto-map mouse symbols -> human Ensembl IDs.
    gene_col = args.gene_col
    if args.auto_map_symbols and gene_col is None and args.gene_mapper_path is None:
        print("\n=== Auto-mapping mouse gene symbols -> human Ensembl ===")
        adata = add_human_ortholog_ensembl_ids(
            adata, species=args.species, verbose=True, inplace=True,
        )
        gene_col = "human_ensembl_id"

    # 3. Embed ONCE with Nicheformer (forward-pass only).
    print("\n=== Nicheformer zero-shot embedding ===")
    adata = compute_nicheformer_embedding(
        adata=adata,
        pretrained_model_path=str(args.pretrained_model_path),
        model_h5ad_path=str(args.model_h5ad_path),
        technology_mean_path=str(args.technology_mean_path),
        technology=args.technology,
        species=args.species,
        modality=args.modality,
        gene_col=gene_col,
        gene_mapper_path=(str(args.gene_mapper_path)
                         if args.gene_mapper_path is not None else None),
        obsm_key=DEFAULT_LATENT_KEY,
        batch_size=int(args.nicheformer_batch_size),
        device=args.device,
        max_seq_len=int(args.max_seq_len),
        n_latent=args.n_latent,
    )
    print(f"  -> adata.obsm[{DEFAULT_LATENT_KEY!r}] shape: "
          f"{adata.obsm[DEFAULT_LATENT_KEY].shape}")
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + Nicheformer extraction): "
          f"{shared_setup_seconds:.1f}s")

    # 4. Per-seed downstream loop.
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
        # TIMED block: Leiden binary search on the shared Nicheformer
        # latent. Below `seed_seconds = ...` runs UNTIMED.
        seed_t0 = time.time()
        leiden_key, n_found, resolution = _leiden_binary_search_on_latent(
            adata, n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors,
            latent_key=DEFAULT_LATENT_KEY, seed=seed,
        )
        seed_seconds = time.time() - seed_t0
        _record_seed_runtime(
            runtime_tracker, seed=seed,
            local_seconds=seed_seconds,
            shared_setup_seconds=shared_setup_seconds,
            run_dir=args.out_dir, method="Nicheformer",
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

        print(f"\n  -- Batch integration on {DEFAULT_LATENT_KEY} --")
        # Alias for the helper that reads obsm['X_pca'].
        adata.obsm["X_pca"] = adata.obsm[DEFAULT_LATENT_KEY]
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
            bint_df["emb_key"] = DEFAULT_LATENT_KEY
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

    # 5. Per-seed long + aggregated mean CSVs.
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

    # 6. Console summary.
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
            print(f"  {DEFAULT_LATENT_KEY} {metric:<5s} = "
                  f"{vals.mean():.4f} ± {std:.4f}  "
                  f"(min={vals.min():.4f}, max={vals.max():.4f})")

    # 7. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["nicheformer_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["nicheformer_resolution"] = float(seed0_state["resolution"])
        adata.uns["nicheformer_seeds"]      = seeds
        adata.uns["nicheformer_model_path"] = str(args.pretrained_model_path)
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

    # 8. Stub config.
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "Nicheformer zero-shot baseline (multi-seed downstream)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "Nicheformer-zero-shot"},
        "nicheformer": {
            "pretrained_model_path": str(args.pretrained_model_path),
            "model_h5ad_path":       str(args.model_h5ad_path),
            "technology_mean_path":  str(args.technology_mean_path),
            "gene_mapper_path":      (str(args.gene_mapper_path)
                                       if args.gene_mapper_path else None),
            "gene_col":              args.gene_col,
            "auto_map_symbols":      bool(args.auto_map_symbols),
            "technology":            args.technology,
            "species":               args.species,
            "modality":              args.modality,
            "batch_size":            int(args.nicheformer_batch_size),
            "max_seq_len":           int(args.max_seq_len),
            "n_latent":              args.n_latent,
            "device":                args.device,
            "batch_key":             args.batch_key,
            "n_clusters_target":     int(args.n_clusters),
            "n_neighbors":           int(args.n_neighbors),
            "seeds":                 seeds,
            "seed_summary":          seed_summary,
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
