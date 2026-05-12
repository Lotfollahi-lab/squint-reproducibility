"""
scGPT-spatial (foundation-model) baseline for the SQUINT cell-type-
identification benchmark — zero-shot embedding mode.

scGPT-spatial is a transformer pretrained on **SpatialHuman30M**: ~30M
human spatial-transcriptomics cells/spots from Visium, Visium HD,
MERFISH, and Xenium (Cui et al., scGPT-spatial preprint 2025;
biorxiv 2025.02.05.636714). Compared to vanilla scGPT (which was
trained on dissociated scRNA-seq), scGPT-spatial sees data with the
same low-count, narrow-panel characteristics as your MERFISH/STARmap
silver h5ads — so it's a much fairer foundation-model baseline than
either scGPT_human or Geneformer V2 on this benchmark.

Key facts about scGPT-spatial (source-verified from the preprint +
the scGPT v0.2.1 docs + scGPT-spatial DeepWiki):
  - Pretrained on HUMAN ONLY data. For mouse benchmarks (mmb-smb),
    pass `--map-via-human-orthologs` to map mouse symbols to human
    HGNC symbols via mygene + Ensembl REST homology, same recipe
    as `run_scgpt.py` and `compute_scgpt_spatial_embedding_mouse`.
  - Output dim: 512 (inherited from scGPT_human; scGPT_human itself
    is 768).
  - Architecture: TransformerModel + MoE decoder (protocol-aware) +
    adversarial components (implicit batch correction during
    pretraining). Spatial coordinates are NOT used at inference time
    — spatial awareness is encoded implicitly through training-time
    sampling.
  - Internal preprocessing: filter -> normalize_total(10_000) -> log1p
    (with check_logged) -> optional HVG/cell-norm -> binning.
  - Public API (canonical, scgpt v0.2.1+):
      embed_data(adata_or_file, model_dir, gene_col='feature_name',
                 max_length=1200, batch_size=64, obs_to_save=None,
                 device='cuda', use_fast_transformer=True,
                 return_new_adata=False)
    With `return_new_adata=True` (what we use), the embedding is on
    `.X` of the returned AnnData. Without, it's `obsm['X_scGPT']`.

Why "embed once, vary downstream" applies here too
--------------------------------------------------
scGPT-spatial in zero-shot mode is forward-pass-only — the latent is
approximately deterministic (modulo CUDA-attention non-determinism,
~1e-4 per dim, negligible for cluster identity). So we EMBED ONCE up
front and the seed loop only varies the kNN graph + Leiden + UMAP +
iLISI/MMD subsampling, same convention as scGPT / Geneformer /
Nicheformer / PCA + Leiden.

Pre-requisites
--------------
1. `pip install scgpt-spatial`. Or, if a separate package isn't
   distributed yet, `pip install scgpt` — the script tries
   `scgpt_spatial.tasks.embed_data` first and falls back to
   `scgpt.tasks.embed_data` automatically.
2. Download the scGPT-spatial pretrained checkpoint directory. It
   must contain ALL FIVE of:
      - best_model.pt
      - vocab.json
      - args.json
      - batch_mapping_dict.json
      - all_dict_mean_std.csv
   The bowang-lab/scGPT-spatial GitHub README documents the download
   link (typically Google Drive). Pass the path via `--model-dir`.
3. Gene names matching the checkpoint's vocab (HUMAN HGNC symbols).
   For mouse data, use `--map-via-human-orthologs` to convert
   mouse symbols -> human HGNC via the same pipeline that
   `compute_scgpt_spatial_embedding_mouse` uses (mygene + Ensembl
   REST homology + NCBI HomoloGene fallbacks).

Output layout mirrors the other baselines
(run_pca_leiden.py / run_harmony.py / run_scvi.py / run_scgpt.py) so
compare_variants.py picks them all up identically.

Approximate runtime: forward-pass embedding takes ~2-5 minutes on a
single GPU for ~100k cells (one-time). Per-seed downstream pipeline
is ~1-2 minutes, so 5 seeds = ~12-15 minutes total. Same speed
class as scGPT_human; way faster than scVI / training-based methods.

Usage:
    # Recommended for mouse data:
    python analysis/benchmarking/cell_type_identification/run_scgpt_spatial.py \\
        --model-dir /path/to/scGPT_spatial_human/ \\
        --map-via-human-orthologs

    # Human data (no ortholog mapping needed):
    python analysis/benchmarking/cell_type_identification/run_scgpt_spatial.py \\
        --model-dir /path/to/scGPT_spatial_human/

    # Custom dataset (chl59, etc.):
    python analysis/benchmarking/cell_type_identification/run_scgpt_spatial.py \\
        --model-dir /path/.../ --silver-dir /alt/silver \\
        --dataset-tag chl59-8b_1p
"""

import argparse
import inspect
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

# Reuse shared helpers from the sibling baselines.
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
# Reuse the mouse->human symbol mapping from run_scgpt — same recipe
# as the user's `compute_scgpt_spatial_embedding_mouse` (mygene +
# Ensembl REST homology + NCBI HomoloGene fallbacks via
# `_nicheformer_embedding.add_human_ortholog_ensembl_ids`).
from run_scgpt import (  # noqa: E402
    _mouse_to_human_symbols,
    scgpt_vocab_match_and_filter,
)


DEFAULT_VARIANT_TAG = "baseline-scgpt-spatial"
DEFAULT_LATENT_KEY  = "X_scgpt_spatial"


# ---------------------------------------------------------------------------
# scGPT-spatial zero-shot embedding
# ---------------------------------------------------------------------------

def _embed_with_scgpt_spatial(
        adata: ad.AnnData,
        model_dir: Path,
        gene_col: Optional[str],
        batch_size: int,
        max_length: int,
        device: Optional[str],
        use_fast_transformer: bool,
        latent_key: str,
    ) -> np.ndarray:
    """Zero-shot embedding via the scGPT-spatial public API.

    Tries `scgpt_spatial.tasks.embed_data` first, then falls back to
    `scgpt.tasks.embed_data` (recent scgpt versions support both
    standard and spatial checkpoints through the same entry point).

    Returns the latent as a (n_obs, embed_dim) NumPy array. embed_dim
    is 512 for canonical scGPT-spatial checkpoints.
    """
    embed_data = None
    backend = None

    # The scGPT-spatial repo is typically NOT pip-installable — the
    # canonical recipe is to clone bowang-lab/scGPT-spatial as a local
    # directory and add it to sys.path (see the scGPT-spatial-bench
    # notebook). We try, in order:
    #   1. `scgpt_spatial` already importable (pip installed)
    #   2. The repo is at `<model_dir>/../..` (model_dir is
    #      `<repo>/checkpoints/scGPT_spatial_v1`)
    #   3. `<scgpt_spatial_repo>` from CLI flag (passed in as a global
    #      via the `model_dir`'s grandparent — auto-detected)
    # If none of these find scgpt_spatial, fall back to vanilla scgpt
    # WITH A LOUD WARNING — the fallback gives wrong-ish embeddings
    # (uses scgpt_human vocab against the scGPT-spatial checkpoint;
    # bypasses the MoE decoder + spatial-aware attention specific to
    # scGPT-spatial).
    # Log the underlying import error from each scgpt_spatial attempt
    # so the user can see WHY the package isn't loading (e.g., a
    # missing dep, a relative-import issue, etc.) instead of silently
    # falling back to vanilla scgpt.
    import_errors: List[str] = []

    try:
        from scgpt_spatial.tasks import embed_data as _ed
        embed_data, backend = _ed, "scgpt_spatial.tasks.embed_data"
    except Exception as e:  # capture broadly; log later if all paths fail
        import_errors.append(
            f"  attempt 1 (scgpt_spatial already importable): "
            f"{type(e).__name__}: {e}"
        )

    if embed_data is None:
        # Try to auto-discover the scGPT-spatial local repo from
        # model_dir. Layout: <repo>/checkpoints/<model_name>
        candidates = []
        md = Path(model_dir).resolve()
        if md.parent.name == "checkpoints":
            candidates.append(md.parent.parent)
        if md.name == "checkpoints":
            candidates.append(md.parent)

        for c in candidates:
            init_py = c / "scgpt_spatial" / "__init__.py"
            if not init_py.is_file():
                import_errors.append(
                    f"  attempt 2: candidate {c} has no scgpt_spatial/"
                    "__init__.py — skipping"
                )
                continue
            sys.path.insert(0, str(c))
            try:
                from scgpt_spatial.tasks import embed_data as _ed
                embed_data = _ed
                backend = (
                    f"scgpt_spatial.tasks.embed_data "
                    f"(via sys.path={c})"
                )
                break
            except Exception as e:
                import_errors.append(
                    f"  attempt 2 (via sys.path={c}): "
                    f"{type(e).__name__}: {e}"
                )

    if embed_data is None:
        # Print the per-attempt errors so the user can fix the
        # underlying cause (most often a missing dep in the venv, or
        # scgpt-spatial vendors a modified scgpt that conflicts with
        # the pip-installed one).
        print(
            "\n"
            "*** scgpt_spatial NOT importable. Per-attempt errors: ***"
        )
        for line in import_errors:
            print(line)

        try:
            from scgpt.tasks import embed_data as _ed
            embed_data, backend = _ed, "scgpt.tasks.embed_data (FALLBACK)"
            md = Path(model_dir).resolve()
            print(
                "\n"
                "*** Falling back to vanilla scgpt.tasks.embed_data, "
                "which loads the scGPT-spatial checkpoint but "
                "BYPASSES scGPT-spatial's MoE decoder + "
                "spatial-aware attention. Resulting embeddings are "
                "structurally different from scGPT-spatial's intended "
                "output — you'll get bad NMI vs cell_type / niche.\n"
                "***\n"
                "*** Fix: address the import error printed above. "
                f"The script auto-detects the repo at:\n"
                f"      <model_dir>/../..  =  {md.parent.parent}\n"
                "*** Layout expected:\n"
                "***     <repo>/scgpt_spatial/__init__.py\n"
                "***     <repo>/checkpoints/<model_name>/\n"
                "*** \n"
                "*** Reproduce the error standalone with:\n"
                f"***     python -c \"import sys; "
                f"sys.path.insert(0, '{md.parent.parent}'); "
                f"from scgpt_spatial.tasks import embed_data\"\n"
                "***\n"
            )
        except ImportError as exc:
            raise SystemExit(
                "Neither `scgpt_spatial` nor `scgpt` is importable.\n"
                "scgpt_spatial errors above; final scgpt fallback: "
                f"{exc}"
            )

    if not Path(model_dir).is_dir():
        raise SystemExit(
            f"--model-dir={model_dir!r} is not a directory. Pass the "
            "full path to the scGPT-spatial checkpoint folder (must "
            "contain best_model.pt, vocab.json, args.json, "
            "batch_mapping_dict.json, all_dict_mean_std.csv)."
        )

    print(f"  backend: {backend}")
    print(f"  scGPT-spatial zero-shot embed: model_dir={model_dir}")
    print(f"    n_cells={adata.n_obs}, n_genes={adata.n_vars}, "
          f"batch_size={batch_size}, max_length={max_length}")

    # `embed_data` requires the gene-symbol column to live in
    # `adata.var[gene_col]` (assertion). Default scGPT-spatial gene_col
    # is "feature_name". If the caller didn't specify --gene-col and
    # there's no matching var column, copy from var_names.
    effective_gene_col = gene_col if gene_col is not None else "feature_name"
    if effective_gene_col not in adata.var.columns:
        print(f"  adata.var has no {effective_gene_col!r} column — "
              f"copying from var_names ({adata.var_names[:3].tolist()}...).")
        adata.var[effective_gene_col] = adata.var_names.astype(str)

    # Densify X if sparse — matches the notebook's
    # `adata.X = adata.X.toarray()` step. scGPT-spatial has separate
    # code paths for sparse vs dense and the dense path is the one
    # exercised by the canonical README + benchmarking notebook;
    # forcing dense here removes that as a source of variance.
    try:
        from scipy.sparse import issparse as _sp_issparse
        if _sp_issparse(adata.X):
            print("  densifying adata.X (was sparse) to match the "
                  "notebook's pre-embed densification.")
            adata.X = adata.X.toarray()
    except Exception:
        pass

    # Build the kwargs dict and forward only the ones the installed
    # `embed_data` actually accepts (signature varies across scgpt /
    # scgpt-spatial versions).
    #
    # Default values match the notebook recipe:
    #   - obs_to_save = list(adata.obs.columns) — preserves cell-level
    #     metadata in the returned AnnData (the notebook does this).
    #   - max_length is ONLY passed when the caller set it explicitly
    #     (max_length is None by default at the CLI). Forcing a
    #     specific value can subtly affect scgpt-spatial's tokenization
    #     (different from scgpt's default 1200) — letting the library
    #     pick its own default is safer. Override with --scgpt-max-length
    #     if you want to truncate.
    desired: Dict = dict(
        adata_or_file=adata,
        model_dir=str(model_dir),
        gene_col=effective_gene_col,
        batch_size=int(batch_size),
        obs_to_save=list(adata.obs.columns),
        return_new_adata=True,
        use_fast_transformer=bool(use_fast_transformer),
    )
    if max_length is not None:
        desired["max_length"] = int(max_length)
    if device is not None:
        desired["device"] = device

    sig = inspect.signature(embed_data)
    accepted = set(sig.parameters)
    kwargs = {k: v for k, v in desired.items() if k in accepted and v is not None}
    # Some scgpt versions name the first positional arg differently
    # (e.g. `adata_or_file` vs `adata`). Fall back to the first
    # parameter name if `adata_or_file` isn't accepted.
    if "adata_or_file" not in accepted and adata is not None:
        first_arg = next(iter(sig.parameters))
        kwargs.pop("adata_or_file", None)
        kwargs[first_arg] = adata
    print(f"  embed_data kwargs: {sorted(kwargs.keys())}")

    out = embed_data(**kwargs)
    if out is None:
        # Some versions write into adata in place and return None.
        out = adata

    # Pull the embedding. With `return_new_adata=True` it lands on
    # `.X` of the returned AnnData. Otherwise check obsm fallbacks
    # (covers older scgpt versions and the in-place return path).
    latent = None
    fallback_keys = ("X_scGPT", "X_scgpt_spatial", "scgpt_spatial", "scgpt")
    if isinstance(out, ad.AnnData):
        for cand in fallback_keys:
            if cand in out.obsm:
                latent = np.asarray(out.obsm[cand], dtype=np.float32)
                print(f"  -> latent extracted from out.obsm[{cand!r}]")
                break
        if latent is None and out.X is not None:
            latent = np.asarray(out.X, dtype=np.float32)
            print(f"  -> latent extracted from out.X")
    if latent is None:
        raise RuntimeError(
            f"scGPT-spatial returned an unexpected object: "
            f"{type(out).__name__}; could not locate the embedding "
            f"in obsm{list(fallback_keys)} or .X."
        )

    print(f"  -> latent shape: {latent.shape}")
    return latent


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
    p.add_argument("--out-dir", type=Path, default=None)
    # scGPT-spatial-specific.
    p.add_argument(
        "--model-dir", type=Path, required=True,
        help="Path to a downloaded scGPT-spatial checkpoint directory. "
             "Must contain best_model.pt, vocab.json, args.json, "
             "batch_mapping_dict.json, all_dict_mean_std.csv. See the "
             "bowang-lab/scGPT-spatial README for download instructions.",
    )
    p.add_argument(
        "--gene-col", type=str, default=None,
        help="adata.var column holding gene symbols matching the "
             "scGPT-spatial vocab. Default None means we auto-copy "
             "var_names into 'feature_name' (the canonical scGPT key).",
    )
    p.add_argument("--scgpt-batch-size", type=int, default=64,
                   help="Cells per forward pass (scGPT default 64).")
    p.add_argument("--scgpt-max-length", type=int, default=None,
                   help="Max gene-token sequence length. Default None — "
                        "uses scgpt-spatial's own internal default "
                        "(matches the canonical scgpt-spatial-bench "
                        "notebook, which doesn't pass max_length). "
                        "Override only if you have a specific reason: "
                        "scgpt's default is 1200 (for whole-"
                        "transcriptome scRNA-seq); for tiny spatial "
                        "panels (~few hundred genes) the default is "
                        "fine since cell sentences are shorter than "
                        "any reasonable max_length anyway.")
    p.add_argument("--scgpt-device", type=str, default=None,
                   help="Override device. Default lets scGPT pick "
                        "(typically cuda if available, else cpu).")
    p.add_argument("--use-fast-transformer", action="store_true",
                   help="Use flash-attn for inference (faster, "
                        "requires flash-attn install). Default off "
                        "for portability.")
    p.add_argument("--map-via-human-orthologs", action="store_true",
                   help="Map mouse gene symbols to human HGNC symbols "
                        "via the shared 4-step ortholog pipeline "
                        "(mygene mouse-DB -> Ensembl REST homology -> "
                        "mygene human-DB direct -> NCBI HomoloGene "
                        "fallback) followed by ENSG -> HGNC symbol "
                        "lookup. REQUIRED for stock human-only "
                        "scGPT-spatial checkpoints on mouse data — "
                        "without this, gene-vocab match rate is 0% "
                        "because scGPT vocab is uppercase HGNC and "
                        "mouse symbols are mixed-case (Gad1 vs GAD1). "
                        "Same shared helper as run_geneformer.py / "
                        "run_nicheformer.py / run_scgpt.py — "
                        "improvements propagate. Slow first call (~1-3 "
                        "min for ~500 genes); needs internet. Typical "
                        "coverage: ~85-92%.")
    p.add_argument("--species", type=str, default="mouse",
                   help="Species of the input data. Used by "
                        "--map-via-human-orthologs. Default: mouse.")
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
    print(f"Model   : {args.model_dir}")

    # 1. Load + concat + scGPT-spatial embedding extraction (one-time
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

    # 2a. Optionally remap mouse symbols -> human HGNC symbols. Required
    #     for stock human-only scGPT-spatial checkpoints on mouse data
    #     (else gene-vocab match rate is 0% and embed_data crashes).
    if args.map_via_human_orthologs:
        print("\n=== Mapping mouse genes to human orthologs ===")
        adata_for_embed = _mouse_to_human_symbols(
            adata.copy(), species=args.species,
        )
        if adata_for_embed.n_obs != adata.n_obs:
            raise SystemExit(
                f"Ortholog mapping changed n_obs ({adata.n_obs} -> "
                f"{adata_for_embed.n_obs}); cannot align latent."
            )
    else:
        adata_for_embed = adata

    # 2b. Pre-embedding QC. Identify cells with 0 in-vocab nonzero
    #     genes and report the vocab match rate. Cells that fail this
    #     check would produce degenerate CLS-only embeddings — drop
    #     them upfront and from `adata` itself so downstream metrics
    #     see only cells with meaningful embeddings. Same shared
    #     helper as run_scgpt.py — fixes propagate.
    print("\n=== Pre-embedding QC ===")
    effective_gene_col = args.gene_col if args.gene_col is not None else "feature_name"
    if effective_gene_col not in adata_for_embed.var.columns:
        adata_for_embed.var[effective_gene_col] = adata_for_embed.var_names.astype(str)
    valid_pre_mask, _n_in_vocab, _n_total = scgpt_vocab_match_and_filter(
        adata_for_embed, model_dir=args.model_dir,
        gene_col=effective_gene_col,
    )
    n_zero_genes = int((~valid_pre_mask).sum())
    if n_zero_genes > 0:
        adata_for_embed = adata_for_embed[valid_pre_mask].copy()
        adata = adata[valid_pre_mask].copy()

    # 2c. Embed ONCE with scGPT-spatial (forward-pass only;
    #     ~deterministic).
    print("\n=== scGPT-spatial zero-shot embedding ===")
    latent = _embed_with_scgpt_spatial(
        adata=adata_for_embed,
        model_dir=args.model_dir,
        gene_col=args.gene_col,
        batch_size=args.scgpt_batch_size,
        max_length=args.scgpt_max_length,
        device=args.scgpt_device,
        use_fast_transformer=args.use_fast_transformer,
        latent_key=DEFAULT_LATENT_KEY,
    )

    # 2d. Defensive valid-mask: scGPT-spatial's L2-normalisation step
    #     divides by zero on any all-zero embedding row, producing
    #     NaN/Inf — same footgun as UCE/Geneformer/scGPT. Subset adata
    #     to cells with finite embeddings so all downstream
    #     computation (Leiden, UMAP, NMI/ARI, iLISI, MMD, per-seed
    #     h5ad) sees only valid cells.
    valid_mask = np.isfinite(latent).all(axis=1)
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        n_total_post = adata.n_obs
        frac = 100.0 * n_invalid / n_total_post
        print(f"  Excluding {n_invalid}/{n_total_post} ({frac:.1f}%) "
              "cells without a valid scGPT-spatial embedding (NaN/Inf "
              "rows). Downstream metrics, UMAPs, and per-seed h5ad "
              f"exports are restricted to the {n_total_post - n_invalid} "
              "cells with valid embeddings only.")
        if n_invalid == n_total_post:
            raise SystemExit(
                "ALL scGPT-spatial embeddings are NaN/Inf — likely the "
                "L2-norm step divided by zero for every cell. Check "
                "vocab match rate and that adata.X holds raw counts."
            )
        adata = adata[valid_mask].copy()
        latent = latent[valid_mask]
    assert np.isfinite(latent).all(), (
        "scGPT-spatial latent still contains non-finite values after "
        "valid-embedding subset — refusing to run downstream metrics "
        "on poisoned input."
    )

    # Attach to the ORIGINAL (mouse-symbol) adata so downstream Leiden /
    # UMAP / metrics see the unmodified obs/X.
    adata.obsm[DEFAULT_LATENT_KEY] = latent
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + scGPT-spatial extraction): "
          f"{shared_setup_seconds:.1f}s")

    # 3. Per-seed downstream loop.
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
        # TIMED block: Leiden binary search on the shared scGPT-spatial
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
            run_dir=args.out_dir, method="scGPT-spatial",
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

    # 4. Per-seed long + aggregated mean CSVs.
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

    # 5. Console summary.
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

    # 6. seed[0] AnnData snapshot + UMAPs.
    if seed0_state is not None:
        adata.uns["scgpt_spatial_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["scgpt_spatial_resolution"] = float(seed0_state["resolution"])
        adata.uns["scgpt_spatial_seeds"] = seeds
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

    # 7. Stub config.
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "scGPT-spatial zero-shot baseline "
                                      "(multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "scGPT-spatial"},
        "scgpt_spatial": {
            "model_dir":             str(args.model_dir),
            "gene_col":              args.gene_col,
            "scgpt_batch_size":      int(args.scgpt_batch_size),
            "scgpt_max_length":      int(args.scgpt_max_length),
            "use_fast_transformer":  bool(args.use_fast_transformer),
            "map_via_human_orthologs": bool(args.map_via_human_orthologs),
            "species":               args.species,
            "n_clusters_target":     int(args.n_clusters),
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
