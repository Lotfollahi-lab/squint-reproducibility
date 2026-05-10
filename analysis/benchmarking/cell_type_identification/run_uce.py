"""
UCE (Universal Cell Embedding) baseline for the SQUINT cell-type-
identification benchmark — zero-shot embedding mode.

UCE is a transformer pretrained across MULTIPLE species (Rosen et al.
2023, snap-stanford/UCE). Unlike scGPT / Geneformer (which are
human-only), UCE has **native mouse support**: its `species_chrom.csv`
+ species-specific protein-embedding files mean you can pass mouse
gene SYMBOLS directly with `--uce-species mouse` and the model
produces meaningful embeddings without any ortholog conversion. No
mouse->human symbol mapping is needed — just point UCE at mouse data
and tell it the species.

Why "embed once, vary downstream" applies here too
--------------------------------------------------
UCE in zero-shot mode is forward-pass-only (subprocess to the
`uce-eval-single-anndata` console script). The latent is approximately
deterministic, so we embed ONCE and the seed loop only varies kNN /
Leiden / UMAP / iLISI/MMD subsampling — same convention as
scGPT / Geneformer / Nicheformer / PCA + Leiden.

Pre-requisites
--------------
1. `pip install uce-model`. This installs the `uce-eval-single-anndata`
   console script in the same `bin/` as the active Python interpreter.
2. UCE support files. The pretrained 33-layer (1280-dim) or 4-layer
   (1024-dim) checkpoint plus auxiliary files:
     - `33l_8ep_1024t_1280.torch`  (or `4layer_model.torch`)
     - `species_chrom.csv`
     - `species_offsets.pkl`
     - `all_tokens.torch`
     - `protein_embeddings/<species>_proteome.embeddings.torch`
   Pass the directory containing these via `--model-files-dir`. The
   script will symlink it to `<work_dir>/model_files/` so UCE finds
   them locally without trying to fetch from Google Drive.
3. Path to the model weights `.torch` file — `--model-loc`.
4. Gene symbols (NOT Ensembl IDs) in `adata.var_names`. For mouse
   data, these should be mouse-cased symbols ('Cd3d', 'Sox2', ...);
   for human data, uppercase HGNC ('CD3D', 'SOX2', ...). UCE's
   internal alignment to its species_chrom.csv handles the rest.

Usage:
    # Default: native UCE mouse mode (recommended for mouse data)
    python analysis/benchmarking/cell_type_identification/run_uce.py \\
        --model-loc /path/to/33l_8ep_1024t_1280.torch \\
        --model-files-dir /path/to/uce_model_files/ \\
        --uce-species mouse

    # Human data:
    python analysis/benchmarking/cell_type_identification/run_uce.py \\
        --model-loc /path/.../ --model-files-dir /path/.../ \\
        --uce-species human

    # Custom dataset:
    python analysis/benchmarking/cell_type_identification/run_uce.py \\
        --model-loc /path/.../ --model-files-dir /path/.../ \\
        --silver-dir /alt/silver --dataset-tag chl59-8b_1p
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
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


DEFAULT_VARIANT_TAG = "baseline-uce"
DEFAULT_LATENT_KEY  = "X_uce"


# ---------------------------------------------------------------------------
# UCE console-script helpers (adapted from the user's `uce_embedder.py`).
# ---------------------------------------------------------------------------

def _resolve_uce_console_script() -> Path:
    """Locate the `uce-eval-single-anndata` console script installed by
    uce-model. Lives in the same bin/ as the active Python."""
    python_bin = Path(sys.executable)
    candidate = python_bin.parent / "uce-eval-single-anndata"
    if candidate.exists():
        return candidate
    found = shutil.which("uce-eval-single-anndata")
    if found is not None:
        return Path(found)
    raise SystemExit(
        "Could not find `uce-eval-single-anndata` console script. "
        "Install via `uv pip install uce-model` in the active venv."
    )


def _ensure_model_files_layout(work_dir: Path, model_files_dir: Path) -> None:
    """Symlink `<work_dir>/model_files` -> `model_files_dir` so UCE
    finds its support files locally rather than trying to download them
    from Google Drive."""
    expected = work_dir / "model_files"
    if expected.exists() or expected.is_symlink():
        if expected.is_symlink() and expected.resolve() == model_files_dir.resolve():
            return
        if expected.is_dir() and not expected.is_symlink():
            return
        expected.unlink()
    expected.symlink_to(model_files_dir.resolve(), target_is_directory=True)


def _embed_with_uce(
        adata: ad.AnnData,
        model_loc: Path,
        model_files_dir: Path,
        species: str,
        model_size: str,
        batch_size: int,
        sample_size: int,
        pad_length: int,
        filter_cells: bool,
        force_rerun: bool,
        work_dir: Path,
    ) -> np.ndarray:
    """Forward-pass UCE embedding via the `uce-eval-single-anndata`
    subprocess. Returns the per-cell latent as (n_obs, n_dim) NumPy
    array, with NaN rows for any cells UCE filtered out (caller must
    handle these — UCE's gene-coverage check can drop cells when the
    panel has too few genes that match its species_chrom.csv).
    """
    # 1. Sanity checks adapted from the user's compute_uce_embedding.
    if (
        adata.var_names.str.startswith("ENSG").any()
        or adata.var_names.str.startswith("ENSMUSG").any()
    ):
        raise SystemExit(
            "adata.var_names look like Ensembl IDs. UCE needs gene "
            "symbols (e.g. 'Cd3d' for mouse, 'CD3D' for human). For "
            "Ensembl-ID input, convert to symbols first or wrap in a "
            "preprocessing step that sets var_names to symbols."
        )
    if hasattr(adata.X, "max"):
        xmax = adata.X.max()
    else:
        xmax = float(np.asarray(adata.X).max())
    if xmax < 30:
        raise SystemExit(
            f"adata.X.max() = {xmax:.2f} — UCE expects RAW counts, "
            "not normalised expression."
        )

    nlayers = 33 if model_size == "large" else 4
    if model_size == "large" and model_loc is None:
        raise SystemExit(
            "--model-size=large requires --model-loc pointing to the "
            "33-layer .torch file."
        )

    # 2. Place model_files where UCE expects them.
    work_dir.mkdir(parents=True, exist_ok=True)
    _ensure_model_files_layout(work_dir, Path(model_files_dir))

    # 3. Write input.h5ad. UCE's CLI builds the output path via string
    # concat — pass --dir with a trailing separator so the resulting
    # path is well-formed.
    input_stem = "uce_input"
    input_path = work_dir / f"{input_stem}.h5ad"
    uce_dir_arg = str(work_dir).rstrip(os.sep) + os.sep
    output_path = Path(uce_dir_arg + f"{input_stem}_uce_adata.h5ad")
    adata.write_h5ad(input_path)
    print(f"  wrote temp h5ad: {input_path}")

    # 4. Invoke the console script.
    #
    # IMPORTANT --filter argparse footgun: UCE's eval_single_anndata.py
    # declares `parser.add_argument('--filter', type=bool, default=True)`.
    # Python's bool('False') is True (any non-empty string is truthy),
    # so passing '--filter False' silently keeps cell filtering ON. The
    # ONLY way to get additional_filter=False is to pass an empty
    # string (bool('') == False) — argparse forwards "" verbatim to
    # `type=bool`, which evaluates to False. Same trick for `--skip`.
    console_script = _resolve_uce_console_script()
    filter_arg = "true" if filter_cells else ""
    cmd = [
        str(console_script),
        "--adata_path",  str(input_path),
        "--dir",         uce_dir_arg,
        "--species",     species,
        "--nlayers",     str(nlayers),
        "--batch_size",  str(batch_size),
        "--sample_size", str(sample_size),
        "--pad_length",  str(pad_length),
        "--filter",      filter_arg,
    ]
    if model_loc is not None:
        cmd += ["--model_loc", str(model_loc)]

    if output_path.exists() and not force_rerun:
        print(f"  reusing existing UCE output: {output_path}  "
              f"(pass --uce-force-rerun to recompute)")
    else:
        if force_rerun and output_path.exists():
            output_path.unlink()
        print(f"  running UCE (cwd={work_dir}):")
        print(f"    {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=work_dir)

    # 5. Load result.
    if not output_path.exists():
        raise SystemExit(
            f"UCE finished but expected output {output_path} is missing. "
            f"Inspect {work_dir} for files matching *_uce_adata.h5ad."
        )
    out_adata = sc.read_h5ad(output_path)
    if "X_uce" not in out_adata.obsm:
        raise SystemExit(
            f"obsm['X_uce'] missing in {output_path}. UCE may have "
            "failed silently — check the subprocess stdout above."
        )

    uce_emb = np.asarray(out_adata.obsm["X_uce"], dtype=np.float32)
    n_nan = int(np.isnan(uce_emb).any(axis=1).sum())
    print(f"  raw UCE embedding shape: {uce_emb.shape}")
    if n_nan > 0:
        print(f"  WARNING: {n_nan}/{uce_emb.shape[0]} cells "
              f"({100 * n_nan / uce_emb.shape[0]:.1f}%) have NaN "
              "embeddings. This typically happens when the gene panel "
              "is too small for UCE's sampling regime. Consider "
              "reducing --uce-sample-size / --uce-pad-length.")

    # 6. Align to original adata if cells were dropped/reordered.
    if uce_emb.shape[0] != adata.n_obs or not (
        out_adata.obs_names.equals(adata.obs_names)
    ):
        common = adata.obs_names.intersection(out_adata.obs_names)
        n_aligned = len(common)
        if n_aligned < adata.n_obs:
            print(f"  WARNING: only {n_aligned}/{adata.n_obs} cells "
                  "survived UCE preprocessing; aligning by obs_names "
                  "(missing cells -> NaN row in latent).")
        idx_out = out_adata.obs_names.get_indexer(common)
        idx_orig = adata.obs_names.get_indexer(common)
        full = np.full((adata.n_obs, uce_emb.shape[1]), np.nan,
                       dtype=uce_emb.dtype)
        full[idx_orig] = uce_emb[idx_out]
        uce_emb = full

    return uce_emb


# ---------------------------------------------------------------------------
# NaN-row scrubbing for downstream Leiden / UMAP / iLISI / MMD
# ---------------------------------------------------------------------------

def _scrub_nan_rows(
        latent: np.ndarray,
        fill_strategy: str = "fail_loud",
        max_nan_fraction: float = 0.02,
    ) -> np.ndarray:
    """Handle rows of `latent` that are entirely NaN — these are cells
    UCE filtered out via `--filter true` (its internal QC drops cells
    with too few genes in the species_chrom.csv).

    `fill_strategy='fail_loud'` (default): if more than
    `max_nan_fraction` (default 2%) of rows are NaN, abort with a clear
    error pointing the user at the right knob to flip. The historical
    fallback ('mean': replace all NaN rows with the column-wise mean of
    valid rows) is a footgun on spatial panels: it turns 30%+ of cells
    into identical-mean embeddings, which Leiden merges into one
    cluster that destroys NMI vs cell_type. Use 'mean' only if you
    explicitly want that behaviour.

    Spatial-data heuristic: if you see this error, set
    `--no-uce-filter-cells` (don't drop low-count cells) and reduce
    `--uce-sample-size` / `--uce-pad-length` to match your panel
    (defaults assume whole-transcriptome scRNA-seq, ~5-20k genes).
    """
    if not np.isnan(latent).any():
        return latent

    nan_rows = np.isnan(latent).any(axis=1)
    n_nan = int(nan_rows.sum())
    n_total = latent.shape[0]
    frac_nan = n_nan / n_total

    if fill_strategy == "fail_loud":
        if frac_nan > max_nan_fraction:
            raise SystemExit(
                f"\n*** UCE produced NaN embeddings for "
                f"{n_nan}/{n_total} ({100*frac_nan:.1f}%) cells, well "
                f"above the {100*max_nan_fraction:.1f}% threshold. ***\n"
                "\n"
                "Most likely cause for spatial transcriptomics: UCE's "
                "default `--filter true` drops cells with too few genes "
                "in its species_chrom.csv, AND the default cell-sentence "
                "length (sample_size=1024, pad_length=1536) is far "
                "larger than your panel size, so most cells fail UCE's "
                "internal coverage check. Mitigations:\n"
                "  1. Pass `--no-uce-filter-cells` to disable the\n"
                "     internal cell-quality filter.\n"
                "  2. Reduce `--uce-sample-size` to ~64-128 (match\n"
                "     your panel's expressed-genes-per-cell).\n"
                "  3. Reduce `--uce-pad-length` to ~128-256 (just\n"
                "     above sample_size).\n"
                "\n"
                "If you genuinely want the old 'fill NaN rows with the\n"
                "column-mean' behaviour (NOT recommended — creates a\n"
                "fake mega-cluster of identical embeddings that\n"
                "destroys cell-type NMI), call _scrub_nan_rows with\n"
                "fill_strategy='mean'."
            )
        # ≤ threshold — warn and drop the rows from downstream metrics
        # by leaving them as NaN. (Caller is expected to mask these
        # cells before sc.pp.neighbors / Leiden.)
        print(f"  NaN-row scrub: {n_nan}/{n_total} "
              f"({100*frac_nan:.2f}%) cells have NaN embeddings; "
              "leaving as NaN (caller should mask before clustering).")
        return latent

    if fill_strategy == "mean":
        valid = latent[~nan_rows]
        if valid.shape[0] == 0:
            raise SystemExit(
                "ALL UCE embeddings are NaN — cannot fall back to "
                "mean-fill. Reduce --uce-sample-size / --uce-pad-length "
                "or check that your gene symbols match UCE's "
                "species_chrom.csv."
            )
        fill = valid.mean(axis=0).astype(latent.dtype)
        out = latent.copy()
        out[nan_rows] = fill
        print(f"  NaN-row scrub (mean-fill): replaced {n_nan}/{n_total} "
              "all-NaN rows with column-wise mean of valid rows. "
              "WARNING: these cells now share an identical embedding "
              "and Leiden will merge them into one fake cluster.")
        return out

    raise ValueError(
        f"_scrub_nan_rows: unknown fill_strategy={fill_strategy!r} "
        "(expected 'fail_loud' or 'mean')."
    )


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
    # UCE-specific.
    p.add_argument("--model-loc", type=Path, required=True,
                   help="Path to the UCE pretrained .torch weights "
                        "(e.g. 33l_8ep_1024t_1280.torch for the "
                        "33-layer / 1280-dim model).")
    p.add_argument("--model-files-dir", type=Path, required=True,
                   help="Directory holding UCE's support files: "
                        "species_chrom.csv, species_offsets.pkl, "
                        "all_tokens.torch, protein_embeddings/, plus "
                        "the model .torch. Symlinked into work_dir as "
                        "model_files/ so UCE doesn't try to download "
                        "from Google Drive.")
    p.add_argument("--uce-species", type=str, default="mouse",
                   choices=["human", "mouse", "zebrafish",
                            "mouse_lemur", "macaca_fascicularis",
                            "macaca_mulatta", "xenopus_tropicalis", "pig"],
                   help="Species token UCE uses (controls which "
                        "species_chrom.csv rows + protein-embedding "
                        "file are used). Default 'mouse'. UCE has "
                        "native multi-species support, so just set "
                        "this to whatever species your data is from "
                        "and pass the corresponding gene symbols "
                        "(mouse-cased for mouse, uppercase HGNC for "
                        "human, etc.).")
    p.add_argument("--uce-model-size", type=str, default="large",
                   choices=["small", "large"],
                   help="'small' = 4-layer / 1024-dim. "
                        "'large' = 33-layer / 1280-dim (paper default).")
    p.add_argument("--uce-batch-size", type=int, default=25,
                   help="Per-GPU batch size (UCE recommends 25 for "
                        "the 33-layer model on an 80 GB A100, 100 "
                        "for the 4-layer model).")
    p.add_argument("--uce-sample-size", type=int, default=1024,
                   help="Number of genes sampled per cell to "
                        "construct the cell sentence. Default 1024 "
                        "(matches UCE upstream). On a targeted "
                        "spatial panel (~400 genes) this oversamples "
                        "with replacement, but empirically the "
                        "upstream value still gives the cleanest "
                        "comparison to UCE's published embeddings.")
    p.add_argument("--uce-pad-length", type=int, default=1536,
                   help="Total cell-sentence length after padding. "
                        "Default 1536 (matches UCE upstream).")
    p.add_argument("--uce-filter-cells", action="store_true", default=False,
                   help="Run UCE's internal cell-quality filter "
                        "(`sc.pp.filter_genes(min_cells=10)` + "
                        "`sc.pp.filter_cells(min_genes=25)`). UCE "
                        "upstream defaults this ON because it was "
                        "designed for whole-transcriptome scRNA-seq "
                        "(~20k genes), where `min_genes=25` is a low "
                        "noise floor. We default OFF: this benchmark "
                        "runs on targeted spatial panels (~400 genes) "
                        "where `min_genes=25` becomes a coverage "
                        "filter rather than a quality filter and "
                        "drops ~30%+ of cells regardless of true "
                        "quality. Flip ON only if you genuinely want "
                        "the upstream behaviour. Note: the upstream "
                        "`--filter` argparse declaration is `type=bool` "
                        "which mishandles 'False' → True; we work "
                        "around this by forwarding '' (empty str) "
                        "instead, which `bool('')` resolves to False "
                        "correctly.")
    p.add_argument("--no-uce-filter-cells", dest="uce_filter_cells",
                   action="store_false",
                   help="(default) Disable UCE's internal cell "
                        "filter — appropriate for targeted spatial "
                        "panels.")
    p.add_argument("--uce-force-rerun", action="store_true",
                   help="Re-run UCE even if cached output exists in "
                        "the work_dir (useful when changing "
                        "--uce-sample-size / --uce-pad-length).")
    p.add_argument("--uce-work-dir", type=Path, default=None,
                   help="Where UCE writes intermediate files. Default: "
                        "system tempdir (auto-cleaned at exit).")
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
    print(f"Run dir         : {args.out_dir}")
    print(f"Seeds           : {seeds}")
    print(f"UCE model       : {args.model_loc}")
    print(f"UCE model_files : {args.model_files_dir}")
    print(f"UCE species     : {args.uce_species}  (UCE handles species "
          "natively via species_chrom.csv + protein_embeddings)")

    # 1. Load + concat. NO normalisation: UCE expects raw counts.
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from obs.")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    # 2. Embed ONCE with UCE (subprocess; ~deterministic). UCE handles
    #    species natively — no ortholog mapping needed for mouse data.
    print("\n=== UCE zero-shot embedding ===")
    if args.uce_work_dir is None:
        # Use a tempdir that auto-cleans at process exit. UCE writes a
        # few hundred MB to model_files/ symlink + a couple of GB of
        # cell-sentence intermediates per run.
        tmp_work = Path(tempfile.mkdtemp(prefix="uce_emb_"))
        cleanup_work = True
    else:
        tmp_work = Path(args.uce_work_dir)
        tmp_work.mkdir(parents=True, exist_ok=True)
        cleanup_work = False
    print(f"  work_dir: {tmp_work}  (cleanup_at_exit={cleanup_work})")

    try:
        latent = _embed_with_uce(
            adata=adata,
            model_loc=args.model_loc,
            model_files_dir=args.model_files_dir,
            species=args.uce_species,
            model_size=args.uce_model_size,
            batch_size=args.uce_batch_size,
            sample_size=args.uce_sample_size,
            pad_length=args.uce_pad_length,
            filter_cells=args.uce_filter_cells,
            force_rerun=args.uce_force_rerun,
            work_dir=tmp_work,
        )
        # Restrict EVERYTHING downstream (Leiden / UMAP / NMI / ARI /
        # iLISI / MMD, plus per-seed h5ad/UMAP exports) to cells that
        # actually got a valid UCE embedding. UCE's internal
        # preprocessing can drop cells (chromosome coverage /
        # gene-mapping / min-counts filter), in which case
        # `_embed_with_uce` aligns by obs_names and leaves NaN rows for
        # the missing cells. We also reject any non-finite (Inf) row
        # defensively — a non-finite embedding row poisons downstream
        # Leiden (NaN distances) and creates fake mega-clusters under
        # mean-fill imputation. The cleanest fix is to subset `adata`
        # itself, so EVERY downstream call (`sc.pp.neighbors`,
        # `sc.tl.umap`, niche/batch metrics, `adata.write_h5ad` per
        # seed) sees only the valid-embedding cells.
        valid_mask = np.isfinite(latent).all(axis=1)
        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            n_total = adata.n_obs
            frac = 100.0 * n_invalid / n_total
            print(f"  Excluding {n_invalid}/{n_total} ({frac:.1f}%) "
                  "cells without a valid UCE embedding (NaN/Inf rows "
                  "from UCE's internal preprocessing). Downstream "
                  "metrics, UMAPs, and per-seed h5ad exports are "
                  f"restricted to the {n_total - n_invalid} cells with "
                  "valid embeddings only.")
            if n_invalid == n_total:
                raise SystemExit(
                    "ALL UCE embeddings are NaN/Inf — UCE rejected "
                    "every cell. Check gene-symbol mapping against "
                    "UCE's species_chrom.csv, or pass "
                    "--no-uce-filter-cells."
                )
            adata = adata[valid_mask].copy()
            latent = latent[valid_mask]
        # Defensive postcondition: anything that reaches Leiden /
        # neighbors must be finite. This is cheap and prevents future
        # regressions where a NaN-handling change silently re-introduces
        # bad rows.
        assert np.isfinite(latent).all(), (
            "UCE latent still contains non-finite values after "
            "valid-embedding subset — refusing to run downstream "
            "metrics on poisoned input."
        )
    finally:
        if cleanup_work:
            try:
                shutil.rmtree(tmp_work)
            except OSError:
                pass

    adata.obsm[DEFAULT_LATENT_KEY] = latent
    print(f"  attached: adata.obsm[{DEFAULT_LATENT_KEY!r}].shape = "
          f"{adata.obsm[DEFAULT_LATENT_KEY].shape}")

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
        seed_t0 = time.time()

        leiden_key, n_found, resolution = _leiden_binary_search_on_latent(
            adata, n_clusters=args.n_clusters,
            n_neighbors=args.n_neighbors,
            latent_key=DEFAULT_LATENT_KEY, seed=seed,
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

        seed_seconds = time.time() - seed_t0
        _record_seed_runtime(
            runtime_tracker, seed=seed, seconds=seed_seconds,
            run_dir=args.out_dir, method="UCE",
        )
        print(f"  runtime (seed {seed}): {seed_seconds:.1f}s")

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
        adata.uns["uce_n_clusters"] = int(seed0_state["n_found"])
        adata.uns["uce_resolution"] = float(seed0_state["resolution"])
        adata.uns["uce_seeds"] = seeds
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
                       "description": "UCE zero-shot baseline (multi-seed)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "UCE"},
        "uce": {
            "model_loc":             str(args.model_loc),
            "model_files_dir":       str(args.model_files_dir),
            "uce_species":           args.uce_species,
            "uce_model_size":        args.uce_model_size,
            "uce_batch_size":        int(args.uce_batch_size),
            "uce_sample_size":       int(args.uce_sample_size),
            "uce_pad_length":        int(args.uce_pad_length),
            "uce_filter_cells":      bool(args.uce_filter_cells),
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
