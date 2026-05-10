"""
scGPT (foundation-model) baseline for the SQUINT cell-identification
benchmark — zero-shot embedding mode.

scGPT is a transformer pre-trained on tens of millions of single-cell
profiles (Cui et al., 2024). In zero-shot mode (no fine-tuning) it
exposes a forward-pass embedder that maps each cell's gene-expression
vector to a learned latent representation; we then cluster + evaluate
that latent the same way we do for PCA / Harmony / scVI.

Reference workflow (canonical scGPT README):
    https://github.com/bowang-lab/scGPT
The README documents `scgpt.tasks.embed_data(adata, model_dir, ...)`
as the public entry point for zero-shot inference. We use only that.

Why per-seed variance is mostly downstream
-----------------------------------------
Unlike scVI (which re-trains per seed), scGPT zero-shot is a
forward-pass-only embedder — the latent is approximately deterministic
modulo CUDA-attention non-determinism (~1e-4 per dim, negligible for
cluster identity). So we embed ONCE up front (matching the PCA + Leiden
script's "embed once, vary downstream" strategy) and the seed loop only
varies the kNN graph + Leiden + UMAP + iLISI/MMD subsampling. That's
where the actual variance lives for this baseline.

Pre-requisites
--------------
1. `pip install scgpt` (and its torch + flash-attn deps; see the
   scGPT install instructions for GPU-specific notes).
2. Download a pre-trained checkpoint directory (~1.5 GB). The
   "whole-human" pretraining is the canonical default; download
   instructions are on the scGPT GitHub. Pass its path via
   `--model-dir`.
3. Gene names matching the checkpoint's vocabulary. For HUMAN data
   (Ensembl HGNC symbols) the human checkpoints work directly. For
   MOUSE data (e.g. mmb-smb), only mouse-symbol matches that the
   tokenizer recognises will contribute — the rest are masked. A
   ortholog-aware preprocessing layer would close this gap; out of
   scope for this baseline.

Output layout mirrors the other baselines
(run_pca_leiden.py / run_harmony.py / run_scvi.py) so
compare_variants.py picks them all up identically.

Approximate runtime: forward-pass embedding takes ~1-3 minutes on a
single GPU for ~100k cells (one-time). Per-seed downstream pipeline
is ~1-2 minutes, so 5 seeds = ~10 minutes total. Way faster than
scVI since there is no training.

Usage:
    python analysis/benchmarking/cell_type_identification/run_scgpt.py \\
        --model-dir /path/to/scGPT_human/

    # different dataset:
    python analysis/benchmarking/cell_type_identification/run_scgpt.py \\
        --silver-dir /nfs/.../silver/chl59-8b_1p \\
        --dataset-tag chl59-8b_1p \\
        --model-dir /path/to/scGPT_human/

    # if your gene symbol lives in adata.var["gene_name"] instead of var_names:
    python analysis/benchmarking/cell_type_identification/run_scgpt.py \\
        --model-dir /path/.../ --gene-col gene_name
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


DEFAULT_VARIANT_TAG = "baseline-scgpt"
DEFAULT_LATENT_KEY  = "X_scgpt"


# ---------------------------------------------------------------------------
# scGPT pre-embedding QC (shared between run_scgpt.py and run_scgpt_spatial.py)
# ---------------------------------------------------------------------------

def scgpt_vocab_match_and_filter(
        adata: ad.AnnData,
        model_dir: Path,
        gene_col: str,
    ):
    """Diagnose vocab coverage and remove cells with 0 in-vocab
    expressed genes.

    Why: scGPT/scGPT-spatial's `embed_data` does NOT filter cells —
    it just builds a (genes, expressions) sequence from each row's
    nonzero entries and prepends `<cls>`. A cell with 0 non-zero entries
    in the IN-VOCAB gene set ends up as `[<cls>]` only — a 1-token
    sequence. Every such cell receives the same CLS embedding, and
    after the function's L2-normalisation step they collapse onto a
    single point on the unit sphere. Leiden then merges them into one
    fake mega-cluster that destroys cell-type NMI.

    Returns
    -------
    valid_mask : np.ndarray of shape (n_obs,), dtype bool
        True for cells with >=1 in-vocab expressed gene; False for the
        cells we should drop before calling embed_data.
    n_in_vocab : int
        Number of genes in adata.var[gene_col] that map to the model's
        vocab (information for the user — explains coverage).
    n_total_genes : int
        len(adata.var) — for printing match rate.
    """
    from scipy.sparse import issparse
    try:
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
    except ImportError as exc:
        raise SystemExit(
            "Could not import scgpt's GeneVocab — is scgpt installed? "
            f"Original error: {exc}"
        )

    vocab_file = Path(model_dir) / "vocab.json"
    if not vocab_file.is_file():
        raise SystemExit(
            f"vocab.json not found at {vocab_file}. The scGPT-format "
            "checkpoint dir must contain vocab.json + args.json + "
            "best_model.pt."
        )
    vocab = GeneVocab.from_file(vocab_file)

    genes = adata.var[gene_col].astype(str).to_numpy()
    in_vocab = np.asarray([g in vocab for g in genes], dtype=bool)
    n_in_vocab = int(in_vocab.sum())
    n_total = int(len(genes))
    pct = 100.0 * n_in_vocab / max(n_total, 1)
    print(f"  scGPT vocab coverage: {n_in_vocab}/{n_total} "
          f"({pct:.1f}%) genes match the model's vocab "
          f"(size={len(vocab)}).")
    if n_in_vocab == 0:
        raise SystemExit(
            "ZERO genes match scGPT's vocabulary. For mouse data on a "
            "human-pretrained checkpoint you MUST map mouse symbols to "
            "human HGNC via --map-via-human-orthologs. Check that "
            f"adata.var[{gene_col!r}] holds gene SYMBOLS (e.g. 'CD3D'), "
            "not Ensembl IDs."
        )

    # Per-cell count of nonzero entries restricted to in-vocab genes.
    X = adata.X
    if n_in_vocab < n_total:
        X_sub = X[:, in_vocab]
    else:
        X_sub = X
    if issparse(X_sub):
        # nnz per row.
        n_per_cell = np.asarray((X_sub != 0).sum(axis=1)).ravel()
    else:
        n_per_cell = (np.asarray(X_sub) != 0).sum(axis=1).astype(np.int64)
    valid_mask = n_per_cell > 0
    n_zero = int((~valid_mask).sum())
    if n_zero > 0:
        print(f"  Pre-filter: {n_zero}/{adata.n_obs} cells have 0 "
              "in-vocab nonzero genes — these would produce CLS-only "
              "(degenerate) embeddings that collapse into a fake "
              "mega-cluster after L2-normalisation. Excluding them "
              "from the embedding pass and from all downstream "
              "metrics.")
    else:
        print(f"  Pre-filter: every cell has >=1 in-vocab nonzero "
              "gene — no cells dropped upfront.")
    return valid_mask, n_in_vocab, n_total


# ---------------------------------------------------------------------------
# Mouse -> human symbol mapping (for stock human-only scGPT checkpoints)
# ---------------------------------------------------------------------------

def _mouse_to_human_symbols(adata: ad.AnnData, species: str) -> ad.AnnData:
    """Convert mouse-symbol AnnData -> human-symbol AnnData via:
        1. mouse symbol -> human ENSG (shared helper
           `_nicheformer_embedding.add_human_ortholog_ensembl_ids`,
           which itself runs a 4-step pipeline:
             1a. mygene mouse-DB: mouse symbol -> mouse Ensembl ID
             1b. Ensembl REST homology: mouse Ensembl -> human Ensembl
             1c. mygene human-DB direct: original symbol -> human Ensembl
                 (catches cross-species symbol matches that 1b misses)
             1d. NCBI HomoloGene via mygene: covers Ensembl-curation gaps
           Typical coverage on a 500-gene mouse panel: ~85-92%.)
        2. human ENSG -> human HGNC symbol (mygene human-DB lookup).

    Genes without an ortholog are dropped. When multiple mouse genes
    map to the same human symbol (rare for a 431-gene panel), their
    expression is SUMMED — same convention as the user's
    `compute_scgpt_spatial_embedding_mouse` pipeline.

    Returns a NEW AnnData (n_obs, n_human_genes) with `var_names` =
    human symbols and `var['feature_name']` = same. The original
    `adata.obs` is preserved verbatim.

    Improvements to the shared `add_human_ortholog_ensembl_ids` helper
    (e.g. adding more fallback sources) propagate to scGPT,
    scGPT-spatial, Geneformer, and Nicheformer simultaneously — single
    code path.
    """
    try:
        from _nicheformer_embedding import add_human_ortholog_ensembl_ids
    except ImportError as exc:
        raise SystemExit(
            "--map-via-human-orthologs needs the helper from "
            "_nicheformer_embedding.py (in the same folder). "
            f"Original error: {exc}"
        )
    try:
        import mygene
    except ImportError as exc:
        raise SystemExit(
            "--map-via-human-orthologs needs the `mygene` package. "
            f"Install: `pip install mygene`. Original error: {exc}"
        )
    import scipy.sparse as sp

    # Step 1: mouse symbol -> human ENSG via the shared 4-step helper
    # (mygene mouse-DB -> Ensembl REST homology -> mygene human-DB
    # direct -> NCBI HomoloGene fallback). See the helper's docstring
    # in _nicheformer_embedding.py for full details.
    print("  Step 1/2: mouse symbols -> human ENSG (shared 4-step "
          "helper: mygene mouse + Ensembl REST + mygene human + "
          "NCBI HomoloGene)...")
    add_human_ortholog_ensembl_ids(
        adata, species=species, gene_col_in=None,
        gene_col_out="human_ensembl_id", fallback_to_input=False,
        verbose=True, inplace=True,
    )
    ensg = adata.var["human_ensembl_id"]
    valid = ensg.notna() & ensg.astype(str).str.startswith("ENSG")
    print(f"    {int(valid.sum())}/{len(ensg)} mouse genes mapped to ENSG.")
    adata_h = adata[:, valid.to_numpy()].copy()

    # Step 2: human ENSG -> human HGNC symbol.
    print("  Step 2/2: human ENSG -> human HGNC symbol (mygene)...")
    unique_ensg = adata_h.var["human_ensembl_id"].astype(str).unique().tolist()
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        unique_ensg,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
    )
    res = res[~res.index.duplicated(keep="first")].dropna(subset=["symbol"])
    ensg_to_sym = res["symbol"].to_dict()
    new_syms = adata_h.var["human_ensembl_id"].astype(str).map(ensg_to_sym)
    keep_mask = new_syms.notna().to_numpy()
    print(f"    {int(keep_mask.sum())}/{len(new_syms)} ENSG mapped to "
          "human HGNC symbol.")
    adata_h = adata_h[:, keep_mask].copy()
    new_syms = new_syms[keep_mask].astype(str).to_numpy()

    # Handle duplicates: sum expression across mouse genes that map to
    # the same human symbol.
    unique_syms, inverse = np.unique(new_syms, return_inverse=True)
    n_dup = len(new_syms) - len(unique_syms)
    if n_dup > 0:
        print(f"    {n_dup} duplicate symbols after mapping; summing "
              "expression for duplicates.")
        agg = sp.csr_matrix(
            (np.ones(len(inverse), dtype=np.float32),
             (np.arange(len(inverse)), inverse)),
            shape=(len(inverse), len(unique_syms)),
        )
        X = adata_h.X
        X_human = (X @ agg) if sp.issparse(X) else (sp.csr_matrix(X) @ agg)
    else:
        X_human = adata_h.X

    new_var = pd.DataFrame(
        {"feature_name": unique_syms},
        index=pd.Index(unique_syms, name="gene_name"),
    )
    out = ad.AnnData(
        X=X_human,
        obs=adata_h.obs.copy(),
        var=new_var,
        obsm=dict(adata_h.obsm),
    )
    print(f"  -> human-symbol AnnData: n_obs={out.n_obs}, "
          f"n_vars={out.n_vars} (from original {adata.n_vars} mouse genes).")
    return out


# ---------------------------------------------------------------------------
# scGPT zero-shot embedding
# ---------------------------------------------------------------------------

def _embed_with_scgpt(
        adata: ad.AnnData,
        model_dir: Path,
        gene_col: Optional[str],
        batch_size: int,
        max_length: int,
        device: Optional[str],
        latent_key: str,
    ) -> np.ndarray:
    """Zero-shot embedding via the scGPT public API.

    `scgpt.tasks.embed_data` (the canonical entry point per the scGPT
    GitHub README) loads the pretrained checkpoint at `model_dir` and
    runs a forward pass over the cells in `adata`, writing the latent
    to `obsm[latent_key]` (default `X_scGPT` in scGPT, but we use the
    user-supplied key for parity with the other baselines).

    Parameters mirror the public API. We DO NOT pass any params not
    documented in the README so the call stays compatible across
    minor scGPT version bumps.
    """
    try:
        from scgpt.tasks import embed_data  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "scGPT is not installed in this environment. Install via "
            "`pip install scgpt` (see GitHub for GPU / flash-attn "
            "notes), then re-run. Original error: " + str(exc)
        )

    print(f"  scGPT zero-shot embed: model_dir={model_dir}")
    print(f"    n_cells={adata.n_obs}, n_genes={adata.n_vars}, "
          f"batch_size={batch_size}, max_length={max_length}")

    # scGPT's `embed_data` requires the gene-symbol column to live in
    # `adata.var[gene_col]` (it asserts `gene_col in adata.var`).
    # Default scGPT gene_col is "feature_name". If the caller didn't
    # specify --gene-col and there's no matching var column, copy from
    # var_names so the assertion passes.
    effective_gene_col = gene_col if gene_col is not None else "feature_name"
    if effective_gene_col not in adata.var.columns:
        print(f"  adata.var has no {effective_gene_col!r} column — "
              f"copying from var_names ({adata.var_names[:3].tolist()}...).")
        adata.var[effective_gene_col] = adata.var_names.astype(str)

    # Densify X if sparse — matches the canonical scGPT benchmarking
    # notebook's `adata.X = adata.X.toarray()` step. The dense path is
    # the one exercised by the README example; forcing dense here
    # removes that as a source of variance.
    try:
        from scipy.sparse import issparse as _sp_issparse
        if _sp_issparse(adata.X):
            print("  densifying adata.X (was sparse) to match the "
                  "notebook's pre-embed densification.")
            adata.X = adata.X.toarray()
    except Exception:
        pass

    # Defaults match the notebook recipe:
    #   - obs_to_save = list(adata.obs.columns) — preserves cell-level
    #     metadata in the returned AnnData.
    #   - max_length is ONLY passed when the caller set it explicitly
    #     (max_length is None by default at the CLI). Letting the
    #     library pick its own default avoids subtly changing
    #     tokenization vs the notebook recipe.
    embed_kwargs: Dict = dict(
        adata_or_file=adata,
        model_dir=str(model_dir),
        gene_col=effective_gene_col,
        batch_size=int(batch_size),
        obs_to_save=list(adata.obs.columns),
        return_new_adata=True,
    )
    if max_length is not None:
        embed_kwargs["max_length"] = int(max_length)
    if device is not None:
        embed_kwargs["device"] = device

    out = embed_data(**embed_kwargs)
    # `embed_data` returns a NEW AnnData with the latent on `.X`
    # (since `return_new_adata=True`). The latent matrix has shape
    # (n_obs, embed_dim).
    latent = np.asarray(out.X)
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
    # scGPT-specific.
    p.add_argument(
        "--model-dir", type=Path, required=True,
        help="Path to a downloaded scGPT pretrained checkpoint "
             "directory (e.g. the whole_human checkpoint). See "
             "https://github.com/bowang-lab/scGPT for download links.",
    )
    p.add_argument(
        "--gene-col", type=str, default=None,
        help="adata.var column holding gene symbols matching the "
             "scGPT vocab. Default None means var_names is the "
             "symbol column (typical for silver h5ads).",
    )
    p.add_argument("--scgpt-batch-size", type=int, default=64,
                   help="Cells per forward pass (scGPT default 64).")
    p.add_argument("--scgpt-max-length", type=int, default=None,
                   help="Max gene-token sequence length. Default None — "
                        "uses scGPT's own internal default (1200), "
                        "matching the canonical scGPT benchmarking "
                        "notebook which doesn't pass max_length. "
                        "Override only if you have a specific reason: "
                        "for tiny spatial panels (~few hundred genes) "
                        "the cell sentence is shorter than max_length "
                        "anyway, so the default is fine.")
    p.add_argument("--scgpt-device", type=str, default=None,
                   help="Override device. Default lets scGPT pick "
                        "(typically cuda if available, else cpu).")
    p.add_argument("--map-via-human-orthologs", action="store_true",
                   help="Map mouse gene symbols to human HGNC symbols "
                        "via the shared 4-step ortholog pipeline "
                        "(mygene mouse-DB -> Ensembl REST homology -> "
                        "mygene human-DB direct -> NCBI HomoloGene "
                        "fallback) followed by ENSG -> HGNC symbol "
                        "lookup. REQUIRED for stock human-only scGPT "
                        "checkpoints (e.g. scGPT_human) on mouse data — "
                        "without this, gene-vocab match rate is 0% "
                        "because scGPT vocab is uppercase HGNC and "
                        "mouse symbols are mixed-case (Gad1 vs GAD1). "
                        "Same shared helper as run_geneformer.py / "
                        "run_nicheformer.py / run_scgpt_spatial.py — "
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

    if not args.model_dir.is_dir():
        raise SystemExit(
            f"--model-dir does not exist or is not a directory: "
            f"{args.model_dir}"
        )

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

    # 1. Load + concat. NO normalisation / scaling / HVG: scGPT expects
    #    raw counts and handles its own preprocessing internally.
    adata = _load_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"\nConcatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(
            f"--batch-key={args.batch_key!r} missing from obs."
        )
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    # 2a. Optionally remap mouse symbols -> human HGNC symbols. Required
    #     for stock human-only scGPT checkpoints on mouse data (else
    #     gene-vocab match rate is 0% and scGPT crashes on empty cells).
    if args.map_via_human_orthologs:
        print("\n=== Mapping mouse genes to human orthologs ===")
        adata_for_embed = _mouse_to_human_symbols(
            adata.copy(), species=args.species,
        )
        # n_obs preserved -> latent rows align 1:1 with the original adata.
        if adata_for_embed.n_obs != adata.n_obs:
            raise SystemExit(
                f"Ortholog mapping changed n_obs ({adata.n_obs} -> "
                f"{adata_for_embed.n_obs}); cannot align latent."
            )
    else:
        adata_for_embed = adata

    # 2b. Pre-embedding QC. Identify cells with 0 in-vocab nonzero
    #     genes and report the vocab match rate. Cells that fail this
    #     check would produce degenerate CLS-only embeddings (see
    #     `scgpt_vocab_match_and_filter` docstring) — drop them upfront
    #     and from `adata` itself so downstream metrics see only cells
    #     with meaningful embeddings.
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

    # 2c. Embed ONCE with scGPT (forward-pass only; ~deterministic).
    print("\n=== scGPT zero-shot embedding ===")
    latent = _embed_with_scgpt(
        adata=adata_for_embed,
        model_dir=args.model_dir,
        gene_col=args.gene_col,
        batch_size=args.scgpt_batch_size,
        max_length=args.scgpt_max_length,
        device=args.scgpt_device,
        latent_key=DEFAULT_LATENT_KEY,
    )

    # 2d. Defensive valid-mask: scGPT's L2-normalisation step
    #     (`emb / np.linalg.norm(emb, axis=1, keepdims=True)`) divides
    #     by zero on any all-zero embedding row, producing NaN/Inf —
    #     same footgun as UCE/Geneformer. Subset adata to cells with
    #     finite embeddings so all downstream computation (Leiden,
    #     UMAP, NMI/ARI, iLISI, MMD, per-seed h5ad) sees only valid
    #     cells.
    valid_mask = np.isfinite(latent).all(axis=1)
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        n_total_post = adata.n_obs
        frac = 100.0 * n_invalid / n_total_post
        print(f"  Excluding {n_invalid}/{n_total_post} ({frac:.1f}%) "
              "cells without a valid scGPT embedding (NaN/Inf rows). "
              "Downstream metrics, UMAPs, and per-seed h5ad exports "
              f"are restricted to the {n_total_post - n_invalid} "
              "cells with valid embeddings only.")
        if n_invalid == n_total_post:
            raise SystemExit(
                "ALL scGPT embeddings are NaN/Inf — likely the L2-norm "
                "step divided by zero for every cell. Check vocab "
                "match rate and that adata.X holds raw counts."
            )
        adata = adata[valid_mask].copy()
        latent = latent[valid_mask]
    assert np.isfinite(latent).all(), (
        "scGPT latent still contains non-finite values after "
        "valid-embedding subset — refusing to run downstream metrics "
        "on poisoned input."
    )

    # Attach to the ORIGINAL (mouse-symbol) adata so downstream Leiden /
    # UMAP / metrics see the unmodified obs/X.
    adata.obsm[DEFAULT_LATENT_KEY] = latent

    # 3. Per-seed loop on the (shared) scGPT latent. Variance comes
    #    from kNN graph + Leiden + UMAP + iLISI/MMD subsampling.
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

        seed_seconds = time.time() - seed_t0
        _record_seed_runtime(
            runtime_tracker, seed=seed, seconds=seed_seconds,
            run_dir=args.out_dir, method="scGPT",
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
        adata.uns["scgpt_n_clusters"]        = int(seed0_state["n_found"])
        adata.uns["scgpt_resolution"]        = float(seed0_state["resolution"])
        adata.uns["scgpt_seeds"]             = seeds
        adata.uns["scgpt_model_dir"]         = str(args.model_dir)
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
                       "description": "scGPT zero-shot baseline (multi-seed downstream)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "scGPT-zero-shot"},
        "scgpt": {
            "model_dir":          str(args.model_dir),
            "gene_col":           args.gene_col,
            "batch_size":         int(args.scgpt_batch_size),
            "max_length":         int(args.scgpt_max_length),
            "device":             args.scgpt_device,
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
