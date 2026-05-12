"""
Geneformer baseline for the SQUINT cell-type-identification benchmark.

Targets **Geneformer V2** (e.g. `Geneformer-V2-104M` or
`Geneformer-V2-95M`) — pretrained on Genecorpus-104M, which is HUMAN
ONLY. The token dictionary contains only `ENSG...` (human Ensembl
gene IDs). For MOUSE data, mouse symbols must therefore be mapped to
their human orthologs first; pass `--map-via-human-orthologs` to do this.
(An earlier version of this script claimed V2 was pan-species; that
was wrong — gc104M's vocab is human-only.)

Reference workflow (canonical Geneformer README on HuggingFace):
    https://huggingface.co/ctheodoris/Geneformer

Pipeline:
  1. Set per-cell `obs["n_counts"]` and per-gene `var["ensembl_id"]`
     (Geneformer's tokenizer reads these directly).
  2. Write a single .h5ad to a temp directory.
  3. `TranscriptomeTokenizer.tokenize_data(...)` -> HuggingFace
     Dataset on disk (rank-value-encoded gene sequences per cell).
  4. `EmbExtractor.extract_embs(...)` -> per-cell embeddings as a
     pandas DataFrame.
  5. Standard 5-seed downstream loop (kNN + Leiden + UMAP + metrics).

Geneformer is forward-pass-only in zero-shot mode, so we EMBED ONCE
(approximately deterministic) and the seed loop only varies the
downstream clustering — same convention as scGPT / Nicheformer /
PCA + Leiden.

Pre-requisites
--------------
1. `pip install geneformer` (and PyTorch + transformers).
2. A downloaded Geneformer V2 checkpoint directory (~500 MB - 1 GB).
   Get one via:
     hf download ctheodoris/Geneformer \\
         --local-dir /nfs/team361/sb75/squint-reproducibility/analysis/benchmarking/geneformer \\
         --include "Geneformer-V2-104M/*"
   Pass the resulting path via `--model-dir`.
3. The package's auxiliary `*.pkl` dictionaries must be the real
   files (not Git-LFS pointer text). If you see
   `_pickle.UnpicklingError: invalid load key, 'v'.` re-download via
   `hf download ... --include "*.pkl"` and copy into the geneformer
   site-packages dir (the LFS pointers are 132 bytes; the real files
   are MB-scale).
4. Ensembl IDs in adata.var. For mouse data:
     - **Recommended**: `--map-via-human-orthologs` to map mouse
       symbols to human ENSG IDs via the Nicheformer helper
       (`_nicheformer_embedding.add_human_ortholog_ensembl_ids` —
       mygene + Ensembl REST homology + NCBI HomoloGene fallbacks).
       Required because V2 is human-only. Slow first call (~1-3 min
       for ~500 genes); needs internet.
     - `--auto-map-symbols` (mygene; mouse symbols -> mouse ENSMUSG)
       is only useful with a mouse-specific Geneformer checkpoint —
       NOT with stock V2 / V1.

Usage:
    # Recommended for mouse data + Geneformer V2:
    python analysis/benchmarking/cell_type_identification/run_geneformer.py \\
        --model-dir /nfs/team361/sb75/squint-reproducibility/analysis/benchmarking/geneformer/Geneformer-V2-104M \\
        --model-input-size 4096 \\
        --map-via-human-orthologs

    # If your var already has Ensembl IDs (e.g. from upstream pipeline):
    python analysis/benchmarking/cell_type_identification/run_geneformer.py \\
        --model-dir /nfs/team361/sb75/squint-reproducibility/analysis/benchmarking/geneformer/Geneformer-V2-104M \\
        --ensembl-id-col human_ensembl_id

    # Custom dataset:
    python analysis/benchmarking/cell_type_identification/run_geneformer.py \\
        --model-dir /nfs/team361/sb75/squint-reproducibility/analysis/benchmarking/geneformer/Geneformer-V2-104M \\
        --silver-dir /alt/silver --dataset-tag chl59-8b_1p
"""

import argparse
import sys
import shutil
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

# Reuse shared helpers.
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


DEFAULT_VARIANT_TAG = "baseline-geneformer"
DEFAULT_LATENT_KEY  = "X_geneformer"


# ---------------------------------------------------------------------------
# transformers compatibility shim (called before importing Geneformer)
# ---------------------------------------------------------------------------

def _shim_transformers_for_geneformer() -> None:
    """Re-attach top-level `transformers` names that newer releases
    moved out of the package `__init__` but kept inside their
    submodules.

    Specifically targets the recurring `cannot import name
    'SpecialTokensMixin' from 'transformers'` failure that hits
    Geneformer on transformers >= ~4.45: the class still lives in
    `transformers.tokenization_utils_base` — this shim sets it back as
    a top-level attribute so `from transformers import SpecialTokensMixin`
    succeeds.

    We also re-attach a few sibling names that have caused the same
    Goldilocks failure across transformers versions in the wild
    (BatchEncoding, PreTrainedTokenizerBase, EncoderDecoderCache).

    Idempotent and safe — no-ops when transformers isn't installed or
    when the names are already at the top level.
    """
    try:
        import transformers
    except ImportError:
        return
    # (top-level name, submodule path)
    candidates = [
        ("SpecialTokensMixin", "transformers.tokenization_utils_base"),
        ("BatchEncoding", "transformers.tokenization_utils_base"),
        ("PreTrainedTokenizerBase", "transformers.tokenization_utils_base"),
        ("EncoderDecoderCache", "transformers.cache_utils"),
        ("Cache", "transformers.cache_utils"),
        ("DynamicCache", "transformers.cache_utils"),
    ]
    for name, src in candidates:
        if hasattr(transformers, name):
            continue
        try:
            mod = __import__(src, fromlist=[name])
            setattr(transformers, name, getattr(mod, name))
        except (ImportError, AttributeError):
            # Submodule may itself not exist, or no longer expose the
            # name; skip silently — the import will then fail with a
            # specific error and the caller's fallback message will
            # tell the user to extend this list.
            pass


# ---------------------------------------------------------------------------
# Geneformer-specific helpers
# ---------------------------------------------------------------------------

def _ensure_ensembl_ids(
        adata: ad.AnnData,
        ensembl_id_col: Optional[str],
        auto_map_symbols: bool,
        species: str,
        map_via_human_orthologs: bool = False,
    ) -> ad.AnnData:
    """Make sure `adata.var["ensembl_id"]` is populated with Ensembl IDs
    that Geneformer's tokenizer can match to its vocabulary. Routes
    (in priority order):
      - `ensembl_id_col` already in adata.var  -> copy that column.
      - `var_names` already look like Ensembl IDs -> copy var_names.
      - `map_via_human_orthologs=True` -> use the Nicheformer helper
        `add_human_ortholog_ensembl_ids` (mygene + Ensembl REST) to
        populate `adata.var['human_ensembl_id']` with HUMAN ENSG IDs,
        then copy that to `ensembl_id`. This is what you want for
        Geneformer V2 on mouse data (V2 is human-only).
      - `auto_map_symbols=True` -> derive via mygene (mouse symbols ->
        MOUSE Ensembl IDs / ENSMUSG). Only useful with a mouse-specific
        Geneformer checkpoint, which V2 isn't.
        SLOW (~1 min per 100 genes) on first call; needs internet.
    Otherwise raises with a clear error.
    """
    if "ensembl_id" in adata.var.columns and adata.var["ensembl_id"].notna().any():
        # Already populated; trust the caller.
        print("  Using pre-existing adata.var['ensembl_id'].")
        return adata

    if ensembl_id_col is not None:
        if ensembl_id_col not in adata.var.columns:
            raise SystemExit(
                f"--ensembl-id-col={ensembl_id_col!r} not in adata.var. "
                f"Available columns: {list(adata.var.columns)[:10]}."
            )
        adata.var["ensembl_id"] = adata.var[ensembl_id_col].astype(str)
        print(f"  Copied adata.var[{ensembl_id_col!r}] -> "
              "adata.var['ensembl_id'].")
        return adata

    # Auto-detect var_names = Ensembl IDs.
    var_names = adata.var_names.astype(str).to_numpy()
    is_ensembl = np.array([
        v.startswith("ENSG") or v.startswith("ENSMUSG")
        for v in var_names
    ], dtype=bool)
    if is_ensembl.mean() > 0.5:
        adata.var["ensembl_id"] = var_names
        print(f"  var_names look like Ensembl IDs "
              f"({int(is_ensembl.sum())}/{len(var_names)} matched); "
              "using var_names as adata.var['ensembl_id'].")
        return adata

    # Mouse data + Geneformer V2 (human-only token dict) -> need to
    # map mouse symbols to HUMAN Ensembl IDs. Reuse the Nicheformer
    # helper, which uses mygene + Ensembl REST homology to populate
    # adata.var['human_ensembl_id'].
    if map_via_human_orthologs:
        if species.lower() not in ("mouse", "mus musculus"):
            print(f"  WARNING: --map-via-human-orthologs given but "
                  f"--species={species!r} (not mouse); the helper "
                  "no-ops for non-mouse and just copies var_names.")
        print("  Mapping mouse genes -> human Ensembl IDs via "
              "_nicheformer_embedding.add_human_ortholog_ensembl_ids "
              "(mygene + Ensembl REST homology, slow first call)...")
        try:
            from _nicheformer_embedding import add_human_ortholog_ensembl_ids
        except ImportError as exc:
            raise SystemExit(
                "--map-via-human-orthologs needs the helper from "
                "_nicheformer_embedding.py (in the same folder). "
                f"Original error: {exc}"
            )
        adata = add_human_ortholog_ensembl_ids(
            adata,
            species=species,
            gene_col_in=None,           # use var_names (= mouse symbols)
            gene_col_out="human_ensembl_id",
            fallback_to_input=True,     # leave unmapped genes as their
                                        # input symbol; Geneformer's
                                        # tokenizer drops non-vocab
                                        # entries automatically.
            verbose=True,
            inplace=True,
        )
        ens_array = adata.var["human_ensembl_id"].astype(str).to_numpy()
        adata.var["ensembl_id"] = ens_array
        n_mapped = int(sum(str(v).startswith("ENSG") for v in ens_array))
        print(f"  Mapped {n_mapped}/{len(ens_array)} mouse genes to "
              f"human ENSG IDs.")
        if n_mapped == 0:
            raise SystemExit(
                "Zero genes mapped to human ENSG. Either the input "
                "symbols are unrecognised by mygene, or there's no "
                "internet access for the Ensembl REST homology call. "
                "Inspect adata.var['human_ensembl_id'] to debug."
            )
        return adata

    if auto_map_symbols:
        print("  Auto-mapping gene symbols -> mouse Ensembl IDs via mygene...")
        try:
            from _nicheformer_embedding import add_human_ortholog_ensembl_ids
        except ImportError as exc:
            raise SystemExit(
                "Auto-map needs the helper from _nicheformer_embedding.py "
                "(it reuses the mygene-based mapping). "
                f"Original error: {exc}"
            )
        # Use the Nicheformer helper but keep the MOUSE Ensembl IDs
        # (we don't want human orthologs for Geneformer V2 — its mouse
        # tokens use ENSMUSG directly). The helper's internal "Step 1"
        # output (symbol -> mouse Ensembl) is what we want, but it's
        # not exposed cleanly. Workaround: call mygene directly here.
        try:
            import mygene
        except ImportError as exc:
            raise SystemExit(
                "Auto-map needs the `mygene` package. "
                f"Install with `pip install mygene`. Original error: {exc}"
            )
        symbols = adata.var_names.astype(str).to_numpy()
        mg = mygene.MyGeneInfo()
        species_for_query = "mouse" if species.lower() in ("mouse", "mus musculus") else "human"
        df = mg.querymany(
            symbols.tolist(),
            scopes=["symbol", "alias", "ensembl.gene"],
            species=species_for_query,
            fields="ensembl.gene",
            returnall=False, as_dataframe=True,
        )
        def _first_ens(x):
            if isinstance(x, list) and x and isinstance(x[0], dict):
                return x[0].get("gene")
            if isinstance(x, dict):
                return x.get("gene")
            return x if isinstance(x, str) else None
        col = "ensembl.gene" if "ensembl.gene" in df.columns else "ensembl"
        sym_to_ens = df[col].apply(_first_ens) if col in df.columns else pd.Series(dtype=str)
        sym_to_ens = sym_to_ens[~sym_to_ens.index.duplicated(keep="first")]
        ens_array = np.array([
            str(sym_to_ens.get(s, s)) for s in symbols
        ], dtype=object)
        adata.var["ensembl_id"] = ens_array
        n_mapped = int(sum(
            (str(s).startswith("ENSG") or str(s).startswith("ENSMUSG"))
            for s in ens_array
        ))
        print(f"  Mapped {n_mapped}/{len(symbols)} symbols to Ensembl IDs.")
        return adata

    raise SystemExit(
        "Could not resolve Ensembl IDs for adata.var. Provide ONE of:\n"
        "  --ensembl-id-col <var-col-with-Ensembl-IDs>\n"
        "  --map-via-human-orthologs  (mouse symbols -> human ENSG; "
        "REQUIRED for Geneformer V2 on mouse data, since V2 is human-only)\n"
        "  --auto-map-symbols         (mouse symbols -> mouse ENSMUSG; "
        "only useful with a mouse-specific Geneformer checkpoint)\n"
        "or pre-populate adata.var['ensembl_id'] before running."
    )


def _embed_with_geneformer(
        adata: ad.AnnData,
        model_dir: Path,
        model_input_size: int,
        forward_batch_size: int,
        emb_layer: int,
        nproc: int,
    ) -> np.ndarray:
    """Tokenize + extract embeddings with Geneformer V2.

    Uses only the documented public API:
      - `geneformer.TranscriptomeTokenizer.tokenize_data(...)`
      - `geneformer.EmbExtractor.extract_embs(...)`

    Writes the AnnData to a temp directory, runs the two-step pipeline,
    parses the resulting per-cell embedding DataFrame, then cleans up.
    """
    # Defensive shim: re-attach top-level transformers names that newer
    # transformers releases (>= ~4.45) removed from the package
    # __init__ but kept in their respective submodules. Geneformer's
    # imports historically rely on these top-level re-exports, so this
    # avoids a hard "cannot import name 'X' from 'transformers'" failure
    # when the user is on a transformers version that's just slightly
    # newer than Geneformer expects.
    _shim_transformers_for_geneformer()

    try:
        from geneformer import TranscriptomeTokenizer, EmbExtractor
    except ImportError as exc:
        # Distinguish "geneformer not installed" from "geneformer
        # imports a version of transformers that's incompatible with
        # the one in this venv". The latter is a common foot-gun
        # because newer transformers releases (>=4.45) drop public
        # re-exports like `SpecialTokensMixin` that older Geneformer
        # versions still import from the top-level namespace.
        msg = str(exc)
        from_transformers = (
            "transformers" in msg or "SpecialTokensMixin" in msg
            or "tokenization_utils" in msg
        )
        if from_transformers:
            raise SystemExit(
                "Geneformer IS installed, but its `from transformers "
                "import ...` failed because the installed transformers "
                "version dropped a name Geneformer expects, and our "
                "shim couldn't find it in any of the usual submodules.\n"
                "Try one of:\n"
                "  1) Pin transformers to a known-good range:\n"
                "       uv pip install 'transformers>=4.37,<4.41'\n"
                "  2) Check the actual missing name: it's printed in the\n"
                "     ImportError below. Then look at where it lives in\n"
                "     your installed transformers (e.g. via\n"
                "     `python -c \"import transformers, sys; "
                "print([m for m in sys.modules if 'transformers' in m])\"`)"
                " and add it to `_shim_transformers_for_geneformer()` "
                "in this file.\n"
                "  3) Inspect Geneformer's pin via\n"
                "     `python -c \"import importlib.metadata as m; "
                "print(m.metadata('geneformer').get_all('Requires-Dist'))\"`.\n"
                f"Original ImportError: {exc}"
            )
        raise SystemExit(
            "Geneformer is not installed (or not importable). Install "
            "via `pip install git+https://huggingface.co/ctheodoris/"
            "Geneformer` (see HuggingFace README for torch + "
            "transformers version notes). "
            f"Original error: {exc}"
        )

    # Geneformer's tokenizer requires obs["n_counts"] (total raw counts
    # per cell) and var["ensembl_id"]. Add n_counts if absent.
    if "n_counts" not in adata.obs.columns:
        from scipy import sparse
        X = adata.X
        n_counts = (np.asarray(X.sum(axis=1)).ravel()
                    if not sparse.issparse(X)
                    else np.asarray(X.sum(axis=1)).ravel())
        adata.obs["n_counts"] = n_counts.astype(np.int64)
        print("  Added obs['n_counts'] from row sums of adata.X.")

    if "ensembl_id" not in adata.var.columns:
        raise SystemExit(
            "adata.var['ensembl_id'] is missing -- _ensure_ensembl_ids "
            "should have populated it. This is a script bug."
        )

    # CRITICAL: Geneformer's `extract_embs` returns a DataFrame with a
    # default RangeIndex (0..N-1) over POST-FILTER, POST-DOWNSAMPLE
    # cells — it does NOT preserve adata.obs_names order. The tokenizer
    # silently drops cells with too few expressed genes after gene-vocab
    # filtering, and the only way to recover the true cell-to-embedding
    # mapping is to carry `obs_names` through tokenization explicitly via
    # `custom_attr_name_dict` and then ask `EmbExtractor` to attach it as
    # a column via `emb_label`. The previous "embs_df.index.astype(int)"
    # alignment was wrong: it assigned the i-th surviving cell's
    # embedding to adata row i (a near-random permutation), and zero-
    # padded the trailing rows — destroying cell-type NMI on any panel
    # where the tokenizer drops cells (almost always, for spatial data).
    cell_id_col = "geneformer_cell_id_orig"
    adata.obs[cell_id_col] = adata.obs_names.astype(str).to_numpy()

    # Two-step pipeline runs on filesystem; use a temp dir we clean up.
    tmpdir = Path(tempfile.mkdtemp(prefix="geneformer_emb_"))
    try:
        h5ad_dir = tmpdir / "h5ads"; h5ad_dir.mkdir()
        out_dir  = tmpdir / "tokens"; out_dir.mkdir()

        # The tokenizer reads ALL h5ads in `data_directory`; we put
        # exactly one there.
        h5ad_path = h5ad_dir / "input.h5ad"
        # Sanitize before write: pandas may hand us ArrowStringArray
        # columns (default for newer pandas + pyarrow), which the H5AD
        # writer can't serialize. Same shared helper used at the
        # `predicted_adata.h5ad` site.
        _sanitize_for_h5ad(adata)
        adata.write_h5ad(h5ad_path)
        print(f"  wrote temp h5ad: {h5ad_path}")

        # Step 1: tokenize. `model_input_size` MUST match the
        # checkpoint's context length (V2-95M-i4096 -> 4096; V1 -> 2048).
        # `special_token=True` adds the CLS-like special token V2 was
        # trained with; if your checkpoint is V1 set --model-input-size
        # 2048 and disable special_token via env if needed.
        # `custom_attr_name_dict` carries obs[cell_id_col] (the original
        # obs_names string) through tokenization so we can recover the
        # cell-to-embedding mapping after some cells get dropped by the
        # tokenizer's min-expressed-gene filter. Without this we cannot
        # align Geneformer's filtered output back to adata.obs order.
        print(f"  tokenizing (model_input_size={model_input_size}, nproc={nproc})...")
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={cell_id_col: cell_id_col},
            nproc=int(nproc),
            model_input_size=int(model_input_size),
            special_token=True,
        )
        tk.tokenize_data(
            data_directory=str(h5ad_dir),
            output_directory=str(out_dir),
            output_prefix="data",
            file_format="h5ad",
        )
        token_dataset = out_dir / "data.dataset"
        if not token_dataset.exists():
            # Some Geneformer versions write directly to .arrow rather than
            # a `.dataset` directory; check both.
            candidates = list(out_dir.glob("data*"))
            raise RuntimeError(
                f"tokenized output not found at {token_dataset}. "
                f"Files in {out_dir}: {[p.name for p in candidates]}"
            )

        # Step 2: extract embeddings.
        print(f"  extracting embeddings (model_dir={model_dir}, "
              f"layer={emb_layer}, batch_size={forward_batch_size})...")
        emb_out_dir = tmpdir / "embs"; emb_out_dir.mkdir()
        # Geneformer V2 trains with a CLS-like <cls> special token at
        # position 0 (we set `special_token=True` in the tokenizer above),
        # and the model learns to put cell-level information there. To
        # get the CLS embedding we set `emb_mode="cls"` — NOT
        # `cell_emb_style="cls"`. (Geneformer's API is:
        # `emb_mode={"cls", "cell", "gene"}` controls pooling;
        # `cell_emb_style` only accepts `"mean_pool"` per its
        # `valid_option_dict`. Earlier versions of this script set
        # `emb_mode="cell" + cell_emb_style="cls"`, which raises in the
        # validator and silently falls back to `mean_pool` — so the CLS
        # path was never actually taken. Fixed.)
        # `emb_label=[cell_id_col]` makes the returned DataFrame include
        # a column with the original adata obs_names, which we need to
        # align embeddings back to adata. (The default RangeIndex on the
        # DataFrame is post-filter / post-downsample order, NOT adata
        # order — using it as a row-index is the bug that randomized
        # cell-to-embedding correspondence in earlier versions.)
        # Fallback: if `emb_mode="cls"` is rejected (older geneformer
        # versions don't have it), drop to `emb_mode="cell"` (mean-pool
        # over gene tokens, V1 behaviour). Still correct — just not
        # leveraging V2's CLS training signal.
        try:
            embex = EmbExtractor(
                model_type="Pretrained",
                num_classes=0,
                emb_mode="cls",
                max_ncells=None,
                emb_layer=int(emb_layer),
                forward_batch_size=int(forward_batch_size),
                nproc=int(nproc),
                emb_label=[cell_id_col],
                summary_stat=None,
            )
            print("  EmbExtractor: emb_mode='cls' (V2-recommended CLS pooling)")
        except Exception as exc:
            print(f"  EmbExtractor: emb_mode='cls' not supported in this "
                  f"Geneformer version ({type(exc).__name__}); falling "
                  "back to emb_mode='cell' (mean-pool over gene tokens, "
                  "V1 behaviour). For V2 CLS pooling, upgrade geneformer "
                  "to a version that includes 'cls' in valid emb_mode "
                  "options.")
            embex = EmbExtractor(
                model_type="Pretrained",
                num_classes=0,
                emb_mode="cell",
                max_ncells=None,
                emb_layer=int(emb_layer),
                forward_batch_size=int(forward_batch_size),
                nproc=int(nproc),
                emb_label=[cell_id_col],
                summary_stat=None,
            )
        embs_df = embex.extract_embs(
            model_directory=str(model_dir),
            input_data_file=str(token_dataset),
            output_directory=str(emb_out_dir),
            output_prefix="data",
        )
        if not isinstance(embs_df, pd.DataFrame):
            raise RuntimeError(
                f"EmbExtractor returned unexpected type "
                f"{type(embs_df).__name__}; expected pandas DataFrame."
            )
        if cell_id_col not in embs_df.columns:
            raise RuntimeError(
                f"Geneformer's returned DataFrame does not contain the "
                f"`{cell_id_col}` column we asked for via "
                "`emb_label`. Without it we cannot align embeddings "
                "back to adata.obs. Available columns: "
                f"{list(embs_df.columns)[:10]}..."
            )
        # Embedding columns are the integer-named columns produced by
        # `pd.DataFrame(np.ndarray)`. Everything else (cell_id_col, plus
        # any other emb_label) is metadata.
        emb_cols = [c for c in embs_df.columns if c != cell_id_col]
        emb_array = embs_df[emb_cols].to_numpy(dtype=np.float32, copy=True)
        returned_ids = embs_df[cell_id_col].astype(str).to_numpy()
        n_returned, n_dim = emb_array.shape
        n_total = adata.n_obs
        # Build alignment: original adata obs_name -> embedding row.
        id_to_row = {oid: i for i, oid in enumerate(returned_ids)}
        adata_obs = adata.obs_names.astype(str).to_numpy()
        full = np.full((n_total, n_dim), np.nan, dtype=np.float32)
        n_aligned = 0
        for i, oid in enumerate(adata_obs):
            row = id_to_row.get(oid)
            if row is not None:
                full[i] = emb_array[row]
                n_aligned += 1
        if n_returned != n_total or n_aligned != n_total:
            print(f"  Geneformer returned {n_returned} embeddings for "
                  f"{n_total} input cells; aligned {n_aligned} by "
                  f"obs_names. Cells without an embedding "
                  f"({n_total - n_aligned}) get NaN rows — caller must "
                  "subset adata to valid-embedding cells before running "
                  "downstream metrics.")
        else:
            print(f"  Geneformer returned all {n_returned} embeddings; "
                  "alignment by obs_names complete.")
        latent = full
        print(f"  embedding shape: {latent.shape} "
              f"(NaN rows: {n_total - n_aligned}/{n_total})")
        return latent
    finally:
        # Clean up tokenized intermediates (can be GBs for big datasets).
        try:
            shutil.rmtree(tmpdir)
        except OSError:
            pass


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
    # Geneformer-specific.
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Path to a downloaded Geneformer V2 checkpoint "
                        "directory (e.g. Geneformer-V2-104M/). Get one "
                        "via `hf download ctheodoris/Geneformer "
                        "--local-dir <PARENT_DIR> --include "
                        "'Geneformer-V2-104M/*'`. Example default: "
                        "/nfs/team361/sb75/squint-reproducibility/"
                        "analysis/benchmarking/geneformer/Geneformer-V2-104M")
    p.add_argument("--model-input-size", type=int, default=4096,
                   help="Context length. MUST match the checkpoint: "
                        "V2-95M-i4096 -> 4096 (default), V1 -> 2048.")
    p.add_argument("--ensembl-id-col", type=str, default=None,
                   help="adata.var column with Ensembl IDs already "
                        "populated. Pre-mapped is the recommended path.")
    p.add_argument("--auto-map-symbols", action="store_true",
                   help="Map gene symbols -> Ensembl IDs via mygene. "
                        "For mouse data this gives MOUSE Ensembl IDs "
                        "(ENSMUSG), which only work with a mouse-"
                        "specific Geneformer checkpoint. For stock V2 "
                        "(human-only) prefer --map-via-human-orthologs.")
    p.add_argument("--map-via-human-orthologs", action="store_true",
                   help="Map mouse genes to HUMAN Ensembl IDs (ENSG) "
                        "via the Nicheformer ortholog helper "
                        "(_nicheformer_embedding.add_human_ortholog_"
                        "ensembl_ids — mygene + Ensembl REST homology). "
                        "This is what you want for Geneformer V2 on "
                        "mouse data, since V2 (gc104M) is human-only. "
                        "Slow first call (~1-3 min for ~500 genes); "
                        "needs internet. Takes priority over "
                        "--auto-map-symbols.")
    p.add_argument("--species", type=str, default="mouse",
                   help="Used by --auto-map-symbols / "
                        "--map-via-human-orthologs to pick the species. "
                        "Default: mouse.")
    p.add_argument("--forward-batch-size", type=int, default=24,
                   help="Cells per forward pass during emb extraction.")
    p.add_argument("--emb-layer", type=int, default=-1,
                   help="Which transformer layer's representation to "
                        "extract. -1 = final layer (Geneformer default).")
    p.add_argument("--nproc", type=int, default=4,
                   help="Worker processes for tokenization.")
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

    # 1. Load + concat + Geneformer embedding extraction (one-time
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

    # 2. Resolve Ensembl IDs. For Geneformer V2 (gc104M) on mouse data,
    #    use --map-via-human-orthologs to map mouse symbols -> human
    #    ENSG via mygene + Ensembl REST homology, since V2 is human-only.
    print("\n=== Ensembl ID resolution ===")
    adata = _ensure_ensembl_ids(
        adata,
        ensembl_id_col=args.ensembl_id_col,
        auto_map_symbols=bool(args.auto_map_symbols),
        species=args.species,
        map_via_human_orthologs=bool(args.map_via_human_orthologs),
    )

    # 3. Embed ONCE with Geneformer (forward-pass only).
    print("\n=== Geneformer V2 zero-shot embedding ===")
    latent = _embed_with_geneformer(
        adata=adata,
        model_dir=args.model_dir,
        model_input_size=int(args.model_input_size),
        forward_batch_size=int(args.forward_batch_size),
        emb_layer=int(args.emb_layer),
        nproc=int(args.nproc),
    )

    # Restrict EVERYTHING downstream (Leiden / UMAP / NMI / ARI / iLISI
    # / MMD, plus per-seed h5ad/UMAP exports) to cells that actually got
    # a valid Geneformer embedding. The tokenizer drops cells with too
    # few expressed genes after gene-vocab filtering — those rows come
    # back as NaN from `_embed_with_geneformer`. We also reject any
    # non-finite (Inf) row defensively. Cleanest fix is to subset
    # `adata` itself so EVERY downstream call sees only valid cells —
    # avoids any chance of one consumer using the original 86k-row adata
    # while another uses the latent.
    valid_mask = np.isfinite(latent).all(axis=1)
    n_invalid = int((~valid_mask).sum())
    if n_invalid > 0:
        n_total = adata.n_obs
        frac = 100.0 * n_invalid / n_total
        print(f"  Excluding {n_invalid}/{n_total} ({frac:.1f}%) cells "
              "without a valid Geneformer embedding (NaN/Inf rows from "
              "Geneformer's tokenizer dropping low-gene-coverage "
              "cells). Downstream metrics, UMAPs, and per-seed h5ad "
              f"exports are restricted to the {n_total - n_invalid} "
              "cells with valid embeddings only.")
        if n_invalid == n_total:
            raise SystemExit(
                "ALL Geneformer embeddings are NaN/Inf — Geneformer's "
                "tokenizer rejected every cell. Check that "
                "adata.var['ensembl_id'] maps to Geneformer's vocab "
                "(>0 mapped genes per cell required), and that "
                "obs['n_counts'] is set."
            )
        adata = adata[valid_mask].copy()
        latent = latent[valid_mask]
    assert np.isfinite(latent).all(), (
        "Geneformer latent still contains non-finite values after "
        "valid-embedding subset — refusing to run downstream metrics "
        "on poisoned input."
    )

    adata.obsm[DEFAULT_LATENT_KEY] = latent
    shared_setup_seconds = time.time() - _shared_t0
    print(f"shared setup (load + Geneformer extraction): "
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
        # TIMED block: Leiden binary search on the shared Geneformer
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
            run_dir=args.out_dir, method="Geneformer",
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
        adata.uns["geneformer_n_clusters"]      = int(seed0_state["n_found"])
        adata.uns["geneformer_resolution"]      = float(seed0_state["resolution"])
        adata.uns["geneformer_seeds"]           = seeds
        adata.uns["geneformer_model_dir"]       = str(args.model_dir)
        adata.uns["geneformer_model_input_size"] = int(args.model_input_size)
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
                       "description": "Geneformer V2 zero-shot baseline (multi-seed downstream)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "Geneformer-V2-zero-shot"},
        "geneformer": {
            "model_dir":          str(args.model_dir),
            "model_input_size":   int(args.model_input_size),
            "ensembl_id_col":     args.ensembl_id_col,
            "auto_map_symbols":   bool(args.auto_map_symbols),
            "species":            args.species,
            "forward_batch_size": int(args.forward_batch_size),
            "emb_layer":          int(args.emb_layer),
            "nproc":              int(args.nproc),
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
