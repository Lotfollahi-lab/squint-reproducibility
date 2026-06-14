#!/usr/bin/env python3
"""
Fetch the CosMx Human Lymph Node dataset WITH MANUAL NICHE ANNOTATIONS from
the spatial-niche-benchmark study and store it as AnnData in the SQUINT
silver layout.

    "spatial-niche-benchmark" (WYXNICK), bioRxiv 2026
    https://www.biorxiv.org/content/10.64898/2026.02.27.708202v1
    https://github.com/WYXNICK/spatial-niche-benchmark

Why this dataset: it is HUMAN, SINGLE-CELL resolution (CosMx SMI), with BOTH
released per-cell CELL-TYPE labels (NanoString) AND released MANUAL
niche/region labels added by this paper (4 niches: B cell Zone, T cell Zone,
Medulla, Germinal Center). That's the single-cell-resolution human
dual-label combination missing from the broader survey.

The authors provide a PRE-ANNOTATED .h5ad on Google Drive
(`lymph_node_niche_annotated.h5ad`) that already bundles expression + cell
types + niches + coordinates, so this script just downloads it and
re-exports it into the silver layout with SQUINT-convention fields:

    obs["annotation"]      = cell type   (from --celltype-key,
                             default 'cell_type_annotation')
    obs["spatial_cluster"] = niche/region (from --niche-key,
                             default 'niche_annotation')
    obs["batch"]           = sample id (constant for a single section)
    obsm["spatial"]        = (x, y) coordinates
The original columns are preserved.

CAVEAT: this is a SINGLE tissue section (one CosMx lymph node) — great as a
human single-cell dual-label reference for niche/cell-type code QUALITY, but
it does NOT give cross-sample/cross-section retrieval (only one sample). The
manual niche annotation here is the rare, valuable part.

Requirements: gdown (pip install gdown), anndata, numpy, pandas, scipy.
Run on a node WITH internet (Google Drive download).

Usage:
    python fetch_cosmx_lymph_node.py \
        --out-dir /nfs/team361/sb75/DATASETS/silver/squint_hln
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Pre-annotated h5ad on Google Drive (from the repo's README).
_DRIVE_FILE_ID = "13z9_l3Z9dhFcJ6GMtwfYYBH_rOGQtd-H"
_DEFAULT_CELLTYPE_KEYS = ["cell_type_annotation", "cell_type", "cellType",
                          "celltype", "cell_types", "nb_clus", "annotation"]
_DEFAULT_NICHE_KEYS = ["niche_annotation", "niche", "manual_niche",
                       "spatial_niche", "region", "spatial_cluster"]
_SAMPLE_KEY_CANDIDATES = ["sample_id", "sample", "Run_Tissue_name", "tissue",
                          "slide", "section", "donor", "donor_id"]
_XY_PAIRS = [("x", "y"), ("x_global_px", "y_global_px"),
             ("CenterX_global_px", "CenterY_global_px"),
             ("x_centroid", "y_centroid"), ("spatial_x", "spatial_y")]


def _download_drive(file_id: str, out_path: str) -> None:
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        print(f"[fetch] reusing existing download: {out_path}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    try:
        import gdown  # noqa
    except ImportError:
        raise SystemExit(
            "gdown not installed. Install it (pip install gdown) and re-run, "
            "or download the file manually from "
            f"https://drive.google.com/file/d/{file_id}/view and pass it via "
            "--h5ad.")
    import gdown
    print(f"[fetch] downloading pre-annotated h5ad from Google Drive "
          f"(id={file_id}) -> {out_path}")
    got = gdown.download(id=file_id, output=out_path, quiet=False)
    if not got or not os.path.isfile(out_path):
        raise SystemExit(
            "Google Drive download failed. Try `pip install -U gdown`, or "
            "download manually from "
            f"https://drive.google.com/file/d/{file_id}/view and pass --h5ad.")


def _pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    return None


def _ensure_spatial(adata):
    import numpy as np
    if "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2:
        return
    for cx, cy in _XY_PAIRS:
        if cx in adata.obs.columns and cy in adata.obs.columns:
            adata.obsm["spatial"] = np.c_[
                adata.obs[cx].to_numpy(dtype=float),
                adata.obs[cy].to_numpy(dtype=float)]
            print(f"[fetch] built obsm['spatial'] from obs[{cx!r},{cy!r}]")
            return
    print("[fetch] WARNING: no spatial coordinates found — obsm['spatial'] "
          "missing (SQUINT needs it).", file=sys.stderr)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "sample"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir",
                   default="/nfs/team361/sb75/DATASETS/silver/squint_hln",
                   help="Silver output dir for the per-sample .h5ad(s).")
    p.add_argument("--h5ad", default=None,
                   help="Use a local pre-annotated h5ad instead of "
                        "downloading (skips Google Drive).")
    p.add_argument("--cache", default=None,
                   help="Where to cache the downloaded h5ad "
                        "(default: <out-dir>/_lymph_node_niche_annotated.h5ad).")
    p.add_argument("--celltype-key", default=None,
                   help="obs column with the cell-type label "
                        "(default: auto-detect, prefers cell_type_annotation).")
    p.add_argument("--niche-key", default=None,
                   help="obs column with the manual niche label "
                        "(default: auto-detect, prefers niche_annotation).")
    p.add_argument("--sample-key", default=None,
                   help="obs column identifying sample/section "
                        "(default: auto-detect; if none, treated as a single "
                        "section named --sample-name).")
    p.add_argument("--sample-name", default="cosmx_human_lymph_node",
                   help="Name for the single section when there is no "
                        "sample column.")
    p.add_argument("--keep-cache", action="store_true")
    args = p.parse_args()

    import anndata as ad

    src = args.h5ad
    cache = args.cache or os.path.join(
        args.out_dir, "_lymph_node_niche_annotated.h5ad")
    if src is None:
        _download_drive(_DRIVE_FILE_ID, cache)
        src = cache

    print(f"[fetch] reading {src}")
    adata = ad.read_h5ad(src)
    print(f"[fetch] loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"[fetch] obs columns: {list(adata.obs.columns)}")
    print(f"[fetch] obsm keys  : {list(adata.obsm.keys())}")

    ct_key = args.celltype_key or _pick(adata.obs.columns,
                                        _DEFAULT_CELLTYPE_KEYS, "cell-type")
    ni_key = args.niche_key or _pick(adata.obs.columns,
                                     _DEFAULT_NICHE_KEYS, "niche")
    if ni_key is None:
        raise SystemExit(
            "Could not find a niche column. Pass --niche-key. "
            f"obs columns: {list(adata.obs.columns)}")
    if ct_key is None:
        print("[fetch] WARNING: no cell-type column auto-detected; pass "
              "--celltype-key if needed.", file=sys.stderr)
    print(f"[fetch] cell-type key: {ct_key!r} | niche key: {ni_key!r}")
    print(f"[fetch] niches ({adata.obs[ni_key].nunique()}): "
          f"{sorted(map(str, adata.obs[ni_key].unique()))}")
    if ct_key:
        print(f"[fetch] cell types: {adata.obs[ct_key].nunique()}")

    s_key = args.sample_key or _pick(adata.obs.columns,
                                     _SAMPLE_KEY_CANDIDATES, "sample")
    if s_key is not None and adata.obs[s_key].nunique() > 1:
        samples = [(str(s), adata[adata.obs[s_key].astype(str) == str(s)])
                   for s in adata.obs[s_key].astype(str).unique()]
        print(f"[fetch] sample key {s_key!r}: {len(samples)} samples")
    else:
        samples = [(args.sample_name, adata)]
        print(f"[fetch] single section -> {args.sample_name!r} "
              f"(no usable sample column; this dataset is one CosMx section)")

    os.makedirs(args.out_dir, exist_ok=True)
    for bidx, (sname, sub) in enumerate(samples):
        sub = sub.copy()
        _ensure_spatial(sub)
        # Keep the label columns under their ORIGINAL names — they're
        # registered in the SQUINT blob via label_names (cell_types=<col> /
        # niche_types=<col>). Only add the infra fields the blob builder
        # requires: obs['cell_id'], obs['batch'] (graph batch_key), and the
        # canonical per-section id uns['batch'] (int).
        sub.obs["batch"] = str(bidx)
        if "cell_id" not in sub.obs.columns:
            sub.obs["cell_id"] = sub.obs_names.astype(str)
        sub.uns["batch"] = int(bidx)
        sub.uns["squint_source"] = "spatial-niche-benchmark_CosMx_lymph_node"
        sub.uns["sample_id"] = str(sname)
        fp = os.path.join(args.out_dir, f"{_sanitize(sname)}.h5ad")
        sub.write_h5ad(fp)
        print(f"[fetch]   wrote {fp}  ({sub.n_obs} cells; "
              f"cell-type col={ct_key!r} "
              f"({sub.obs[ct_key].nunique() if ct_key else 'n/a'}); "
              f"niche col={ni_key!r} ({sub.obs[ni_key].nunique()}))")

    if not args.keep_cache and args.h5ad is None and os.path.isfile(cache):
        os.unlink(cache)
        print(f"[fetch] removed cache {cache} (use --keep-cache to retain)")
    print(f"[fetch] DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
