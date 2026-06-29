#!/usr/bin/env python3
"""
Gene-filter the CosMx Human Lymph Node (squint_hln) silver AnnData into the
THREE feature-selection scenarios used by the niche-identification benchmark
(Wang et al., bioRxiv 2026, doi 10.64898/2026.02.27.708202), so SQUINT is
compared apples-to-apples with NicheCompass / CellCharter / BANKSY / ...:

    all   -> all detected genes        (filter_genes only)
    hvg   -> top-N highly variable genes (scanpy seurat_v3 on RAW counts)
    svg   -> top-N spatially variable genes (squidpy Moran's I)

Negative-control probes are removed UPSTREAM by fetch_cosmx_lymph_node.py
(--drop-controls), which also restores raw counts into X. This script therefore
does GENE FILTERING ONLY (filter_genes + HVG/SVG/all selection); it warns if any
control probes are still present (i.e. the silver wasn't regenerated).

The benchmark ran every method across {all genes, top-2000 HVG, top-2000 SVG,
curated-2000 core-lineage DE} on this exact dataset. NicheCompass's own recipe
(train_nichecompass_reference_model.py) is: filter_genes(min_cells=0.0005*N) ->
seurat_v3 HVG -> squidpy Moran's I SVG (top 3000) -> union with gene-program
genes. SQUINT has no gene programs, so we expose HVG and SVG directly (N=2000 to
match the benchmark; override with --n-top).

Each scenario is written into its OWN silver dir so the SQUINT blob builder can
ingest it unchanged:

    <silver_root>/squint_hln_allgenes/cosmx_human_lymph_node.h5ad
    <silver_root>/squint_hln_hvg2k/cosmx_human_lymph_node.h5ad
    <silver_root>/squint_hln_svg2k/cosmx_human_lymph_node.h5ad

CRITICAL: X is kept as RAW COUNTS (SQUINT's NB decoder consumes raw counts;
HVG/SVG are computed on temporary copies). All obs / obsm / uns (labels,
spatial, batch) are inherited from the input by subsetting `var` only — so the
output matches what `make_dataset_blob_config_squint_hln` expects.

Then build the blobs and run the reference variant on each:
    python ../../squint/examples/run_squint.py --build-blob \
        --build-blob-dataset squint_hln_svg2k
    bash ../../squint/examples/submit_multi_seed.sh \
        "dualvq+...+filmscale+squint_hln_svg2k" 0,1,2,3,4
(see submit_squint_hln_genesets.sh for the full sweep.)

Requires: anndata, numpy, scipy, scanpy (HVG), squidpy (SVG).
Run on the farm. `--self-test` runs an scanpy/squidpy-free sanity check.

Usage:
    python preprocess_squint_hln.py \
        --input /nfs/team361/sb75/DATASETS/silver/squint_hln/cosmx_human_lymph_node.h5ad \
        --silver-root /nfs/team361/sb75/DATASETS/silver \
        --gene-sets all,hvg,svg --n-top 2000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

# Negative-control / control-probe name prefixes for imaging panels
# (CosMx uses Negative* and SystemControl*; others kept for safety).
DEFAULT_CONTROL_PREFIXES = (
    "Negative", "SystemControl", "NegPrb", "NegPrb_", "FalseCode",
    "Blank", "BLANK", "control_probe", "antisense",
)

# tag -> (silver dataset name, selection method)
GENE_SET_SPEC = {
    "all": ("squint_hln_allgenes", "all"),
    "hvg": ("squint_hln_hvg2k", "hvg"),
    "svg": ("squint_hln_svg2k", "svg"),
}

OUT_FILENAME = "cosmx_human_lymph_node.h5ad"


# ---------------------------------------------------------------------------
# control-probe detection (pure; anndata/numpy only -> locally testable)
# ---------------------------------------------------------------------------
def detect_control_genes(var_names, prefixes=DEFAULT_CONTROL_PREFIXES, var=None):
    """Boolean mask (len = n_vars), True == negative control / control probe.

    Matches by case-insensitive name prefix; additionally honours a boolean /
    categorical control flag in `var` if one of the usual columns is present.
    """
    names = np.asarray([str(n) for n in var_names])
    pat = re.compile(r"^(" + "|".join(re.escape(p) for p in prefixes) + r")",
                     re.IGNORECASE)
    mask = np.array([bool(pat.match(n)) for n in names])

    if var is not None:
        for col in ("feature_types", "control", "is_control", "probe_type"):
            if col in getattr(var, "columns", []):
                vals = var[col].astype(str).str.lower()
                mask = mask | vals.str.contains(
                    "control|negative|blank|neg_prb|falsecode", regex=True
                ).to_numpy()
    return mask


# ---------------------------------------------------------------------------
# feature selection (scanpy / squidpy; farm only)
# ---------------------------------------------------------------------------
def select_hvg(adata, n_top):
    """Top-N highly variable genes via scanpy seurat_v3 on RAW counts."""
    import scanpy as sc

    n_top = int(min(n_top, adata.n_vars))
    a = adata.copy()                       # X = raw counts (seurat_v3 expects counts)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_top)
    genes = a.var_names[a.var["highly_variable"].to_numpy()].tolist()
    return genes


def select_svg(adata, n_top, n_neighs, spatial_key="spatial"):
    """Top-N spatially variable genes via squidpy Moran's I (on lognorm)."""
    import scanpy as sc
    import squidpy as sq

    n_top = int(min(n_top, adata.n_vars))
    a = adata.copy()
    sc.pp.normalize_total(a)               # median library size (target_sum=None)
    sc.pp.log1p(a)
    sq.gr.spatial_neighbors(a, coord_type="generic", spatial_key=spatial_key,
                            n_neighs=n_neighs)
    sq.gr.spatial_autocorr(a, mode="moran", genes=a.var_names.tolist(), n_jobs=1)
    moran = a.uns["moranI"]
    moran = moran.sort_values("I", ascending=False)
    return moran.index[:n_top].tolist()


# ---------------------------------------------------------------------------
# writing one gene set
# ---------------------------------------------------------------------------
def write_geneset(adata, genes, out_dir, fname=OUT_FILENAME, manifest_extra=None):
    """Subset `adata` to `genes` (var only; X = raw counts preserved) and write.

    Returns the output path. obs / obsm / uns are inherited untouched.
    """
    os.makedirs(out_dir, exist_ok=True)
    genes = [g for g in genes if g in set(adata.var_names)]
    sub = adata[:, genes].copy()
    out_path = os.path.join(out_dir, fname)
    sub.write_h5ad(out_path)
    # gene manifest (for the paper / reproducibility)
    with open(os.path.join(out_dir, "selected_genes.txt"), "w") as f:
        f.write("\n".join(genes) + "\n")
    manifest = {"n_genes": int(sub.n_vars), "n_cells": int(sub.n_obs),
                "out_path": out_path}
    if manifest_extra:
        manifest.update(manifest_extra)
    with open(os.path.join(out_dir, "preprocess_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _looks_like_counts(X) -> bool:
    sample = X[:50] if X.shape[0] > 50 else X
    arr = sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)
    if arr.size == 0:
        return True
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return bool(np.all(finite >= 0) and np.allclose(finite, np.round(finite)))


def restore_raw_if_needed(adata, norm_layer="norm"):
    """If X is NOT raw counts but adata.raw holds them, move counts into X.

    The released CosMx file stores normalised values in X and raw counts in
    .raw; SQUINT needs RAW counts. This mirrors the fix in
    fetch_cosmx_lymph_node.py so preprocessing is correct even when run directly
    on a silver file that hasn't been regenerated. Reindexes raw to adata.var so
    X / layers / var stay aligned. Returns (adata, did_restore)."""
    if _looks_like_counts(adata.X):
        return adata, False
    if adata.raw is None:
        print("[prep] WARNING: X is not raw counts and adata.raw is None — "
              "cannot recover counts. SQUINT's NB decoder expects RAW counts.",
              file=sys.stderr)
        return adata, False
    raw_ad = adata.raw.to_adata()
    raw_set = set(map(str, raw_ad.var_names))
    present = [g for g in adata.var_names if str(g) in raw_set]
    if len(present) < adata.n_vars:
        print(f"[prep] WARNING: {adata.n_vars - len(present)} genes absent from "
              f"adata.raw — dropping them during raw restore.", file=sys.stderr)
    adata = adata[:, present].copy()
    adata.layers[norm_layer] = adata.X.copy()
    adata.X = raw_ad[:, present].X.copy()
    adata.raw = None
    print(f"[prep] X was normalised; restored RAW counts from adata.raw for "
          f"{len(present)} genes (normalised kept in layers['{norm_layer}']).")
    return adata, True


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",
                   default="/nfs/team361/sb75/DATASETS/silver/squint_hln/"
                           "cosmx_human_lymph_node.h5ad",
                   help="Raw squint_hln silver AnnData (X = raw counts).")
    p.add_argument("--silver-root",
                   default="/nfs/team361/sb75/DATASETS/silver",
                   help="Root silver dir; outputs go to <root>/<dataset_name>/.")
    p.add_argument("--gene-sets", default="all,hvg,svg",
                   help="Comma list of {all,hvg,svg}.")
    p.add_argument("--n-top", type=int, default=2000,
                   help="N for HVG/SVG (benchmark used 2000).")
    p.add_argument("--min-cell-gene-ratio", type=float, default=0.0005,
                   help="filter_genes min_cells = ceil(ratio * n_cells) "
                        "(NicheCompass default 0.0005).")
    p.add_argument("--svg-n-neighs", type=int, default=6,
                   help="kNN for the Moran's I spatial graph (squidpy default 6).")
    p.add_argument("--out-filename", default=OUT_FILENAME)
    p.add_argument("--self-test", action="store_true",
                   help="Run a scanpy/squidpy-free sanity check and exit.")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    import anndata as ad

    requested = [g.strip() for g in args.gene_sets.split(",") if g.strip()]
    for g in requested:
        if g not in GENE_SET_SPEC:
            raise SystemExit(f"unknown gene set {g!r}; choices: {list(GENE_SET_SPEC)}")

    print(f"[prep] reading {args.input}")
    adata = ad.read_h5ad(args.input)
    n0_cells, n0_genes = adata.n_obs, adata.n_vars
    print(f"[prep] loaded {n0_cells} cells x {n0_genes} genes")
    # ensure X is RAW counts (no-op if the silver already has counts; restores
    # from adata.raw if a not-yet-regenerated silver still has normalised X).
    adata, _ = restore_raw_if_needed(adata)

    # Negative-control probes are removed UPSTREAM by fetch_cosmx_lymph_node.py
    # (--drop-controls). We do NOT remove them here — only gene filtering. But
    # warn loudly if any slipped through, since that means the silver wasn't
    # regenerated and the gene sets (esp. "all") would be contaminated.
    ctrl = detect_control_genes(adata.var_names, DEFAULT_CONTROL_PREFIXES, var=adata.var)
    n_ctrl = int(ctrl.sum())
    if n_ctrl:
        print(f"[prep] WARNING: {n_ctrl} control probes still present "
              f"(e.g. {list(adata.var_names[ctrl][:5])}). These should have been "
              f"removed by fetch_cosmx_lymph_node.py (--drop-controls) — re-run it "
              f"and rebuild. Proceeding WITHOUT removing them.", file=sys.stderr)

    # filter low-coverage genes (NicheCompass: min_cells = 0.0005 * n_cells)
    import scanpy as sc
    min_cells = int(np.ceil(args.min_cell_gene_ratio * adata.n_obs))
    before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=max(1, min_cells))
    print(f"[prep] filter_genes(min_cells={max(1, min_cells)}): "
          f"{before} -> {adata.n_vars} genes")

    detected = adata.var_names.tolist()
    common = {"n_input_genes": int(n0_genes), "n_controls_present": n_ctrl,
              "min_cells": int(max(1, min_cells)),
              "n_detected_genes": int(len(detected)), "n_top": int(args.n_top),
              "input": os.path.abspath(args.input)}

    # 3) per-scenario selection + write
    for tag in requested:
        dataset_name, method = GENE_SET_SPEC[tag]
        out_dir = os.path.join(args.silver_root, dataset_name)
        if method == "all":
            genes = detected
        elif method == "hvg":
            print(f"[prep] selecting top-{args.n_top} HVGs (seurat_v3)...")
            genes = select_hvg(adata, args.n_top)
        elif method == "svg":
            print(f"[prep] selecting top-{args.n_top} SVGs (Moran's I, "
                  f"n_neighs={args.svg_n_neighs})...")
            genes = select_svg(adata, args.n_top, args.svg_n_neighs)
        else:
            raise AssertionError(method)
        out_path = write_geneset(
            adata, genes, out_dir, fname=args.out_filename,
            manifest_extra={**common, "gene_set": tag, "method": method,
                            "dataset_name": dataset_name})
        print(f"[prep] {tag:>3} -> {out_path}  ({len(genes)} genes)")

    print("[prep] DONE. Next: build the blobs + run the reference variant, e.g.\n"
          "       bash analysis/data_preparation/submit_squint_hln_genesets.sh")
    return 0


# ---------------------------------------------------------------------------
# self-test (no scanpy / squidpy)
# ---------------------------------------------------------------------------
def _self_test():
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(0)
    n, g = 200, 30
    counts = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    var_names = [f"Gene{i}" for i in range(g - 5)] + \
                ["Negative1", "Negative2", "SystemControl1", "Blank3", "NegPrb7"]
    a = ad.AnnData(X=counts)
    a.var_names = var_names
    a.obs["cell_type_annotation"] = pd.Categorical(rng.integers(0, 4, n).astype(str))
    a.obs["niche_annotation"] = pd.Categorical(rng.integers(0, 4, n).astype(str))
    a.obs["batch"] = "batch0"
    a.uns["batch"] = "batch0"
    a.obsm["spatial"] = rng.normal(size=(n, 2))

    # control detection
    ctrl = detect_control_genes(a.var_names, var=a.var)
    assert ctrl.sum() == 5, ctrl.sum()
    assert set(np.asarray(a.var_names)[ctrl]) == {
        "Negative1", "Negative2", "SystemControl1", "Blank3", "NegPrb7"}
    a2 = a[:, ~ctrl].copy()
    assert a2.n_vars == 25

    # write a gene subset to a temp dir; check X is raw counts + metadata kept
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        genes = a2.var_names[:10].tolist()
        out = write_geneset(a2, genes, d)
        back = ad.read_h5ad(out)
        assert back.n_vars == 10
        assert back.n_obs == n
        Xb = back.X.toarray() if hasattr(back.X, "toarray") else np.asarray(back.X)
        assert np.allclose(Xb, np.round(Xb)) and Xb.min() >= 0, "X not raw counts"
        assert "cell_type_annotation" in back.obs and "niche_annotation" in back.obs
        assert "spatial" in back.obsm and back.uns["batch"] == "batch0"
        assert _looks_like_counts(back.X)
        manifest = json.load(open(os.path.join(d, "preprocess_manifest.json")))
        assert manifest["n_genes"] == 10
    print("ok  self-test: control detection (5), subset (25->10), raw-count + "
          "metadata preservation, manifest")

    # raw-restore path: normalized X + raw counts in .raw -> counts moved to X
    raw = ad.AnnData(X=counts.copy())
    raw.var_names = var_names
    norm = counts / counts.sum(1, keepdims=True)
    a2 = ad.AnnData(X=norm.astype(np.float32))
    a2.var_names = var_names
    a2.obsm["spatial"] = a.obsm["spatial"]
    a2.raw = raw
    assert not _looks_like_counts(a2.X)
    restored, did = restore_raw_if_needed(a2.copy())
    assert did and _looks_like_counts(restored.X) and "norm" in restored.layers
    Xr = restored.X.toarray() if hasattr(restored.X, "toarray") else np.asarray(restored.X)
    assert np.allclose(Xr, counts), "raw restore did not recover counts"
    assert restored.raw is None
    print("ok  self-test: restore_raw_if_needed (normalized X + .raw -> raw counts in X)")
    print("SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
