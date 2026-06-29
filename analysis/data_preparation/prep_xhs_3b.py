#!/usr/bin/env python3
"""
Prepare the 3-section Xenium-human-skin silver dir (xhs1000-3b_1p) for SQUINT.

The raw xhs1000-39b_1p section files lack the fields the SQUINT blob builder
requires and have non-unique cell names, so a plain symlink would (a) hard-error
in `_derive_adata_batch_id` (which REQUIRES `uns['batch']`) and (b) collide
obs_names across sections. This script reads the chosen batches, stamps the
infra fields, makes obs_names globally unique, and writes them into the
xhs1000-3b_1p silver dir:

  * uns['batch'] = 'batch<i>' and obs['batch'] = 'batch<i>'  (i = 0,1,2 in the
    order given) — distinct per section so the blob treats them as 3 sections;
  * obs_names prefixed with 'b<orig>_' (orig = the source batch number) so cell
    ids are unique across the 3 concatenated sections;
  * obs['cell_id'] kept; original section id kept in uns['sample_id'].

Labels (obs['new_annotation'] / obs['niche_type']) and X (RAW counts) are left
untouched. Verifies X looks like counts and the label columns exist.

Usage:
    python prep_xhs_3b.py
    python prep_xhs_3b.py --batches 11,19,32 \
        --src-dir /nfs/team361/sb75/DATASETS/silver/xhs1000-39b_1p \
        --out-dir /nfs/team361/sb75/DATASETS/silver/xhs1000-3b_1p
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

CELL_KEY = "new_annotation"
NICHE_KEY = "niche_type"


def _looks_like_counts(X) -> bool:
    sub = X[:50] if X.shape[0] > 50 else X
    arr = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
    fin = arr[np.isfinite(arr)]
    return bool(fin.size and fin.min() >= 0 and np.allclose(fin, np.round(fin)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batches", default="11,19,32",
                   help="Comma list of source batch numbers (order = section 0,1,2).")
    p.add_argument("--src-dir", default="/nfs/team361/sb75/DATASETS/silver/xhs1000-39b_1p")
    p.add_argument("--out-dir", default="/nfs/team361/sb75/DATASETS/silver/xhs1000-3b_1p")
    p.add_argument("--cell-key", default=CELL_KEY)
    p.add_argument("--niche-key", default=NICHE_KEY)
    args = p.parse_args(argv)

    import anndata as ad

    batches = [b.strip() for b in args.batches.split(",") if b.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[prep-xhs] {len(batches)} sections -> {args.out_dir}")

    for i, orig in enumerate(batches):
        src = os.path.join(args.src_dir, f"adata_batch{orig}.h5ad")
        if not os.path.isfile(src):
            raise SystemExit(f"missing source file: {src}")
        adata = ad.read_h5ad(src)

        # checks
        if not _looks_like_counts(adata.X):
            print(f"[prep-xhs] WARNING batch{orig}: X does not look like raw counts "
                  f"— SQUINT's NB decoder expects RAW counts.", file=sys.stderr)
        for k in (args.cell_key, args.niche_key):
            if k not in adata.obs:
                print(f"[prep-xhs] WARNING batch{orig}: obs['{k}'] missing "
                      f"(label will be unscored).", file=sys.stderr)

        # stamp section identity (dense 0..N-1; blob _derive parses 'batchN')
        adata.obs["batch"] = f"batch{i}"
        adata.uns["batch"] = f"batch{i}"
        adata.uns["sample_id"] = f"xhs_batch{orig}"
        if "cell_id" not in adata.obs.columns:
            adata.obs["cell_id"] = adata.obs_names.astype(str)
        # globally-unique obs_names across the 3 sections
        adata.obs_names = [f"b{orig}_{n}" for n in adata.obs_names.astype(str)]
        adata.obs_names_make_unique()

        out = os.path.join(args.out_dir, f"adata_batch{orig}.h5ad")
        adata.write_h5ad(out)
        nct = adata.obs[args.cell_key].nunique() if args.cell_key in adata.obs else "n/a"
        nni = adata.obs[args.niche_key].nunique() if args.niche_key in adata.obs else "n/a"
        print(f"[prep-xhs] section {i} (src batch{orig}): {adata.n_obs} cells x "
              f"{adata.n_vars} genes; uns['batch']='batch{i}'; "
              f"{args.cell_key}={nct} cats, {args.niche_key}={nni} cats -> {out}")

    print(f"[prep-xhs] DONE. Next: build the blob:\n"
          f"    python examples/run_squint.py --build-blob --build-blob-dataset xhs1000-3b_1p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
