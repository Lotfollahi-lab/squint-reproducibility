#!/usr/bin/env python
"""
Cache human Ensembl IDs into a silver dataset's var, so the foundation-model
benchmarks (Geneformer / Nicheformer) can read them directly instead of doing a
per-run, internet-dependent `mygene` lookup on the compute node.

Maps each h5ad's `var_names` (HGNC symbols) -> human ENSG via mygene and writes
`var['ensembl_id']` IN PLACE, then re-saves. Run this ONCE, on a node WITH
INTERNET (e.g. the login node), then point the benchmarks at the cached column
(`--ensembl-id-col ensembl_id` for Geneformer, `--gene-col ensembl_id` for
Nicheformer) -- exactly how chl59-8b_1p already works.

Mirrors run_geneformer.py's `_ensure_ensembl_ids` auto-map path (same scopes /
species / first-ENSG extraction), so the cached IDs match what the per-run path
would have produced.

Usage (login node, with internet):
    source /nfs/team361/sb75/.venvs/geneformer/bin/activate   # has mygene + anndata
    python analysis/data_preparation/add_ensembl_ids.py \
        --silver-dir /nfs/team361/sb75/DATASETS/silver/xhs1000-3b_1p
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _map_symbols_to_ensembl(symbols, species: str):
    """HGNC symbols -> Ensembl gene IDs via mygene. Unmapped -> kept as the
    input symbol (so the column is all-strings; Geneformer drops non-ENSG)."""
    import mygene

    mg = mygene.MyGeneInfo()
    sp = "mouse" if species.lower() in ("mouse", "mus musculus") else "human"
    df = mg.querymany(
        list(symbols), scopes=["symbol", "alias", "ensembl.gene"],
        species=sp, fields="ensembl.gene", returnall=False, as_dataframe=True,
    )

    def _first_ens(x):
        if isinstance(x, list) and x and isinstance(x[0], dict):
            return x[0].get("gene")
        if isinstance(x, dict):
            return x.get("gene")
        return x if isinstance(x, str) else None

    col = "ensembl.gene" if "ensembl.gene" in df.columns else "ensembl"
    sym_to_ens = (df[col].apply(_first_ens) if col in df.columns
                  else pd.Series(dtype=str))
    sym_to_ens = sym_to_ens[~sym_to_ens.index.duplicated(keep="first")]
    return np.array([str(sym_to_ens.get(s, s)) for s in symbols], dtype=object)


def _n_ensembl(ens) -> int:
    return int(sum(str(v).startswith(("ENSG", "ENSMUSG")) for v in ens))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--silver-dir", required=True,
                   help="Dir of silver h5ads to annotate in place.")
    p.add_argument("--species", default="human")
    p.add_argument("--gene-col-out", default="ensembl_id")
    p.add_argument("--glob", default="*.h5ad")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-map even if the column already has Ensembl IDs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Map + report, but do NOT write the h5ads back.")
    args = p.parse_args(argv)

    import anndata as ad

    silver = Path(args.silver_dir)
    files = sorted(silver.glob(args.glob))
    if not files:
        print(f"ERROR: no {args.glob} under {silver}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} h5ad(s) under {silver}")

    any_mapped = False
    for fp in files:
        print(f"\n=== {fp.name} ===")
        adata = ad.read_h5ad(fp)
        col = args.gene_col_out
        if (col in adata.var.columns and not args.overwrite
                and _n_ensembl(adata.var[col].astype(str)) > 0):
            n = _n_ensembl(adata.var[col].astype(str))
            print(f"  already has {col} ({n}/{adata.n_vars} Ensembl); skip "
                  f"(use --overwrite to remap).")
            any_mapped = True
            continue
        symbols = adata.var_names.astype(str).to_numpy()
        print(f"  mapping {len(symbols)} symbols -> {args.species} Ensembl via mygene...")
        ens = _map_symbols_to_ensembl(symbols, args.species)
        n_mapped = _n_ensembl(ens)
        print(f"  mapped {n_mapped}/{len(symbols)} to Ensembl IDs.")
        if n_mapped == 0:
            print("  ERROR: 0 mapped — is there internet on this node? "
                  "(mygene needs it). Not writing this file.", file=sys.stderr)
            continue
        any_mapped = True
        adata.var[col] = ens
        if args.dry_run:
            print("  [dry-run] not writing.")
            continue
        adata.write_h5ad(fp)
        print(f"  wrote {col} -> {fp}")

    if not any_mapped:
        print("\nNO files were mapped (0 Ensembl everywhere). Check internet / "
              "symbols.", file=sys.stderr)
        return 2
    print("\nDONE. Now run the benchmarks with the cached column:")
    print("  Geneformer:  --ensembl-id-col ensembl_id")
    print("  Nicheformer: --gene-col ensembl_id")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
