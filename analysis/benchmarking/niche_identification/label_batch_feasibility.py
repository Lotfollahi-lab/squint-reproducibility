#!/usr/bin/env python
"""
label_batch_feasibility.py — is a label column shared across batches, or split by them?
=============================================================================
Two questions, one crosstab.

1. HOW MANY CELL TYPES DOES EACH SECTION ACTUALLY HAVE? On the mouse brain the
   49-class `cell_type` column looks like a union of two per-section annotation
   vocabularies rather than one harmonised set: the skip log from
   `kbet_per_label` contained both "Astrocytes" and "astrocyte", both
   "Microglia" and "microglial cell". If that is right, no cell type spans both
   sections and any statement of the form "the dataset has 49 cell types" needs
   qualifying, because no single section does.

2. WHERE CAN kbet_per_label BE COMPUTED AT ALL? scib-metrics 0.5.6,
   `_kbet.py:151`, skips a label outright when

       n_obs < 10  or  len(np.unique(batches_sub)) == 1

   and then averages with `np.nanmean`. If every label sits in one batch, every
   label is skipped and the mean of an empty slice is NaN. That is a property of
   the annotation, not a bug, and it cannot be fixed by changing the code. So
   before running the metric anywhere, check which (dataset, label column) pairs
   have labels that genuinely straddle batches.

Reported per label column: classes per batch, how many classes are shared, and
the fraction of cells that fall in a label kbet would actually score. That last
number is the one that matters: a column can be nominally shared and still leave
kbet averaging over a handful of unrepresentative classes.

USAGE
  python label_batch_feasibility.py --adata .../predicted_adata.h5ad [--adata ...]
  python label_batch_feasibility.py --adata ... --label-keys cell_type,niche_type
  python label_batch_feasibility.py --adata ... --list-classes   # full vocabularies

Reads with backed="r"; obs and obsm keys only, so it is cheap and needs no GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# scib-metrics 0.5.6 _kbet.py:151. Kept as a named constant so the feasibility
# verdict below cannot drift away from the condition it is predicting.
KBET_MIN_CELLS_PER_LABEL = 10

BATCH_HINTS = ("batch", "section", "sample", "donor", "slide", "assay",
               "library", "fov_batch")
LABEL_HINTS = ("cell_type", "celltype", "cell_class", "subclass", "annotation",
               "niche", "region", "domain", "cluster", "leiden", "tissue",
               "molecular", "label")
MAX_CLASSES = 500          # above this it is an id column, not an annotation


def pick_columns(obs, label_keys):
    """(batch columns, label columns). Explicit --label-keys wins."""
    batch_cols = [c for c in obs.columns
                  if any(h in c.lower() for h in BATCH_HINTS)
                  and 2 <= obs[c].nunique(dropna=True) <= 50]
    if label_keys:
        return batch_cols, label_keys
    label_cols = []
    for c in obs.columns:
        if c in batch_cols:
            continue
        n = obs[c].nunique(dropna=True)
        if not (2 <= n <= MAX_CLASSES):
            continue
        if any(h in c.lower() for h in LABEL_HINTS):
            label_cols.append(c)
    return batch_cols, label_cols


def report(path: Path, label_keys, batch_key, list_classes: bool) -> None:
    import anndata as ad
    import pandas as pd

    a = ad.read_h5ad(path, backed="r")
    obs = a.obs
    print("\n" + "=" * 88)
    print(f"{path}")
    print(f"  n_obs={a.n_obs}  n_vars={a.n_vars}")
    print("=" * 88)

    batch_cols, label_cols = pick_columns(obs, label_keys)
    if batch_key:
        if batch_key not in obs.columns:
            print(f"  !! --batch-key {batch_key!r} not in obs; columns that look "
                  f"like batches: {batch_cols}")
            return
        batch_cols = [batch_key]
    if not batch_cols:
        print("  !! no batch-like column found")
        return
    bkey = batch_cols[0]
    if len(batch_cols) > 1:
        print(f"  batch-like columns: {batch_cols}  -> using {bkey!r}")

    b = obs[bkey].astype(str)
    vc = b.value_counts()
    print(f"\n  BATCH {bkey!r}: {vc.size} batches")
    for k, v in vc.items():
        print(f"    {k:<44s} {v:>8d} cells")

    if not label_cols:
        print("\n  !! no label-like column found; pass --label-keys explicitly")
        return

    print(f"\n  {'label column':<34s}{'classes':>8s}{'NaN':>8s}"
          f"{'per batch':>22s}{'shared':>8s}{'scorable cells':>16s}  verdict")
    for c in label_cols:
        if c not in obs.columns:
            print(f"  {c:<34s}  !! not in obs")
            continue
        col = obs[c]
        n_nan = int(col.isna().sum())
        lab = col.astype(str)
        ct = pd.crosstab(lab, b)
        if n_nan:                       # 'nan' becomes its own row via astype(str)
            ct = ct.drop(index="nan", errors="ignore")

        per_batch = (ct > 0).sum(axis=0)                 # classes present per batch
        n_batches_per_class = (ct > 0).sum(axis=1)
        size_per_class = ct.sum(axis=1)
        # exactly the condition scib-metrics applies before averaging
        scorable = (n_batches_per_class >= 2) & (size_per_class >=
                                                KBET_MIN_CELLS_PER_LABEL)
        n_scorable_cells = int(size_per_class[scorable].sum())
        frac = n_scorable_cells / max(int(size_per_class.sum()), 1)
        shared_all = int((n_batches_per_class == vc.size).sum())

        pb = "/".join(str(int(per_batch[k])) for k in vc.index)
        verdict = ("NOT FEASIBLE (every label single-batch)"
                   if int(scorable.sum()) == 0 else
                   f"OK ({int(scorable.sum())}/{ct.shape[0]} labels)"
                   if frac >= 0.75 else
                   f"PARTIAL ({int(scorable.sum())}/{ct.shape[0]} labels, "
                   f"{frac:.0%} of cells)")
        print(f"  {c:<34s}{ct.shape[0]:>8d}{n_nan:>8d}{pb:>22s}"
              f"{shared_all:>8d}{n_scorable_cells:>10d} {frac:>5.1%}  {verdict}")

        if list_classes:
            print(f"      per-batch vocabularies for {c!r}:")
            for k in vc.index:
                names = sorted(ct.index[ct[k] > 0].tolist())
                print(f"        [{k}] {len(names)} classes: "
                      + ", ".join(names[:60])
                      + (" ..." if len(names) > 60 else ""))
            both = sorted(ct.index[n_batches_per_class >= 2].tolist())
            print(f"        shared by >=2 batches: {len(both)}"
                  + (": " + ", ".join(both[:60]) if both else ""))

    print("\n  'per batch' counts classes PRESENT in each batch, in the batch order"
          f" listed above.\n  'shared' counts classes present in all {vc.size} "
          f"batches. 'scorable cells' applies\n  scib-metrics' own filter "
          f"(>=2 batches and >={KBET_MIN_CELLS_PER_LABEL} cells): a column whose"
          "\n  scorable fraction is 0% makes kbet_per_label return NaN, whatever "
          "the code does.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adata", type=Path, action="append", required=True)
    p.add_argument("--label-keys", default=None,
                   help="Comma list. Default: auto-detect annotation-like columns.")
    p.add_argument("--batch-key", default=None,
                   help="Default: first batch-like column found.")
    p.add_argument("--list-classes", action="store_true",
                   help="Print the vocabulary of each batch, to see whether two "
                        "annotation schemes were concatenated.")
    a = p.parse_args(argv)
    keys = ([s.strip() for s in a.label_keys.split(",") if s.strip()]
            if a.label_keys else None)
    for f in a.adata:
        report(f, keys, a.batch_key, a.list_classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
