#!/usr/bin/env python
"""
summarize_label_conditioned_metrics.py — collapse the per-method csvs into one table.
=============================================================================
compute_label_conditioned_metrics.py writes one csv per method into
artifacts/label_conditioned_metrics/. This concatenates them, averages over
seeds, and writes a single long csv plus a wide comparison csv.

WHAT THE COLUMNS MEAN, and which of them answers R2-W1b
  kbet   batch mixing WITHIN each cell type, averaged over cell types. Higher is
         better. THIS is the label-conditioned integration metric R2 asked for:
         unlike iLISI it cannot be satisfied by mixing cells of different types.
  clisi  cell-type LISI. Bio-conservation: higher means cell types stay separated.
  casw   cell-type silhouette, a second bio-conservation view.
  ilisi  plain batch mixing, the metric already in Table 1. Reported here as a
         consistency check on the same graph, NOT as a new result.

A NaN kbet is not a bad score, it means the metric was skipped. scib-metrics
drops any cell type confined to a single batch and then averages with
np.nanmean, so a dataset whose annotations are section-specific yields NaN for
every method. On the mouse brain that is all 49 labels. Treat NaN as "not
measurable on this dataset" and say so, rather than reporting it as zero.

ONE ROW PER (method, latent_key). SQUINT contributes two rows per codebook run,
cell_* and neighborhood_*, because the two codebooks are separate
representations; baselines contribute one, on the emb_key their published iLISI
was computed on.

USAGE
  python summarize_label_conditioned_metrics.py
  python summarize_label_conditioned_metrics.py --dataset xhs1000-3b_1p
  python summarize_label_conditioned_metrics.py --out-prefix /tmp/lcm --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["kbet", "clisi", "casw", "ilisi"]
DEFAULT_DIR = (Path("/nfs/team361/sb75/squint-reproducibility/artifacts")
               / "label_conditioned_metrics")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                   help="Folder holding the per-method lcm_*.csv files.")
    p.add_argument("--dataset", default=None,
                   help="Restrict to one dataset tag, e.g. xhs1000-3b_1p. "
                        "Default: every dataset found, with a `dataset` column.")
    p.add_argument("--out-prefix", type=Path, default=None,
                   help="Default: <dir>/summary_label_conditioned_metrics")
    p.add_argument("--force", action="store_true",
                   help="Overwrite the two summary csvs. The per-method inputs "
                        "are only ever read.")
    p.add_argument("--sort-by", default="kbet", choices=METRICS)
    a = p.parse_args(argv)

    files = sorted(f for f in a.dir.glob("lcm_*.csv"))
    if not files:
        raise SystemExit(f"no lcm_*.csv in {a.dir}. Run "
                         f"submit_label_conditioned_metrics.sh first.")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        # filename layout: lcm_<dataset>_<method>.csv
        stem = f.stem[len("lcm_"):]
        df["source_csv"] = f.name
        if "dataset" not in df.columns:
            df["dataset"] = stem.rsplit("_", 1)[0] if "_" in stem else stem
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    if a.dataset:
        raw = raw[raw["dataset"] == a.dataset]
        if raw.empty:
            raise SystemExit(f"no rows for dataset {a.dataset!r}. Found: "
                             f"{sorted(pd.concat(frames)['dataset'].unique())}")

    for m in METRICS:
        if m not in raw.columns:
            raw[m] = np.nan

    errs = raw[raw.get("error", "").astype(str).str.len() > 0]
    if len(errs):
        print(f"!! {len(errs)} of {len(raw)} rows recorded an error; they are kept "
              f"in the long csv and excluded from the means:")
        for _, r in errs.head(12).iterrows():
            print(f"   {r['method']:24s} {r['latent_key']:26s} "
                  f"seed {r.get('seed','?')}  {str(r['error'])[:110]}")
        raw = raw[raw.get("error", "").astype(str).str.len() == 0]

    keys = ["dataset", "method", "latent_key"]
    g = raw.groupby(keys, dropna=False)
    wide = g.agg(
        n_seeds=("seed", "nunique"),
        n_cells=("n_cells", "first"),
        cell_type_key=("cell_type_key", "first"),
        **{f"{m}_{s}": (m, s) for m in METRICS for s in ("mean", "std")},
    ).reset_index()
    # np.nanmean over an all-NaN group is NaN, which is what we want, but pandas
    # `count` is the honest record of how many seeds actually produced a number.
    for m in METRICS:
        wide[f"{m}_n"] = g[m].count().to_numpy()

    prefix = a.out_prefix or (a.dir / "summary_label_conditioned_metrics")
    long_csv = Path(f"{prefix}_long.csv")
    wide_csv = Path(f"{prefix}_wide.csv")
    for out in (long_csv, wide_csv):
        if out.exists() and not a.force:
            raise SystemExit(f"{out} exists. Pass --force to replace it.")

    order = wide.sort_values([f"{a.sort_by}_mean"], ascending=False,
                             na_position="last")
    for ds, sub in order.groupby("dataset", sort=False):
        print("\n" + "=" * 100)
        print(f"{ds}   label = {sub['cell_type_key'].iloc[0]!r}   "
              f"sorted by {a.sort_by}")
        print("=" * 100)
        print(f"  {'method':24s}{'representation':28s}{'seeds':>6s}"
              + "".join(f"{m:>18s}" for m in METRICS))
        for _, r in sub.iterrows():
            cells = ""
            for m in METRICS:
                mu, sd, n = r[f"{m}_mean"], r[f"{m}_std"], int(r[f"{m}_n"])
                cells += ("           n/a    " if not np.isfinite(mu) else
                          f"{mu:>11.4f}+/-{sd:<5.3f}" if n > 1 and np.isfinite(sd)
                          else f"{mu:>11.4f}        ")
            print(f"  {r['method']:24s}{r['latent_key']:28s}"
                  f"{int(r['n_seeds']):>6d}{cells}")
        if not np.isfinite(sub["kbet_mean"]).any():
            print("\n  kbet is n/a for EVERY method here. That is a property of the"
                  "\n  annotation, not of the methods: scib-metrics skips any cell"
                  "\n  type confined to one batch. Check with"
                  "\n  label_batch_feasibility.py before reporting anything.")

    raw.to_csv(long_csv, index=False)
    order.to_csv(wide_csv, index=False)
    print(f"\nwrote {long_csv}  ({len(raw)} per-seed rows)")
    print(f"wrote {wide_csv}  ({len(order)} method rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
