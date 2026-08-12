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

AN EXACT 0.0000 kbet IS ALSO NOT A SCORE. Measured on eczema seed 0:

  representation        distinct rows      tied nbr distances   kbet   labels scored
  cell_code_indices     633 / 53,655             94.12%        0.046      1 of 21
  cell_latent        53,655 / 53,655              0.00%        0.457     20 of 21

kbet is diffusion-based, and obsm['cell_code_indices'] is TWO INTEGER COLUMNS, so
85 cells share each distinct coordinate. The neighbour graph collapses, the
per-label subgraph breaks into many components below the 3*k0 size floor, and
_kbet.py assigns `score = 0  # i.e. 100% rejection` without measuring anything.
That is a property of scoring a discrete index with a diffusion metric, NOT a
statement about batch mixing.

So report kbet (and casw, which the same ties distort) on the CONTINUOUS latents.
iLISI is unaffected and stays on the discrete codes, where it is the metric Table 1
reports: it is purely kNN-based, so a tied neighbourhood simply means "the cells
sharing this code", and asking how batches mix within a code is exactly the
intended question. This script warns whenever a kbet mean is exactly zero.

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

METRICS = ["basw", "kbet", "graph_conn", "ilisi", "clisi", "casw"]
# Printed as two narrower tables rather than one 150-column one, and the split is
# the conceptual one: the first group is batch mixing computed WITHIN cell types
# (what R2-W1b asked for), the second is unconditioned batch mixing plus
# bio-conservation.
GROUPS = [("label-conditioned integration", ["basw", "kbet", "graph_conn"]),
          ("batch mixing (unconditioned) and bio conservation",
           ["ilisi", "clisi", "casw"])]
DEFAULT_DIR = (Path("/nfs/team361/sb75/squint-reproducibility/artifacts")
               / "label_conditioned_metrics")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                   help="Folder holding the per-method csv files.")
    p.add_argument("--pattern", default="lcm_*.csv",
                   help="Which csvs to combine. 'lcm_*.csv' is the full metric set; "
                        "'basw_*.csv' is the batch-ASW-only set from compute_basw.py. "
                        "Keep them separate: they cover different method sets, so "
                        "mixing them yields rows whose metrics are not comparable.")
    p.add_argument("--dataset", default=None,
                   help="Restrict to one dataset tag, e.g. xhs1000-3b_1p. "
                        "Default: every dataset found, with a `dataset` column.")
    p.add_argument("--out-prefix", type=Path, default=None,
                   help="Default: <dir>/summary_label_conditioned_metrics")
    p.add_argument("--force", action="store_true",
                   help="Overwrite the two summary csvs. The per-method inputs "
                        "are only ever read.")
    p.add_argument("--sort-by", default="basw", choices=METRICS,
                   help="Default basw: the label-conditioned integration metric "
                        "that is defined on the discrete codes as well as the "
                        "continuous latents, so every row is comparable.")
    a = p.parse_args(argv)

    files = sorted(f for f in a.dir.glob(a.pattern)
                   if not f.name.startswith("summary_"))
    if not files:
        raise SystemExit(f"no {a.pattern} in {a.dir}. Run "
                         f"submit_label_conditioned_metrics.sh (lcm_*) or "
                         f"submit_basw.sh (basw_*) first.")
    prefix_len = len(a.pattern.split("*")[0])

    def dataset_of(path, fallback):
        """
        The dataset tag is taken from artifacts/<tag>/... in the recorded path, not
        from the filename. Filenames are not parseable: lcm_<dataset>_<method>.csv
        splits wrongly whenever a method name itself contains an underscore
        (lcm_xhs1000-3b_1p_SQUINT_codes.csv invented a dataset called
        "xhs1000-3b_1p_SQUINT"). The path is authoritative.
        """
        parts = Path(str(path)).parts
        if "artifacts" in parts:
            i = parts.index("artifacts")
            if i + 1 < len(parts):
                return parts[i + 1]
        return fallback

    frames = []
    for f in files:
        df = pd.read_csv(f)
        stem = f.stem[prefix_len:]
        df["source_csv"] = f.name
        fallback = stem.rsplit("_", 1)[0] if "_" in stem else stem
        df["dataset"] = [dataset_of(p, fallback) for p in df.get("path", [])] \
            if "path" in df.columns else fallback
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    # The same (dataset, method, representation, seed, file) can arrive twice when an
    # earlier ad-hoc run wrote its own csv alongside the launcher's. Identical rows,
    # so collapsing them is lossless and stops the mean being weighted twice.
    dup_on = [c for c in ("dataset", "method", "latent_key", "seed", "path")
              if c in raw.columns]
    n_before = len(raw)
    raw = raw.drop_duplicates(subset=dup_on, keep="first")
    if len(raw) < n_before:
        print(f"note: dropped {n_before - len(raw)} duplicate rows (the same file "
              f"scored by two csvs); {len(raw)} remain")
    if a.dataset:
        raw = raw[raw["dataset"] == a.dataset]
        if raw.empty:
            raise SystemExit(f"no rows for dataset {a.dataset!r}. Found: "
                             f"{sorted(pd.concat(frames)['dataset'].unique())}")

    # Which metrics were actually ATTEMPTED, i.e. present as a column in at least one
    # input csv. Needed to keep the diagnostics below honest: the basw_*.csv files
    # carry only basw, so kbet is absent there, and reporting "kbet is n/a for every
    # method, therefore the annotation is batch-confounded" would be flatly wrong
    # when the truth is that kbet was never computed.
    attempted = {m for m in METRICS if m in raw.columns}
    for m in METRICS:
        if m not in raw.columns:
            raw[m] = np.nan

    # An empty `error` cell round-trips through csv as NaN, and NaN.astype(str) is
    # the 3-character string "nan", so a naive length test marks EVERY row as
    # failed. Normalise first.
    if "error" in raw.columns:
        err = (raw["error"].fillna("").astype(str).str.strip()
               .replace({"nan": "", "NaN": "", "None": ""}))
    else:
        err = pd.Series("", index=raw.index)
    errs = raw[err != ""]
    if len(errs):
        print(f"!! {len(errs)} of {len(raw)} rows recorded an error; they are kept "
              f"in the long csv and excluded from the means:")
        for _, r in errs.head(12).iterrows():
            print(f"   {r['method']:24s} {r['latent_key']:26s} "
                  f"seed {r.get('seed','?')}  {str(r['error'])[:110]}")
        raw = raw[err == ""]

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
        for gname, gmetrics in GROUPS:
            gmetrics = [m for m in gmetrics if m in attempted]
            if not gmetrics:
                continue
            print(f"\n  -- {gname} --")
            print(f"  {'method':24s}{'representation':28s}{'seeds':>6s}"
                  + "".join(f"{m:>18s}" for m in gmetrics))
            for _, r in sub.iterrows():
                cells = ""
                for m in gmetrics:
                    mu, sd, n = r[f"{m}_mean"], r[f"{m}_std"], int(r[f"{m}_n"])
                    cells += ("           n/a    " if not np.isfinite(mu) else
                              f"{mu:>11.4f}+/-{sd:<5.3f}"
                              if n > 1 and np.isfinite(sd)
                              else f"{mu:>11.4f}        ")
                print(f"  {r['method']:24s}{r['latent_key']:28s}"
                      f"{int(r['n_seeds']):>6d}{cells}")
        zeros = (sub[np.isclose(sub["kbet_mean"].to_numpy(dtype=float), 0.0)]
                 if "kbet" in attempted else sub.iloc[:0])
        if len(zeros):
            print("\n  kbet is EXACTLY 0.0000 for: "
                  + ", ".join(f"{r['method']}/{r['latent_key']}"
                              for _, r in zeros.iterrows()))
            print("  That is scib-metrics' `score = 0  # 100% rejection` fallback,"
                  "\n  reached when the per-label subgraph fragments, NOT a measured"
                  "\n  score. It is what a low-dimensional INTEGER representation"
                  "\n  does to a diffusion-based metric: eczema cell_code_indices has"
                  "\n  633 distinct rows for 53,655 cells and 94% tied neighbour"
                  "\n  distances. Report kbet on the continuous latent instead; iLISI"
                  "\n  on the codes is unaffected because it is purely kNN-based.")
        if "kbet" in attempted and not np.isfinite(sub["kbet_mean"]).any():
            print("\n  kbet is n/a for EVERY method here. That is a property of the"
                  "\n  annotation, not of the methods: scib-metrics skips any cell"
                  "\n  type confined to one batch. Check with"
                  "\n  label_batch_feasibility.py before reporting anything."
                  "\n"
                  "\n  AND DO NOT REPORT clisi OR casw FOR THIS DATASET EITHER. The"
                  "\n  same cause that voids kbet inverts them: if each label lives"
                  "\n  in one batch, then a representation that merely SEPARATES the"
                  "\n  batches gives every cell neighbours from its own batch only,"
                  "\n  hence from its own label vocabulary only, which scores as"
                  "\n  excellent label separation. clisi and casw here reward the"
                  "\n  opposite of integration. Report them on a dataset whose"
                  "\n  annotation is shared across batches.")

    raw.to_csv(long_csv, index=False)
    order.to_csv(wide_csv, index=False)
    print(f"\nwrote {long_csv}  ({len(raw)} per-seed rows)")
    print(f"wrote {wide_csv}  ({len(order)} method rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
