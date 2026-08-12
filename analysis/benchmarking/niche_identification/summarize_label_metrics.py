#!/usr/bin/env python
"""
summarize_label_metrics.py — pivot the per-(metric, dataset, method) csvs into a table.
=============================================================================
compute_label_metric.py writes LONG csvs, one row per (metric, seed, latent_key) with
the number in `value`. This concatenates every lm_*.csv, averages over seeds, and
pivots metrics into columns, one block per dataset.

Rows are (method, representation), and each dataset is split into the paper's TWO
comparisons: cell-type identification and niche identification. The method-to-task
mapping is read off the METHODS table in submit_all_benchmarks.sh:465, whose section
headers define it. SQUINT appears in both, split by representation rather than by
method: cell_emb / cell_latent are its cell-type entry, neighborhood_emb /
neighborhood_latent its niche entry, the same pairing Table 1 uses.

SQUINT contributes four representations, the four the paper's own integration metrics
use (see any run's metrics/batch_integration_metrics.csv). Baselines contribute one,
on the emb_key their published iLISI was computed on.

READ cilisi AND kbet_strat AS THE ANSWER TO R2-W1b: both are batch mixing computed
WITHIN each cell type. basw and bras are the silhouette-based alternatives; note that
Rautenstrauch & Ohler (Nat Biotechnol 2025) document Batch ASW behaving erratically
and propose bras as its replacement, so prefer bras over basw if a silhouette metric
is reported at all. ilisi is the UNCONDITIONED metric already in Table 1, present as
the anchor the conditioned ones should be read against, not as a new result.

n_types_scored is carried through: every label-conditioned metric averages only over
cell types with >=10 cells and >=2 batches, so the denominator matters. Where it is 0
the metric is undefined rather than bad, which is the mouse brain, whose 49 cell types
are two disjoint per-section vocabularies.

USAGE
  python summarize_label_metrics.py
  python summarize_label_metrics.py --dataset xhs1000-3b_1p --sort-by cilisi
  python summarize_label_metrics.py --task "niche identification"   # one block only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["cilisi", "cilisi_means", "kbet_strat", "kbet_label", "bras", "basw",
         "cmmd", "cmmd_means", "graph_conn", "ilisi", "mmd", "clisi", "casw"]
# MMD is a DISTANCE, so lower is better; every other column here is
# higher-is-better. Sorting by one of these ascends instead of descends.
LOWER_IS_BETTER = {"mmd", "cmmd", "cmmd_means"}
DEFAULT_DIR = (Path("/nfs/team361/sb75/squint-reproducibility/artifacts")
               / "label_conditioned_metrics")

# The two benchmark families, split exactly as the paper splits them. Taken from the
# METHODS table in analysis/benchmarking/submit_all_benchmarks.sh:465, which carries
# the section headers "Cell-type identification" and "Niche identification", so this is
# read off the submission table rather than inferred from method names.
CELL_TYPE_METHODS = {"pca-leiden", "harmony", "scvi", "geneformer", "nicheformer",
                     "scgpt", "scgpt-spatial", "uce"}
NICHE_METHODS = {"banksy", "cellcharter", "graphst", "novae", "nichecompass",
                 "neigh-expr-pca"}
TASKS = ("cell-type identification", "niche identification", "unassigned")

# The label each comparison must be CONDITIONED on. The paper's two comparisons use
# different annotations, so a label-conditioned metric has to follow: scoring niche
# methods against the cell-type annotation answers a different question from the one
# the niche comparison asks. Niche keys from DEFAULT_NICHE_LABEL_KEYS
# (plot_ablations.py:122) plus xhs1000's explicit niche_type override.
NICHE_LABEL_KEYS = {"niche", "niche_type", "spatial_cluster", "NICHE_NAMES",
                    "Sub_molecular_tissue_region", "ccf_region_name",
                    "Main_molecular_tissue_region", "major_brain_region"}


def label_kind(cell_type_key: str) -> str:
    """'niche' if the metric was conditioned on a niche annotation, else 'cell'."""
    return "niche" if str(cell_type_key) in NICHE_LABEL_KEYS else "cell"


def task_of(method: str, latent_key: str) -> str:
    """
    Which of the paper's two comparisons a row belongs to.

    SQUINT is split by REPRESENTATION rather than by method, because it contributes to
    both: the cell codebook (cell_emb / cell_latent) is the cell-type entry and the
    niche codebook (neighborhood_emb / neighborhood_latent) the niche entry. That is
    the same pairing Table 1 uses.
    """
    if str(method).upper().startswith("SQUINT"):
        return ("cell-type identification" if str(latent_key).startswith("cell")
                else "niche identification")
    m = str(method)
    if m.startswith("baseline-"):
        m = m[len("baseline-"):]
    if m in CELL_TYPE_METHODS:
        return "cell-type identification"
    if m in NICHE_METHODS:
        return "niche identification"
    return "unassigned"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--pattern", default="lm_*.csv")
    p.add_argument("--dataset", default=None)
    p.add_argument("--sort-by", default="cilisi")
    p.add_argument("--task", default=None, choices=TASKS,
                   help="Restrict to one of the paper's two comparisons.")
    p.add_argument("--out-prefix", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    files = sorted(f for f in a.dir.glob(a.pattern)
                   if not f.name.startswith("summary_"))
    if not files:
        raise SystemExit(f"no {a.pattern} in {a.dir}; run submit_label_metrics.sh")
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # cell_type_key MUST be part of the identity. The same (metric, dataset, method,
    # representation, seed) is computed TWICE, once conditioned on the cell-type label
    # and once on the niche label, and those are different numbers answering different
    # questions. Leaving the key out made them look like duplicates, and keep="first"
    # silently dropped every niche-conditioned row (the non-_niche csv sorts first),
    # so the niche block appeared empty while 27 *_niche.csv files sat on disk.
    raw = raw.drop_duplicates(
        subset=[c for c in ("metric", "dataset", "method", "latent_key", "seed",
                            "cell_type_key", "batch_key") if c in raw.columns],
        keep="first")
    if a.dataset:
        raw = raw[raw["dataset"] == a.dataset]
        if raw.empty:
            raise SystemExit(f"no rows for {a.dataset!r}")

    err = (raw["error"].fillna("").astype(str).str.strip()
           .replace({"nan": "", "None": ""}) if "error" in raw else
           pd.Series("", index=raw.index))
    if (err != "").any():
        bad = raw[err != ""]
        print(f"!! {len(bad)} of {len(raw)} rows recorded an error, excluded from the "
              f"means (kept in the long csv):")
        for (mt, ds), g in bad.groupby(["metric", "dataset"]):
            print(f"   {mt:14s} {ds:22s} {len(g)} rows  "
                  f"{str(g['error'].iloc[0])[:70]}")
        raw = raw[err == ""]

    # cell_type_key is part of the grouping for the same reason.
    keys = ["dataset", "method", "latent_key", "cell_type_key", "metric"]
    g = raw.groupby(keys, dropna=False)["value"]
    stat = g.agg(["mean", "std", "count"]).reset_index()
    types = (raw.groupby(keys)["n_types_scored"].first().reset_index()
             if "n_types_scored" in raw else None)

    prefix = a.out_prefix or (a.dir / "summary_label_metrics")
    long_csv, wide_csv = Path(f"{prefix}_long.csv"), Path(f"{prefix}_wide.csv")
    for out in (long_csv, wide_csv):
        if out.exists() and not a.force:
            raise SystemExit(f"{out} exists; pass --force")

    IDX = ["dataset", "method", "latent_key", "cell_type_key"]
    wide = stat.pivot_table(index=IDX, columns="metric",
                            values="mean").reset_index()
    sd = stat.pivot_table(index=IDX, columns="metric",
                          values="std").reset_index()
    cols = [m for m in ORDER if m in wide.columns]
    cols += [c for c in wide.columns if c not in cols
             and c not in ("dataset", "method", "latent_key", "cell_type_key")]

    wide["task"] = [task_of(m, k) for m, k in
                    zip(wide["method"], wide["latent_key"])]
    wide["label_kind"] = [label_kind(k) for k in wide["cell_type_key"]]
    # Each block shows only the rows conditioned on ITS label. Rows conditioned on the
    # other one are held back and counted, rather than mixed into the table, because a
    # niche method scored against cell-type labels is not the niche comparison.
    WANT = {"cell-type identification": "cell", "niche identification": "niche"}
    for ds, dsub in wide.groupby("dataset", sort=False):
        sortcol = a.sort_by if a.sort_by in dsub.columns else cols[0]
        asc = sortcol in LOWER_IS_BETTER
        print("\n" + "=" * 104)
        print(f"{ds}   sorted by {sortcol} "
              f"({'lower' if asc else 'higher'} is better); "
              f"cmmd/mmd are DISTANCES so lower is better, all others higher")
        print("=" * 104)
        for task in ([a.task] if a.task else TASKS):
            sub = dsub[dsub["task"] == task]
            want = WANT.get(task)
            if want is not None:
                mismatched = sub[sub["label_kind"] != want]
                sub = sub[sub["label_kind"] == want]
            note = ""
            if want is not None and len(mismatched):
                keys = sorted(set(mismatched["cell_type_key"]))
                if sub.empty:
                    # Nothing to show: this comparison has not been computed yet.
                    note = (f"     NOT COMPUTED. {len(mismatched)} row(s) exist but "
                            f"are conditioned on {keys}; this comparison needs a "
                            f"{want} label. Run LABEL_KIND={want}.")
                else:
                    # Both label kinds are present. The extras are the off-diagonal,
                    # e.g. cell-type methods scored against the niche annotation,
                    # which the niche run also produces. Not missing work, so no
                    # rerun advice here.
                    note = (f"     ({len(mismatched)} off-diagonal row(s) not shown: "
                            f"other methods scored against {keys}.)")
            if sub.empty:
                if note:
                    print(f"\n  -- {task} --")
                    print(note)
                continue
            sub = sub.sort_values(sortcol, ascending=asc, na_position="last")
            print()
            if task == "unassigned":
                print("     (method not in submit_all_benchmarks.sh's METHODS table; "
                      "assign it there or in CELL_TYPE_METHODS / NICHE_METHODS)")
            print(f"  -- {task} (label: "
                  f"{sorted(set(sub['cell_type_key']))[0]}) --"
                  if want else f"  -- {task} --")
            if note:
                print(note)
            print(f"  {'method':25s}{'representation':26s}"
                  + "".join(f"{m:>13s}" for m in cols))
            for _, r in sub.iterrows():
                cells = ""
                for m in cols:
                    v = r.get(m, np.nan)
                    cells += "          n/a" if not np.isfinite(v) else f"{v:>13.4f}"
                print(f"  {r['method']:25s}{r['latent_key']:26s}{cells}")

    long_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(long_csv, index=False)
    wide.merge(sd, on=IDX, suffixes=("", "_sd")).to_csv(wide_csv, index=False)
    print(f"\nwrote {long_csv}  ({len(raw)} per-seed rows)")
    print(f"wrote {wide_csv}  ({len(wide)} method rows; _sd columns hold the sd)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
