#!/usr/bin/env python
"""
compute_label_conditioned_metrics.py — the scIB label-conditioned scores R2 asked for.
=============================================================================
R2-W1b: "MMD is useful but can be sensitive to representation scaling and does not
by itself establish preservation of cell-type-specific biology; label-conditioned
integration metrics would be preferable."

Table 1 reports MMD and iLISI, which measure batch mixing but say nothing about
whether cell-type structure survives it. The metrics that answer this are already
implemented in `vqniche.metrics.benchmarking` through `scib_metrics`, and
`scib-metrics>=0.5` is already a dependency. Nothing needs retraining: these are
computed on the saved `predicted_adata.h5ad` latents.

WHAT IS COMPUTED
  kbet   `scib_metrics.kbet_per_label` — batch mixing evaluated WITHIN each cell
         type. This is literally the label-conditioned integration metric R2
         asked for, and the one that answers the objection directly.
  clisi  cell-type LISI. The bio-conservation mirror of the iLISI we already
         report, so it drops straight into Table 1's existing columns.
  casw   cell-type silhouette. A second bio-conservation view.
  blisi  `scib_metrics.ilisi_knn`, i.e. OUR REPORTED iLISI. Not a new result —
         it is the REPRODUCTION GATE (see below).

THE REPRODUCTION GATE, READ THIS BEFORE TRUSTING ANY NUMBER
Choosing the wrong `--latent-key` or `--cell-type-key` produces plausible numbers
silently; we have been bitten by exactly that before (a sibling script defaulted
to a 40-class `cell_type` column where the paper used `new_annotation`). So this
script recomputes `blisi` alongside the new metrics and compares it against the
paper's published iLISI for the same representation. If blisi does not land near
that value, the keys are wrong and the new metrics are meaningless. Pass
`--expect-blisi 0.609` (mouse-brain niche latent, Table 1) to make the check
explicit; the script reports the discrepancy and exits non-zero on a bad match.

Run `--inspect` first. It prints every obsm key with its shape and every obs
column that looks like a label or batch with its cardinality, and computes
nothing. Pick the keys from that output rather than trusting the defaults.

USAGE
  # 1. look before you leap
  python compute_label_conditioned_metrics.py --inspect \\
      --adata /nfs/.../20260628_145939_seed0/predicted_adata.h5ad

  # 2. one seed, with the gate armed
  python compute_label_conditioned_metrics.py \\
      --adata /nfs/.../seed0/predicted_adata.h5ad \\
      --latent-keys H_adj --cell-type-key cell_type --batch-key adata_batch_id \\
      --expect-blisi 0.609

  # 3. all five seeds -> CSV
  python compute_label_conditioned_metrics.py --method SQUINT \\
      --adata /nfs/.../seed0/predicted_adata.h5ad \\
      --adata /nfs/.../seed1/predicted_adata.h5ad ... \\
      --out label_conditioned_metrics_mmb.csv

Needs the squint venv (the one that can `import vqniche`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

METRICS = ["blisi", "kbet", "clisi", "casw"]
LABEL_HINTS = ("cell_type", "cell_types", "celltype", "annotation", "new_annotation",
               "niche", "region", "domain", "leiden", "cluster")
BATCH_HINTS = ("batch", "section", "sample", "donor", "slide", "assay")


def inspect(path: Path) -> None:
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    print(f"\n=== {path} ===")
    print(f"  n_obs={a.n_obs}  n_vars={a.n_vars}")
    print("\n  obsm keys (candidates for --latent-keys):")
    for k in a.obsm.keys():
        try:
            shp = a.obsm[k].shape
        except Exception:                                        # noqa: BLE001
            shp = "?"
        print(f"    {k:32s} {shp}")
    print("\n  obs columns that look like labels or batches:")
    for c in a.obs.columns:
        lc = c.lower()
        tag = ("LABEL" if any(h in lc for h in LABEL_HINTS) else
               "BATCH" if any(h in lc for h in BATCH_HINTS) else None)
        if tag is None:
            continue
        col = a.obs[c]
        n = col.nunique(dropna=True)
        nan = int(col.isna().sum())
        print(f"    [{tag}] {c:30s} {n:6d} classes, {nan} NaN")
    print("\n  Pick --latent-keys / --cell-type-key / --batch-key from the above.")
    print("  The cell-type key MUST be the one Table 1 used, or the numbers are")
    print("  not comparable with the published columns.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adata", type=Path, action="append", required=True,
                   help="Repeat once per seed.")
    p.add_argument("--inspect", action="store_true",
                   help="Print available keys and exit without computing.")
    p.add_argument("--method", default="SQUINT")
    p.add_argument("--latent-keys", default="H_adj",
                   help="Comma list. Table 1 reports cell AND niche columns, so "
                        "pass both representations to fill both.")
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--batch-key", default="adata_batch_id")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--k", type=int, default=8,
                   help="Matches compute_benchmarking_metrics' default; the "
                        "metric-specific neighbour counts (50 for kbet, 90 for "
                        "LISI) are added internally, as in the paper's runs.")
    p.add_argument("--expect-blisi", type=float, default=None,
                   help="Published iLISI for this representation. Arms the "
                        "reproduction gate. Mouse-brain niche latent = 0.609.")
    p.add_argument("--blisi-tol", type=float, default=0.05,
                   help="Absolute tolerance for the gate.")
    p.add_argument("--out", type=Path, default=None,
                   help="CSV to write. Default: "
                        "label_conditioned_metrics_<method>.csv in the CWD. The "
                        "script REFUSES to overwrite an existing file (use "
                        "--force), so no previously computed table can be lost.")
    p.add_argument("--force", action="store_true",
                   help="Permit overwriting an existing --out file.")
    a = p.parse_args(argv)

    if a.inspect:
        for f in a.adata:
            inspect(f)
        return 0

    # Resolve and guard the output BEFORE computing anything: a name clash should
    # cost a second, not a full metric run.
    out = a.out or Path(f"label_conditioned_metrics_{a.method}.csv")
    if out.exists() and not a.force:
        raise SystemExit(
            f"{out} already exists. This script never overwrites results.\n"
            f"  Pass a different --out, or --force if you really mean to replace it.")

    import anndata as ad
    import pandas as pd
    try:
        from vqniche.metrics.benchmarking import compute_benchmarking_metrics
    except ImportError as ex:                                     # noqa: BLE001
        raise SystemExit(
            f"cannot import vqniche ({ex}). Activate the squint venv: the same "
            f"one that produced Table 1, so the metric code is identical.")

    latent_keys = [s.strip() for s in a.latent_keys.split(",") if s.strip()]
    rows = []
    for f in a.adata:
        seed = next((part.split("seed")[-1] for part in f.parts[::-1]
                     if "seed" in part), "?")
        adata = ad.read_h5ad(f)
        for key in ("cell_type_key", "batch_key"):
            col = getattr(a, key)
            if col not in adata.obs.columns:
                raise SystemExit(
                    f"{f}\n  --{key.replace('_','-')} {col!r} is not in obs. "
                    f"Present: {sorted(adata.obs.columns)[:25]} ...\n"
                    f"  Run with --inspect and pick the column Table 1 used.")
        for lk in latent_keys:
            if lk not in adata.obsm:
                raise SystemExit(
                    f"{f}\n  --latent-keys {lk!r} is not in obsm. "
                    f"Present: {list(adata.obsm.keys())}")
            print(f"\n--- seed {seed} | latent {lk} ---")
            d = compute_benchmarking_metrics(
                adata=adata, metrics=METRICS,
                cell_type_key=a.cell_type_key, batch_key=a.batch_key,
                spatial_key=a.spatial_key, latent_key=lk,
                k=a.k, seed=int(seed) if str(seed).isdigit() else 0)
            row = {"method": a.method, "seed": seed, "latent_key": lk,
                   "cell_type_key": a.cell_type_key, "batch_key": a.batch_key,
                   "n_cells": int(adata.n_obs), "path": str(f)}
            row.update({m: float(d.get(m, float("nan"))) for m in METRICS})
            rows.append(row)
            print("    " + "  ".join(f"{m}={row[m]:.4f}" for m in METRICS))

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78 + "\nMEAN +/- SD ACROSS SEEDS\n" + "=" * 78)
    print(f"  {'latent':22s}" + "".join(f"{m:>18s}" for m in METRICS))
    for lk, g in df.groupby("latent_key"):
        cells = "".join(f"{g[m].mean():>10.4f}+/-{g[m].std(ddof=1):<7.4f}"
                        for m in METRICS)
        print(f"  {lk:22s}{cells}")

    ok = True
    if a.expect_blisi is not None:
        print("\n" + "=" * 78 + "\nREPRODUCTION GATE\n" + "=" * 78)
        for lk, g in df.groupby("latent_key"):
            got = g["blisi"].mean()
            d = abs(got - a.expect_blisi)
            verdict = "PASS" if d <= a.blisi_tol else "FAIL"
            ok &= (d <= a.blisi_tol)
            print(f"  {lk:22s} blisi {got:.4f} vs published iLISI "
                  f"{a.expect_blisi:.4f}  |diff| {d:.4f}  {verdict}")
        if not ok:
            print("\n  *** The gate FAILED. blisi is our own reported iLISI, so a")
            print("  mismatch means the latent or label keys differ from the ones")
            print("  Table 1 used. kbet/clisi/casw above are therefore NOT")
            print("  comparable with the published columns. Re-run --inspect and")
            print("  fix the keys before quoting anything. ***")
        else:
            print("\n  Gate passed: the keys reproduce our published iLISI, so the")
            print("  label-conditioned columns are on the same footing as Table 1.")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:      # re-check: a concurrent run may have won
        raise SystemExit(f"{out} appeared while computing; refusing to overwrite.")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
