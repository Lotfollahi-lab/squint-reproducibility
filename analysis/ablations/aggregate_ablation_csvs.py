#!/usr/bin/env python
"""
Combine the per-axis ablation CSVs (+ the discretization comparison) into ONE
file for a paper table.

`plot_ablations.py` writes one long-format CSV per ablation axis next to each
SVG (e.g. `axis_1_adjacency.csv`, ..., `axis_11_*.csv`), columns
`axis, prefix, variant, label, is_default, metric, value, *_key`.

`compare_discrete_vs_continuous.py` writes the discrete-vs-continuous
("discretization") comparison separately as `discretization_summary.csv`
(`condition, branch, metric, mean, std, n`). This script folds that in as an
extra pseudo-axis `axis_12_discretization` so it lands in the same table.

Outputs (to --ablations-dir, or --out-*):
  ablations_all_metrics_long.csv   every (axis, variant, metric) row, stacked
                                   (incl. discretization; carries std/n where
                                   available -- NaN for the point-value axes).
  ablations_all_metrics_wide.csv   table-ready: one row per (axis, variant),
                                   one column per metric, default variant first.

No re-run of the plotting scripts needed -- reads the CSVs already on disk.

Usage:
  python analysis/ablations/aggregate_ablation_csvs.py \
    --ablations-dir /nfs/team361/sb75/squint-reproducibility/artifacts/mmb0-1b_smb1-1b_1p/ablations
  # explicit discretization CSV (else auto-discovered near the ablations dir):
  #   --discretization-summary /nfs/.../comparison_vs_discrete/discretization_summary.csv
  #   --no-discretization   to skip it
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

METRIC_ORDER = [
    "Niche NMI", "Niche ARI", "Niche MMD", "Niche iLISI",
    "Cell NMI", "Cell ARI", "Cell MMD", "Cell iLISI",
]
DISCRETIZATION_AXIS = "axis_12_discretization"


def _axis_num(axis_key: str) -> int:
    """'axis_10_coupling' -> 10 (so axis_2 sorts before axis_10)."""
    m = re.search(r"axis_(\d+)", str(axis_key))
    return int(m.group(1)) if m else 999


def _find_discretization(ablations_dir: Path, explicit) -> "Path | None":
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    # bounded-depth search near the ablations dir (artifacts/<dataset>/...)
    cands = []
    for root in (ablations_dir.parent, ablations_dir.parent.parent):
        if root.is_dir():
            cands += list(root.glob("*/*/comparison_vs_discrete/discretization_summary.csv"))
            cands += list(root.glob("*/comparison_vs_discrete/discretization_summary.csv"))
    cands = list(dict.fromkeys(cands))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _load_discretization(path: Path) -> pd.DataFrame:
    """discretization_summary.csv -> ablation-long schema rows."""
    d = pd.read_csv(path)
    need = {"condition", "branch", "metric", "mean"}
    if not need.issubset(d.columns):
        raise ValueError(f"{path} missing columns {need - set(d.columns)}")
    branch = d["branch"].astype(str).str.strip().str.capitalize()   # cell->Cell, niche->Niche
    metric = d["metric"].astype(str).str.strip()                    # NMI/ARI/iLISI/MMD (as written)
    out = pd.DataFrame({
        "axis": DISCRETIZATION_AXIS,
        "variant": d["condition"].astype(str),
        "label": d["condition"].astype(str),
        # the discrete-VQ-codes condition is the reference/default
        "is_default": d["condition"].astype(str).str.strip().str.lower().eq("discrete codes"),
        "metric": branch + " " + metric,                            # -> "Cell NMI", "Niche iLISI"
        "value": d["mean"].astype(float),
    })
    if "std" in d.columns:
        out["std"] = d["std"].astype(float)
    if "n" in d.columns:
        out["n"] = d["n"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablations-dir", required=True, type=Path,
                   help="Dir holding the per-axis axis_*.csv (next to the SVGs).")
    p.add_argument("--glob", default="axis_*.csv")
    p.add_argument("--discretization-summary", default=None,
                   help="Path to discretization_summary.csv (else auto-discovered).")
    p.add_argument("--no-discretization", action="store_true",
                   help="Skip the discrete-vs-continuous comparison.")
    p.add_argument("--out-long", type=Path, default=None)
    p.add_argument("--out-wide", type=Path, default=None)
    args = p.parse_args(argv)

    files = sorted(args.ablations_dir.glob(args.glob),
                   key=lambda f: (_axis_num(f.stem), f.name))
    files = [f for f in files if not f.name.startswith("ablations_all_metrics")]
    if not files:
        print(f"ERROR: no files matching {args.glob} in {args.ablations_dir}",
              file=sys.stderr)
        return 1
    print(f"Found {len(files)} per-axis CSV(s):")
    for f in files:
        print(f"  {f.name}")

    parts = [pd.read_csv(f) for f in files]

    # --- discretization comparison (extra pseudo-axis) --------------------
    if not args.no_discretization:
        disc = _find_discretization(args.ablations_dir, args.discretization_summary)
        if disc is not None:
            try:
                parts.append(_load_discretization(disc))
                print(f"  + discretization: {disc}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ! discretization skipped ({disc}): {exc}", file=sys.stderr)
        else:
            print("  ! discretization_summary.csv not found near the ablations dir "
                  "(pass --discretization-summary, or --no-discretization to silence).",
                  file=sys.stderr)

    long_df = pd.concat(parts, ignore_index=True)
    long_df["_axis_num"] = long_df["axis"].map(_axis_num)
    long_df = long_df.sort_values(["_axis_num", "axis"], kind="stable").drop(columns="_axis_num")

    out_long = args.out_long or (args.ablations_dir / "ablations_all_metrics_long.csv")
    long_df.to_csv(out_long, index=False)
    print(f"\n-> {out_long}  ({len(long_df)} rows, "
          f"{long_df['axis'].nunique()} axes, {long_df['metric'].nunique()} metrics)")

    # --- wide / table-ready ------------------------------------------------
    idx_cols = [c for c in ("axis", "variant", "label", "is_default")
                if c in long_df.columns]
    wide = long_df.pivot_table(
        index=idx_cols, columns="metric", values="value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    metric_cols = ([m for m in METRIC_ORDER if m in wide.columns]
                   + [c for c in wide.columns
                      if c not in METRIC_ORDER and c not in idx_cols])
    wide = wide[idx_cols + metric_cols]

    wide["_axis_num"] = wide["axis"].map(_axis_num)
    sort_cols, asc = ["_axis_num"], [True]
    if "is_default" in wide.columns:
        sort_cols.append("is_default"); asc.append(False)   # default first
    wide = (wide.sort_values(sort_cols, ascending=asc, kind="stable")
                .drop(columns="_axis_num").reset_index(drop=True))

    out_wide = args.out_wide or (args.ablations_dir / "ablations_all_metrics_wide.csv")
    wide.to_csv(out_wide, index=False)
    print(f"-> {out_wide}  ({len(wide)} variant rows x {len(metric_cols)} metrics)")
    print("   axes:", list(dict.fromkeys(wide["axis"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
