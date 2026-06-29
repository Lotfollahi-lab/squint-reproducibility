#!/usr/bin/env python3
"""
Summarize the s64 squint_hln SINGLE-SEED diagnostic sweep.

s64 was run single-seed with the aggregator OFF (submit_s64_squint_hln_diag.sh),
so there are NO `<variant>__multiseed/<TS>/metrics/per_seed_*.csv` files for
summarize_ablation_multiseed.py to read. Instead each variant has a per-run
metrics dir:

    <artifacts>/<dataset>/<variant>/<latest_TS>/metrics/
        niche_identification_metrics.csv   (split, code_key, label_key, NMI, ARI)
        pearson_reconstruction_metrics.csv (optional, recon-quality proxy)

This reads those directly, tabulates Cell/Niche NMI & ARI for every s64_v*
variant PLUS the un-ablated FiLM-scale squint_hln reference, ranks them, and
shows the delta vs the reference on the chosen metric — so you can see which
knob (adjacency weight, batch size, recon weight, ...) actually moved
resolution.

pandas/numpy only (no torch/anndata). Run on the farm (reads the artifacts).

Usage:
    python summarize_s64.py
    python summarize_s64.py --sort-by "Niche NMI"
    python summarize_s64.py --dataset squint_hln --artifacts-root /nfs/.../artifacts
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"
DEFAULT_DATASET = "squint_hln"
# the un-ablated FiLM-scale squint_hln reference (full-panel) the s64 variants modify
REFERENCE_VARIANT = (
    "dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16"
    "+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5"
    "+filmscale+squint_hln"
)

CELL_CODE_KEY = "cell_code_indices[level_0]"
NICHE_CODE_KEY = "neighborhood_code_indices[level_0]"
CELL_LABELS = ("cell_type", "cell_types", "annotation")
NICHE_LABELS = ("niche", "niche_annotation", "spatial_cluster")
# (display, code_key, branch, column)
METRICS = [
    ("Cell NMI", CELL_CODE_KEY, "cell", "NMI"),
    ("Cell ARI", CELL_CODE_KEY, "cell", "ARI"),
    ("Niche NMI", NICHE_CODE_KEY, "niche", "NMI"),
    ("Niche ARI", NICHE_CODE_KEY, "niche", "ARI"),
]
METRIC_NAMES = [m[0] for m in METRICS]


def _latest_run_dir(variant_dir: Path):
    """Latest <TS>(_seedN)/ subdir that has metrics/niche_identification_metrics.csv."""
    cands = []
    for d in variant_dir.iterdir() if variant_dir.is_dir() else []:
        if d.is_dir() and (d / "metrics" / "niche_identification_metrics.csv").is_file():
            cands.append(d)
    if not cands:
        return None
    # sort by the leading timestamp in the dir name (YYYYmmdd_HHMMSS...), newest last
    cands.sort(key=lambda p: p.name)
    return cands[-1]


def _effective_code_key(d, requested, branch):
    """RVQ exposes `..._code_indices[level_0]`; single-level VQ (the no-l1
    variants) writes the BARE key. Prefer level_0, fall back to the bare key."""
    if "code_key" not in d.columns:
        return None
    keys = set(d["code_key"].astype(str).unique())
    if requested in keys:
        return requested
    stem = "cell_code_ind" if branch == "cell" else "neighborhood_code_ind"
    cands = [k for k in keys if k.startswith(stem) and not k.endswith("[composite]")]
    if not cands:
        return None
    bare = [k for k in cands if "[" not in k]
    return sorted(bare)[0] if bare else sorted(cands, key=len)[0]


def _read_metric(run_dir: Path, code_key, branch, col):
    f = run_dir / "metrics" / "niche_identification_metrics.csv"
    if not f.is_file():
        return np.nan
    d = pd.read_csv(f)
    if "split" in d.columns and (d["split"] == "all").any():
        d = d[d["split"] == "all"]
    ck = _effective_code_key(d, code_key, branch)
    if ck is None:
        return np.nan
    d = d[d["code_key"].astype(str) == ck]
    pref = CELL_LABELS if branch == "cell" else NICHE_LABELS
    have = set(d["label_key"].astype(str).unique()) if "label_key" in d.columns else set()
    lk = next((l for l in pref if l in have), None)
    if lk is None:
        return np.nan
    vals = pd.to_numeric(d[d["label_key"].astype(str) == lk][col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else np.nan


def _discover_variants(base: Path):
    """All s64_v* variant dirs (not __multiseed), sorted by version index."""
    out = []
    for d in sorted(base.glob("s64_v*")):
        if d.is_dir() and not d.name.endswith("__multiseed"):
            out.append(d)
    def _vidx(p):
        m = re.match(r"s64_v(\d+)_", p.name)
        return int(m.group(1)) if m else 9999
    out.sort(key=_vidx)
    return out


def _label(variant_name: str) -> str:
    """Short label from the variant dir name."""
    m = re.match(r"(s64_v\d+_[^+]+)", variant_name)
    if m:
        return m.group(1)
    return variant_name[:40]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--reference", default=REFERENCE_VARIANT,
                   help="Variant dir name of the un-ablated reference (baseline row).")
    p.add_argument("--sort-by", default="Niche NMI", choices=METRIC_NAMES + ["mean4"])
    p.add_argument("--out", default=None,
                   help="Output CSV (default: <artifacts>/<dataset>/_s64_summary/s64_summary.csv).")
    args = p.parse_args(argv)

    base = Path(args.artifacts_root) / args.dataset
    if not base.is_dir():
        raise SystemExit(f"not found: {base}")

    rows = []
    # reference first (if present), then the s64 variants
    targets = []
    ref_dir = base / args.reference
    if ref_dir.is_dir():
        targets.append((ref_dir, "REFERENCE (FiLM-scale)", True))
    else:
        print(f"[s64] WARNING: reference dir not found ({ref_dir.name}); "
              f"deltas will be vs NaN.", file=sys.stderr)
    for d in _discover_variants(base):
        targets.append((d, _label(d.name), False))

    if len(targets) == (1 if ref_dir.is_dir() else 0):
        raise SystemExit(f"No s64_v* variant runs found under {base}")

    for variant_dir, label, is_ref in targets:
        run = _latest_run_dir(variant_dir)
        row = {"variant": label, "is_ref": is_ref,
               "run": run.name if run else "(no run)"}
        if run is None:
            for name in METRIC_NAMES:
                row[name] = np.nan
        else:
            for name, ck, branch, col in METRICS:
                row[name] = _read_metric(run, ck, branch, col)
        row["mean4"] = float(np.nanmean([row[n] for n in METRIC_NAMES]))
        rows.append(row)

    df = pd.DataFrame(rows)

    # delta vs reference on each metric
    ref_row = df[df["is_ref"]]
    ref_vals = {n: (float(ref_row[n].iloc[0]) if len(ref_row) else np.nan)
                for n in METRIC_NAMES + ["mean4"]}
    for n in METRIC_NAMES:
        df[f"Δ {n}"] = df[n] - ref_vals[n]

    # rank the non-reference variants by the chosen metric (desc); keep ref pinned on top
    sort_col = args.sort_by
    variants = df[~df["is_ref"]].sort_values(sort_col, ascending=False, na_position="last")
    ordered = pd.concat([df[df["is_ref"]], variants], ignore_index=True)

    out = args.out or str(base / "_s64_summary" / "s64_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ordered.to_csv(out, index=False)

    # pretty print
    show = ["variant"] + METRIC_NAMES + [f"Δ {sort_col}", "run"]
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(f"\n=== s64 squint_hln diagnostic — ranked by {sort_col} "
              f"(↑ better) ===")
        print(ordered[show].to_string(index=False))
    print(f"\n[s64] reference {sort_col} = {ref_vals.get(sort_col, float('nan')):.3f}")
    best = variants.iloc[0] if len(variants) else None
    if best is not None and np.isfinite(best[sort_col]):
        print(f"[s64] best variant: {best['variant']}  {sort_col}={best[sort_col]:.3f} "
              f"(Δ {best[sort_col] - ref_vals.get(sort_col, np.nan):+.3f} vs ref)")
    print(f"[s64] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
