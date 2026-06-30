#!/usr/bin/env python3
"""
Spatial-IMPUTATION benchmark figure — a separate task from the reconstruction
benchmark (`plot_pearson_benchmark.py`), so it gets its own figure.

Task: predict a held-out region's per-cell expression from spatial context, with
the held-out cells' expression NEVER seen. Distinct from reconstruction (where
each method encodes the cell's own measured expression), hence a dedicated plot.

Bars (cell-level only; both methods are cell-branch):
  - SQUINT (imputed)  codes predicted from spatial context (expression unseen).
  - GeST (imputed)    reimplemented GeST baseline (Hao et al. MLCB 2025).
  (- SQUINT (recon) is shown in the reconstruction figure, not here.)

Each method's metrics are located flexibly via `--*-path`, which accepts:
  * a direct per_seed_pearson_reconstruction.csv / pearson_reconstruction_metrics.csv,
  * a run dir (searched, incl. a `metrics/` subdir and `<TS>/metrics/`),
  * or a variant name under <artifacts_root>/<dataset_tag>/.
This handles the SQUINT-imputed CSV living under the stage-1 run's
`.../<TS>/stage2/<TS2>/metrics/` rather than a top-level variant dir.

Pearson is computed by the SAME util as the reconstruction benchmark (read from
the CSV), on the TEST split (held-out region). Reuses plot_pearson_benchmark's
panel renderer / style.

Outputs (to --out-dir, default <artifacts>/benchmarking/figures/):
  <out_prefix>_<split>.{svg,png}     one figure, 1 x len(METRIC_VARIANTS) panels
  <out_prefix>.csv                   long-format values behind the figure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from plot_pearson_benchmark import (  # noqa: E402  (reuse the tested machinery)
    DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATASET_TAG, METRIC_VARIANTS,
    _plot_panel, _apply_nature_style,
)

# SQUINT (imputed) = the GeST-architecture transformer trained on SQUINT's
# discrete codes (best of the stage-2 imputation variants on held-out regions),
# decoded back to expression through the frozen SQUINT decoder. Using the SAME
# (GeST) architecture for both bars isolates the REPRESENTATION being imputed —
# SQUINT codes vs GeST meta-cell tokens — rather than confounding it with the
# stage-2 architecture. Override with --squint-imputed-path (variant / dir / CSV);
# the native MaskGIT stage-2 lives at "squint-imputed+region-holdout".
DEFAULT_SQUINT_IMPUTED = "squint-gestarch+region-holdout"   # variant OR a path
DEFAULT_GEST_IMPUTED = "gest-imputed+region-holdout"

COLOURS = {"SQUINT (imputed)": "#FF7AB6", "GeST (imputed)": "#2A9D8F",
           "SQUINT (recon)": "#FF006E"}
_CSV_NAMES = ("per_seed_pearson_reconstruction.csv", "pearson_reconstruction_metrics.csv")


def _resolve_metrics_csv(locator: str, artifacts_root: Path, dataset_tag: str):
    """Find a Pearson CSV from a file path, a run/metrics dir, or a variant name."""
    p = Path(locator)
    if p.is_file():
        return p
    search: List[Path] = []
    if p.is_dir():
        search += [p, p / "metrics"]
        # newest <TS>/metrics/ and <TS>/ underneath (handles stage2/<TS>/metrics)
        search += sorted([d / "metrics" for d in p.glob("*") if (d / "metrics").is_dir()],
                         reverse=True)
        search += sorted([d for d in p.glob("*") if d.is_dir()], reverse=True)
    else:
        vdir = artifacts_root / dataset_tag / locator      # variant under artifacts
        if vdir.is_dir():
            search += sorted([d / "metrics" for d in vdir.glob("*") if (d / "metrics").is_dir()],
                             reverse=True)
            search += [vdir / "metrics", vdir]
    for d in search:
        for n in _CSV_NAMES:
            if (d / n).is_file():
                return d / n
    return None


def _load_csv_filtered(csv_path: Path, axis: str, transform: str,
                       gene_subset: str, value_col: str = "pearson_mean"):
    """branch='cell' slice as (seed, split, value) reading `value_col` (returns
    empty if that column is absent — e.g. an OLD CSV that predates the panel
    metrics)."""
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        return pd.DataFrame()
    if "seed" not in df.columns:
        df = df.copy(); df["seed"] = 0
    df = df[df["branch"] == "cell"]
    for col, val in (("axis", axis), ("transform", transform), ("gene_subset", gene_subset)):
        if col in df.columns:
            df = df[df[col] == val]
    df = df[df[value_col].notna()]
    if df.empty:
        return pd.DataFrame()
    return df[["seed", "split", value_col]].rename(columns={value_col: "value"})


# Panel specs per --metric. Each entry: (suffix, axis, transform, gene_subset,
# value_col, title, higher_is_better). "panel" is the reviewer figure: one
# representative bar-group per complementary metric.
_PANEL_SPECS = {
    "panel": [
        ("pearson",      "gene_wise", "log1p",  "all",     "pearson_mean",  "Pearson (log1p) ↑",     True),
        ("spearman",     "gene_wise", "log1p",  "all",     "spearman_mean", "Spearman ↑",            True),
        ("rmse_log1p",   "gene_wise", "log1p",  "all",     "rmse_mean",     "RMSE (log1p) ↓",        False),
        ("rmse_counts",  "gene_wise", "raw",    "all",     "rmse_mean",     "RMSE (counts) ↓",       False),
        ("zero_nonzero", "entrywise", "counts", "all",     "auroc_zero",    "Zero/nonzero AUROC ↑",  True),
        ("markers",      "gene_wise", "log1p",  "markers", "pearson_mean",  "Marker Pearson ↑",      True),
    ],
    "pearson":  [(s, ax, tr, gs, "pearson_mean",  lbl + " ↑", True)
                 for s, ax, tr, gs, lbl in METRIC_VARIANTS],
    "spearman": [(s, ax, tr, gs, "spearman_mean", lbl + " ↑", True)
                 for s, ax, tr, gs, lbl in METRIC_VARIANTS],
    "mse":      [(s, ax, tr, gs, "mse_mean",      lbl + " ↓", False)
                 for s, ax, tr, gs, lbl in METRIC_VARIANTS],
    "rmse":     [(s, ax, tr, gs, "rmse_mean",     lbl + " ↓", False)
                 for s, ax, tr, gs, lbl in METRIC_VARIANTS],
    "zero_nonzero": [
        ("zero_nonzero_all",     "entrywise", "counts", "all",     "auroc_zero", "Zero/nonzero AUROC (all) ↑",     True),
        ("zero_nonzero_markers", "entrywise", "counts", "markers", "auroc_zero", "Zero/nonzero AUROC (markers) ↑", True),
    ],
}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-prefix", type=str, default="imputation_benchmark")
    p.add_argument("--split", type=str, default="test",
                   help="Pearson split to plot (held-out region = 'test').")
    p.add_argument("--metric", type=str, default="panel", choices=list(_PANEL_SPECS),
                   help="Which metric figure to render. 'panel' (default) = one "
                        "bar-group per complementary metric (Pearson / Spearman / "
                        "RMSE log1p+counts / zero-nonzero AUROC / marker Pearson) — "
                        "the reviewer panel. Others render the per-variant breakdown "
                        "for a single metric. Needs CSVs rebuilt with the new columns.")
    p.add_argument("--squint-imputed-path", type=str, default=DEFAULT_SQUINT_IMPUTED,
                   help="CSV / run dir / variant for SQUINT (imputed).")
    p.add_argument("--gest-imputed-path", type=str, default=DEFAULT_GEST_IMPUTED,
                   help="CSV / run dir / variant for GeST (imputed).")
    p.add_argument("--squint-recon-path", type=str, default=None,
                   help="Optional: add a SQUINT (recon) ceiling bar from this "
                        "CSV/dir/variant (off by default — it's in the recon figure).")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    methods: Dict[str, str] = {
        "SQUINT (imputed)": args.squint_imputed_path,
        "GeST (imputed)": args.gest_imputed_path,
    }
    if args.squint_recon_path:
        methods = {"SQUINT (recon)": args.squint_recon_path, **methods}

    _apply_nature_style()
    import plot_pearson_benchmark as ppb
    ppb.HATCH_METHODS |= {"SQUINT (imputed)", "GeST (imputed)"}   # recon stays solid

    # resolve each method's CSV once
    resolved: Dict[str, Optional[Path]] = {}
    print(f"Artifacts root: {args.artifacts_root}\nDataset: {args.dataset_tag}\nSplit: {args.split}")
    for label, locator in methods.items():
        csv = _resolve_metrics_csv(locator, args.artifacts_root, args.dataset_tag)
        resolved[label] = csv
        print(f"  {label:<18s} <- {csv if csv else f'MISSING (from {locator!r})'}")

    panels = _PANEL_SPECS[args.metric]
    long_rows: List[dict] = []
    per_panel_values: Dict[str, Dict[str, np.ndarray]] = {}
    per_panel_meta: Dict[str, Tuple[str, bool]] = {}
    for suffix, axis, transform, gene_subset, value_col, title, hib in panels:
        vals: Dict[str, np.ndarray] = {}
        for label, csv in resolved.items():
            if csv is None:
                continue
            df = _load_csv_filtered(csv, axis, transform, gene_subset, value_col)
            if df.empty:
                continue
            sub = df[df["split"] == args.split]
            v = sub["value"].astype(float).to_numpy()
            if v.size == 0:
                continue
            vals[label] = v
            for vv, sd in zip(v, sub["seed"].astype(int)):
                long_rows.append({
                    "metric": value_col, "panel": suffix, "axis": axis,
                    "transform": transform, "gene_subset": gene_subset,
                    "branch": "cell", "method": label, "split": args.split,
                    "seed": int(sd), "value": float(vv),
                })
        per_panel_values[suffix] = vals
        per_panel_meta[suffix] = (title, hib)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n + 0.6,
                                            max(1.35, 0.30 * len(methods) + 0.6)))
    if n == 1:
        axes = [axes]
    for ax, (suffix, _ax, _tr, _gs, _vc, mlabel, hib) in zip(axes, panels):
        vals = per_panel_values.get(suffix, {})
        # rank best-first: descending for higher-is-better, ascending otherwise.
        sign = -1.0 if hib else 1.0
        order = sorted(vals.keys(), key=lambda m: sign * float(np.nanmean(vals.get(m, [np.nan]))))
        _plot_panel(ax=ax, method_order=order, per_method_values=vals,
                    colour_for=COLOURS, title=mlabel)
    fig.suptitle(f"Spatial imputation — held-out region (expression unseen)  "
                 f"[{args.split}]", fontsize=7, fontweight="medium", y=1.03)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.55)

    base = args.out_dir / f"{args.out_prefix}_{args.metric}_{args.split}"
    for ext in ("svg", "png"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)

    if long_rows:
        csv_out = args.out_dir / f"{args.out_prefix}_{args.metric}.csv"
        pd.DataFrame(long_rows).to_csv(csv_out, index=False)
        print(f"  -> {csv_out}")


if __name__ == "__main__":
    sys.exit(main())
