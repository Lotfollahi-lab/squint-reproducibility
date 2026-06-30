#!/usr/bin/env python
"""
Gene-expression RECONSTRUCTION benchmark — the SAME multi-metric grid as the
imputation figure, but for reconstruction: every method reconstructs each cell's
OWN expression (expression IS seen), so this measures decoder fidelity rather
than spatial in-painting.

Compares (cell-level, default): scVI vs NicheCompass vs SQUINT
(+ optional Vanilla VQ-VAE). Reads each method's
`per_seed_pearson_reconstruction.csv` (written by `_holdout_utils.
build_pearson_dataframe`, which now carries the full metric panel), so this is
a thin wrapper over `plot_imputation_benchmark.render_metric_grid`.

Grid (rows = metric, cols = view): Pearson / Spearman / RMSE (log1p + counts)
each gene-wise AND cell-wise, marker Pearson, and zero/nonzero AUROC. Each panel
shows one direction arrow (↑ higher-better / ↓ lower-better).

Default split = 'all' (full reconstruction). The CSVs must be rebuilt with the
new metric columns (re-run the imputation/recon runners, or rescore the saved
adata) — otherwise only the Pearson panels fill.

Usage:
  python analysis/benchmarking/plots/plot_reconstruction_benchmark.py
  python ... --dataset-tag xhs1000-3b_1p --split all
  python ... --squint-path <variant-or-dir-or-csv> --include-vanilla
  python ... --metric pearson           # single-metric breakdown
  python ... --extra-variant <dir> "SQUINT (small cb)"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from plot_imputation_benchmark import (  # noqa: E402  (reuse the grid machinery)
    render_metric_grid, _GRID_ROWS, _PANEL_SPECS, _resolve_metrics_csv,
)
from plot_pearson_benchmark import (  # noqa: E402  (shared style + recon defaults)
    _apply_nature_style, DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATASET_TAG,
    DEFAULT_SQUINT_VARIANT, METHOD_COLOURS,
)

# label -> variant/dir/CSV (resolved by _resolve_metrics_csv). Mirrors the
# cell-level row of plot_pearson_benchmark.ROWS so the same runs are read.
DEFAULT_SCVI = "baseline-scvi+region-holdout"
DEFAULT_NICHECOMPASS = "baseline-nichecompass+region-holdout"
DEFAULT_VANILLA = "vanilla-vq-cell+region-holdout"
# extra (non-default-coloured) method shades
_EXTRA_SHADES = ["#FF7AB6", "#B5179E", "#7209B7", "#F72585"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-prefix", type=str, default="reconstruction_benchmark")
    p.add_argument("--split", type=str, default="all",
                   help="Pearson split to plot. Default 'all' (full reconstruction).")
    p.add_argument("--metric", type=str, default="panel",
                   choices=["panel", *_PANEL_SPECS],
                   help="'panel' (default) = full multi-metric grid; else the "
                        "single-metric breakdown.")
    p.add_argument("--scvi-path", type=str, default=DEFAULT_SCVI,
                   help="scVI variant / run dir / CSV.")
    p.add_argument("--nichecompass-path", type=str, default=DEFAULT_NICHECOMPASS,
                   help="NicheCompass variant / run dir / CSV.")
    p.add_argument("--squint-path", type=str, default=DEFAULT_SQUINT_VARIANT,
                   help="SQUINT variant / run dir / CSV (default: the wide-decoder "
                        "reconstruction reference).")
    p.add_argument("--include-vanilla", action="store_true",
                   help="Also add a Vanilla VQ-VAE bar (vanilla-vq-cell+region-holdout).")
    p.add_argument("--extra-variant", nargs=2, action="append",
                   metavar=("VARIANT", "LABEL"), default=None,
                   help="Add an EXTRA method bar (VARIANT = __multiseed dir / run / "
                        "CSV, no timestamp; LABEL = display name). Repeatable.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Method -> locator, in plot order. scVI / NicheCompass / SQUINT by default.
    methods: Dict[str, str] = {
        "scVI": args.scvi_path,
        "NicheCompass": args.nichecompass_path,
        "SQUINT": args.squint_path,
    }
    if args.include_vanilla:
        methods["Vanilla VQ-VAE"] = DEFAULT_VANILLA
    colours = dict(METHOD_COLOURS)
    for i, (variant, label) in enumerate(args.extra_variant or []):
        methods[label] = variant
        colours.setdefault(label, _EXTRA_SHADES[i % len(_EXTRA_SHADES)])

    _apply_nature_style()
    print(f"Artifacts root: {args.artifacts_root}\nDataset: {args.dataset_tag}\n"
          f"Split: {args.split}")
    resolved: Dict[str, Optional[Path]] = {}
    for label, locator in methods.items():
        csv = _resolve_metrics_csv(locator, args.artifacts_root, args.dataset_tag)
        resolved[label] = csv
        print(f"  {label:<16s} <- {csv if csv else f'MISSING (from {locator!r})'}")

    grid_rows = _GRID_ROWS if args.metric == "panel" else [_PANEL_SPECS[args.metric]]
    suptitle = f"Gene-expression reconstruction  [{args.split}]"
    render_metric_grid(
        resolved, grid_rows, args.split,
        fig_base=args.out_dir / f"{args.out_prefix}_{args.metric}_{args.split}",
        long_csv_path=args.out_dir / f"{args.out_prefix}_{args.metric}.csv",
        suptitle=suptitle, colours=colours)


if __name__ == "__main__":
    sys.exit(main())
