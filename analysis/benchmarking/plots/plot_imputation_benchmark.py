#!/usr/bin/env python3
"""
Spatial-IMPUTATION benchmark figure — a separate task from the reconstruction
benchmark (`plot_pearson_benchmark.py`), so it gets its own figure.

Task: predict a held-out region's per-cell expression from spatial context, with
the held-out cells' expression NEVER seen (not in training, not at inference).
Distinct from reconstruction, where each method encodes the cell's own measured
expression. Hence a dedicated plot rather than mixing the bars.

Bars (cell-level only; both imputation methods are cell-branch):
  - SQUINT (recon)    the SAME dec-w32 model's reconstruction on the held-out
                      cells (expression SEEN) — the within-task ceiling/reference.
  - SQUINT (imputed)  codes predicted from spatial context (expression unseen).
  - GeST (imputed)    reimplemented GeST baseline (Hao et al. MLCB 2025), same task.

Pearson is computed by the SAME util as the reconstruction benchmark, read from
each variant's metrics CSV, on the TEST split (the held-out region). Reuses
`plot_pearson_benchmark`'s loader / panel renderer / palette so the numbers and
styling stay identical.

Outputs (to --out-dir, default <artifacts>/benchmarking/figures/):
  <out_prefix>_<split>.{svg,png}     one figure, 1 x len(METRIC_VARIANTS) panels
  <out_prefix>.csv                   long-format values behind the figure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from plot_pearson_benchmark import (  # noqa: E402  (reuse the tested machinery)
    DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATASET_TAG, METRIC_VARIANTS,
    load_method_pearson, _plot_panel, _apply_nature_style,
)

# Same dec-w32 FiLM-scale region-holdout run that SQUINT (imputed) was decoded
# from, so recon vs imputed is a clean SAME-MODEL comparison (not the wide-decoder
# headline reference used in the reconstruction figure).
DEFAULT_SQUINT_RECON = (
    "dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32"
    "+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc"
    "+diversity-w10+filmscale+crossmnn-wt10-k1+region-holdout+mmb0-1b_smb1-1b_1p"
)
DEFAULT_SQUINT_IMPUTED = "squint-imputed+region-holdout"
DEFAULT_GEST_IMPUTED = "gest-imputed+region-holdout"

# magenta family for SQUINT (recon solid / imputed lighter+hatched), teal for GeST.
COLOURS = {
    "SQUINT (recon)":   "#FF006E",
    "SQUINT (imputed)": "#FF7AB6",
    "GeST (imputed)":   "#2A9D8F",
}
HATCH = {"SQUINT (imputed)", "GeST (imputed)"}     # imputation bars hatched


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-prefix", type=str, default="imputation_benchmark")
    p.add_argument("--split", type=str, default="test",
                   help="Pearson split to plot (held-out region = 'test').")
    p.add_argument("--squint-recon-variant", type=str, default=DEFAULT_SQUINT_RECON)
    p.add_argument("--squint-imputed-variant", type=str, default=DEFAULT_SQUINT_IMPUTED)
    p.add_argument("--gest-imputed-variant", type=str, default=DEFAULT_GEST_IMPUTED)
    p.add_argument("--no-recon", action="store_true",
                   help="Drop the SQUINT (recon) ceiling bar.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # method label -> variant dir (order = top-to-bottom intent; panels re-sort
    # by value like the reconstruction figure).
    methods: Dict[str, str] = {}
    if not args.no_recon:
        methods["SQUINT (recon)"] = args.squint_recon_variant
    methods["SQUINT (imputed)"] = args.squint_imputed_variant
    methods["GeST (imputed)"] = args.gest_imputed_variant

    _apply_nature_style()
    # _plot_panel reads HATCH_METHODS from plot_pearson_benchmark's module; make
    # sure our labels hatch there too (idempotent).
    import plot_pearson_benchmark as ppb
    ppb.HATCH_METHODS |= HATCH

    print(f"Artifacts root: {args.artifacts_root}")
    print(f"Dataset tag:    {args.dataset_tag}")
    print(f"Split:          {args.split}")
    for label, variant in methods.items():
        print(f"  {label:<18s} <- {variant}")

    # collect values: per metric-variant -> {label: per-seed array} on the split
    long_rows: List[dict] = []
    per_variant_values: Dict[str, Dict[str, np.ndarray]] = {}
    for suffix, axis, transform, gene_subset, _label in METRIC_VARIANTS:
        vals: Dict[str, np.ndarray] = {}
        for label, variant in methods.items():
            df = load_method_pearson(
                args.artifacts_root / args.dataset_tag / variant,
                branch="cell", axis=axis, transform=transform, gene_subset=gene_subset)
            if df.empty:
                print(f"  [{_label:<22s}] [{label:<16s}] MISSING")
                continue
            v = df[df["split"] == args.split]["value"].astype(float).to_numpy()
            vals[label] = v
            for vv, sd in zip(v, df[df["split"] == args.split]["seed"].astype(int)):
                long_rows.append({
                    "metric_variant": suffix, "axis": axis, "transform": transform,
                    "gene_subset": gene_subset, "branch": "cell", "method": label,
                    "split": args.split, "seed": int(sd), "pearson_mean": float(vv),
                })
        per_variant_values[suffix] = vals

    # --- render: 1 row x N metric-variant panels --------------------------
    variants = METRIC_VARIANTS
    n = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n + 0.6,
                                            max(1.35, 0.30 * len(methods) + 0.6)))
    if n == 1:
        axes = [axes]
    for ax, (suffix, _ax, _tr, _gs, mlabel) in zip(axes, variants):
        vals = per_variant_values.get(suffix, {})
        order = sorted(vals.keys(), key=lambda m: -float(np.nanmean(vals.get(m, [np.nan]))))
        _plot_panel(ax=ax, method_order=order, per_method_values=vals,
                    colour_for=COLOURS, title=mlabel)
    fig.suptitle(f"Spatial imputation — held-out region (expression unseen)  "
                 f"[{args.split}]", fontsize=7, fontweight="medium", y=1.03)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.55)

    base = args.out_dir / f"{args.out_prefix}_{args.split}"
    for ext in ("svg", "png"):
        out = base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)

    if long_rows:
        csv_out = args.out_dir / f"{args.out_prefix}.csv"
        pd.DataFrame(long_rows).to_csv(csv_out, index=False)
        print(f"  -> {csv_out}")


if __name__ == "__main__":
    sys.exit(main())
