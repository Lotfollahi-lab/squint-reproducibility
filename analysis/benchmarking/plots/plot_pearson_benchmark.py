#!/usr/bin/env python3
"""
Pearson reconstruction benchmark figure (Nature style).

Loads per-seed Pearson metrics for the gene-expression-imputation
methods + SQUINT's region-holdout variant, and renders SIX separate
1×2-panel figures — one per (metric-variant × split) combination —
matching the layout of `plot_niche_identification_benchmark.py`.

Metric variants (rows of `METRIC_VARIANTS`):
  - cell_wise_raw          axis=cell_wise, transform=raw,   gene_subset=all
  - gene_wise_raw          axis=gene_wise, transform=raw,   gene_subset=all
  - gene_wise_raw_hvg50    axis=gene_wise, transform=raw,   gene_subset=hvg50

For each metric variant, the script emits two figures (full split
across all cells, and the held-out test cells only):

  pearson_benchmark_cell_wise_raw_full.{svg,png}
  pearson_benchmark_cell_wise_raw_test.{svg,png}
  pearson_benchmark_gene_wise_raw_full.{svg,png}
  pearson_benchmark_gene_wise_raw_test.{svg,png}
  pearson_benchmark_gene_wise_raw_hvg50_full.{svg,png}
  pearson_benchmark_gene_wise_raw_hvg50_test.{svg,png}

Each figure is a 1×2 grid (cell-level | neighborhood-level Pearson),
showing the per-seed mean as a translucent bar, the SEM as an error
bar (when >=2 seeds), and individual seeds as overlaid dots.

Inputs (per method):
  <artifacts_root>/<dataset_tag>/<variant>/<latest_TS>/metrics/
      per_seed_pearson_reconstruction.csv      (5-seed baselines, this benchmark)
   OR
      pearson_reconstruction_metrics.csv       (SQUINT inference, 1 seed)

Outputs (default <out_dir> = <artifacts_root>/benchmarking/figures/):
  pearson_benchmark_<metric>_<split>.{svg,png}    (6 figures)
  pearson_benchmark.csv                            (long-format, all metric variants + splits)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ARTIFACTS_ROOT = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts"
)
DEFAULT_DATASET_TAG = "mmb0-1b_smb1-1b_1p"

# SQUINT region-holdout multi-seed variant DEFAULT. Shared between the
# cell- and niche-level rows below. Overridable on the CLI via
# `--squint-variant`. As of 2026-06-29 the default tracks the NEW SQUINT
# reference = the s57_v19 coupling (cross-batch MNN wt=10/k=1 + cell-cond
# niche FiLM scale-only on the s49_v23 decoupled spine) with the STANDARD
# decoder (dec-w32) and the 25%x25% region-holdout, run multi-seed at
# .../20260628_230209. NOTE: the previous default used a WIDER decoder
# (dec-w128-256) which buys more NB-rate capacity (the thing this benchmark
# measures), so SQUINT's Pearson may read lower here than in the old figure
# — this is the standard-decoder reference, not a wide-decoder one.
DEFAULT_SQUINT_VARIANT = (
    "dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32"
    "+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc"
    "+diversity-w10+filmscale+crossmnn-wt10-k1+region-holdout"
    "+mmb0-1b_smb1-1b_1p__multiseed"
)
# Legacy alias kept so any external import / reference doesn't break.
_SQUINT_VARIANT = DEFAULT_SQUINT_VARIANT

# variant directory name -> display label, per row (branch).
ROWS = {
    "cell": {
        "title": "Cell-level Pearson",
        "branch": "cell",
        "methods": {
            "baseline-scvi+region-holdout":           "scVI",
            "baseline-nichecompass+region-holdout":   "NicheCompass",
            "vanilla-vq-cell+region-holdout":         "Vanilla VQ-VAE",
            _SQUINT_VARIANT:                          "SQUINT",
        },
    },
    "niche": {
        "title": "Neighborhood-level Pearson",
        "branch": "niche",
        "methods": {
            "baseline-nichecompass+region-holdout":   "NicheCompass",
            "baseline-scvi-nbr+region-holdout":       "scVI (X_nbr)",
            "vanilla-vq-nbr+region-holdout":          "Vanilla VQ-VAE",
            _SQUINT_VARIANT:                          "SQUINT",
        },
    },
}

# Palette is aligned with the niche- and cell-type-identification
# figures so the same baseline keeps the same colour across all three
# panels of the paper:
#   - scVI gets the cell-type-ID light blue-gray (#A8DADC). Both the
#     cell-branch and the niche-branch scVI rows share it — same model
#     class, different input.
#   - NicheCompass gets the niche-ID dark brown (#3D2817).
#   - SQUINT keeps its accent magenta (#FF006E).
#   - Vanilla VQ-VAE keeps a neutral grey — it doesn't appear in the
#     other two figures, so there's no shared shade to inherit.
METHOD_COLOURS: Dict[str, str] = {
    "scVI":            "#A8DADC",   # cell-type-ID light blue-gray
    "scVI (X_nbr)":    "#A8DADC",   # same as scVI — same model class
    "NicheCompass":    "#3D2817",   # niche-ID dark brown
    "Vanilla VQ-VAE":  "#888888",   # neutral grey — minimal-model baseline
    "SQUINT":          "#FF006E",   # accent magenta — our method
}

# Splits we render — one figure per entry.
SPLITS = [
    ("all",  "full",    "Full Pearson"),
    ("test", "test",    "Test Pearson (held-out)"),
]

# Pearson metric variants — one (file-suffix, axis, transform,
# gene_subset, human-readable label) per entry. The Pearson CSV emitted
# by `_holdout_utils.build_pearson_dataframe` contains all 6 axis ×
# transform × gene_subset combinations per (branch × split); this list
# selects the slice each output figure pulls. Add / remove entries to
# expand / contract the matrix without touching the loader.
METRIC_VARIANTS = [
    ("cell_wise_raw",       "cell_wise", "raw",   "all",     "Cell-wise raw"),
    ("gene_wise_raw",       "gene_wise", "raw",   "all",     "Gene-wise raw"),
    ("gene_wise_raw_hvg50", "gene_wise", "raw",   "hvg50",   "Gene-wise raw (HVG50)"),
]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _apply_nature_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


# ---------------------------------------------------------------------------
# Per-method metric loaders
# ---------------------------------------------------------------------------

def _find_latest_metrics_dir(variant_dir: Path) -> Optional[Path]:
    if not variant_dir.is_dir():
        return None
    for ts in sorted(variant_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not ts.is_dir():
            continue
        m = ts / "metrics"
        if m.is_dir() and any(
            (m / fn).is_file() for fn in (
                "per_seed_pearson_reconstruction.csv",
                "pearson_reconstruction_metrics.csv",
            )
        ):
            return m
    return None


def load_method_pearson(
        variant_dir: Path,
        branch: str,
        axis: str,
        transform: str,
        gene_subset: str,
    ) -> pd.DataFrame:
    """Return tidy DataFrame: columns = (seed, split, value) for the
    given branch ("cell" / "niche") × (axis, transform, gene_subset)
    slice. `axis` ∈ {"cell_wise", "gene_wise"}, `transform` ∈ {"raw",
    "log1p"}, `gene_subset` ∈ {"all", f"hvg{N}"} — must match the
    values written by `_holdout_utils.build_pearson_dataframe`.

    The Pearson CSV is missing certain combinations by design
    (cell_wise × hvg{N} is skipped — Pearson per cell over only N
    genes is noisy). If the requested slice is absent, returns an
    empty DataFrame and the caller renders an "n/a" tile."""
    m = _find_latest_metrics_dir(variant_dir)
    if m is None:
        return pd.DataFrame()

    per_seed_csv = m / "per_seed_pearson_reconstruction.csv"
    mean_csv = m / "pearson_reconstruction_metrics.csv"

    df = None
    if per_seed_csv.is_file():
        df = pd.read_csv(per_seed_csv)
    elif mean_csv.is_file():
        df = pd.read_csv(mean_csv)
        if "seed" not in df.columns:
            df = df.copy()
            df["seed"] = 0

    if df is None or df.empty:
        return pd.DataFrame()

    df = df[df["branch"] == branch]
    if "axis" in df.columns:
        df = df[df["axis"] == axis]
    if "transform" in df.columns:
        df = df[df["transform"] == transform]
    if "gene_subset" in df.columns:
        df = df[df["gene_subset"] == gene_subset]
    if df.empty:
        return pd.DataFrame()

    return df[["seed", "split", "pearson_mean"]].rename(
        columns={"pearson_mean": "value"}
    )


# ---------------------------------------------------------------------------
# Plotting (matches plot_niche_identification_benchmark layout exactly)
# ---------------------------------------------------------------------------

def _plot_panel(
        ax,
        method_order: List[str],
        per_method_values: Dict[str, np.ndarray],
        colour_for: Dict[str, str],
        title: str,
    ) -> None:
    n = len(method_order)
    # BAR_HEIGHT in axis units: 0.55 leaves 0.45 of whitespace per row.
    # Was 0.40 (60% whitespace) — bumped to make panels feel less sparse.
    BAR_HEIGHT = 0.55
    DOT_JITTER = 0.12

    for j, method in enumerate(method_order):
        vals = np.asarray(per_method_values.get(method, np.array([])), dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            ax.text(0.5, j, "n/a", va="center", ha="center",
                    fontsize=5.5, fontstyle="italic", color="0.5",
                    transform=ax.get_yaxis_transform())
            continue
        mean_val = float(vals.mean())
        colour = colour_for.get(method, "#888888")

        ax.barh(j, mean_val, height=BAR_HEIGHT,
                color=colour, edgecolor=colour,
                linewidth=0.5, alpha=0.35, zorder=2)
        if vals.size > 1:
            sem = float(vals.std(ddof=1) / np.sqrt(vals.size))
            ax.errorbar(mean_val, j, xerr=sem, fmt="none",
                        ecolor="0.3", elinewidth=0.5,
                        capsize=1.2, capthick=0.5, zorder=3)
        rng = np.random.default_rng(42 + j)
        yj = rng.uniform(-DOT_JITTER, DOT_JITTER, size=vals.size)
        ax.scatter(vals, np.full_like(vals, j, dtype=float) + yj,
                   s=12, color=colour, edgecolors="white",
                   linewidths=0.3, zorder=4, alpha=0.92)

    ax.set_yticks(range(n))
    ax.set_yticklabels(method_order, fontweight="medium")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Tighter top/bottom margins (was -0.5 / n-0.5 → 1.0 unit padding total).
    # Pulls the topmost / bottommost bars closer to the panel edges so
    # vertical whitespace shrinks with the figure height below.
    ax.set_ylim(-0.4, n - 0.6)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linewidth=0.2, alpha=0.4, color="0.65", linestyle="--")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_title(f"{title} (↑)", fontsize=7, fontweight="medium", pad=4)


def render_one_split(
        per_method_values: Dict[str, Dict[str, np.ndarray]],
        out_path_base: Path,
        split_title: str,
    ) -> None:
    """Render a single split's figure (cell-level | nbr-level) as
    .svg + .png next to `out_path_base`.

    `per_method_values[row_key]` is a dict[method_label -> per-seed
    np.array] for ROWS["cell"] / ROWS["niche"]."""
    row_keys = list(ROWS.keys())                                     # ["cell", "niche"]
    n_cols = len(row_keys)

    panel_width_in = 1.7
    fig_width_in = panel_width_in * n_cols + 0.6
    # Each panel height depends on its own method count — Pearson rows
    # have ≤4 methods each so the figure is compact.
    # Per-method coefficient lowered (0.34 -> 0.22) and floor lowered
    # (2.0 -> 1.35) to shrink vertical whitespace between methods. For
    # 4 methods this changes the figure height from 2.0 in to 1.38 in.
    max_methods = max(len(per_method_values[rk]) for rk in row_keys)
    fig_height_in = max(1.35, 0.22 * max_methods + 0.5)

    fig, axes = plt.subplots(
        1, n_cols, figsize=(fig_width_in, fig_height_in), sharey=False,
    )
    if n_cols == 1:
        axes = [axes]

    for i, rk in enumerate(row_keys):
        # Sort methods by mean Pearson DESC within this row + split.
        method_means = {
            m: float(np.nanmean(per_method_values[rk].get(m, [np.nan])))
            for m in per_method_values[rk].keys()
        }
        method_order = sorted(method_means, key=lambda m: -method_means[m])

        _plot_panel(
            ax=axes[i],
            method_order=method_order,
            per_method_values=per_method_values[rk],
            colour_for=METHOD_COLOURS,
            title=ROWS[rk]["title"],
        )

    fig.suptitle(split_title, fontsize=7, fontweight="medium", y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.55)

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <artifacts_root>/benchmarking/figures/")
    p.add_argument("--out-prefix", type=str, default="pearson_benchmark",
                   help="Basename for output files. Default: "
                        "pearson_benchmark. Resulting names: "
                        "<prefix>_full.{svg,png}, <prefix>_test.{svg,png}, "
                        "<prefix>.csv.")
    p.add_argument("--squint-variant", type=str, default=DEFAULT_SQUINT_VARIANT,
                   help="Directory name of the SQUINT multi-seed sweep to "
                        "score for BOTH the cell-level and neighborhood-"
                        "level Pearson panels. Must be a `__multiseed` "
                        "sweep so its `metrics/per_seed_pearson_reconstruction"
                        ".csv` is populated. Default tracks the current "
                        "best Pearson variant — see DEFAULT_SQUINT_VARIANT.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"

    # Override the SQUINT variant in both branch rows with the CLI value
    # (or the default if --squint-variant wasn't passed). We rebuild the
    # `methods` dicts so the old _SQUINT_VARIANT key is replaced by
    # `args.squint_variant`. No-op if the user didn't override.
    for row_key in ROWS:
        old_methods = ROWS[row_key]["methods"]
        new_methods: Dict[str, str] = {}
        for variant_dir, label in old_methods.items():
            if label == "SQUINT":
                new_methods[args.squint_variant] = label
            else:
                new_methods[variant_dir] = label
        ROWS[row_key]["methods"] = new_methods

    _apply_nature_style()

    print(f"Artifacts root: {args.artifacts_root}")
    print(f"Dataset tag:    {args.dataset_tag}")
    print(f"Output dir:     {args.out_dir}")

    # Load everything once. Nested dict shape:
    #   loaded[metric_suffix][row_key][split_key][method_label] -> np.array
    loaded: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {
        mv[0]: {rk: {sk: {} for sk, _, _ in SPLITS} for rk in ROWS}
        for mv in METRIC_VARIANTS
    }
    long_rows: List[dict] = []

    for metric_suffix, axis, transform, gene_subset, metric_label in METRIC_VARIANTS:
        print(f"\n=== Metric variant: {metric_label} "
              f"(axis={axis}, transform={transform}, gene_subset={gene_subset}) ===")
        for row_key, row_spec in ROWS.items():
            branch = row_spec["branch"]
            print(f"  --- {row_spec['title']} ({branch}) ---")
            for variant, label in row_spec["methods"].items():
                variant_dir = args.artifacts_root / args.dataset_tag / variant
                df = load_method_pearson(
                    variant_dir,
                    branch=branch,
                    axis=axis,
                    transform=transform,
                    gene_subset=gene_subset,
                )
                if df.empty:
                    print(f"    [{label:<22s}]  MISSING  ({variant_dir})")
                    continue
                n_seeds = df["seed"].nunique()
                splits_present = sorted(df["split"].unique().tolist())
                print(f"    [{label:<22s}]  {n_seeds} seed(s)  splits={splits_present}")
                for split_key, _, _ in SPLITS:
                    vals = (
                        df[df["split"] == split_key]["value"].astype(float).to_numpy()
                    )
                    loaded[metric_suffix][row_key][split_key][label] = vals
                    for v, sd in zip(
                        vals,
                        df[df["split"] == split_key]["seed"].astype(int).to_numpy(),
                    ):
                        long_rows.append({
                            "metric_variant": metric_suffix,
                            "axis":           axis,
                            "transform":      transform,
                            "gene_subset":    gene_subset,
                            "branch":         branch,
                            "method":         label,
                            "split":          split_key,
                            "seed":           int(sd),
                            "pearson_mean":   float(v),
                        })

    # --- Render one figure per (metric_variant × split) ------------------
    print()
    print("Rendering figures:")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for metric_suffix, _axis, _transform, _gene_subset, metric_label in METRIC_VARIANTS:
        for split_key, split_suffix, split_title in SPLITS:
            # Pull just this metric variant's slice for this split.
            per_method_for_split = {
                rk: loaded[metric_suffix][rk][split_key] for rk in ROWS
            }
            base = args.out_dir / (
                f"{args.out_prefix}_{metric_suffix}_{split_suffix}"
            )
            combined_title = f"{split_title} — {metric_label}"
            render_one_split(
                per_method_for_split, base, split_title=combined_title,
            )

    # --- Long-format CSV (all metric variants + splits, for downstream re-use) ---
    if long_rows:
        csv_out = args.out_dir / f"{args.out_prefix}.csv"
        pd.DataFrame(long_rows).to_csv(csv_out, index=False)
        print(f"  -> {csv_out}")


if __name__ == "__main__":
    sys.exit(main())
