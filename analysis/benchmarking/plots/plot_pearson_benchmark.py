#!/usr/bin/env python3
"""
Pearson reconstruction benchmark figure (Nature style).

Loads per-seed Pearson metrics for the gene-expression-imputation
methods + SQUINT's region-holdout variant, and renders TWO separate
1×2-panel figures matching the layout of
`plot_niche_identification_benchmark.py`:

  pearson_benchmark_full.{svg,png}      Full Pearson (all cells)
  pearson_benchmark_test.{svg,png}      Test Pearson (held-out cells only)

Each figure is a 1×2 grid:

    ┌────────────────────────┐  ┌────────────────────────┐
    │  Cell-level Pearson   │  │  Nbr-level Pearson    │
    │  scVI                 │  │  NicheCompass         │
    │  NicheCompass         │  │  scVI (X_nbr)         │
    │  Vanilla VQ-VAE       │  │  Vanilla VQ-VAE       │
    │  SQUINT               │  │  SQUINT               │
    └────────────────────────┘  └────────────────────────┘

Each panel shows the per-seed mean as a translucent bar, the SEM as an
error bar (when >=2 seeds), and individual seeds as overlaid dots.

Inputs (per method):
  <artifacts_root>/<dataset_tag>/<variant>/<latest_TS>/metrics/
      per_seed_pearson_reconstruction.csv      (5-seed baselines, this benchmark)
   OR
      pearson_reconstruction_metrics.csv       (SQUINT inference, 1 seed)

Outputs (default <out_dir> = <artifacts_root>/benchmarking/figures/):
  pearson_benchmark_full.{svg,png}
  pearson_benchmark_test.{svg,png}
  pearson_benchmark.csv                       (long-format, both splits)
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

# variant directory name -> display label, per row (branch).
ROWS = {
    "cell": {
        "title": "Cell-level Pearson",
        "branch": "cell",
        "methods": {
            "baseline-scvi+region-holdout":           "scVI",
            "baseline-nichecompass+region-holdout":   "NicheCompass",
            "vanilla-vq-cell+region-holdout":         "Vanilla VQ-VAE",
            "dualvq+rvq-both+decoder-cov+adv+region-holdout+mmb0-1b_smb1-1b_1p":
                "SQUINT",
        },
    },
    "niche": {
        "title": "Neighborhood-level Pearson",
        "branch": "niche",
        "methods": {
            "baseline-nichecompass+region-holdout":   "NicheCompass",
            "baseline-scvi-nbr+region-holdout":       "scVI (X_nbr)",
            "vanilla-vq-nbr+region-holdout":          "Vanilla VQ-VAE",
            "dualvq+rvq-both+decoder-cov+adv+region-holdout+mmb0-1b_smb1-1b_1p":
                "SQUINT",
        },
    },
}

# Distinct, colour-blind-friendly palette. SQUINT keeps its accent
# magenta (consistent with the niche- and cell-type-identification
# figures). scVI in both branches shares teal — same model class,
# different input.
METHOD_COLOURS: Dict[str, str] = {
    "scVI":            "#06D6A0",
    "scVI (X_nbr)":    "#06D6A0",
    "NicheCompass":    "#118AB2",
    "Vanilla VQ-VAE":  "#888888",   # neutral grey — minimal-model baseline
    "SQUINT":          "#FF006E",   # accent magenta — our method
}

# Splits we render — one figure per entry.
SPLITS = [
    ("all",  "full",    "Full Pearson"),
    ("test", "test",    "Test Pearson (held-out)"),
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
    ) -> pd.DataFrame:
    """Return tidy DataFrame: columns = (seed, split, value) for the
    given branch ("cell" / "niche"). Picks the canonical Pearson row:
    cell_wise × log1p × all-genes (matches SQUINT's convention)."""
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

    # Filter to branch + canonical (cell_wise × log1p × all-genes).
    df = df[df["branch"] == branch]
    if "axis" in df.columns:
        df = df[df["axis"] == "cell_wise"]
    if "transform" in df.columns:
        df = df[df["transform"] == "log1p"]
    if "gene_subset" in df.columns:
        df = df[df["gene_subset"] == "all"]
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
    BAR_HEIGHT = 0.4
    DOT_JITTER = 0.10

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
    ax.set_ylim(-0.5, n - 0.5)
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
    max_methods = max(len(per_method_values[rk]) for rk in row_keys)
    fig_height_in = max(2.0, 0.34 * max_methods + 0.6)

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
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"

    _apply_nature_style()

    print(f"Artifacts root: {args.artifacts_root}")
    print(f"Dataset tag:    {args.dataset_tag}")
    print(f"Output dir:     {args.out_dir}")

    # Load everything once. `loaded[row_key][split_key][method_label] -> np.array`.
    loaded: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
        rk: {sk: {} for sk, _, _ in SPLITS} for rk in ROWS
    }
    long_rows: List[dict] = []

    for row_key, row_spec in ROWS.items():
        branch = row_spec["branch"]
        print(f"\n--- {row_spec['title']} ({branch}) ---")
        for variant, label in row_spec["methods"].items():
            variant_dir = args.artifacts_root / args.dataset_tag / variant
            df = load_method_pearson(variant_dir, branch=branch)
            if df.empty:
                print(f"  [{label:<22s}]  MISSING  ({variant_dir})")
                continue
            n_seeds = df["seed"].nunique()
            splits_present = sorted(df["split"].unique().tolist())
            print(f"  [{label:<22s}]  {n_seeds} seed(s)  splits={splits_present}")
            for split_key, _, _ in SPLITS:
                vals = (
                    df[df["split"] == split_key]["value"].astype(float).to_numpy()
                )
                loaded[row_key][split_key][label] = vals
                for v, sd in zip(
                    vals,
                    df[df["split"] == split_key]["seed"].astype(int).to_numpy(),
                ):
                    long_rows.append({
                        "branch": branch, "method": label,
                        "split": split_key, "seed": int(sd),
                        "pearson_mean": float(v),
                    })

    # --- Render one figure per split -----------------------------------
    print()
    print("Rendering figures:")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_key, split_suffix, split_title in SPLITS:
        # Pull just this split's slice from the loaded dict.
        per_method_for_split = {
            rk: loaded[rk][split_key] for rk in ROWS
        }
        base = args.out_dir / f"{args.out_prefix}_{split_suffix}"
        render_one_split(per_method_for_split, base, split_title=split_title)

    # --- Long-format CSV (both splits, for downstream re-use) ---------
    if long_rows:
        csv_out = args.out_dir / f"{args.out_prefix}.csv"
        pd.DataFrame(long_rows).to_csv(csv_out, index=False)
        print(f"  -> {csv_out}")


if __name__ == "__main__":
    sys.exit(main())
