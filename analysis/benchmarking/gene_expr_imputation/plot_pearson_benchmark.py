"""
Pearson reconstruction benchmark figure (Nature style).

Loads per-seed Pearson metrics for the gene-expression-imputation
methods + SQUINT's region-holdout variant, and renders a 2×2 panel
figure:

                  Full Pearson      Test Pearson (held-out)
              ┌───────────────────┐ ┌───────────────────┐
  Cell-level  │ scVI              │ │ scVI              │
              │ NicheCompass      │ │ NicheCompass      │
              │ Vanilla VQ-VAE    │ │ Vanilla VQ-VAE    │
              │ SQUINT            │ │ SQUINT            │
              └───────────────────┘ └───────────────────┘
              ┌───────────────────┐ ┌───────────────────┐
  Nbr-level   │ NicheCompass      │ │ NicheCompass      │
              │ CellCharter+dec.  │ │ CellCharter+dec.  │
              │ Vanilla VQ-VAE    │ │ Vanilla VQ-VAE    │
              │ SQUINT            │ │ SQUINT            │
              └───────────────────┘ └───────────────────┘

Each cell shows the per-seed-mean as a translucent bar, SEM as an
error bar, and individual seeds as overlaid dots — same style as the
niche- and cell-type-identification benchmark figures.

Inputs (per method):
  <artifacts_root>/<dataset_tag>/<variant>/<latest_TS>/metrics/
      per_seed_pearson_reconstruction.csv      (this benchmark — new)
   OR
      pearson_reconstruction_metrics.csv       (SQUINT inference, 1 seed)

Outputs:
  <out_dir>/pearson_benchmark.{svg,png,csv}
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

# Per-row method registry: {row_label: {variant_dir: display_label}}.
# `branch` keys ("cell" / "niche") tell the loader which Pearson rows to
# pick out of the per-seed CSVs. The ordering inside each row determines
# top-to-bottom plot order *before* sort-by-mean kicks in.
ROWS = {
    "cell": {
        "title": "Cell-level Pearson",
        "branch": "cell",
        "methods": {
            "baseline-scvi+region-holdout":             "scVI",
            "baseline-nichecompass+region-holdout":     "NicheCompass",
            "vanilla-vq-cell+region-holdout":           "Vanilla VQ-VAE",
            "dualvq+rvq-both+decoder-cov+adv+region-holdout+mmb0-1b_smb1-1b_1p":
                "SQUINT",
        },
    },
    "niche": {
        "title": "Neighborhood-level Pearson",
        "branch": "niche",
        "methods": {
            "baseline-nichecompass+region-holdout":     "NicheCompass",
            "baseline-scvi-nbr+region-holdout":         "scVI (X_nbr)",
            "vanilla-vq-nbr+region-holdout":            "Vanilla VQ-VAE",
            "dualvq+rvq-both+decoder-cov+adv+region-holdout+mmb0-1b_smb1-1b_1p":
                "SQUINT",
        },
    },
}

# Methods come from up to two rows; pick consistent colours. The
# cell-level scVI ("scVI") and the nbr-level scVI-on-X_nbr ("scVI (X_nbr)")
# share the same teal — they're the same model class, just different
# input/target.
METHOD_COLOURS: Dict[str, str] = {
    "scVI":            "#06D6A0",
    "scVI (X_nbr)":    "#06D6A0",
    "NicheCompass":    "#118AB2",
    "Vanilla VQ-VAE":  "#888888",   # neutral grey — "minimal model"
    "SQUINT":          "#FF006E",
}

# Splits to plot — one column per split.
SPLITS = [
    ("all",  "Full Pearson"),
    ("test", "Test Pearson (held-out)"),
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
# Loaders
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
    given branch ("cell" / "niche"). Falls back to single-seed mean
    CSV when per-seed isn't available (SQUINT case)."""
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

    # Filter to branch + cell-wise / log1p / all-genes (the panels-of-record).
    df = df[df["branch"] == branch]
    if "axis" in df.columns:
        df = df[df["axis"] == "cell_wise"]
    if "transform" in df.columns:
        df = df[df["transform"] == "log1p"]
    if "gene_subset" in df.columns:
        df = df[df["gene_subset"] == "all"]
    if df.empty:
        return pd.DataFrame()

    # The rows we want carry one value per (seed, split). pearson_mean is
    # the per-cell-mean Pearson; we use it as "the" score for the bar.
    out = df[["seed", "split", "pearson_mean"]].rename(
        columns={"pearson_mean": "value"}
    )
    return out


# ---------------------------------------------------------------------------
# Plotting (re-uses the layout from plot_niche_identification_benchmark)
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


def make_figure(
        per_method_per_split: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        out_path_base: Path,
    ) -> None:
    """`per_method_per_split[row_key][split][method]` -> 1D np.array of seeds."""
    if not per_method_per_split:
        raise SystemExit("No data to plot.")

    rows = list(per_method_per_split.keys())   # ["cell", "niche"]
    n_rows = len(rows)
    n_cols = len(SPLITS)

    panel_width_in = 1.7
    fig_width_in = panel_width_in * n_cols + 0.6
    # Each row's height grows with its method count.
    method_count_per_row = {
        rk: max(len(per_method_per_split[rk][SPLITS[0][0]]), 1)
        for rk in rows
    }
    fig_height_in = sum(0.34 * method_count_per_row[rk] + 0.6 for rk in rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width_in, fig_height_in), sharey=False,
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r, row_key in enumerate(rows):
        # Sort methods by mean of "all"-split Pearson, descending — same
        # convention as the existing benchmark figures. SQUINT keeps its
        # accent colour wherever it lands.
        method_means = {
            m: float(np.nanmean(per_method_per_split[row_key]["all"].get(m, [np.nan])))
            for m in per_method_per_split[row_key]["all"].keys()
        }
        method_order = sorted(method_means, key=lambda m: -method_means[m])

        for c, (split_key, split_label) in enumerate(SPLITS):
            ax = axes[r, c]
            per_method = per_method_per_split[row_key][split_key]
            row_title = ROWS[row_key]["title"]
            col_label = split_label
            _plot_panel(
                ax=ax,
                method_order=method_order,
                per_method_values=per_method,
                colour_for=METHOD_COLOURS,
                title=f"{row_title} · {col_label}",
            )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.55, hspace=0.55)

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
    p.add_argument("--out-name", type=str, default="pearson_benchmark")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"

    _apply_nature_style()

    print(f"Artifacts root: {args.artifacts_root}")
    print(f"Dataset tag:    {args.dataset_tag}")
    print(f"Output dir:     {args.out_dir}")
    print()

    # Build the nested dict structure: per_method_per_split[row][split][method] -> values
    per_method_per_split: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    long_rows: List[dict] = []   # for the saved long-format CSV
    for row_key, row_spec in ROWS.items():
        branch = row_spec["branch"]
        per_method_per_split[row_key] = {sk: {} for sk, _ in SPLITS}

        print(f"--- {row_spec['title']} ({branch}) ---")
        for variant, label in row_spec["methods"].items():
            variant_dir = args.artifacts_root / args.dataset_tag / variant
            df = load_method_pearson(variant_dir, branch=branch)
            if df.empty:
                print(f"  [{label:<22s}]  MISSING  ({variant_dir})")
                continue
            n_seeds = df["seed"].nunique()
            splits_present = sorted(df["split"].unique().tolist())
            print(f"  [{label:<22s}]  {n_seeds} seed(s)  splits={splits_present}")

            for split_key, _ in SPLITS:
                vals = df[df["split"] == split_key]["value"].astype(float).to_numpy()
                per_method_per_split[row_key][split_key][label] = vals
                for v, sd in zip(
                    vals,
                    df[df["split"] == split_key]["seed"].astype(int).to_numpy(),
                ):
                    long_rows.append({
                        "branch": branch, "method": label, "split": split_key,
                        "seed": int(sd), "pearson_mean": float(v),
                    })

    # --- Render figure --------------------------------------------------
    print()
    print("Rendering figure:")
    base = args.out_dir / args.out_name
    make_figure(per_method_per_split, base)

    # Long-format CSV next to the figure for downstream re-use.
    if long_rows:
        long_df = pd.DataFrame(long_rows)
        csv_out = base.with_suffix(".csv")
        long_df.to_csv(csv_out, index=False)
        print(f"  -> {csv_out}")


if __name__ == "__main__":
    sys.exit(main())
