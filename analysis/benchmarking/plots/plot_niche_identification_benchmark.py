#!/usr/bin/env python3
"""
Niche-identification benchmark figure (Nature style).

Loads per-seed metrics for a configurable set of methods (niche-
identification baselines + one or more SQUINT variants), and renders
one figure with FOUR horizontal-bar panels:

  Niche NMI   (↑ better)
  Niche ARI   (↑ better)
  iLISI       (↑ better)
  MMD         (↓ better)

Each panel shows the mean across seeds as a translucent bar, the SEM as
an error bar, and individual seeds overlaid as dots (matching the
reference notebook style).

Inputs (per method):
  <artifacts_root>/<dataset_tag>/<variant>/<latest_TS>/metrics/
      per_seed_niche_identification.csv      (5-seed baselines)
      per_seed_batch_integration.csv         (5-seed baselines)
   OR
      niche_identification_metrics.csv       (SQUINT inference, 1-seed)
      batch_integration_metrics.csv          (SQUINT inference, 1-seed)

The script auto-detects which CSV layout is present and unifies them.

Outputs:
  <out_dir>/niche_identification_benchmark.{svg,png}
  <out_dir>/niche_identification_benchmark_data.csv

Default <out_dir> is `<artifacts_root>/benchmarking/figures/`.

Run from any directory:
  python analysis/benchmarking/plots/plot_niche_identification_benchmark.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Defaults: paths + the methods to include in the FIRST figure
# ---------------------------------------------------------------------------

DEFAULT_ARTIFACTS_ROOT = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts"
)
DEFAULT_DATASET_TAG = "mmb0-1b_smb1-1b_1p"

# variant directory name -> display label
# Display order in the plot is determined later by mean Niche NMI; SQUINT is
# always plotted with a distinct accent colour regardless of position.
DEFAULT_METHODS: Dict[str, str] = {
    "baseline-banksy":          "BANKSY",
    "baseline-cellcharter":     "CellCharter",
    "baseline-graphst":         "GraphST",
    "baseline-novae":           "Novae",
    "baseline-nichecompass":    "NicheCompass",
    "baseline-neigh-expr-pca":  "Neighbour expr. PCA",
    "dualvq+wide+rvq-both+decoder-cov+adv-warmup10+mmb0-1b_smb1-1b_1p":
        "SQUINT",
}

# Distinct, colour-blind-friendly palette. SQUINT is the accent (magenta)
# matching the reference Nature-style notebook.
METHOD_COLOURS: Dict[str, str] = {
    "BANKSY":              "#3A86FF",
    "CellCharter":         "#06D6A0",
    "GraphST":             "#073B4C",
    "Novae":               "#8338EC",
    "NicheCompass":        "#118AB2",
    "Neighbour expr. PCA": "#FFD166",
    "SQUINT":              "#FF006E",
}

# Metric panels (in the requested order) and direction-of-merit annotation.
METRICS = [
    ("Niche NMI", "↑"),
    ("Niche ARI", "↑"),
    ("iLISI",     "↑"),
    ("MMD",       "↓"),
]


# ---------------------------------------------------------------------------
# Nature style
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
    """Return the most recent `<TS>/metrics` dir under `variant_dir`,
    or None if `variant_dir` doesn't exist or has no completed runs."""
    if not variant_dir.is_dir():
        return None
    ts_dirs = sorted(
        (p for p in variant_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for ts in ts_dirs:
        m = ts / "metrics"
        if m.is_dir():
            # Require at least one of the metric CSVs to call this a "completed run".
            # (Avoids picking up an in-progress run dir whose `metrics/` exists but is empty.)
            if any(
                (m / fn).is_file() for fn in (
                    "per_seed_niche_identification.csv",
                    "niche_identification_metrics.csv",
                    "per_seed_batch_integration.csv",
                    "batch_integration_metrics.csv",
                )
            ):
                return m
    return None


def _pick_niche_code_key(code_keys: List[str]) -> Optional[str]:
    """Out of the available `code_key` values in a niche metrics CSV,
    pick the one that corresponds to the niche-side codes.

    Baselines write a single `code_key="leiden"`. SQUINT writes
    `code_key` values like `cell_code_index`, `neighborhood_code_index`,
    `cell_code_indices[level_0]`, `neighborhood_code_indices[composite]`,
    etc. We prefer the niche-side composite (full-resolution) over
    level_0; fall back to the single-level niche key; and finally to
    `leiden` for baselines.
    """
    # 1. Niche composite (RVQ-style multi-level, full-resolution leaf)
    for ck in code_keys:
        if ck.startswith("neighborhood_code_indices") and "composite" in ck:
            return ck
    # 2. Niche single-level
    for ck in code_keys:
        if ck == "neighborhood_code_index":
            return ck
    # 3. Niche level_0 (RVQ macro cluster) - last resort niche-side
    for ck in code_keys:
        if ck.startswith("neighborhood_code_indices") and "level_0" in ck:
            return ck
    # 4. Baseline default
    if "leiden" in code_keys:
        return "leiden"
    # 5. Fall back to anything (e.g. a future custom code_key)
    return code_keys[0] if code_keys else None


def _pick_batch_emb_key(emb_keys: List[str]) -> Optional[str]:
    """Out of the available `emb_key` values in a batch integration
    CSV, pick the one to score for batch integration.

    Baselines write a single per-method emb_key (e.g. "novae_latent_corrected").
    SQUINT writes multiple (cell_emb, neighborhood_emb, cell_latent,
    neighborhood_latent, X_squint, X_squint_quantized, ...). For the
    NICHE benchmark we pick the niche-side embedding, preferring the
    adversarially-corrected variant when available.
    """
    if not emb_keys:
        return None
    # 1. Niche corrected (post-adversarial)
    for ek in emb_keys:
        if "neighborhood" in ek and "corrected" in ek:
            return ek
    # 2. Niche raw
    for ek in emb_keys:
        if "neighborhood" in ek:
            return ek
    # 3. Single emb_key (baselines): just take it
    if len(emb_keys) == 1:
        return emb_keys[0]
    # 4. Fall back to first
    return emb_keys[0]


def load_method_metrics(variant_dir: Path, method_label: str) -> pd.DataFrame:
    """Return tidy DataFrame: columns = (method, seed, metric, value).

    For methods with `per_seed_*.csv` (baselines), one row per seed per metric.
    For methods with only `*_metrics.csv` (SQUINT 1-seed inference), one
    row per metric with seed=0 (synthetic — represents the single inference).
    Returns an empty DataFrame if no metrics are found.
    """
    m = _find_latest_metrics_dir(variant_dir)
    if m is None:
        return pd.DataFrame()

    rows: List[dict] = []

    # ---- Niche identification (NMI / ARI) -------------------------------
    per_seed_niche = m / "per_seed_niche_identification.csv"
    mean_niche = m / "niche_identification_metrics.csv"

    df_n: Optional[pd.DataFrame] = None
    if per_seed_niche.is_file():
        df_n = pd.read_csv(per_seed_niche)
    elif mean_niche.is_file():
        df_n = pd.read_csv(mean_niche)
        if "seed" not in df_n.columns:
            df_n = df_n.copy()
            df_n["seed"] = 0

    if df_n is not None and not df_n.empty:
        if "split" in df_n.columns:
            df_n = df_n[df_n["split"] == "all"]
        df_n = df_n[df_n["label_key"] == "niche"]
        if not df_n.empty and "code_key" in df_n.columns:
            ck = _pick_niche_code_key(sorted(df_n["code_key"].unique().tolist()))
            if ck is not None:
                df_n = df_n[df_n["code_key"] == ck]
        for _, r in df_n.iterrows():
            rows.append({
                "method": method_label,
                "seed":   int(r.get("seed", 0)),
                "metric": "Niche NMI",
                "value":  float(r["NMI"]),
            })
            rows.append({
                "method": method_label,
                "seed":   int(r.get("seed", 0)),
                "metric": "Niche ARI",
                "value":  float(r["ARI"]),
            })

    # ---- Batch integration (iLISI / MMD) --------------------------------
    per_seed_batch = m / "per_seed_batch_integration.csv"
    mean_batch = m / "batch_integration_metrics.csv"

    df_b: Optional[pd.DataFrame] = None
    if per_seed_batch.is_file():
        df_b = pd.read_csv(per_seed_batch)
    elif mean_batch.is_file():
        df_b = pd.read_csv(mean_batch)
        if "seed" not in df_b.columns:
            df_b = df_b.copy()
            df_b["seed"] = 0

    if df_b is not None and not df_b.empty:
        if "emb_key" in df_b.columns:
            ek = _pick_batch_emb_key(sorted(df_b["emb_key"].unique().tolist()))
            if ek is not None:
                df_b = df_b[df_b["emb_key"] == ek]
        for _, r in df_b.iterrows():
            metric = r.get("metric")
            if metric not in ("iLISI", "MMD"):
                continue
            rows.append({
                "method": method_label,
                "seed":   int(r.get("seed", 0)),
                "metric": str(metric),
                "value":  float(r["score"]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel(
        ax,
        method_order: List[str],
        per_method_values: Dict[str, np.ndarray],
        colour_for: Dict[str, str],
        metric_label: str,
        show_y_ticklabels: bool,
        x_invert_better: bool,
    ) -> None:
    n = len(method_order)
    # Bar height: thinner than the canonical 0.7 default so the dots and
    # the underlying mean read as separate visual elements (the bar
    # becomes a "swatch + lollipop" rather than a chunky block).
    BAR_HEIGHT = 0.4
    DOT_JITTER = 0.10  # < BAR_HEIGHT/2 so dots stay within the bar band

    for j, method in enumerate(method_order):
        vals = per_method_values.get(method, np.array([]))
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            ax.text(0.5, j, "n/a", va="center", ha="center",
                    fontsize=5.5, fontstyle="italic", color="0.5",
                    transform=ax.get_yaxis_transform())
            continue
        mean_val = float(vals.mean())
        colour = colour_for.get(method, "#888888")

        # Bar (mean) — thin
        ax.barh(j, mean_val, height=BAR_HEIGHT,
                color=colour, edgecolor=colour,
                linewidth=0.5, alpha=0.35, zorder=2)
        # SEM error bar (only when >=2 seeds)
        if vals.size > 1:
            sem = float(vals.std(ddof=1) / np.sqrt(vals.size))
            ax.errorbar(mean_val, j, xerr=sem, fmt="none",
                        ecolor="0.3", elinewidth=0.5,
                        capsize=1.2, capthick=0.5, zorder=3)
        # Individual seeds — light vertical jitter so overlapping points are visible.
        rng = np.random.default_rng(42 + j)
        yj = rng.uniform(-DOT_JITTER, DOT_JITTER, size=vals.size)
        ax.scatter(vals, np.full_like(vals, j, dtype=float) + yj,
                   s=12, color=colour, edgecolors="white",
                   linewidths=0.3, zorder=4, alpha=0.92)

    ax.set_yticks(range(n))
    # Always show the method names on every panel — easier to scan
    # individual rows without relying on a shared first-panel column.
    ax.set_yticklabels(method_order, fontweight="medium")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()  # top-of-list = top of axis

    ax.xaxis.grid(True, linewidth=0.2, alpha=0.4, color="0.65", linestyle="--")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))

    arrow = "↓" if x_invert_better else "↑"
    ax.set_title(f"{metric_label} ({arrow})",
                 fontsize=7, fontweight="medium", pad=4)


def make_figure(
        df: pd.DataFrame,
        out_path_base: Path,
        method_colours: Dict[str, str] = METHOD_COLOURS,
    ) -> None:
    """Build the 1×4 panel figure and save .svg / .png next to `out_path_base`."""
    if df.empty:
        raise SystemExit("No metrics loaded; refusing to plot empty figure.")

    pivot = df.pivot_table(
        index=["method", "seed"], columns="metric", values="value",
    )

    # --- Method order: sort by mean Niche NMI descending. -----------------
    if "Niche NMI" in pivot.columns:
        nmi_means = pivot["Niche NMI"].groupby(level=0).mean()
        method_order = nmi_means.sort_values(ascending=False).index.tolist()
    else:
        method_order = sorted(pivot.index.get_level_values(0).unique())

    # Build {method: per-method value array} for each metric.
    per_metric: Dict[str, Dict[str, np.ndarray]] = {}
    for m_label, _ in METRICS:
        per_metric[m_label] = {}
        if m_label not in pivot.columns:
            continue
        for method in method_order:
            try:
                v = pivot.loc[method][m_label].to_numpy()
            except KeyError:
                v = np.array([])
            per_metric[m_label][method] = v

    # --- Figure size: tuned to the Nature notebook's per-panel width. -----
    # Each panel now carries its own y-tick labels (method names), so the
    # per-panel allocated width has to include label space, and we bump
    # wspace so labels of one panel don't bleed into the bars of the next.
    panel_width_in = 1.7
    fig_width_in = panel_width_in * len(METRICS) + 0.6
    fig_height_in = max(2.0, 0.34 * len(method_order) + 0.6)

    fig, axes = plt.subplots(
        1, len(METRICS),
        figsize=(fig_width_in, fig_height_in),
        sharey=False,
    )
    if len(METRICS) == 1:
        axes = [axes]

    for i, (m_label, arrow) in enumerate(METRICS):
        _plot_panel(
            ax=axes[i],
            method_order=method_order,
            per_method_values=per_metric[m_label],
            colour_for=method_colours,
            metric_label=m_label,
            show_y_ticklabels=True,
            x_invert_better=(arrow == "↓"),
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.55)

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)

    # Also persist the tidy DataFrame for downstream re-use.
    csv_out = out_path_base.with_suffix(".csv")
    pivot.to_csv(csv_out)
    print(f"  -> {csv_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: "
                        "<artifacts_root>/benchmarking/figures/).")
    p.add_argument("--out-name", type=str,
                   default="niche_identification_benchmark",
                   help="Output basename (no extension). Default: "
                        "'niche_identification_benchmark'.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"

    _apply_nature_style()

    print(f"Artifacts root: {args.artifacts_root}")
    print(f"Dataset tag:    {args.dataset_tag}")
    print(f"Output dir:     {args.out_dir}")
    print()
    print("Loading metrics:")

    frames: List[pd.DataFrame] = []
    for variant, label in DEFAULT_METHODS.items():
        variant_dir = args.artifacts_root / args.dataset_tag / variant
        df = load_method_metrics(variant_dir, label)
        n_seeds = df["seed"].nunique() if not df.empty else 0
        n_metrics = df["metric"].nunique() if not df.empty else 0
        status = (f"{n_seeds} seed(s), {n_metrics} metric(s)"
                  if not df.empty else "MISSING")
        print(f"  [{label:<22s}]  {status:<25s}  ({variant_dir})")
        if not df.empty:
            frames.append(df)

    if not frames:
        raise SystemExit(
            "\nNo metric CSVs found for any configured method. Check "
            "that --artifacts-root + --dataset-tag point to a directory "
            "with completed runs."
        )

    all_df = pd.concat(frames, ignore_index=True)

    print()
    print("Rendering figure:")
    base = args.out_dir / args.out_name
    make_figure(all_df, base)


if __name__ == "__main__":
    sys.exit(main())
