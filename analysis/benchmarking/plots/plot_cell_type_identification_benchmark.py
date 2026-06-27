#!/usr/bin/env python3
"""
Cell-type-identification benchmark figure (Nature style).

Loads per-seed metrics for cell-type-identification baselines + a
SQUINT multi-seed sweep (scored against its CELL-side codes), and
renders one figure with FOUR horizontal-bar panels:

  Cell-type NMI   (↑ better)
  Cell-type ARI   (↑ better)
  iLISI           (↑ better)
  MMD             (↓ better)

Each panel shows the mean across seeds as a translucent bar, the SEM as
an error bar, and individual seeds overlaid as dots (matching the
reference notebook style and the niche-identification figure).

Inputs (per method):
  <artifacts_root>/<dataset_tag>/<variant>/<latest_TS>/metrics/
      per_seed_niche_identification.csv
      per_seed_batch_integration.csv

`<variant>` for baselines is the `baseline-<name>` directory; for
SQUINT it's the multi-seed sweep directory (suffix `__multiseed`).
Both produce the SAME per_seed_*.csv layout — the legacy single-seed
SQUINT path (`niche_identification_metrics.csv` etc.) is no longer
supported and the loader refuses to fall back to it.

`<latest_TS>` is auto-selected per variant: the most recent timestamp
subdirectory whose `metrics/` contains at least one of the per-seed
CSVs is used. Older / in-progress runs in the same variant dir are
ignored.

Cell-type-identification baselines (PCA+Leiden, Harmony, scVI,
Geneformer, Nicheformer, scGPT, scGPT-spatial, UCE) write per-seed
CSVs with `code_key="leiden"` and `label_key="cell_type"`. SQUINT
multi-seed writes per-seed CSVs with multiple code_keys / emb_keys; for
THIS figure we pick the CELL-side codes (`cell_code_index` /
`cell_code_indices[composite]`) and the cell-side embedding
(`cell_emb_corrected` / `cell_emb`).

Outputs:
  <out_dir>/cell_type_identification_benchmark.{svg,png,csv}

Default <out_dir> is `<artifacts_root>/benchmarking/figures/`.

Usage:
  # Default SQUINT variant (winner across the v8/v9 sweeps):
  python analysis/benchmarking/plots/plot_cell_type_identification_benchmark.py

  # Compare a different SQUINT multi-seed sweep:
  python analysis/benchmarking/plots/plot_cell_type_identification_benchmark.py \\
      --squint-variant \\
      dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+mmb0-1b_smb1-1b_1p__multiseed

  # Different dataset:
  python analysis/benchmarking/plots/plot_cell_type_identification_benchmark.py \\
      --dataset-tag chl59-8b_1p
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

# Baseline variant directories -> display labels.
# Display order in the plot is determined later by mean Cell-type NMI;
# SQUINT is always plotted with a distinct accent colour regardless of
# position. PCA+Leiden and Harmony are intentionally OMITTED — for
# cell-type identification on spatial data the comparison of interest is
# generative / foundation models + SQUINT, not the two linear baselines
# (which historically over-cluster at this granularity). Add them back
# here if you want them in the plot again.
DEFAULT_BASELINES: Dict[str, str] = {
    "baseline-scvi":           "scVI",
    "baseline-geneformer":     "Geneformer",
    "baseline-nicheformer":    "Nicheformer",
    "baseline-scgpt":          "scGPT",
    "baseline-scgpt-spatial":  "scGPT-spatial",
    "baseline-uce":            "UCE",
}

# Default SQUINT variant. Always a `__multiseed` sweep directory
# (produced by `examples/submit_multi_seed.sh`) — its
# `metrics/per_seed_*.csv` files have the same schema as the baseline
# per_seed_*.csv files, so the loader treats them uniformly.
# Override on the CLI via `--squint-variant`.
DEFAULT_SQUINT_VARIANT = (
    # s49_v23: decoupled-enc + diversity wt=10 + within-batch contrastive wt=10
    # on the s48_v2 spine (cell-w=1, no-batch-int, enc-deeper, within-sec).
    # Replaces the previous default (s42 winner, cell-w=5.0) as of the
    # s49 sweep results — see project_squint.md.
    "s49_v23_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p__multiseed"
)
SQUINT_LABEL = "SQUINT"

# Colour family: SHADES OF BLUE for all cell-type-ID baselines,
# SQUINT keeps its accent red. The niche-ID plot uses SHADES OF
# BROWN for its baselines (see plot_niche_identification_benchmark.py),
# so the two metric plots are immediately distinguishable as
# "cell-type-ID = blue family" vs "niche-ID = brown family", with
# SQUINT as the consistent red accent across both. Shades are
# arranged light -> dark and chosen for clear pair-wise contrast
# within the family.
METHOD_COLOURS: Dict[str, str] = {
    "scVI":           "#A8DADC",   # very light blue-gray  — classical generative
    "Geneformer":     "#48CAE4",   # light cyan-blue        — foundation model
    "Nicheformer":    "#0096C7",   # mid cyan               — foundation model
    "scGPT":          "#0077B6",   # mid blue               — foundation model
    "scGPT-spatial":  "#023E8A",   # dark blue              — foundation model (spatial-trained)
    "UCE":            "#03045E",   # very dark navy         — foundation model
    "SQUINT":         "#FF006E",   # accent red/magenta     — our method (unchanged)
}

# Metric panels (in the requested order) and direction-of-merit annotation.
METRICS = [
    ("Cell-type NMI", "↑"),
    ("Cell-type ARI", "↑"),
    ("iLISI",         "↑"),
    ("MMD",           "↓"),
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
    """Return the most recent `<TS>/metrics` dir under `variant_dir`
    that contains at least one of the per-seed metric CSVs.

    Timestamps are sorted lexicographically — works because every
    upstream writer uses `YYYYMMDD_HHMMSS` which sorts identically to
    calendar order. An in-progress run with an empty `metrics/` dir is
    skipped (the per_seed_*.csv files aren't written until the runner /
    aggregator's final step).

    Returns None if `variant_dir` is missing or no timestamp under it
    has a populated metrics dir.

    Only `per_seed_*.csv` files count as "completed". The legacy
    single-seed SQUINT layout (mean-only `niche_identification_metrics.csv`)
    is no longer supported: SQUINT now runs through the multi-seed
    pipeline and writes per_seed_*.csv just like the baselines.
    """
    if not variant_dir.is_dir():
        return None
    ts_dirs = sorted(
        (p for p in variant_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for ts in ts_dirs:
        m = ts / "metrics"
        if m.is_dir() and any(
            (m / fn).is_file() for fn in (
                "per_seed_niche_identification.csv",
                "per_seed_batch_integration.csv",
            )
        ):
            return m
    return None


def _pick_cell_code_key(code_keys: List[str]) -> Optional[str]:
    """Out of the available `code_key` values in a niche-metrics CSV,
    pick the one that corresponds to the CELL-side codes.

    Baselines write a single `code_key="leiden"`. SQUINT writes
    `code_key` values like `cell_code_index`, `cell_code_indices[level_0]`,
    `cell_code_indices[composite]`, plus their niche-side counterparts.

    Preference order (cell-side):
      1. `cell_code_indices[level_0]` — the K1=30 macro RVQ partition,
         which is the HEADLINE cell-type-NMI metric across the rest of
         the tooling (compare_variants.py reports this column as
         `niche|all|cell_code_indices[level_0]|cell_type|NMI`). Using
         it here keeps this figure consistent with the per-variant
         comparison heatmaps.
      2. `cell_code_index` — non-residual VQ single-level fallback.
      3. `cell_code_indices[composite]` — K1*K2 leaf clusters
         (factorised RVQ levels). Last cell-side fallback.
      4. `leiden` — baseline default.

    Notes
    -----
    Previously this picker preferred `[composite]` over `[level_0]`,
    which silently reported a DIFFERENT cell-NMI than the rest of the
    analysis tooling (composite has up to 2700 leaves on a 2-level
    RVQ vs level_0's 30 macro codes — generally produces a higher
    NMI). Switched on user request so the figure matches the
    `compare_variants.py` headline.
    """
    # 1. Cell level_0 (RVQ macro cluster) — HEADLINE metric across tooling.
    for ck in code_keys:
        if ck.startswith("cell_code_indices") and "level_0" in ck:
            return ck
    # 2. Cell single-level (non-residual VQ fallback)
    for ck in code_keys:
        if ck == "cell_code_index":
            return ck
    # 3. Cell composite (multi-level RVQ leaf cluster) — last resort cell-side
    for ck in code_keys:
        if ck.startswith("cell_code_indices") and "composite" in ck:
            return ck
    # 4. Baseline default
    if "leiden" in code_keys:
        return "leiden"
    # 5. Catch-all
    return code_keys[0] if code_keys else None


def _pick_batch_emb_key_cell(emb_keys: List[str]) -> Optional[str]:
    """Out of available `emb_key` values, pick the cell-side embedding.

    Baselines write a single per-method emb_key (X_pca_leiden, X_harmony,
    X_scvi, X_geneformer, X_nicheformer, X_scgpt, X_scgpt_spatial,
    X_uce). SQUINT writes multiple emb keys; for the CELL benchmark we
    pick `cell_emb` (the POST-VQ quantized cell embedding, written from
    `H_quantized_cell` in run_squint.py:29316).

    Naming caveat
    -------------
    Despite the name, `cell_emb` is the QUANTIZED embedding (= the
    discrete code lookup, ~K1*K2 unique vectors per cell). The
    CONTINUOUS pre-VQ latent lives under `cell_latent` (= H_latent_cell).
    iLISI / MMD on the quantized embedding is by design here — the
    figure reports SQUINT's CODEBOOK-level batch integration, which is
    the property the paper claims (codes are batch-invariant
    representations of biology). If you want to A/B against the
    continuous pre-VQ representation, override on the CLI with
    `--squint-cell-emb-key cell_latent` (not currently exposed; add
    if needed).

    Preference order:
      1. `cell_emb`            — quantized, the headline reporting key
      2. `cell_emb_corrected`  — optional adversarial-corrected variant
      3. `cell_latent`         — continuous pre-VQ (fallback)
      4. `cell_latent_corrected`
    Then per-row fallbacks: any cell-prefixed key, then any non-niche key,
    then arbitrary.
    """
    if not emb_keys:
        return None
    # Explicit preference order — first match wins.
    for target in (
        "cell_emb",
        "cell_emb_corrected",
        "cell_latent",
        "cell_latent_corrected",
    ):
        if target in emb_keys:
            return target
    # Single emb_key (baselines): just take it.
    if len(emb_keys) == 1:
        return emb_keys[0]
    # Fall back: any cell-side key (guard against accidentally picking
    # a neighborhood-side key that happens to contain the substring
    # "cell").
    for ek in emb_keys:
        if ek.startswith("cell_"):
            return ek
    # Last resort: first non-niche key, else first.
    for ek in emb_keys:
        if "neighborhood" not in ek:
            return ek
    return emb_keys[0]


def _warn_duplicate_seed_values(df: pd.DataFrame, method_label: str) -> None:
    """Warn if, within a metric, multiple seeds carry IDENTICAL values — the
    fingerprint of a multi-seed run-dir collision (seeds that aren't actually
    independent; see run_squint_multi_seed.py's duplicate-run-dir guard). Such
    duplication biases the mean and understates the per-seed error bars."""
    if df.empty or "seed" not in df.columns:
        return
    for metric, g in df.groupby("metric"):
        n_seed = g["seed"].nunique()
        n_val = g["value"].round(6).nunique()
        if n_seed >= 2 and n_val < n_seed:
            print(f"  WARNING [{method_label}] {metric}: {n_seed} seeds but only "
                  f"{n_val} distinct value(s) — likely a run-dir collision "
                  f"(seeds not independent; error bars understated).",
                  file=sys.stderr)


def load_method_metrics(variant_dir: Path, method_label: str) -> pd.DataFrame:
    """Return tidy DataFrame: columns = (method, seed, metric, value).

    Reads ONLY the per-seed CSVs:
      <variant_dir>/<latest_TS>/metrics/per_seed_niche_identification.csv
      <variant_dir>/<latest_TS>/metrics/per_seed_batch_integration.csv

    Both files have one row per (seed, code_key / emb_key, label_key,
    ...). This unified schema applies to baselines AND to SQUINT
    multi-seed sweeps. Legacy single-seed SQUINT outputs
    (`niche_identification_metrics.csv` etc.) are intentionally NOT
    loaded — SQUINT is now expected to run through the multi-seed
    pipeline.

    Returns an empty DataFrame if no metrics are found.
    """
    m = _find_latest_metrics_dir(variant_dir)
    if m is None:
        return pd.DataFrame()

    rows: List[dict] = []

    # ---- Cell-type identification (NMI / ARI) ---------------------------
    per_seed_niche = m / "per_seed_niche_identification.csv"
    if per_seed_niche.is_file():
        df_n = pd.read_csv(per_seed_niche)
        if "split" in df_n.columns:
            df_n = df_n[df_n["split"] == "all"]
        df_n = df_n[df_n["label_key"] == "cell_type"]
        if not df_n.empty and "code_key" in df_n.columns:
            ck = _pick_cell_code_key(sorted(df_n["code_key"].unique().tolist()))
            if ck is not None:
                df_n = df_n[df_n["code_key"] == ck]
        for _, r in df_n.iterrows():
            rows.append({
                "method": method_label,
                "seed":   int(r.get("seed", 0)),
                "metric": "Cell-type NMI",
                "value":  float(r["NMI"]),
            })
            rows.append({
                "method": method_label,
                "seed":   int(r.get("seed", 0)),
                "metric": "Cell-type ARI",
                "value":  float(r["ARI"]),
            })

    # ---- Batch integration (iLISI / MMD) --------------------------------
    per_seed_batch = m / "per_seed_batch_integration.csv"
    if per_seed_batch.is_file():
        df_b = pd.read_csv(per_seed_batch)
        if "emb_key" in df_b.columns:
            ek = _pick_batch_emb_key_cell(sorted(df_b["emb_key"].unique().tolist()))
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

    out_df = pd.DataFrame(rows)
    _warn_duplicate_seed_values(out_df, method_label)
    return out_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel(
        ax,
        method_order: List[str],
        per_method_values: Dict[str, np.ndarray],
        colour_for: Dict[str, str],
        metric_label: str,
        x_invert_better: bool,
        show_y_ticklabels: bool = True,
    ) -> None:
    n = len(method_order)
    # Thick bars: 0.65 → ~65% of the row band. Combined with the tight
    # row spacing in `make_figure`, this leaves a thin inter-row gap and
    # makes each method's bar read as a solid swatch rather than a thin
    # line. DOT_JITTER stays < BAR_HEIGHT/2 so dots stay inside the bar.
    BAR_HEIGHT = 0.65
    DOT_JITTER = 0.18

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

        # Bar (mean)
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

    # Only set y-tick labels on the leftmost panel. With `sharey=True`
    # the y-axis is shared across all panels, so calling
    # `set_yticklabels([])` here would CLEAR labels from the shared
    # axis (including the leftmost one). Instead we set ticks on every
    # panel (cheap, consistent) and only set labels on the leftmost —
    # `sharey=True` auto-applies `labelleft=False` to the others.
    ax.set_yticks(range(n))
    if show_y_ticklabels:
        ax.set_yticklabels(method_order, fontweight="medium")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Tight ylim — each row gets 1.0 unit of axis space by default, but
    # we shrink to -0.4/n-0.6 to remove the vertical whitespace above
    # the topmost bar and below the bottommost.
    ax.set_ylim(-0.4, n - 0.6)
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

    # --- Method order: sort by mean Cell-type NMI descending so the
    # best-performing method ends up at the TOP of every panel.
    # The plot loop assigns method_order[0] to y=0; `ax.invert_yaxis()`
    # then flips y=0 to the top of the figure, so highest-NMI-on-top
    # falls out for free across all 4 panels (they share the y-axis
    # since `sharey=True`). Tie-broken by name to make the output
    # deterministic across reruns.
    if "Cell-type NMI" in pivot.columns:
        nmi_means = pivot["Cell-type NMI"].groupby(level=0).mean()
        method_order = (
            nmi_means
            .sort_values(ascending=False, kind="stable")
            .index.tolist()
        )
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

    # --- Figure size --------------------------------------------------
    # Compact layout: only the leftmost panel shows method names on the
    # y-axis, the other 3 share the same y-axis (no per-panel label
    # space wasted). Panels are narrow (1.1 in each) so bars don't
    # stretch too far horizontally; row spacing is tight (0.18 in/row)
    # which, combined with BAR_HEIGHT=0.65, leaves only a thin gap
    # between bars.
    panel_width_in = 1.1
    label_pad_in = 1.0                  # left margin for the y-tick labels
    fig_width_in = panel_width_in * len(METRICS) + label_pad_in
    fig_height_in = max(1.4, 0.18 * len(method_order) + 0.5)

    fig, axes = plt.subplots(
        1, len(METRICS),
        figsize=(fig_width_in, fig_height_in),
        sharey=True,
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
            x_invert_better=(arrow == "↓"),
            show_y_ticklabels=(i == 0),
        )

    plt.tight_layout()
    # Tighter inter-panel spacing now that only one panel carries
    # y-tick labels — no labels for the right-side panels to bleed
    # into the previous panel's bars.
    plt.subplots_adjust(wspace=0.12)

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)

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
    p.add_argument("--squint-variant", type=str, default=DEFAULT_SQUINT_VARIANT,
                   help="SQUINT multi-seed variant directory name (under "
                        "<artifacts_root>/<dataset_tag>/). Must be a "
                        "`__multiseed` sweep so the loader finds "
                        "per_seed_*.csv. Default: "
                        f"'{DEFAULT_SQUINT_VARIANT}'.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: "
                        "<artifacts_root>/benchmarking/figures/).")
    p.add_argument("--out-name", type=str,
                   default="cell_type_identification_benchmark",
                   help="Output basename (no extension). Default: "
                        "'cell_type_identification_benchmark'.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"

    _apply_nature_style()

    # Build the final variant->label mapping: baselines (fixed) + the
    # SQUINT multi-seed sweep chosen on the CLI. Insertion order is
    # preserved by dict in Py3.7+, so iteration order is deterministic
    # (baselines first, then SQUINT) — final plot order is then
    # determined by mean Cell-type NMI in `make_figure`.
    methods: Dict[str, str] = dict(DEFAULT_BASELINES)
    methods[args.squint_variant] = SQUINT_LABEL

    print(f"Artifacts root: {args.artifacts_root}")
    print(f"Dataset tag:    {args.dataset_tag}")
    print(f"SQUINT variant: {args.squint_variant}")
    print(f"Output dir:     {args.out_dir}")
    print()
    print("Loading metrics:")

    frames: List[pd.DataFrame] = []
    for variant, label in methods.items():
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
