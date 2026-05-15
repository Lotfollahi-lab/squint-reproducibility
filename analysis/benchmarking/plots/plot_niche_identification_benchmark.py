#!/usr/bin/env python3
"""
Niche-identification benchmark figure (Nature style).

Loads per-seed metrics for niche-identification baselines + a SQUINT
multi-seed sweep, and renders one figure with FOUR horizontal-bar panels:

  Niche NMI   (↑ better)
  Niche ARI   (↑ better)
  iLISI       (↑ better)
  MMD         (↓ better)

Each panel shows the mean across seeds as a translucent bar, the SEM as
an error bar, and individual seeds overlaid as dots (matching the
reference notebook style).

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

Outputs:
  <out_dir>/niche_identification_benchmark.{svg,png}
  <out_dir>/niche_identification_benchmark.csv

Default <out_dir> is `<artifacts_root>/benchmarking/figures/`.

Usage:
  # Default SQUINT variant (winner across the v8/v9 sweeps):
  python analysis/benchmarking/plots/plot_niche_identification_benchmark.py

  # Compare a different SQUINT multi-seed sweep:
  python analysis/benchmarking/plots/plot_niche_identification_benchmark.py \\
      --squint-variant \\
      dualvq+small-h64+rvq-both-3level-30-20-10+decoder-cov+adv+mmb0-1b_smb1-1b_1p__multiseed

  # Different dataset:
  python analysis/benchmarking/plots/plot_niche_identification_benchmark.py \\
      --dataset-tag chl59-8b_1p
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

# Baseline variant directories -> display labels.
# Display order in the plot is determined later by mean Niche NMI; SQUINT
# is always plotted with a distinct accent colour regardless of position.
# `baseline-neigh-expr-pca` (Neighbour expr. PCA) is intentionally OMITTED —
# it's the linear-baseline counterpart of cell-type-side PCA+Leiden, and
# the comparison of interest on this figure is learned niche-identification
# methods + SQUINT. Add it back here if you want the linear reference.
DEFAULT_BASELINES: Dict[str, str] = {
    "baseline-banksy":          "BANKSY",
    "baseline-cellcharter":     "CellCharter",
    "baseline-graphst":         "GraphST",
    "baseline-novae":           "Novae",
    "baseline-nichecompass":    "NicheCompass",
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

# Colour family: SHADES OF BROWN for all niche-ID baselines, SQUINT
# keeps its accent red. The cell-type-ID plot uses SHADES OF BLUE
# for its baselines (see plot_cell_type_identification_benchmark.py),
# so the two metric plots are immediately distinguishable as
# "niche-ID = brown family" vs "cell-type-ID = blue family", with
# SQUINT as the consistent red accent across both. Shades are
# arranged light -> dark and chosen for clear pair-wise contrast
# within the family.
METHOD_COLOURS: Dict[str, str] = {
    "BANKSY":              "#E8C19D",   # light tan          — baseline
    "CellCharter":         "#C8A27C",   # medium tan         — baseline
    "GraphST":             "#A0522D",   # sienna             — baseline
    "Novae":               "#7C3F00",   # dark sienna        — baseline
    "NicheCompass":        "#3D2817",   # very dark brown    — baseline
    "SQUINT":              "#FF006E",   # accent red/magenta — our method (unchanged)
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


def _pick_niche_code_key(code_keys: List[str]) -> Optional[str]:
    """Out of the available `code_key` values in a niche metrics CSV,
    pick the one that corresponds to the niche-side codes.

    Baselines write a single `code_key="leiden"`. SQUINT writes
    `code_key` values like `cell_code_index`, `neighborhood_code_index`,
    `cell_code_indices[level_0]`, `neighborhood_code_indices[composite]`,
    etc.

    Preference order (niche-side):
      1. `neighborhood_code_indices[level_0]` — K1=30 macro RVQ
         partition. HEADLINE niche-NMI metric across the rest of the
         tooling (compare_variants.py reports
         `niche|all|neighborhood_code_indices[level_0]|<niche_label>|NMI`).
      2. `neighborhood_code_index` — single-level VQ fallback.
      3. `neighborhood_code_indices[composite]` — K1*K2 leaf clusters
         (factorised RVQ levels). Last resort niche-side.
      4. `leiden` — baseline default.

    Notes
    -----
    Previously preferred `[composite]` over `[level_0]`, which silently
    reported a DIFFERENT niche-NMI than the rest of the analysis
    tooling. Switched on user request so this figure matches the
    `compare_variants.py` headline.
    """
    # 1. Niche level_0 (RVQ macro cluster) — HEADLINE metric across tooling.
    for ck in code_keys:
        if ck.startswith("neighborhood_code_indices") and "level_0" in ck:
            return ck
    # 2. Niche single-level (non-residual VQ fallback)
    for ck in code_keys:
        if ck == "neighborhood_code_index":
            return ck
    # 3. Niche composite (RVQ-style multi-level, full-resolution leaf) — last resort niche-side
    for ck in code_keys:
        if ck.startswith("neighborhood_code_indices") and "composite" in ck:
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
    neighborhood_latent, plus optional `_corrected` adversarial variants).
    For the NICHE benchmark we pick `neighborhood_emb` (the POST-VQ
    quantized niche embedding, written from `H_quantized_niche` in
    run_squint.py:29318).

    Naming caveat
    -------------
    Despite the name, `neighborhood_emb` is the QUANTIZED embedding
    (= the discrete code lookup, ~K1*K2 unique vectors per cell). The
    CONTINUOUS post-GNN latent lives under `neighborhood_latent` (=
    H_latent_niche). iLISI / MMD on the quantized embedding is by
    design here — the figure reports SQUINT's CODEBOOK-level batch
    integration, which is the property the paper claims (codes are
    batch-invariant representations of niche identity).

    Preference order:
      1. `neighborhood_emb`            — quantized, the headline reporting key
      2. `neighborhood_emb_corrected`  — optional adversarial-corrected variant
      3. `neighborhood_latent`         — continuous post-GNN (fallback)
      4. `neighborhood_latent_corrected`
    """
    if not emb_keys:
        return None
    # Explicit preference order — first match wins.
    for target in (
        "neighborhood_emb",
        "neighborhood_emb_corrected",
        "neighborhood_latent",
        "neighborhood_latent_corrected",
    ):
        if target in emb_keys:
            return target
    # Single emb_key (baselines): just take it.
    if len(emb_keys) == 1:
        return emb_keys[0]
    # Fall back: any niche-side key.
    for ek in emb_keys:
        if "neighborhood" in ek:
            return ek
    # Last resort: first.
    return emb_keys[0]


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

    # ---- Niche identification (NMI / ARI) -------------------------------
    per_seed_niche = m / "per_seed_niche_identification.csv"
    if per_seed_niche.is_file():
        df_n = pd.read_csv(per_seed_niche)
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
    if per_seed_batch.is_file():
        df_b = pd.read_csv(per_seed_batch)
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
    # Thick bars: 0.65 → ~65% of the row band. Combined with the tight
    # row spacing (0.18 in/row) in `make_figure`, this leaves only a
    # thin gap between bars — same convention as the cell-type
    # benchmark figure. DOT_JITTER stays < BAR_HEIGHT/2 so the per-seed
    # dots stay within the bar band.
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

    # Only set y-tick labels on the leftmost panel. With `sharey=True`
    # the y-axis is shared across all panels, so calling
    # `set_yticklabels([])` here would CLEAR labels from the shared
    # axis (including the leftmost one). Set ticks on every panel
    # (cheap, consistent), but only set labels on the leftmost —
    # `sharey=True` auto-applies `labelleft=False` to the others.
    ax.set_yticks(range(n))
    if show_y_ticklabels:
        ax.set_yticklabels(method_order, fontweight="medium")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Tight ylim — shrinks the row band so each thick bar (BAR_HEIGHT
    # = 0.65) is separated only by a thin gap, with no whitespace above
    # the topmost or below the bottommost bar.
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

    # --- Method order: sort by mean Niche NMI descending. -----------------
    # `kind="stable"` keeps the original methods-dict insertion order for
    # ties, so re-runs with identical metrics produce identical figures.
    if "Niche NMI" in pivot.columns:
        nmi_means = pivot["Niche NMI"].groupby(level=0).mean()
        method_order = nmi_means.sort_values(
            ascending=False, kind="stable"
        ).index.tolist()
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

    # --- Figure size: compact Nature layout. ------------------------------
    # Only the leftmost panel carries y-tick labels (method names), so the
    # remaining panels can be narrower and packed close together. Panels
    # are narrow (1.1 in each) so bars don't stretch too far horizontally;
    # row spacing is tight (0.18 in/row) which, combined with BAR_HEIGHT
    # = 0.65, leaves only a thin gap between bars. `label_pad_in` is the
    # extra width reserved on the left edge for the method names.
    panel_width_in = 1.1
    label_pad_in = 1.0
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
            show_y_ticklabels=(i == 0),
            x_invert_better=(arrow == "↓"),
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.12)

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
                   default="niche_identification_benchmark",
                   help="Output basename (no extension). Default: "
                        "'niche_identification_benchmark'.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"

    _apply_nature_style()

    # Build the final variant->label mapping: baselines (fixed) + the
    # SQUINT multi-seed sweep chosen on the CLI. Insertion order is
    # preserved by dict in Py3.7+, so iteration order is deterministic
    # (baselines first, then SQUINT) — final plot order is then
    # determined by mean Niche NMI in `make_figure`.
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
