#!/usr/bin/env python3
"""
Runtime benchmark figure (Nature style).

Loads per-seed runtimes for ALL configured baselines (cell-type
identification + niche identification) plus a SQUINT multi-seed sweep,
and renders ONE horizontal-bar panel: mean as a translucent bar, SEM
as an error bar, individual seeds overlaid as dots.

Methods are sorted ascending by mean runtime (fastest at the top of
the figure). The x-axis is **log-scaled** because runtimes span 2-3
orders of magnitude: linear PCA + Leiden (~10-30 s) up to
NicheCompass + foundation models (~1-2 h). On a linear axis the
faster methods collapse into a single pixel column; on log they're
all readable.

Inputs (per method):
  <artifacts_root>/<dataset_tag>/<variant>/<latest_TS>/metrics/
      per_seed_runtimes.csv

`<variant>` is the `baseline-<name>` directory for the 14 baselines,
or the SQUINT `__multiseed` sweep directory. Both produce the same
`per_seed_runtimes.csv` schema (one row per seed; columns:
seed, method, runtime_seconds, local_seconds, shared_setup_seconds).
`<latest_TS>` is auto-selected per variant (most recent timestamp
whose `metrics/per_seed_runtimes.csv` exists).

Runtime semantics (all methods, apples-to-apples):
  - `runtime_seconds = local_seconds + shared_setup_seconds`. Each
    row is the "time to obtain clusters from raw data for one seed":
    per-seed clustering (Leiden binary search, or the method's own
    domain assignment for Novae) plus the shared embedding compute
    (BANKSY+Harmony, PCA, FM extraction, etc.) amortised back to
    each seed.
  - For SQUINT it's "train(seed) + predict(seed)" — directly
    comparable.
  - EXCLUDED everywhere: metric computation (NMI/ARI/iLISI/MMD),
    UMAP for visualisation, per-seed plot writes. Benchmark
    scaffolding, not method cost.

Outputs:
  <out_dir>/runtime_benchmark.{svg,png,csv}

Default <out_dir> is `<artifacts_root>/benchmarking/figures/`.

Usage:
  # Default — all 14 baselines + the canonical SQUINT variant:
  python analysis/benchmarking/plots/plot_runtime_benchmark.py

  # Subset of methods (comma-separated baseline variant dirs):
  python analysis/benchmarking/plots/plot_runtime_benchmark.py \\
      --methods baseline-banksy,baseline-cellcharter,baseline-pca-leiden

  # Different SQUINT multi-seed sweep:
  python analysis/benchmarking/plots/plot_runtime_benchmark.py \\
      --squint-variant \\
      dualvq+small-h64+rvq-both-3level+decoder-cov+adv+mmb0-1b_smb1-1b_1p__multiseed

  # Different dataset:
  python analysis/benchmarking/plots/plot_runtime_benchmark.py \\
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
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullLocator


# ---------------------------------------------------------------------------
# Defaults: paths + which methods to include
# ---------------------------------------------------------------------------

DEFAULT_ARTIFACTS_ROOT = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts"
)
DEFAULT_DATASET_TAG = "mmb0-1b_smb1-1b_1p"

# Cell-type-identification baselines. variant_dir -> display label.
# PCA+Leiden and Harmony are intentionally OMITTED from the default
# — they're the linear-baseline floor (~10-60 s) and squash the log
# axis: the heavier learned methods (scVI, foundation models, SQUINT)
# get pushed into the right half of the panel. The default figure
# compares learned methods on like-for-like compute. Add either back
# with `--methods baseline-pca-leiden,baseline-harmony,...` when you
# specifically want to show the linear-baseline reference.
DEFAULT_BASELINES_CELL_TYPE: Dict[str, str] = {
    "baseline-scvi":           "scVI",
    "baseline-geneformer":     "Geneformer",
    "baseline-nicheformer":    "Nicheformer",
    "baseline-scgpt":          "scGPT",
    "baseline-scgpt-spatial":  "scGPT-spatial",
    "baseline-uce":            "UCE",
}

# Niche-identification baselines. variant_dir -> display label.
# `baseline-neigh-expr-pca` (Neigh. expr. PCA) is intentionally
# OMITTED from the default: like PCA+Leiden on the cell-type side,
# it's the linear-baseline floor (~10-30 s) and squashes the log
# axis against the heavier learned methods. Add back via
# `--methods baseline-neigh-expr-pca,...` when you specifically
# want to show the spatial-linear comparator.
DEFAULT_BASELINES_NICHE_ID: Dict[str, str] = {
    "baseline-banksy":          "BANKSY",
    "baseline-cellcharter":     "CellCharter",
    "baseline-graphst":         "GraphST",
    "baseline-novae":           "Novae",
    "baseline-nichecompass":    "NicheCompass",
}

# Default SQUINT variant (matches the metric figures). Must be a
# `__multiseed` sweep so its `metrics/per_seed_runtimes.csv` is
# populated the same way as the baselines'.
DEFAULT_SQUINT_VARIANT = (
    # s49_v23: decoupled-enc + diversity wt=10 + within-batch contrastive wt=10
    # on the s48_v2 spine (cell-w=1, no-batch-int, enc-deeper, within-sec).
    # Replaces the previous default (s42 winner, cell-w=5.0) as of the
    # s49 sweep results — see project_squint.md.
    "s49_v23_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p__multiseed"
)
SQUINT_LABEL = "SQUINT"

# Category tags drive the colour assignment + the legend group order.
# Keep these aligned with the per-category palettes below. These
# strings are used BOTH as the legend labels rendered on the figure
# AND as the values written into the `category` column of the
# companion CSV — so picking short, plain labels here keeps the CSV
# readable too.
CATEGORY_CELL_TYPE = "Cell Type ID"
CATEGORY_NICHE_ID  = "Niche ID"
CATEGORY_SQUINT    = "SQUINT"

# Per-METHOD colour. Each method keeps the same colour it has on its
# native metric plot, so a reader can identify methods across the
# three figures by their colour alone:
#   - Cell-type-ID methods use SHADES OF BLUE (light scVI → dark UCE),
#     matching `METHOD_COLOURS` in `plot_cell_type_identification_benchmark.py`.
#   - Niche-ID methods use SHADES OF BROWN (light BANKSY → dark
#     NicheCompass), matching `METHOD_COLOURS` in
#     `plot_niche_identification_benchmark.py`.
#   - SQUINT keeps its accent red across all three plots.
# The bars / dots in the runtime plot use this per-method dict;
# the LEGEND uses the category-mean colours below.
METHOD_COLOURS: Dict[str, str] = {
    # Cell-type-ID — blue family (light → dark, matching the cell-type-ID plot)
    "scVI":           "#A8DADC",
    "Geneformer":     "#48CAE4",
    "Nicheformer":    "#0096C7",
    "scGPT":          "#0077B6",
    "scGPT-spatial":  "#023E8A",
    "UCE":            "#03045E",
    # Niche-ID — brown family (light → dark, matching the niche-ID plot)
    "BANKSY":         "#E8C19D",
    "CellCharter":    "#C8A27C",
    "GraphST":        "#A0522D",
    "Novae":          "#7C3F00",
    "NicheCompass":   "#3D2817",
    # Our method
    "SQUINT":         "#FF006E",
}

# Per-CATEGORY colour. Used ONLY for the legend swatches — one entry
# per method category, picking a representative "mid" shade from
# each family. The bars themselves are per-method (see
# `METHOD_COLOURS` above); the legend just summarises the family
# membership so a reader scanning the figure knows "blue ≈ cell-type
# baseline, brown ≈ niche-id baseline, red = SQUINT".
CATEGORY_COLOURS: Dict[str, str] = {
    CATEGORY_CELL_TYPE: "#0077B6",   # mid blue   — cell-type baselines (legend swatch)
    CATEGORY_NICHE_ID:  "#A0522D",   # sienna     — niche-id baselines (legend swatch)
    CATEGORY_SQUINT:    "#FF006E",   # accent red — our method (unchanged)
}


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
# Per-method runtime loader
# ---------------------------------------------------------------------------

def _find_latest_metrics_dir(variant_dir: Path) -> Optional[Path]:
    """Return the most recent `<TS>/metrics` dir under `variant_dir`
    that contains `per_seed_runtimes.csv`.

    Timestamps are sorted lexicographically (works for the
    `YYYYMMDD_HHMMSS` convention used by every upstream writer).
    Returns None if `variant_dir` is missing or no timestamp under
    it has a populated runtime CSV.
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
        if (m / "per_seed_runtimes.csv").is_file():
            return m
    return None


def load_method_runtimes(
        variant_dir: Path,
        method_label: str,
        category: str,
    ) -> pd.DataFrame:
    """Return tidy DataFrame: columns = (method, category, seed, runtime_seconds).

    Reads ONLY `per_seed_runtimes.csv`. Other columns from the source
    CSV (`local_seconds`, `shared_setup_seconds`, source `method`
    field) are dropped — we keep just the comparable per-seed total.

    Returns an empty DataFrame if no runtime CSV is found (e.g. the
    method's run hasn't finished, or it lives at a different path).
    """
    m = _find_latest_metrics_dir(variant_dir)
    if m is None:
        return pd.DataFrame()
    csv = m / "per_seed_runtimes.csv"
    df = pd.read_csv(csv)
    if df.empty or "runtime_seconds" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "method":          method_label,
        "category":        category,
        "seed":            df["seed"].astype(int) if "seed" in df.columns
                           else np.arange(len(df)),
        "runtime_seconds": df["runtime_seconds"].astype(float),
    })
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _format_seconds(s: float) -> str:
    """Human-readable runtime label (used for the inline 'mean' annotation
    next to each bar). Picks the right unit so a sub-minute method
    doesn't display as '0.0 h'."""
    if not np.isfinite(s) or s <= 0:
        return "n/a"
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


def make_figure(
        df: pd.DataFrame,
        out_path_base: Path,
        category_colours: Dict[str, str] = CATEGORY_COLOURS,
        method_colours: Dict[str, str] = METHOD_COLOURS,
    ) -> None:
    """Build the 1-panel runtime figure and save .svg / .png /.csv next
    to `out_path_base`.

    Plot conventions (matching the metric figures):
      - Translucent horizontal bar = mean across seeds
      - Black error bar = SEM (only if >=2 seeds)
      - Coloured dots overlaid = individual seeds (jittered vertically
        so overlapping points stay visible)
      - Method order = mean runtime ascending (fastest at the top
        after invert_yaxis)
      - X-axis log10, ticks in seconds; an inline `mean` annotation
        next to each bar gives a human-readable conversion (min/h)
    """
    if df.empty:
        raise SystemExit("No runtime CSVs loaded; refusing to plot empty figure.")

    # --- Per-method mean / SEM / dots -------------------------------------
    grouped = df.groupby(["method", "category"])
    method_stats = (
        grouped["runtime_seconds"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    method_stats = method_stats.sort_values(
        "mean", ascending=True, kind="stable"
    ).reset_index(drop=True)
    method_order = method_stats["method"].tolist()
    n = len(method_order)

    # --- Figure size ------------------------------------------------------
    # Compact, portrait-oriented layout. Tuned for a typical Nature
    # half-column slot. Panel width 0.55 in is near the floor before
    # the 3 log-tick labels start crowding (matplotlib's LogLocator
    # auto-thins to 2 ticks if it can't fit 3); the 0.9-in label-pad
    # column carries the y-tick method names ("scGPT-spatial" is the
    # widest, fits at ~6.5 pt with room to spare). Per-row height
    # 0.18 in keeps the whole figure under ~2.6 in vertically for
    # ~12 methods while leaving each bar physically thick enough to
    # read at print scale.
    panel_width_in = 0.55
    label_pad_in   = 0.9
    fig_width_in   = panel_width_in + label_pad_in
    fig_height_in  = max(1.8, 0.18 * n + 0.4)

    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))

    # --- Resolve x-axis limits UP-FRONT --------------------------------
    # We need `x_min` BEFORE drawing any bars: matplotlib's `barh`
    # defaults to `left=0`, and on a log axis `log(0) = -inf`. The
    # visible rendering is clipped correctly, but the SVG path for
    # each bar still encodes an extreme negative x coordinate
    # (~1e308). Adobe Illustrator parses those literally; when the
    # SVG is copied across documents it fails with
    # "Can't paste the objects. The requested transformation would
    # make some objects too large." Anchoring `left=x_min` keeps the
    # SVG geometry inside Illustrator's coordinate envelope without
    # changing what's visible.
    finite_runtimes = df[df["runtime_seconds"] > 0]["runtime_seconds"]
    if not finite_runtimes.empty:
        x_min = float(finite_runtimes.min()) / 2.5
        x_max = float(finite_runtimes.max()) * 1.3
    else:
        x_min, x_max = 1.0, 1e4

    # Bar thickness (vertical) is a fraction of the 1.0-unit row band.
    # 0.6 leaves a thin gap between rows at the new 0.18-in spacing.
    # DOT_JITTER stays below BAR_HEIGHT/2 so per-seed dots remain
    # inside their row's bar band.
    BAR_HEIGHT = 0.6
    DOT_JITTER = 0.13

    for j, method in enumerate(method_order):
        category = method_stats.loc[
            method_stats["method"] == method, "category"
        ].iloc[0]
        # Per-method colour (a specific shade of the family — blue
        # for cell-type-ID, brown for niche-ID, red for SQUINT). The
        # legend below uses the category-mean colour instead, so the
        # legend tells you "this family is blue/brown" while each
        # individual bar's shade tells you "this method specifically".
        # Fall back to the category-mean colour if the method isn't
        # in `method_colours` (defensive — e.g. when `--methods` adds
        # a baseline that wasn't in the runtime plot's METHOD_COLOURS).
        colour = method_colours.get(
            method,
            category_colours.get(category, "#888888"),
        )
        sub = df[df["method"] == method]["runtime_seconds"].astype(float).values
        sub = sub[np.isfinite(sub) & (sub > 0)]
        if sub.size == 0:
            ax.text(1.0, j, "n/a", va="center", ha="left",
                    fontsize=5.5, fontstyle="italic", color="0.5")
            continue
        mean_val = float(sub.mean())

        # Bar — anchored at `x_min` (left edge of the panel) so the
        # underlying Rectangle path stays inside Illustrator's
        # coordinate envelope. `width = mean_val - x_min` keeps the
        # bar visually ending at the mean, identical to the
        # `left=0, width=mean_val` default but without the log(0)
        # path-coordinate explosion.
        ax.barh(j, mean_val - x_min, left=x_min, height=BAR_HEIGHT,
                color=colour, edgecolor=colour,
                linewidth=0.5, alpha=0.35, zorder=2)
        # SEM (only when >=2 seeds)
        if sub.size > 1:
            sem = float(sub.std(ddof=1) / np.sqrt(sub.size))
            # Don't draw an error bar smaller than the marker size — on
            # log10 it can render as a phantom horizontal line below the
            # bar. Skip the cap whenever SEM is essentially zero.
            if sem > mean_val * 1e-3:
                ax.errorbar(mean_val, j, xerr=sem, fmt="none",
                            ecolor="0.3", elinewidth=0.5,
                            capsize=1.2, capthick=0.5, zorder=3)
        # Per-seed dots (jittered vertically)
        rng = np.random.default_rng(42 + j)
        yj = rng.uniform(-DOT_JITTER, DOT_JITTER, size=sub.size)
        ax.scatter(sub, np.full_like(sub, j, dtype=float) + yj,
                   s=12, color=colour, edgecolors="white",
                   linewidths=0.3, zorder=4, alpha=0.92)

    # --- Axes -------------------------------------------------------------
    ax.set_yticks(range(n))
    ax.set_yticklabels(method_order, fontweight="medium")
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()  # fastest-on-top after the ascending sort
    ax.set_xscale("log")
    ax.set_xlabel("Runtime per seed (s, log scale)")
    # Major ticks at powers of 10 only, capped to ~3 visible labels so
    # the narrow 1.5-in panel doesn't read as a wall of numbers.
    # `LogLocator(numticks=3)` is a soft cap: matplotlib picks the
    # subset that best brackets the data. Minor ticks + minor
    # gridlines are suppressed entirely (`NullLocator`) — they were
    # the dominant source of x-axis label clutter on the previous
    # `which="both"` config.
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    ax.xaxis.grid(True, which="major", linewidth=0.2, alpha=0.35,
                  color="0.65", linestyle="--")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Inline mean-time annotation ('1.2 h', '45 s', ...) --------------
    # Placed just to the LEFT of each method's leftmost dot (= fastest
    # seed), with `ha="right"` so the label text terminates at the
    # offset position. This guarantees the label never overlays a dot:
    # the dot cloud spans [min_seed, max_seed], the bar tip is at the
    # mean inside that range, so any label position to the LEFT of
    # min_seed is in empty space. The previous design (right of the
    # bar tip at 1.25 * mean) overlapped the slower-seed dots whenever
    # per-seed variance was high.
    #
    # `1.4 ×` log-offset (~0.15 decade gap) keeps a consistent visual
    # gap between label and leftmost dot across the whole range.
    # `x_min` / `x_max` were computed at the top of make_figure (before
    # the bar loop) so they could anchor `barh(left=x_min)`. Apply them
    # now that all artists are drawn.
    ax.set_xlim(x_min, x_max)
    for j, method in enumerate(method_order):
        sub = df[df["method"] == method]["runtime_seconds"].astype(float).values
        sub = sub[np.isfinite(sub) & (sub > 0)]
        if sub.size == 0:
            continue
        mean_val = float(sub.mean())
        leftmost = float(sub.min())
        # `leftmost / 2.0` ≈ 0.30-decade offset to the left of the
        # leftmost dot. The panel is only 0.55 in wide so we need a
        # generous log offset to keep a visible pixel gap between
        # the label's right edge and the dot.
        ax.text(leftmost / 2.0, j, _format_seconds(mean_val),
                va="center", ha="right", fontsize=5.5, color="0.25")

    # --- Legend (categories) ---------------------------------------------
    # Only show categories that actually have a method in the figure —
    # avoids a phantom entry if e.g. SQUINT is missing.
    present_cats = [
        c for c in (CATEGORY_CELL_TYPE, CATEGORY_NICHE_ID, CATEGORY_SQUINT)
        if c in method_stats["category"].values
    ]
    handles = [
        mpl.patches.Patch(facecolor=category_colours[c], alpha=0.55,
                          edgecolor=category_colours[c], label=c)
        for c in present_cats
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            fontsize=5.5,
            handlelength=1.2,
            handleheight=0.8,
            borderpad=0.3,
            labelspacing=0.3,
        )

    plt.tight_layout()

    # White figure + axes background. Earlier the runtime plot was saved
    # with `transparent=True` so its SVG bbox was tight when imported into
    # Illustrator — but the PNGs came out with a transparent BG which made
    # them hard to read in slides / preview. Restoring an explicit white
    # background here. If you need the Illustrator-tight-bbox behaviour
    # back, swap to `transparent=True` on the savefig calls below.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(
            out,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor=fig.get_facecolor(),
        )
        print(f"  -> {out}")
    plt.close(fig)

    # Tidy CSV companion (one row per (method, seed)).
    csv_out = out_path_base.with_suffix(".csv")
    df.sort_values(["method", "seed"]).to_csv(csv_out, index=False)
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
                        "<artifacts_root>/<dataset_tag>/). Default: "
                        f"'{DEFAULT_SQUINT_VARIANT}'.")
    p.add_argument("--methods", type=str, default=None,
                   help="Optional comma-separated subset of baseline variant "
                        "dirs to include (e.g. "
                        "'baseline-banksy,baseline-cellcharter'). The SQUINT "
                        "variant is always included on top of any subset — "
                        "pass `--no-squint` to drop it. Default: all 14 "
                        "configured baselines.")
    p.add_argument("--no-squint", action="store_true",
                   help="Don't include the SQUINT variant in the plot.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: "
                        "<artifacts_root>/benchmarking/figures/).")
    p.add_argument("--out-name", type=str,
                   default="runtime_benchmark",
                   help="Output basename (no extension). Default: "
                        "'runtime_benchmark'.")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"

    _apply_nature_style()

    # Resolve baselines:
    #   - default: every cell-type + niche-id baseline registered above
    #   - user subset: filter down by the variant-dir slugs they passed
    if args.methods is None:
        chosen_cell_type = dict(DEFAULT_BASELINES_CELL_TYPE)
        chosen_niche_id  = dict(DEFAULT_BASELINES_NICHE_ID)
    else:
        wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
        chosen_cell_type = {
            k: v for k, v in DEFAULT_BASELINES_CELL_TYPE.items() if k in wanted
        }
        chosen_niche_id = {
            k: v for k, v in DEFAULT_BASELINES_NICHE_ID.items() if k in wanted
        }
        unknown = wanted - set(DEFAULT_BASELINES_CELL_TYPE) - set(DEFAULT_BASELINES_NICHE_ID)
        if unknown:
            print(f"WARNING: --methods entries not in the known baseline "
                  f"list (ignored): {sorted(unknown)}", file=sys.stderr)

    print(f"Artifacts root: {args.artifacts_root}")
    print(f"Dataset tag:    {args.dataset_tag}")
    if not args.no_squint:
        print(f"SQUINT variant: {args.squint_variant}")
    print(f"Output dir:     {args.out_dir}")
    print()
    print("Loading per-seed runtimes:")

    frames: List[pd.DataFrame] = []

    def _load(variant: str, label: str, category: str) -> None:
        variant_dir = args.artifacts_root / args.dataset_tag / variant
        sub = load_method_runtimes(variant_dir, label, category)
        n_seeds = sub["seed"].nunique() if not sub.empty else 0
        status = f"{n_seeds} seed(s)" if not sub.empty else "MISSING"
        print(f"  [{label:<22s}]  {status:<12s}  ({variant_dir})")
        if not sub.empty:
            frames.append(sub)

    for variant, label in chosen_cell_type.items():
        _load(variant, label, CATEGORY_CELL_TYPE)
    for variant, label in chosen_niche_id.items():
        _load(variant, label, CATEGORY_NICHE_ID)
    if not args.no_squint:
        _load(args.squint_variant, SQUINT_LABEL, CATEGORY_SQUINT)

    if not frames:
        raise SystemExit(
            "\nNo runtime CSVs found for any configured method. Check "
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
