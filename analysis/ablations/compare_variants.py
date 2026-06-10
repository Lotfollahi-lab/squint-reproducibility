"""
Compare variant metrics across a dataset's run dirs.

Loads `metrics/niche_identification_metrics.csv`,
`metrics/batch_integration_metrics.csv`, and
`metrics/pearson_reconstruction_metrics.csv` for every variant under
`<artifacts_root>/<dataset>/<variant>/<timestamp>/metrics/` and renders
a set of comparison plots.

Output:
  <out-dir>/
    summary_long.csv
        Concatenated long-format DataFrame of all loaded metrics, tagged
        with `variant`. Easy to drop into a notebook for further
        analysis.
    key_metrics/
        One PNG + SVG per "headline" metric (the list the user cares
        about most), horizontal-bar chart with variants ranked.
    nmi_heatmap.{png,svg}
    ari_heatmap.{png,svg}
        Variants × (code_key, label_key) heatmaps for the niche-
        identification table (split=all only). One full overview of
        which variants do well on which (code, label) pairing.
        Priority columns (the user's headline (code, label) pairs)
        are framed in red and rendered with a bold tick label.
    batch_integration.{png,svg}
        Variants × (emb_key, metric) heatmap for iLISI / ASW / MMD.
        Priority columns (iLISI on cell_latent and
        neighborhood_latent) are framed in red.
    pearson_reconstruction.{png,svg}
        Variants × (branch, axis, transform, gene_subset) heatmap of
        `pearson_mean` (split=all only). Priority columns
        (gene_wise log1p) are framed in red.

Default headline metrics:
  - all, neighborhood_code_indices[level_0],  niche
  - all, cell_code_indices[level_0],          cell_type
  - neighborhood_latent, iLISI
  - neighborhood_latent, MMD
  - cell_latent,         iLISI
  - cell_latent,         MMD

Usage:
  # Default: only `dualvq*` variants, each variant's most recent timestamp.
  python analysis/ablations/compare_variants.py

  # Only sweep-aliased variants (the `s<sweep>_v<variant>_<base>`
  # convention introduced in sweep 17 — `s17_*`, `s18_*`, ...):
  python analysis/ablations/compare_variants.py --prefix s

  # Include baselines + smoke / region-holdout subdirs too:
  python analysis/ablations/compare_variants.py --include-baselines

  # Restrict to specific variants (explicit list ALWAYS bypasses the
  # prefix filter, no --include-baselines needed):
  python analysis/ablations/compare_variants.py \\
      --variants 'dualvq+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p,\\
                  baseline-nichecompass'

  # Different dataset:
  python analysis/ablations/compare_variants.py \\
      --dataset chl59-8b_1p

  # Different artifacts root (e.g. on a non-default cluster install):
  python analysis/ablations/compare_variants.py \\
      --artifacts-root /alt/artifacts/path
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# Defaults --------------------------------------------------------------------

DEFAULT_ARTIFACTS_ROOT = Path("/nfs/team361/sb75/squint-reproducibility/artifacts")
DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"

# (split, code_key, label_key, table) -> friendly label.
# Table is one of "niche" (niche_identification_metrics.csv) or
# "batchint" (batch_integration_metrics.csv).
KEY_METRICS: List[Tuple[str, str]] = [
    # (selector_str, friendly_label)
    # selectors are "split,code_key,label_key,score_col" for niche table
    # or "emb_key,metric" for batchint table.
    ("niche|all|neighborhood_code_indices[level_0]|niche|NMI",
     "neighborhood codes vs niche labels  —  NMI"),
    ("niche|all|neighborhood_code_indices[level_0]|niche|ARI",
     "neighborhood codes vs niche labels  —  ARI"),
    ("niche|all|cell_code_indices[level_0]|cell_type|NMI",
     "cell codes vs cell_type  —  NMI"),
    ("niche|all|cell_code_indices[level_0]|cell_type|ARI",
     "cell codes vs cell_type  —  ARI"),
    ("batchint|neighborhood_latent|iLISI",
     "neighborhood_latent  —  iLISI  (higher = more batch-mixed)"),
    ("batchint|neighborhood_latent|MMD",
     "neighborhood_latent  —  MMD   (lower = better integration)"),
    ("batchint|cell_latent|iLISI",
     "cell_latent  —  iLISI  (higher = more batch-mixed)"),
    ("batchint|cell_latent|MMD",
     "cell_latent  —  MMD   (lower = better integration)"),
]

# Heatmap column labels (formed as `<col>  →  <label>` / `<emb>  —  <metric>`
# / `<branch> · <axis> · <transform> · <gene_subset>`) that the user wants
# called out visually. These are matched verbatim against the pivoted
# DataFrame's column index — keep in sync with the formatters below.
PRIORITY_NICHE_PAIRS: List[str] = [
    "neighborhood_code_indices[level_0]  →  niche",
    "cell_code_indices[level_0]  →  cell_type",
]
PRIORITY_BATCHINT_PAIRS: List[str] = [
    "neighborhood_latent  —  iLISI",
    "cell_latent  —  iLISI",
]
# Pearson columns most relevant for ranking variants: gene-wise log1p on
# both the full gene set and the top-N HVGs, for cell + niche branches.
# Highlighted on the pearson heatmap.
PRIORITY_PEARSON_COLS: List[str] = [
    "cell · gene_wise · log1p · all",
    "cell · gene_wise · log1p · hvg50",
    "niche · gene_wise · log1p · all",
    "niche · gene_wise · log1p · hvg50",
]


# Loading ---------------------------------------------------------------------

def _resolve_run_dir(
        variant_dir: Path,
        timestamp_strategy: str = "latest",
    ) -> Optional[Path]:
    """
    Pick a single run dir under `variant_dir` based on `timestamp_strategy`.

    `latest`  - the most recent timestamp by lexicographic order
                (timestamps are `YYYYMMDD_HHMMSS` so this matches calendar
                order).
    `<ts>`    - exact timestamp string (errors if not found).
    """
    if not variant_dir.is_dir():
        return None
    candidates = sorted(p for p in variant_dir.iterdir() if p.is_dir())
    if not candidates:
        return None
    if timestamp_strategy == "latest":
        return candidates[-1]
    matched = [p for p in candidates if p.name == timestamp_strategy]
    return matched[0] if matched else None


def _load_metrics_for_variant(
        run_dir: Path,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame],
               Optional[pd.DataFrame], List[str]]:
    """
    Returns (niche_df, batchint_df, pearson_df, missing_files). Any
    DataFrame is None if its CSV is missing (tracked in `missing_files`).
    """
    metrics_dir = run_dir / "metrics"
    niche_csv    = metrics_dir / "niche_identification_metrics.csv"
    batchint_csv = metrics_dir / "batch_integration_metrics.csv"
    pearson_csv  = metrics_dir / "pearson_reconstruction_metrics.csv"

    missing: List[str] = []
    niche_df = batchint_df = pearson_df = None

    if niche_csv.exists():
        niche_df = pd.read_csv(niche_csv)
    else:
        missing.append(str(niche_csv))

    if batchint_csv.exists():
        batchint_df = pd.read_csv(batchint_csv)
    else:
        missing.append(str(batchint_csv))

    if pearson_csv.exists():
        pearson_df = pd.read_csv(pearson_csv)
    else:
        missing.append(str(pearson_csv))

    return niche_df, batchint_df, pearson_df, missing


def load_all_variants(
        artifacts_root: Path,
        dataset: str,
        variants: Optional[List[str]] = None,
        timestamp_strategy: str = "latest",
        prefix_filter: Optional[Union[str, Sequence[str]]] = "dualvq",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    For every variant under `<artifacts_root>/<dataset>/`, load all
    metric CSVs (niche identification, batch integration, pearson
    reconstruction) and return them as long-format DataFrames tagged
    with `variant` and `timestamp`.

    Returns (niche_long, batchint_long, pearson_long, info_dict).
    `info_dict` carries diagnostic info: variants_found,
    variants_missing_files (list of paths), variants_no_run_dir
    (variants with no timestamp subdir), variants_filtered_out (the
    subdirs that auto-discovery skipped because they didn't match
    `prefix_filter`), and `prefix_filter` (the prefix actually used
    — echoed back so callers can render accurate diagnostic
    messages).

    `prefix_filter` (default `"dualvq"`): when auto-discovering (i.e.
    `variants is None`), restrict to subdirs whose name starts with
    this string. Can be a single string OR a sequence of strings —
    in the sequence case a subdir is kept iff its name starts with
    ANY of the prefixes.

      - "dualvq" (default): only historic SQUINT ablation runs
        (pre-sweep-17 naming convention).
      - "s":               only sweep-aliased variants
                           (s17_*, s18_*, ... — the
                           `s<sweep>_v<variant>_<base>` convention
                           introduced in sweep 17).
      - "baseline-":       only baseline runs (banksy / scvi / ...).
      - ("s51", "s52"):    OR-union — keep subdirs starting with
                           either prefix. Useful when one sweep's
                           ablations spill across multiple sweep ids
                           (e.g. cell-codebook ablation in s51,
                           niche-codebook ablation in s52).
      - "":                disable the filter, include every subdir.
      - None:              also disables the filter.

    Pass an explicit `variants` list to bypass the filter entirely —
    the filter only applies during auto-discovery.
    """
    dataset_root = artifacts_root / dataset
    if not dataset_root.is_dir():
        raise SystemExit(
            f"No artifacts dir at {dataset_root}. Override with "
            f"--artifacts-root or --dataset."
        )

    # Normalise: empty string / empty sequence / None all -> "no filter".
    # A single string and a sequence-of-strings are unified into a tuple
    # so the rest of the function can do `any(name.startswith(p) for ...)`
    # uniformly.
    prefixes: Optional[Tuple[str, ...]]
    if not prefix_filter:
        prefixes = None
    elif isinstance(prefix_filter, str):
        prefixes = (prefix_filter,)
    else:
        prefixes = tuple(p for p in prefix_filter if p)
        if not prefixes:
            prefixes = None

    variants_filtered_out: List[str] = []
    if variants is None:
        # Auto-discover: every subdir under dataset_root that contains
        # at least one timestamp subdir.
        all_subdirs = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
        if prefixes is not None:
            variants = [
                v for v in all_subdirs
                if any(v.startswith(p) for p in prefixes)
            ]
            variants_filtered_out = [
                v for v in all_subdirs
                if not any(v.startswith(p) for p in prefixes)
            ]
        else:
            variants = all_subdirs

    niche_frames:    List[pd.DataFrame] = []
    batchint_frames: List[pd.DataFrame] = []
    pearson_frames:  List[pd.DataFrame] = []
    variants_found:           List[str] = []
    variants_missing_files:   List[Tuple[str, List[str]]] = []
    variants_no_run_dir:      List[str] = []

    for variant in variants:
        variant_dir = dataset_root / variant
        run_dir = _resolve_run_dir(variant_dir, timestamp_strategy)
        if run_dir is None:
            variants_no_run_dir.append(variant)
            continue

        niche_df, batchint_df, pearson_df, missing = (
            _load_metrics_for_variant(run_dir)
        )
        if missing:
            variants_missing_files.append((variant, missing))
        if niche_df is None and batchint_df is None and pearson_df is None:
            # Nothing usable.
            continue

        variants_found.append(variant)
        if niche_df is not None:
            niche_df = niche_df.copy()
            niche_df["variant"]   = variant
            niche_df["timestamp"] = run_dir.name
            niche_frames.append(niche_df)
        if batchint_df is not None:
            batchint_df = batchint_df.copy()
            batchint_df["variant"]   = variant
            batchint_df["timestamp"] = run_dir.name
            batchint_frames.append(batchint_df)
        if pearson_df is not None:
            pearson_df = pearson_df.copy()
            pearson_df["variant"]   = variant
            pearson_df["timestamp"] = run_dir.name
            pearson_frames.append(pearson_df)

    niche_long    = pd.concat(niche_frames,    ignore_index=True) if niche_frames    else pd.DataFrame()
    batchint_long = pd.concat(batchint_frames, ignore_index=True) if batchint_frames else pd.DataFrame()
    pearson_long  = pd.concat(pearson_frames,  ignore_index=True) if pearson_frames  else pd.DataFrame()

    info = {
        "variants_found":         variants_found,
        "variants_no_run_dir":    variants_no_run_dir,
        "variants_missing_files": variants_missing_files,
        "variants_filtered_out":  variants_filtered_out,
        # Echo back the prefix(es) actually used so the CLI's diagnostic
        # message can show e.g. "non-'s*' subdir(s)" instead of
        # always saying "dualvq". Always returned as a normalised tuple
        # (or None) so callers don't need to handle the str/sequence
        # ambiguity.
        "prefix_filter":          prefixes,
        # `variants_attempted` is every variant we TRIED to load (passes the
        # prefix filter / user's --variants list). Includes variants that
        # ended up with no metrics — used by `render_all` to keep them as
        # empty rows in the heatmaps so the reader can see at a glance
        # which configured variants are missing data.
        "variants_attempted":     list(variants),
    }
    return niche_long, batchint_long, pearson_long, info


# Plotting --------------------------------------------------------------------

def _save_dual(fig, out_path: Path, **kw) -> None:
    """Save the figure as both .png and .svg siblings."""
    out_path = Path(out_path)
    for ext in (".png", ".svg"):
        fig.savefig(out_path.with_suffix(ext), **kw)


# Y-axis label sizing -------------------------------------------------------
# Variant names in this codebase can be >150 characters (e.g. the s36/s37
# sweeps stack a dozen `+knob` chunks onto a 30-char `dualvq+...` stem).
# At the historical `fontsize=8`, anything past ~70 chars overflows the
# left axis margin and gets truncated by `bbox_inches="tight"` cropping
# the rendered image — the user sees ".../knn16+sampler16+mmb0-1b_smb..."
# instead of the full name.
#
# We expand the figure width to make room for the longest label, and
# drop the y-tick font size slightly. `bbox_inches="tight"` then crops
# any leftover whitespace, so the resulting PNG/SVG stays tight without
# truncating the names.
_VARIANT_TICK_FONTSIZE = 6     # was 8; ~12% denser, still legible on screen
_INCH_PER_CHAR_AT_TICK_FS = 0.045   # ~0.045 in/char at fontsize 6, DPI 150


def _variant_label_left_margin_inches(labels) -> float:
    """
    Width budget (in inches) for the left-side y-axis labels of any
    variants × ... plot. Scales with the longest label so the plot
    canvas always has room; floors at 2.0in so short-label sweeps
    don't get a cramped left margin.
    """
    max_chars = max((len(str(s)) for s in labels), default=0)
    return max(2.0, _INCH_PER_CHAR_AT_TICK_FS * max_chars + 0.4)


def _select_key_metric(
        niche_long: pd.DataFrame,
        batchint_long: pd.DataFrame,
        selector: str,
    ) -> pd.DataFrame:
    """
    Resolve a selector string to a (variant, value) DataFrame.

    Selector forms:
      "niche|<split>|<code_key>|<label_key>|<NMI|ARI>"
      "batchint|<emb_key>|<metric>"
    """
    parts = selector.split("|")
    if parts[0] == "niche":
        _, split, code_key, label_key, score_col = parts
        df = niche_long
        if df.empty:
            return pd.DataFrame(columns=["variant", "value"])
        m = (
            (df["split"]     == split)
            & (df["code_key"]  == code_key)
            & (df["label_key"] == label_key)
        )
        out = df.loc[m, ["variant", score_col]].rename(columns={score_col: "value"})
    elif parts[0] == "batchint":
        _, emb_key, metric = parts
        df = batchint_long
        if df.empty:
            return pd.DataFrame(columns=["variant", "value"])
        m = (df["emb_key"] == emb_key) & (df["metric"] == metric)
        out = df.loc[m, ["variant", "score"]].rename(columns={"score": "value"})
    else:
        raise ValueError(f"unknown selector kind: {parts[0]!r}")
    return out.dropna(subset=["value"]).reset_index(drop=True)


def _plot_key_metric(
        df: pd.DataFrame,
        title: str,
        out_path: Path,
        higher_is_better: bool = True,
    ) -> None:
    """Horizontal bar chart, variants sorted by metric value."""
    if df.empty:
        return
    df = df.sort_values("value", ascending=higher_is_better).reset_index(drop=True)
    n = len(df)
    # Width scales with the longest variant name so long s37-style names
    # (up to ~150 chars) don't overflow the left margin.
    left_margin = _variant_label_left_margin_inches(df["variant"].astype(str))
    width = max(10.0, left_margin + 6.0)
    fig, ax = plt.subplots(figsize=(width, max(2.5, 0.32 * n + 1.0)))
    colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, n))
    bars = ax.barh(df["variant"].astype(str), df["value"].astype(float), color=colours)
    for bar, val in zip(bars, df["value"].astype(float)):
        ax.text(
            bar.get_width() + (df["value"].max() * 0.005 if df["value"].max() > 0 else 0.0),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            ha="left", va="center", fontsize=8,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("value")
    ax.tick_params(axis="y", labelsize=_VARIANT_TICK_FONTSIZE)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    _save_dual(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
        wide: pd.DataFrame,
        title: str,
        out_path: Path,
        cbar_label: str = "value",
        cmap: str = "viridis",
        highlight_columns: Optional[List[str]] = None,
        highlight_color: str = "#d62728",
    ) -> None:
    """
    Variants × metric_columns heatmap with annotated cells.

    `highlight_columns`: column-name substrings (matched against
    `wide.columns`) to call out — drawn with a thick coloured frame
    around the column and a bold tick label.
    """
    if wide.empty:
        return
    # Width scales with: (a) number of metric columns (existing), AND
    # (b) the longest variant name on the y-axis so long s37-style
    # names (up to ~150 chars) don't overflow the left margin and get
    # cropped by `bbox_inches="tight"` on save.
    left_margin = _variant_label_left_margin_inches(wide.index.astype(str))
    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * len(wide.columns) + left_margin + 2.0),
                 max(3, 0.32 * len(wide.index) + 1.5)),
    )
    im = ax.imshow(wide.values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(wide.columns)))
    ax.set_xticklabels(wide.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(wide.index)))
    ax.set_yticklabels(wide.index, fontsize=_VARIANT_TICK_FONTSIZE)
    # Mean is used only for the text-vs-background contrast heuristic;
    # an all-NaN matrix (when every reindexed variant is missing data)
    # would otherwise trigger a RuntimeWarning and emit NaN colours.
    finite_vals = wide.values[np.isfinite(wide.values)]
    cell_text_thresh = float(finite_vals.mean()) if finite_vals.size else 0.0
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            v = wide.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if v < cell_text_thresh else "black")

    # Frame priority columns and bold their tick labels. Use a generous
    # rectangle that hugs the column gridline; matplotlib data
    # coordinates for `imshow` are pixel-centred so the column spans
    # x in [j-0.5, j+0.5] and y in [-0.5, n_rows-0.5].
    if highlight_columns:
        n_rows = wide.shape[0]
        cols = list(wide.columns.astype(str))
        x_labels = ax.get_xticklabels()
        for col_name in highlight_columns:
            if col_name not in cols:
                continue
            j = cols.index(col_name)
            # Bold + colour the tick label.
            x_labels[j].set_fontweight("bold")
            x_labels[j].set_color(highlight_color)
            # Border around the column (clip_on=False so the top/bottom
            # edges aren't trimmed at the axis frame).
            rect = plt.Rectangle(
                (j - 0.5, -0.5), 1.0, n_rows,
                linewidth=2.5, edgecolor=highlight_color, facecolor="none",
                clip_on=False, zorder=5,
            )
            ax.add_patch(rect)

    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.025, pad=0.02)
    fig.tight_layout()
    _save_dual(fig, out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_all(
        niche_long: pd.DataFrame,
        batchint_long: pd.DataFrame,
        pearson_long: pd.DataFrame,
        out_dir: Path,
        all_variants: Optional[List[str]] = None,
    ) -> None:
    """Top-level renderer: writes summary CSV + key-metric bars + heatmaps.

    `all_variants`: when provided, every heatmap is REINDEXED to this
    full list of variant names. Variants that have no metric rows in
    the long tables show up as all-NaN rows (rendered as blank cells)
    rather than being dropped — useful when some runs are still in
    flight but you want to see at a glance which configured variants
    haven't produced metrics yet. Pass `info["variants_attempted"]`
    from `load_all_variants` for the default behaviour.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir = out_dir / "key_metrics"
    key_dir.mkdir(exist_ok=True)

    def _reindex_to_all(wide: pd.DataFrame) -> pd.DataFrame:
        """If `all_variants` was provided, expand `wide` to include any
        missing variants as all-NaN rows. Preserves the user's intended
        ordering (the `all_variants` list order); variants already in
        `wide` but not in `all_variants` are appended at the end so we
        never silently drop data."""
        if not all_variants:
            return wide
        present = list(wide.index.astype(str))
        full = list(all_variants) + [v for v in present if v not in all_variants]
        return wide.reindex(full)

    # Long-format summary CSV (one row per metric/variant).
    summary_rows: List[pd.DataFrame] = []
    if not niche_long.empty:
        cols = ["variant", "timestamp", "split", "code_key", "label_key",
                "NMI", "ARI", "n_cells"]
        existing = [c for c in cols if c in niche_long.columns]
        n = niche_long[existing].copy()
        n["table"] = "niche_identification"
        summary_rows.append(n)
    if not batchint_long.empty:
        cols = ["variant", "timestamp", "emb_key", "metric", "score"]
        existing = [c for c in cols if c in batchint_long.columns]
        b = batchint_long[existing].copy()
        b["table"] = "batch_integration"
        summary_rows.append(b)
    if not pearson_long.empty:
        cols = ["variant", "timestamp", "split", "branch", "axis",
                "transform", "gene_subset",
                "pearson_mean", "pearson_median", "n_cells", "n_genes"]
        existing = [c for c in cols if c in pearson_long.columns]
        p = pearson_long[existing].copy()
        p["table"] = "pearson_reconstruction"
        summary_rows.append(p)
    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True, sort=False)
        summary.to_csv(out_dir / "summary_long.csv", index=False)
        print(f"  -> wrote {out_dir / 'summary_long.csv'}")

    # Key-metric bars (the user's headline list).
    print("\nHeadline metrics:")
    for selector, label in KEY_METRICS:
        df = _select_key_metric(niche_long, batchint_long, selector)
        higher_is_better = "MMD" not in selector  # MMD: lower is better
        if df.empty:
            print(f"  [skip] {label}: no rows match selector {selector!r}")
            continue
        slug = (
            label.replace("/", "_").replace(" ", "_").replace("—", "-")
        )
        _plot_key_metric(
            df, title=label, out_path=key_dir / slug,
            higher_is_better=higher_is_better,
        )
        print(f"  -> {label}  (n={len(df)} variants)")

    # NMI / ARI heatmaps (split=all only).
    if not niche_long.empty:
        all_split = niche_long[niche_long["split"] == "all"].copy()
        all_split["pair"] = (
            all_split["code_key"].astype(str) + "  →  " + all_split["label_key"].astype(str)
        )
        nmi_wide = all_split.pivot_table(
            index="variant", columns="pair", values="NMI", aggfunc="mean",
        )
        ari_wide = all_split.pivot_table(
            index="variant", columns="pair", values="ARI", aggfunc="mean",
        )
        nmi_wide = _reindex_to_all(nmi_wide)
        ari_wide = _reindex_to_all(ari_wide)
        if not nmi_wide.empty:
            _plot_heatmap(
                nmi_wide,
                title="NMI  —  variants × (code → label)  (split=all)",
                out_path=out_dir / "nmi_heatmap",
                cbar_label="NMI",
                cmap="viridis",
                highlight_columns=PRIORITY_NICHE_PAIRS,
            )
            print(f"  -> {out_dir / 'nmi_heatmap.png'}")
        if not ari_wide.empty:
            _plot_heatmap(
                ari_wide,
                title="ARI  —  variants × (code → label)  (split=all)",
                out_path=out_dir / "ari_heatmap",
                cbar_label="ARI",
                cmap="viridis",
                highlight_columns=PRIORITY_NICHE_PAIRS,
            )
            print(f"  -> {out_dir / 'ari_heatmap.png'}")
    elif all_variants:
        # No niche_long rows at all but the user wanted heatmaps anyway —
        # render blank-heatmap placeholders so the missing variants are
        # visible in the figure rather than only in stdout.
        blank = pd.DataFrame(
            index=list(all_variants),
            columns=[PRIORITY_NICHE_PAIRS[0]] if PRIORITY_NICHE_PAIRS else ["(no data)"],
            dtype=float,
        )
        _plot_heatmap(
            blank,
            title="NMI  —  variants × (code → label)  (split=all)  [no data]",
            out_path=out_dir / "nmi_heatmap",
            cbar_label="NMI", cmap="viridis",
            highlight_columns=PRIORITY_NICHE_PAIRS,
        )
        print(f"  -> {out_dir / 'nmi_heatmap.png'} (all blank)")

    # Batch-integration heatmap.
    if not batchint_long.empty:
        bi = batchint_long.copy()
        bi["pair"] = bi["emb_key"].astype(str) + "  —  " + bi["metric"].astype(str)
        bi_wide = bi.pivot_table(
            index="variant", columns="pair", values="score", aggfunc="mean",
        )
        bi_wide = _reindex_to_all(bi_wide)
        if not bi_wide.empty:
            _plot_heatmap(
                bi_wide,
                title="Batch integration  —  variants × (emb × metric)",
                out_path=out_dir / "batch_integration",
                cbar_label="score",
                cmap="viridis",
                highlight_columns=PRIORITY_BATCHINT_PAIRS,
            )
            print(f"  -> {out_dir / 'batch_integration.png'}")

    # Pearson reconstruction heatmap (split=all only, pearson_mean).
    if not pearson_long.empty:
        # Reproducibility-runs sometimes only have split=all populated;
        # fall back to whatever's there if "all" isn't.
        if "split" in pearson_long.columns and (pearson_long["split"] == "all").any():
            pe = pearson_long[pearson_long["split"] == "all"].copy()
        else:
            pe = pearson_long.copy()
        for col in ("branch", "axis", "transform", "gene_subset"):
            if col not in pe.columns:
                pe[col] = "?"
        pe["pair"] = (
            pe["branch"].astype(str)    + " · " +
            pe["axis"].astype(str)      + " · " +
            pe["transform"].astype(str) + " · " +
            pe["gene_subset"].astype(str)
        )
        pe_wide = pe.pivot_table(
            index="variant", columns="pair", values="pearson_mean",
            aggfunc="mean",
        )
        # Order columns so priority cols are leftmost (others by name).
        if not pe_wide.empty:
            ordered = [c for c in PRIORITY_PEARSON_COLS if c in pe_wide.columns]
            ordered += sorted(c for c in pe_wide.columns if c not in ordered)
            pe_wide = pe_wide[ordered]
            pe_wide = _reindex_to_all(pe_wide)
            _plot_heatmap(
                pe_wide,
                title="Pearson reconstruction  —  variants × "
                      "(branch · axis · transform · gene_subset)  "
                      "[higher = better]",
                out_path=out_dir / "pearson_reconstruction",
                cbar_label="pearson_mean",
                cmap="viridis",
                highlight_columns=PRIORITY_PEARSON_COLS,
            )
            print(f"  -> {out_dir / 'pearson_reconstruction.png'}")


# CLI -------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT,
                   help=f"Root containing <dataset>/<variant>/<timestamp>/. "
                        f"Default: {DEFAULT_ARTIFACTS_ROOT}")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                   help=f"Dataset short tag. Default: {DEFAULT_DATASET}")
    p.add_argument("--variants", type=str, default=None,
                   help="Optional comma-separated list of variant names. "
                        "Default: every subdir of <artifacts-root>/<dataset>/ "
                        "whose name starts with the --prefix string. "
                        "Explicit --variants always bypasses the prefix "
                        "filter.")
    p.add_argument("--prefix", type=str, default="dualvq",
                   help="When auto-discovering (i.e. --variants not "
                        "passed), only include subdirs whose name starts "
                        "with this prefix. Common values: 'dualvq' "
                        "(default — historic SQUINT ablation runs), 's' "
                        "(sweep-aliased variants under the s<sweep>_v<N>_ "
                        "convention introduced in sweep 17), 'baseline-' "
                        "(baselines only). Multiple comma-separated "
                        "prefixes are OR'd (e.g. '--prefix s51,s52' "
                        "keeps subdirs starting with EITHER s51 OR s52). "
                        "Pass --include-baselines or --prefix '' to "
                        "disable the filter entirely. No effect when "
                        "--variants is passed explicitly.")
    p.add_argument("--include-baselines", action="store_true",
                   help="When auto-discovering (i.e. --variants not "
                        "passed), DISABLE the --prefix filter entirely — "
                        "include every subdir (baselines, smoke tests, "
                        "region-holdout, sweep aliases, all). Equivalent "
                        "to --prefix ''. Overrides --prefix.")
    p.add_argument("--timestamp-strategy", type=str, default="latest",
                   help="'latest' to pick the most recent timestamp per "
                        "variant, or an exact YYYYMMDD_HHMMSS string. "
                        "Default: latest.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write summary CSV + plots. "
                        "Default: <artifacts-root>/ablation_summaries/"
                        "<dataset>/<NOW>/.")
    args = p.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = (
            args.artifacts_root / "ablation_summaries" / args.dataset / ts
        )

    variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants else None
    )

    print(f"Artifacts root : {args.artifacts_root}")
    print(f"Dataset        : {args.dataset}")
    print(f"Out dir        : {args.out_dir}")
    print()

    # Resolve the effective prefix filter:
    #   --include-baselines wins over --prefix (sets filter to None).
    #   --prefix "" also disables the filter (treated as None inside
    #   load_all_variants).
    #   --prefix "s51,s52" splits into a tuple — load_all_variants
    #   keeps subdirs starting with ANY of the listed prefixes.
    effective_prefix: Optional[Union[str, Tuple[str, ...]]]
    if args.include_baselines:
        effective_prefix = None
    elif not args.prefix:
        effective_prefix = None
    else:
        parts = tuple(p.strip() for p in args.prefix.split(",") if p.strip())
        if len(parts) == 0:
            effective_prefix = None
        elif len(parts) == 1:
            effective_prefix = parts[0]
        else:
            effective_prefix = parts

    niche_long, batchint_long, pearson_long, info = load_all_variants(
        artifacts_root     = args.artifacts_root,
        dataset            = args.dataset,
        variants           = variants,
        timestamp_strategy = args.timestamp_strategy,
        prefix_filter      = effective_prefix,
    )

    print(f"Variants with metrics      : {len(info['variants_found'])}")
    for v in info["variants_found"]:
        print(f"  {v}")

    if info.get("variants_filtered_out"):
        used_prefixes = info.get("prefix_filter") or ()
        used_str = " | ".join(f"{p}*" for p in used_prefixes) if used_prefixes else ""
        print(f"\nVariants filtered out      : "
              f"{len(info['variants_filtered_out'])} subdir(s) not matching "
              f"'{used_str}' "
              f"(pass --include-baselines or --prefix '' to keep them, or "
              f"--prefix <other[,other2,...]> to use a different filter)")
        for v in info["variants_filtered_out"]:
            print(f"  {v}")

    if info["variants_no_run_dir"]:
        print(f"\nVariants WITHOUT run dirs  : "
              f"{len(info['variants_no_run_dir'])}")
        for v in info["variants_no_run_dir"]:
            print(f"  {v}")

    if info["variants_missing_files"]:
        print(f"\nVariants with MISSING metric files :")
        for v, paths in info["variants_missing_files"]:
            print(f"  {v}")
            for path in paths:
                print(f"    - {path}")

    if niche_long.empty and batchint_long.empty and pearson_long.empty:
        raise SystemExit(
            "\nNo metrics loaded — nothing to render. Check that runs "
            "actually completed the metrics step."
        )

    print()
    # Pass `variants_attempted` so the heatmaps reserve an empty row for
    # every configured variant — including those that haven't finished
    # producing metrics yet — instead of silently dropping them.
    render_all(
        niche_long, batchint_long, pearson_long, args.out_dir,
        all_variants=info.get("variants_attempted"),
    )
    print(f"\nDone. Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
