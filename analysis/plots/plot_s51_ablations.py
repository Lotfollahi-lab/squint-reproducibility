#!/usr/bin/env python3
"""
s51 ablation visualisations.

Reads `summary_long.csv` (produced by
`analysis/ablations/compare_variants.py`) and renders one figure per
ablation axis. Each figure is a 1×8 grid: 4 cell-side metrics +
4 niche-side metrics, all on one row.

    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Cell NMI │ │Cell ARI │ │Cell iLI │ │Cell MMD │ │Niche NMI│ │Niche ARI│ │Niche iLI│ │Niche MMD│
    │  (↑)   │ │  (↑)   │ │  (↑)   │ │  (↓)   │ │  (↑)   │ │  (↑)   │ │  (↑)   │ │  (↓)   │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘

For each axis the chosen (default-architecture) variant is rendered
in RED (`#FF006E`, the SQUINT accent) and the non-selected variants
in GREY (`#888888`, matching the "Vanilla VQ-VAE" colour in the
Pearson plots).

Default variant (RED in every axis):
  s51_v2_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32
    +knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec
    +decoupled-enc+diversity-w10
    +mmb0-1b_smb1-1b_1p

Layout choices:
  - 1 row × 8 panels.  All panels share the y-axis (`sharey=True`)
    so y-tick labels appear only on the LEFTMOST panel.
  - Bars are short + lean (low BAR_HEIGHT in axis-units + tight
    figure height) so the figure is compact enough to fit several
    side-by-side on a paper page.
  - All text is written as `<text>` elements (`svg.fonttype="none"`)
    so titles / labels are editable in Illustrator / Inkscape.

Axes (one figure each):
  1. ADJACENCY RECONSTRUCTION LOSS  — default vs s51_v3 (no-adj)
  2. CONTRASTIVE CELL LOSS          — default vs s51_v1 (with-contrastive)
  3. DECODER COVARIATE              — default vs s51_v4 (no-decoder-cov)
  4. GNN DEPTH                      — default vs s51_v11 (gnn-l2)
  5. NUMBER OF NEIGHBOURS           — default vs s51_v8 (knn8),
                                       s51_v9 (knn16+sampler8), s51_v10 (knn24)
  6. CELL CODEBOOK SIZE             — default vs s51_v6 (30×10),
                                       s51_v5 (30×30), s51_v7 (30×300)

Usage:
  python analysis/ablations/plots/plot_s51_ablations.py \\
      --summary-csv /nfs/team361/sb75/squint-reproducibility/artifacts/ablation_summaries/mmb0-1b_smb1-1b_1p/20260514_140209/summary_long.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SUMMARY_CSV = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts/"
    "ablation_summaries/mmb0-1b_smb1-1b_1p/20260514_140209/summary_long.csv"
)

# Figures land in a SHARED ablation-figures directory on the cluster
# so multiple ablation studies can be browsed alongside one another.
DEFAULT_OUT_DIR = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts/ablations/figures"
)

# Visual style — RED for the default architecture, GREY for the
# comparators. Picked to match the existing benchmarking plots.
COLOUR_DEFAULT = "#FF006E"
COLOUR_OTHER   = "#888888"

# Niche-side code_key for NMI/ARI (matches the niche-ID benchmark plot
# convention: `[level_0]` is the K1=30 macro partition, the headline
# NMI metric across the rest of the tooling).
CELL_CODE_KEY  = "cell_code_indices[level_0]"
NICHE_CODE_KEY = "neighborhood_code_indices[level_0]"

# Embedding keys for iLISI/MMD. By design we score the QUANTIZED
# embeddings (post-VQ), matching the cell-type-ID + niche-ID benchmark
# plot conventions.
CELL_EMB_KEY  = "cell_emb"
NICHE_EMB_KEY = "neighborhood_emb"

# Cell-type / niche label preference orders. For each variant the FIRST
# label_key with data is used (lets the same script handle multiple
# datasets without changes).
DEFAULT_CELL_LABEL_KEYS = (
    "cell_type",    # mmb mouse brain, chl59 CosMx Lung
    "cell_types",   # blob-canonical alias
    "annotation",   # spatch tissue subsets
)
DEFAULT_NICHE_LABEL_KEYS = (
    "niche",                        # chl59 CosMx Lung
    "Sub_molecular_tissue_region",  # mmb mouse brain
    "ccf_region_name",              # mmb mouse brain
    "spatial_cluster",              # spatch tissue subsets
)


# ---------------------------------------------------------------------------
# Axis specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantEntry:
    """One bar in an axis: a variant slug + the y-axis label to show."""
    prefix: str              # e.g. "s51_v2_" — matched via str.startswith
    label: str               # display label, e.g. "16 Neigh."
    is_default: bool = False # True -> RED; False -> GREY


@dataclass(frozen=True)
class AxisSpec:
    """One ablation axis -> one figure."""
    key: str                       # filename slug, e.g. "axis_1_adjacency"
    title: str                     # figure suptitle (Title Case)
    entries: Tuple[VariantEntry, ...]  # in plotted order, top-to-bottom


AXES: Tuple[AxisSpec, ...] = (
    AxisSpec(
        key="axis_1_adjacency",
        title="Adjacency Reconstruction Loss",
        entries=(
            VariantEntry("s51_v2_", "w. Adj. Recon.",  is_default=True),
            VariantEntry("s51_v3_", "w/o Adj. Recon."),
        ),
    ),
    AxisSpec(
        key="axis_2_contrastive",
        title="Contrastive Cell Loss",
        # User's chosen default DROPS the contrastive loss (s51_v2).
        # s51_v1 (with contrastive) is the comparator in grey.
        entries=(
            VariantEntry("s51_v1_", "w. Contrastive"),
            VariantEntry("s51_v2_", "w/o Contrastive", is_default=True),
        ),
    ),
    AxisSpec(
        key="axis_3_decoder_cov",
        title="Decoder Covariate",
        entries=(
            VariantEntry("s51_v2_", "w. Dec. Cov.",  is_default=True),
            VariantEntry("s51_v4_", "w/o Dec. Cov."),
        ),
    ),
    AxisSpec(
        key="axis_4_gnn_layers",
        title="GNN Depth",
        entries=(
            VariantEntry("s51_v2_",  "1 Layer",  is_default=True),
            VariantEntry("s51_v11_", "2 Layers"),
        ),
    ),
    AxisSpec(
        key="axis_5_neighbors",
        title="Number of Neighbours",
        # Ordered from sparsest to densest.
        # NB: the user's label list mentioned "12 Neigh." / "14 Neigh."
        # which don't correspond to any registered s51 variant — those
        # were a typo. The actual variants are 8 / 16 / 16-with-sampler-
        # 8 / 24 neighbours, which is what we label here.
        entries=(
            VariantEntry("s51_v8_",  "8 Neigh."),
            VariantEntry("s51_v9_",  "16 Neigh. (8 Sampled)"),
            VariantEntry("s51_v2_",  "16 Neigh.", is_default=True),
            VariantEntry("s51_v10_", "24 Neigh."),
        ),
    ),
    AxisSpec(
        key="axis_6_codebook_size",
        title="Cell Codebook Size",
        entries=(
            VariantEntry("s51_v6_", "(30, 10) Codebook"),
            VariantEntry("s51_v5_", "(30, 30) Codebook"),
            VariantEntry("s51_v2_", "(30, 90) Codebook", is_default=True),
            VariantEntry("s51_v7_", "(30, 300) Codebook"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _apply_nature_style() -> None:
    """rcParams matched to plot_niche_identification_benchmark.py so
    these ablation figures slot in next to the existing paper plots.
    `svg.fonttype="none"` keeps all titles / labels editable in
    Illustrator / Inkscape (no text-as-paths)."""
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

def _resolve_variants_for_axis(
        axis: AxisSpec,
        all_variants: List[str],
    ) -> Dict[str, str]:
    """Return {prefix: full_variant_name} for every entry in `axis`."""
    out: Dict[str, str] = {}
    for entry in axis.entries:
        matches = [v for v in all_variants if v.startswith(entry.prefix)]
        if not matches:
            raise SystemExit(
                f"[{axis.key}] no variant in summary_long.csv starts with "
                f"prefix {entry.prefix!r}."
            )
        if len(matches) > 1:
            raise SystemExit(
                f"[{axis.key}] prefix {entry.prefix!r} matches multiple "
                f"variants in summary_long.csv: {matches}."
            )
        out[entry.prefix] = matches[0]
    return out


def _label_for_variant(
        niche_df: pd.DataFrame,
        full_variant: str,
        preference: Tuple[str, ...],
    ) -> Optional[str]:
    """Pick the first label_key in `preference` that has data for
    `full_variant`. Returns None if none match."""
    sub = niche_df[niche_df["variant"] == full_variant]
    if sub.empty:
        return None
    have = set(sub["label_key"].unique().tolist())
    for lk in preference:
        if lk in have:
            return lk
    return None


def _extract_metric(
        df: pd.DataFrame,
        full_variant: str,
        metric: str,
        cell_label_key: Optional[str] = None,
        niche_label_key: Optional[str] = None,
    ) -> Optional[float]:
    """Extract one scalar value for `(full_variant, metric)` from the
    long-format DataFrame.

    `metric` is one of:
      "Cell NMI",  "Cell ARI",  "Cell iLISI",  "Cell MMD",
      "Niche NMI", "Niche ARI", "Niche iLISI", "Niche MMD"
    """
    # ----- NMI / ARI (cell or niche side) -------------------------------
    if metric in ("Cell NMI", "Cell ARI", "Niche NMI", "Niche ARI"):
        is_cell = metric.startswith("Cell")
        code_key = CELL_CODE_KEY if is_cell else NICHE_CODE_KEY
        label_key = cell_label_key if is_cell else niche_label_key
        if label_key is None:
            return None
        sub = df[
            (df.get("table") == "niche_identification")
            & (df["variant"] == full_variant)
            & (df.get("split") == "all")
            & (df.get("code_key") == code_key)
            & (df.get("label_key") == label_key)
        ]
        col = "NMI" if metric.endswith("NMI") else "ARI"
        if sub.empty or col not in sub.columns:
            return None
        val = pd.to_numeric(sub[col], errors="coerce").dropna()
        return float(val.mean()) if not val.empty else None

    # ----- iLISI / MMD (cell or niche side) -----------------------------
    if metric in ("Cell iLISI", "Cell MMD", "Niche iLISI", "Niche MMD"):
        is_cell = metric.startswith("Cell")
        emb_key = CELL_EMB_KEY if is_cell else NICHE_EMB_KEY
        metric_tag = "iLISI" if metric.endswith("iLISI") else "MMD"
        sub = df[
            (df.get("table") == "batch_integration")
            & (df["variant"] == full_variant)
            & (df.get("emb_key") == emb_key)
            & (df.get("metric") == metric_tag)
        ]
        if sub.empty or "score" not in sub.columns:
            return None
        val = pd.to_numeric(sub["score"], errors="coerce").dropna()
        return float(val.mean()) if not val.empty else None

    raise ValueError(f"unknown metric {metric!r}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# 8 metric panels, in left-to-right order. Title text + direction-of-merit
# arrow. Cell-side first (4), then niche-side (4) — mirrors the column
# order in the paper's compact figure layout.
METRICS: Tuple[Tuple[str, str], ...] = (
    ("Cell NMI",   "↑"),
    ("Cell ARI",   "↑"),
    ("Cell iLISI", "↑"),
    ("Cell MMD",   "↓"),
    ("Niche NMI",  "↑"),
    ("Niche ARI",  "↑"),
    ("Niche iLISI","↑"),
    ("Niche MMD",  "↓"),
)


def _plot_panel(
        ax,
        method_order: List[str],
        per_method_value: Dict[str, Optional[float]],
        colour_for: Dict[str, str],
        metric_label: str,
        direction: str,
        show_y_ticklabels: bool = True,
    ) -> None:
    n = len(method_order)
    # Lean bars: BAR_HEIGHT in axis-units. 0.45 leaves 55% of each row
    # as whitespace, which combined with the tight figure height below
    # makes each bar visually thin.
    BAR_HEIGHT = 0.45

    for j, method in enumerate(method_order):
        val = per_method_value.get(method)
        if val is None or not np.isfinite(val):
            ax.text(0.5, j, "n/a", va="center", ha="center",
                    fontsize=5.5, fontstyle="italic", color="0.5",
                    transform=ax.get_yaxis_transform())
            continue
        colour = colour_for.get(method, COLOUR_OTHER)
        ax.barh(j, val, height=BAR_HEIGHT,
                color=colour, edgecolor=colour,
                linewidth=0.5, alpha=0.55, zorder=2)

    ax.set_yticks(range(n))
    if show_y_ticklabels:
        ax.set_yticklabels(method_order, fontweight="medium")
    else:
        ax.set_yticklabels([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.xaxis.grid(True, linewidth=0.2, alpha=0.4, color="0.65", linestyle="--")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.set_title(f"{metric_label}  ({direction})",
                 fontsize=7, fontweight="medium", pad=3)


def render_axis(
        axis: AxisSpec,
        per_metric: Dict[str, Dict[str, Optional[float]]],
        labels: Dict[str, str],
        out_path_base: Path,
    ) -> None:
    """Render one axis figure: 1 row × 8 panels."""
    method_order = [labels[e.prefix] for e in axis.entries]
    colour_for = {
        labels[e.prefix]: (COLOUR_DEFAULT if e.is_default else COLOUR_OTHER)
        for e in axis.entries
    }

    n_methods = len(method_order)
    n_panels = len(METRICS)

    # Compact dimensions: each panel ~1.05 inches wide; total ~9.0 inches
    # including the left margin for y-tick labels (handled implicitly by
    # tight_layout). Height scales with method count but stays under
    # 1.6 inches even for 4 methods.
    panel_width_in = 1.05
    fig_width_in = panel_width_in * n_panels + 0.6
    fig_height_in = max(0.95, 0.26 * n_methods + 0.55)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_width_in, fig_height_in), sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for i, (metric_label, direction) in enumerate(METRICS):
        _plot_panel(
            ax=axes[i],
            method_order=method_order,
            per_method_value=per_metric[metric_label],
            colour_for=colour_for,
            metric_label=metric_label,
            direction=direction,
            # With sharey=True only the leftmost axis owns the tick
            # labels — but we set them explicitly here for safety in
            # case a future matplotlib changes the sharey default.
            show_y_ticklabels=(i == 0),
        )

    fig.suptitle(axis.title, fontsize=8, fontweight="medium", y=1.04)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)

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
    p.add_argument(
        "--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV,
        help="Path to summary_long.csv (default: %(default)s).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=(
            "Output directory (default: %(default)s). Six "
            "axis_<N>_<slug>.{svg,png,csv} files are written there."
        ),
    )
    p.add_argument(
        "--cell-label-keys", type=str,
        default=",".join(DEFAULT_CELL_LABEL_KEYS),
        help="Cell-type label preference order. For each variant the "
             "FIRST label_key with data is used. Default: %(default)s.",
    )
    p.add_argument(
        "--niche-label-keys", type=str,
        default=",".join(DEFAULT_NICHE_LABEL_KEYS),
        help="Niche label preference order. For each variant the "
             "FIRST label_key with data is used. Default: %(default)s.",
    )
    args = p.parse_args(argv)

    if not args.summary_csv.is_file():
        raise SystemExit(f"summary_long.csv not found at {args.summary_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _apply_nature_style()

    print(f"Reading: {args.summary_csv}")
    df = pd.read_csv(args.summary_csv)
    print(f"  shape: {df.shape}")
    if "variant" not in df.columns or "table" not in df.columns:
        raise SystemExit(
            "summary_long.csv must have 'variant' and 'table' columns; got "
            f"{list(df.columns)}"
        )
    all_variants = sorted(df["variant"].unique().tolist())
    print(f"  unique variants: {len(all_variants)}")
    print(f"  output dir: {args.out_dir}")

    cell_label_preference = tuple(
        k.strip() for k in args.cell_label_keys.split(",") if k.strip()
    )
    niche_label_preference = tuple(
        k.strip() for k in args.niche_label_keys.split(",") if k.strip()
    )
    niche_df = df[df.get("table") == "niche_identification"]

    # ---- Render each axis -------------------------------------------------
    for axis in AXES:
        print(f"\n=== {axis.key}: {axis.title} ===")
        variant_map = _resolve_variants_for_axis(axis, all_variants)
        labels = {e.prefix: e.label for e in axis.entries}

        default_prefix = next(e.prefix for e in axis.entries if e.is_default)
        default_full   = variant_map[default_prefix]
        cell_label  = _label_for_variant(
            niche_df, default_full, cell_label_preference,
        )
        niche_label = _label_for_variant(
            niche_df, default_full, niche_label_preference,
        )
        if cell_label is None:
            print(f"  WARN: no cell label_key from {cell_label_preference} "
                  f"available for the default variant. Cell NMI/ARI panels "
                  f"will show n/a.")
        else:
            print(f"  cell label used:  {cell_label!r}")
        if niche_label is None:
            print(f"  WARN: no niche label_key from {niche_label_preference} "
                  f"available for the default variant. Niche NMI/ARI panels "
                  f"will show n/a.")
        else:
            print(f"  niche label used: {niche_label!r}")

        per_metric: Dict[str, Dict[str, Optional[float]]] = {
            m: {} for m, _ in METRICS
        }
        for entry in axis.entries:
            full = variant_map[entry.prefix]
            for metric_label, _ in METRICS:
                v = _extract_metric(
                    df, full, metric_label,
                    cell_label_key=cell_label,
                    niche_label_key=niche_label,
                )
                per_metric[metric_label][entry.label] = v

        # --- Write a small per-axis CSV for downstream analysis -------
        records = []
        for entry in axis.entries:
            for metric_label, _ in METRICS:
                records.append({
                    "axis":            axis.key,
                    "prefix":          entry.prefix,
                    "variant":         variant_map[entry.prefix],
                    "label":           entry.label,
                    "is_default":      entry.is_default,
                    "metric":          metric_label,
                    "value":           per_metric[metric_label][entry.label],
                    "cell_label_key":  cell_label,
                    "niche_label_key": niche_label,
                    "cell_code_key":   CELL_CODE_KEY,
                    "niche_code_key":  NICHE_CODE_KEY,
                    "cell_emb_key":    CELL_EMB_KEY,
                    "niche_emb_key":   NICHE_EMB_KEY,
                })
        csv_out = args.out_dir / f"{axis.key}.csv"
        pd.DataFrame(records).to_csv(csv_out, index=False)
        print(f"  -> {csv_out}")

        render_axis(
            axis=axis,
            per_metric=per_metric,
            labels=labels,
            out_path_base=args.out_dir / axis.key,
        )

    print(f"\nAll 6 figures written to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
