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
        Priority columns (iLISI / MMD on cell_latent and
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
  # Default: scan every variant's most recent timestamp.
  python analysis/ablations/compare_variants.py

  # Restrict to specific variants:
  python analysis/ablations/compare_variants.py \\
      --variants 'dualvq+rvq-both+decoder-cov+adv+mmb0-1b_smb1-1b_1p,\\
                  dualvq+rvq-both+decoder-cov+adv+gatv2+mmb0-1b_smb1-1b_1p'

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
from typing import List, Optional, Tuple

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
    "neighborhood_latent  —  MMD",
    "cell_latent  —  iLISI",
    "cell_latent  —  MMD",
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    For every variant under `<artifacts_root>/<dataset>/`, load all
    metric CSVs (niche identification, batch integration, pearson
    reconstruction) and return them as long-format DataFrames tagged
    with `variant` and `timestamp`.

    Returns (niche_long, batchint_long, pearson_long, info_dict).
    `info_dict` carries diagnostic info: variants_found,
    variants_missing_files (list of paths), variants_no_run_dir
    (variants with no timestamp subdir).
    """
    dataset_root = artifacts_root / dataset
    if not dataset_root.is_dir():
        raise SystemExit(
            f"No artifacts dir at {dataset_root}. Override with "
            f"--artifacts-root or --dataset."
        )

    if variants is None:
        # Auto-discover: every subdir under dataset_root that contains
        # at least one timestamp subdir.
        variants = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())

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
    }
    return niche_long, batchint_long, pearson_long, info


# Plotting --------------------------------------------------------------------

def _save_dual(fig, out_path: Path, **kw) -> None:
    """Save the figure as both .png and .svg siblings."""
    out_path = Path(out_path)
    for ext in (".png", ".svg"):
        fig.savefig(out_path.with_suffix(ext), **kw)


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
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.32 * n + 1.0)))
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
    ax.tick_params(axis="y", labelsize=8)
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
    fig, ax = plt.subplots(
        figsize=(max(6, 0.6 * len(wide.columns) + 4),
                 max(3, 0.32 * len(wide.index) + 1.5)),
    )
    im = ax.imshow(wide.values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(wide.columns)))
    ax.set_xticklabels(wide.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(wide.index)))
    ax.set_yticklabels(wide.index, fontsize=8)
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            v = wide.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if v < (np.nanmean(wide.values)) else "black")

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
    ) -> None:
    """Top-level renderer: writes summary CSV + key-metric bars + heatmaps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key_dir = out_dir / "key_metrics"
    key_dir.mkdir(exist_ok=True)

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

    # Batch-integration heatmap.
    if not batchint_long.empty:
        bi = batchint_long.copy()
        bi["pair"] = bi["emb_key"].astype(str) + "  —  " + bi["metric"].astype(str)
        bi_wide = bi.pivot_table(
            index="variant", columns="pair", values="score", aggfunc="mean",
        )
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
                        "Default: all subdirs of <artifacts-root>/<dataset>/.")
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

    niche_long, batchint_long, pearson_long, info = load_all_variants(
        artifacts_root     = args.artifacts_root,
        dataset            = args.dataset,
        variants           = variants,
        timestamp_strategy = args.timestamp_strategy,
    )

    print(f"Variants with metrics      : {len(info['variants_found'])}")
    for v in info["variants_found"]:
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
    render_all(niche_long, batchint_long, pearson_long, args.out_dir)
    print(f"\nDone. Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
