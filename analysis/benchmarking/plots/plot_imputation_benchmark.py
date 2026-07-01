#!/usr/bin/env python3
"""
Spatial-IMPUTATION benchmark figure — a separate task from the reconstruction
benchmark (`plot_pearson_benchmark.py`), so it gets its own figure.

Task: predict a held-out region's per-cell expression from spatial context, with
the held-out cells' expression NEVER seen. Distinct from reconstruction (where
each method encodes the cell's own measured expression), hence a dedicated plot.

Bars (cell-level only; both methods are cell-branch):
  - SQUINT (imputed)  codes predicted from spatial context (expression unseen).
  - GeST (imputed)    reimplemented GeST baseline (Hao et al. MLCB 2025).
  (- SQUINT (recon) is shown in the reconstruction figure, not here.)

Each method's metrics are located flexibly via `--*-path`, which accepts:
  * a direct per_seed_pearson_reconstruction.csv / pearson_reconstruction_metrics.csv,
  * a run dir (searched, incl. a `metrics/` subdir and `<TS>/metrics/`),
  * or a variant name under <artifacts_root>/<dataset_tag>/.
This handles the SQUINT-imputed CSV living under the stage-1 run's
`.../<TS>/stage2/<TS2>/metrics/` rather than a top-level variant dir.

Pearson is computed by the SAME util as the reconstruction benchmark (read from
the CSV), on the TEST split (held-out region). Reuses plot_pearson_benchmark's
panel renderer / style.

Outputs (to --out-dir, default <artifacts>/benchmarking/figures/):
  <out_prefix>_<split>.{svg,png}     one figure, 1 x len(METRIC_VARIANTS) panels
  <out_prefix>.csv                   long-format values behind the figure
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from plot_pearson_benchmark import (  # noqa: E402  (reuse the tested machinery)
    DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATASET_TAG, METRIC_VARIANTS,
    _plot_panel, _apply_nature_style,
)

# SQUINT (imputed) = the GeST-architecture transformer trained on SQUINT's
# discrete codes (best of the stage-2 imputation variants on held-out regions),
# decoded back to expression through the frozen SQUINT decoder. Using the SAME
# (GeST) architecture for both bars isolates the REPRESENTATION being imputed —
# SQUINT codes vs GeST meta-cell tokens — rather than confounding it with the
# stage-2 architecture. Override with --squint-imputed-path (variant / dir / CSV);
# the native MaskGIT stage-2 lives at "squint-imputed+region-holdout".
DEFAULT_SQUINT_IMPUTED = "squint-gestarch+region-holdout"   # variant OR a path
DEFAULT_GEST_IMPUTED = "gest-imputed+region-holdout"

COLOURS = {"SQUINT (imputed)": "#FF7AB6", "GeST (imputed)": "#2A9D8F",
           "SQUINT (recon)": "#FF006E"}
_CSV_NAMES = ("per_seed_pearson_reconstruction.csv", "pearson_reconstruction_metrics.csv")


def _resolve_metrics_csv(locator: str, artifacts_root: Path, dataset_tag: str):
    """Find a Pearson CSV from a file path, a run/metrics dir, or a variant name."""
    p = Path(locator)
    if p.is_file():
        return p
    search: List[Path] = []
    if p.is_dir():
        search += [p, p / "metrics"]
        # newest <TS>/metrics/ and <TS>/ underneath (handles stage2/<TS>/metrics)
        search += sorted([d / "metrics" for d in p.glob("*") if (d / "metrics").is_dir()],
                         reverse=True)
        search += sorted([d for d in p.glob("*") if d.is_dir()], reverse=True)
    else:
        vdir = artifacts_root / dataset_tag / locator      # variant under artifacts
        if vdir.is_dir():
            search += sorted([d / "metrics" for d in vdir.glob("*") if (d / "metrics").is_dir()],
                             reverse=True)
            search += [vdir / "metrics", vdir]
    for d in search:
        for n in _CSV_NAMES:
            if (d / n).is_file():
                return d / n
    return None


def _resolve_metrics_csvs(locator: str, artifacts_root: Path, dataset_tag: str):
    """ALL per-seed metrics CSVs for a method (concatenated by _load_csv_filtered).

    Handles the stage2-ablation layout where each seed is its OWN run dir
    (`<TS>_seedN/metrics/per_seed_pearson_reconstruction.csv`, one seed each) —
    _resolve_metrics_csv returns only the first, so SQUINT showed a single dot.
    Globs one level of subdirs; falls back to the single-CSV resolver for a
    file / aggregate (all seeds in one CSV) / single run dir."""
    p = Path(locator)
    if p.is_file():
        return [p]
    base = p if p.is_dir() else (artifacts_root / dataset_tag / locator)
    csvs: List[Path] = []
    if base.is_dir():
        for sub in sorted(base.glob("*")):          # ascending TS -> latest last
            if not sub.is_dir():
                continue
            for n in _CSV_NAMES:
                f = sub / "metrics" / n
                if f.is_file():
                    csvs.append(f)
                    break
    if csvs:
        return csvs
    single = _resolve_metrics_csv(locator, artifacts_root, dataset_tag)
    return [single] if single else []


def _load_csv_filtered(csv_path, axis: str, transform: str,
                       gene_subset: str, value_col: str = "pearson_mean",
                       branch: str = "cell"):
    """`branch` slice ('cell' = per-cell recon; 'niche' = neighborhood-level)
    as (seed, split, value) reading `value_col` (empty if that column/branch is
    absent — e.g. an OLD CSV, or a method with no neighborhood branch).

    `csv_path` may be a single path OR a list of per-seed CSVs — a list is read
    and concatenated (de-duplicated on (seed, split), keeping the latest path)
    so multi-seed-dir variants contribute every seed."""
    if isinstance(csv_path, (list, tuple)):
        frames = []
        for c in csv_path:
            if c is None:
                continue
            f = _load_csv_filtered(c, axis, transform, gene_subset, value_col, branch)
            if f.empty:
                continue
            # Derive the seed from the PATH (`..._seedN` / `seed_N`), overriding
            # the CSV's seed column. The stage2-ablation decode step doesn't pass
            # --seed, so every per-seed dir's CSV is stamped seed=0 — trusting it
            # would collapse all 5 seeds to one in the de-dup below. A single
            # aggregate CSV (no `seed` token in its path) keeps its own seeds.
            mm = re.findall(r"seed_?(\d+)", str(c))
            if mm:
                f = f.copy(); f["seed"] = int(mm[-1])
            frames.append(f)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return out.drop_duplicates(subset=["seed", "split"], keep="last")
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        return pd.DataFrame()
    if "seed" not in df.columns:
        df = df.copy(); df["seed"] = 0
    df = df[df["branch"] == branch]
    for col, val in (("axis", axis), ("transform", transform)):
        if col in df.columns:
            df = df[df[col] == val]
    if "gene_subset" in df.columns:
        if gene_subset == "hvg":          # match any hvg{N} (N varies w/ panel size)
            df = df[df["gene_subset"].astype(str).str.startswith("hvg")]
        else:
            df = df[df["gene_subset"] == gene_subset]
    df = df[df[value_col].notna()]
    if df.empty:
        return pd.DataFrame()
    return df[["seed", "split", value_col]].rename(columns={value_col: "value"})


# Panel specs per --metric. Each entry: (suffix, axis, transform, gene_subset,
# value_col, title, higher_is_better). "panel" is the reviewer figure: one
# representative bar-group per complementary metric.
# "panel" = the full multi-metric GRID. Rows = metric family; columns = view.
# Covers BOTH gene-wise and cell-wise for the axis-based metrics, plus the
# entry-wise zero/nonzero recovery (no gene/cell axis). Each tuple:
# (suffix, axis, transform, gene_subset, value_col, title, higher_is_better).
# Suffixes must be unique (they key per_panel_values / the long CSV).
_GRID_ROWS = [
    [  # Pearson (log1p)
        ("pe_gw_log", "gene_wise", "log1p", "all",     "pearson_mean", "Pearson · gene-wise (log1p)", True),
        ("pe_cw_log", "cell_wise", "log1p", "all",     "pearson_mean", "Pearson · cell-wise (log1p)", True),
        ("pe_hv_log", "gene_wise", "log1p", "hvg",     "pearson_mean", "Pearson · HVG (log1p)",       True),
        ("pe_mk_log", "gene_wise", "log1p", "markers", "pearson_mean", "Pearson · markers (log1p)",   True),
    ],
    [  # Pearson (raw counts)
        ("pe_gw_raw", "gene_wise", "raw", "all",     "pearson_mean", "Pearson · gene-wise (counts)", True),
        ("pe_cw_raw", "cell_wise", "raw", "all",     "pearson_mean", "Pearson · cell-wise (counts)", True),
        ("pe_hv_raw", "gene_wise", "raw", "hvg",     "pearson_mean", "Pearson · HVG (counts)",       True),
        ("pe_mk_raw", "gene_wise", "raw", "markers", "pearson_mean", "Pearson · markers (counts)",   True),
    ],
    [  # Spearman (rank-based -> transform-invariant; log1p rows)
        ("sp_gw", "gene_wise", "log1p", "all",     "spearman_mean", "Spearman · gene-wise", True),
        ("sp_cw", "cell_wise", "log1p", "all",     "spearman_mean", "Spearman · cell-wise", True),
        ("sp_hv", "gene_wise", "log1p", "hvg",     "spearman_mean", "Spearman · HVG",       True),
        ("sp_mk", "gene_wise", "log1p", "markers", "spearman_mean", "Spearman · markers",   True),
    ],
    [  # RMSE (log1p)
        ("rm_gw_log", "gene_wise", "log1p", "all",     "rmse_mean", "RMSE · gene-wise (log1p)", False),
        ("rm_cw_log", "cell_wise", "log1p", "all",     "rmse_mean", "RMSE · cell-wise (log1p)", False),
        ("rm_hv_log", "gene_wise", "log1p", "hvg",     "rmse_mean", "RMSE · HVG (log1p)",       False),
        ("rm_mk_log", "gene_wise", "log1p", "markers", "rmse_mean", "RMSE · markers (log1p)",   False),
    ],
    [  # RMSE (raw counts; calibration)
        ("rm_gw_raw", "gene_wise", "raw", "all",     "rmse_mean", "RMSE · gene-wise (counts)", False),
        ("rm_cw_raw", "cell_wise", "raw", "all",     "rmse_mean", "RMSE · cell-wise (counts)", False),
        ("rm_hv_raw", "gene_wise", "raw", "hvg",     "rmse_mean", "RMSE · HVG (counts)",       False),
        ("rm_mk_raw", "gene_wise", "raw", "markers", "rmse_mean", "RMSE · markers (counts)",   False),
    ],
    [  # zero/nonzero recovery (entry-wise; no gene/cell axis)
        ("zn_au_all", "entrywise", "counts", "all",     "auroc_zero", "Zero/nonzero AUROC",           True),
        ("zn_au_mk",  "entrywise", "counts", "markers", "auroc_zero", "Zero/nonzero AUROC (markers)", True),
        ("zn_ap_all", "entrywise", "counts", "all",     "auprc_zero", "Zero/nonzero AUPRC",           True),
        ("zn_ap_mk",  "entrywise", "counts", "markers", "auprc_zero", "Zero/nonzero AUPRC (markers)", True),
    ],
]

# Single-metric breakdowns (--metric pearson/spearman/mse/rmse/zero_nonzero):
# one row of per-(axis,transform,gene_subset) panels. Tuple shape as above.
_PANEL_SPECS = {
    "pearson":  [(s, ax, tr, gs, "pearson_mean",  lbl, True)
                 for s, ax, tr, gs, lbl in METRIC_VARIANTS],
    "spearman": [(s, ax, tr, gs, "spearman_mean", lbl, True)
                 for s, ax, tr, gs, lbl in METRIC_VARIANTS],
    "mse":      [(s, ax, tr, gs, "mse_mean",      lbl, False)
                 for s, ax, tr, gs, lbl in METRIC_VARIANTS],
    "rmse":     [(s, ax, tr, gs, "rmse_mean",     lbl, False)
                 for s, ax, tr, gs, lbl in METRIC_VARIANTS],
    "zero_nonzero": [
        ("zero_nonzero_all",     "entrywise", "counts", "all",     "auroc_zero", "Zero/nonzero AUROC (all)",     True),
        ("zero_nonzero_markers", "entrywise", "counts", "markers", "auroc_zero", "Zero/nonzero AUROC (markers)", True),
    ],
}


def render_metric_grid(resolved, grid_rows, split, fig_base, long_csv_path,
                       suptitle, colours, branch="cell"):
    """Render a metric GRID (rows × cols) comparing methods, and write the long
    CSV. Reusable across tasks (imputation / reconstruction).

    resolved      {label: per_seed CSV Path or None}
    grid_rows     list of rows; each row a list of 7-tuples
                  (suffix, axis, transform, gene_subset, value_col, title, hib)
    split         which `split` column value to plot (e.g. 'test' or 'all')
    fig_base      output Path WITHOUT extension (.svg/.png appended)
    long_csv_path output Path for the tidy long CSV
    suptitle      figure suptitle
    colours       {method_label: hex}
    One FIXED method order across the grid (y-labels on column 0 only); each
    panel title carries one direction arrow (↑ higher-better / ↓ lower-better).
    """
    panels = [p for row in grid_rows for p in row]
    long_rows: List[dict] = []
    per_panel_values: Dict[str, Dict[str, np.ndarray]] = {}
    for suffix, axis, transform, gene_subset, value_col, _title, _hib in panels:
        vals: Dict[str, np.ndarray] = {}
        for label, csv in resolved.items():
            if csv is None:
                continue
            df = _load_csv_filtered(csv, axis, transform, gene_subset, value_col,
                                    branch=branch)
            if df.empty:
                continue
            sub = df[df["split"] == split]
            v = sub["value"].astype(float).to_numpy()
            if v.size == 0:
                continue
            vals[label] = v
            for vv, sd in zip(v, sub["seed"].astype(int)):
                long_rows.append({
                    "metric": value_col, "panel": suffix, "axis": axis,
                    "transform": transform, "gene_subset": gene_subset,
                    "branch": branch, "method": label, "split": split,
                    "seed": int(sd), "value": float(vv),
                })
        per_panel_values[suffix] = vals

    seen: List[str] = []
    for row in grid_rows:
        for (suffix, *_rest) in row:
            for m in per_panel_values.get(suffix, {}):
                if m not in seen:
                    seen.append(m)
    fv = per_panel_values.get(panels[0][0], {})
    method_order = sorted(
        seen, key=lambda m: -float(np.nanmean(fv.get(m, [np.nan]))) if m in fv else 0.0)

    nrows = len(grid_rows)
    ncols = max(len(r) for r in grid_rows)
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(1.9 * ncols + 1.0,
                 nrows * max(1.05, 0.28 * max(1, len(method_order)) + 0.45)))
    for r, row in enumerate(grid_rows):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(row):
                ax.set_visible(False)
                continue
            suffix, _axn, _tr, _gs, _vc, mlabel, hib = row[c]
            vals = per_panel_values.get(suffix, {})
            _plot_panel(ax=ax, method_order=method_order, per_method_values=vals,
                        colour_for=colours, title=mlabel)
            # _plot_panel hard-appends " (↑)"; override with the CORRECT single
            # direction arrow (↑ higher-better / ↓ lower-better).
            ax.set_title(f"{mlabel} ({'↑' if hib else '↓'})",
                         fontsize=7, fontweight="medium", pad=4)
            if c > 0:                      # method names only on the left column
                ax.tick_params(axis="y", labelleft=False)
    fig.suptitle(suptitle, fontsize=8, fontweight="medium", y=1.005)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18, hspace=0.55)

    fig_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = fig_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)
    if long_rows:
        pd.DataFrame(long_rows).to_csv(long_csv_path, index=False)
        print(f"  -> {long_csv_path}")
    return pd.DataFrame(long_rows)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out-prefix", type=str, default="imputation_benchmark")
    p.add_argument("--split", type=str, default="test",
                   help="Pearson split to plot (held-out region = 'test').")
    p.add_argument("--branch", type=str, default="cell", choices=["cell", "niche"],
                   help="'cell' = per-cell reconstruction; 'niche' = "
                        "neighborhood-level (X_hat_nbr vs X_nbr). For methods with "
                        "no native neighborhood branch the runner aggregates the "
                        "cell prediction over the spatial graph.")
    p.add_argument("--metric", type=str, default="panel",
                   choices=["panel", *_PANEL_SPECS],
                   help="Which figure to render. 'panel' (default) = the full "
                        "multi-metric GRID: Pearson / Spearman / RMSE (log1p+counts) "
                        "each gene-wise AND cell-wise, + marker Pearson + zero-nonzero "
                        "AUROC. Others render the per-(axis,transform,subset) "
                        "breakdown for a single metric. Needs CSVs rebuilt with the "
                        "new columns.")
    p.add_argument("--squint-imputed-path", type=str, default=DEFAULT_SQUINT_IMPUTED,
                   help="CSV / run dir / variant for SQUINT (imputed).")
    p.add_argument("--gest-imputed-path", type=str, default=DEFAULT_GEST_IMPUTED,
                   help="CSV / run dir / variant for GeST (imputed).")
    p.add_argument("--squint-recon-path", type=str, default=None,
                   help="Optional: add a SQUINT (recon) ceiling bar from this "
                        "CSV/dir/variant (off by default — it's in the recon figure).")
    p.add_argument("--squint-imputed-label", type=str, default="SQUINT (imputed)",
                   help="Display label for the --squint-imputed-path bar "
                        "(e.g. 'SQUINT (MC)').")
    p.add_argument("--extra-variant", nargs=2, action="append",
                   metavar=("PATH", "LABEL"), default=None,
                   help="Add an EXTRA imputed bar. PATH = CSV / run dir / variant; "
                        "LABEL = display name (e.g. 'SQUINT (No MC)'). Repeatable — "
                        "use to compare decode configs side by side.")
    p.add_argument("--gest-imputed-label", type=str, default="GeST",
                   help="Display label for the GeST bar (default 'GeST').")
    p.add_argument("--color", nargs=2, action="append",
                   metavar=("LABEL", "HEX"), default=None,
                   help="Override a bar's colour: LABEL = display name, HEX = "
                        "'#RRGGBB'. Repeatable (e.g. swap two bars' colours).")
    args = p.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = args.artifacts_root / "benchmarking" / "figures"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    squint_label = args.squint_imputed_label
    gest_label = args.gest_imputed_label
    methods: Dict[str, str] = {
        squint_label: args.squint_imputed_path,
        gest_label: args.gest_imputed_path,
    }
    if args.squint_recon_path:
        methods = {"SQUINT (recon)": args.squint_recon_path, **methods}
    # Extra imputed bars (e.g. a second SQUINT decode config to compare).
    for _path, _label in (args.extra_variant or []):
        methods[_label] = _path

    _apply_nature_style()
    import plot_pearson_benchmark as ppb
    # Colours: start from COLOURS; map the (possibly renamed) squint/gest bars to
    # their base shades, give each extra bar a distinct SQUINT-family shade, then
    # apply any explicit --color overrides LAST (so they win).
    _extra_shades = ["#B5179E", "#7209B7", "#F72585", "#4361EE"]
    colours = dict(COLOURS)
    colours.setdefault(squint_label, COLOURS["SQUINT (imputed)"])
    colours.setdefault(gest_label, COLOURS["GeST (imputed)"])
    for i, (_path, _label) in enumerate(args.extra_variant or []):
        colours.setdefault(_label, _extra_shades[i % len(_extra_shades)])
    for _label, _hex in (args.color or []):
        colours[_label] = _hex
    # Hatch all imputed bars (everything except the solid SQUINT (recon) ceiling).
    ppb.HATCH_METHODS |= (set(methods) - {"SQUINT (recon)"})

    # Resolve each method's per-seed CSV(s). ALL per-seed dirs are collected so
    # multi-seed-dir variants (stage2-ablation) contribute every seed, not one.
    resolved: Dict[str, List[Path]] = {}
    print(f"Artifacts root: {args.artifacts_root}\nDataset: {args.dataset_tag}\nSplit: {args.split}")
    for label, locator in methods.items():
        csvs = _resolve_metrics_csvs(locator, args.artifacts_root, args.dataset_tag)
        resolved[label] = csvs
        if csvs:
            print(f"  {label:<18s} <- {len(csvs)} CSV(s); e.g. {csvs[0]}")
        else:
            print(f"  {label:<18s} <- MISSING (from {locator!r})")

    grid_rows = _GRID_ROWS if args.metric == "panel" else [_PANEL_SPECS[args.metric]]
    blabel = {"cell": "cell-level", "niche": "neighborhood-level"}[args.branch]
    suptitle = (f"Spatial imputation — held-out region (expression unseen)  "
                f"[{blabel}, {args.split}]")
    render_metric_grid(
        resolved, grid_rows, args.split,
        fig_base=args.out_dir / f"{args.out_prefix}_{args.metric}_{args.branch}_{args.split}",
        long_csv_path=args.out_dir / f"{args.out_prefix}_{args.metric}_{args.branch}.csv",
        suptitle=suptitle, colours=colours, branch=args.branch)


if __name__ == "__main__":
    sys.exit(main())
