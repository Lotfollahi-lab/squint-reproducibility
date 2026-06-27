#!/usr/bin/env python3
"""
Multi-seed ablation figures — same axes as plot_ablations.py, but each bar is
the MEAN across the training seeds with: the individual-seed DOTS overlaid, a
95% CONFIDENCE INTERVAL whisker, and a SIGNIFICANCE marker for each comparator
vs the axis default (red).

Unlike plot_ablations.py (which reads one mean value per variant from
summary_long.csv), this reads PER-SEED metrics directly from each variant's
multi-seed sweep — the same files the benchmark figures use:
    <ARTIFACTS_ROOT>/<dataset>/<variant>__multiseed/<latest_TS>/metrics/
        per_seed_niche_identification.csv   (NMI / ARI, by code_key + label_key)
        per_seed_batch_integration.csv      (iLISI / MMD, by emb_key)

The axis structure, metric panels, code/emb keys, label preferences, colours
and Nature style are imported from plot_ablations.py, so the axes are defined in
exactly one place (incl. the s51_v1 reference fix and the s54 L0 axes).

Run AFTER the ablation multi-seed sweeps finish
(squint/examples/submit_all_ablation_multiseed.sh) and their aggregators have
written per_seed_*.csv:
    python plot_ablations_multiseed.py
    python plot_ablations_multiseed.py --artifacts-root <...> --dataset mmb0-1b_smb1-1b_1p
    python plot_ablations_multiseed.py --test mannwhitney --error sem

Significance: each non-default bar is tested against the axis default
(Welch's t-test by default; --test mannwhitney for the non-parametric rank
test). Stars: *** p<1e-3, ** p<1e-2, * p<0.05, ns otherwise.

Requirements: pandas, numpy, matplotlib, scipy (scipy optional — CIs fall back
to the normal approx and p-values to NaN if it's missing).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import the axis definitions + shared constants from the single-seed plotter so
# there is ONE source of truth for the axes / panels / keys / style.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ablations import (  # noqa: E402
    AXES, METRICS,
    CELL_CODE_KEY, NICHE_CODE_KEY, CELL_EMB_KEY, NICHE_EMB_KEY,
    DEFAULT_CELL_LABEL_KEYS, DEFAULT_NICHE_LABEL_KEYS,
    COLOUR_DEFAULT, COLOUR_OTHER, DEFAULT_DATASET,
    _apply_nature_style,
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Where the <variant>__multiseed sweep dirs live (same root the benchmark
# plots use).
DEFAULT_ARTIFACTS_ROOT = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts"
)


# ---------------------------------------------------------------------------
# Per-seed data loading (from the __multiseed sweeps)
# ---------------------------------------------------------------------------
# Config-identical aliases: if a prefix has no __multiseed sweep, fall back to
# these. s51_v1 (the cross-axis reference in axes 1-7) IS the s49_v23 benchmark
# winner under a different key — so a figure can reuse an existing s49_v23 sweep
# for the reference slot instead of re-running s51_v1. Tries the primary prefix
# FIRST, then the alias.
REF_ALIASES: Dict[str, Tuple[str, ...]] = {
    "s51_v1_": ("s49_v23_",),
}


def _multiseed_metrics_dir(prefix: str, dataset: str,
                           artifacts_root: Path) -> Optional[Path]:
    """Resolve an axis-entry prefix (e.g. 's51_v5_') to the latest
    `<TS>/metrics` dir under `<root>/<dataset>/<prefix>*__multiseed/` that has
    per_seed_*.csv. Falls back to REF_ALIASES (config-identical variants) if the
    primary prefix has no sweep. Returns None if nothing is found."""
    base = artifacts_root / dataset
    if not base.is_dir():
        return None
    for pref in (prefix, *REF_ALIASES.get(prefix, ())):
        cands = sorted(d for d in base.glob(f"{pref}*__multiseed") if d.is_dir())
        if not cands:
            continue
        if len(cands) > 1:
            print(f"  WARN: prefix {pref!r} matched {len(cands)} __multiseed "
                  f"dirs; using {cands[0].name}", file=sys.stderr)
        for ts in sorted((p for p in cands[0].iterdir() if p.is_dir()),
                         key=lambda p: p.name, reverse=True):
            m = ts / "metrics"
            if m.is_dir() and (
                (m / "per_seed_niche_identification.csv").is_file()
                or (m / "per_seed_batch_integration.csv").is_file()
            ):
                if pref != prefix:
                    print(f"  (alias: using {pref!r} sweep for {prefix!r} "
                          f"-> {cands[0].name})")
                return m
    return None


def _pick_label_key(metrics_dir: Path, is_cell: bool,
                    preference: Tuple[str, ...]) -> Optional[str]:
    """First label_key in `preference` that has rows for the relevant code_key
    in this variant's per_seed_niche_identification.csv."""
    f = metrics_dir / "per_seed_niche_identification.csv"
    if not f.is_file():
        return None
    d = pd.read_csv(f)
    if "split" in d.columns:
        d = d[d["split"] == "all"]
    ck = CELL_CODE_KEY if is_cell else NICHE_CODE_KEY
    if "code_key" in d.columns:
        d = d[d["code_key"] == ck]
    have = set(d["label_key"].unique().tolist()) if "label_key" in d.columns else set()
    return next((lk for lk in preference if lk in have), None)


def _per_seed_values(metrics_dir: Path, metric: str,
                     cell_label: Optional[str],
                     niche_label: Optional[str]) -> np.ndarray:
    """Per-seed values for one (variant, metric). Empty array if unavailable."""
    if metric in ("Cell NMI", "Cell ARI", "Niche NMI", "Niche ARI"):
        f = metrics_dir / "per_seed_niche_identification.csv"
        if not f.is_file():
            return np.array([])
        d = pd.read_csv(f)
        if "split" in d.columns:
            d = d[d["split"] == "all"]
        is_cell = metric.startswith("Cell")
        ck = CELL_CODE_KEY if is_cell else NICHE_CODE_KEY
        lk = cell_label if is_cell else niche_label
        if lk is None:
            return np.array([])
        d = d[(d["code_key"] == ck) & (d["label_key"] == lk)]
        col = "NMI" if metric.endswith("NMI") else "ARI"
        if d.empty or col not in d.columns:
            return np.array([])
        return pd.to_numeric(d[col], errors="coerce").dropna().to_numpy()

    if metric in ("Cell iLISI", "Cell MMD", "Niche iLISI", "Niche MMD"):
        f = metrics_dir / "per_seed_batch_integration.csv"
        if not f.is_file():
            return np.array([])
        d = pd.read_csv(f)
        is_cell = metric.startswith("Cell")
        ek = CELL_EMB_KEY if is_cell else NICHE_EMB_KEY
        tag = "iLISI" if metric.endswith("iLISI") else "MMD"
        d = d[(d["emb_key"] == ek) & (d["metric"] == tag)]
        if d.empty or "score" not in d.columns:
            return np.array([])
        return pd.to_numeric(d["score"], errors="coerce").dropna().to_numpy()

    raise ValueError(f"unknown metric {metric!r}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _err_halfwidth(vals: np.ndarray, kind: str) -> float:
    """Error-bar half-width: 95% CI (default), SEM, or STD."""
    n = vals.size
    if n < 2:
        return 0.0
    sd = float(vals.std(ddof=1))
    if kind == "std":
        return sd
    sem = sd / np.sqrt(n)
    if kind == "sem":
        return sem
    # ci95
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        t = 1.96
    return t * sem


def _pvalue(a: np.ndarray, b: np.ndarray, test: str) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        from scipy import stats
    except Exception:
        return float("nan")
    try:
        if test == "mannwhitney":
            return float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])
        return float(stats.ttest_ind(a, b, equal_var=False)[1])  # Welch
    except Exception:
        return float("nan")


def _stars(p: float) -> str:
    if p != p:  # NaN
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _darken(colour, f: float = 0.62):
    """Darken a colour for the per-seed dots so they read against their bar."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(colour)
    return (r * f, g * f, b * f)


def _plot_panel(ax, method_order: List[str],
                per_method_values: Dict[str, np.ndarray],
                default_method: str, colour_for: Dict[str, str],
                metric_label: str, direction: str, test: str, err_kind: str,
                show_y_ticklabels: bool, zoom: bool = False) -> None:
    n = len(method_order)
    BAR_HEIGHT = 0.52
    DOT_SPREAD = 0.13

    def _clean(m):
        v = np.asarray(per_method_values.get(m, np.array([])), float)
        return v[np.isfinite(v)]

    default_vals = _clean(default_method)

    # Pre-pass: per-method (vals, mean, err) + panel extent, so the
    # significance stars can be aligned in a single column past the longest
    # bar and the x-limits leave room for them.
    stats = {}
    for m in method_order:
        v = _clean(m)
        stats[m] = (v, float(v.mean()), _err_halfwidth(v, err_kind)) if v.size else None
    ends = [max(mn + er, float(v.max())) for v, mn, er in (s for s in stats.values() if s)]
    lows = [min(mn - er, float(v.min())) for v, mn, er in (s for s in stats.values() if s)]
    max_end = max(ends) if ends else 1.0
    min_low = min(lows) if lows else 0.0
    span = max(max_end - min(min_low, 0.0), 1e-6)
    star_x = max_end + 0.05 * span

    # Faint reference line at the default's mean — shows at a glance which
    # comparators beat the reference.
    if stats.get(default_method) is not None:
        ax.axvline(stats[default_method][1], color="0.6", lw=0.6,
                   ls=(0, (3, 2)), alpha=0.8, zorder=1)

    for j, method in enumerate(method_order):
        s = stats[method]
        if s is None:
            ax.text(0.5, j, "n/a", va="center", ha="center", fontsize=5.5,
                    fontstyle="italic", color="0.55",
                    transform=ax.get_yaxis_transform())
            continue
        v, mean, err = s
        colour = colour_for.get(method, COLOUR_OTHER)
        ax.barh(j, mean, height=BAR_HEIGHT, color=colour, edgecolor="none",
                alpha=0.85, zorder=2)
        if err > 0:
            ax.errorbar(mean, j, xerr=err, fmt="none", ecolor="0.25",
                        elinewidth=0.8, capsize=2.0, capthick=0.6, zorder=3)
        # per-seed dots: deterministic value-ordered spread (no random jitter),
        # white-edged, darker shade of the bar colour -> crisp + tied to bar.
        if v.size > 1:
            yoff = np.empty(v.size)
            yoff[np.argsort(v)] = np.linspace(-DOT_SPREAD, DOT_SPREAD, v.size)
        else:
            yoff = np.zeros(1)
        ax.scatter(v, np.full(v.size, j, dtype=float) + yoff, s=11,
                   facecolor=_darken(colour), edgecolor="white",
                   linewidths=0.4, zorder=6)
        # significance vs the default — single aligned column, "ns" omitted.
        if method != default_method and default_vals.size >= 2 and v.size >= 2:
            st = _stars(_pvalue(default_vals, v, test))
            if st and st != "ns":
                ax.text(star_x, j, st, fontsize=6.5, fontweight="bold",
                        va="center", ha="left", color="0.3",
                        clip_on=False, zorder=7)

    ax.set_yticks(range(n))
    if show_y_ticklabels:
        ax.set_yticklabels(method_order, fontweight="medium")
        ax.tick_params(axis="y", labelleft=True, length=0)
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    # Honest baseline (bars from 0) by default; --zoom truncates to the data
    # range to amplify small differences.
    left = (min_low - 0.04 * span) if zoom else min(0.0, min_low)
    ax.set_xlim(left=left, right=star_x + 0.12 * span)
    ax.xaxis.grid(True, linewidth=0.3, alpha=0.25, color="0.75")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=1))
    ax.set_title(f"{metric_label}  ({direction})", fontsize=7,
                 fontweight="medium", pad=3)


def render_axis(axis, per_metric_values: Dict[str, Dict[str, np.ndarray]],
                labels: Dict[str, str], default_label: str,
                out_path_base: Path, test: str, err_kind: str,
                zoom: bool = False) -> None:
    method_order = [labels[e.prefix] for e in axis.entries]
    colour_for = {labels[e.prefix]: (COLOUR_DEFAULT if e.is_default else COLOUR_OTHER)
                  for e in axis.entries}
    n_methods = len(method_order)
    n_panels = len(METRICS)
    panel_width_in = 0.80
    fig_width_in = panel_width_in * n_panels + 0.7
    fig_height_in = max(1.0, 0.30 * n_methods + 0.7)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width_in, fig_height_in),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]
    for i, (metric_label, direction) in enumerate(METRICS):
        _plot_panel(ax=axes[i], method_order=method_order,
                    per_method_values=per_metric_values[metric_label],
                    default_method=default_label, colour_for=colour_for,
                    metric_label=metric_label, direction=direction,
                    test=test, err_kind=err_kind, show_y_ticklabels=(i == 0),
                    zoom=zoom)
    err_name = {"ci95": "95% CI", "sem": "SEM", "std": "SD"}[err_kind]
    test_name = {"ttest": "Welch t-test", "mannwhitney": "Mann–Whitney"}[test]
    # Title big + bold; metadata/legend as a small grey line beneath it (both
    # above the panels, so they never collide with the bottom x-tick labels).
    fig.suptitle(axis.title, fontsize=9, fontweight="bold", y=1.12)
    fig.text(0.5, 1.03,
             f"bars: mean ± {err_name}   ·   dots: per-seed   ·   "
             f"dashed: default mean   ·   "
             f"* p<0.05  ** p<0.01  *** p<0.001 ({test_name} vs default)",
             ha="center", va="center", fontsize=5.5, color="0.4")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.28)
    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png", "pdf"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT,
                   help="Root containing <dataset>/<variant>__multiseed/ dirs.")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir (default: <artifacts-root>/<dataset>/"
                        "_ablation_multiseed_figures/).")
    p.add_argument("--test", choices=["ttest", "mannwhitney"], default="ttest",
                   help="Significance test vs the axis default (ttest = Welch).")
    p.add_argument("--error", choices=["ci95", "sem", "std"], default="ci95",
                   help="Error-bar half-width (default 95%% CI).")
    p.add_argument("--zoom", action="store_true",
                   help="Zoom the x-axis to the data range (amplifies small "
                        "differences; truncates the bar baseline). Default: "
                        "honest bars from 0.")
    p.add_argument("--cell-label-keys", type=str,
                   default=",".join(DEFAULT_CELL_LABEL_KEYS))
    p.add_argument("--niche-label-keys", type=str,
                   default=",".join(DEFAULT_NICHE_LABEL_KEYS))
    args = p.parse_args(argv)

    out_dir = args.out_dir or (args.artifacts_root / args.dataset
                               / "_ablation_multiseed_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_nature_style()
    cell_pref = tuple(k.strip() for k in args.cell_label_keys.split(",") if k.strip())
    niche_pref = tuple(k.strip() for k in args.niche_label_keys.split(",") if k.strip())

    print(f"artifacts root: {args.artifacts_root}")
    print(f"dataset       : {args.dataset}")
    print(f"output dir    : {out_dir}")
    print(f"test={args.test}  error={args.error}")

    for axis in AXES:
        print(f"\n=== {axis.key}: {axis.title} ===")
        entry_dir = {e.prefix: _multiseed_metrics_dir(e.prefix, args.dataset,
                                                      args.artifacts_root)
                     for e in axis.entries}
        missing = [e.prefix for e in axis.entries if entry_dir[e.prefix] is None]
        if missing:
            print(f"  SKIP {axis.key}: no __multiseed per_seed data for "
                  f"{missing} — run those sweeps first.")
            continue
        labels = {e.prefix: e.label for e in axis.entries}
        default_prefix = next(e.prefix for e in axis.entries if e.is_default)
        default_dir = entry_dir[default_prefix]
        cell_label = _pick_label_key(default_dir, True, cell_pref)
        niche_label = _pick_label_key(default_dir, False, niche_pref)
        print(f"  cell label: {cell_label!r}   niche label: {niche_label!r}")

        per_metric_values: Dict[str, Dict[str, np.ndarray]] = {}
        for metric_label, _direction in METRICS:
            per_metric_values[metric_label] = {
                labels[e.prefix]: _per_seed_values(
                    entry_dir[e.prefix], metric_label, cell_label, niche_label)
                for e in axis.entries
            }
        # report per-seed counts for the default (sanity)
        nseed = {m: per_metric_values[m][labels[default_prefix]].size for m, _ in METRICS}
        print(f"  default per-seed n: {nseed}")
        render_axis(axis, per_metric_values, labels, labels[default_prefix],
                    out_dir / axis.key, args.test, args.error,
                    zoom=args.zoom)

    print(f"\n[ablations-multiseed] DONE -> {out_dir}")


if __name__ == "__main__":
    main()
