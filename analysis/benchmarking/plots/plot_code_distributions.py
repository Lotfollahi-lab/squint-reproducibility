#!/usr/bin/env python3
"""
SQUINT code-distribution visualisation across sections (Nature style).

Given a `predicted_adata.h5ad` written by SQUINT, this script renders
publication-quality figures comparing the empirical distribution of
CELL codes and NICHE codes across the sections (samples / tissue
slides / batches) present in the AnnData.

The intuition: two sections that share tissue state will yield similar
code histograms; sections that differ biologically (different tumour
sub-types, drug vs vehicle, healthy vs disease) will diverge. This
gives a discrete, codebook-level handle on inter-section similarity
that does not depend on continuous embedding choices.

Outputs (under --out-dir):

  1. code_distributions.{svg,png}
        Small-multiples grid. Rows = sections, columns = (cell codes,
        niche codes). Each panel is a vertical bar chart of code
        frequency for that section. Bar COLOUR is consistent across
        panels for the same code, so the same colour in two panels
        means the same codebook entry — making structural similarity
        immediately legible by eye. Y-axes shared within a column so
        heights are directly comparable.

  2. similarity_heatmap.{svg,png}
        Two heatmaps side-by-side: cell-code and niche-code section
        similarity matrices. Diagonal = 1.0. Cell ordering of both
        heatmaps respects the section order in --section-order (or the
        sorted order seen in the AnnData if not given).

  3. per_section_ranking/<section>.{svg,png}   (one figure per section)
        For each section S, a 2-panel figure showing every OTHER
        section ranked from most- to least-similar to S, separately
        for cell codes and niche codes. The most-similar section is
        highlighted in the SQUINT accent colour to draw the eye.

  4. similarity_long.csv
        Tidy CSV with one row per (source, target, code_kind, score)
        triple. Diagonal entries (source == target) are included with
        score=1.0 for downstream convenience.

Similarity score
----------------
We use **1 - Jensen-Shannon distance** with log base 2. The
Jensen-Shannon DIVERGENCE with log base 2 is bounded [0, 1], so its
square-root (the DISTANCE, which is a true metric) is also bounded
[0, 1], and our similarity = 1 - distance is bounded [0, 1] with 1.0
meaning identical distributions. This is the natural symmetric
information-theoretic distance between probability mass functions, and
is the standard choice for comparing categorical histograms.

Code-key selection
------------------
Code keys are searched in `adata.obs` FIRST, then in `adata.obsm`:

  Cell codes  : cell_code_indices[level_0]    (obs, legacy 1-D layout)
                cell_code_indices[composite]  (obs)
                cell_code_index               (obs)
                cell_code_indices             (obsm, NEWER 2-D layout
                                                 — shape `(N, L)`, one
                                                 column per RVQ level)
  Niche codes : neighborhood_code_indices[level_0]    (obs)
                neighborhood_code_indices[composite]  (obs)
                neighborhood_code_index               (obs)
                neighborhood_code_indices             (obsm 2-D)
  Section     : section, adata_batch_id, batch_id, batch, sample,
                sample_id      (always in obs)

For the obsm 2-D layout, --cell-level / --niche-level choose which
column to plot:
  '0' / 'l0' / 'level_0'   — K1 macro partition (default, ~10-30 bars)
  '1' / 'l1' / 'level_1'   — K2 sub-partition
  'composite'              — L1*max(L2)+L2, one bar per unique leaf

The level flags are IGNORED for obs columns (the level is then baked
into the column name like `[level_0]`).

Usage
-----
  # Defaults: read the predicted AnnData and write all figures next to it.
  python analysis/benchmarking/plots/plot_code_distributions.py \\
      --adata /nfs/team361/sb75/squint-reproducibility/artifacts/\\
chl59-8b_1p/<variant>/<TS>/predicted_adata.h5ad

  # Use the high-resolution composite codes instead of level_0:
  python analysis/benchmarking/plots/plot_code_distributions.py \\
      --adata <path> \\
      --cell-code-col cell_code_indices[composite] \\
      --niche-code-col neighborhood_code_indices[composite]

  # Custom section order (top-to-bottom in the small-multiples grid):
  python analysis/benchmarking/plots/plot_code_distributions.py \\
      --adata <path> \\
      --section-order 0,1,4,5,6,7,2,3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Auto-detection order. Each entry is searched in `adata.obs.columns`
# FIRST, then in `adata.obsm.keys()`. obs entries are 1-D code vectors
# whose name encodes the level (e.g. `[level_0]`); obsm entries are 2-D
# `(N, L)` arrays whose level is picked via --cell-level / --niche-level
# (default: 0, i.e. the K1 macro RVQ partition).
DEFAULT_CELL_CODE_KEYS: Tuple[str, ...] = (
    "cell_code_indices[level_0]",      # obs  — K1 macro partition (preferred)
    "cell_code_indices[composite]",    # obs  — K1*K2 leaf clusters
    "cell_code_index",                 # obs  — single-level VQ fallback
    "cell_code_indices",               # obsm — 2-D (N, L), newer layout
)
DEFAULT_NICHE_CODE_KEYS: Tuple[str, ...] = (
    "neighborhood_code_indices[level_0]",
    "neighborhood_code_indices[composite]",
    "neighborhood_code_index",
    "neighborhood_code_indices",       # obsm — 2-D (N, L)
)
DEFAULT_SECTION_COLS: Tuple[str, ...] = (
    "section",
    "adata_batch_id",
    "batch_id",
    "batch",
    "sample",
    "sample_id",
)

# Visual style.
ACCENT      = "#FF006E"  # SQUINT accent — used for the top-ranked section
ACCENT_DARK = "#A8004A"  # darker accent for hover/emphasis
NEUTRAL     = "#BDBDBD"  # neutral grey for non-highlighted bars / NaNs
HEATMAP_CMAP_NAME = "squint_blues"  # registered below


def _apply_nature_style() -> None:
    """rcParams aligned with plot_niche_identification_benchmark.py so
    these figures slot in next to the existing paper plots."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
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
    # Register a SQUINT-blue colormap for the similarity heatmaps —
    # off-white -> deep blue, matching the cell-type-ID family. The
    # accent red is reserved for the per-section "most similar"
    # highlight so the eye never confuses the two.
    if HEATMAP_CMAP_NAME not in plt.colormaps():
        cmap = LinearSegmentedColormap.from_list(
            HEATMAP_CMAP_NAME,
            [
                (0.00, "#F7FBFF"),  # very pale blue
                (0.25, "#C6DBEF"),
                (0.50, "#6BAED6"),
                (0.75, "#2171B5"),
                (1.00, "#08306B"),  # deep navy
            ],
        )
        mpl.colormaps.register(cmap=cmap, name=HEATMAP_CMAP_NAME)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _autodetect_code_key(
        adata: ad.AnnData,
        candidates: Sequence[str],
        kind: str,
    ) -> Tuple[str, str]:
    """Return `(key, source)` where source is `"obs"` or `"obsm"`. obs
    is preferred when both contain a matching key.

    Raises with a helpful message listing both obs columns and obsm
    keys if no candidate matches.
    """
    for c in candidates:
        if c in adata.obs.columns:
            return c, "obs"
        if c in adata.obsm.keys():
            return c, "obsm"
    raise SystemExit(
        f"Could not auto-detect a {kind} key. None of {list(candidates)} "
        f"are present in adata.obs OR adata.obsm.\n"
        f"  obs columns:  {sorted(adata.obs.columns.tolist())}\n"
        f"  obsm keys:    {sorted(adata.obsm.keys())}\n"
        f"Pass the key explicitly via the CLI (it may live in either "
        f"obs or obsm; the script searches both)."
    )


def _autodetect_section(
        obs: pd.DataFrame,
        candidates: Sequence[str],
    ) -> str:
    """Return the first obs column present in `candidates`. Section
    column is ALWAYS in obs."""
    for c in candidates:
        if c in obs.columns:
            return c
    raise SystemExit(
        f"Could not auto-detect a section column. None of "
        f"{list(candidates)} are present in adata.obs. Available "
        f"columns: {sorted(obs.columns.tolist())}. Pass --section-col "
        f"explicitly."
    )


def _parse_level(level_str: str) -> "int | str":
    """Map CLI level string -> column index or 'composite'."""
    s = level_str.strip().lower()
    aliases = {
        "0": 0, "l0": 0, "level_0": 0, "level0": 0,
        "1": 1, "l1": 1, "level_1": 1, "level1": 1,
        "2": 2, "l2": 2, "level_2": 2, "level2": 2,
        "composite": "composite", "comp": "composite",
    }
    if s not in aliases:
        raise SystemExit(
            f"--*-level got {level_str!r}; expected one of "
            f"{sorted(set(aliases.keys()))}."
        )
    return aliases[s]


def _resolve_code_vector(
        adata: ad.AnnData,
        key: str,
        source: str,
        level: "int | str",
        kind: str,
    ) -> Tuple[np.ndarray, str]:
    """Return `(codes_1d, descriptor)`.

    - If `source == "obs"`: the obs column is already 1-D; `level` is
      ignored (the level is baked into the obs column name like
      `[level_0]`). We surface that name in the descriptor.

    - If `source == "obsm"`: read the 2-D `(N, L)` array, pick column
      `level` (int), or synthesize a composite leaf index when
      `level == "composite"`. Composite = `col_0 * (max(col_1)+1) + col_1`,
      which gives one unique index per (L1, L2) leaf — matching the
      semantics of the legacy `[composite]` obs column.
    """
    if source == "obs":
        vec = adata.obs[key].to_numpy()
        return vec, f"obs['{key}']"

    if source != "obsm":
        raise ValueError(f"unknown source {source!r}")

    arr = np.asarray(adata.obsm[key])
    if arr.ndim == 1:
        # Already 1-D in obsm — treat like an obs column.
        return arr, f"obsm['{key}']"
    if arr.ndim != 2:
        raise SystemExit(
            f"adata.obsm['{key}'] has ndim={arr.ndim}; expected 1 or 2."
        )
    n_levels = arr.shape[1]
    if isinstance(level, int):
        if level < 0 or level >= n_levels:
            raise SystemExit(
                f"--{kind}-level={level} out of range for "
                f"adata.obsm['{key}'] which has shape {arr.shape}. "
                f"Valid integer levels: 0..{n_levels - 1}, or 'composite'."
            )
        return arr[:, level], f"obsm['{key}'][:, {level}]"
    # composite
    if n_levels < 2:
        raise SystemExit(
            f"--{kind}-level=composite needs at least 2 columns in "
            f"adata.obsm['{key}']; got shape {arr.shape}."
        )
    L2_max = int(arr[:, 1].max()) + 1
    comp = arr[:, 0].astype(np.int64) * L2_max + arr[:, 1].astype(np.int64)
    return comp, f"obsm['{key}'][composite L1*{L2_max}+L2]"


def _coerce_section_label(x) -> str:
    """Normalise a section identifier to a stable display string."""
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)) and float(x).is_integer():
        return str(int(x))
    return str(x)


def _derive_section_labels(
        obs: pd.DataFrame,
        section_col: str,
        sections_raw: List,
        source_file_col: Optional[str],
    ) -> Dict:
    """Map each raw section value to a human-readable display label.

    If `source_file_col` is a real column in `obs`, the label is the
    substring of that column's value BEFORE the first `+`. So for a
    CosMx Lung file like `Lung9_Rep1+SMI+Flat+data.tar.h5ad` the label
    becomes `Lung9_Rep1`.

    If the column is absent / empty / `None`, falls back to the
    stringified raw section id (the previous behaviour).

    When multiple `source_file` values exist for the same section, we
    use the most common one and print a warning — this can happen if
    the AnnData was concatenated with overlapping batch ids and the
    user should know.
    """
    fallback = {s: _coerce_section_label(s) for s in sections_raw}
    if not source_file_col or source_file_col not in obs.columns:
        return fallback

    mapping: Dict = {}
    for s in sections_raw:
        sub = obs.loc[obs[section_col] == s, source_file_col].dropna()
        if sub.empty:
            mapping[s] = fallback[s]
            continue
        # value_counts() on a pandas Categorical returns ALL defined
        # categories — even ones with count 0. Filter to those actually
        # present so the "multiple values" warning only fires for real.
        vc = sub.value_counts()
        vc = vc[vc > 0]
        if len(vc) > 1:
            print(
                f"  WARN: section {s!r} has multiple distinct "
                f"{source_file_col!r} values: {list(vc.index)}. "
                f"Using the most common one ({vc.index[0]!r}) for the "
                f"display label."
            )
        prefix = str(vc.index[0]).split("+", 1)[0]
        mapping[s] = prefix if prefix else fallback[s]
    return mapping


def _derive_donor(label: str, pattern: str) -> str:
    """Extract the donor id from a section display label using a regex.

    Default pattern matches `<donor>_Rep<N>` (e.g. `Lung9_Rep1` ->
    `Lung9`). When the pattern doesn't match, returns the label
    unchanged — meaning donor == section (one rep per donor).
    """
    import re
    m = re.match(pattern, label)
    if m and m.groups():
        donor = m.group(1)
        if donor:
            return donor
    return label


def _donor_palette(donors_ordered: List[str]) -> Dict[str, tuple]:
    """Stable {donor: rgba} palette. Order preserved across calls (so
    legends in the heatmap and the ranking figures use the same
    colour for the same donor)."""
    # tab10 has 10 reasonably distinct categorical colours; tab20 has
    # 20 (paired light/dark). For 1-10 donors use tab10; 11-20 use
    # tab20; else cycle a 40-colour stitched palette.
    n = len(donors_ordered)
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        cols = [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        cols = [cmap(i) for i in range(n)]
    else:
        a = plt.get_cmap("tab20")
        b = plt.get_cmap("tab20b")
        stitched = [a(i) for i in range(20)] + [b(i) for i in range(20)]
        cols = [stitched[i % 40] for i in range(n)]
    return dict(zip(donors_ordered, cols))


def _build_code_universe(obs: pd.DataFrame, code_col: str,
                         nominal_k: "int | None" = None) -> np.ndarray:
    """Return the sorted array of code values to plot for `code_col`.

    By default this is the set of codes that ACTUALLY APPEAR in the data
    (`.unique()`), so a dead code (a codebook entry never assigned to any
    cell) is silently absent — e.g. a nominal-90 codebook with one dead code
    yields 89 bars. Pass `nominal_k` (the codebook size for this level) to
    instead span the FULL codebook `0..K-1`, so dead codes show as zero-height
    bars and the axis matches the reported K. Only applies to INTEGER codes;
    ignored for non-integer (label-name) columns."""
    vals = obs[code_col].dropna().unique()
    # Try to coerce to int -> numeric sort; fall back to string sort.
    try:
        arr = np.array([int(v) for v in vals])
        arr.sort()
        if nominal_k is not None:
            # Span the full nominal codebook. Guard: never SHRINK below the
            # observed max (in case an index unexpectedly exceeds nominal_k),
            # so no assigned code is ever dropped.
            observed_max = int(arr.max()) + 1 if arr.size else 0
            return np.arange(max(int(nominal_k), observed_max), dtype=int)
        return arr
    except (TypeError, ValueError):
        arr = np.array([str(v) for v in vals])
        arr.sort()
        return arr


def _nominal_codebook_k(adata, branch: str, src: str, key: str,
                        level, cli_override: "int | None" = None) -> "int | None":
    """Nominal codebook size K for the resolved (branch, level) column.

    Lets the distribution axis span the FULL codebook (dead codes -> zero
    bars) instead of only the observed-unique codes. Priority:
      cli_override  >  adata.uns['codebook_sizes_<branch>'][level_idx]
    Returns None (=> keep the observed-unique behaviour) for composite levels
    or when the sizes metadata is missing / the level is out of range.

    `level` is the parsed --*-level (int or 'composite') for obsm columns; for
    obs columns the level is baked into the key name (e.g. '[level_1]')."""
    if cli_override is not None:
        return int(cli_override)
    # Which residual level does the resolved column correspond to?
    if src == "obsm":
        level_idx = level if isinstance(level, int) else None      # composite -> None
    else:                                    # obs column: level baked into name
        low = str(key).lower()
        if "composite" in low:
            level_idx = None
        elif "level_2" in low or "level2" in low:
            level_idx = 2
        elif "level_1" in low or "level1" in low:
            level_idx = 1
        else:
            level_idx = 0                    # 'level_0' or a bare single-level col
    if level_idx is None:
        return None
    sizes = adata.uns.get(f"codebook_sizes_{branch}")
    if sizes is None and branch == "niche":
        sizes = adata.uns.get("codebook_sizes")                    # legacy alias
    if sizes is None:
        return None
    try:
        sizes = [int(s) for s in np.asarray(sizes).ravel().tolist()]
    except (TypeError, ValueError):
        return None
    if 0 <= level_idx < len(sizes):
        return int(sizes[level_idx])
    return None


def _section_distribution(
        obs: pd.DataFrame,
        section,
        section_col: str,
        code_col: str,
        code_universe: np.ndarray,
    ) -> np.ndarray:
    """Empirical probability distribution over `code_universe` for the
    rows of `obs` in the given section. Returns a length-K float array
    summing to 1 (or all zeros if the section is empty)."""
    sub = obs.loc[obs[section_col] == section, code_col]
    if sub.empty:
        return np.zeros(len(code_universe), dtype=float)
    counts = sub.value_counts()
    # Reindex onto the global universe; sections that never use a code
    # contribute 0 there. Cast to float for division.
    aligned = counts.reindex(code_universe, fill_value=0).to_numpy(dtype=float)
    total = aligned.sum()
    return aligned / total if total > 0 else aligned


def _jsd_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """Similarity = 1 - Jensen-Shannon DISTANCE (log base 2). Bounded
    [0, 1]; 1.0 ⇔ identical distributions.

    Implemented inline (no scipy.spatial.distance.jensenshannon dep)
    so the script is portable across older scipy installs."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # KL with the convention 0*log(0/x) = 0, x*log(x/0) = inf; but
    # because m = (p+q)/2, whenever p>0 we also have m>0, so the only
    # problematic term is p==0 (where the contribution is 0). Mask it.
    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask])))
    js_div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # Numerical safety: JS divergence with log2 is in [0, 1]; clamp.
    js_div = float(np.clip(js_div, 0.0, 1.0))
    js_dist = float(np.sqrt(js_div))
    return 1.0 - js_dist


# ---------------------------------------------------------------------------
# Palette: one stable colour per code (consistent across panels)
# ---------------------------------------------------------------------------

def _code_palette(n_codes: int) -> List:
    """Return `n_codes` colours, ordered. Designed so neighbouring
    code indices get easily-distinguishable colours; for n>20 we cycle
    `tab20` (acceptable when n<=60-ish — beyond that the user is
    probably plotting [composite] codes and should switch to a heatmap
    representation anyway)."""
    if n_codes <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n_codes)]
    if n_codes <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n_codes)]
    if n_codes <= 40:
        # Interleave tab20 + tab20b for 40 distinguishable colours.
        a = plt.get_cmap("tab20")
        b = plt.get_cmap("tab20b")
        return [a(i) for i in range(20)] + [b(i) for i in range(n_codes - 20)]
    # Fallback: cycle the husl-like tab20c continuously; the script
    # warns the user later (see main()).
    a = plt.get_cmap("tab20")
    b = plt.get_cmap("tab20b")
    c = plt.get_cmap("tab20c")
    base = ([a(i) for i in range(20)] +
            [b(i) for i in range(20)] +
            [c(i) for i in range(20)])
    return [base[i % 60] for i in range(n_codes)]


# ---------------------------------------------------------------------------
# Per-code colours MATCHING the spatial code_index_plots
# (squint/examples/plot_code_indices_spatial.py). The spatial scatter colours
# each cell by its RAW code id: categorical tab20/tab20b for <=max_categorical
# codes with colour(code k) = cmap(k % cmap.N) (so "code k -> colour k" is
# stable across sections), else a cyclic hsv with colour(code) = hsv(code /
# max_code). We mirror that exactly here so a given code is the SAME colour in
# the distribution bars, the stacked-proportion bars, and the spatial plots.
# (The functions are copied — not imported — to keep this benchmark script
# self-contained / independent of the squint examples dir.)
# ---------------------------------------------------------------------------
def _build_palette(num_codes: int, max_categorical: int = 30):
    """Mirror of plot_code_indices_spatial.py::_build_palette."""
    if num_codes <= max_categorical:
        if num_codes <= 20:
            cmap = plt.get_cmap("tab20", num_codes)
        else:
            tab20 = plt.get_cmap("tab20").colors
            tab20b = plt.get_cmap("tab20b").colors
            cmap = ListedColormap((list(tab20) + list(tab20b))[:num_codes])
        return cmap, True
    return plt.get_cmap("hsv"), False


def _code_index_colors(code_universe: np.ndarray, max_categorical: int = 30) -> List:
    """Colour-per-bar list aligned to `code_universe` (the ordered unique code
    ids drawn as bars), so bar for code k gets the SAME colour as the spatial
    code_index_plots scatter. Categorical: cmap(k % N). Large: hsv(k/max)."""
    cu = np.asarray(code_universe)
    if cu.size == 0:
        return []
    # Only integer CODE universes get the spatial-plot palette. A LABEL
    # universe (cell-type / niche names — strings) has no code id, so fall
    # back to the rank-based palette (its old behaviour) rather than crashing.
    try:
        cu = cu.astype(int)
    except (ValueError, TypeError):
        return _code_palette(len(cu))
    max_code = int(cu.max())
    cmap, is_cat = _build_palette(max_code + 1, max_categorical)
    if is_cat:
        return [cmap(int(c) % cmap.N) for c in cu]
    md = max(max_code, 1)
    return [cmap(int(c) / float(md)) for c in cu]


# ---------------------------------------------------------------------------
# Plot: small-multiples grid of distributions
# ---------------------------------------------------------------------------

def render_distribution_grid(
        sections: List[str],
        code_universe_cell: np.ndarray,
        code_universe_niche: np.ndarray,
        dist_cell: Dict[str, np.ndarray],
        dist_niche: Dict[str, np.ndarray],
        cell_code_col: str,
        niche_code_col: str,
        out_path_base: Path,
        cell_axis_title: str = "Cell codes",
        niche_axis_title: str = "Niche codes",
        suptitle: str = "Code distribution per section",
        code_colored: bool = True,
    ) -> None:
    """Render the small-multiples grid (rows = sections, cols = (cell,
    niche)) and save to `<out_path_base>.{svg,png}`.

    `cell_axis_title` / `niche_axis_title` / `suptitle` allow the same
    layout to be reused for a labels-based pass (e.g. cell_type /
    niche obs columns) instead of model code indices.
    """
    n_sec = len(sections)
    n_cell = len(code_universe_cell)
    n_niche = len(code_universe_niche)
    # Colour bars by RAW code id to match the spatial code_index_plots (a code
    # is the same colour in both figures), not by bar rank. The labels pass
    # (cell-type / niche label names) sets code_colored=False -> keep the old
    # rank-based palette (labels have no code id).
    if code_colored:
        pal_cell  = _code_index_colors(code_universe_cell)
        pal_niche = _code_index_colors(code_universe_niche)
    else:
        pal_cell  = _code_palette(n_cell)
        pal_niche = _code_palette(n_niche)

    # Panel sizes. Width scales gently with code count so 30 vs 90 vs
    # composite all stay legible without overflowing a Nature column.
    def _panel_width(n_codes: int) -> float:
        # 0.07 inches per bar, with sensible min/max. Slightly more
        # generous than 0.05 so individual bars stay readable at 30
        # codes without padding too much at 90.
        return float(np.clip(0.07 * n_codes + 0.5, 2.0, 6.0))
    w_cell   = _panel_width(n_cell)
    w_niche  = _panel_width(n_niche)
    h_row    = 0.85   # inches per section row
    # Reserve left margin for the row label. Adaptive: ~0.055" per
    # character at 7.5 pt + 0.18" padding. Lower-bounded at 0.7" so
    # short labels ("0", "1") still get sensible whitespace; upper-
    # bounded at 1.6" so a stray long label doesn't dominate the figure.
    max_label_len = max((len(s) for s in sections), default=2)
    left_pad = float(np.clip(0.055 * max_label_len + 0.18, 0.7, 1.6))
    fig_w = left_pad + w_cell + w_niche + 0.25
    fig_h = h_row * n_sec + 0.85   # extra for column titles + footer

    # Use width_ratios so the cell column and the niche column get the
    # space they need given their respective code-counts. height_ratios
    # is uniform — every section gets the same vertical real estate.
    fig, axes = plt.subplots(
        n_sec, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            "width_ratios": [w_cell, w_niche],
            "hspace": 0.28,
            "wspace": 0.14,
        },
        sharex=False,
    )
    if n_sec == 1:
        axes = np.array([axes])  # ensure 2-D indexing

    # Cap y-axis per column so heights are directly comparable across
    # all sections within a column.
    ymax_cell  = max((dist_cell[s].max()  for s in sections), default=0.0)
    ymax_niche = max((dist_niche[s].max() for s in sections), default=0.0)
    # Round up to a nice number for tick aesthetics.
    def _nice_ylim(m: float) -> float:
        if m <= 0: return 1.0
        # round to the next 0.05 above m, with a small headroom.
        return float(np.ceil(m / 0.05) * 0.05 + 0.01)
    ylim_cell  = _nice_ylim(ymax_cell)
    ylim_niche = _nice_ylim(ymax_niche)

    for i, sec in enumerate(sections):
        # ---- CELL panel ------------------------------------------------
        ax = axes[i, 0]
        vals = dist_cell[sec]
        ax.bar(
            np.arange(n_cell), vals,
            color=pal_cell, edgecolor="none",
            width=0.85, zorder=2,
        )
        ax.set_ylim(0, ylim_cell)
        ax.set_xlim(-0.6, n_cell - 0.4)
        # X ticks: every code if <=30, else sparse.
        if n_cell <= 30:
            ax.set_xticks(np.arange(n_cell))
            ax.set_xticklabels(
                [str(c) for c in code_universe_cell],
                rotation=0, fontsize=5,
            )
        else:
            step = max(1, n_cell // 12)
            xt = np.arange(0, n_cell, step)
            ax.set_xticks(xt)
            ax.set_xticklabels(
                [str(code_universe_cell[i]) for i in xt],
                rotation=0, fontsize=5,
            )
        ax.tick_params(axis="x", pad=1.5)
        # Y ticks: just 2 (0 and max) to keep panels uncluttered.
        ax.set_yticks([0.0, ylim_cell])
        ax.set_yticklabels(["0", f"{ylim_cell:.2f}"], fontsize=5.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.2, alpha=0.35,
                      color="0.65", linestyle="--")
        ax.set_axisbelow(True)
        # Row label = section display name (e.g. `Lung9_Rep1`, derived
        # from the source_file column). Drawn as the leftmost panel's
        # ylabel with rotation=0, which places it naturally to the left
        # of the axes and is laid-out correctly by matplotlib (no
        # manual point-offset arithmetic to get wrong). labelpad in
        # points; 14 pt ≈ 0.19" of clearance.
        ax.set_ylabel(
            sec,
            fontsize=7.5, fontweight="semibold",
            rotation=0, ha="right", va="center", labelpad=14,
        )
        # Column title only on the top row. Keep it short to avoid
        # collisions with the adjacent column; the full source
        # descriptor is rendered as a smaller figure-level subtitle
        # below the suptitle.
        if i == 0:
            ax.set_title(
                cell_axis_title,
                fontsize=8.5, fontweight="medium", pad=4,
            )
        # X-axis label only on the bottom row.
        if i == n_sec - 1:
            ax.set_xlabel("Code index", fontsize=7, labelpad=2)

        # ---- NICHE panel -----------------------------------------------
        ax = axes[i, 1]
        vals = dist_niche[sec]
        ax.bar(
            np.arange(n_niche), vals,
            color=pal_niche, edgecolor="none",
            width=0.85, zorder=2,
        )
        ax.set_ylim(0, ylim_niche)
        ax.set_xlim(-0.6, n_niche - 0.4)
        if n_niche <= 30:
            ax.set_xticks(np.arange(n_niche))
            ax.set_xticklabels(
                [str(c) for c in code_universe_niche],
                rotation=0, fontsize=5,
            )
        else:
            step = max(1, n_niche // 12)
            xt = np.arange(0, n_niche, step)
            ax.set_xticks(xt)
            ax.set_xticklabels(
                [str(code_universe_niche[i]) for i in xt],
                rotation=0, fontsize=5,
            )
        ax.tick_params(axis="x", pad=1.5)
        ax.set_yticks([0.0, ylim_niche])
        ax.set_yticklabels(["0", f"{ylim_niche:.2f}"], fontsize=5.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.2, alpha=0.35,
                      color="0.65", linestyle="--")
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_title(
                niche_axis_title,
                fontsize=8.5, fontweight="medium", pad=4,
            )
        if i == n_sec - 1:
            ax.set_xlabel("Code index", fontsize=7, labelpad=2)

    fig.suptitle(
        suptitle,
        fontsize=9.5, fontweight="semibold", y=0.998,
    )
    # Smaller italic subtitle showing the precise source / level for
    # the cell and niche columns. Sits just below the suptitle and
    # above the column titles. Coordinates in figure fraction.
    fig.text(
        0.5, 0.998 - 0.30 / fig_h,
        f"cell:  {cell_code_col}      •      niche:  {niche_code_col}",
        ha="center", va="top",
        fontsize=6.5, fontstyle="italic", color="0.4",
    )

    # Leave room on the left for the rotated row labels (ylabel of the
    # leftmost panels) and below for the x-axis labels of the bottom
    # row. The top margin is enlarged to ~0.75" so the suptitle, the
    # italic source-descriptor subtitle and the column titles all fit
    # without crowding each other.
    fig.subplots_adjust(
        left=left_pad / fig_w,
        right=1 - 0.06 / fig_w,
        top=1 - 0.75 / fig_h,
        bottom=0.40 / fig_h,
    )

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: stacked code-proportion bars (one stacked bar per section)
# ---------------------------------------------------------------------------

def render_proportions_stacked(
        sections: List[str],
        code_universe_cell: np.ndarray,
        code_universe_niche: np.ndarray,
        dist_cell: Dict[str, np.ndarray],
        dist_niche: Dict[str, np.ndarray],
        out_path_base: Path,
        cell_axis_title: str = "Cell codes",
        niche_axis_title: str = "Niche codes",
        suptitle: str = "Code proportions per section",
        code_colored: bool = True,
    ) -> None:
    """One STACKED bar per section showing the full code-proportion
    composition (segments sum to 1), coloured by the SAME per-code palette as
    the spatial code_index_plots / the distribution grid. Two panels (cell,
    niche). Saved to `<out_path_base>.{svg,png}`.

    `dist_cell[sec]` / `dist_niche[sec]` are the per-section proportion vectors
    (aligned to `code_universe_*`, summing to ~1) already computed for the
    distribution grid — so this is a complementary view of the same data."""
    n_sec = len(sections)
    if code_colored:
        pal_cell  = _code_index_colors(code_universe_cell)
        pal_niche = _code_index_colors(code_universe_niche)
    else:
        pal_cell  = _code_palette(len(code_universe_cell))
        pal_niche = _code_palette(len(code_universe_niche))

    # HORIZONTAL stacked bars: sections on the y-axis, code proportion on the
    # x-axis. Height scales with the number of sections; width fixed (2 panels).
    fig_h = float(np.clip(0.34 * n_sec + 1.4, 2.4, 16.0))
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, fig_h), gridspec_kw={"wspace": 0.3},
    )
    y = np.arange(n_sec)
    for ax, universe, dist, pal, title in (
        (axes[0], code_universe_cell,  dist_cell,  pal_cell,  cell_axis_title),
        (axes[1], code_universe_niche, dist_niche, pal_niche, niche_axis_title),
    ):
        if len(universe) == 0:
            ax.set_visible(False)
            continue
        # (n_sec, n_codes) proportion matrix, rows = sections.
        mat = np.vstack([np.asarray(dist[s], dtype=float) for s in sections])
        left = np.zeros(n_sec)
        for k in range(len(universe)):
            ax.barh(
                y, mat[:, k], left=left, color=pal[k],
                height=0.85, edgecolor="white", linewidth=0.1, zorder=2,
            )
            left = left + mat[:, k]
        ax.set_title(title, fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(sections, fontsize=6)
        ax.set_ylim(-0.6, n_sec - 0.4)
        ax.invert_yaxis()                 # first section at the top
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("code proportion", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(suptitle, fontsize=9)

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: paired similarity heatmaps
# ---------------------------------------------------------------------------

def render_similarity_heatmaps(
        sections: List[str],
        sim_cell: np.ndarray,
        sim_niche: np.ndarray,
        out_path_base: Path,
        donor_for_section: Optional[Dict[str, str]] = None,
        donor_palette: Optional[Dict[str, tuple]] = None,
        query_sections: Optional[List[str]] = None,
        cell_axis_title: str = "Cell codes",
        niche_axis_title: str = "Niche codes",
        suptitle: str = "Section similarity",
    ) -> None:
    """Render two heatmaps (cell, niche) side-by-side, with optional
    coloured donor annotation strips along the top + left edges of each
    panel. When `donor_for_section` is provided, a small legend in the
    upper-right of the figure maps donor -> colour.

    When `query_sections` is provided, the matching tick labels (x and
    y axes) are coloured in SQUINT-accent red and bold so the held-out
    targets are immediately legible.
    """
    n = len(sections)
    query_set = set(query_sections or [])
    # Adapt panel width to label length so long names like `Lung9_Rep1`
    # don't get squished or overlapping. Slightly more generous than
    # before to give the larger tick / cell-annotation fonts room.
    max_label_len = max((len(s) for s in sections), default=2)
    panel_w = max(3.0, 0.42 * n + 0.06 * max_label_len + 1.2)
    fig_w = panel_w * 2 + 1.1
    fig_h = max(3.4, 0.42 * n + 0.08 * max_label_len + 1.8)

    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.30},
    )

    # Both heatmaps share the SAME colour scale ([0, 1]) so the eye can
    # compare cell-side vs niche-side similarities directly.
    cmap = plt.get_cmap(HEATMAP_CMAP_NAME)
    vmin, vmax = 0.0, 1.0

    # Donor strip width in axis-data units: 0.4 is a thin tab roughly
    # half the height of a heatmap cell. It sits OUTSIDE the main grid
    # extent so the similarity heatmap stays clean.
    strip_w = 0.42 if donor_for_section else 0.0

    have_donor_annot = (
        donor_for_section is not None and donor_palette is not None
    )

    for ax, mat, ttl in zip(
        axes,
        [sim_cell, sim_niche],
        [cell_axis_title, niche_axis_title],
    ):
        im = ax.imshow(
            mat, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest",
        )
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        xtl = ax.set_xticklabels(sections, rotation=45, ha="right",
                                  fontsize=9)
        ytl = ax.set_yticklabels(sections, fontsize=9)
        # Highlight query sections in SQUINT accent red + bold so the
        # held-out targets pop next to the donor-coloured strips.
        if query_set:
            for lbl in xtl:
                if lbl.get_text() in query_set:
                    lbl.set_color(ACCENT)
                    lbl.set_fontweight("bold")
            for lbl in ytl:
                if lbl.get_text() in query_set:
                    lbl.set_color(ACCENT)
                    lbl.set_fontweight("bold")
        ax.set_title(ttl, fontsize=10.5, fontweight="medium", pad=8)
        # Subtle grid between cells.
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

        # --- Donor annotation strips ---------------------------------
        # Draw two thin coloured strips: one above the heatmap (top
        # axis) and one to the left of it. Each square = one section,
        # coloured by donor. Sections from the same donor get the same
        # colour, so block-diagonal donor structure becomes visible
        # alongside the similarity heatmap.
        if have_donor_annot:
            from matplotlib.patches import Rectangle
            # Top strip: at y in [-0.5 - strip_w, -0.5).
            for j, sec in enumerate(sections):
                colour = donor_palette[donor_for_section[sec]]
                ax.add_patch(Rectangle(
                    (j - 0.5, -0.5 - strip_w), 1.0, strip_w,
                    facecolor=colour, edgecolor="white",
                    linewidth=0.6, clip_on=False, zorder=4,
                ))
            # Left strip: at x in [-0.5 - strip_w, -0.5).
            for i, sec in enumerate(sections):
                colour = donor_palette[donor_for_section[sec]]
                ax.add_patch(Rectangle(
                    (-0.5 - strip_w, i - 0.5), strip_w, 1.0,
                    facecolor=colour, edgecolor="white",
                    linewidth=0.6, clip_on=False, zorder=4,
                ))
            # Tiny "Donor" caption above the top-left corner of the
            # strip. Keeps the figure self-describing.
            ax.text(
                -0.5 - strip_w / 2, -0.5 - strip_w - 0.25,
                "Donor", ha="center", va="bottom",
                fontsize=7, color="0.25", fontstyle="italic",
            )
            # Expand the axes data limits a hair so the strip isn't
            # clipped by the default bbox_inches="tight".
            ax.set_xlim(-0.5 - strip_w - 0.05, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5 - strip_w - 0.05)

        # Annotate cells with the similarity to 2 dp. Black/white text
        # depending on background lightness (>0.55 of the [0,1] scale
        # = use white).
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isnan(v):
                    txt = "n/a"
                    col = "0.4"
                else:
                    txt = f"{v:.2f}"
                    col = "white" if v > 0.55 else "0.15"
                ax.text(
                    j, i, txt, ha="center", va="center",
                    fontsize=8, color=col,
                )

    # Shared colourbar on the right.
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(),
        fraction=0.025, pad=0.02, shrink=0.85,
    )
    cbar.set_label("Similarity  (1 − JS distance)", fontsize=8.5, labelpad=5)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_linewidth(0.4)

    # Donor legend — placed BELOW the figure so it doesn't compete
    # with the colourbar on the right. Hidden when no donor info.
    if have_donor_annot:
        from matplotlib.patches import Patch
        # Preserve donor insertion order for a stable legend.
        seen = []
        for sec in sections:
            d = donor_for_section[sec]
            if d not in seen:
                seen.append(d)
        handles = [
            Patch(facecolor=donor_palette[d], edgecolor="white",
                  linewidth=0.6, label=d)
            for d in seen
        ]
        # Up to 6 donors per row in the legend.
        ncol = min(len(handles), 6)
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=ncol,
            frameon=False,
            fontsize=8.5,
            title="Donor",
            title_fontsize=9.5,
            handlelength=1.4, handleheight=1.2,
            columnspacing=1.4,
        )

    fig.suptitle(
        suptitle,
        fontsize=12, fontweight="semibold", y=1.02,
    )

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: per-section ranking (one figure per section)
# ---------------------------------------------------------------------------

def render_per_section_ranking(
        anchor: str,
        sections: List[str],
        sim_cell_row: np.ndarray,
        sim_niche_row: np.ndarray,
        out_path_base: Path,
        donor_for_section: Optional[Dict[str, str]] = None,
        donor_palette: Optional[Dict[str, tuple]] = None,
        cell_axis_title: str = "Cell codes",
        niche_axis_title: str = "Niche codes",
        suptitle_prefix: str = "Sections most similar to",
    ) -> None:
    """Render a 1×2 figure ranking the OTHER sections by similarity to
    `anchor`, separately for cell and niche codes.

    When `donor_for_section` is given, bars are coloured by donor (so
    same-donor sections are immediately legible as a colour family) and
    a small legend appears at the bottom. The top-ranked bar in each
    panel gets a dark edge + full opacity so it still pops; lower-
    ranked bars are drawn at reduced opacity. When donor info is not
    provided, falls back to the original red-top / grey-rest scheme.
    """
    # Drop the self-entry (similarity = 1.0 by construction).
    other_idx = [i for i, s in enumerate(sections) if s != anchor]
    other_labels = [sections[i] for i in other_idx]
    cell_vals = sim_cell_row[other_idx]
    niche_vals = sim_niche_row[other_idx]
    n = len(other_idx)
    have_donor_annot = (
        donor_for_section is not None and donor_palette is not None
    )

    def _ranked(labels, vals):
        order = np.argsort(-np.nan_to_num(vals, nan=-1.0))  # NaNs last
        return [labels[i] for i in order], vals[order], order

    # Width adapts to longest section label so e.g. `Lung9_Rep1` doesn't
    # collide with the bar area. Bumped from the earlier compact sizing
    # to accommodate larger fonts.
    max_label_len = max((len(s) for s in sections), default=2)
    fig_w = float(np.clip(0.09 * max_label_len + 6.4, 6.4, 9.5))
    fig_h = max(2.4, 0.42 * n + 1.5)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.45},
        sharex=True,
        constrained_layout=True,
    )

    for ax, vals, label_kind in zip(
        axes,
        [cell_vals, niche_vals],
        [cell_axis_title, niche_axis_title],
    ):
        ranked_labels, ranked_vals, _order = _ranked(other_labels, vals)
        ys = np.arange(n)

        if have_donor_annot:
            face_cols = [
                donor_palette[donor_for_section[lbl]]
                for lbl in ranked_labels
            ]
            # Top-ranked: full alpha + dark edge for emphasis.
            # Others: 60% alpha so the eye is drawn to the leader.
            alphas = [1.0 if k == 0 else 0.55 for k in range(n)]
            edge_cols = ["0.15" if k == 0 else fc
                         for k, fc in enumerate(face_cols)]
            edge_widths = [0.9 if k == 0 else 0.0 for k in range(n)]
            # barh doesn't support a sequence of alphas → draw bars
            # one at a time so each can carry its own alpha.
            for k in range(n):
                ax.barh(
                    ys[k], ranked_vals[k],
                    color=face_cols[k], edgecolor=edge_cols[k],
                    height=0.68, linewidth=edge_widths[k],
                    alpha=alphas[k], zorder=2,
                )
        else:
            # Original red-top / grey-rest fallback.
            colours = [ACCENT if k == 0 else NEUTRAL for k in range(n)]
            ax.barh(
                ys, ranked_vals,
                color=colours, edgecolor=colours,
                height=0.65, linewidth=0,
                alpha=0.92, zorder=2,
            )

        # Numerical annotation on each bar.
        for k, v in enumerate(ranked_vals):
            if not np.isfinite(v):
                ax.text(0.005, k, "n/a", va="center", ha="left",
                        fontsize=7, fontstyle="italic", color="0.45")
                continue
            inside = v > 0.18
            ax.text(
                v - 0.015 if inside else v + 0.015,
                k, f"{v:.3f}",
                va="center", ha="right" if inside else "left",
                fontsize=7.5,
                color="white" if inside else "0.2",
                fontweight="medium",
            )
        ax.set_yticks(ys)
        ax.set_yticklabels(ranked_labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel("Similarity (1 − JS dist.)",
                      fontsize=9, labelpad=3)
        ax.set_title(label_kind, fontsize=10.5,
                     fontweight="medium", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.grid(True, linewidth=0.2, alpha=0.35,
                      color="0.65", linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle(
        f"{suptitle_prefix} {anchor}",
        fontsize=12, fontweight="semibold", y=1.02,
    )

    # Donor legend at the bottom (omit if not annotated).
    if have_donor_annot:
        from matplotlib.patches import Patch
        seen = []
        for sec in sections:
            d = donor_for_section[sec]
            if d not in seen:
                seen.append(d)
        handles = [
            Patch(facecolor=donor_palette[d], edgecolor="0.4",
                  linewidth=0.4, label=d)
            for d in seen
        ]
        ncol = min(len(handles), 6)
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=ncol,
            frameon=False,
            fontsize=8.5,
            title="Donor",
            title_fontsize=9.5,
            handlelength=1.4, handleheight=1.2,
            columnspacing=1.4,
        )

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        out = out_path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Labels-obs builder: load from an external "all sections labelled" AnnData
# ---------------------------------------------------------------------------

def _build_labels_obs_from_external(
        label_adata_path: Path,
        sections: List[str],
        label_section_col: Optional[str],
        cell_label_col: str,
        niche_label_col: str,
    ) -> "Tuple[pd.DataFrame, List[str]]":
    """Load a single external AnnData and build a tidy obs DataFrame
    for the labels pass.

    Designed for the chl59 workflow where the predicted AnnData has
    labels stripped on held-out test sections, but a separate
    `reference_query_mapping.h5ad`-style file carries the per-cell
    `cell_type` and `niche` annotations for EVERY section.

    Section matching:
      Each row's section identity is read from a single obs column in
      the external file — either the explicit `label_section_col`, or
      the first present from a list of common candidates
      (`source_file`, `batch`, `patient`, `sample`, `section`,
      `adata_batch_id`, ...). Each value is `astype(str)` and split on
      the first `+` (harmless for non-`source_file`-style values like
      `Lung5_Rep3`; effective for `Lung5_Rep3+SMI+Flat+...`). Rows
      whose resulting section identity matches one of `sections`
      (the codes-pass list) are kept; the rest are dropped.

    Returns (labels_obs, kept_sections). `labels_obs` has columns
    [_section, cell_label_col, niche_label_col]; `kept_sections` is the
    subset of `sections` that had matching rows in the external file.
    """
    if not label_adata_path.is_file():
        raise SystemExit(
            f"--label-adata not a file: {label_adata_path}"
        )
    print(f"  Loading labels AnnData: {label_adata_path} ...")
    labels_adata = ad.read_h5ad(label_adata_path)
    print(f"    n_obs = {labels_adata.n_obs:,d}, "
          f"obs cols = {sorted(labels_adata.obs.columns.tolist())}")

    # ---- Resolve section column --------------------------------------
    # Explicit user choice wins; otherwise auto-detect from a list of
    # common naming conventions seen across SQUINT / Nanostring /
    # cellxgene / public spatial-transcriptomics datasets.
    candidate_section_cols: Tuple[str, ...] = (
        "source_file", "batch", "patient", "sample", "section",
        "adata_batch_id", "sample_id", "batch_id", "donor",
    )
    if label_section_col is not None:
        if label_section_col not in labels_adata.obs.columns:
            raise SystemExit(
                f"--label-section-col {label_section_col!r} not in "
                f"--label-adata obs. Available: "
                f"{sorted(labels_adata.obs.columns.tolist())}"
            )
        resolved_sec_col = label_section_col
    else:
        resolved_sec_col = None
        for cand in candidate_section_cols:
            if cand in labels_adata.obs.columns:
                resolved_sec_col = cand
                break
        if resolved_sec_col is None:
            raise SystemExit(
                f"Could not auto-detect a section column in --label-"
                f"adata. Tried {list(candidate_section_cols)}. Pass "
                f"--label-section-col explicitly. Available obs cols: "
                f"{sorted(labels_adata.obs.columns.tolist())}"
            )
    print(f"  Section column in --label-adata: obs['{resolved_sec_col}']")

    # ---- Check label columns -----------------------------------------
    missing = [c for c in (cell_label_col, niche_label_col)
               if c not in labels_adata.obs.columns]
    if missing:
        raise SystemExit(
            f"--label-adata is missing required obs columns {missing}. "
            f"Available: {sorted(labels_adata.obs.columns.tolist())}. "
            f"Pass --cell-label-col / --niche-label-col to point at "
            f"the right ones."
        )

    # ---- Derive per-row section identity -----------------------------
    # Split on first '+' to handle source_file-style values like
    # `Lung5_Rep3+SMI+Flat+...`; harmless for plain values like
    # `Lung5_Rep3` (no `+` present).
    sec_vals = labels_adata.obs[resolved_sec_col].astype(str)
    row_section_raw = sec_vals.str.split("+", n=1).str[0]

    # Section matching is CASE-INSENSITIVE so chl59-style labels with
    # `Lung*` (codes-pass) and `lung*` (Nanostring reference) line up
    # automatically. We lowercase both sides for comparison, then map
    # back to the codes-pass canonical capitalisation so all figures /
    # CSV records carry the same display label.
    sections_lc_to_canonical: Dict[str, str] = {s.lower(): s for s in sections}
    row_section_lc = row_section_raw.str.lower()
    keep_mask = row_section_lc.isin(sections_lc_to_canonical.keys()).to_numpy()
    n_kept_total = int(keep_mask.sum())
    if n_kept_total == 0:
        present = sorted(set(row_section_raw.dropna().tolist()))[:30]
        raise SystemExit(
            f"--label-adata has no rows whose "
            f"obs['{resolved_sec_col}'] matches any codes-pass section "
            f"(case-insensitive comparison).\n"
            f"  codes-pass sections: {sections}\n"
            f"  --label-adata section values (first 30 unique): "
            f"{present}\n"
            f"  Pass --label-section-col to point at the right column, "
            f"or rename one of the section identifiers so they match."
        )

    # Diagnostic: per-canonical-section row counts + report which codes-
    # pass sections have no matching rows (so the user knows up-front
    # which sections will be absent from the labels figures).
    print(f"  Section matching (case-insensitive):")
    matched_lc = set(row_section_lc[keep_mask].dropna().unique())
    for sec in sections:
        sec_lc = sec.lower()
        if sec_lc in matched_lc:
            # Find the literal raw value(s) actually used in the file —
            # there may be one OR many distinct casings.
            raw_for_sec = sorted(set(
                row_section_raw[row_section_lc == sec_lc]
            ))
            n_for_sec = int((row_section_lc == sec_lc).sum())
            literal = (
                f"{raw_for_sec[0]!r}"
                if len(raw_for_sec) == 1
                else f"{raw_for_sec[0]!r} (+{len(raw_for_sec) - 1} casings)"
            )
            print(f"    codes-pass {sec!r:>16s}  <-  "
                  f"label-adata {literal:<18s}  "
                  f"({n_for_sec:>6,d} rows)")
        else:
            print(f"    codes-pass {sec!r:>16s}  <-  "
                  f"NO MATCH                    (skipped)")
    print(f"  Match count: {n_kept_total:,d} rows total.")

    parts: List[pd.DataFrame] = []
    kept: List[str] = []
    # Iterate in codes-pass order so the labels pass renders sections
    # in the same order on every figure (heatmaps line up axis-for-axis).
    for sec in sections:
        sec_lc = sec.lower()
        sec_mask = (row_section_lc == sec_lc).to_numpy()
        if not sec_mask.any():
            continue
        n = int(sec_mask.sum())
        parts.append(pd.DataFrame({
            "_section": [sec] * n,
            cell_label_col:  labels_adata.obs.loc[sec_mask, cell_label_col]
                                              .astype("object").values,
            niche_label_col: labels_adata.obs.loc[sec_mask, niche_label_col]
                                              .astype("object").values,
        }))
        kept.append(sec)

    if not parts:
        raise SystemExit("Labels pass: no sections matched in --label-adata.")
    labels_obs = pd.concat(parts, ignore_index=True)
    print(f"  Labels-pass obs assembled (external): "
          f"{len(kept)} sections, {labels_obs.shape[0]:,d} cells total.")
    return labels_obs, kept


# ---------------------------------------------------------------------------
# Pipeline pass: build universes + distributions + similarities + render
# ---------------------------------------------------------------------------

def _run_distribution_pass(
        *,
        obs: pd.DataFrame,
        section_col: str,
        sections: List[str],
        sections_raw: List,
        cell_col_internal: str,
        niche_col_internal: str,
        cell_descriptor: str,
        niche_descriptor: str,
        out_dir: Path,
        grid_basename: str,
        stacked_basename: str,
        heatmap_basename: str,
        ranking_subdirname: str,
        grid_suptitle: str,
        stacked_suptitle: str,
        heatmap_suptitle: str,
        ranking_suptitle_prefix: str,
        cell_axis_title: str,
        niche_axis_title: str,
        csv_kind_label: str,
        donor_for_section: Optional[Dict[str, str]],
        donor_palette: Optional[Dict[str, tuple]],
        query_sections: Optional[List[str]],
        skip_per_section: bool,
        progress_label: str,
        code_colored: bool = True,
        cell_nominal_k: "int | None" = None,
        niche_nominal_k: "int | None" = None,
    ) -> List[Dict]:
    """Run one full analysis pass (codes or labels) on a working obs
    DataFrame containing `cell_col_internal` and `niche_col_internal`.

    Writes three figure types (distribution grid, similarity heatmap,
    one per-section ranking each) to `out_dir` and returns the list of
    tidy CSV records (with `source_kind=csv_kind_label`) for the
    caller to combine into a single similarity_long.csv.
    """
    # ---- Build code universes ------------------------------------------
    # nominal_k (when known) spans the FULL codebook so dead codes appear as
    # zero bars; None -> observed-unique (the labels pass always passes None).
    code_universe_cell  = _build_code_universe(obs, cell_col_internal, cell_nominal_k)
    code_universe_niche = _build_code_universe(obs, niche_col_internal, niche_nominal_k)
    print(f"  {progress_label}: unique cell  values: "
          f"{len(code_universe_cell)}")
    print(f"  {progress_label}: unique niche values: "
          f"{len(code_universe_niche)}")
    if len(code_universe_cell) > 60 or len(code_universe_niche) > 60:
        print("  NOTE: >60 unique values in at least one column — bar-"
              "chart panels may become dense.")

    # ---- Per-section distributions ------------------------------------
    dist_cell:  Dict[str, np.ndarray] = {}
    dist_niche: Dict[str, np.ndarray] = {}
    for s_str, s_raw in zip(sections, sections_raw):
        dist_cell[s_str]  = _section_distribution(
            obs, s_raw, section_col, cell_col_internal,  code_universe_cell)
        dist_niche[s_str] = _section_distribution(
            obs, s_raw, section_col, niche_col_internal, code_universe_niche)

    # ---- Pairwise similarity ------------------------------------------
    n = len(sections)
    sim_cell  = np.full((n, n), np.nan, dtype=float)
    sim_niche = np.full((n, n), np.nan, dtype=float)
    for i, s1 in enumerate(sections):
        for j, s2 in enumerate(sections):
            if j < i:
                continue
            sim_cell[i, j]  = _jsd_similarity(dist_cell[s1],  dist_cell[s2])
            sim_niche[i, j] = _jsd_similarity(dist_niche[s1], dist_niche[s2])
    sim_cell  = np.where(np.isnan(sim_cell),  sim_cell.T,  sim_cell)
    sim_niche = np.where(np.isnan(sim_niche), sim_niche.T, sim_niche)

    # ---- Figures ------------------------------------------------------
    print(f"\n[{progress_label}] Distribution grid")
    render_distribution_grid(
        sections=sections,
        code_universe_cell=code_universe_cell,
        code_universe_niche=code_universe_niche,
        dist_cell=dist_cell,
        dist_niche=dist_niche,
        cell_code_col=cell_descriptor,
        niche_code_col=niche_descriptor,
        out_path_base=out_dir / grid_basename,
        cell_axis_title=cell_axis_title,
        niche_axis_title=niche_axis_title,
        suptitle=grid_suptitle,
        code_colored=code_colored,
    )

    print(f"\n[{progress_label}] Stacked code proportions")
    render_proportions_stacked(
        sections=sections,
        code_universe_cell=code_universe_cell,
        code_universe_niche=code_universe_niche,
        dist_cell=dist_cell,
        dist_niche=dist_niche,
        out_path_base=out_dir / stacked_basename,
        cell_axis_title=cell_axis_title,
        niche_axis_title=niche_axis_title,
        suptitle=stacked_suptitle,
        code_colored=code_colored,
    )

    print(f"\n[{progress_label}] Similarity heatmap")
    render_similarity_heatmaps(
        sections=sections,
        sim_cell=sim_cell,
        sim_niche=sim_niche,
        out_path_base=out_dir / heatmap_basename,
        donor_for_section=donor_for_section,
        donor_palette=donor_palette,
        query_sections=query_sections or None,
        cell_axis_title=cell_axis_title,
        niche_axis_title=niche_axis_title,
        suptitle=heatmap_suptitle,
    )

    if not skip_per_section:
        print(f"\n[{progress_label}] Per-section ranking figures")
        sub_dir = out_dir / ranking_subdirname
        for i, sec in enumerate(sections):
            safe = sec.replace("/", "_").replace(" ", "_")
            render_per_section_ranking(
                anchor=sec,
                sections=sections,
                sim_cell_row=sim_cell[i],
                sim_niche_row=sim_niche[i],
                out_path_base=sub_dir / f"section_{safe}",
                donor_for_section=donor_for_section,
                donor_palette=donor_palette,
                cell_axis_title=cell_axis_title,
                niche_axis_title=niche_axis_title,
                suptitle_prefix=ranking_suptitle_prefix,
            )

    # ---- CSV records --------------------------------------------------
    def _donor(s):
        return donor_for_section[s] if donor_for_section else ""
    records: List[Dict] = []
    for i, src in enumerate(sections):
        for j, tgt in enumerate(sections):
            for axis_name, mat in (
                ("cell",  sim_cell),
                ("niche", sim_niche),
            ):
                records.append({
                    "source_section": src,
                    "source_donor":   _donor(src),
                    "target_section": tgt,
                    "target_donor":   _donor(tgt),
                    "source_kind":    csv_kind_label,  # 'codes' / 'labels'
                    "code_kind":      axis_name,
                    "jsd_similarity": mat[i, j],
                })
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--adata", type=Path, required=True,
        help="Path to predicted_adata.h5ad produced by SQUINT.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory. Default: <adata>.parent/"
             "code_distribution_figures/",
    )
    p.add_argument(
        "--cell-code-col", "--cell-code-key", type=str, default=None,
        help=f"Key for cell codes. Searched in adata.obs THEN adata.obsm. "
             f"Default: first present from {list(DEFAULT_CELL_CODE_KEYS)}. "
             f"obs values are 1-D and ignore --cell-level; obsm values "
             f"are (N, L) 2-D and use --cell-level to pick the column.",
    )
    p.add_argument(
        "--niche-code-col", "--niche-code-key", type=str, default=None,
        help=f"Key for niche codes. Searched in adata.obs THEN adata.obsm. "
             f"Default: first present from {list(DEFAULT_NICHE_CODE_KEYS)}.",
    )
    p.add_argument(
        "--cell-level", type=str, default="0",
        help="When cell codes come from a 2-D obsm array, which level "
             "to plot. One of '0' / 'l0' / 'level_0' (default — K1 "
             "macro partition), '1' / 'l1' / 'level_1' (K2 sub-"
             "partition), or 'composite' (L1*max_L2 + L2, one bar per "
             "unique leaf). Ignored when the code source is an obs "
             "column (the level is then encoded in the column name).",
    )
    p.add_argument(
        "--niche-level", type=str, default="0",
        help="Same as --cell-level but for niche codes. Default: '0'.",
    )
    p.add_argument(
        "--cell-codebook-size", type=int, default=None,
        help="Nominal CELL codebook size (K) for the plotted level. The code "
             "axis then spans the FULL codebook 0..K-1 so DEAD codes (never "
             "assigned) show as zero-height bars — otherwise the axis only "
             "covers observed codes (e.g. 89 bars for a nominal-90 codebook "
             "with one dead code). Default: auto from "
             "adata.uns['codebook_sizes_cell'] for the resolved level; only "
             "needed for old artifacts without that metadata, or to override.",
    )
    p.add_argument(
        "--niche-codebook-size", type=int, default=None,
        help="Nominal NICHE codebook size (K) for the plotted level. See "
             "--cell-codebook-size.",
    )
    p.add_argument(
        "--section-col", type=str, default=None,
        help=f"obs column for section ids. Default: first present from "
             f"{list(DEFAULT_SECTION_COLS)}.",
    )
    p.add_argument(
        "--source-file-col", type=str, default="source_file",
        help="obs column to derive section DISPLAY labels from. The "
             "label is the substring BEFORE the first '+' (e.g. "
             "`Lung9_Rep1+SMI+Flat+data.tar.h5ad` -> `Lung9_Rep1`). "
             "Pass '' (empty string) to disable and use raw section "
             "ids as labels. Default: %(default)s.",
    )
    p.add_argument(
        "--donor-pattern", type=str, default=r"^(.*?)_Rep\d+$",
        help="Regex matched against each section label to extract a "
             "donor id (group 1). Sections sharing the same donor get "
             "the same colour in the similarity heatmap and per-section "
             "ranking. Default %(default)r matches `<donor>_Rep<N>` -> "
             "`<donor>`. Pass '' to disable donor colouring (each "
             "section is then its own 'donor').",
    )
    p.add_argument(
        "--section-order", type=str, default=None,
        help="Comma-separated explicit section order for the small-"
             "multiples grid and heatmap axes. Default: sorted (numeric "
             "if all-numeric, else lexical).",
    )
    p.add_argument(
        "--cell-label-col", type=str, default="cell_type",
        help="obs column for the GROUND-TRUTH cell-type labels. When "
             "present, the script ALSO runs the full distribution / "
             "similarity / ranking analysis on cell-type proportions "
             "(in parallel to the code-based analysis). Default: "
             "%(default)s. Pass '' to disable.",
    )
    p.add_argument(
        "--niche-label-col", type=str, default="niche",
        help="obs column for the GROUND-TRUTH niche labels. Pairs with "
             "--cell-label-col for the labels-based analysis pass. "
             "Default: %(default)s. Pass '' to disable.",
    )
    p.add_argument(
        "--skip-labels", action="store_true",
        help="Skip the labels-based analysis pass even if the cell- and "
             "niche-label columns are present in obs.",
    )
    p.add_argument(
        "--label-adata", type=Path, default=None,
        help="Optional path to a SINGLE external AnnData file that "
             "carries `cell_type` and `niche` labels for EVERY section "
             "(including held-out queries). When provided, the labels "
             "pass reads from this file instead of from the predicted "
             "AnnData's obs columns. Section matching uses the column "
             "named by --label-section-col (auto-detected from common "
             "names if not given). Example for chl59-8b_1p: "
             "/nfs/team361/sb75/DATASETS/bronze/chl59-8b_1p/"
             "nanostring_cosmx_human_nsclc_reference_query_mapping.h5ad",
    )
    p.add_argument(
        "--label-section-col", type=str, default=None,
        help="obs column in --label-adata used to identify the section "
             "for each row. Values are `astype(str)` and split on the "
             "first '+' before matching against the codes-pass section "
             "labels (so both `Lung5_Rep3+SMI+Flat+...` and plain "
             "`Lung5_Rep3` work). Default: auto-detect by trying "
             "(source_file, batch, patient, sample, section, "
             "adata_batch_id, sample_id, batch_id, donor) in order.",
    )
    p.add_argument(
        "--query-sections", type=str, default=None,
        help="Comma-separated section labels (post-source_file-prefix, "
             "e.g. 'Lung5_Rep3,Lung13') to mark as QUERY sections in "
             "the similarity heatmap. Tick labels for these sections "
             "are rendered in SQUINT-accent red and bold so the held-"
             "out targets are immediately legible alongside the donor-"
             "color strips. Default: no query highlight.",
    )
    p.add_argument(
        "--skip-per-section", action="store_true",
        help="Skip writing one ranking figure per section (useful for "
             "datasets with very many sections).",
    )
    args = p.parse_args(argv)

    if not args.adata.is_file():
        raise SystemExit(f"AnnData not found: {args.adata}")
    if args.out_dir is None:
        args.out_dir = args.adata.parent / "code_distribution_figures"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _apply_nature_style()

    # ---- Load AnnData (read obs only, lazily — full X not needed) -----
    print(f"Reading: {args.adata}")
    adata = ad.read_h5ad(args.adata)
    print(f"  n_obs = {adata.n_obs:,d}   n_vars = {adata.n_vars:,d}")

    # ---- Resolve code keys (obs OR obsm) ------------------------------
    # Cell codes
    if args.cell_code_col is None:
        cell_key, cell_src = _autodetect_code_key(
            adata, DEFAULT_CELL_CODE_KEYS, "cell-code")
    else:
        if args.cell_code_col in adata.obs.columns:
            cell_key, cell_src = args.cell_code_col, "obs"
        elif args.cell_code_col in adata.obsm.keys():
            cell_key, cell_src = args.cell_code_col, "obsm"
        else:
            raise SystemExit(
                f"--cell-code-col {args.cell_code_col!r} not in "
                f"adata.obs OR adata.obsm.\n"
                f"  obs:  {sorted(adata.obs.columns.tolist())}\n"
                f"  obsm: {sorted(adata.obsm.keys())}"
            )
    # Niche codes
    if args.niche_code_col is None:
        niche_key, niche_src = _autodetect_code_key(
            adata, DEFAULT_NICHE_CODE_KEYS, "niche-code")
    else:
        if args.niche_code_col in adata.obs.columns:
            niche_key, niche_src = args.niche_code_col, "obs"
        elif args.niche_code_col in adata.obsm.keys():
            niche_key, niche_src = args.niche_code_col, "obsm"
        else:
            raise SystemExit(
                f"--niche-code-col {args.niche_code_col!r} not in "
                f"adata.obs OR adata.obsm.\n"
                f"  obs:  {sorted(adata.obs.columns.tolist())}\n"
                f"  obsm: {sorted(adata.obsm.keys())}"
            )
    # Section column (always in obs)
    if args.section_col is None:
        args.section_col = _autodetect_section(adata.obs, DEFAULT_SECTION_COLS)
    elif args.section_col not in adata.obs.columns:
        raise SystemExit(
            f"--section-col {args.section_col!r} not in adata.obs.")

    cell_level  = _parse_level(args.cell_level)
    niche_level = _parse_level(args.niche_level)

    # Extract 1-D code vectors. `cell_desc` / `niche_desc` are display
    # strings used in panel column titles ("Cell codes (obsm['...'][:, 0])").
    cell_codes_1d,  cell_desc  = _resolve_code_vector(
        adata, cell_key,  cell_src,  cell_level,  kind="cell")
    niche_codes_1d, niche_desc = _resolve_code_vector(
        adata, niche_key, niche_src, niche_level, kind="niche")

    print(f"  cell-code:   {cell_desc}")
    print(f"  niche-code:  {niche_desc}")
    print(f"  section col: {args.section_col}")

    # Nominal codebook size for each plotted level -> the code axis spans the
    # FULL codebook (0..K-1) so DEAD codes show as zero bars. None => plot only
    # the observed-unique codes (previous behaviour; also used for composite).
    cell_nominal_k = _nominal_codebook_k(
        adata, "cell", cell_src, cell_key, cell_level, args.cell_codebook_size)
    niche_nominal_k = _nominal_codebook_k(
        adata, "niche", niche_src, niche_key, niche_level, args.niche_codebook_size)
    print(f"  cell  codebook axis span: "
          f"{cell_nominal_k if cell_nominal_k is not None else 'observed-only'}")
    print(f"  niche codebook axis span: "
          f"{niche_nominal_k if niche_nominal_k is not None else 'observed-only'}")

    # Build a fixed-schema working DataFrame so the downstream pipeline
    # (which still does `df[df[section_col] == section][code_col]`-style
    # lookups) doesn't have to know whether codes came from obs or obsm.
    obs = pd.DataFrame({
        args.section_col: adata.obs[args.section_col].values,
        "_cell_code":     cell_codes_1d,
        "_niche_code":    niche_codes_1d,
    })
    cell_code_col_internal  = "_cell_code"
    niche_code_col_internal = "_niche_code"

    # ---- Build section order ------------------------------------------
    raw_sections = obs[args.section_col].dropna().unique().tolist()
    # Try numeric sort first; fall back to lexical.
    try:
        sorted_sections = sorted(raw_sections, key=lambda v: float(v))
    except (TypeError, ValueError):
        sorted_sections = sorted(raw_sections, key=lambda v: str(v))

    if args.section_order:
        explicit = [s.strip() for s in args.section_order.split(",") if s.strip()]
        # Map back from string -> the raw section value in obs.
        raw_by_str = {_coerce_section_label(s): s for s in raw_sections}
        missing = [s for s in explicit if s not in raw_by_str]
        if missing:
            raise SystemExit(
                f"--section-order references sections not in obs: {missing}. "
                f"Available: {sorted(raw_by_str.keys())}"
            )
        sections_raw = [raw_by_str[s] for s in explicit]
    else:
        sections_raw = sorted_sections

    # ---- Derive section DISPLAY labels --------------------------------
    # Default: take the substring of `source_file` before the first `+`
    # (e.g. `Lung9_Rep1+SMI+Flat+data.tar.h5ad` -> `Lung9_Rep1`).
    # If --source-file-col is empty or the column is absent, fall back
    # to stringified raw section ids — the prior behaviour.
    src_col = args.source_file_col or None
    label_map = _derive_section_labels(
        adata.obs, args.section_col, sections_raw, src_col,
    )
    sections = [label_map[s] for s in sections_raw]
    if src_col and src_col in adata.obs.columns:
        print(f"  section labels (from obs['{src_col}'] prefix): {sections}")
    else:
        print(f"  sections: {sections}")

    # ---- Derive donor per section + donor palette ---------------------
    if args.donor_pattern:
        donor_for_section: Dict[str, str] = {
            s: _derive_donor(s, args.donor_pattern) for s in sections
        }
        # Stable insertion-order list of unique donors (preserves the
        # plotted section order in the legend).
        donors_ordered: List[str] = []
        for s in sections:
            d = donor_for_section[s]
            if d not in donors_ordered:
                donors_ordered.append(d)
        donor_palette = _donor_palette(donors_ordered)
        print(f"  donors detected: {donors_ordered}")
    else:
        donor_for_section = None
        donor_palette = None

    # ---- Resolve --query-sections (shared across both passes) ---------
    query_sections: List[str] = []
    if args.query_sections:
        requested = [
            s.strip() for s in args.query_sections.split(",") if s.strip()
        ]
        unknown = [s for s in requested if s not in sections]
        if unknown:
            raise SystemExit(
                f"--query-sections references labels not in the "
                f"resolved section list: {unknown}. Available: {sections}"
            )
        query_sections = requested
        print(f"  query sections (highlighted red): {query_sections}")

    # ---- Resolve labels-pass columns (cell_type + niche) --------------
    # The labels pass reads from either (a) the silver-tier per-section
    # files when --label-source-dir is set — recommended for chl59-
    # style splits where held-out sections have labels stripped in the
    # training AnnData; OR (b) the predicted AnnData's own obs columns.
    cell_label_col  = (args.cell_label_col or "").strip() or None
    niche_label_col = (args.niche_label_col or "").strip() or None
    run_labels_pass = not args.skip_labels and \
                      cell_label_col is not None and \
                      niche_label_col is not None
    use_external_labels = (
        run_labels_pass and args.label_adata is not None
    )

    if run_labels_pass and not use_external_labels:
        # Mode (b): read labels from predicted obs. Soft-skip if the
        # defaults are missing; hard-error if the user explicitly named
        # a non-default column that doesn't exist.
        missing = [c for c in (cell_label_col, niche_label_col)
                   if c not in adata.obs.columns]
        is_default_cell  = (cell_label_col  == "cell_type")
        is_default_niche = (niche_label_col == "niche")
        if missing and not (is_default_cell and is_default_niche):
            raise SystemExit(
                f"--cell-label-col / --niche-label-col reference obs "
                f"columns not in adata.obs: {missing}. Available: "
                f"{sorted(adata.obs.columns.tolist())}"
            )
        if missing:
            print(f"  labels pass: skipped (missing obs columns "
                  f"{missing}; defaults are 'cell_type' / 'niche'). "
                  f"Pass --label-adata to read from an external "
                  f"labels AnnData instead.")
            run_labels_pass = False

    all_records: List[Dict] = []

    # ---- Pass 1: code-based analysis (existing, default filenames) ----
    all_records.extend(_run_distribution_pass(
        obs=obs,
        section_col=args.section_col,
        sections=sections,
        sections_raw=sections_raw,
        cell_col_internal=cell_code_col_internal,
        niche_col_internal=niche_code_col_internal,
        cell_descriptor=cell_desc,
        niche_descriptor=niche_desc,
        out_dir=args.out_dir,
        grid_basename="code_distributions",
        stacked_basename="code_proportions_stacked",
        heatmap_basename="similarity_heatmap",
        ranking_subdirname="per_section_ranking",
        grid_suptitle="Code distribution per section",
        stacked_suptitle="Code proportions per section",
        heatmap_suptitle="Section similarity  (SQUINT codes)",
        ranking_suptitle_prefix="Sections most similar to",
        cell_axis_title="Cell codes",
        niche_axis_title="Niche codes",
        csv_kind_label="codes",
        donor_for_section=donor_for_section,
        donor_palette=donor_palette,
        query_sections=query_sections,
        skip_per_section=args.skip_per_section,
        progress_label="Codes",
        cell_nominal_k=cell_nominal_k,
        niche_nominal_k=niche_nominal_k,
    ))

    # ---- Pass 2: labels-based analysis --------------------------------
    if run_labels_pass:
        # Two possible label sources:
        #   (a) a separate "all-sections-labelled" AnnData passed via
        #       --label-adata. PREDICTED IS NOT USED for the labels in
        #       this path — only the external file is read. Sections,
        #       donors, palette, and order are STILL inherited from
        #       the codes pass so the two heatmaps line up axis-for-axis.
        #   (b) the predicted AnnData's own obs columns (--cell-label-col
        #       / --niche-label-col).
        if use_external_labels:
            print(f"\n  Labels pass: reading cell_type + niche from "
                  f"external AnnData ({args.label_adata}). Predicted "
                  f"obs labels are NOT consulted in this pass.")
            labels_obs, kept_sections = _build_labels_obs_from_external(
                label_adata_path=args.label_adata,
                sections=sections,
                label_section_col=args.label_section_col,
                cell_label_col=cell_label_col,
                niche_label_col=niche_label_col,
            )
            # Rename label columns to stable internal names.
            labels_obs = labels_obs.rename(columns={
                cell_label_col:  "_cell_label",
                niche_label_col: "_niche_label",
            })
            # Section ordering: tightened to the sections that actually
            # had matching rows in the external file. Display label IS
            # the raw key in the labels pass (no batch_id concept).
            sections_to_use     = kept_sections
            sections_raw_to_use = list(kept_sections)
            cell_desc_labels    = (
                f"external: obs['{cell_label_col}']  "
                f"(from {args.label_adata.name})"
            )
            niche_desc_labels   = (
                f"external: obs['{niche_label_col}']  "
                f"(from {args.label_adata.name})"
            )
            labels_pass_obs     = labels_obs
            labels_section_col  = "_section"
            # Re-use the codes-pass donor info / palette — sections in
            # both passes share display labels, so colours stay stable.
            donor_for_section_labels = donor_for_section
            donor_palette_pass       = donor_palette
        else:
            # Predicted-obs path: reuse the codes-pass sections + donors.
            obs = obs.copy()
            obs["_cell_label"]  = adata.obs[cell_label_col].values
            obs["_niche_label"] = adata.obs[niche_label_col].values
            cell_desc_labels    = f"obs['{cell_label_col}']"
            niche_desc_labels   = f"obs['{niche_label_col}']"
            sections_to_use     = sections
            sections_raw_to_use = sections_raw
            labels_pass_obs     = obs
            labels_section_col  = args.section_col
            donor_for_section_labels = donor_for_section
            donor_palette_pass       = donor_palette

        all_records.extend(_run_distribution_pass(
            obs=labels_pass_obs,
            section_col=labels_section_col,
            sections=sections_to_use,
            sections_raw=sections_raw_to_use,
            cell_col_internal="_cell_label",
            niche_col_internal="_niche_label",
            cell_descriptor=cell_desc_labels,
            niche_descriptor=niche_desc_labels,
            out_dir=args.out_dir,
            grid_basename="label_distributions",
            stacked_basename="label_proportions_stacked",
            heatmap_basename="similarity_heatmap_labels",
            ranking_subdirname="per_section_ranking_labels",
            grid_suptitle="Cell-type and niche distribution per section",
            stacked_suptitle="Cell-type and niche proportions per section",
            heatmap_suptitle="Section similarity  (cell-type & niche labels)",
            # labels are cell-type/niche NAMES (not code ids) -> keep the
            # rank-based palette, not the spatial code_index_plots palette.
            code_colored=False,
            ranking_suptitle_prefix=(
                "Sections with most similar cell-type & niche composition to"
            ),
            cell_axis_title="Cell types",
            niche_axis_title="Niches",
            csv_kind_label="labels",
            donor_for_section=donor_for_section_labels,
            donor_palette=donor_palette_pass,
            query_sections=query_sections,
            skip_per_section=args.skip_per_section,
            progress_label="Labels",
        ))

    # ---- Combined long CSV (both passes, one file) --------------------
    csv_path = args.out_dir / "similarity_long.csv"
    pd.DataFrame(all_records).to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")
    print(f"\nDone. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
