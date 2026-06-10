#!/usr/bin/env python3
"""
chl59-8b_1p holdout UMAP — reference / replicate query / new donor query.

For the CosMx Lung dataset where Lung5_Rep3 (a technical replicate of
Lung5 from a donor that's already in training) and Lung13 (a different
donor) are held out, this script renders three UMAP panels from a
trained SQUINT run's `predicted_adata.h5ad`:

  1. UMAP coloured by SAMPLE ROLE
       - "Reference"          training batches (data_split == "train")
       - "Replicate query"    held-out replicate of an existing donor
       - "New donor query"    held-out donor not seen during training
  2. UMAP coloured by ASSIGNED CELL CODE  (`obs['cell_code_index']`)
  3. UMAP coloured by ASSIGNED NICHE CODE (`obs['neighborhood_code_index']`)

Panels 2 & 3 include cells from BOTH the training and held-out
batches — the codes are read straight from the predict-time output, so
the held-out cells already have indices assigned.

The same UMAP coordinates (computed once on `cell_emb`) back all three
panels, so spatial proximity between cells is consistent across the
figures.

Inputs (positional / required):
  --predicted-adata PATH   predicted_adata.h5ad written by SQUINT's
                           predict pipeline. Must contain `obsm['cell_emb']`,
                           `obs['cell_code_index']`, `obs['neighborhood_code_index']`,
                           `obs['data_split']`, and `obs['batch']` (or
                           --batch-key).

Outputs:
  Default --out-dir: /nfs/team361/sb75/squint-reproducibility/artifacts/
                     dataset_preparation/chl59-8b_1p
  Per panel:
    chl59_holdout_umap_by_sample_role.{svg,png}
    chl59_holdout_umap_by_cell_code.{svg,png}
    chl59_holdout_umap_by_niche_code.{svg,png}
  Plus a combined 1x3 figure: chl59_holdout_umap_combined.{svg,png}

GPU acceleration: if `rapids-singlecell` + `cupy` are importable AND
a CUDA device is visible, the UMAP step runs on GPU; otherwise it
falls back to scanpy CPU. Force the path via --use-gpu / --cpu-only.

Usage:
  python analysis/data_preparation/plot_chl59_holdout_umap.py \\
      --predicted-adata /nfs/.../<variant>/<TS>/inference/<run>/predicted_adata.h5ad
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Silence the two upstream FutureWarnings that also leak in run_squint.py
# (legacy Dask DataFrame, anndata `read_text` re-export) — they're
# noise that makes the diagnostic preamble harder to read.
warnings.filterwarnings(
    "ignore",
    message=r".*Importing read_text from `anndata` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*legacy Dask DataFrame implementation is deprecated.*",
    category=FutureWarning,
)

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Editable text in SVG (don't outline glyphs to paths). Same convention
# as plot_chl59_ground_truth.py / plot_holdout_regions.py.
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUT_DIR = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts/"
    "dataset_preparation/chl59-8b_1p"
)

# Substring patterns used to assign sample roles. Both match against the
# batch label (obs[batch_key], string-cast). Lung5_Rep3 is a replicate
# of Lung5 (same donor, different replicate); Lung13 is a distinct donor.
# Override on the CLI if your dataset uses different filenames.
DEFAULT_REPLICATE_QUERY_PATTERN = "Lung5_Rep3"
DEFAULT_NEW_DONOR_QUERY_PATTERN = "Lung13"

# Role palette — chosen so the two queries POP against the grey-of-reference
# but stay distinguishable from each other.
ROLE_PALETTE: Dict[str, str] = {
    "Reference":         "#9aa0a6",   # neutral grey  (large)
    "Replicate query":   "#3A86FF",   # blue           (Lung5_Rep3)
    "New donor query":   "#FF006E",   # magenta accent (Lung13)
}
ROLE_ORDER = ["Reference", "Replicate query", "New donor query"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save_dual(fig: plt.Figure, out_path_base: Path, **kwargs) -> None:
    """Save `fig` as both `.png` and `.svg` siblings sharing the same stem.
    Mirrors the helper used in plot_chl59_ground_truth.py."""
    out_path_base = Path(out_path_base)
    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.05)
    for ext in (".png", ".svg"):
        out = out_path_base.with_suffix(ext)
        fig.savefig(out, format=ext.lstrip("."), **kwargs)
        print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Sample-role resolution
# ---------------------------------------------------------------------------

def _resolve_sample_role(
        adata: ad.AnnData,
        batch_key: str,
        replicate_pattern: str,
        new_donor_pattern: str,
        data_split_key: str = "data_split",
    ) -> pd.Categorical:
    """Return per-cell role: Reference / Replicate query / New donor query.

    Logic (in priority order):
      1. If `obs[batch_key]` matches `new_donor_pattern` (case-insensitive
         substring) -> "New donor query".
      2. Else if `obs[batch_key]` matches `replicate_pattern` -> "Replicate query".
      3. Else "Reference".

    `data_split_key` (if present) is used as a CROSS-CHECK only — we
    assert that every "*query*" cell has `data_split == "test"` and vice
    versa, and warn on any mismatch. The role itself is derived from
    the batch label so that misconfigured data_split values don't
    silently mis-tag the figure.
    """
    if batch_key not in adata.obs.columns:
        raise SystemExit(
            f"adata.obs has no column '{batch_key}'. Available: "
            f"{list(adata.obs.columns)}.\n"
            f"Pass --batch-key explicitly if your column is named differently."
        )

    batch_str = adata.obs[batch_key].astype(str)
    rep_re = re.compile(re.escape(replicate_pattern), re.IGNORECASE)
    new_re = re.compile(re.escape(new_donor_pattern), re.IGNORECASE)

    new_mask = batch_str.str.contains(new_re, na=False)
    rep_mask = batch_str.str.contains(rep_re, na=False) & ~new_mask
    role = np.where(
        new_mask, "New donor query",
        np.where(rep_mask, "Replicate query", "Reference"),
    )

    # Cross-check against data_split, if it exists. The predict pipeline
    # stamps "test" on the held-out cells; mismatch means our pattern
    # didn't catch the right files (or the variant used a different
    # holdout set than this script assumes).
    if data_split_key in adata.obs.columns:
        ds = adata.obs[data_split_key].astype(str)
        is_test = (ds == "test").to_numpy()
        is_query = (role != "Reference")
        mism_query_not_test = is_query & ~is_test
        mism_test_not_query = is_test & ~is_query
        if mism_query_not_test.any() or mism_test_not_query.any():
            print(
                f"WARN: {data_split_key} <-> sample_role mismatch.\n"
                f"  cells tagged as a query but data_split != 'test': "
                f"{int(mism_query_not_test.sum())}\n"
                f"  cells in data_split == 'test' but no query pattern matched: "
                f"{int(mism_test_not_query.sum())}\n"
                f"  -> Check --replicate-query-pattern / --new-donor-query-pattern"
                f" against your obs['{batch_key}'] values."
            )
    else:
        print(
            f"NOTE: obs has no '{data_split_key}' column — skipping the "
            f"data_split <-> sample_role consistency check."
        )

    cat = pd.Categorical(role, categories=ROLE_ORDER, ordered=True)
    return cat


# ---------------------------------------------------------------------------
# UMAP computation
# ---------------------------------------------------------------------------

def _detect_gpu_backend() -> str:
    """Return 'rapids' if rapids-singlecell + cupy + GPU are all available,
    else 'scanpy'."""
    try:
        import rapids_singlecell  # noqa: F401
        import cupy  # noqa: F401
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return "rapids"
    except Exception:  # noqa: BLE001 — any failure -> CPU
        pass
    return "scanpy"


def _compute_umap_inplace(
        adata: ad.AnnData,
        emb_key: str,
        n_neighbors: int,
        backend: str,
    ) -> None:
    """Build kNN graph + UMAP coords IN-PLACE on `adata` (writes
    `adata.obsm['X_umap']`)."""
    import scanpy as sc

    if backend == "rapids":
        try:
            import cupy as cp
            import rapids_singlecell as rsc
            np_emb = np.asarray(adata.obsm[emb_key])
            adata.obsm[emb_key] = cp.asarray(np_emb)
            try:
                rsc.pp.neighbors(
                    adata, n_neighbors=n_neighbors, use_rep=emb_key,
                )
                rsc.tl.umap(adata)
                if "X_umap" in adata.obsm and hasattr(adata.obsm["X_umap"], "get"):
                    adata.obsm["X_umap"] = adata.obsm["X_umap"].get()
            finally:
                adata.obsm[emb_key] = np_emb
            return
        except (MemoryError, RuntimeError) as e:
            print(
                f"  rapids GPU UMAP failed ({type(e).__name__}: {e}); "
                f"falling back to scanpy CPU."
            )

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=emb_key)
    sc.tl.umap(adata)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _spot_size_for(n_cells: int) -> float:
    """Pick a scatter `s` (point area) based on cell count.

    Same heuristic as plot_chl59_ground_truth.py but for UMAP (which
    has a fixed coordinate range so density depends only on n_cells).
    Tested against the chl59 silver h5ads in our 6-inch figure layout.
    """
    if n_cells >= 200_000:
        return 0.5
    if n_cells >= 50_000:
        return 1.0
    if n_cells >= 10_000:
        return 2.0
    return 4.0


def _plot_role_panel(
        ax: plt.Axes,
        xy: np.ndarray,
        roles: pd.Categorical,
        spot_size: float,
        legend: bool = True,
    ) -> None:
    """Scatter cells coloured by sample role.

    Drawing order: Reference first (so the smaller query categories
    sit ON TOP and stay visible against the grey background of
    training cells).
    """
    from matplotlib.lines import Line2D

    for role in ROLE_ORDER:
        mask = (np.asarray(roles) == role)
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            c=ROLE_PALETTE[role], s=spot_size,
            linewidths=0, marker="o", rasterized=True,
            # Higher zorder for queries puts them above Reference.
            zorder=2 if role == "Reference" else 3,
        )
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Sample role", fontsize=10)
    if legend:
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=ROLE_PALETTE[role],
                   markeredgewidth=0, markersize=7,
                   label=f"{role} (n={int((np.asarray(roles) == role).sum()):,})")
            for role in ROLE_ORDER
            if (np.asarray(roles) == role).any()
        ]
        ax.legend(
            handles=handles, loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize=8,
            handletextpad=0.4, borderaxespad=0.0,
        )


def _categorical_cmap_for(n: int) -> List[Tuple[float, float, float]]:
    """Pick a high-distinguishability colour list of length `n` for a
    categorical (cell-code / niche-code) panel. Stitches tab20+20b+20c
    for up to 60; falls back to HSV for >60 (typical for RVQ composite
    codes which can run to ~1000s of distinct values)."""
    if n <= 20:
        return list(plt.get_cmap("tab20").colors[:n])
    pool: List[Tuple[float, float, float]] = []
    for name in ("tab20", "tab20b", "tab20c"):
        pool += list(plt.get_cmap(name).colors)
    if n <= len(pool):
        return pool[:n]
    cmap = plt.get_cmap("hsv")
    return [cmap((i % n) / n)[:3] for i in range(n)]


def _plot_code_panel(
        ax: plt.Axes,
        xy: np.ndarray,
        codes: np.ndarray,
        title: str,
        spot_size: float,
        legend: bool = False,
        max_legend_entries: int = 25,
    ) -> None:
    """Scatter cells coloured by an integer code (cell-code or niche-code).

    Codes are cast through the palette via `code % palette_size` so even
    multi-level / composite codes with thousands of distinct values
    still produce a stable, deterministic colour assignment. The
    legend is suppressed by default because typical chl59 SQUINT runs
    have 30 / 90 / 216k codes — too many to label individually.
    """
    from matplotlib.lines import Line2D

    codes = np.asarray(codes, dtype=int)
    uniq = np.unique(codes)
    palette = _categorical_cmap_for(len(uniq))
    code_to_colour = {int(c): palette[i % len(palette)] for i, c in enumerate(uniq)}

    colours = np.array([code_to_colour[int(c)] for c in codes])
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=colours, s=spot_size,
        linewidths=0, marker="o", rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{title}  (n_codes = {len(uniq)})", fontsize=10)

    if legend and len(uniq) <= max_legend_entries:
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=code_to_colour[int(c)],
                   markeredgewidth=0, markersize=6,
                   label=str(int(c)))
            for c in uniq
        ]
        ax.legend(
            handles=handles, loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize=6, ncol=2,
            handletextpad=0.3, borderaxespad=0.0,
            columnspacing=0.8,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_code_obs_key(adata: ad.AnnData, candidates: List[str]) -> Optional[str]:
    """Return the first obs column from `candidates` that exists in adata."""
    for k in candidates:
        if k in adata.obs.columns:
            return k
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--predicted-adata", type=Path, required=True,
                   help="Path to predicted_adata.h5ad written by SQUINT's predict pipeline.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--emb-key", type=str, default="cell_emb",
                   help="obsm key used as the UMAP input embedding "
                        "(default: cell_emb).")
    p.add_argument("--batch-key", type=str, default="batch",
                   help="obs column carrying the per-cell sample name "
                        "(e.g. 'Lung13', 'Lung5_Rep3'). Default: 'batch'. "
                        "Some predicted_adatas use 'adata_batch_id' (an "
                        "int); in that case set this AND make sure your "
                        "patterns map to those ints.")
    p.add_argument("--cell-code-key", type=str, default=None,
                   help="obs column for cell-side codes (default: auto-"
                        "detect cell_code_index / cell_code_indices[*]).")
    p.add_argument("--niche-code-key", type=str, default=None,
                   help="obs column for niche-side codes (default: auto-"
                        "detect neighborhood_code_index / neighborhood_code_indices[*]).")
    p.add_argument("--replicate-query-pattern", type=str,
                   default=DEFAULT_REPLICATE_QUERY_PATTERN,
                   help=f"Substring identifying the replicate-query batch "
                        f"(default: {DEFAULT_REPLICATE_QUERY_PATTERN}).")
    p.add_argument("--new-donor-query-pattern", type=str,
                   default=DEFAULT_NEW_DONOR_QUERY_PATTERN,
                   help=f"Substring identifying the new-donor-query batch "
                        f"(default: {DEFAULT_NEW_DONOR_QUERY_PATTERN}).")
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="kNN size for sc.pp.neighbors (default 15).")
    p.add_argument("--spot-size", type=float, default=None,
                   help="Override scatter point size. Default: auto-scale by n_cells.")
    backend_grp = p.add_mutually_exclusive_group()
    backend_grp.add_argument("--use-gpu", action="store_true",
                             help="Force rapids-singlecell GPU UMAP "
                                  "(errors out if unavailable).")
    backend_grp.add_argument("--cpu-only", action="store_true",
                             help="Force scanpy CPU UMAP.")
    args = p.parse_args(argv)

    # ---- Load ------------------------------------------------------------
    print(f"Loading {args.predicted_adata}")
    if not args.predicted_adata.is_file():
        raise SystemExit(f"File not found: {args.predicted_adata}")
    adata = ad.read_h5ad(args.predicted_adata)
    print(f"  n_obs = {adata.n_obs:,}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    print(f"  obs columns: {list(adata.obs.columns)}")

    if args.emb_key not in adata.obsm:
        raise SystemExit(
            f"adata.obsm has no key '{args.emb_key}'. Available: "
            f"{list(adata.obsm.keys())}. Pass --emb-key explicitly."
        )

    # ---- Resolve obs keys for cell + niche codes -------------------------
    cell_code_key = args.cell_code_key or _resolve_code_obs_key(
        adata,
        [
            "cell_code_index",
            "cell_code_indices[composite]",
            "cell_code_indices[level_0]",
        ],
    )
    if cell_code_key is None:
        raise SystemExit(
            "Could not find a cell-side code column in obs. Tried: "
            "cell_code_index, cell_code_indices[composite], "
            "cell_code_indices[level_0]. Pass --cell-code-key explicitly."
        )
    niche_code_key = args.niche_code_key or _resolve_code_obs_key(
        adata,
        [
            "neighborhood_code_index",
            "neighborhood_code_indices[composite]",
            "neighborhood_code_indices[level_0]",
        ],
    )
    if niche_code_key is None:
        raise SystemExit(
            "Could not find a niche-side code column in obs. Tried: "
            "neighborhood_code_index, neighborhood_code_indices[composite], "
            "neighborhood_code_indices[level_0]. Pass --niche-code-key explicitly."
        )
    print(f"  cell-code obs  : {cell_code_key}")
    print(f"  niche-code obs : {niche_code_key}")

    # ---- Resolve sample roles -------------------------------------------
    roles = _resolve_sample_role(
        adata,
        batch_key=args.batch_key,
        replicate_pattern=args.replicate_query_pattern,
        new_donor_pattern=args.new_donor_query_pattern,
    )
    counts = pd.Series(roles).value_counts().reindex(ROLE_ORDER, fill_value=0)
    print("Sample-role assignment:")
    for r in ROLE_ORDER:
        print(f"  {r:<20s} n = {int(counts[r]):>10,}")
    if int(counts["Replicate query"]) == 0 and int(counts["New donor query"]) == 0:
        raise SystemExit(
            "No cells matched either query pattern. Check "
            "--replicate-query-pattern / --new-donor-query-pattern against "
            f"unique values of obs['{args.batch_key}']: "
            f"{adata.obs[args.batch_key].astype(str).unique().tolist()}"
        )

    # Stamp roles on adata so the categorical is reusable downstream
    # (e.g. if the user re-runs this script after caching the predicted_adata).
    adata.obs["sample_role"] = roles

    # ---- Compute UMAP ----------------------------------------------------
    if args.cpu_only:
        backend = "scanpy"
    elif args.use_gpu:
        backend = _detect_gpu_backend()
        if backend != "rapids":
            raise SystemExit(
                "--use-gpu requested but rapids-singlecell / cupy / a CUDA "
                "device were not all available."
            )
    else:
        backend = _detect_gpu_backend()
    print(f"UMAP backend: {backend}")

    print(f"Computing UMAP on obsm['{args.emb_key}'] "
          f"(n_neighbors={args.n_neighbors}) ...")
    _compute_umap_inplace(
        adata, emb_key=args.emb_key,
        n_neighbors=args.n_neighbors, backend=backend,
    )
    xy = np.asarray(adata.obsm["X_umap"])

    # ---- Plot ------------------------------------------------------------
    spot = args.spot_size if args.spot_size is not None else _spot_size_for(adata.n_obs)
    print(f"Spot size: {spot}")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single panels (3 separate figures)
    print()
    print("Rendering individual panels:")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _plot_role_panel(ax, xy, roles, spot_size=spot, legend=True)
    _save_dual(fig, out_dir / "chl59_holdout_umap_by_sample_role")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    _plot_code_panel(ax, xy,
                     codes=adata.obs[cell_code_key].to_numpy(),
                     title=f"Cell code ({cell_code_key})",
                     spot_size=spot, legend=False)
    _save_dual(fig, out_dir / "chl59_holdout_umap_by_cell_code")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    _plot_code_panel(ax, xy,
                     codes=adata.obs[niche_code_key].to_numpy(),
                     title=f"Niche code ({niche_code_key})",
                     spot_size=spot, legend=False)
    _save_dual(fig, out_dir / "chl59_holdout_umap_by_niche_code")
    plt.close(fig)

    # Combined 1x3 figure (shared UMAP coords for cross-panel reading).
    print()
    print("Rendering combined 1x3 figure:")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    _plot_role_panel(axes[0], xy, roles, spot_size=spot, legend=True)
    _plot_code_panel(axes[1], xy,
                     codes=adata.obs[cell_code_key].to_numpy(),
                     title=f"Cell code ({cell_code_key})",
                     spot_size=spot, legend=False)
    _plot_code_panel(axes[2], xy,
                     codes=adata.obs[niche_code_key].to_numpy(),
                     title=f"Niche code ({niche_code_key})",
                     spot_size=spot, legend=False)
    fig.suptitle(
        f"chl59-8b_1p — UMAP on obsm['{args.emb_key}']  "
        f"(n_cells = {adata.n_obs:,})",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save_dual(fig, out_dir / "chl59_holdout_umap_combined")
    plt.close(fig)

    print()
    print("=" * 78)
    print(f"Done. Output dir: {out_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
