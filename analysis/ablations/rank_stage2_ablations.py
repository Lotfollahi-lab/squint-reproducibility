#!/usr/bin/env python
"""
Rank the stage-2 (decode / MaskGIT) ablation variants by RMSE.

Every variant under ``--ablation-dir`` is a folder that contains one or more
per-seed run dirs, each with a decode->Pearson metrics CSV::

    <ablation-dir>/
      <variant>/
        <TS>_seed<N>/
          metrics/
            per_seed_pearson_reconstruction.csv
        <TS>_seed<M>/
          metrics/
            per_seed_pearson_reconstruction.csv
      ...

Each CSV is written by ``squint/examples/stage2_decode_pearson.py`` and carries
the FULL metric panel per (branch, split): one row per
(axis, transform, gene_subset) with ``pearson_mean / spearman_mean /
mse_mean / rmse_mean`` (+ ``*_median``), plus entry-wise zero/nonzero
AUROC/AUPRC rows. Columns::

    seed, branch, split, axis, transform, gene_subset,
    pearson_mean, pearson_median, spearman_mean, spearman_median,
    mse_mean, mse_median, rmse_mean, n_cells, n_genes            # corr rows
    seed, branch, split, axis="entrywise", transform="counts",
    gene_subset, auroc_zero, auprc_zero, n_cells, n_genes        # entry-wise

RMSE has several flavours (gene_wise / cell_wise x log1p / raw x
all / hvg{N} / markers). This script:

  1. Aggregates ``rmse_mean`` across seeds per variant (mean / std / sem / ci95),
     de-duplicating re-runs of the same seed (keeps the latest timestamp).
  2. Ranks variants on ONE primary RMSE slice (``--split/--axis/--transform/
     --gene-subset``; default: test / cell_wise / log1p / all — per-held-out-cell
     imputation error on the variance-stabilised scale). Lower is better.
  3. Also computes a CONSENSUS rank = mean rank across every RMSE flavour in the
     chosen split, so you can see whether the winner is robust to the slice.

Outputs (to ``--out-dir``, default ``<ablation-dir>/_ranking``):
  stage2_ablation_rmse_ranking.csv    primary slice, ranked (best first)
  stage2_ablation_rmse_consensus.csv  mean-rank-across-flavours, ranked
  stage2_ablation_rmse_all.csv        wide: variant x every RMSE flavour (mean)

Usage:
  python analysis/ablations/rank_stage2_ablations.py
  python analysis/ablations/rank_stage2_ablations.py --axis gene_wise --transform raw
  python analysis/ablations/rank_stage2_ablations.py --split all --gene-subset markers
  python analysis/ablations/rank_stage2_ablations.py --include 'decode*'   # subset
"""
from __future__ import annotations

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_ABLATION_DIR = Path(
    "/nfs/team361/sb75/squint-reproducibility/artifacts/"
    "mmb0-1b_smb1-1b_1p/stage2-ablation"
)
DEFAULT_CSV_NAME = "per_seed_pearson_reconstruction.csv"
# The (axis, transform, gene_subset) grid the writer emits for RMSE. cell_wise
# only carries gene_subset="all" (markers/hvg are gene_wise-only). Used to build
# the wide table + the consensus rank. hvg is matched by PREFIX ("hvg") because
# the subset label embeds the count (e.g. "hvg50").
_RMSE_FLAVOURS = [
    ("gene_wise", "log1p", "all"),
    ("gene_wise", "log1p", "hvg"),
    ("gene_wise", "log1p", "markers"),
    ("gene_wise", "raw",   "all"),
    ("gene_wise", "raw",   "hvg"),
    ("gene_wise", "raw",   "markers"),
    ("cell_wise", "log1p", "all"),
    ("cell_wise", "raw",   "all"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_variant(variant_dir: Path, csv_name: str) -> pd.DataFrame:
    """Concatenate every ``*/metrics/<csv_name>`` under ``variant_dir``, one
    per seed-run dir. De-duplicate on (seed, split, axis, transform,
    gene_subset): if the same seed was re-run, keep the row from the LATEST
    timestamped run dir (dir names sort as YYYYMMDD_HHMMSS_seedN)."""
    files = sorted(variant_dir.glob(f"*/metrics/{csv_name}"),
                   key=lambda p: p.parent.parent.name, reverse=True)
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception as e:  # noqa: BLE001
            print(f"    ! failed to read {f}: {e}", file=sys.stderr)
            continue
        if d.empty:
            continue
        d = d.copy()
        d["_run_dir"] = f.parent.parent.name          # for provenance/debug
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "seed" not in df.columns:
        df["seed"] = 0
    key = [c for c in ("seed", "branch", "split", "axis", "transform",
                       "gene_subset") if c in df.columns]
    # files are sorted latest-first -> keep='first' keeps the latest re-run.
    df = df.drop_duplicates(subset=key, keep="first")
    return df


def load_all(ablation_dir: Path, csv_name: str, branch: str,
             include: Optional[str], exclude: Optional[str]
             ) -> Dict[str, pd.DataFrame]:
    """Return {variant_name: per-seed metrics DataFrame (branch-filtered)}."""
    out: Dict[str, pd.DataFrame] = {}
    variant_dirs = sorted(p for p in ablation_dir.iterdir() if p.is_dir())
    for vd in variant_dirs:
        name = vd.name
        if name.startswith("_"):                       # skip _ranking / _* helpers
            continue
        if include and not fnmatch(name, include):
            continue
        if exclude and fnmatch(name, exclude):
            continue
        df = _load_variant(vd, csv_name)
        if df.empty:
            print(f"  [{name:<28s}] MISSING ({csv_name} not found)")
            continue
        if "branch" in df.columns:
            df = df[df["branch"] == branch]
        n_seeds = df["seed"].nunique() if "seed" in df.columns else 0
        print(f"  [{name:<28s}] {n_seeds} seed(s)")
        if not df.empty:
            out[name] = df
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _ci95(v: np.ndarray) -> float:
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return 0.0
    sem = v.std(ddof=1) / np.sqrt(n)
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:  # noqa: BLE001
        t = 1.96
    return float(t * sem)


def _slice(df: pd.DataFrame, split: str, axis: str, transform: str,
           gene_subset: str, value_col: str = "rmse_mean") -> np.ndarray:
    """Per-seed ``value_col`` for one (split, axis, transform, gene_subset)
    slice. ``gene_subset='hvg'`` matches any ``hvg{N}`` by prefix."""
    if value_col not in df.columns:
        return np.array([])
    d = df
    for col, val in (("split", split), ("axis", axis), ("transform", transform)):
        if col in d.columns:
            d = d[d[col] == val]
    if "gene_subset" in d.columns:
        if gene_subset == "hvg":
            d = d[d["gene_subset"].astype(str).str.startswith("hvg")]
        else:
            d = d[d["gene_subset"] == gene_subset]
    v = pd.to_numeric(d.get(value_col, pd.Series(dtype=float)), errors="coerce")
    return v[v.notna()].to_numpy()


def _agg(v: np.ndarray) -> dict:
    n = int(v.size)
    return {
        "n_seeds": n,
        "mean": float(np.mean(v)) if n else float("nan"),
        "std": float(np.std(v, ddof=1)) if n > 1 else 0.0,
        "sem": float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "ci95": _ci95(v),
    }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def build_primary_ranking(data: Dict[str, pd.DataFrame], split: str, axis: str,
                          transform: str, gene_subset: str) -> pd.DataFrame:
    """Rank variants by mean RMSE on ONE slice (ascending = best first).
    Also carries the co-located Pearson/Spearman means for context."""
    rows = []
    for name, df in data.items():
        rmse = _slice(df, split, axis, transform, gene_subset, "rmse_mean")
        if rmse.size == 0:
            continue
        a = _agg(rmse)
        pear = _slice(df, split, axis, transform, gene_subset, "pearson_mean")
        spear = _slice(df, split, axis, transform, gene_subset, "spearman_mean")
        rows.append({
            "variant": name,
            "n_seeds": a["n_seeds"],
            "rmse_mean": a["mean"], "rmse_std": a["std"],
            "rmse_sem": a["sem"], "rmse_ci95": a["ci95"],
            "pearson_mean": float(np.mean(pear)) if pear.size else float("nan"),
            "spearman_mean": float(np.mean(spear)) if spear.size else float("nan"),
        })
    cols = ["rank", "variant", "n_seeds", "rmse_mean", "rmse_std", "rmse_sem",
            "rmse_ci95", "pearson_mean", "spearman_mean"]
    if not rows:                                   # no variant had this slice
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_values("rmse_mean", kind="stable").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_wide(data: Dict[str, pd.DataFrame], split: str) -> pd.DataFrame:
    """variant x every RMSE flavour (mean over seeds) for the chosen split."""
    rows = []
    for name, df in data.items():
        row = {"variant": name}
        for axis, transform, gs in _RMSE_FLAVOURS:
            v = _slice(df, split, axis, transform, gs, "rmse_mean")
            col = f"{axis}|{transform}|{gs}"
            row[col] = float(np.mean(v)) if v.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def build_consensus(wide: pd.DataFrame) -> pd.DataFrame:
    """Mean rank across all RMSE flavours (rank 1 = lowest RMSE in that column).
    Only columns present for a variant contribute to its mean rank."""
    ranks = wide.rank(axis=0, method="min", ascending=True)   # lower RMSE -> rank 1
    cons = pd.DataFrame({
        "mean_rank": ranks.mean(axis=1, skipna=True),
        "n_flavours": ranks.notna().sum(axis=1),
        "best_flavours": (ranks == 1).sum(axis=1),
    })
    cons = cons.sort_values(["mean_rank", "variant"] if "variant" in cons.columns
                            else "mean_rank", kind="stable")
    cons = cons.reset_index()  # variant back as a column
    cons.insert(0, "rank", np.arange(1, len(cons) + 1))
    return cons


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------

def _print_primary(rank_df: pd.DataFrame, split: str, axis: str,
                   transform: str, gene_subset: str) -> None:
    print(f"\n=== RMSE ranking  (split={split}, {axis}, {transform}, "
          f"{gene_subset}; lower is better) ===")
    if rank_df.empty:
        print("  (no variants had this RMSE slice)")
        return
    print(f"  {'#':>2}  {'variant':<30s} {'RMSE':>10s} {'±ci95':>8s} "
          f"{'n':>3s}  {'Pearson':>8s} {'Spearman':>8s}")
    for _, r in rank_df.iterrows():
        print(f"  {int(r['rank']):>2d}  {r['variant']:<30s} "
              f"{r['rmse_mean']:>10.4f} {r['rmse_ci95']:>8.4f} "
              f"{int(r['n_seeds']):>3d}  {r['pearson_mean']:>8.4f} "
              f"{r['spearman_mean']:>8.4f}")


def _print_consensus(cons_df: pd.DataFrame, split: str) -> None:
    print(f"\n=== Consensus RMSE rank  (mean rank across all RMSE flavours, "
          f"split={split}) ===")
    if cons_df.empty:
        print("  (nothing to rank)")
        return
    print(f"  {'#':>2}  {'variant':<30s} {'mean_rank':>9s} "
          f"{'#flav':>5s} {'#best':>5s}")
    for _, r in cons_df.iterrows():
        print(f"  {int(r['rank']):>2d}  {r['variant']:<30s} "
              f"{r['mean_rank']:>9.2f} {int(r['n_flavours']):>5d} "
              f"{int(r['best_flavours']):>5d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR,
                   help=f"Dir of stage-2 ablation variants. Default: {DEFAULT_ABLATION_DIR}")
    p.add_argument("--csv-name", type=str, default=DEFAULT_CSV_NAME,
                   help=f"Per-seed metrics filename. Default: {DEFAULT_CSV_NAME}")
    p.add_argument("--branch", type=str, default="cell", choices=["cell", "niche"],
                   help="Metric branch (stage-2 decode is cell-only). Default: cell.")
    p.add_argument("--split", type=str, default="test",
                   help="Split to rank on. 'test' = held-out region (imputation). "
                        "Default: test.")
    p.add_argument("--axis", type=str, default="cell_wise",
                   choices=["cell_wise", "gene_wise"],
                   help="RMSE axis for the PRIMARY ranking. Default: cell_wise "
                        "(per-held-out-cell error across genes).")
    p.add_argument("--transform", type=str, default="log1p",
                   choices=["log1p", "raw"],
                   help="Scale for the PRIMARY ranking. Default: log1p "
                        "(variance-stabilised; raw is count-depth dominated).")
    p.add_argument("--gene-subset", type=str, default="all",
                   help="Gene subset for the PRIMARY ranking ('all', 'hvg', "
                        "'markers'). Default: all. Note cell_wise only has 'all'.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir (default: <ablation-dir>/_ranking).")
    p.add_argument("--include", type=str, default=None,
                   help="Only variants whose name matches this glob (e.g. 'decode*').")
    p.add_argument("--exclude", type=str, default=None,
                   help="Skip variants whose name matches this glob.")
    args = p.parse_args(argv)

    if not args.ablation_dir.is_dir():
        raise SystemExit(f"--ablation-dir not found: {args.ablation_dir}")
    if args.out_dir is None:
        args.out_dir = args.ablation_dir / "_ranking"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ablation dir: {args.ablation_dir}")
    print(f"CSV:          {args.csv_name}  (branch={args.branch})")
    print("\nLoading variants:")
    data = load_all(args.ablation_dir, args.csv_name, args.branch,
                    args.include, args.exclude)
    if not data:
        raise SystemExit("No variants with metrics found.")

    rank_df = build_primary_ranking(
        data, args.split, args.axis, args.transform, args.gene_subset)
    wide = build_wide(data, args.split)
    cons = build_consensus(wide)

    _print_primary(rank_df, args.split, args.axis, args.transform, args.gene_subset)
    _print_consensus(cons, args.split)

    rank_csv = args.out_dir / "stage2_ablation_rmse_ranking.csv"
    cons_csv = args.out_dir / "stage2_ablation_rmse_consensus.csv"
    wide_csv = args.out_dir / "stage2_ablation_rmse_all.csv"
    rank_df.to_csv(rank_csv, index=False)
    cons.to_csv(cons_csv, index=False)
    wide.round(6).to_csv(wide_csv)
    print("\nWrote:")
    for c in (rank_csv, cons_csv, wide_csv):
        print(f"  -> {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
