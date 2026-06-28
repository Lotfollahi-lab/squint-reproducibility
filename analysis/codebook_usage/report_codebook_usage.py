#!/usr/bin/env python3
"""
Report CODEBOOK USAGE for a trained SQUINT dual-VQ run (addresses the
reviewer comment: "Codebook usage is not reported ... can the authors report the
codebook usage (e.g., active-code fraction or perplexity) at the chosen
(L, K) setting, to confirm the [EMA-reinit collapse-prevention] mechanism is
working as intended?").

It reads a run's `predicted_adata.h5ad` (the deterministic argmin code
assignments produced at inference — no GPU / model reload needed) and, for
EACH codebook level of EACH branch (cell + niche RVQ), reports:

  * K                     codebook size at that level (the chosen K)
  * n_used                # distinct codes actually assigned
  * active_fraction       n_used / K                       (1.0 = no dead codes)
  * n_dead                K - n_used
  * perplexity            exp(entropy of the usage distribution), in [1, K]
  * normalized_perplexity perplexity / K                   (1.0 = perfectly uniform)
  * entropy_bits          Shannon entropy of usage (bits)
  * max_code_share        fraction of cells in the single most-used code
                          (a collapse-to-one-code indicator)

For the residual VQ it also reports a JOINT summary over the full code tuple
(L0, L1, ...): how many of the prod(K_level) possible discrete tokens are
realised, and the joint perplexity — the effective vocabulary size.

active-code fraction and perplexity are the two numbers the reviewer asked
for. A high active fraction (≈1.0, no dead codes) plus a perplexity that is a
large fraction of K together demonstrate the EMA dead-code reinitialisation
is preventing collapse.

Outputs (to --out-dir, default <run_dir>/codebook_usage_plots/):
  codebook_usage.csv     one row per (branch, level) + joint rows
  codebook_usage.json    same, machine-readable
  codebook_usage.png/svg per-(branch, level) sorted usage bar charts

Usage (auto-resolve the latest run of the default s49_v23 variant):
    python report_codebook_usage.py

Or point it at a specific artifact / run / timestamp:
    python report_codebook_usage.py --predicted-adata /path/to/predicted_adata.h5ad
    python report_codebook_usage.py --run-dir /path/to/<timestamp>
    python report_codebook_usage.py --variant <KEY> --timestamp 20260601_120000
    python report_codebook_usage.py --variant <KEY> --dataset mmb0-1b_smb1-1b_1p

Requirements: anndata, numpy, pandas, matplotlib (all in the squint env).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Defaults — the variant the reviewer asked about + standard farm layout.
DEFAULT_VARIANT = (
    "s49_v23_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+"
    "knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+"
    "diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"
)
DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"
DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"

# Per-branch storage keys in the predicted adata (with fallbacks).
_BRANCH_KEYS = {
    "cell":  {"uns": "Indices_cell",  "obsm": "cell_code_indices",
              "obs": "cell_code_index",  "sizes": "codebook_sizes_cell"},
    "niche": {"uns": "Indices_niche", "obsm": "neighborhood_code_indices",
              "obs": "neighborhood_code_index", "sizes": "codebook_sizes_niche"},
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _to_numpy(x) -> np.ndarray:
    """Coerce a torch.Tensor / np.ndarray / list to a numpy array."""
    if x is None:
        return None
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_predicted_adata(args) -> str:
    """Figure out which predicted_adata.h5ad to read from the CLI args."""
    if args.predicted_adata:
        return args.predicted_adata
    if args.run_dir:
        p = os.path.join(args.run_dir, "predicted_adata.h5ad")
        if not os.path.isfile(p):
            raise SystemExit(f"No predicted_adata.h5ad in --run-dir {args.run_dir}")
        return p
    # Resolve from <artifacts>/<dataset>/<variant>/<timestamp>/.
    variant_slug = args.variant.replace("/", "_").replace(" ", "_")
    base = os.path.join(args.artifacts_root, args.dataset, variant_slug)
    if not os.path.isdir(base):
        raise SystemExit(
            f"Variant dir not found: {base}\n"
            f"Check --artifacts-root / --dataset / --variant, or pass "
            f"--predicted-adata / --run-dir directly.")
    if args.timestamp and args.timestamp != "latest":
        run_dir = os.path.join(base, args.timestamp)
    else:
        stamps = sorted(d for d in os.listdir(base)
                        if os.path.isdir(os.path.join(base, d)))
        if not stamps:
            raise SystemExit(f"No timestamped run dirs under {base}")
        run_dir = os.path.join(base, stamps[-1])
        print(f"[usage] latest timestamp: {os.path.basename(run_dir)} "
              f"(of {len(stamps)} run(s))")
    p = os.path.join(run_dir, "predicted_adata.h5ad")
    if not os.path.isfile(p):
        raise SystemExit(f"No predicted_adata.h5ad in {run_dir}")
    return p


def _get_indices(adata, branch: str) -> np.ndarray:
    """Per-cell code indices for a branch, shape (N, num_levels)."""
    k = _BRANCH_KEYS[branch]
    idx = None
    if k["uns"] in adata.uns:
        idx = _to_numpy(adata.uns[k["uns"]])
    elif k["obsm"] in adata.obsm:
        idx = _to_numpy(adata.obsm[k["obsm"]])
    elif k["obs"] in adata.obs:
        idx = _to_numpy(adata.obs[k["obs"].values if hasattr(adata.obs[k["obs"]], "values") else k["obs"]])
    if idx is None:
        return None
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx[:, None]            # single-level -> (N, 1)
    return idx.astype(np.int64)


def _get_sizes(adata, branch: str, idx: np.ndarray) -> list:
    """Codebook size per level for a branch. Prefer stored metadata."""
    k = _BRANCH_KEYS[branch]
    sizes = None
    if k["sizes"] in adata.uns:
        sizes = adata.uns[k["sizes"]]
    elif branch == "niche" and "codebook_sizes" in adata.uns:   # legacy alias
        sizes = adata.uns["codebook_sizes"]
    if sizes is not None:
        sizes = [int(s) for s in np.asarray(sizes).ravel().tolist()]
    if not sizes or len(sizes) != idx.shape[1]:
        # Fallback: infer K per level from the max observed index (+1).
        # This UNDER-counts K if the top codes are never used, so it's a
        # last resort and is flagged in the output.
        sizes = [int(idx[:, q].max()) + 1 for q in range(idx.shape[1])]
        print(f"[usage] WARNING: codebook sizes for '{branch}' not found in "
              f"adata.uns; inferred {sizes} from max index (+1). Active "
              f"fraction will read 1.0 by construction — pass the real K via "
              f"the run's metadata for an accurate dead-code count.",
              file=sys.stderr)
    return sizes


def _level_metrics(idx_q: np.ndarray, K: int) -> dict:
    """active-code fraction + perplexity + collapse indicators for one level."""
    counts = np.bincount(idx_q, minlength=K).astype(np.float64)
    N = float(counts.sum())
    p = counts / N
    nz = p[p > 0]
    entropy_nats = float(-(nz * np.log(nz)).sum())
    n_used = int((counts > 0).sum())
    return {
        "n_cells": int(N),
        "K": int(K),
        "n_used": n_used,
        "active_fraction": n_used / K,
        "n_dead": int(K - n_used),
        "perplexity": float(np.exp(entropy_nats)),
        "normalized_perplexity": float(np.exp(entropy_nats) / K),
        "entropy_bits": entropy_nats / np.log(2.0),
        "max_code_share": float(p.max()),
    }, counts


def _joint_metrics(idx: np.ndarray, sizes: list) -> dict:
    """Effective vocabulary over the full residual code tuple (L0, L1, ...)."""
    # Encode the multi-level tuple as a single mixed-radix integer id.
    radix = np.asarray(sizes, dtype=np.int64)
    flat = np.zeros(idx.shape[0], dtype=np.int64)
    for q in range(idx.shape[1]):
        flat = flat * radix[q] + idx[:, q]
    uniq, cnt = np.unique(flat, return_counts=True)
    total_possible = int(np.prod(radix))
    p = cnt.astype(np.float64) / cnt.sum()
    entropy_nats = float(-(p * np.log(p)).sum())
    return {
        "n_cells": int(idx.shape[0]),
        "K": total_possible,
        "n_used": int(uniq.size),
        "active_fraction": uniq.size / total_possible,
        "n_dead": int(total_possible - uniq.size),
        "perplexity": float(np.exp(entropy_nats)),
        "normalized_perplexity": float(np.exp(entropy_nats) / total_possible),
        "entropy_bits": entropy_nats / np.log(2.0),
        "max_code_share": float(cnt.max() / cnt.sum()),
    }


# ----------------------------------------------------------------------------
# Per-run report (reusable for single-run AND --all-seeds)
# ----------------------------------------------------------------------------
_COLS = ["branch", "level", "K", "n_used", "active_fraction", "n_dead",
         "perplexity", "normalized_perplexity", "entropy_bits",
         "max_code_share", "n_cells"]


def _compute_rows(adata):
    """Return (rows, usage_for_plot) of per (branch, level) codebook metrics."""
    rows, usage_for_plot = [], {}
    for branch in ("cell", "niche"):
        idx = _get_indices(adata, branch)
        if idx is None:
            print(f"[usage] WARNING: no code indices for '{branch}' "
                  f"(uns/{_BRANCH_KEYS[branch]['uns']}, "
                  f"obsm/{_BRANCH_KEYS[branch]['obsm']}). Skipping.", file=sys.stderr)
            continue
        sizes = _get_sizes(adata, branch, idx)
        for q in range(idx.shape[1]):
            m, counts = _level_metrics(idx[:, q], sizes[q])
            rows.append({"branch": branch, "level": f"L{q}", **m})
            usage_for_plot[(branch, q)] = counts
        if idx.shape[1] > 1:
            rows.append({"branch": branch, "level": "joint",
                         **_joint_metrics(idx, sizes)})
    return rows, usage_for_plot


def _plot_usage(usage_for_plot, df, out_dir, stem, suptitle):
    """Sorted per-code usage bar charts (per branch/level) -> png/svg/pdf."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        # editable text in Illustrator (svg=named fonts, pdf/ps=embedded TrueType)
        matplotlib.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        items = sorted(usage_for_plot.keys())
        ncol = max(len(set(q for _, q in items)), 1)
        nrow = max(len(set(b for b, _ in items)), 1)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow),
                                 squeeze=False)
        branch_order = ["cell", "niche"]
        for (branch, q), counts in usage_for_plot.items():
            r = branch_order.index(branch) if branch in branch_order else 0
            ax = axes[r][q]
            order = np.argsort(counts)[::-1]
            frac = counts[order] / counts.sum()
            ax.bar(np.arange(len(frac)), frac, width=1.0)
            K = len(counts)
            n_used = int((counts > 0).sum())
            ax.axhline(1.0 / K, ls="--", lw=0.8, color="0.4")  # uniform 1/K
            mrow = df[(df.branch == branch) & (df.level == f"L{q}")].iloc[0]
            ax.set_title(f"{branch.capitalize()} L{q}: K={K}, Used={n_used} "
                         f"({mrow.active_fraction:.0%}), "
                         f"Perplexity={mrow.perplexity:.1f}/{K}", fontsize=9)
            ax.set_xlabel("Code (Sorted by Usage)")
            ax.set_ylabel("Fraction of Cells")
        for r in range(nrow):
            for c in range(ncol):
                if (branch_order[r] if r < len(branch_order) else None, c) \
                        not in usage_for_plot:
                    axes[r][c].axis("off")
        fig.suptitle(suptitle, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "svg", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=150,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"[usage] wrote {os.path.join(out_dir, stem + '.png')} (+ .svg, .pdf)")
    except Exception as e:                       # plotting is a nicety; never fail
        print(f"[usage] plot skipped ({type(e).__name__}: {e})", file=sys.stderr)


def _report_one(path, out_dir, variant, stem, make_plot):
    """Compute + write csv/json + plot for one run. Returns the metrics df (or None)."""
    import anndata as ad
    import pandas as pd
    print(f"[usage] reading {path}")
    adata = ad.read_h5ad(path)
    rows, usage_for_plot = _compute_rows(adata)
    if not rows:
        print(f"[usage] no code indices in {path} (continuous / old artifact?) — skip",
              file=sys.stderr)
        return None
    df = pd.DataFrame(rows)[_COLS]
    df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as fh:
        json.dump({"source": path, "variant": variant,
                   "n_cells": int(adata.n_obs), "metrics": rows}, fh, indent=2)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(df.to_string(index=False))
    if make_plot:
        _plot_usage(usage_for_plot, df, out_dir, stem,
                    "SQUINT Codebook Usage — Sorted Per-Code Assignment "
                    "Frequency (Dashed = Uniform 1/K)")
    return df


def _print_interpretation(df):
    per_level = df[df["level"].str.startswith("L")]
    print("\nInterpretation:")
    print("  active_fraction       -> 1.00 = every code used (no dead codes).")
    print("  normalized_perplexity -> 1.00 = perfectly uniform; >> 1/K = broad.")
    print("  max_code_share        -> ~1.0 = collapse onto a single code.")
    if not per_level.empty:
        print(f"  min active_fraction = {per_level['active_fraction'].min():.3f}; "
              f"min normalized_perplexity = "
              f"{per_level['normalized_perplexity'].min():.3f}.")


def _plot_aggregate(alldf, out_dir):
    """Across-seed bars (mean + per-seed dots + SD) for active_fraction and
    normalized_perplexity, per (branch, level)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        groups = (alldf[["branch", "level"]].drop_duplicates()
                  .sort_values(["branch", "level"]).itertuples(index=False))
        groups = [(b, l) for b, l in groups]
        xlabels = [f"{b[0].upper()}-{l}" for b, l in groups]
        x = np.arange(len(groups))
        metrics = [("active_fraction", "Active fraction"),
                   ("normalized_perplexity", "Normalized perplexity")]
        fig, axes = plt.subplots(1, 2, figsize=(max(6, 1.1 * len(groups)), 4.2),
                                 squeeze=False)
        for ax, (col, title) in zip(axes[0], metrics):
            means, pts = [], []
            for b, l in groups:
                v = alldf[(alldf.branch == b) & (alldf.level == l)][col].to_numpy()
                v = v[np.isfinite(v)]
                means.append(float(v.mean()) if v.size else np.nan)
                pts.append(v)
            errs = [float(v.std(ddof=1)) if v.size > 1 else 0.0 for v in pts]
            ax.bar(x, means, yerr=errs, width=0.66, color="#FF006E", alpha=0.85,
                   error_kw=dict(ecolor="0.25", elinewidth=0.9, capsize=2.5))
            for xi, v in zip(x, pts):
                if v.size:
                    jit = np.linspace(-0.14, 0.14, v.size) if v.size > 1 else np.zeros(1)
                    ax.scatter(np.full(v.size, xi) + jit, v, s=16,
                               facecolor="#7A0033", edgecolor="white", linewidths=0.4,
                               zorder=5)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7.5)
            ax.set_ylabel(title)
            ax.set_title(title, fontsize=9)
        fig.suptitle("SQUINT Codebook Usage Across Seeds (mean ± SD, dots = seeds)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for ext in ("png", "svg", "pdf"):
            fig.savefig(os.path.join(out_dir, f"codebook_usage_aggregate.{ext}"),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[usage] wrote {os.path.join(out_dir, 'codebook_usage_aggregate.png')} "
              f"(+ .svg, .pdf)")
    except Exception as e:
        print(f"[usage] aggregate plot skipped ({type(e).__name__}: {e})",
              file=sys.stderr)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("run selection (pick one; else auto-resolve)")
    src.add_argument("--predicted-adata", default=None,
                     help="Path to a predicted_adata.h5ad directly.")
    src.add_argument("--run-dir", default=None,
                     help="A <timestamp> run dir containing predicted_adata.h5ad.")
    src.add_argument("--variant", default=DEFAULT_VARIANT,
                     help="Variant key (default: the s49_v23 mouse-brain winner).")
    src.add_argument("--dataset", default=DEFAULT_DATASET,
                     help="Dataset tag dir under the artifacts root.")
    src.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    src.add_argument("--timestamp", default="latest",
                     help="Run timestamp, or 'latest' (default).")
    src.add_argument("--all-seeds", action="store_true",
                     help="Process EVERY <timestamp> run dir under the variant "
                          "(one per seed): per-seed reports + an across-seed "
                          "aggregate (mean ± SD CSV + plot). out-dir default "
                          "becomes <variant>/codebook_usage_plots/.")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir (default: <run_dir>/codebook_usage_plots/, or "
                         "<variant>/codebook_usage_plots/ for --all-seeds).")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip the usage bar-chart figures.")
    args = ap.parse_args(argv)

    if args.all_seeds:
        _run_all_seeds(args)
        return

    path = _resolve_predicted_adata(args)
    out_dir = args.out_dir or os.path.join(os.path.dirname(path), "codebook_usage_plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[usage] out -> {out_dir}\n")
    df = _report_one(path, out_dir, args.variant, "codebook_usage", not args.no_plot)
    if df is None:
        raise SystemExit("No codebook indices found in the predicted adata — "
                         "continuous/non-VQ run, or an old artifact without codes?")
    _print_interpretation(df)
    print("\n[usage] DONE")


def _run_all_seeds(args):
    """Per-seed codebook reports for ALL timestamp dirs under the variant, plus
    an across-seed aggregate."""
    import pandas as pd
    variant_slug = args.variant.replace("/", "_").replace(" ", "_")
    base = os.path.join(args.artifacts_root, args.dataset, variant_slug)
    if not os.path.isdir(base):
        raise SystemExit(f"Variant dir not found: {base}")
    stamps = sorted(d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d))
                    and os.path.isfile(os.path.join(base, d, "predicted_adata.h5ad")))
    if not stamps:
        raise SystemExit(f"No <timestamp>/predicted_adata.h5ad run dirs under {base}")
    out_dir = args.out_dir or os.path.join(base, "codebook_usage_plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[usage] --all-seeds: {len(stamps)} run(s) under\n  {base}\n  out -> {out_dir}\n")
    per_seed = []
    for ts in stamps:
        print(f"\n===== {ts} =====")
        path = os.path.join(base, ts, "predicted_adata.h5ad")
        df = _report_one(path, out_dir, args.variant,
                         f"codebook_usage_{ts}", not args.no_plot)
        if df is not None:
            df = df.copy()
            df.insert(0, "seed_run", ts)
            per_seed.append(df)
    if not per_seed:
        raise SystemExit("No codebook indices found in any run.")
    alldf = pd.concat(per_seed, ignore_index=True)
    alldf.to_csv(os.path.join(out_dir, "codebook_usage_per_seed.csv"), index=False)
    agg = (alldf.groupby(["branch", "level"])
           .agg(n=("active_fraction", "size"), K=("K", "first"),
                active_fraction_mean=("active_fraction", "mean"),
                active_fraction_std=("active_fraction", "std"),
                normalized_perplexity_mean=("normalized_perplexity", "mean"),
                normalized_perplexity_std=("normalized_perplexity", "std"),
                perplexity_mean=("perplexity", "mean"),
                perplexity_std=("perplexity", "std"))
           .reset_index())
    agg.to_csv(os.path.join(out_dir, "codebook_usage_aggregate.csv"), index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 78)
    print(f"ACROSS-SEED AGGREGATE (n={alldf['seed_run'].nunique()} seeds; mean ± SD)")
    print("=" * 78)
    print(agg.to_string(index=False))
    _print_interpretation(alldf)
    if not args.no_plot:
        _plot_aggregate(alldf, out_dir)
    print(f"\n[usage] wrote {out_dir}/codebook_usage_per_seed.csv + "
          f"codebook_usage_aggregate.csv")
    print("[usage] DONE")


if __name__ == "__main__":
    main()
