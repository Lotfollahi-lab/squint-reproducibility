#!/usr/bin/env python3
"""
EMA-decay (beta) SENSITIVITY figure for codebook health.

Addresses the reviewer comment: "EMA decay beta = 0.8 still asserted, not swept
... a short beta sweep would preempt the obvious question." Collates the s65
EMA-decay sweep (variants that differ ONLY in the codebook EMA decay, on the
s57_v19 FiLM-scale reference) plus the reference itself (beta=0.8) into ONE
figure: codebook health vs beta.

For each variant it reads the per-seed predicted_adata.h5ad run dirs, computes
per-(branch, level) codebook metrics with the SAME code as report_codebook_usage
(`_compute_rows`), tags each row with its beta, and plots — per branch (cell /
niche) and level (L0 / L1) — the two health metrics the reviewer named:

  * active-code fraction   (n_used / K; 1.0 = no dead codes)
  * normalized perplexity  (perplexity / K; 1.0 = perfectly uniform usage)

as a function of beta, with per-seed dots + the across-seed mean, and beta=0.8
(the adopted reference) highlighted. A flat, healthy plateau around 0.8 is the
evidence that the choice isn't a knife-edge.

Beta is parsed from the variant dir name (`...emadecay-0p9...` -> 0.9); the
`s57_v19_reference-filmscale...` dir maps to beta=0.8 (the reference).

Usage (auto-discover s65_v* + the s57_v19 reference under the dataset dir):
    python plot_ema_decay_sweep.py
    python plot_ema_decay_sweep.py --dataset mmb0-1b_smb1-1b_1p --out-name ema_decay_sweep
    # explicit variants (dir names under <artifacts>/<dataset>/):
    python plot_ema_decay_sweep.py --variant-dirs s65_v3_emadecay-0p9+... s57_v19_reference-...

Requires: anndata + numpy + pandas + matplotlib (all in the squint env), and
report_codebook_usage.py in the same dir. Run the s65 sweep first so the
per-seed predicted_adata.h5ad files exist.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"
DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"
# The reference variant IS beta=0.8 (not part of the s65 sweep dirs).
REFERENCE_VARIANT = "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p"
REFERENCE_BETA = 0.8

_METRICS = [
    ("active_fraction",       "Active-code fraction"),
    ("normalized_perplexity", "Normalized perplexity"),
]
_LEVELS = ["L0", "L1"]
_BRANCHES = ["cell", "niche"]


# ---------------------------------------------------------------------------
# beta parsing + variant discovery
# ---------------------------------------------------------------------------
def _beta_from_variant(name: str):
    """0.9 from '...emadecay-0p9...'; 0.8 for the s57_v19 reference; else None."""
    m = re.search(r"emadecay-(\d+p\d+)", name)
    if m:
        return float(m.group(1).replace("p", "."))
    if name.startswith("s57_v19_reference-filmscale"):
        return REFERENCE_BETA
    return None


def _discover_variants(artifacts_root: str, dataset: str) -> list:
    """Variant dirs under <artifacts>/<dataset>/ that carry a beta: every
    `s65_v*_emadecay-*` sweep dir + the s57_v19 reference (beta=0.8)."""
    base = os.path.join(artifacts_root, dataset)
    found = []
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if not os.path.isdir(os.path.join(base, d)):
                continue
            # per-seed predicted_adata.h5ad live in the NON-multiseed variant
            # dir; the `__multiseed` sibling only holds aggregated metrics.
            if d.endswith("__multiseed"):
                continue
            if d.startswith("s65_v") and "emadecay-" in d:
                found.append(d)
    if REFERENCE_VARIANT not in found and os.path.isdir(
            os.path.join(base, REFERENCE_VARIANT)):
        found.append(REFERENCE_VARIANT)
    return found


# ---------------------------------------------------------------------------
# Data collection (farm: needs anndata + report_codebook_usage._compute_rows)
# ---------------------------------------------------------------------------
def collect_rows(variant_dir: str, beta: float, sizes_override) -> list:
    """Per-seed codebook rows for one variant. Globs the variant's per-seed
    `<TS>/predicted_adata.h5ad` run dirs (same discovery as
    report_codebook_usage --all-seeds), computes rows via `_compute_rows`, and
    tags each with beta + seed_run. Returns [] (with a warning) if the variant
    dir or its predicted_adata files are missing."""
    import anndata as ad
    import report_codebook_usage as rcu

    if not os.path.isdir(variant_dir):
        print(f"  [skip] variant dir missing: {variant_dir}", file=sys.stderr)
        return []
    stamps = sorted(d for d in os.listdir(variant_dir)
                    if os.path.isfile(os.path.join(variant_dir, d,
                                                   "predicted_adata.h5ad")))
    if not stamps:
        print(f"  [skip] no <TS>/predicted_adata.h5ad under {variant_dir}",
              file=sys.stderr)
        return []
    out = []
    for ts in stamps:
        adata = ad.read_h5ad(os.path.join(variant_dir, ts, "predicted_adata.h5ad"))
        rows, _ = rcu._compute_rows(adata, sizes_override=sizes_override)
        for r in rows:
            r = dict(r)
            r["beta"] = float(beta)
            r["seed_run"] = ts
            out.append(r)
    print(f"  beta={beta:<5g}  {len(stamps)} seed(s)  ({os.path.basename(variant_dir)})")
    return out


# ---------------------------------------------------------------------------
# Rendering (pure pandas + matplotlib — no anndata; unit-testable)
# ---------------------------------------------------------------------------
def render_sweep(df: pd.DataFrame, out_base: str,
                 ref_beta: float = REFERENCE_BETA) -> None:
    """Grid of codebook-health metrics vs beta. Rows = metric, cols = branch,
    lines = RVQ level; per-seed dots + across-seed mean; ref beta highlighted.
    Writes <out_base>.{svg,png} + <out_base>.csv (the collated long table)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8, "axes.titlesize": 9, "figure.dpi": 150,
    })
    import matplotlib.pyplot as plt

    if df.empty:
        raise SystemExit("No codebook rows collected — refusing to plot empty figure.")

    lvl = df[df["level"].isin(_LEVELS)].copy()
    betas = sorted(lvl["beta"].dropna().unique())
    level_style = {"L0": dict(marker="o", ls="-"), "L1": dict(marker="s", ls="--")}
    level_color = {"L0": "#FF006E", "L1": "#3A86FF"}

    nrow, ncol = len(_METRICS), len(_BRANCHES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow),
                             squeeze=False, sharex=True)
    for ri, (mcol, mlabel) in enumerate(_METRICS):
        for ci, branch in enumerate(_BRANCHES):
            ax = axes[ri][ci]
            if ref_beta in betas:
                ax.axvline(ref_beta, color="0.6", ls=":", lw=1.0, zorder=1)
                if ri == 0:
                    ax.text(ref_beta, 1.01, "ref (0.8)", transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=6, color="0.4")
            for level in _LEVELS:
                sub = lvl[(lvl["branch"] == branch) & (lvl["level"] == level)]
                if sub.empty:
                    continue
                col = level_color[level]
                # per-seed dots
                ax.scatter(sub["beta"], sub[mcol], s=14, color=col, alpha=0.5,
                           edgecolors="white", linewidths=0.3, zorder=3)
                # across-seed mean line
                g = sub.groupby("beta")[mcol].agg(["mean", "std", "count"]).reindex(betas)
                yerr = g["std"].fillna(0.0).where(g["count"] > 1, 0.0)
                ax.errorbar(g.index, g["mean"], yerr=yerr, color=col, zorder=4,
                            capsize=2, lw=1.3, **level_style[level],
                            label=f"{branch.capitalize()} {level}")
            if ri == nrow - 1:
                ax.set_xlabel("EMA decay β")
            if ci == 0:
                ax.set_ylabel(mlabel)
            if ri == 0:
                ax.set_title(branch.capitalize())
            if mcol == "active_fraction":
                ax.set_ylim(-0.02, 1.05)
            ax.grid(True, lw=0.3, alpha=0.4)
            ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.suptitle("Codebook health vs EMA decay β  (per-seed dots, across-seed mean)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(f"{out_base}.{ext}", bbox_inches="tight")
        print(f"  -> {out_base}.{ext}")
    plt.close(fig)
    lvl.sort_values(["beta", "branch", "level", "seed_run"]).to_csv(
        f"{out_base}.csv", index=False)
    print(f"  -> {out_base}.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--variant-dirs", nargs="*", default=None,
                    help="Explicit variant dir names under <artifacts>/<dataset>/ "
                         "(default: auto-discover s65_v*_emadecay-* + the s57_v19 "
                         "reference).")
    ap.add_argument("--codebook-sizes-cell", default="30,90",
                    help="Nominal cell codebook sizes per level (comma ints). "
                         "Passed to _compute_rows so K is correct even if a run "
                         "lacks stored metadata. Default 30,90.")
    ap.add_argument("--codebook-sizes-niche", default="30,90")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <artifacts>/benchmarking/figures/.")
    ap.add_argument("--out-name", default="ema_decay_sweep")
    args = ap.parse_args(argv)

    variants = args.variant_dirs or _discover_variants(args.artifacts_root, args.dataset)
    variants = [(v, _beta_from_variant(v)) for v in variants]
    variants = [(v, b) for v, b in variants if b is not None]
    if not variants:
        raise SystemExit(
            f"No beta-tagged variants found under {args.artifacts_root}/{args.dataset} "
            "(expected s65_v*_emadecay-* dirs + the s57_v19 reference). Run the "
            "s65 sweep first: bash examples/submit_s65_emadecay_multiseed.sh")
    variants.sort(key=lambda vb: vb[1])
    sizes_override = {
        "cell":  [int(x) for x in args.codebook_sizes_cell.split(",") if x.strip()],
        "niche": [int(x) for x in args.codebook_sizes_niche.split(",") if x.strip()],
    }
    print(f"Artifacts: {args.artifacts_root}\nDataset:   {args.dataset}")
    print(f"Variants ({len(variants)}) betas: {[b for _, b in variants]}")

    rows = []
    for vdir, beta in variants:
        rows += collect_rows(os.path.join(args.artifacts_root, args.dataset, vdir),
                             beta, sizes_override)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows collected — did the s65 runs finish (per-seed "
                         "predicted_adata.h5ad present)?")

    out_dir = args.out_dir or os.path.join(args.artifacts_root, "benchmarking", "figures")
    print("\nRendering:")
    render_sweep(df, os.path.join(out_dir, args.out_name))
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
