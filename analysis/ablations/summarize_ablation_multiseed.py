#!/usr/bin/env python3
"""
Multi-seed ablation DATA SUMMARY — aggregate per-seed metrics across a set of
variants' multi-seed sweeps into one tidy table (mean ± 95% CI per metric),
with significance vs a reference variant. Lightweight: reads the already-
aggregated per_seed_*.csv files (pandas/numpy/scipy only — no anndata,
matplotlib or sklearn).

For each variant it reads
    <artifacts>/<dataset>/<variant>*__multiseed/<latest_TS>/metrics/
        per_seed_niche_identification.csv   (NMI / ARI by code_key + label_key)
        per_seed_batch_integration.csv      (iLISI / MMD by emb_key)
and summarises 8 metrics, split into the two axes of the trade-off:
    resolution : Cell NMI, Cell ARI, Niche NMI, Niche ARI   (↑ better)
    integration: Cell iLISI (↑), Cell MMD (↓), Niche iLISI (↑), Niche MMD (↓)

Pick a named preset with --set. The DEFAULT is "coupling" = EVERY cell/niche
coupling experiment in one table (s56 trunk-sharing + s58 info-flow/
complementarity + s59 soft-L2/parameter-efficient + s60 novel cross-branch),
all vs the decoupled s55_v3 reference. Individual presets: s55 (cross-batch-MNN
weight sweep vs s49_v23), s56 / s58 / s59 / s60 (the coupling families), s57
(all s51/s52/s54 ablations on the cross-batch spine vs s55_v3). Override with
--variants / --reference for an arbitrary set.

Outputs (to --out, default <artifacts>/<dataset>/_ablation_summary/):
    ablation_summary_long.csv   one row per (variant, metric): n, mean, std,
                                sem, ci95, p_vs_ref, stars
    ablation_summary_wide.csv   variants × metrics, "mean ± ci95" cells
and prints the wide table.

Usage:
    python summarize_ablation_multiseed.py                       # ALL coupling experiments (default)
    python summarize_ablation_multiseed.py --set s60            # novel cross-branch coupling
    python summarize_ablation_multiseed.py --set s59            # soft-L2 + param-efficient
    python summarize_ablation_multiseed.py --set s58            # info-flow / complementarity
    python summarize_ablation_multiseed.py --set s56            # encoder coupling methods
    python summarize_ablation_multiseed.py --set s55            # cross-batch weight sweep
    python summarize_ablation_multiseed.py --set s57            # all ablations, cross spine
    python summarize_ablation_multiseed.py \
        --variants s51_v1_ s51_v3_ s51_v4_ --reference s51_v1_
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"
DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"

CELL_CODE_KEY = "cell_code_indices[level_0]"
NICHE_CODE_KEY = "neighborhood_code_indices[level_0]"
CELL_LABELS = ("cell_type", "cell_types", "annotation")
NICHE_LABELS = ("niche", "Sub_molecular_tissue_region", "ccf_region_name",
                "spatial_cluster")

# (display name, higher_is_better) for the niche_identification table
NICHE_METRICS = [
    ("Cell NMI", CELL_CODE_KEY, "cell", "NMI", True),
    ("Cell ARI", CELL_CODE_KEY, "cell", "ARI", True),
    ("Niche NMI", NICHE_CODE_KEY, "niche", "NMI", True),
    ("Niche ARI", NICHE_CODE_KEY, "niche", "ARI", True),
]
# (display name, emb_key, metric_tag, higher_is_better) for batch_integration
BATCH_METRICS = [
    ("Cell iLISI", "cell_emb", "iLISI", True),
    ("Cell MMD", "cell_emb", "MMD", False),
    ("Niche iLISI", "neighborhood_emb", "iLISI", True),
    ("Niche MMD", "neighborhood_emb", "MMD", False),
]
METRIC_ORDER = [m[0] for m in NICHE_METRICS] + [m[0] for m in BATCH_METRICS]

# ---------------------------------------------------------------------------
# Named preset sweeps. Each is (list of (prefix, label), reference_prefix).
# Pick one with --set; override with --variants / --reference.
#
# s55 — cross-batch-MNN weight/k sweep vs the s49_v23 within-batch reference.
S55_SET = [
    ("s49_v23_", "Reference (wt_cross=0)"),
    ("s55_v1_", "cross wt=1 k=1"),
    ("s55_v2_", "cross wt=5 k=1"),
    ("s55_v3_", "cross wt=10 k=1"),
    ("s55_v4_", "cross wt=5 k=2"),
    ("s55_v5_", "cross wt=2 k=1 floor=0.5"),
]
S55_REFERENCE = "s49_v23_"

# s56 — encoder cell/niche COUPLING-METHOD sweep on the cross-batch-MNN spine.
# Reference = s55_v3 (the decoupled endpoint that all s56 variants modify).
S56_SET = [
    ("s55_v3_", "Decoupled (ref)"),
    ("s56_v1_", "Coupled (shared trunk)"),
    ("s56_v2_", "Y-shape 192/64"),
    ("s56_v3_", "Y-shape 128/128"),
    ("s56_v4_", "Y-shape 64/192"),
    ("s56_v5_", "Cross-stitch (scalar)"),
    ("s56_v6_", "Cross-stitch (per-ch)"),
    ("s56_v7_", "Stop-gradient (coupled)"),
    ("s56_v8_", "Soft L2 coupling"),
]
S56_REFERENCE = "s55_v3_"

# s57 — SELF-CONTAINED PAPER SET. Everything lives under the s57 namespace so
# you can run only `s57_v*`. Reference = s57_v19 (cross-MNN decoupled spine).
# v19 reference; v20/v21 contrastive axis; v1-v18 component ablations;
# v22-v25 coupling mechanisms; v26/v27 two-separate-models; v28 continuous.
# NOTE: the continuous row (v28) has placeholder code-based NMI/ARI — for the
# fair continuous-vs-VQ comparison use compare_discrete_vs_continuous.py
# (embedding path) with v28 (continuous) vs v20 (discrete VQ).
S57_SET = [
    ("s57_v19_", "Reference: FiLM scale-only"),
    ("s57_v20_", "Contrastive within-batch"),
    ("s57_v21_", "Contrastive none"),
    ("s57_v1_", "No adjacency"),
    ("s57_v2_", "No decoder cov"),
    ("s57_v3_", "GNN 2 layers"),
    ("s57_v4_", "knn8"),
    ("s57_v5_", "knn16/sampler8"),
    ("s57_v6_", "knn24"),
    ("s57_v7_", "Cell RVQ 30/10"),
    ("s57_v8_", "Cell RVQ 30/30"),
    ("s57_v9_", "Cell RVQ 30/300"),
    ("s57_v10_", "Niche RVQ 30/10"),
    ("s57_v11_", "Niche RVQ 30/30"),
    ("s57_v12_", "Niche RVQ 30/300"),
    ("s57_v13_", "Cell RVQ 10/30"),
    ("s57_v14_", "Cell RVQ 90/30"),
    ("s57_v15_", "Cell RVQ 300/30"),
    ("s57_v16_", "Niche RVQ 10/30"),
    ("s57_v17_", "Niche RVQ 90/30"),
    ("s57_v18_", "Niche RVQ 300/30"),
    ("s57_v25_", "Coupling: decoupled"),
    ("s57_v22_", "Coupling: coupled"),
    ("s57_v23_", "Coupling: cross-stitch"),
    ("s57_v24_", "Coupling: affine adapter"),
    ("s57_v26_", "Two-model: cell-only"),
    ("s57_v27_", "Two-model: niche-only"),
    ("s57_v28_", "Continuous latent (vs v29)"),
    ("s57_v29_", "Discrete VQ ref (continuous pair)"),
    ("s57_v30_", "L=1 VQ K=2700 (vs L=2)"),
    ("s57_v31_", "L=1 VQ K=120 (vs L=2)"),
    ("s57_v32_", "L=1 VQ K=30 (vs L=2)"),
]
S57_REFERENCE = "s57_v19_"

# s58 — information-flow / complementarity coupling (beyond trunk-sharing).
# Reference = s55_v3 (the decoupled baseline these variants modify).
S58_SET = [
    ("s55_v3_", "Decoupled (ref)"),
    ("s58_v1_", "Compose z_q_cell"),
    ("s58_v2_", "Compose z_q_cell proj64"),
    ("s58_v3_", "Compose z_mlp (cont.)"),
    ("s58_v4_", "Compose z_q_cell no-detach"),
    ("s58_v5_", "Disentangle w=100"),
    ("s58_v6_", "Disentangle w=1000"),
    ("s58_v7_", "Compose + disentangle"),
]
S58_REFERENCE = "s55_v3_"

# s59 — soft-L2 weight sweep + parameter-efficient coupling (shared trunk +
# per-branch adapter). Reference = s55_v3 (decoupled).
S59_SET = [
    ("s55_v3_", "Decoupled (ref)"),
    ("s56_v1_", "Coupled (no adapter)"),
    ("s56_v8_", "Soft L2 w=1e-3"),
    ("s59_v1_", "Soft L2 w=1e-4"),
    ("s59_v2_", "Soft L2 w=1e-2"),
    ("s59_v3_", "Soft L2 w=1e-1"),
    ("s59_v4_", "Coupled + affine (~50%)"),
    ("s59_v5_", "Coupled + LoRA r16 (~52%)"),
    ("s59_v6_", "Coupled + MLP head (~80%)"),
]
S59_REFERENCE = "s55_v3_"

# s60 — novel cross-branch couplings (VQ/codebook-level, conditional, attention,
# alignment, domain-borrowed). Reference = s55_v3 (decoupled).
S60_SET = [
    ("s55_v3_", "Decoupled (ref)"),
    ("s60_v1_", "Shared token"),
    ("s60_v2_", "Cross-branch residual VQ"),
    ("s60_v3_", "Shared codebook"),
    ("s60_v4_", "Cell-cond niche (bias)"),
    ("s60_v5_", "Cell-cond niche (FiLM)"),
    ("s60_v6_", "Niche attends cell cb"),
    ("s60_v7_", "Mutual alignment"),
    ("s60_v8_", "Nbr-expr augment"),
]
S60_REFERENCE = "s55_v3_"

# s62 — follow-ups to the on-par / slightly-better threads (Y-shape specialised,
# cell-cond FiLM, multi-head attention). Includes the s56/s60 anchors they tweak.
S62_SET = [
    ("s55_v3_", "Decoupled (ref)"),
    ("s56_v4_", "s56 Y-shape 64/192 (anchor)"),
    ("s62_v1_", "Y-shape 48/208"),
    ("s62_v2_", "Y-shape 32/224"),
    ("s62_v3_", "Y-shape 16/240"),
    ("s60_v5_", "s60 cell-cond FiLM (anchor)"),
    ("s62_v4_", "Cell-cond FiLM pre-VQ"),
    ("s62_v5_", "Cell-cond FiLM scale-only"),
    ("s62_v6_", "Cell-cond FiLM continuous"),
    ("s60_v6_", "s60 niche-attends (anchor)"),
    ("s62_v7_", "Niche-attends multi-head"),
    ("s62_v8_", "Y-shape 64/192 + FiLM combo"),
]
S62_REFERENCE = "s55_v3_"

# "coupling" — EVERY cell/niche coupling experiment in one table (s56 trunk-
# sharing + s58 info-flow/complementarity + s59 soft-L2/param-efficient + s60
# novel cross-branch), all vs the decoupled s55_v3 reference. Built from the
# per-family sets (so it never drifts); family-prefixed labels; the per-family
# reference rows are dropped and a single decoupled ref is kept at the top;
# s59's echoes of s56_v1/s56_v8 are de-duplicated.
def _family_rows(set_list, fam, drop=()):
    return [(p, f"{fam}: {lbl}") for p, lbl in set_list
            if p != "s55_v3_" and p not in drop]

COUPLING_SET = (
    [("s55_v3_", "Decoupled (ref)")]
    + _family_rows(S56_SET, "s56")
    + _family_rows(S58_SET, "s58")
    + _family_rows(S59_SET, "s59", drop=("s56_v1_", "s56_v8_"))
    + _family_rows(S60_SET, "s60")
    + _family_rows(S62_SET, "s62", drop=("s56_v4_", "s60_v5_", "s60_v6_"))
)
COUPLING_REFERENCE = "s55_v3_"

# "joint_vs_separate" — the reviewer's baseline: does the JOINT model beat two
# strong SEPARATE single-task models? Read s61_v1's CELL columns vs the joint
# Cell columns, and s61_v2's NICHE columns vs the joint Niche columns (the
# off-task columns of the single-task models are untrained — ignore them). The
# shared-encoder rows (coupled / affine adapter) carry the efficiency case
# (both tasks at ~half the encoder params). Reference = the joint SQUINT.
JOINT_VS_SEPARATE_SET = [
    ("s55_v3_", "SQUINT joint (decoupled)"),
    ("s61_v1_", "Separate cell-only (CELL cols)"),
    ("s61_v2_", "Separate niche-only (NICHE cols)"),
    ("s56_v1_", "SQUINT shared-trunk (~50% params)"),
    ("s59_v4_", "SQUINT shared+affine (~50% params)"),
]
JOINT_VS_SEPARATE_REFERENCE = "s55_v3_"

# "coupling_compare" — the 5 conceptually-distinct coupling mechanisms for the
# paper figure (matches plot_ablations.py axis_10): decoupled (ref) vs coupled
# vs cross-stitch vs coupled+affine vs cell-cond FiLM scale-only (the winner).
COUPLING_COMPARE_SET = [
    ("s57_v19_", "Cell-cond FiLM (scale)"),   # reference (the SQUINT coupling)
    ("s57_v25_", "Decoupled"),
    ("s57_v22_", "Coupled (shared trunk)"),
    ("s57_v23_", "Cross-stitch"),
    ("s57_v24_", "Coupled + affine"),
]
COUPLING_COMPARE_REFERENCE = "s57_v19_"

# s63 — ALL ablations re-run on the NEW DEFAULT spine (cross-batch MNN + cell-cond
# FiLM scale-only, == s62_v5). Reference = s62_v5 (the un-ablated default).
S63_SET = [
    ("s62_v5_", "Default: FiLM-scale (ref)"),
    ("s63_v1_", "No adjacency"),
    ("s63_v2_", "No decoder cov"),
    ("s63_v3_", "GNN 2 layers"),
    ("s63_v4_", "knn8"),
    ("s63_v5_", "knn16/sampler8"),
    ("s63_v6_", "knn24"),
    ("s63_v7_", "Cell RVQ 30/10"),
    ("s63_v8_", "Cell RVQ 30/30"),
    ("s63_v9_", "Cell RVQ 30/300"),
    ("s63_v10_", "Niche RVQ 30/10"),
    ("s63_v11_", "Niche RVQ 30/30"),
    ("s63_v12_", "Niche RVQ 30/300"),
    ("s63_v13_", "Cell RVQ 10/30"),
    ("s63_v14_", "Cell RVQ 90/30"),
    ("s63_v15_", "Cell RVQ 300/30"),
    ("s63_v16_", "Niche RVQ 10/30"),
    ("s63_v17_", "Niche RVQ 90/30"),
    ("s63_v18_", "Niche RVQ 300/30"),
]
S63_REFERENCE = "s62_v5_"

NAMED_SETS = {
    "s55": (S55_SET, S55_REFERENCE),
    "s56": (S56_SET, S56_REFERENCE),
    "s57": (S57_SET, S57_REFERENCE),
    "s58": (S58_SET, S58_REFERENCE),
    "s59": (S59_SET, S59_REFERENCE),
    "s60": (S60_SET, S60_REFERENCE),
    "s62": (S62_SET, S62_REFERENCE),
    "s63": (S63_SET, S63_REFERENCE),
    "coupling": (COUPLING_SET, COUPLING_REFERENCE),
    "coupling_compare": (COUPLING_COMPARE_SET, COUPLING_COMPARE_REFERENCE),
    "joint_vs_separate": (JOINT_VS_SEPARATE_SET, JOINT_VS_SEPARATE_REFERENCE),
}
DEFAULT_SET_NAME = "coupling_compare"


# ---------------------------------------------------------------------------
def _resolve_metrics_dir(prefix, artifacts_root, dataset):
    """Latest <TS>/metrics dir under <root>/<dataset>/<prefix>*__multiseed/."""
    base = Path(artifacts_root) / dataset
    cands = sorted(d for d in base.glob(f"{prefix}*__multiseed") if d.is_dir())
    if not cands:
        return None
    if len(cands) > 1:
        print(f"  WARN: {prefix!r} matched {len(cands)} sweeps; using {cands[0].name}",
              file=sys.stderr)
    for ts in sorted((p for p in cands[0].iterdir() if p.is_dir()),
                     key=lambda p: p.name, reverse=True):
        m = ts / "metrics"
        if m.is_dir() and (
            (m / "per_seed_niche_identification.csv").is_file()
            or (m / "per_seed_batch_integration.csv").is_file()
        ):
            return m
    return None


def _niche_vals(metrics_dir, code_key, branch, col):
    f = metrics_dir / "per_seed_niche_identification.csv"
    if not f.is_file():
        return np.array([])
    d = pd.read_csv(f)
    if "split" in d.columns:
        d = d[d["split"] == "all"]
    d = d[d["code_key"] == code_key]
    pref = CELL_LABELS if branch == "cell" else NICHE_LABELS
    have = set(d["label_key"].unique()) if "label_key" in d.columns else set()
    lk = next((l for l in pref if l in have), None)
    if lk is None:
        return np.array([])
    d = d[d["label_key"] == lk]
    return pd.to_numeric(d[col], errors="coerce").dropna().to_numpy()


def _batch_vals(metrics_dir, emb_key, tag):
    f = metrics_dir / "per_seed_batch_integration.csv"
    if not f.is_file():
        return np.array([])
    d = pd.read_csv(f)
    d = d[(d["emb_key"] == emb_key) & (d["metric"] == tag)]
    return pd.to_numeric(d["score"], errors="coerce").dropna().to_numpy()


def _per_seed_for_metric(metrics_dir, name):
    for disp, ck, branch, col, _hib in NICHE_METRICS:
        if disp == name:
            return _niche_vals(metrics_dir, ck, branch, col)
    for disp, ek, tag, _hib in BATCH_METRICS:
        if disp == name:
            return _batch_vals(metrics_dir, ek, tag)
    raise ValueError(name)


def _ci95(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return 0.0
    sem = v.std(ddof=1) / np.sqrt(n)
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        t = 1.96
    return float(t * sem)


def _pvalue(a, b, test):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        from scipy import stats
        if test == "mannwhitney":
            return float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])
        return float(stats.ttest_ind(a, b, equal_var=False)[1])
    except Exception:
        return float("nan")


def _stars(p):
    if p != p:
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="preset", choices=list(NAMED_SETS),
                    default=DEFAULT_SET_NAME,
                    help=f"Named preset sweep to summarise (default {DEFAULT_SET_NAME!r}). "
                         f"s55=cross-batch weight sweep; s56=encoder coupling methods; "
                         f"s57=all ablations on the cross-batch spine; s58=information-"
                         f"flow / complementarity coupling; s59=soft-L2 sweep + "
                         f"parameter-efficient coupling; s60=novel cross-branch coupling; "
                         f"coupling=ALL coupling experiments (s56+s58+s59+s60) in one "
                         f"table. Ignored when --variants is given.")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Variant key prefixes (e.g. s56_v1_). Overrides --set.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional display labels matching --variants.")
    ap.add_argument("--reference", default=None,
                    help="Reference variant prefix for significance. Default: the "
                         "chosen preset's reference (s56 -> s55_v3_), or the first "
                         "--variants entry when --variants is given.")
    ap.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--test", choices=["ttest", "mannwhitney"], default="ttest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.variants:
        pairs = list(zip(args.variants,
                         args.labels if args.labels else args.variants))
        # default reference = explicit --reference, else the first variant.
        reference = args.reference or args.variants[0]
    else:
        pairs, set_reference = NAMED_SETS[args.preset]
        reference = args.reference or set_reference

    out_dir = Path(args.out) if args.out else (
        Path(args.artifacts_root) / args.dataset / "_ablation_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference per-seed values (for significance).
    ref_dir = _resolve_metrics_dir(reference, args.artifacts_root, args.dataset)
    ref_vals = {}
    if ref_dir is not None:
        ref_vals = {m: _per_seed_for_metric(ref_dir, m) for m in METRIC_ORDER}
    else:
        print(f"WARN: reference {reference!r} has no sweep — p-values will be NaN",
              file=sys.stderr)

    rows = []
    set_label = "custom" if args.variants else args.preset
    print(f"set={set_label}  dataset={args.dataset}  test={args.test}  "
          f"reference={reference}\n")
    for prefix, label in pairs:
        md = _resolve_metrics_dir(prefix, args.artifacts_root, args.dataset)
        if md is None:
            print(f"  SKIP {label} ({prefix}): no multiseed per_seed data")
            continue
        is_ref = (prefix == reference)
        for m in METRIC_ORDER:
            v = _per_seed_for_metric(md, m)
            if v.size == 0:
                continue
            p = (float("nan") if is_ref or not ref_vals
                 else _pvalue(ref_vals.get(m, np.array([])), v, args.test))
            rows.append({
                "variant": label, "prefix": prefix, "metric": m,
                "n": int(v.size), "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
                "ci95": _ci95(v), "p_vs_ref": p, "stars": "" if is_ref else _stars(p),
            })
        print(f"  OK   {label} ({prefix})  <- {md}")

    if not rows:
        raise SystemExit("No data found for any variant.")
    long_df = pd.DataFrame(rows)

    # Wide table: variants × metrics, "mean ± ci (stars)".
    def _cell(r):
        s = f"{r['mean']:.3f}±{r['ci95']:.3f}"
        return s + (f" {r['stars']}" if r["stars"] and r["stars"] != "ns" else "")
    long_df["_cell"] = long_df.apply(_cell, axis=1)
    var_order = [lbl for _, lbl in pairs if lbl in set(long_df["variant"])]
    wide = (long_df.pivot_table(index="variant", columns="metric", values="_cell",
                                aggfunc="first")
            .reindex(index=var_order, columns=METRIC_ORDER))

    long_out = out_dir / "ablation_summary_long.csv"
    wide_out = out_dir / "ablation_summary_wide.csv"
    long_df.drop(columns="_cell").to_csv(long_out, index=False)
    wide.to_csv(wide_out)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 78)
    print("ABLATION SUMMARY  (mean ± 95% CI across seeds; stars = vs reference)")
    print("  resolution: Cell/Niche NMI, ARI (↑)   |   "
          "integration: iLISI (↑), MMD (↓)")
    print("=" * 78)
    print(wide.to_string())
    print(f"\n[summary] wrote {long_out}")
    print(f"[summary] wrote {wide_out}")


if __name__ == "__main__":
    main()
