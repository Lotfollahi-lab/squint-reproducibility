#!/usr/bin/env python3
"""
4-FOLD, multi-seed comparison of a DISCRETE (dual-VQ) SQUINT model against its
CONTINUOUS-latent twin (same architecture, VQ removed) — the
discrete-vs-continuous ablation answering the reviewer comment "the rationale
for discrete codebooks over continuous latents is not sufficiently supported".

For each branch (cell, niche) it scores FOUR representations against the SAME
ground-truth labels (cell-type for cell, niche for niche), via NMI and ARI:

  (1) DISCRETE CODES               — the discrete model's level-0 code
                                     assignment, used directly (native output).
  (2) VQ-VAE QUANTIZED EMB, CLUSTERED   — k-means of the discrete model's
                                     QUANTIZED embedding (obsm['cell_emb'] /
                                     ['neighborhood_emb'] = z_q).
  (3) VQ-VAE PRE-QUANT EMB, CLUSTERED   — k-means of the discrete model's
                                     PRE-quantization latent (obsm['cell_latent']
                                     / ['neighborhood_latent'] = z).
  (4) CONTINUOUS EMB, CLUSTERED    — k-means of the continuous model's embedding
                                     (obsm['cell_emb']; z_q==z for that model).

k-means k for folds (2),(3),(4) = the number of discrete codes
(--match nominal [default] = the L0 codebook size; --match used = the #unique
L0 codes the discrete model assigns), so all clustered folds are at matched
granularity. k-means seed is fixed (--kmeans-seed); the VARIANCE comes from the
5 TRAINING SEEDS (5 discrete + 5 continuous run dirs).

MULTI-SEED: pass several run dirs per model (training seeds 0-4). Each path may
be a run dir (has predicted_adata.h5ad), a multi-seed sweep dir / a
seed_run_index.csv (auto-expanded to its per-seed run dirs), or a variant
parent dir (all timestamp subdirs with a predicted_adata.h5ad). Identical paths
are de-duplicated with a warning.

Per branch x metric it reports mean +/- std across seeds, the individual seed
points, and pairwise SIGNIFICANCE (folds 1-3 are paired across the discrete
runs; any-vs-fold-4 is an independent two-sample test). Test: t-test (default)
or Mann-Whitney (--test mannwhitney).

Outputs (to --out-dir, default <first continuous run>/comparison_vs_discrete/):
  comparison_per_seed.csv     one row per (branch, condition, seed-run)
  comparison_summary.csv      mean/std/n per (branch, condition)
  comparison_significance.csv pairwise p-values + stars
  comparison.json             everything machine-readable
  comparison.png / .svg / .pdf  grouped bars + points + error bars + sig brackets
                                (editable text in Illustrator)

Usage:
    python compare_discrete_vs_continuous.py \
        --discrete-runs   <s49_v23 multiseed sweep dir | run dirs...> \
        --continuous-runs <s53_v1 multiseed sweep dir | run dirs...>
    python compare_discrete_vs_continuous.py --match nominal --test ttest

Requirements: anndata, numpy, pandas, scikit-learn (scipy + matplotlib optional).
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys

import numpy as np

# --- default discrete run dirs (the 5 s49_v23 seeds; de-duped at runtime) ----
_ARTROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts/mmb0-1b_smb1-1b_1p"
_DISCRETE_VARIANT = ("s49_v23_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+"
                     "dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+"
                     "decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p")
DEFAULT_DISCRETE_RUNS = [os.path.join(_ARTROOT, _DISCRETE_VARIANT, ts) for ts in (
    "20260513_214133", "20260513_220352", "20260513_214135")]

# Per-branch keys.
_QUANT_KEY    = {"cell": "cell_emb",    "niche": "neighborhood_emb"}      # z_q
_PREQUANT_KEY = {"cell": "cell_latent", "niche": "neighborhood_latent"}   # z (pre-VQ)
_CELL_LABELS  = ["cell_type", "cell_types", "annotation"]
_NICHE_LABELS = ["niche", "Sub_molecular_tissue_region", "ccf_region_name",
                 "spatial_cluster"]
_CODE_KEYS = {
    "cell":  {"uns": "Indices_cell",  "obsm": "cell_code_indices",
              "obs": "cell_code_index",         "sizes": "codebook_sizes_cell"},
    "niche": {"uns": "Indices_niche", "obsm": "neighborhood_code_indices",
              "obs": "neighborhood_code_index", "sizes": "codebook_sizes_niche"},
}
# Condition labels (the 4 folds) and short labels for the plot.
C1, C2, C3, C4 = ("1. Discrete codes (L0)",
                  "2. VQ-VAE quant. emb, clustered",
                  "3. VQ-VAE pre-quant emb, clustered",
                  "4. Continuous emb, clustered")
SHORT = {C1: "Codes", C2: "Quant.\nemb", C3: "Pre-quant\nemb", C4: "Continuous\nemb"}
BRANCHES = [("cell", "Cell-type"), ("niche", "Niche")]


# ----------------------------------------------------------------------------
# Run-dir discovery
# ----------------------------------------------------------------------------
def _read_run_dirs_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    col = next((c for c in ("run_dir", "run_directory", "rundir", "dir")
                if c in df.columns), None)
    if col is None:
        print(f"    (no run_dir column in {path})", file=sys.stderr)
        return []
    return [str(x) for x in df[col].tolist() if isinstance(x, str) or not pd.isna(x)]


def _expand_runs(paths, tag):
    out = []
    for p in paths:
        if p.endswith(".csv") and os.path.isfile(p):
            out += _read_run_dirs_csv(p)
        elif os.path.isdir(p):
            if os.path.isfile(os.path.join(p, "predicted_adata.h5ad")):
                out.append(p)                                         # a run dir
            elif os.path.isfile(os.path.join(p, "seed_run_index.csv")):
                out += _read_run_dirs_csv(os.path.join(p, "seed_run_index.csv"))
            elif glob.glob(os.path.join(p, "seed_runs", "seed_*_run_dir.txt")):
                for f in sorted(glob.glob(os.path.join(p, "seed_runs",
                                                       "seed_*_run_dir.txt"))):
                    with open(f) as fh:
                        d = fh.read().strip()
                        if d:
                            out.append(d)
            else:                                                    # variant parent
                out += [d for d in sorted(glob.glob(os.path.join(p, "*")))
                        if os.path.isfile(os.path.join(d, "predicted_adata.h5ad"))]
        else:
            print(f"    (skip {tag} path, not found: {p})", file=sys.stderr)
    # de-dupe, preserve order; warn on duplicates.
    seen, uniq, dups = set(), [], 0
    for d in out:
        d = os.path.normpath(d)
        if d in seen:
            dups += 1
            continue
        seen.add(d)
        uniq.append(d)
    if dups:
        print(f"    [{tag}] WARNING: dropped {dups} duplicate run dir(s); "
              f"{len(uniq)} unique remain.", file=sys.stderr)
    return uniq


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _adata_in(run_dir):
    p = os.path.join(run_dir, "predicted_adata.h5ad")
    return p if os.path.isfile(p) else None


def _first_present(container, keys):
    for k in keys:
        if k in container:
            return k
    return None


def _label_mask_factorize(labels):
    import pandas as pd
    s = pd.Series(np.asarray(labels))
    mask = (~pd.isna(s)).to_numpy()
    lab = pd.factorize(s[mask].to_numpy())[0]
    return mask, lab


def _nmi_ari(true_lab, pred_lab):
    from sklearn.metrics import (normalized_mutual_info_score as _nmi,
                                 adjusted_rand_score as _ari)
    return float(_nmi(true_lab, pred_lab)), float(_ari(true_lab, pred_lab))


def _kmeans(emb, k, seed):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit_predict(
        np.asarray(emb, dtype=np.float32))


def _codes_l0(adata, branch):
    k = _CODE_KEYS[branch]
    idx = None
    if k["uns"] in adata.uns:
        idx = _to_numpy(adata.uns[k["uns"]])
    elif k["obsm"] in adata.obsm:
        idx = _to_numpy(adata.obsm[k["obsm"]])
    elif k["obs"] in adata.obs:
        idx = _to_numpy(adata.obs[k["obs"]].to_numpy())
    if idx is None:
        return None
    idx = np.asarray(idx)
    if idx.ndim == 2:
        idx = idx[:, 0]
    return idx.astype(np.int64)


def _nominal_l0(adata, branch):
    k = _CODE_KEYS[branch]
    s = adata.uns.get(k["sizes"], None)
    if s is None and branch == "niche":
        s = adata.uns.get("codebook_sizes", None)
    return int(np.asarray(s).ravel()[0]) if s is not None else None


def _label_key(adata, branch, override):
    if override:
        return override
    return _first_present(adata.obs, _CELL_LABELS if branch == "cell" else _NICHE_LABELS)


def _stars(p):
    if p != p:
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def _pvalue(a, b, paired, test):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        from scipy import stats
    except Exception:
        return float("nan")
    paired = paired and (len(a) == len(b))
    try:
        if test == "mannwhitney":
            _, p = stats.wilcoxon(a, b) if paired else stats.mannwhitneyu(
                a, b, alternative="two-sided")
        else:
            _, p = stats.ttest_rel(a, b) if paired else stats.ttest_ind(
                a, b, equal_var=False)
        return float(p)
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discrete-runs", nargs="+", default=DEFAULT_DISCRETE_RUNS,
                    help="Discrete (VQ-VAE) run dirs / multiseed sweep dir / "
                         "seed_run_index.csv (auto-expanded).")
    ap.add_argument("--continuous-runs", nargs="+", required=False, default=None,
                    help="Continuous run dirs / multiseed sweep dir / "
                         "seed_run_index.csv (auto-expanded).")
    ap.add_argument("--discrete-label", default="Discrete VQ")
    ap.add_argument("--continuous-label", default="Continuous")
    ap.add_argument("--cell-label-key", default=None)
    ap.add_argument("--niche-label-key", default=None)
    ap.add_argument("--match", choices=["nominal", "used"], default="nominal",
                    help="k for embedding clustering = #discrete codes. "
                         "nominal (default) = L0 codebook size; used = #unique "
                         "L0 codes assigned (per discrete run; fold-4 uses the mean).")
    ap.add_argument("--kmeans-seed", type=int, default=0,
                    help="Fixed k-means seed (variance comes from training seeds).")
    ap.add_argument("--test", choices=["ttest", "mannwhitney"], default="ttest")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if not args.continuous_runs:
        raise SystemExit("--continuous-runs is required (the 5 s53_v1 seed runs, "
                         "or the s53_v1 multiseed sweep dir / seed_run_index.csv).")

    import anndata as ad
    import pandas as pd

    print("[compare] resolving run dirs...")
    disc_runs = _expand_runs(args.discrete_runs, "discrete")
    cont_runs = _expand_runs(args.continuous_runs, "continuous")
    print(f"    discrete  : {len(disc_runs)} run(s)")
    print(f"    continuous: {len(cont_runs)} run(s)")
    if not disc_runs or not cont_runs:
        raise SystemExit("Need >=1 discrete and >=1 continuous run dir.")
    out_dir = args.out_dir or os.path.join(cont_runs[0], "comparison_vs_discrete")
    os.makedirs(out_dir, exist_ok=True)
    dlab, clab = args.discrete_label, args.continuous_label

    per_seed = []                       # tidy rows: branch, condition, model, seed_idx, run_dir, NMI, ARI, k
    label_per_branch = {}
    nominal_k = {}                      # branch -> L0 codebook size (from discrete)
    used_counts = {"cell": [], "niche": []}

    # ---- DISCRETE runs -> folds 1, 2, 3 ----
    for si, rd in enumerate(disc_runs):
        p = _adata_in(rd)
        if p is None:
            print(f"[compare] discrete seed {si}: no predicted_adata in {rd}", file=sys.stderr)
            continue
        print(f"[compare] discrete seed {si}: {p}")
        A = ad.read_h5ad(p)
        for branch, _bn in BRANCHES:
            lab_key = _label_key(A, branch, args.cell_label_key if branch == "cell"
                                 else args.niche_label_key)
            codes = _codes_l0(A, branch)
            qkey, pkey = _QUANT_KEY[branch], _PREQUANT_KEY[branch]
            if lab_key is None or qkey not in A.obsm or codes is None:
                print(f"    [{branch}] missing label/{qkey}/codes — skip", file=sys.stderr)
                continue
            label_per_branch[branch] = lab_key
            nominal_k.setdefault(branch, _nominal_l0(A, branch) or
                                 int(len(np.unique(codes))))
            mask, true_lab = _label_mask_factorize(A.obs[lab_key])
            codes_v = codes[mask]
            k_used = int(len(np.unique(codes_v)))
            used_counts[branch].append(k_used)
            k = nominal_k[branch] if args.match == "nominal" else k_used
            # fold 1: codes directly
            nmi, ari = _nmi_ari(true_lab, codes_v)
            per_seed.append(dict(branch=branch, condition=C1, model=dlab, seed_idx=si,
                                 run_dir=rd, NMI=nmi, ARI=ari, k=k_used))
            # fold 2: quantized embedding clustered
            nmi, ari = _nmi_ari(true_lab, _kmeans(A.obsm[qkey][mask], k, args.kmeans_seed))
            per_seed.append(dict(branch=branch, condition=C2, model=dlab, seed_idx=si,
                                 run_dir=rd, NMI=nmi, ARI=ari, k=k))
            # fold 3: pre-quantization latent clustered
            if pkey in A.obsm:
                nmi, ari = _nmi_ari(true_lab, _kmeans(A.obsm[pkey][mask], k, args.kmeans_seed))
                per_seed.append(dict(branch=branch, condition=C3, model=dlab, seed_idx=si,
                                     run_dir=rd, NMI=nmi, ARI=ari, k=k))
            else:
                print(f"    [{branch}] no pre-quant key {pkey!r}; skipping fold 3",
                      file=sys.stderr)
        del A
        gc.collect()

    # k for fold 4 (continuous): single value per branch from the discrete codes.
    k_f4 = {}
    for branch, _ in BRANCHES:
        if branch not in nominal_k:
            continue
        k_f4[branch] = (nominal_k[branch] if args.match == "nominal"
                        else int(round(np.mean(used_counts[branch]))) if used_counts[branch]
                        else nominal_k[branch])

    # ---- CONTINUOUS runs -> fold 4 ----
    for si, rd in enumerate(cont_runs):
        p = _adata_in(rd)
        if p is None:
            print(f"[compare] continuous seed {si}: no predicted_adata in {rd}", file=sys.stderr)
            continue
        print(f"[compare] continuous seed {si}: {p}")
        A = ad.read_h5ad(p)
        for branch, _bn in BRANCHES:
            if branch not in label_per_branch or branch not in k_f4:
                continue
            lab_key = label_per_branch[branch]
            qkey = _QUANT_KEY[branch]
            if lab_key not in A.obs or qkey not in A.obsm:
                print(f"    [{branch}] missing label/{qkey} in continuous run — skip",
                      file=sys.stderr)
                continue
            mask, true_lab = _label_mask_factorize(A.obs[lab_key])
            nmi, ari = _nmi_ari(true_lab, _kmeans(A.obsm[qkey][mask], k_f4[branch],
                                                  args.kmeans_seed))
            per_seed.append(dict(branch=branch, condition=C4, model=clab, seed_idx=si,
                                 run_dir=rd, NMI=nmi, ARI=ari, k=k_f4[branch]))
        del A
        gc.collect()

    if not per_seed:
        raise SystemExit("No metrics computed — check run dirs / labels / obsm keys.")
    ps_df = pd.DataFrame(per_seed)

    # ---- summary (mean/std/n) ----
    summ = (ps_df.groupby(["branch", "condition", "model"])
            .agg(NMI_mean=("NMI", "mean"), NMI_std=("NMI", "std"),
                 ARI_mean=("ARI", "mean"), ARI_std=("ARI", "std"),
                 n=("NMI", "size"), k=("k", "first"))
            .reset_index())

    # ---- pairwise significance per branch x metric ----
    order = [C1, C2, C3, C4]
    model_of = {C1: dlab, C2: dlab, C3: dlab, C4: clab}
    sig_rows = []
    for branch, _bn in BRANCHES:
        for metric in ("NMI", "ARI"):
            vals = {c: ps_df[(ps_df.branch == branch) & (ps_df.condition == c)]
                    .sort_values("seed_idx")[metric].to_numpy() for c in order}
            for i in range(len(order)):
                for j in range(i + 1, len(order)):
                    ci, cj = order[i], order[j]
                    if len(vals[ci]) == 0 or len(vals[cj]) == 0:
                        continue
                    paired = (model_of[ci] == model_of[cj])
                    p = _pvalue(vals[ci], vals[cj], paired, args.test)
                    sig_rows.append(dict(
                        branch=branch, metric=metric, cond_a=ci, cond_b=cj,
                        mean_a=float(np.nanmean(vals[ci])),
                        mean_b=float(np.nanmean(vals[cj])),
                        test=("paired-" if paired else "indep-") + args.test,
                        p_value=p, stars=_stars(p)))
    sig_df = pd.DataFrame(sig_rows)

    # ---- write ----
    ps_csv = os.path.join(out_dir, "comparison_per_seed.csv")
    su_csv = os.path.join(out_dir, "comparison_summary.csv")
    sg_csv = os.path.join(out_dir, "comparison_significance.csv")
    ps_df.to_csv(ps_csv, index=False)
    summ.to_csv(su_csv, index=False)
    sig_df.to_csv(sg_csv, index=False)
    with open(os.path.join(out_dir, "comparison.json"), "w") as fh:
        json.dump(dict(discrete=dict(label=dlab, runs=disc_runs),
                       continuous=dict(label=clab, runs=cont_runs),
                       match=args.match, k_per_branch=k_f4,
                       per_seed=per_seed,
                       summary=summ.to_dict(orient="records"),
                       significance=sig_rows), fh, indent=2, default=str)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 80)
    print(f"4-FOLD comparison across seeds (discrete n={ps_df[ps_df.model==dlab].seed_idx.nunique()}, "
          f"continuous n={ps_df[ps_df.model==clab].seed_idx.nunique()}); "
          f"k=#codes [{args.match}], k-means seed={args.kmeans_seed}")
    print("=" * 80)
    print(summ.to_string(index=False))
    print("\nPairwise significance:")
    print(sig_df.to_string(index=False))
    print(f"\n[compare] wrote {ps_csv}\n[compare] wrote {su_csv}\n[compare] wrote {sg_csv}")

    # ---- figure: 2 metrics x 2 branches, 4 bars each + points + std + brackets ----
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.rcParams["svg.fonttype"] = "none"   # editable text in Illustrator
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42
            import matplotlib.pyplot as plt
            rng = np.random.default_rng(0)
            fig, axes = plt.subplots(2, 2, figsize=(11, 9), squeeze=False)
            for r, metric in enumerate(("NMI", "ARI")):
                for c, (branch, bn) in enumerate(BRANCHES):
                    ax = axes[r][c]
                    means, stds, pts = [], [], []
                    for cond in order:
                        v = ps_df[(ps_df.branch == branch) & (ps_df.condition == cond)][metric].to_numpy()
                        means.append(np.nanmean(v) if len(v) else np.nan)
                        stds.append(np.nanstd(v, ddof=1) if len(v) > 1 else 0.0)
                        pts.append(v)
                    x = np.arange(len(order))
                    ax.bar(x, means, yerr=stds, capsize=4, color="0.8",
                           edgecolor="0.3", zorder=1)
                    for xi, v in zip(x, pts):
                        if len(v):
                            ax.scatter(np.full(len(v), xi) + rng.uniform(-.12, .12, len(v)),
                                       v, s=22, zorder=3, color="0.15", alpha=0.85)
                    # significance brackets: each discrete fold vs continuous (fold 4)
                    sub = sig_df[(sig_df.branch == branch) & (sig_df.metric == metric)]
                    top = np.nanmax([np.nanmax(v) if len(v) else 0 for v in pts])
                    step = 0.05 * (top if top > 0 else 1.0)
                    lvl = 0
                    for a_i, cond in enumerate((C1, C2, C3)):
                        row = sub[(sub.cond_a == cond) & (sub.cond_b == C4)]
                        if row.empty:
                            continue
                        star = row.iloc[0]["stars"]
                        y = top + step * (1.5 + lvl)
                        ax.plot([a_i, a_i, 3, 3], [y, y + step * .4, y + step * .4, y],
                                lw=1.0, c="0.3", zorder=4)
                        ax.text((a_i + 3) / 2, y + step * .4, star, ha="center",
                                va="bottom", fontsize=8)
                        lvl += 1
                    ax.set_xticks(x)
                    ax.set_xticklabels([SHORT[c2] for c2 in order], fontsize=8)
                    ax.set_ylabel(metric)
                    ax.set_title(f"{bn} — {metric} vs Ground-Truth Labels", fontsize=10)
            fig.suptitle("Discrete Codes vs Clustered Embeddings Across 5 Seeds "
                         "(k = Number of Discrete Codes)", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            for ext in ("png", "svg", "pdf"):
                fig.savefig(os.path.join(out_dir, f"comparison.{ext}"),
                            dpi=150, bbox_inches="tight")
            print(f"[compare] wrote {os.path.join(out_dir, 'comparison.png')} "
                  f"(+ .svg, .pdf — editable text in Illustrator)")
        except Exception as e:
            print(f"[compare] plot skipped ({type(e).__name__}: {e})", file=sys.stderr)

    print("\n[compare] DONE")


if __name__ == "__main__":
    main()
