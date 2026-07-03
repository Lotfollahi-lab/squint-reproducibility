#!/usr/bin/env python3
"""
Multi-seed DISCRETE-vs-CONTINUOUS ablation, rendered in the SAME horizontal
format as the other ablation figures (reuses plot_ablations_multiseed.render_axis
— per-seed dots, 95% CI, pink default, significance vs default). Answers the
reviewer comment "the rationale for discrete codebooks over continuous latents
is not sufficiently supported".

THREE conditions (bars), each scored on ALL 8 ablation metrics — resolution
(Cell/Niche NMI & ARI) AND integration (Cell/Niche iLISI & MMD). To reconcile
with the main benchmark, the DISCRETE model is the HEADLINE SQUINT reference
(s57_v19) and the CONTINUOUS model is its exact continuous counterpart (s57_v33
= identical config with the ResidualVQ bottleneck replaced by a ContinuousVQ
passthrough), so discreteness is the ONLY thing that varies between them:

  - "SQUINT (codes)"    headline SQUINT: NMI/ARI from the level-0 CODES directly
                        (no clustering step — the codes ARE the assignment);
                        iLISI/MMD from the quantized z_q embedding. This bar is
                        the headline niche/cell number reported in Table 1.
  - "SQUINT (Leiden)"   headline SQUINT: NMI/ARI from Leiden clustering of the
                        PRE-VQ latent z (the model's own continuous
                        representation), iLISI/MMD from the pre-VQ z embedding.
  - "Continuous (Leiden)"  s57_v33 continuous counterpart: NMI/ARI from Leiden
                        clustering of its embedding; iLISI/MMD from that emb.

So the headline model contributes both its DISCRETE (codes / z_q) and its
CONTINUOUS (pre-VQ z) representations, and the continuous-trained counterpart
its embedding — isolating whether the discrete codes match (or beat) clustering
a continuous latent. RESOLUTION (NMI/ARI) for the two Leiden bars is computed
here by Leiden with resolution binary-search to k = #discrete codes clusters
(--match nominal = L0 codebook size = 30; --match used = #unique codes), the
IDENTICAL protocol used for the continuous baselines in the main niche/cell-type
benchmark (analysis/benchmarking/.../run_pca_leiden.py::_leiden_n_clusters:
sc.pp.neighbors n_neighbors=15 then bisect resolution in [0.05, 10.0]).
INTEGRATION (iLISI/MMD) is read straight from each run's precomputed
metrics/batch_integration_metrics.csv (emb keys cell_emb / neighborhood_emb =
z_q ; cell_latent / neighborhood_latent = pre-VQ z) — no recomputation, so it
matches the numbers in the other ablation plots. Leiden seed fixed
(--leiden-seed); VARIANCE = the training seeds. Significance: each condition vs
the "SQUINT (codes)" default.

MULTI-SEED: pass run dirs per model; each may be a run dir, a multiseed sweep /
seed_run_index.csv (auto-expanded), or a variant parent dir. Default runs =
s57_v19 (headline discrete) and s57_v33 (continuous counterpart).

Outputs (to --out-dir, default <first continuous run>/comparison_vs_discrete/):
  discretization_per_seed.csv      one row per (condition, branch, metric, seed)
  discretization_summary.csv       mean/std/n per (condition, branch, metric)
  discretization_comparison.{svg,png,pdf}  horizontal ablation-style figure

Usage:
    python compare_discrete_vs_continuous.py            # s57_v19 (discrete) vs s57_v33 (continuous)
    python compare_discrete_vs_continuous.py --test mannwhitney --error sem

Requirements: anndata, numpy, pandas, scikit-learn, matplotlib (+ the ablation
plotters under analysis/ablations/plots/).
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys

import numpy as np

# --- default sweep dirs ------------------------------------------------------
# Matched-pair discretization ablation on the HEADLINE config:
#   discrete   = s57_v19 (the paper's headline SQUINT reference; its level-0
#                codes ARE the Table-1 niche/cell-type clustering, so the
#                "SQUINT (codes)" bar reconciles with Table 1).
#   continuous = s57_v33 (the EXACT continuous counterpart of s57_v19 — same
#                FiLM-scale coupling + cross-batch-MNN contrastive spine, with
#                the ResidualVQ bottleneck swapped for a ContinuousVQ
#                passthrough on both branches). Isolates discreteness alone.
# Both are multi-seed sweep PARENT dirs; `_expand_runs` auto-descends each to
# its latest <timestamp>/seed_run_index.csv. Override with
# --discrete-runs / --continuous-runs.
_ARTROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts/mmb0-1b_smb1-1b_1p"
_DISCRETE_VARIANT = "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p"
_CONTINUOUS_VARIANT = "s57_v33_continuous-ref-filmscale+mmb0-1b_smb1-1b_1p"
DEFAULT_DISCRETE_RUNS = [os.path.join(_ARTROOT, _DISCRETE_VARIANT + "__multiseed")]
DEFAULT_CONTINUOUS_RUNS = [os.path.join(_ARTROOT, _CONTINUOUS_VARIANT + "__multiseed")]

# Per-branch keys.
_QUANT_KEY    = {"cell": "cell_emb",    "niche": "neighborhood_emb"}      # z_q
_PREQUANT_KEY = {"cell": "cell_latent", "niche": "neighborhood_latent"}   # z (pre-VQ)
# Ground-truth label columns, mirroring compute_inference_metrics.py's
# DEFAULT_{CELL,NICHE}_LABEL_KEYS so the resolution NMI/ARI here matches the
# main benchmark. Niche NMI/ARI is the cell-count-weighted mean over ALL
# present niche labels (= Table 1's aggregate "niche" row); cell uses one.
_CELL_LABELS  = ["cell_type", "cell_types", "annotation", "new_annotation"]
_NICHE_LABELS = ["niche", "Sub_molecular_tissue_region", "ccf_region_name",
                 "spatial_cluster", "niche_type"]
_CODE_KEYS = {
    "cell":  {"uns": "Indices_cell",  "obsm": "cell_code_indices",
              "obs": "cell_code_index",         "sizes": "codebook_sizes_cell"},
    "niche": {"uns": "Indices_niche", "obsm": "neighborhood_code_indices",
              "obs": "neighborhood_code_index", "sizes": "codebook_sizes_niche"},
}
BRANCHES = [("cell", "Cell-type"), ("niche", "Niche")]


def _darken(colour, f: float = 0.62):
    """Darker shade for the per-seed dots (so they read against their bar)."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(colour)
    return (r * f, g * f, b * f)


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
            elif glob.glob(os.path.join(p, "*", "seed_run_index.csv")):
                # a <variant>__multiseed PARENT dir: descend to the latest
                # timestamped sweep and read its seed_run_index.csv.
                idx = sorted(glob.glob(os.path.join(p, "*", "seed_run_index.csv")))[-1]
                print(f"    [{tag}] using multiseed sweep {os.path.dirname(idx)}")
                out += _read_run_dirs_csv(idx)
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
    # Drop NaN AND the literal "nan"/"None" strings (h5ad object columns can
    # smuggle these through), exactly as compute_inference_metrics.py does, so
    # the labelled-cell set matches the benchmark.
    s = pd.Series(np.asarray(labels)).astype("object").map(
        lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v)))
        else str(v))
    mask = (s.notna() & (s != "nan") & (s != "None")).to_numpy()
    lab = pd.factorize(s[mask].to_numpy())[0]
    return mask, lab


def _nmi_ari(true_lab, pred_lab):
    from sklearn.metrics import (normalized_mutual_info_score as _nmi,
                                 adjusted_rand_score as _ari)
    return float(_nmi(true_lab, pred_lab)), float(_ari(true_lab, pred_lab))


def _present_labels(adata, branch, override=None):
    """Ground-truth label columns present in `adata` for a branch, mirroring
    compute_inference_metrics.py: the niche resolution is scored against EVERY
    present niche label (then cell-count-weighted-averaged), while the cell
    resolution uses a single primary label."""
    keys = [override] if override else (_CELL_LABELS if branch == "cell"
                                        else _NICHE_LABELS)
    present = [k for k in keys if k in adata.obs]
    if branch == "cell":
        present = present[:1]
    return present


def _res_weighted(pred_all, adata, present):
    """Cell-count-weighted mean (NMI, ARI) of the per-cell labels `pred_all`
    (codes or Leiden clusters, defined for ALL cells) against each present
    ground-truth label -- the same aggregation compute_inference_metrics.py
    uses for Table 1's niche row, so the "SQUINT (codes)" bar reconciles with
    the headline. Returns None if no labelled cells."""
    pred_all = np.asarray(pred_all)
    tot = 0
    wn = wa = 0.0
    for key in present:
        mask, true_lab = _label_mask_factorize(adata.obs[key])
        n = int(mask.sum())
        if n == 0:
            continue
        nmi, ari = _nmi_ari(true_lab, pred_all[mask])
        tot += n
        wn += nmi * n
        wa += ari * n
    return (wn / tot, wa / tot) if tot else None


def _leiden(emb, k, seed, n_neighbors: int = 15, max_iters: int = 25):
    """Leiden clustering with resolution binary-search to hit `k` clusters,
    returning integer labels aligned to `emb` rows.

    Mirrors the main niche/cell-type benchmark protocol
    (analysis/benchmarking/.../run_pca_leiden.py::_leiden_n_clusters, CPU path):
    build a k-NN graph on the embedding (n_neighbors=15) then bisect the Leiden
    resolution in [0.05, 10.0] until the cluster count equals `k` (or as close
    as possible within `max_iters`). This is the IDENTICAL clustering the
    continuous baselines get in the main benchmark, so the two Leiden bars here
    are apples-to-apples with Table 1's baselines.
    """
    import anndata as ad
    import scanpy as sc
    X = np.asarray(emb, dtype=np.float32)
    A = ad.AnnData(X)
    A.obsm["X_emb"] = X
    sc.pp.neighbors(A, n_neighbors=n_neighbors, use_rep="X_emb", random_state=seed)
    lo, hi = 0.05, 10.0
    best = None  # (abs-diff, labels)
    for _ in range(int(max_iters)):
        mid = 0.5 * (lo + hi)
        sc.tl.leiden(A, resolution=mid, key_added="leiden", random_state=seed)
        lab = A.obs["leiden"].astype(int).to_numpy()
        n_found = int(np.unique(lab).size)
        diff = abs(n_found - int(k))
        if best is None or diff < best[0]:
            best = (diff, lab)
        if n_found == int(k):
            return lab
        if n_found < int(k):
            lo = mid
        else:
            hi = mid
    return best[1]


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


def _err_halfwidth(vals, kind="ci95"):
    """Error-bar half-width: 95% CI (default), SEM, or STD."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return 0.0
    sd = float(v.std(ddof=1))
    if kind == "std":
        return sd
    sem = sd / np.sqrt(n)
    if kind == "sem":
        return sem
    try:
        from scipy import stats
        t = float(stats.t.ppf(0.975, n - 1))
    except Exception:
        t = 1.96
    return t * sem


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
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discrete-runs", nargs="+", default=DEFAULT_DISCRETE_RUNS,
                    help="Discrete (VQ-VAE) run dirs / multiseed sweep dir / "
                         "seed_run_index.csv (auto-expanded).")
    ap.add_argument("--continuous-runs", nargs="+", default=DEFAULT_CONTINUOUS_RUNS,
                    help="Continuous run dirs / multiseed sweep dir / "
                         "seed_run_index.csv (auto-expanded).")
    ap.add_argument("--discrete-label", default="Discrete VQ")
    ap.add_argument("--continuous-label", default="Continuous")
    ap.add_argument("--cell-label-key", default=None)
    ap.add_argument("--niche-label-key", default=None)
    ap.add_argument("--match", choices=["nominal", "used"], default="nominal",
                    help="target #clusters for Leiden = #discrete codes. "
                         "nominal (default) = L0 codebook size (=30); used = "
                         "#unique L0 codes assigned (per discrete run; the "
                         "continuous fold uses the mean).")
    ap.add_argument("--leiden-seed", type=int, default=0,
                    help="Fixed Leiden/neighbors seed (variance comes from "
                         "training seeds).")
    ap.add_argument("--test", choices=["ttest", "mannwhitney"], default="ttest")
    ap.add_argument("--error", choices=["ci95", "sem", "std"], default="ci95",
                    help="Error-bar half-width (default 95%% CI).")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)


    import anndata as ad
    import pandas as pd
    from pathlib import Path

    # Reuse the ablation plotter so the figure is IDENTICAL in format (horizontal
    # bars, per-seed dots, 95% CI, pink default, significance vs default).
    import sys as _sys
    _plots_dir = Path(__file__).resolve().parents[1] / "ablations" / "plots"
    if str(_plots_dir) not in _sys.path:
        _sys.path.insert(0, str(_plots_dir))
    from plot_ablations import (METRICS, AxisSpec, VariantEntry,        # noqa: E402
                                _apply_nature_style)
    from plot_ablations_multiseed import render_axis                    # noqa: E402

    # The 3 conditions (bars). Each maps a RESOLUTION source (codes directly /
    # Leiden-to-k on an embedding, k=#codes) and an INTEGRATION embedding
    # (iLISI/MMD read straight from each
    # run's precomputed metrics/batch_integration_metrics.csv — cell_emb /
    # neighborhood_emb = z_q; cell_latent / neighborhood_latent = pre-VQ z).
    COND_CODES  = "SQUINT (codes)"       # headline SQUINT: NMI/ARI from L0 codes; iLISI/MMD from z_q (= Table 1)
    COND_DCLUST = "SQUINT (Leiden)"      # headline SQUINT: NMI/ARI from Leiden of pre-VQ z; iLISI/MMD from pre-VQ z
    COND_CONT   = "Continuous (Leiden)"  # s57_v33 continuous counterpart: NMI/ARI from Leiden of emb; iLISI/MMD from emb
    CONDS = [COND_CODES, COND_DCLUST, COND_CONT]
    INTEG_EMB = {
        COND_CODES:  {"cell": "cell_emb",    "niche": "neighborhood_emb"},
        COND_DCLUST: {"cell": "cell_latent", "niche": "neighborhood_latent"},
        COND_CONT:   {"cell": "cell_emb",    "niche": "neighborhood_emb"},
    }

    def _read_integration(run_dir):
        """{(emb_key, metric): score} from <run_dir>/metrics/batch_integration_metrics.csv."""
        f = os.path.join(run_dir, "metrics", "batch_integration_metrics.csv")
        if not os.path.isfile(f):
            print(f"    [integration] missing {f}", file=sys.stderr)
            return {}
        try:
            d = pd.read_csv(f)
            return {(str(r["emb_key"]), str(r["metric"])): float(r["score"])
                    for _, r in d.iterrows()}
        except Exception as exc:  # noqa: BLE001
            print(f"    [integration] read failed {f}: {exc}", file=sys.stderr)
            return {}

    print("[compare] resolving run dirs...")
    disc_runs = _expand_runs(args.discrete_runs, "discrete")
    cont_runs = _expand_runs(args.continuous_runs, "continuous")
    print(f"    discrete  : {len(disc_runs)} run(s)\n    continuous: {len(cont_runs)} run(s)")
    if not disc_runs or not cont_runs:
        raise SystemExit("Need >=1 discrete and >=1 continuous run dir.")
    out_dir = args.out_dir or os.path.join(cont_runs[0], "comparison_vs_discrete")
    os.makedirs(out_dir, exist_ok=True)

    # per-seed accumulators: (cond, branch, kind) -> [values over seeds]
    res, integ = {}, {}
    def _push(store, cond, branch, kind, v):
        store.setdefault((cond, branch, kind), []).append(float(v))

    label_per_branch, nominal_k = {}, {}
    used_counts = {"cell": [], "niche": []}

    # ---- DISCRETE runs: codes + pre-VQ-clustered resolution + z_q/pre-VQ integ ----
    for si, rd in enumerate(disc_runs):
        p = _adata_in(rd)
        if p is None:
            print(f"[compare] discrete seed {si}: no predicted_adata in {rd}", file=sys.stderr)
            continue
        print(f"[compare] discrete seed {si}: {p}")
        A = ad.read_h5ad(p)
        for branch, _bn in BRANCHES:
            present = _present_labels(A, branch, args.cell_label_key if branch == "cell"
                                      else args.niche_label_key)
            codes = _codes_l0(A, branch)
            pkey = _PREQUANT_KEY[branch]
            if not present or codes is None:
                print(f"    [{branch}] missing label/codes — skip", file=sys.stderr)
                continue
            label_per_branch[branch] = present
            nominal_k.setdefault(branch, _nominal_l0(A, branch) or int(len(np.unique(codes))))
            used_counts[branch].append(int(len(np.unique(codes))))
            k = nominal_k[branch] if args.match == "nominal" else int(len(np.unique(codes)))
            # COND_CODES resolution = codes directly (weighted over all labels)
            r = _res_weighted(codes, A, present)
            if r:
                _push(res, COND_CODES, branch, "NMI", r[0]); _push(res, COND_CODES, branch, "ARI", r[1])
            # COND_DCLUST resolution = pre-VQ z Leiden-clustered to k (ALL cells)
            if pkey in A.obsm:
                lab_all = _leiden(A.obsm[pkey], k, args.leiden_seed)
                r = _res_weighted(lab_all, A, present)
                if r:
                    _push(res, COND_DCLUST, branch, "NMI", r[0]); _push(res, COND_DCLUST, branch, "ARI", r[1])
        ig = _read_integration(rd)
        for cond in (COND_CODES, COND_DCLUST):
            for branch, _bn in BRANCHES:
                ek = INTEG_EMB[cond][branch]
                for kind in ("iLISI", "MMD"):
                    if (ek, kind) in ig:
                        _push(integ, cond, branch, kind, ig[(ek, kind)])
        del A
        gc.collect()

    # k for the continuous fold (per branch from the discrete codes)
    k_cont = {b: (nominal_k[b] if args.match == "nominal"
                  else int(round(np.mean(used_counts[b]))) if used_counts[b] else nominal_k[b])
              for b, _ in BRANCHES if b in nominal_k}

    # ---- CONTINUOUS runs: clustered resolution + continuous-emb integ ----
    for si, rd in enumerate(cont_runs):
        p = _adata_in(rd)
        if p is None:
            print(f"[compare] continuous seed {si}: no predicted_adata in {rd}", file=sys.stderr)
            continue
        print(f"[compare] continuous seed {si}: {p}")
        A = ad.read_h5ad(p)
        for branch, _bn in BRANCHES:
            if branch not in label_per_branch or branch not in k_cont:
                continue
            present = [k for k in label_per_branch[branch] if k in A.obs]
            qkey = _QUANT_KEY[branch]
            if not present or qkey not in A.obsm:
                continue
            lab_all = _leiden(A.obsm[qkey], k_cont[branch], args.leiden_seed)
            r = _res_weighted(lab_all, A, present)
            if r:
                _push(res, COND_CONT, branch, "NMI", r[0]); _push(res, COND_CONT, branch, "ARI", r[1])
        ig = _read_integration(rd)
        for branch, _bn in BRANCHES:
            ek = INTEG_EMB[COND_CONT][branch]
            for kind in ("iLISI", "MMD"):
                if (ek, kind) in ig:
                    _push(integ, COND_CONT, branch, kind, ig[(ek, kind)])
        del A
        gc.collect()

    # ---- assemble per_metric_values for the 8 ablation metrics ----
    per_metric_values = {}
    for metric_label, _direction in METRICS:
        branch = "cell" if metric_label.startswith("Cell") else "niche"
        kind = metric_label.split()[-1]                     # NMI / ARI / iLISI / MMD
        src = res if kind in ("NMI", "ARI") else integ
        per_metric_values[metric_label] = {
            cond: np.asarray(src.get((cond, branch, kind), []), dtype=float)
            for cond in CONDS}

    if all(per_metric_values[m][c].size == 0 for m, _ in METRICS for c in CONDS):
        raise SystemExit("No metrics computed — check run dirs / labels / obsm / "
                         "metrics/batch_integration_metrics.csv.")

    # ---- per-seed + summary CSVs ----
    rows = [dict(condition=c, branch=b, metric=k, seed_idx=i, value=v)
            for (c, b, k), vals in {**res, **integ}.items()
            for i, v in enumerate(vals)]
    ps_df = pd.DataFrame(rows)
    summ = (ps_df.groupby(["condition", "branch", "metric"])
            .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "size"))
            .reset_index())
    ps_csv = os.path.join(out_dir, "discretization_per_seed.csv")
    su_csv = os.path.join(out_dir, "discretization_summary.csv")
    ps_df.to_csv(ps_csv, index=False)
    summ.to_csv(su_csv, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 78)
    print("DISCRETE vs CONTINUOUS — codes / clustered / continuous (mean ± std)")
    print("=" * 78)
    print(summ.to_string(index=False))
    print(f"\n[compare] wrote {ps_csv}\n[compare] wrote {su_csv}")

    # ---- figure: identical horizontal format to the ablation axes ----
    if not args.no_plot:
        try:
            _apply_nature_style()
            axis = AxisSpec(
                key="discretization",
                title="Discretization: Discrete Codes vs Leiden-Clustered Embeddings",
                entries=(VariantEntry(COND_CODES, COND_CODES, is_default=True),
                         VariantEntry(COND_DCLUST, COND_DCLUST),
                         VariantEntry(COND_CONT, COND_CONT)))
            render_axis(axis, per_metric_values, {c: c for c in CONDS}, COND_CODES,
                        Path(out_dir) / "discretization_comparison",
                        args.test, args.error)
            print(f"[compare] wrote {out_dir}/discretization_comparison.{{svg,png,pdf}}")
        except Exception as exc:  # noqa: BLE001
            print(f"[compare] plot skipped ({type(exc).__name__}: {exc})", file=sys.stderr)

    print("\n[compare] DONE")


if __name__ == "__main__":
    main()
