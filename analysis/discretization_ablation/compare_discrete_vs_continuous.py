#!/usr/bin/env python3
"""
3-FOLD comparison of a DISCRETE (dual-VQ) SQUINT run against its
CONTINUOUS-latent twin (same architecture, VQ removed) — the
discrete-vs-continuous ablation that answers the reviewer comment "the
rationale for discrete codebooks over continuous latents is not sufficiently
supported".

For each branch (cell, niche) it scores three representations against the SAME
ground-truth labels (cell-type for cell, niche for niche), via NMI and ARI:

  (1) DISCRETE CODES            — the discrete model's level-0 code assignment
                                  used directly as the clustering (its native,
                                  "free" discrete output).
  (2) DISCRETE EMB, CLUSTERED   — k-means of the discrete model's embedding,
                                  with k = the number of discrete codes.
  (3) CONTINUOUS EMB, CLUSTERED — k-means of the continuous model's embedding,
                                  with the SAME k = the number of discrete codes.

The embedding clustering (2, 3) uses k = the number of discrete codes (default:
the number of UNIQUE level-0 codes the discrete model actually uses; use
--match nominal for the codebook size instead), so all three folds produce a
comparable number of clusters. k-means (sklearn, explicit n_clusters, fixed
seed) is run identically for (2) and (3).

This makes the head-to-head fair: comparing (1) vs (3) asks "do discrete codes
beat a continuous latent clustered to the same granularity?"; (2) vs (3) asks
"is the discrete model's learned embedding better than the continuous model's,
independent of the discretization step?"; (1) vs (2) asks "how much does using
codes directly cost vs clustering the discrete model's own embedding?".

Why a dedicated script: the pipeline's niche_identification_metrics.csv scores
NMI only from the discrete codes, so the continuous run reads ~0 there (its
codes are placeholders). This recomputes everything consistently.

Also surfaces, side by side, the already-fair metrics from each run's
metrics/*.csv: reconstruction (pearson_reconstruction_metrics.csv) and batch
integration (batch_integration_metrics.csv).

Outputs (to --out-dir, default <continuous-run>/comparison_vs_discrete/):
  comparison.csv / comparison.json    full 3-fold table (+ side tables)
  comparison.png / .svg / .pdf        grouped bar charts (editable text in Illustrator)

Usage (defaults point at the two runs in question):
    python compare_discrete_vs_continuous.py
    python compare_discrete_vs_continuous.py --match nominal
    python compare_discrete_vs_continuous.py \
        --discrete-run <run_dir> --continuous-run <run_dir>

Requirements: anndata, numpy, pandas, scikit-learn (matplotlib optional).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np

# --- the two runs being compared (override on the CLI) -----------------------
_ARTROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts/mmb0-1b_smb1-1b_1p"
_DISCRETE_VARIANT = ("s49_v23_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+"
                     "dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+"
                     "decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p")
_CONTINUOUS_VARIANT = ("s53_v1_continuous-latent+decoder-cov+no-batch-int+enc-deeper+"
                       "dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+"
                       "decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p")
DEFAULT_DISCRETE_RUN   = os.path.join(_ARTROOT, _DISCRETE_VARIANT,   "20260513_223846")
DEFAULT_CONTINUOUS_RUN = os.path.join(_ARTROOT, _CONTINUOUS_VARIANT, "20260627_074514")

# Per-branch embedding obsm keys (first present wins) + candidate label columns.
_CELL_EMB_KEYS  = ["cell_emb", "cell_latent", "X_squint_quantized", "X_squint"]
_NICHE_EMB_KEYS = ["neighborhood_emb", "neighborhood_latent"]
_CELL_LABELS    = ["cell_type", "cell_types", "annotation"]
_NICHE_LABELS   = ["niche", "Sub_molecular_tissue_region", "ccf_region_name",
                   "spatial_cluster"]
# Per-branch discrete-code sources in the predicted adata.
_CODE_KEYS = {
    "cell":  {"uns": "Indices_cell",  "obsm": "cell_code_indices",
              "obs": "cell_code_index",         "sizes": "codebook_sizes_cell"},
    "niche": {"uns": "Indices_niche", "obsm": "neighborhood_code_indices",
              "obs": "neighborhood_code_index", "sizes": "codebook_sizes_niche"},
}

# Condition labels (the 3 folds).
C_CODES = "1. Discrete codes (L0)"
C_DEMB  = "2. Discrete emb, clustered"
C_CEMB  = "3. Continuous emb, clustered"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _adata_path(run_dir, explicit):
    if explicit:
        return explicit
    p = os.path.join(run_dir, "predicted_adata.h5ad")
    if not os.path.isfile(p):
        raise SystemExit(f"No predicted_adata.h5ad in {run_dir}")
    return p


def _first_present(container, keys):
    for k in keys:
        if k in container:
            return k
    return None


def _label_mask_factorize(labels):
    """Drop NaN labels; return (mask, integer-coded labels on the kept subset)."""
    import pandas as pd
    s = pd.Series(np.asarray(labels))
    mask = ~pd.isna(s)
    lab = pd.factorize(s[mask].to_numpy())[0]
    return mask.to_numpy(), lab


def _nmi_ari(true_lab, pred_lab):
    from sklearn.metrics import (normalized_mutual_info_score as _nmi,
                                 adjusted_rand_score as _ari)
    return float(_nmi(true_lab, pred_lab)), float(_ari(true_lab, pred_lab))


def _kmeans_predict(emb, k, seed):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit_predict(
        np.asarray(emb, dtype=np.float32))


def _get_codes_level0(adata, branch):
    """Level-0 discrete code id per cell, shape (N,). None if unavailable."""
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
        idx = idx[:, 0]               # level-0 (coarse macro cluster)
    return idx.astype(np.int64)


def _nominal_size_level0(adata, branch):
    k = _CODE_KEYS[branch]
    sizes = None
    if k["sizes"] in adata.uns:
        sizes = adata.uns[k["sizes"]]
    elif branch == "niche" and "codebook_sizes" in adata.uns:
        sizes = adata.uns["codebook_sizes"]
    if sizes is None:
        return None
    return int(np.asarray(sizes).ravel()[0])


def _read_csv(run_dir, name):
    import pandas as pd
    p = os.path.join(run_dir, "metrics", name)
    if not os.path.isfile(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"    (could not read {p}: {e})", file=sys.stderr)
        return None


def _merge_side_by_side(df_d, df_c, key_cols, val_cols, dlabel, clabel):
    import pandas as pd
    if df_d is None and df_c is None:
        return None
    frames = []
    if df_d is not None:
        frames.append(df_d.assign(_run=dlabel))
    if df_c is not None:
        frames.append(df_c.assign(_run=clabel))
    cat = pd.concat(frames, ignore_index=True)
    keep_keys = [k for k in key_cols if k in cat.columns]
    keep_vals = [v for v in val_cols if v in cat.columns]
    if not keep_vals or not keep_keys:
        return None
    wide = cat.pivot_table(index=keep_keys, columns="_run", values=keep_vals,
                           aggfunc="first")
    wide.columns = [f"{m} ({r})" for m, r in wide.columns]
    return wide.reset_index()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--discrete-run", default=DEFAULT_DISCRETE_RUN)
    ap.add_argument("--continuous-run", default=DEFAULT_CONTINUOUS_RUN)
    ap.add_argument("--discrete-adata", default=None)
    ap.add_argument("--continuous-adata", default=None)
    ap.add_argument("--discrete-label", default="Discrete VQ")
    ap.add_argument("--continuous-label", default="Continuous")
    ap.add_argument("--cell-label-key", default=None)
    ap.add_argument("--niche-label-key", default=None)
    ap.add_argument("--cell-emb-key", default=None)
    ap.add_argument("--niche-emb-key", default=None)
    ap.add_argument("--match", choices=["used", "nominal"], default="used",
                    help="k for embedding clustering = number of discrete codes. "
                         "'used' (default) = # unique level-0 codes the discrete "
                         "model actually assigns; 'nominal' = the L0 codebook size.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    import anndata as ad
    import pandas as pd

    d_path = _adata_path(args.discrete_run, args.discrete_adata)
    c_path = _adata_path(args.continuous_run, args.continuous_adata)
    out_dir = args.out_dir or os.path.join(args.continuous_run, "comparison_vs_discrete")
    os.makedirs(out_dir, exist_ok=True)
    dlab, clab = args.discrete_label, args.continuous_label

    rows = []            # tidy: branch, condition, model, NMI, ARI, k_clusters, n_cells
    k_per_branch = {}    # branch -> k used for the embedding clustering
    label_per_branch = {}

    # ---- Pass 1: DISCRETE run -> folds (1) codes and (2) discrete emb clustered
    print(f"\n[compare] DISCRETE ({dlab}): {d_path}")
    disc = ad.read_h5ad(d_path)
    print(f"    {disc.n_obs} cells")
    for branch, emb_keys, lab_keys, lab_arg in (
            ("cell",  _CELL_EMB_KEYS,  _CELL_LABELS,  args.cell_label_key),
            ("niche", _NICHE_EMB_KEYS, _NICHE_LABELS, args.niche_label_key)):
        emb_key = (args.cell_emb_key if branch == "cell" else args.niche_emb_key) \
            or _first_present(disc.obsm, emb_keys)
        lab_key = lab_arg or _first_present(disc.obs, lab_keys)
        codes = _get_codes_level0(disc, branch)
        bname = "Cell-type" if branch == "cell" else "Niche"
        print(f"    [{branch}] emb={emb_key!r} label={lab_key!r} "
              f"codes={'yes' if codes is not None else 'MISSING'}")
        if lab_key is None or emb_key is None:
            print(f"    [{branch}] missing embedding or label — skipping branch",
                  file=sys.stderr)
            continue
        label_per_branch[branch] = lab_key
        mask, true_lab = _label_mask_factorize(disc.obs[lab_key])

        # k = number of discrete codes (used or nominal)
        if codes is not None:
            codes_v = codes[mask]
            k_used = int(len(np.unique(codes_v)))
            k_nominal = _nominal_size_level0(disc, branch) or k_used
            k = k_used if args.match == "used" else k_nominal
            k_per_branch[branch] = k
            # Fold (1): discrete codes directly
            nmi, ari = _nmi_ari(true_lab, codes_v)
            rows.append({"branch": bname, "condition": C_CODES, "model": dlab,
                         "NMI": nmi, "ARI": ari, "k_clusters": k_used,
                         "n_cells": int(mask.sum())})
        else:
            # No codes (e.g. old artifact): fall back to k = #classes for (2)/(3).
            k = int(len(np.unique(true_lab)))
            k_per_branch[branch] = k
            print(f"    [{branch}] no discrete codes found; "
                  f"clustering k falls back to #classes={k}", file=sys.stderr)

        # Fold (2): discrete embedding clustered at k = #codes
        pred = _kmeans_predict(disc.obsm[emb_key][mask], k_per_branch[branch], args.seed)
        nmi, ari = _nmi_ari(true_lab, pred)
        rows.append({"branch": bname, "condition": C_DEMB, "model": dlab,
                     "NMI": nmi, "ARI": ari, "k_clusters": k_per_branch[branch],
                     "n_cells": int(mask.sum())})
    del disc
    gc.collect()

    # ---- Pass 2: CONTINUOUS run -> fold (3) continuous emb clustered @ matched k
    print(f"\n[compare] CONTINUOUS ({clab}): {c_path}")
    cont = ad.read_h5ad(c_path)
    print(f"    {cont.n_obs} cells")
    for branch, emb_keys in (("cell", _CELL_EMB_KEYS), ("niche", _NICHE_EMB_KEYS)):
        if branch not in k_per_branch or branch not in label_per_branch:
            continue
        emb_key = (args.cell_emb_key if branch == "cell" else args.niche_emb_key) \
            or _first_present(cont.obsm, emb_keys)
        lab_key = label_per_branch[branch]
        bname = "Cell-type" if branch == "cell" else "Niche"
        if emb_key is None or lab_key not in cont.obs:
            print(f"    [{branch}] missing embedding {emb_key!r} or label "
                  f"{lab_key!r} in continuous run — skipping", file=sys.stderr)
            continue
        print(f"    [{branch}] emb={emb_key!r} label={lab_key!r} k={k_per_branch[branch]}")
        mask, true_lab = _label_mask_factorize(cont.obs[lab_key])
        pred = _kmeans_predict(cont.obsm[emb_key][mask], k_per_branch[branch], args.seed)
        nmi, ari = _nmi_ari(true_lab, pred)
        rows.append({"branch": bname, "condition": C_CEMB, "model": clab,
                     "NMI": nmi, "ARI": ari, "k_clusters": k_per_branch[branch],
                     "n_cells": int(mask.sum())})
    del cont
    gc.collect()

    comp_df = pd.DataFrame(rows)

    # ---- side tables (already-fair, from the metrics CSVs) ----
    extra = {}
    pear = _merge_side_by_side(
        _read_csv(args.discrete_run, "pearson_reconstruction_metrics.csv"),
        _read_csv(args.continuous_run, "pearson_reconstruction_metrics.csv"),
        key_cols=["split", "split_label", "branch", "feature", "metric"],
        val_cols=["pearson", "pearson_r", "spearman", "r", "value", "score"],
        dlabel=dlab, clabel=clab)
    if pear is not None:
        extra["Reconstruction (pearson_reconstruction_metrics.csv)"] = pear
    batch = _merge_side_by_side(
        _read_csv(args.discrete_run, "batch_integration_metrics.csv"),
        _read_csv(args.continuous_run, "batch_integration_metrics.csv"),
        key_cols=["emb_key", "metric"], val_cols=["score", "value"],
        dlabel=dlab, clabel=clab)
    if batch is not None:
        extra["Batch integration (batch_integration_metrics.csv)"] = batch

    # ---- write ----
    comp_csv = os.path.join(out_dir, "comparison.csv")
    comp_json = os.path.join(out_dir, "comparison.json")
    comp_df.to_csv(comp_csv, index=False)
    with open(comp_json, "w") as fh:
        json.dump({"discrete": {"label": dlab, "run": args.discrete_run},
                   "continuous": {"label": clab, "run": args.continuous_run},
                   "match": args.match, "k_per_branch": k_per_branch,
                   "three_fold": rows,
                   "extra_tables": {k: v.to_dict(orient="records")
                                    for k, v in extra.items()}}, fh, indent=2, default=str)

    # ---- print ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 78)
    print("3-FOLD comparison — NMI / ARI vs ground-truth labels")
    print(f"(embedding clustering k = #discrete codes [{args.match}]; "
          f"sklearn k-means, seed={args.seed})")
    print("=" * 78)
    if not comp_df.empty:
        for metric in ("NMI", "ARI"):
            piv = comp_df.pivot_table(index="condition", columns="branch",
                                      values=metric, aggfunc="first")
            piv = piv.reindex([C_CODES, C_DEMB, C_CEMB])
            print(f"\n{metric}:")
            print(piv.to_string())
        print("\nk used per branch:", {b: k_per_branch[b] for b in k_per_branch})
        print("\nFull table:")
        print(comp_df.to_string(index=False))
    for title, tbl in extra.items():
        print(f"\n--- {title} ---")
        print(tbl.to_string(index=False))
    print(f"\n[compare] wrote {comp_csv}\n[compare] wrote {comp_json}")

    # ---- figure: grouped bars, one subplot per metric ----
    if not args.no_plot and not comp_df.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.rcParams["svg.fonttype"] = "none"   # editable text in Illustrator
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42
            import matplotlib.pyplot as plt
            conds = [C_CODES, C_DEMB, C_CEMB]
            branches = [b for b in ("Cell-type", "Niche")
                        if b in set(comp_df["branch"])]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), squeeze=False)
            for ax, metric in zip(axes[0], ("NMI", "ARI")):
                x = np.arange(len(branches))
                w = 0.26
                for i, cond in enumerate(conds):
                    vals = [comp_df[(comp_df.branch == b) & (comp_df.condition == cond)][metric]
                            .mean() for b in branches]
                    bars = ax.bar(x + (i - 1) * w, vals, w,
                                  label=cond.split(". ", 1)[-1])
                    for rect, v in zip(bars, vals):
                        if v == v:
                            ax.annotate(f"{v:.3f}",
                                        (rect.get_x() + rect.get_width() / 2, v),
                                        ha="center", va="bottom", fontsize=7,
                                        xytext=(0, 1), textcoords="offset points")
                ax.set_xticks(x)
                ax.set_xticklabels(branches)
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} vs Ground-Truth Labels")
            axes[0][0].legend(title="Representation", frameon=False, fontsize=8)
            fig.suptitle("Discrete Codes vs Clustered Embeddings "
                         "(k = Number of Discrete Codes)", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
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
