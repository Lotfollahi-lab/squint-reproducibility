#!/usr/bin/env python3
"""
Coupling diagnostics for SQUINT's dual cell / niche encoder.

The s56 coupling-method sweep (coupled, Y-shape, cross-stitch, stop-gradient,
soft-L2) all tied with the DECOUPLED baseline (s55_v3). That tells us
trunk-sharing is not the lever. This script measures, on an ALREADY-TRAINED
decoupled model's saved embeddings (NO retraining, NO GPU), WHICH lever could
actually beat decoupled — so the next farm sweep is informed, not a guess.

It answers three questions, each mapping to a candidate coupling idea:

  1. REDUNDANCY  — are the cell and niche representations already encoding the
     same thing?  (CKA / CCA between z_cell and z_niche, both pre-VQ and
     quantized.)  HIGH redundancy -> a DISENTANGLEMENT penalty (idea #2) is the
     lever: force the niche code to capture only complementary signal.

  2. CROSS-PREDICTABILITY — does each branch leak the OTHER's label?
     (kNN balanced accuracy: niche_emb -> cell-type, cell_emb -> niche-label,
     vs each branch's native task.)  If niche_emb predicts cell-type nearly as
     well as cell_emb does, the niche code is redundantly re-encoding identity.

  3. COMPOSITION PREMISE — does the local CELL-TYPE COMPOSITION around a cell
     predict its niche label?  (Build the within-section spatial kNN graph,
     form each cell's normalized histogram of NEIGHBOUR cell codes, kNN-predict
     the niche label.)  Compared to the niche-code ceiling and the ego-cell
     baseline.  If composition predicts niche ~ as well as the niche code and
     >> the ego baseline, then injecting neighbour cell composition into the
     niche branch (idea #1, "compositional cell->niche coupling") has real
     headroom and is the lever.

Reads `predicted_adata.h5ad`:
    obsm['cell_latent'] / obsm['neighborhood_latent']  -> z_cell / z_niche (pre-VQ)
    obsm['cell_emb']    / obsm['neighborhood_emb']      -> z_q_cell / z_q_niche
    obs['cell_code_index'] / obs['neighborhood_code_index'] -> L0 codes
    obsm['spatial']                                     -> xy coordinates
    obs['adata_batch_id'] (or 'batch')                  -> section id
    obs[<cell-type>] / obs[<niche>]                     -> labels

numpy-only core (linear/RBF CKA, CCA, kNN classifier, composition histogram all
implemented from scratch); uses scikit-learn ONLY to speed up the spatial kNN
graph + classifier when available (graceful numpy fallback otherwise). Requires
anndata to read the h5ad.

Usage:
    # default: the decoupled s55_v3 reference, all seeds of its multiseed sweep
    python diagnose_coupling.py
    python diagnose_coupling.py --variant s55_v3_ --timestamp latest
    python diagnose_coupling.py --predicted-adata /path/to/predicted_adata.h5ad
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except Exception:                                    # pragma: no cover
    pd = None

DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"
DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"
DEFAULT_VARIANT_PREFIX = "s55_v3_"      # the decoupled reference we want to beat

CELL_LABELS = ("cell_type", "cell_types", "annotation", "Cell_class")
NICHE_LABELS = ("niche", "Sub_molecular_tissue_region", "ccf_region_name",
                "spatial_cluster", "region")
SECTION_KEYS = ("adata_batch_id", "batch", "sample", "section")

LATENT_CELL, LATENT_NICHE = "cell_latent", "neighborhood_latent"
EMB_CELL, EMB_NICHE = "cell_emb", "neighborhood_emb"
CODE_CELL, CODE_NICHE = "cell_code_index", "neighborhood_code_index"

# Subsample caps (diagnostics are estimates; full data is unnecessary + slow).
N_SUB_CKA_RBF = 4000      # RBF-CKA Gram matrices are O(n^2)
N_SUB_CLF = 8000          # kNN classifier subsample
KNN_CLF_K = 15            # neighbours for the label classifiers
KNN_GRAPH_K = 15          # spatial neighbours for the composition graph
RNG_SEED = 0


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _read_run_dir_col(csv_path):
    if pd is None:
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    col = next((c for c in ("run_dir", "run_directory", "rundir", "dir")
                if c in df.columns), None)
    if col is None:
        return []
    return [str(x) for x in df[col].dropna().tolist()]


def _latest_ts_dir(parent):
    subs = [p for p in glob.glob(os.path.join(parent, "*")) if os.path.isdir(p)]
    if not subs:
        return None
    return sorted(subs, key=lambda p: os.path.basename(p))[-1]


def _resolve_adata_paths(args):
    """Return a list of (label, predicted_adata_path). Multiseed -> one per seed."""
    if args.predicted_adata:
        return [("single", args.predicted_adata)]
    if args.run_dir:
        return [("single", os.path.join(args.run_dir, "predicted_adata.h5ad"))]

    base = os.path.join(args.artifacts_root, args.dataset)
    variant = args.variant

    # Prefer the multiseed sweep (gives per-seed adata for mean +/- sd).
    ms = sorted(d for d in glob.glob(os.path.join(base, f"{variant}*__multiseed"))
                if os.path.isdir(d))
    if ms:
        sweep = ms[0]
        ts = (os.path.join(sweep, args.timestamp)
              if args.timestamp and args.timestamp != "latest"
              else _latest_ts_dir(sweep))
        if ts and os.path.isfile(os.path.join(ts, "seed_run_index.csv")):
            run_dirs = _read_run_dir_col(os.path.join(ts, "seed_run_index.csv"))
            out = []
            for i, rd in enumerate(run_dirs):
                p = os.path.join(rd, "predicted_adata.h5ad")
                if os.path.isfile(p):
                    out.append((f"seed{i}", p))
            if out:
                return out
            print(f"[diag] multiseed index found but no predicted_adata.h5ad "
                  f"under its run_dirs ({ts}); falling back to single runs.",
                  file=sys.stderr)

    # Fall back to plain timestamped run dirs under the variant.
    var_dirs = sorted(d for d in glob.glob(os.path.join(base, f"{variant}*"))
                      if os.path.isdir(d) and not d.endswith("__multiseed"))
    if not var_dirs:
        raise SystemExit(f"[diag] no run dirs matching {variant!r} under {base}")
    seed_dirs = sorted(d for d in glob.glob(os.path.join(var_dirs[0], "*"))
                       if os.path.isdir(d))
    out = []
    for d in seed_dirs:
        p = os.path.join(d, "predicted_adata.h5ad")
        if os.path.isfile(p):
            out.append((os.path.basename(d), p))
    if not out:
        raise SystemExit(f"[diag] no predicted_adata.h5ad under {var_dirs[0]}")
    return out


# ---------------------------------------------------------------------------
# Similarity / redundancy
# ---------------------------------------------------------------------------
def _center_cols(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    """Linear CKA (feature-space, O(n d^2)) between (n,dx) and (n,dy)."""
    X = _center_cols(np.asarray(X, float))
    Y = _center_cols(np.asarray(Y, float))
    xty = X.T @ Y
    hsic_xy = float(np.sum(xty * xty))                 # ||X^T Y||_F^2
    hsic_xx = float(np.sum((X.T @ X) ** 2))
    hsic_yy = float(np.sum((Y.T @ Y) ** 2))
    denom = (hsic_xx * hsic_yy) ** 0.5
    return hsic_xy / denom if denom > 0 else float("nan")


def _rbf_gram(Z, gamma=None):
    sq = np.sum(Z * Z, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    if gamma is None:
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / (med + 1e-12)                    # median heuristic
    return np.exp(-gamma * d2)


def _centered_hsic(K, L):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float(np.sum(Kc * Lc))


def rbf_cka(X, Y, n_sub=N_SUB_CKA_RBF, seed=RNG_SEED):
    """Kernel (RBF) CKA on a subsample (Gram matrices are O(n^2))."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n = X.shape[0]
    if n > n_sub:
        idx = np.random.default_rng(seed).choice(n, n_sub, replace=False)
        X, Y = X[idx], Y[idx]
    K, L = _rbf_gram(X), _rbf_gram(Y)
    hkl = _centered_hsic(K, L)
    hkk = _centered_hsic(K, K)
    hll = _centered_hsic(L, L)
    denom = (hkk * hll) ** 0.5
    return hkl / denom if denom > 0 else float("nan")


def cca_mean_corr(X, Y, k=10, reg=1e-4):
    """Mean of the top-k canonical correlations between X and Y."""
    X = _center_cols(np.asarray(X, float))
    Y = _center_cols(np.asarray(Y, float))

    def _whiten(A):
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        keep = s > (s.max() * 1e-6) if s.size else np.array([], bool)
        return U[:, keep]                              # orthonormal column space

    Ux, Uy = _whiten(X), _whiten(Y)
    if Ux.shape[1] == 0 or Uy.shape[1] == 0:
        return float("nan")
    s = np.linalg.svd(Ux.T @ Uy, compute_uv=False)     # canonical correlations
    k = min(k, s.size)
    return float(np.mean(np.clip(s[:k], 0.0, 1.0)))


# ---------------------------------------------------------------------------
# kNN classifier (numpy fallback; sklearn if present)
# ---------------------------------------------------------------------------
def _standardize(train, test):
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def _balanced_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        m = y_true == c
        if m.sum() > 0:
            recalls.append(float((y_pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def _knn_predict_numpy(Xtr, ytr, Xte, k):
    pred = np.empty(Xte.shape[0], dtype=ytr.dtype)
    tr_sq = np.sum(Xtr * Xtr, axis=1)
    step = 1024
    for s in range(0, Xte.shape[0], step):
        q = Xte[s:s + step]
        d2 = np.sum(q * q, axis=1)[:, None] + tr_sq[None, :] - 2.0 * (q @ Xtr.T)
        nn = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]
        for j in range(q.shape[0]):
            vals, cnts = np.unique(ytr[nn[j]], return_counts=True)
            pred[s + j] = vals[np.argmax(cnts)]
    return pred


def knn_balanced_accuracy(X, labels, n_sub=N_SUB_CLF, k=KNN_CLF_K, seed=RNG_SEED,
                          min_per_class=4):
    """kNN balanced accuracy predicting `labels` from features X (subsampled,
    standardized, 50/50 split). Returns nan if too few labelled cells."""
    X = np.asarray(X, float)
    y = np.asarray(labels)
    keep = np.array([str(v) not in ("nan", "None", "") and v == v for v in y])
    X, y = X[keep], y[keep]
    if y.size < 50 or np.unique(y).size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if X.shape[0] > n_sub:
        idx = rng.choice(X.shape[0], n_sub, replace=False)
        X, y = X[idx], y[idx]
    # drop ultra-rare classes that can't be split
    vals, cnts = np.unique(y, return_counts=True)
    ok = set(vals[cnts >= min_per_class])
    m = np.array([v in ok for v in y])
    X, y = X[m], y[m]
    if np.unique(y).size < 2:
        return float("nan")
    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]
    cut = X.shape[0] // 2
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]
    Xtr, Xte = _standardize(Xtr, Xte)
    try:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=min(k, len(ytr)))
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
    except Exception:
        ypred = _knn_predict_numpy(Xtr, ytr, Xte, min(k, len(ytr)))
    return _balanced_accuracy(yte, ypred)


# ---------------------------------------------------------------------------
# Neighbour cell-type composition (idea #1 premise)
# ---------------------------------------------------------------------------
def neighbour_code_composition(coords, sections, cell_codes, n_codes,
                               k=KNN_GRAPH_K, max_per_section=60000):
    """Per cell, the normalized histogram over the cell-code of its `k` nearest
    spatial neighbours WITHIN the same section. Returns (n_cells, n_codes)."""
    coords = np.asarray(coords, float)
    sections = np.asarray(sections)
    cell_codes = np.asarray(cell_codes).astype(int)
    comp = np.zeros((coords.shape[0], n_codes), dtype=np.float32)

    for sec in np.unique(sections):
        sidx = np.where(sections == sec)[0]
        if sidx.size < 2:
            continue
        pts = coords[sidx]
        codes = cell_codes[sidx]
        kk = min(k, sidx.size - 1)
        nbr_idx = _spatial_knn(pts, kk, max_per_section)   # (m, kk) local indices
        for row, nbrs in enumerate(nbr_idx):
            vals, cnts = np.unique(codes[nbrs], return_counts=True)
            comp[sidx[row], vals] = cnts / max(1, cnts.sum())
    return comp


def _spatial_knn(pts, k, max_per_section):
    """(n,k) neighbour indices (excluding self). sklearn KDTree if available."""
    n = pts.shape[0]
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1).fit(pts)
        _, idx = nn.kneighbors(pts)
        return idx[:, 1:]                               # drop self
    except Exception:
        pass
    # numpy fallback (chunked brute force). Warn + subsample huge sections.
    if n > max_per_section:
        print(f"    [diag] section has {n} cells and sklearn is unavailable; "
              f"brute-force kNN would be slow — using all but in chunks.",
              file=sys.stderr)
    sq = np.sum(pts * pts, axis=1)
    out = np.empty((n, k), dtype=int)
    step = 1024
    for s in range(0, n, step):
        q = pts[s:s + step]
        d2 = np.sum(q * q, axis=1)[:, None] + sq[None, :] - 2.0 * (q @ pts.T)
        for j in range(q.shape[0]):
            row = d2[j]
            row[s + j] = np.inf                         # exclude self
            out[s + j] = np.argpartition(row, kth=k)[:k]
    return out


# ---------------------------------------------------------------------------
# Per-adata diagnostics
# ---------------------------------------------------------------------------
def _first_present(obs_cols, candidates):
    return next((c for c in candidates if c in obs_cols), None)


def _to_dense(a):
    return np.asarray(a.todense()) if hasattr(a, "todense") else np.asarray(a)


def _resolve_codes(A, obs_key, obsm_key, uns_key):
    """Per-cell L0 code, resolved from obs -> obsm -> uns (RVQ: take level 0).
    Mirrors report_codebook_usage.py's branch-key fallback so it works whether
    the run wrote codes to obs['*_code_index'], obsm['*_code_indices'], or
    uns['Indices_*']."""
    arr = None
    if obs_key in A.obs.columns:
        arr = A.obs[obs_key].to_numpy()
    elif obsm_key in A.obsm:
        arr = _to_dense(A.obsm[obsm_key])
    elif uns_key in A.uns:
        arr = np.asarray(A.uns[uns_key])
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:                                   # (n_cells, n_levels) RVQ
        arr = arr[:, 0]
    return arr.astype(int).reshape(-1)


def diagnose_one(path):
    import anndata as ad
    A = ad.read_h5ad(path)
    obs = A.obs
    out = {"path": path, "n_cells": int(A.n_obs)}

    cell_lab = _first_present(obs.columns, CELL_LABELS)
    niche_lab = _first_present(obs.columns, NICHE_LABELS)
    sec_key = _first_present(obs.columns, SECTION_KEYS)
    out["cell_label_key"] = cell_lab
    out["niche_label_key"] = niche_lab
    out["section_key"] = sec_key

    def _obsm(k):
        return _to_dense(A.obsm[k]) if k in A.obsm else None

    z_cell, z_niche = _obsm(LATENT_CELL), _obsm(LATENT_NICHE)
    e_cell, e_niche = _obsm(EMB_CELL), _obsm(EMB_NICHE)

    # 1. Redundancy ---------------------------------------------------------
    if z_cell is not None and z_niche is not None:
        out["cka_linear_latent"] = linear_cka(z_cell, z_niche)
        out["cka_rbf_latent"] = rbf_cka(z_cell, z_niche)
        out["cca_meancorr_latent"] = cca_mean_corr(z_cell, z_niche)
    if e_cell is not None and e_niche is not None:
        out["cka_linear_emb"] = linear_cka(e_cell, e_niche)

    # 2. Cross-predictability (use quantized emb = what downstream consumes) -
    feat_cell = e_cell if e_cell is not None else z_cell
    feat_niche = e_niche if e_niche is not None else z_niche
    if cell_lab is not None and feat_cell is not None:
        out["acc_cell_emb__celltype"] = knn_balanced_accuracy(
            feat_cell, obs[cell_lab].to_numpy())
    if cell_lab is not None and feat_niche is not None:
        out["acc_niche_emb__celltype"] = knn_balanced_accuracy(
            feat_niche, obs[cell_lab].to_numpy())
    if niche_lab is not None and feat_niche is not None:
        out["acc_niche_emb__nichelabel"] = knn_balanced_accuracy(
            feat_niche, obs[niche_lab].to_numpy())
    if niche_lab is not None and feat_cell is not None:
        out["acc_cell_emb__nichelabel"] = knn_balanced_accuracy(
            feat_cell, obs[niche_lab].to_numpy())

    # 3. Composition premise (idea #1) --------------------------------------
    cell_codes = _resolve_codes(A, CODE_CELL, "cell_code_indices", "Indices_cell")
    have_codes = cell_codes is not None
    have_spatial = "spatial" in A.obsm
    if have_codes and have_spatial and sec_key is not None and niche_lab is not None:
        n_codes = int(cell_codes.max()) + 1
        out["n_cell_codes_seen"] = int(np.unique(cell_codes).size)
        coords = _to_dense(A.obsm["spatial"])[:, :2]
        sections = obs[sec_key].to_numpy()
        comp = neighbour_code_composition(coords, sections, cell_codes, n_codes)
        out["acc_composition__nichelabel"] = knn_balanced_accuracy(
            comp, obs[niche_lab].to_numpy())
        # ego baseline: my OWN cell code (one-hot) predicting my niche label
        ego = np.zeros((len(cell_codes), n_codes), np.float32)
        ego[np.arange(len(cell_codes)), cell_codes] = 1.0
        out["acc_egocode__nichelabel"] = knn_balanced_accuracy(
            ego, obs[niche_lab].to_numpy())
    else:
        miss = [n for n, ok in [("codes", have_codes), ("spatial", have_spatial),
                                ("section", sec_key is not None),
                                ("niche_label", niche_lab is not None)] if not ok]
        # List what IS available so a rerun can pinpoint the right key names.
        print(f"    [diag] available obs cols: {list(obs.columns)[:40]}",
              file=sys.stderr)
        print(f"    [diag] available obsm keys: {list(A.obsm.keys())}",
              file=sys.stderr)
        print(f"    [diag] available uns keys: {list(A.uns.keys())[:40]}",
              file=sys.stderr)
        print(f"    [diag] skipping composition test (missing: {', '.join(miss)})",
              file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
def _agg(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] == r[key]]
    if not vals:
        return (float("nan"), float("nan"), 0)
    a = np.asarray(vals, float)
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, a.size)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=DEFAULT_VARIANT_PREFIX,
                    help="Variant key (full slug or prefix). Default s55_v3_ "
                         "(the decoupled reference).")
    ap.add_argument("--timestamp", default="latest")
    ap.add_argument("--predicted-adata", default=None,
                    help="Path to a single predicted_adata.h5ad (overrides --variant).")
    ap.add_argument("--run-dir", default=None,
                    help="A run dir containing predicted_adata.h5ad.")
    ap.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    paths = _resolve_adata_paths(args)
    print(f"[diag] dataset={args.dataset} variant={args.variant}")
    print(f"[diag] {len(paths)} adata(s) to diagnose:")
    for lab, p in paths:
        print(f"    {lab}: {p}")

    rows = []
    for lab, p in paths:
        print(f"\n[diag] === {lab} ===")
        try:
            r = diagnose_one(p)
            r["seed_label"] = lab
            rows.append(r)
        except Exception as exc:                         # noqa: BLE001
            print(f"    [diag] FAILED on {p}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
    if not rows:
        raise SystemExit("[diag] no adata diagnosed successfully.")

    metric_keys = [
        "cka_linear_latent", "cka_rbf_latent", "cca_meancorr_latent",
        "cka_linear_emb",
        "acc_cell_emb__celltype", "acc_niche_emb__celltype",
        "acc_niche_emb__nichelabel", "acc_cell_emb__nichelabel",
        "acc_composition__nichelabel", "acc_egocode__nichelabel",
    ]

    out_dir = Path(args.out) if args.out else (
        Path(args.artifacts_root) / args.dataset / "_coupling_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {k: _agg(rows, k) for k in metric_keys}

    # Persist per-seed + summary.
    if pd is not None:
        pd.DataFrame(rows).to_csv(out_dir / "coupling_diagnostics_per_seed.csv",
                                  index=False)
        srows = [{"metric": k, "mean": m, "std": s, "n": n}
                 for k, (m, s, n) in summary.items()]
        pd.DataFrame(srows).to_csv(out_dir / "coupling_diagnostics_summary.csv",
                                   index=False)
    with open(out_dir / "coupling_diagnostics_summary.json", "w") as fh:
        json.dump({k: {"mean": m, "std": s, "n": n}
                   for k, (m, s, n) in summary.items()}, fh, indent=2)

    # ---- report -----------------------------------------------------------
    def f(k):
        m, s, n = summary[k]
        return f"{m:.3f}±{s:.3f} (n={n})" if m == m else "n/a"

    print("\n" + "=" * 74)
    print("COUPLING DIAGNOSTICS  (mean ± sd across seeds)")
    print("=" * 74)
    print("\n1. REDUNDANCY  (cell vs niche representation; 1.0 = identical)")
    print(f"   CKA linear  (pre-VQ z)      : {f('cka_linear_latent')}")
    print(f"   CKA rbf     (pre-VQ z)      : {f('cka_rbf_latent')}")
    print(f"   CCA mean-corr (pre-VQ z)    : {f('cca_meancorr_latent')}")
    print(f"   CKA linear  (quantized emb) : {f('cka_linear_emb')}")

    print("\n2. CROSS-PREDICTABILITY  (kNN balanced accuracy)")
    print(f"   cell_emb  -> cell-type   (native) : {f('acc_cell_emb__celltype')}")
    print(f"   niche_emb -> cell-type   (leak?)  : {f('acc_niche_emb__celltype')}")
    print(f"   niche_emb -> niche-label (native) : {f('acc_niche_emb__nichelabel')}")
    print(f"   cell_emb  -> niche-label (leak?)  : {f('acc_cell_emb__nichelabel')}")

    print("\n3. COMPOSITION PREMISE  (predict niche label)")
    print(f"   neighbour cell-code composition   : {f('acc_composition__nichelabel')}")
    print(f"   ego cell-code only (baseline)     : {f('acc_egocode__nichelabel')}")
    print(f"   niche_emb (ceiling)               : {f('acc_niche_emb__nichelabel')}")

    # ---- interpretation heuristics ----------------------------------------
    print("\n" + "-" * 74)
    print("READOUT")
    print("-" * 74)
    cka = summary["cka_linear_latent"][0]
    leak = summary["acc_niche_emb__celltype"][0]
    native_cell = summary["acc_cell_emb__celltype"][0]
    comp = summary["acc_composition__nichelabel"][0]
    ego = summary["acc_egocode__nichelabel"][0]
    ceil = summary["acc_niche_emb__nichelabel"][0]

    if cka == cka:
        red = ("HIGH" if cka > 0.6 else "MODERATE" if cka > 0.35 else "LOW")
        print(f"- Redundancy is {red} (CKA={cka:.2f}). ", end="")
        if cka > 0.35:
            print("Cell & niche codes overlap -> a DISENTANGLEMENT penalty (#2) "
                  "could free the niche code to encode complementary signal.")
        else:
            print("Branches are already fairly complementary -> disentanglement "
                  "(#2) likely has little headroom.")
    if leak == leak and native_cell == native_cell and native_cell > 0:
        ratio = leak / native_cell
        print(f"- niche_emb recovers cell-type at {ratio:.0%} of cell_emb's "
              f"accuracy -> {'substantial identity leakage' if ratio > 0.7 else 'limited identity leakage'}.")
    if comp == comp and ceil == ceil and ego == ego:
        head = comp / ceil if ceil > 0 else float("nan")
        lift = comp - ego
        print(f"- Neighbour cell-COMPOSITION predicts niche at {comp:.2f} "
              f"({head:.0%} of the niche-code ceiling {ceil:.2f}; +{lift:.2f} over "
              f"the ego baseline {ego:.2f}).")
        if head == head and head > 0.7 and lift > 0.05:
            print("  -> STRONG support for COMPOSITIONAL cell->niche coupling (#1): "
                  "neighbour cell identity carries most of the niche signal, so "
                  "injecting it into the niche branch should help.")
        else:
            print("  -> Weak support for #1: neighbour cell composition alone does "
                  "not explain the niche label; the niche code uses more than "
                  "local cell-type identity.")
    print(f"\n[diag] wrote {out_dir}/coupling_diagnostics_{{per_seed,summary}}.csv")


if __name__ == "__main__":
    main()
