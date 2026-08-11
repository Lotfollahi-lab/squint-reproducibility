#!/usr/bin/env python
"""
cifm_marginal_vs_gate.py — the head we mis-signed, and what it fixes.
=============================================================================
`cifm_identify_output_space.py` settled two things about CIFM's output:

  * The space is log1p(1e4-normalised) -- the SAME space the model consumes.
    On the entries that are truly expressed, magnitude-vs-truth Pearson is
    0.8134 with a mean ratio of 1.023. No transform is needed; `expm1(...)`
    then rescaling is only a change of units, not a correction.
  * `sigmoid(mask_cell_dropout)` is P(EXPRESSED), not P(dropped):
    AUROC[1-p] = 0.1230, so AUROC[p] = 0.8770. CIFM's own gate agrees --
    `encode_decode` keeps entries with p > 0.5 -- so the head's NAME is
    misleading but its polarity is unambiguous.

That makes CIFM a factorised zero-inflated predictor: a CONDITIONAL magnitude
`m` times a sparsity probability `p`. Three ways to collapse it to a point
prediction, only the first of which we have ever benchmarked:

  UNGATED   m           run_cifm.py's DEFAULT. 99.76% dense against a truth
                        that is 2.41% dense. Discards the p head entirely, so
                        the 0.877 AUROC of sparsity information is thrown away
                        and to_counts()' row-normalisation then splits each
                        cell's depth across ~18.3k entries instead of ~441.
  HARD      m*(p>0.5)   CIFM's native inference (`--apply-dropout-gate`).
  MARGINAL  m*p         E[expression] under a zero-inflated likelihood. This is
                        the quantity a Pearson-against-observed-counts metric
                        actually asks for, and it is the one variant absent
                        from every diagnostic we have run so far.

Scored in all three spaces (log-direct / unit profile / harness counts) against
the CONSTANT and 16-NN controls, so the choice of space cannot flatter or
penalise any variant. Also reports AUROC/AP for the zero pattern and the
sparsity each variant implies, since that is the axis where UNGATED fails.

Held-out throughout: target cells are removed from the context.

Usage
-----
  python cifm_marginal_vs_gate.py [--n 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"


def pearson(a, b, axis):
    """Mean Pearson r along `axis` (1 = per cell, 0 = per gene)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if axis == 0:
        a, b = a.T, b.T
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    na = np.sqrt((a ** 2).sum(1)); nb = np.sqrt((b ** 2).sum(1))
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        return float("nan")
    return float(np.nanmean((a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])))


def unit(P):
    r = np.clip(np.asarray(P, float), 0, None)
    rs = r.sum(1, keepdims=True)
    return r / np.where(rs > 0, rs, 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import scanpy as sc
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.nn import radius_graph

    repo = a.cifm_repo.resolve(); sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFM.from_pretrained(str(repo),
        args=torch.load(repo / "models_cifm" / "args.pt")).to(dev)
    model.channel2ensembl_ids_source = torch.load(
        repo / "models_cifm" / "channel2ensembl.pt")
    model.eval()

    adata = sc.read_h5ad(repo / "adata.h5ad")
    raw = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    model.channel_matching([[i] for i in adata.var.index.tolist()],
                           model.channel2ensembl_ids_source)

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    C = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    xy = np.asarray(adata.obsm["spatial"], float)[:, :2]
    rng = np.random.default_rng(a.seed)
    sel = rng.choice(adata.n_obs, size=min(a.n, adata.n_obs // 4), replace=False)
    keep = np.setdiff1d(np.arange(adata.n_obs), sel)
    ctx = adata[keep].copy()

    # ---- both heads, ungated (held-out context) -----------------------------
    Xc = ctx.X.toarray() if hasattr(ctx.X, "toarray") else np.asarray(ctx.X)
    n_ctx, G = Xc.shape; n_q = len(sel)
    with torch.no_grad():
        e = torch.tensor(Xc, dtype=torch.float32, device=dev)
        e = torch.cat([e, torch.zeros(n_q, G, device=dev)], 0)
        c = torch.tensor(np.concatenate(
            [np.asarray(ctx.obsm["spatial"])[:, :2], xy[sel]], 0),
            dtype=torch.float32)
        c = torch.cat([c, torch.zeros(c.shape[0], 1)], 1).to(dev)
        ei = radius_graph(c, r=model.radius_spatial_graph,
                          max_num_neighbors=10000, loop=True)
        mp = torch.arange(n_ctx, n_ctx + n_q, device=dev)
        emb = model.encode(e, c, ei)
        emb[mp] = model.mask_embedding(
            torch.zeros(1, dtype=torch.int64, device=dev))
        dec = model.mask_cell_decoder(emb, c, ei)[0][mp]
        m = model.relu(model.mask_cell_expression(dec)).cpu().numpy()
        p = model.sigmoid(model.mask_cell_dropout(dec)).cpu().numpy()

    truth_log = X[sel]
    truth_cnt = C[sel]
    depth = truth_cnt.sum(1)

    variants = {
        "UNGATED  m         (our default)": m,
        "HARD     m*(p>0.5) (CIFM native)": m * (p > 0.5),
        "MARGINAL m*p       (zero-infl E)": m * p,
    }

    # ---- 0. sparsity: the axis where UNGATED fails --------------------------
    print("=" * 78 + "\n0. SPARSITY  (truth nonzero frac "
          f"{float((truth_log > 0).mean()):.4f})\n" + "=" * 78)
    for nm, P in variants.items():
        nz = float((P > 0).mean())
        print(f"  {nm:34s} nonzero {nz:.4f}  ({nz / max(1e-9, float((truth_log>0).mean())):5.1f}x truth)"
              f"  row-sum median {np.median(P.sum(1)):9.1f}"
              f"  (truth {np.median(truth_log.sum(1)):.1f})")

    # ---- 1. zero pattern, independent of magnitude -------------------------
    print("\n" + "=" * 78 + "\n1. ZERO PATTERN: AUROC / AP for (truth > 0)\n" + "=" * 78)
    y = (truth_log > 0).ravel().astype(int)
    for nm, s in (("p            (P expressed)", p.ravel()),
                  ("m            (magnitude)  ", m.ravel()),
                  ("m*p          (marginal)   ", (m * p).ravel())):
        print(f"  {nm}  AUROC {roc_auc_score(y, s):.4f}   AP {average_precision_score(y, s):.4f}")
    print(f"  (baseline AP = prevalence = {y.mean():.4f})")

    # ---- 2. Pearson, every variant x every space ----------------------------
    const_lin = np.repeat(np.expm1(X[keep]).mean(0, keepdims=True), n_q, 0)
    _, idx = NearestNeighbors(n_neighbors=16).fit(xy[keep]).kneighbors(xy[sel])
    knn_lin = np.expm1(X[keep])[idx].mean(1)

    spaces = {
        "A log-direct  (vs adata.X)": (lambda P: P, truth_log),
        "B unit profile (cell 11)  ": (lambda P: unit(np.expm1(P)),
                                       unit(np.expm1(truth_log))),
        "C harness counts          ": (lambda P: np.log1p(unit(np.expm1(P))
                                                         * depth[:, None]),
                                       np.log1p(truth_cnt)),
    }
    rows = {**variants,
            "CONSTANT (train mean)": np.log1p(const_lin),
            "16-NN average": np.log1p(knn_lin)}

    print("\n" + "=" * 78 + "\n2. PEARSON: variant x space (held-out)\n" + "=" * 78)
    best = {}
    for sp, (fn, T) in spaces.items():
        print(f"\n  {sp}")
        print(f"    {'prediction':34s}{'cell-wise':>12s}{'gene-wise':>12s}")
        for nm, P in rows.items():
            Q = fn(P)
            cw, gw = pearson(T, Q, 1), pearson(T, Q, 0)
            print(f"    {nm:34s}{cw:>12.4f}{gw:>12.4f}")
            if nm in variants and (sp not in best or cw > best[sp][1]):
                best[sp] = (nm, cw)
    print("\n  best CIFM variant per space: " +
          "; ".join(f"{s.strip()} -> {v[0].split()[0]} ({v[1]:.4f})"
                    for s, v in best.items()))

    print("\n  If MARGINAL/HARD beat UNGATED by a wide margin, run_cifm.py's "
          "\n  ungated default is the bug and the mmb numbers must be recomputed."
          "\n  If all three still trail CONSTANT, the metric -- not our code -- "
          "\n  is what CIFM's output does not support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
