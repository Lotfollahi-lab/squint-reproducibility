#!/usr/bin/env python
"""
cifm_mse_remaining.py — the MSE readings not yet tested.
=============================================================================
Goal: reproduce the four MSE values in Fig. 2B. Nothing else.

    CIFM 0.144   NeighborAvg 0.205   BernRnd 0.400   UnifRnd 0.409

Already ruled out (104-cell grid in cifm_mse_definition.py, and the target_sum
sweep):
  * space: log1p(1e4-norm) is right, linear is not (NeighborAvg 23.4 vs 0.205)
  * form: pooled is right, balanced-Eq15 is not (NeighborAvg 3.18)
  * target_sum: 1e4 is right; it is the ONLY value at which NeighborAvg
    reproduces 0.205 (we get 0.1894; at the median we get 0.0116)
  * post-processing of the prediction: as-is 1.7682, tutorial renorm 0.4095,
    marginal renorm 0.3361, p>prevalence renorm 0.3121, oracle logsum 0.2056
  * the prediction's per-gene magnitude is CORRECT (3.29 vs truth 3.13); the
    whole row-sum gap is density, 0.2211 nonzero vs 0.0254

So the space, the form and the input normalisation are settled. What remains
untested is how the MSE is AGGREGATED, and over WHICH ENTRIES.

THE KEY REFERENCE THIS SCRIPT ADDS. An all-zeros prediction. In the settled
space its MSE is just mean(truth^2), and it calibrates everything: if all-zeros
lands near 0.28, then their 0.144 (CIFM) and 0.205 (NeighborAvg) are both BETTER
than predicting nothing, and their ~0.40 randoms are worse. That fixes the scale
of the whole figure and tells us whether 0.144 is even attainable by a prediction
of our output's density.

REMAINING AGGREGATIONS TESTED HERE (all on the faithful gated output, plus the
tutorial-renormalised version, plus the controls so every row is comparable):
  pooled-mean            mean over all (cell, gene) entries        [reference]
  per-cell-median        MSE within a cell, then MEDIAN over cells. Fig. 2B is a
                         bar chart with error bars; a median is a plausible
                         summary and is far less sensitive to a few inflated
                         cells than the mean.
  per-gene-mean/median   MSE within a gene, then mean / median over genes. Their
                         other panels are per-gene (Spearman is described as
                         ranking genes), so a per-gene MSE is plausible.
  union-support          restricted to entries where truth>0 OR pred>0, i.e.
                         ignoring the vast agreed-zero background.
  truth-support          restricted to truth>0 (already known: 2.1549).
  top1000-DE             restricted to the 1,000 most differentially expressed
                         genes. Their Appdx C / Fig. 6C reports exactly this
                         variant, so the transform is theirs, not ours. DE is
                         computed on the CONTEXT cells only, one-vs-rest by
                         variance of the log1p truth (no labels available here).
  sqrt-space             MSE after sqrt on both sides -- a variance-stabilising
                         choice some pipelines use instead of log1p.

Every aggregation is applied identically to CIFM, NeighborAvg, UnifRnd, BernRnd,
all-zeros and the constant profile, against one shared truth. A reading is only
a candidate if it puts CIFM near 0.144 AND NeighborAvg near 0.205 AND the two
random baselines near 0.40 -- four constraints, which is a strong filter.

Usage
-----
  python cifm_mse_remaining.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"
TARGETS = {"CIFM": 0.144, "NeighborAvg": 0.205,
           "BernRnd": 0.400, "UnifRnd": 0.409}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    ap.add_argument("--h5ad", type=Path, default=None)
    ap.add_argument("--mask-frac", type=float, default=0.05)
    ap.add_argument("--max-eval", type=int, default=4000)
    ap.add_argument("--knn-k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import scanpy as sc
    import torch
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

    path = a.h5ad if a.h5ad is not None else (repo / "adata.h5ad")
    adata = sc.read_h5ad(path)
    genes = adata.var.index.astype(str).tolist()
    xy = np.asarray(adata.obsm["spatial"], np.float32)[:, :2]
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    model.channel_matching([[g] for g in genes], model.channel2ensembl_ids_source)
    X = (adata.X.toarray() if hasattr(adata.X, "toarray")
         else np.asarray(adata.X)).astype(np.float32)
    r = float(model.radius_spatial_graph)

    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    xth = xmin + (xmax - xmin) * 0.6
    yth = ymin + (ymax - ymin) * 0.5
    te = np.where((xy[:, 0] > xth) & (xy[:, 1] < yth))[0]
    rng = np.random.default_rng(a.seed)
    k = max(1, int(round(a.mask_frac * te.size)))
    q = np.sort(rng.choice(te, size=k, replace=False))
    ctx = np.setdiff1d(te, q)
    if q.size > a.max_eval:
        q = np.sort(rng.choice(q, size=a.max_eval, replace=False))
    truth = X[q].astype(np.float64)
    print(f"{path.name}: scattered {a.mask_frac:.0%} in the test region -> "
          f"context {ctx.size}, scoring {q.size}")

    n_ctx, G = X[ctx].shape; n_q = q.size
    with torch.no_grad():
        e = torch.tensor(X[ctx], dtype=torch.float32, device=dev)
        e = torch.cat([e, torch.zeros(n_q, G, device=dev)], 0)
        c = torch.tensor(np.concatenate([xy[ctx], xy[q]], 0), dtype=torch.float32)
        c = torch.cat([c, torch.zeros(c.shape[0], 1)], 1).to(dev)
        ei = radius_graph(c, r=r, max_num_neighbors=10000, loop=True)
        mp = torch.arange(n_ctx, n_ctx + n_q, device=dev)
        emb = model.encode(e, c, ei)
        emb[mp] = model.mask_embedding(
            torch.zeros(1, dtype=torch.int64, device=dev))
        dec = model.mask_cell_decoder(emb, c, ei)[0][mp]
        mm = model.relu(model.mask_cell_expression(dec))
        ppr = model.sigmoid(model.mask_cell_dropout(dec))
        g = mm.clone(); g[ppr <= 0.5] = 0.0
        cifm = g.cpu().numpy().astype(np.float64)
    del e, c, ei, emb, dec, mm, ppr, g
    if dev == "cuda":
        torch.cuda.empty_cache()

    lin = np.expm1(np.clip(cifm, 0, None))
    rs = lin.sum(1, keepdims=True); rs = np.where(rs > 0, rs, 1.0)
    cifm_rn = np.log1p(lin / rs * 1e4)

    Xc = X[ctx].astype(np.float64)
    nn = NearestNeighbors(n_neighbors=min(a.knn_k, ctx.size)).fit(xy[ctx])
    nbr = nn.radius_neighbors(xy[q], radius=r, return_distance=False)
    cmean = Xc.mean(0)
    navg = np.stack([Xc[i].mean(0) if len(i) else cmean for i in nbr])
    zeros = np.zeros_like(truth)
    const = np.repeat(cmean[None, :], n_q, axis=0)
    unif = rng.uniform(0.0, 1.0, size=truth.shape)
    bern = (rng.random(truth.shape) < 0.5) * 1.0

    PREDS = [("CIFM (gate)", cifm), ("CIFM (tut. renorm)", cifm_rn),
             ("NeighborAvg", navg), ("UnifRnd", unif), ("BernRnd", bern),
             ("ALL-ZEROS", zeros), ("CONSTANT", const)]

    # top-1000 DE genes, computed on CONTEXT cells only (no labels available,
    # so highest-variance in the log1p truth -- their Appdx C uses DE genes)
    de = np.argsort(-Xc.var(0))[:1000]
    tmask = truth > 0

    def pooled(T, P):    return float(np.mean((T - P) ** 2))
    def cell_med(T, P):  return float(np.median(np.mean((T - P) ** 2, axis=1)))
    def gene_mean(T, P): return float(np.mean(np.mean((T - P) ** 2, axis=0)))
    def gene_med(T, P):  return float(np.median(np.mean((T - P) ** 2, axis=0)))

    def union(T, P):
        m = (T > 0) | (P > 0)
        return float(((T[m] - P[m]) ** 2).mean()) if m.any() else float("nan")

    def tsupport(T, P):
        return float(((T[tmask] - P[tmask]) ** 2).mean())

    def de1000(T, P):    return float(np.mean((T[:, de] - P[:, de]) ** 2))
    def sqrt_sp(T, P):
        return float(np.mean((np.sqrt(np.clip(T, 0, None))
                              - np.sqrt(np.clip(P, 0, None))) ** 2))

    FORMS = [("pooled-mean", pooled), ("per-cell-median", cell_med),
             ("per-gene-mean", gene_mean), ("per-gene-median", gene_med),
             ("union-support", union), ("truth-support", tsupport),
             ("top1000-DE", de1000), ("sqrt-space", sqrt_sp)]

    print(f"\ntruth: nonzero {float((truth>0).mean()):.4f}, "
          f"mean(truth^2) = {float(np.mean(truth**2)):.4f}  "
          f"<- this IS the all-zeros MSE, and it calibrates the whole figure")

    print("\n" + "=" * 104)
    print("REMAINING MSE READINGS — a candidate must fit ALL FOUR published "
          "values at once")
    print("   CIFM 0.144   NeighborAvg 0.205   BernRnd 0.400   UnifRnd 0.409")
    print("=" * 104)
    hdr = f"  {'aggregation':18s}" + "".join(f"{n[:13]:>14s}" for n, _ in PREDS)
    print(hdr)
    rows = {}
    for fn, ff in FORMS:
        vals = [ff(truth, P) for _, P in PREDS]
        rows[fn] = vals
        print(f"  {fn:18s}" + "".join(f"{v:>14.4f}" for v in vals))

    print("\n" + "=" * 104)
    print("  fit to the four published values (lower = better; each is |ours - theirs|)")
    print(f"  {'aggregation':18s}{'dCIFM':>9s}{'dNAvg':>9s}{'dBern':>9s}"
          f"{'dUnif':>9s}{'total':>9s}")
    names = [n for n, _ in PREDS]
    best = None
    for fn, vals in rows.items():
        d = {}
        # PREDS labels are longer than the TARGETS keys ("CIFM (gate)" etc.)
        for key in TARGETS:
            col = next(i for i, n in enumerate(names) if n.startswith(key))
            d[key] = abs(vals[col] - TARGETS[key])
        tot = sum(d.values())
        print(f"  {fn:18s}{d['CIFM']:>9.3f}{d['NeighborAvg']:>9.3f}"
              f"{d['BernRnd']:>9.3f}{d['UnifRnd']:>9.3f}{tot:>9.3f}")
        if best is None or tot < best[0]:
            best = (tot, fn)
    print(f"\n  closest aggregation overall: {best[1]} (total residual {best[0]:.3f})")
    print("\n  If NO row fits all four, then the metric is not the variable: the")
    print("  space, form and target_sum are already pinned by NeighborAvg, and")
    print("  what differs is the PREDICTION -- ours calls 8.7x too many genes")
    print("  expressed at otherwise-correct magnitudes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
