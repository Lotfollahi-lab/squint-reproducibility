#!/usr/bin/env python
"""
cifm_vs_knn_own_data.py — on CIFM's OWN demo data, under CIFM's OWN split and
CIFM's OWN metric: does a trivial spatial k-NN beat CIFM?
=============================================================================
Everything here is set up from the paper so that a poor CIFM result cannot be
blamed on our data, our panel, our split or our metric choice:

  * DATA: the released demo `adata.h5ad` (Visium-HD, 24,844 cells x 18,289
    genes). Full transcriptome -- no panel, no ortholog map, no species shift.
  * SPLIT: their regional split, verbatim from Appdx B.1 --
        x_thres = x_min + (x_max - x_min) * 0.6
        y_thres = y_min + (y_max - y_min) * 0.5
        train: x <= x_thres     val: x > x_thres and y >= y_thres
        test:  x > x_thres and y < y_thres
    So the test region is a CONTIGUOUS rectangle (~40% x 50% of the bbox), which
    is their in-sample evaluation and is LARGER than our 25% x 25% held-out
    boxes. Context = train cells only; val cells are used by nobody.
  * MODEL CALL: `encode_decode` semantics via the gate
    (`expressions_dec[dropouts_dec <= 0.5] = 0`), radius graph at the
    checkpoint's own r=20um, single shot -- i.e. `predict_cells_at_locations`.
  * METRIC: their correlation is SPEARMAN ("correlation assesses whether the
    model ranks gene expression correctly", Sec. 3; Fig. 2B/2C axes read
    "Spearman Correlation r"). Pearson is reported alongside. Scored in the
    model's own space, log1p(normalize_total(1e4)) -- no counts round trip,
    because their evaluation does not do one.
  * BASELINES: 16-NN over train cells (our kNN floor) and the train-mean
    profile. Their own published baseline is "a naive neighborhood average
    approach that computes the mean expressions of the neighboring cells"
    (Sec. 3), so the radius-graph version of that is reported too, with its
    coverage, since interior test cells have no train neighbour within 20um.

Reference points from their Fig. 2B (in-sample, Visium-HD, read off the bar
labels): CIFM Spearman ~0.212 vs NeighborAvg ~0.17; MSE 0.144 vs 0.205.

Usage
-----
  python cifm_vs_knn_own_data.py [--max-test 4000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"


def _rank_rows(A):
    A = np.asarray(A, np.float64)
    n, g = A.shape
    order = np.argsort(A, axis=1, kind="stable")
    ranks = np.empty_like(A)
    np.put_along_axis(ranks, order,
                      np.broadcast_to(np.arange(1.0, g + 1.0), (n, g)), axis=1)
    for i in range(n):                       # average ties -> true Spearman
        v = A[i][order[i]]; r = ranks[i][order[i]]
        j = 0
        while j < g:
            k = j
            while k + 1 < g and v[k + 1] == v[j]:
                k += 1
            if k > j:
                r[j:k + 1] = r[j:k + 1].mean()
            j = k + 1
        ranks[i][order[i]] = r
    return ranks


def _corr_rows(T, P):
    T = np.asarray(T, np.float64); P = np.asarray(P, np.float64)
    T = T - T.mean(1, keepdims=True); P = P - P.mean(1, keepdims=True)
    nt = np.sqrt((T ** 2).sum(1)); npd = np.sqrt((P ** 2).sum(1))
    ok = (nt > 0) & (npd > 0)
    if not ok.any():
        return float("nan")
    return float(np.nanmean((T[ok] * P[ok]).sum(1) / (nt[ok] * npd[ok])))


def cell_pearson(T, P):  return _corr_rows(T, P)
def gene_pearson(T, P):  return _corr_rows(np.asarray(T).T, np.asarray(P).T)
def cell_spearman(T, P): return _corr_rows(_rank_rows(T), _rank_rows(P))
def gene_spearman(T, P): return _corr_rows(_rank_rows(np.asarray(T).T),
                                           _rank_rows(np.asarray(P).T))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    ap.add_argument("--max-test", type=int, default=4000,
                    help="Subsample the test region to this many cells (metric "
                         "only; the full train region is always the context).")
    ap.add_argument("--knn-k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import scanpy as sc
    import torch
    from sklearn.metrics import roc_auc_score
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
    genes = adata.var.index.astype(str).tolist()
    xy = np.asarray(adata.obsm["spatial"], np.float32)[:, :2]
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    model.channel_matching([[g] for g in genes], model.channel2ensembl_ids_source)
    X = (adata.X.toarray() if hasattr(adata.X, "toarray")
         else np.asarray(adata.X)).astype(np.float32)
    print(f"demo data: {adata.n_obs} cells x {adata.n_vars} genes")

    # ---- their regional split, Appdx B.1 -----------------------------------
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    xth = xmin + (xmax - xmin) * 0.6
    yth = ymin + (ymax - ymin) * 0.5
    is_train = xy[:, 0] <= xth
    is_test = (xy[:, 0] > xth) & (xy[:, 1] < yth)
    tr = np.where(is_train)[0]; te_all = np.where(is_test)[0]
    print(f"their regional split (Appdx B.1): x_thres={xth:.1f} y_thres={yth:.1f}")
    print(f"  train {tr.size}  test {te_all.size}  val {adata.n_obs-tr.size-te_all.size}"
          f"   (paper: ~3:1:1)")

    rng = np.random.default_rng(a.seed)
    te = (np.sort(rng.choice(te_all, size=a.max_test, replace=False))
          if te_all.size > a.max_test else te_all)
    print(f"  scoring {te.size} test cells")

    # how reachable is the test region at all? 4 hops x 20um
    d_tr, _ = NearestNeighbors(n_neighbors=1).fit(xy[tr]).kneighbors(xy[te])
    r = float(model.radius_spatial_graph)
    print(f"  distance to nearest TRAIN cell: median {np.median(d_tr):.1f}um; "
          f"within {r:.0f}um {float((d_tr[:,0]<=r).mean()):.3f}, "
          f"within {4*r:.0f}um {float((d_tr[:,0]<=4*r).mean()):.3f}")

    truth = X[te]

    # ---- CIFM: encode_decode semantics, single shot ------------------------
    n_ctx, G = X[tr].shape; n_q = te.size
    with torch.no_grad():
        e = torch.tensor(X[tr], dtype=torch.float32, device=dev)
        e = torch.cat([e, torch.zeros(n_q, G, device=dev)], 0)
        c = torch.tensor(np.concatenate([xy[tr], xy[te]], 0), dtype=torch.float32)
        c = torch.cat([c, torch.zeros(c.shape[0], 1)], 1).to(dev)
        ei = radius_graph(c, r=r, max_num_neighbors=10000, loop=True)
        mp = torch.arange(n_ctx, n_ctx + n_q, device=dev)
        emb = model.encode(e, c, ei)
        emb[mp] = model.mask_embedding(
            torch.zeros(1, dtype=torch.int64, device=dev))
        dec = model.mask_cell_decoder(emb, c, ei)[0][mp]
        m = model.relu(model.mask_cell_expression(dec))
        p = model.sigmoid(model.mask_cell_dropout(dec))
        gated = m.clone(); gated[p <= 0.5] = 0.0
        cifm = gated.cpu().numpy().astype(np.float32)
        p_np = p.cpu().numpy().astype(np.float32)
    del e, c, ei, emb, dec, m, p, gated
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- baselines ---------------------------------------------------------
    nn = NearestNeighbors(n_neighbors=a.knn_k).fit(xy[tr])
    dist, idx = nn.kneighbors(xy[te])
    knn_mean = X[tr][idx].mean(1)
    w = 1.0 / np.maximum(dist, 1e-8); w /= w.sum(1, keepdims=True)
    knn_wt = (X[tr][idx] * w[:, :, None].astype(np.float32)).sum(1)
    const = np.repeat(X[tr].mean(0, keepdims=True), n_q, axis=0)

    # their published baseline: mean of neighbours within the radius graph
    nbr_r = nn.radius_neighbors(xy[te], radius=r, return_distance=False)
    navg = np.zeros_like(truth); has = np.zeros(n_q, bool)
    for i, ids in enumerate(nbr_r):
        if len(ids):
            navg[i] = X[tr][ids].mean(0); has[i] = True
    navg[~has] = const[~has]          # no train neighbour in range -> fall back
    print(f"  NeighborAvg(r={r:.0f}um): {has.mean():.3f} of test cells have a "
          f"train neighbour in range (rest fall back to the train mean)")

    # ---- score -------------------------------------------------------------
    y = (truth > 0).ravel().astype(np.int8)
    rows = [("CIFM (gate, 1-shot)", cifm), (f"{a.knn_k}-NN mean", knn_mean),
            (f"{a.knn_k}-NN dist-weighted", knn_wt),
            ("NeighborAvg (their baseline)", navg), ("CONSTANT (train mean)", const)]
    print("\n" + "=" * 86)
    print("CIFM's OWN data, OWN regional split, OWN metric space "
          "(log1p 1e4-norm, 18,289 genes)")
    print("=" * 86)
    print(f"  {'method':30s}{'SPEARMAN':>10s}{'cw Pearson':>12s}"
          f"{'gw Spear':>10s}{'gw Pearson':>12s}{'MSE':>9s}{'AUROC':>8s}")
    res = {}
    for nm, P in rows:
        sp = cell_spearman(truth, P); cw = cell_pearson(truth, P)
        gs = gene_spearman(truth, P); gp = gene_pearson(truth, P)
        mse = float(np.mean((truth - P) ** 2))
        au = roc_auc_score(y, P.ravel())
        res[nm] = sp
        print(f"  {nm:30s}{sp:>10.4f}{cw:>12.4f}{gs:>10.4f}{gp:>12.4f}"
              f"{mse:>9.4f}{au:>8.4f}")
    print(f"\n  CIFM sparsity head alone, AUROC(p) = "
          f"{roc_auc_score(y, p_np.ravel()):.4f}")
    print("\n  Their Fig. 2B in-sample Visium-HD reference: CIFM Spearman ~0.212,"
          "\n  NeighborAvg ~0.17, MSE 0.144 vs 0.205.")
    best = max(res, key=res.get)
    print(f"\n  BEST by their own metric (Spearman): {best} ({res[best]:.4f})")
    d = res["CIFM (gate, 1-shot)"] - res[f"{a.knn_k}-NN mean"]
    print(f"  CIFM - {a.knn_k}NN = {d:+.4f}  -> "
          + ("CIFM wins" if d > 0 else "a trivial k-NN beats CIFM on its own data"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
