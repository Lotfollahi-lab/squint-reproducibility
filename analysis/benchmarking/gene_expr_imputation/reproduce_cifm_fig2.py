#!/usr/bin/env python
"""
reproduce_cifm_fig2.py — reproduce CIFM's Fig. 2B numbers, on their terms.
=============================================================================
Goal: establish whether we can hit the authors' PUBLISHED numbers before any
claim is made about their method on our data. Nothing here uses our dataset, our
split or our metrics.

TARGETS — Fig. 2B, "In-Sample Evaluation on Visium-HD", summary panels
(transcribed from the bar labels; ~ marks a label I read approximately):

    metric                 UnifRnd  BernRnd  NeighborAvg   CIFM
    Spearman r  (higher)     0.022    0.013       ~0.17    0.212
    MSE         (lower)      0.409   ~0.400       0.205    0.144
    Top-3 typing Acc         0.145    0.143      ~0.209    ~0.29

Fig. 2C (Xenium-Prime, cross-platform) adds CIFM-M (zero-shot) and CIFM-VX
(finetuned); its summary labels were not legible in our PDF render, so this
script does not assert them. Fig. 2D's top-k typing panel needs scTab AND
SCimilarity and is out of scope here (see LIMITATIONS).

WHAT THIS IMPLEMENTS, all from the paper
  * SPLIT (Appdx B.1, Fig. 2A): x_thres = x_min + (x_max-x_min)*0.6,
    y_thres = y_min + (y_max-y_min)*0.5; train x <= x_thres; val x > x_thres and
    y >= y_thres; test x > x_thres and y < y_thres. Val is used by nobody.
  * MASKING (Appdx B.3): "we remove 5% of the nodes for masking", randomly and
    uniformly. `--mask-mode scattered` (DEFAULT) masks --mask-frac of the cells
    inside a region and uses the REST OF THAT REGION as context, which is the
    regime the objective was trained under. `--mask-mode region` masks the whole
    test region and uses the train region as context -- our benchmark's geometry,
    kept here so the two can be compared under one metric.
  * GRAPH (Appdx B.1): r_thres = 20um for Visium-HD / Xenium-V1 / Xenium-Prime,
    150um for Visium-Spatial. Taken from the checkpoint's own
    radius_spatial_graph unless --radius is given.
  * MODEL CALL: `encode_decode` semantics -- gate at p > 0.5, single shot.
  * METRICS (Sec. 3): their "correlation" is SPEARMAN ("assesses whether the
    model ranks gene expression correctly"; Fig. 2B/2C axes read "Spearman
    Correlation r"), plus "mismatch error" = MSE. Scored in the model's own
    space, log1p(normalize_total(1e4)) -- no counts round trip, because their
    evaluation does not do one.
  * BASELINES (Sec. 3): "random expressions following parameterized uniform and
    Bernoulli distributions (with parameters learned from the data), as well as a
    naive neighborhood average approach that computes the mean expressions of the
    neighboring cells".
      NeighborAvg  mean over observed cells within r (their radius graph).
      UnifRnd      per-gene Uniform[0, max_g] fitted on observed cells.
      BernRnd      per-gene Bernoulli(p_g) x mean_nonzero_g, both fitted on
                   observed cells.
    The paper does not pin down the two random baselines further, so those two
    definitions are OUR reading -- and they are self-checking: if our UnifRnd
    lands near 0.022 and BernRnd near 0.013, the reading is right.
  16-NN and the observed-mean profile are printed too, as familiar references.

LIMITATIONS, stated up front
  * Fig. 2B is a mean over 10 Visium-HD samples (Fig. 1A: 10 samples /
    5,610,394 cells / 18,070 genes, i.e. ~561k cells each). The released demo
    `adata.h5ad` is 24,844 x 18,289 -- a crop of one sample with a slightly
    different gene set. So expect the ORDER OF MAGNITUDE and the ORDERING of
    methods to reproduce, not the third decimal. Point --h5ad at a real 10x
    sample for a like-for-like number.
  * Cell-typing accuracy (Fig. 2B panel 1, Fig. 2D) needs scTab and SCimilarity.
    Not implemented.
  * For samples far larger than the demo, pass --max-context to cap the context
    (a documented deviation) and --max-eval to cap the scored cells.

Usage
-----
  python reproduce_cifm_fig2.py                          # scattered 5%, demo data
  python reproduce_cifm_fig2.py --mask-mode region       # our benchmark geometry
  python reproduce_cifm_fig2.py --h5ad /path/visium_hd_sample.h5ad
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"

FIG2B = {          # transcribed targets; None = not legible in our render
    "UnifRnd":     dict(spearman=0.022, mse=0.409),
    "BernRnd":     dict(spearman=0.013, mse=0.400),
    "NeighborAvg": dict(spearman=0.17,  mse=0.205),
    "CIFM":        dict(spearman=0.212, mse=0.144),
}


def _rank_rows(A):
    A = np.asarray(A, np.float64)
    n, g = A.shape
    order = np.argsort(A, axis=1, kind="stable")
    ranks = np.empty_like(A)
    np.put_along_axis(ranks, order,
                      np.broadcast_to(np.arange(1.0, g + 1.0), (n, g)), axis=1)
    for i in range(n):
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


def _corr(T, P):
    T = np.asarray(T, np.float64); P = np.asarray(P, np.float64)
    T = T - T.mean(1, keepdims=True); P = P - P.mean(1, keepdims=True)
    nt = np.sqrt((T ** 2).sum(1)); npd = np.sqrt((P ** 2).sum(1))
    ok = (nt > 0) & (npd > 0)
    return (float(np.nanmean((T[ok] * P[ok]).sum(1) / (nt[ok] * npd[ok])))
            if ok.any() else float("nan"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    ap.add_argument("--h5ad", type=Path, default=None,
                    help="Sample to evaluate. Default: the repo's demo adata.h5ad.")
    ap.add_argument("--mask-mode", default="scattered",
                    choices=["scattered", "region"])
    ap.add_argument("--mask-frac", type=float, default=0.05,
                    help="Appdx B.3 uses 5%%. Only used by --mask-mode scattered.")
    ap.add_argument("--region", default="test", choices=["test", "train"],
                    help="Which region to mask inside, for --mask-mode scattered. "
                         "'test' is held out from the checkpoint; 'train' is a "
                         "CEILING reference only.")
    ap.add_argument("--radius", type=float, default=None,
                    help="Override the graph radius (Appdx B.1: 20um imaging / "
                         "150um Visium-Spatial). Default: the checkpoint's own.")
    ap.add_argument("--max-eval", type=int, default=4000)
    ap.add_argument("--max-context", type=int, default=0,
                    help="0 = use every observed cell (faithful). >0 caps the "
                         "context, a documented deviation for large samples.")
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
    r = float(a.radius if a.radius is not None else model.radius_spatial_graph)
    print(f"sample : {path}")
    print(f"shape  : {adata.n_obs} cells x {adata.n_vars} genes;  radius {r:.0f}um")

    # ---- their regional split (Appdx B.1 / Fig. 2A) ------------------------
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    xth = xmin + (xmax - xmin) * 0.6
    yth = ymin + (ymax - ymin) * 0.5
    tr = np.where(xy[:, 0] <= xth)[0]
    te = np.where((xy[:, 0] > xth) & (xy[:, 1] < yth))[0]
    va = np.setdiff1d(np.arange(adata.n_obs), np.union1d(tr, te))
    print(f"split  : train {tr.size}  val {va.size}  test {te.size}  "
          f"(paper: ~3:1:1)")

    rng = np.random.default_rng(a.seed)
    if a.mask_mode == "scattered":
        pool = te if a.region == "test" else tr
        k = max(1, int(round(a.mask_frac * pool.size)))
        q = np.sort(rng.choice(pool, size=k, replace=False))
        ctx = np.setdiff1d(pool, q)
        tag = (f"SCATTERED {a.mask_frac:.0%} inside the {a.region} region "
               f"(Appdx B.3 regime)")
    else:
        q, ctx = te, tr
        tag = "REGION HOLDOUT: whole test region masked, context = train region"

    if q.size > a.max_eval:
        q = np.sort(rng.choice(q, size=a.max_eval, replace=False))
    if a.max_context and ctx.size > a.max_context:
        ctx = np.sort(rng.choice(ctx, size=a.max_context, replace=False))
        print(f"  NOTE context capped at {ctx.size} cells (DEVIATION)")

    print(f"\nprotocol: {tag}")
    print(f"  context {ctx.size} cells, scoring {q.size} masked cells")
    d, _ = NearestNeighbors(n_neighbors=1).fit(xy[ctx]).kneighbors(xy[q])
    print(f"  distance to nearest OBSERVED cell: median {np.median(d):.1f}um; "
          f"within {r:.0f}um {float((d[:,0]<=r).mean()):.3f}")

    truth = X[q]

    # ---- CIFM: encode_decode, gated, single shot ---------------------------
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
        pp = model.sigmoid(model.mask_cell_dropout(dec))
        gated = mm.clone(); gated[pp <= 0.5] = 0.0
        cifm = gated.cpu().numpy().astype(np.float32)
    del e, c, ei, emb, dec, mm, pp, gated
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- their baselines, parameters fitted on OBSERVED cells only ---------
    Xc = X[ctx]
    nn = NearestNeighbors(n_neighbors=min(a.knn_k, ctx.size)).fit(xy[ctx])
    _, idx = nn.kneighbors(xy[q])
    knn = Xc[idx].mean(1)
    const = np.repeat(Xc.mean(0, keepdims=True), n_q, axis=0)

    nbr = nn.radius_neighbors(xy[q], radius=r, return_distance=False)
    navg = np.zeros_like(truth); has = np.zeros(n_q, bool)
    for i, ids in enumerate(nbr):
        if len(ids):
            navg[i] = Xc[ids].mean(0); has[i] = True
    navg[~has] = const[~has]
    print(f"  NeighborAvg: {has.mean():.3f} of masked cells have an observed "
          f"neighbour within {r:.0f}um (rest fall back to the observed mean)")

    gmax = Xc.max(0)
    unif = rng.uniform(0.0, np.maximum(gmax, 1e-12),
                       size=(n_q, G)).astype(np.float32)
    pg = (Xc > 0).mean(0)
    nzsum = Xc.sum(0); nzcnt = (Xc > 0).sum(0)
    mean_nz = np.divide(nzsum, np.maximum(nzcnt, 1)).astype(np.float32)
    bern = ((rng.random((n_q, G)) < pg) * mean_nz).astype(np.float32)

    # ---- score -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("REPRODUCTION vs Fig. 2B (their metric space, their baselines)")
    print("=" * 78)
    print(f"  {'method':22s}{'Spearman':>10s}{'target':>9s}"
          f"{'MSE':>10s}{'target':>9s}")
    rows = [("CIFM", cifm), ("NeighborAvg", navg), ("UnifRnd", unif),
            ("BernRnd", bern), (f"{a.knn_k}-NN mean", knn),
            ("CONSTANT (obs mean)", const)]
    for nm, P in rows:
        sp = _corr(_rank_rows(truth), _rank_rows(P))
        mse = float(np.mean((truth - P) ** 2))
        t = FIG2B.get(nm)
        ts = f"{t['spearman']:.3f}" if t else "-"
        tm = f"{t['mse']:.3f}" if t else "-"
        print(f"  {nm:22s}{sp:>10.4f}{ts:>9s}{mse:>10.4f}{tm:>9s}")

    print(f"\n  calibration: truth row-sum median {np.median(truth.sum(1)):.1f}, "
          f"CIFM {np.median(cifm.sum(1)):.1f} "
          f"(ratio {np.median(cifm.sum(1))/max(1e-9,np.median(truth.sum(1))):.2f}x); "
          f"nonzero frac truth {float((truth>0).mean()):.4f} "
          f"CIFM {float((cifm>0).mean()):.4f}")
    print("\n  If CIFM lands near 0.212 / 0.144 here, the implementation "
          "reproduces the paper.\n  If UnifRnd ~0.022 and BernRnd ~0.013, our "
          "reading of their random baselines is right too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
