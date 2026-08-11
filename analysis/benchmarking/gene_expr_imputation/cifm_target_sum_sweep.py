#!/usr/bin/env python
"""
!! NOTE any "0.144" / "0.205" below is a MISREAD of Fig. 2B's MSE panel.
   The real values are CIFM 0.266 and NeighborAvg 0.280, with large
   overlapping error bars. Corrected 2026-08-11.

cifm_target_sum_sweep.py — is the 9x magnitude inflation just `target_sum`?
=============================================================================
STATE OF PLAY. On CIFM's own demo data, their regional split and their scattered
5% masking, we reproduce their SPEARMAN (CIFM 0.2070 vs published 0.212;
NeighborAvg 0.1842 vs ~0.17) and we get NeighborAvg MSE 0.1894 (their 0.280)
under log1p(1e4-norm) + pooled MSE. A 104-cell grid over post-processings x
spaces x MSE forms bracketed their CIFM MSE of 0.266: the tutorial's own
renormalisation gives 0.4095 (1.54x) and an ORACLE forcing each row's log-space
sum to the truth's gives 0.2056 (0.77x).

The grid also isolated the cause. It is not density -- NeighborAvg is just as
dense (0.239 nonzero vs CIFM's 0.221) and still scores 0.19. It is MAGNITUDE:

    truth       log-space row sum  1452.3
    NeighborAvg                    1452.0   ratio 1.000   MSE 0.1894
    CIFM                          13297.3   ratio 9.156   MSE 1.7682

THE HYPOTHESIS. Our tutorial reproduction (`reproduce_cifm_tutorial.py` matched
`model.embed()` to 4 decimals) proves we match THE TUTORIAL. It does not prove
the tutorial matches TRAINING. Appdx B.1 says only "normalize gene counts and
conduct log1p-transformation" -- with NO target_sum, i.e. scanpy's default, the
MEDIAN library size. The tutorial's 1e4 is a demo convenience. Visium-HD 8um bins
are low-count, so 1e4 inflates every input by (1e4 / median library size); a
decoder trained on the median scale would then emit output inflated by the same
factor.

That predicts median library size ~= 1e4 / 9.156 ~= 1092. This script measures it
and sweeps target_sum. For each value the SAME normalisation is used for the
model input and for the ground truth -- which is what a masking-reconstruction
pipeline does, since input and target are one matrix.

WHAT TO LOOK FOR
  * calib ratio -> 1.00 at the right target_sum. That is the direct test.
  * CIFM MSE -> ~0.266 and NeighborAvg MSE -> ~0.280 at the same value.
  * Spearman should stay ~0.21. It is NOT invariant here (the input changes, not
    just the output scale), so if it collapses, the model is being fed something
    off-distribution and the hypothesis is wrong.

Usage
-----
  python cifm_target_sum_sweep.py
  python cifm_target_sum_sweep.py --target-sums median,1092,3000,10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"
TARGET_CIFM, TARGET_NAVG = 0.266, 0.280  # corrected 2026-08-11 (was a misread)


def _rank_rows(A):
    A = np.asarray(A, np.float64); n, g = A.shape
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
    return (float(np.nanmean((T[ok]*P[ok]).sum(1)/(nt[ok]*npd[ok])))
            if ok.any() else float("nan"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    ap.add_argument("--h5ad", type=Path, default=None)
    ap.add_argument("--target-sums", default="median,1092,2000,5000,10000",
                    help="Comma list; 'median' = scanpy's default (no target_sum).")
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
    base = sc.read_h5ad(path)
    genes = base.var.index.astype(str).tolist()
    xy = np.asarray(base.obsm["spatial"], np.float32)[:, :2]
    counts = (base.X.toarray() if hasattr(base.X, "toarray")
              else np.asarray(base.X)).astype(np.float32)
    # channel_matching depends only on gene identity, so once is enough
    model.channel_matching([[g] for g in genes], model.channel2ensembl_ids_source)
    r = float(model.radius_spatial_graph)

    lib = counts.sum(1)
    med = float(np.median(lib))
    print(f"{path.name}: {base.n_obs} cells x {base.n_vars} genes; r={r:.0f}um")
    print(f"library size: median {med:.1f}, mean {lib.mean():.1f}, "
          f"IQR [{np.percentile(lib,25):.0f}, {np.percentile(lib,75):.0f}]")
    print(f"PREDICTION: if the 9.156x inflation is target_sum, the median should "
          f"be ~{1e4/9.156:.0f}  ->  measured {med:.1f}  "
          f"({'CONSISTENT' if 0.5 < med/(1e4/9.156) < 2.0 else 'NOT consistent'})")

    # their regional split (Appdx B.1)
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
    print(f"scattered {a.mask_frac:.0%} in the test region -> context {ctx.size}, "
          f"scoring {q.size}")

    def norm_log(ts):
        """One normalisation, used for BOTH the model input and the truth."""
        rs = counts.sum(1, keepdims=True, dtype=np.float64)
        rs = np.where(rs > 0, rs, 1.0)
        return np.log1p(counts / rs * ts).astype(np.float32)

    def predict(Xn):
        n_ctx, G = Xn[ctx].shape; n_q = q.size
        with torch.no_grad():
            e = torch.tensor(Xn[ctx], dtype=torch.float32, device=dev)
            e = torch.cat([e, torch.zeros(n_q, G, device=dev)], 0)
            c = torch.tensor(np.concatenate([xy[ctx], xy[q]], 0),
                             dtype=torch.float32)
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
            out = g.cpu().numpy().astype(np.float32)
        del e, c, ei, emb, dec, mm, ppr, g
        if dev == "cuda":
            torch.cuda.empty_cache()
        return out

    nn = NearestNeighbors(n_neighbors=min(a.knn_k, ctx.size)).fit(xy[ctx])
    nbr = nn.radius_neighbors(xy[q], radius=r, return_distance=False)

    tss = []
    for tok in a.target_sums.split(","):
        tok = tok.strip()
        tss.append(("median", med) if tok == "median" else (tok, float(tok)))

    print("\n" + "=" * 100)
    print("TARGET_SUM SWEEP — one normalisation for input AND truth, everything "
          "else fixed")
    print(f"   targets: CIFM MSE {TARGET_CIFM:.3f}, NeighborAvg MSE "
          f"{TARGET_NAVG:.3f}, CIFM Spearman ~0.212")
    print("=" * 100)
    print(f"  {'target_sum':12s}{'in mean':>9s}{'truth Σ':>10s}{'CIFM Σ':>10s}"
          f"{'calib':>8s}{'nz':>7s}{'Spear':>8s}{'MSE':>9s}{'NAvg MSE':>10s}")
    for label, ts in tss:
        Xn = norm_log(ts)
        truth = Xn[q]
        cifm = predict(Xn)
        Xc = Xn[ctx]; cmean = Xc.mean(0)
        navg = np.stack([Xc[i].mean(0) if len(i) else cmean
                         for i in nbr]).astype(np.float32)
        tsum = float(np.median(truth.sum(1))); csum = float(np.median(cifm.sum(1)))
        sp = _corr(_rank_rows(truth), _rank_rows(cifm))
        mse = float(np.mean((truth - cifm) ** 2))
        mse_n = float(np.mean((truth - navg) ** 2))
        print(f"  {label:12s}{Xc.mean():>9.4f}{tsum:>10.1f}{csum:>10.1f}"
              f"{csum/max(1e-9,tsum):>8.3f}{float((cifm>0).mean()):>7.3f}"
              f"{sp:>8.4f}{mse:>9.4f}{mse_n:>10.4f}")

    print("\n  calib -> 1.00 with CIFM MSE ~0.144, NAvg ~0.205 and Spearman ~0.21")
    print("  at one target_sum would settle it: the paper's unspecified")
    print("  normalisation is the median, and 1e4 was the whole discrepancy.")
    print("  If calib stays ~9 at every value, the inflation is not target_sum")
    print("  and lives in the decoder head itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
