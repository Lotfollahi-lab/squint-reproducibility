#!/usr/bin/env python
"""
cifm_mse_definition.py — find the MSE definition that reproduces Fig. 2B.
=============================================================================
`reproduce_cifm_fig2.py` reproduced the authors' SPEARMAN on their own data,
their split and their masking regime:

    CIFM        Spearman 0.2070   (published 0.212)
    NeighborAvg Spearman 0.1842   (published ~0.17)

so the forward path is right. MSE did not follow:

    CIFM        MSE 1.7682        (published 0.144)
    NeighborAvg MSE 0.1894        (published 0.205)   <- already close

That pattern is the whole clue. Our NeighborAvg MSE is within 8% of theirs, so
the metric SPACE and the MSE FORM cannot be badly wrong. What differs is CIFM's
output magnitude: measured row sum 13,297 against a truth of 1,452 (9.16x) and a
nonzero fraction of 0.2211 against 0.0254.

So: which (post-processing x MSE form) makes BOTH published numbers fall out at
once? A definition that fixes CIFM while breaking NeighborAvg is wrong. This
script searches the grid and scores each cell by the joint residual.

POST-PROCESSINGS of the gated prediction
  as-is                 what encode_decode returns
  renorm-1e4            the tutorial's own cell 11: expm1 -> /rowsum -> x1e4,
                        then log1p (their stated way to "convert it into
                        normalize counts")
  renorm-linear-1e4     same but compared in LINEAR normalised space
  logsum-matched        scale so each row's LOG-space sum equals the truth's
                        (diagnostic: isolates sparsity from scale)
  oracle-total          renorm so each row's linear total equals that cell's own
                        true total (an ORACLE upper bound, not a candidate)

MSE FORMS
  pooled                mean over every (cell, gene) entry            <- our current
  balanced              Appdx B.3 Eq. 15: half-weight the X>0 entries and
                        half-weight the X=0 entries. This is their TRAINING loss,
                        and "mismatch error" may well be the same quantity.
  per-cell              mean within a cell, then averaged over cells
  expressed-only        mean over entries where truth > 0

Any combination is applied identically to CIFM and to NeighborAvg, and the same
truth is used throughout, so no cell of the grid can flatter one method.

Usage
-----
  python cifm_mse_definition.py [--mask-frac 0.05] [--max-eval 4000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"
TARGET = {"CIFM": 0.144, "NeighborAvg": 0.205}


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


# ---------- post-processings (all take/return LOG-space matrices) ------------
def _to_lin(L):
    return np.expm1(np.clip(np.asarray(L, np.float64), 0, None))


def _renorm_lin(L, total):
    lin = _to_lin(L)
    rs = lin.sum(1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    return lin / rs * total


def pp_asis(P, T):            return P
def pp_renorm1e4(P, T):       return np.log1p(_renorm_lin(P, 1e4))
def pp_renorm_oracle(P, T):   return np.log1p(_renorm_lin(P, _to_lin(T).sum(1, keepdims=True)))


def pp_logsum(P, T):
    ps = np.asarray(P, np.float64).sum(1, keepdims=True)
    ts = np.asarray(T, np.float64).sum(1, keepdims=True)
    return np.asarray(P, np.float64) * np.where(ps > 0, ts / np.where(ps > 0, ps, 1.0), 0.0)


# ---------- MSE forms --------------------------------------------------------
def mse_pooled(T, P):     return float(np.mean((T - P) ** 2))
def mse_percell(T, P):    return float(np.mean(np.mean((T - P) ** 2, axis=1)))


def mse_balanced(T, P):
    """Appdx B.3 Eq. 15, per sample then averaged."""
    T = np.asarray(T, np.float64); P = np.asarray(P, np.float64)
    pos = T > 0; neg = ~pos
    e2 = (T - P) ** 2
    a = e2[pos].mean() if pos.any() else 0.0
    b = e2[neg].mean() if neg.any() else 0.0
    return float(0.5 * a + 0.5 * b)


def mse_expressed(T, P):
    T = np.asarray(T, np.float64); P = np.asarray(P, np.float64)
    m = T > 0
    return float(((T[m] - P[m]) ** 2).mean()) if m.any() else float("nan")


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
    print(f"{path.name}: {adata.n_obs}x{adata.n_vars}; scattered "
          f"{a.mask_frac:.0%} in the test region -> context {ctx.size}, "
          f"scoring {q.size}")

    truth = X[q]
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
        g = mm.clone(); g[pp <= 0.5] = 0.0
        cifm = g.cpu().numpy().astype(np.float32)
        m_np = mm.cpu().numpy().astype(np.float32)
        p_np = pp.cpu().numpy().astype(np.float32)
    del e, c, ei, emb, dec, mm, pp, g
    if dev == "cuda":
        torch.cuda.empty_cache()

    nn = NearestNeighbors(n_neighbors=min(a.knn_k, ctx.size)).fit(xy[ctx])
    nbr = nn.radius_neighbors(xy[q], radius=r, return_distance=False)
    Xc = X[ctx]; const = Xc.mean(0)
    navg = np.stack([Xc[i].mean(0) if len(i) else const for i in nbr]).astype(np.float32)

    print(f"\ncalibration (log space): truth row-sum median "
          f"{np.median(truth.sum(1)):.1f}; CIFM {np.median(cifm.sum(1)):.1f}; "
          f"NeighborAvg {np.median(navg.sum(1)):.1f}")
    print(f"nonzero frac: truth {float((truth>0).mean()):.4f}; "
          f"CIFM {float((cifm>0).mean()):.4f}; "
          f"NeighborAvg {float((navg>0).mean()):.4f}")
    print(f"Spearman (invariant to every post-processing below except "
          f"linear-space ones): CIFM {_corr(_rank_rows(truth), _rank_rows(cifm)):.4f}, "
          f"NeighborAvg {_corr(_rank_rows(truth), _rank_rows(navg)):.4f}")

    # ---------------- every candidate prediction I can think of -------------
    # All are expressed in the model's own space, log1p(1e4-normalised).
    def renorm(P, total):
        lin = np.expm1(np.clip(np.asarray(P, np.float64), 0, None))
        rs = lin.sum(1, keepdims=True); rs = np.where(rs > 0, rs, 1.0)
        return np.log1p(lin / rs * total)

    def logsum_match(P):
        P = np.asarray(P, np.float64)
        ps = P.sum(1, keepdims=True); ts = truth.sum(1, keepdims=True)
        return P * np.where(ps > 0, ts / np.where(ps > 0, ps, 1.0), 0.0)

    def topk_match(P):
        """Keep, per cell, as many entries as the truth has nonzeros."""
        P = np.asarray(P, np.float64); out = np.zeros_like(P)
        kk = (truth > 0).sum(1)
        for i2 in range(P.shape[0]):
            n = int(kk[i2])
            if n <= 0:
                continue
            idx = np.argpartition(-P[i2], n - 1)[:n]
            out[i2, idx] = P[i2, idx]
        return out

    prev = float((truth > 0).mean())
    thr = float(np.quantile(p_np, 1.0 - prev))          # p-threshold at prevalence
    m_np64 = m_np.astype(np.float64); p_np64 = p_np.astype(np.float64)
    true_lin_tot = np.expm1(truth).sum(1, keepdims=True)

    CANDS = [
        ("gate (as-is)",            cifm),
        ("gate renorm-1e4",         renorm(cifm, 1e4)),
        ("gate renorm-median",      renorm(cifm, float(np.median(np.expm1(truth).sum(1))))),
        ("gate renorm-oracle",      renorm(cifm, true_lin_tot)),
        ("gate logsum-matched",     logsum_match(cifm)),
        ("gate topk-matched",       topk_match(cifm)),
        ("magnitude m (ungated)",   m_np64),
        ("magnitude renorm-1e4",    renorm(m_np64, 1e4)),
        ("marginal m*p",            m_np64 * p_np64),
        ("marginal renorm-1e4",     renorm(m_np64 * p_np64, 1e4)),
        ("lin-marginal p*expm1(m)", np.log1p(p_np64 * np.expm1(m_np64))),
        ("p>prev gate",             np.where(p_np64 > thr, m_np64, 0.0)),
        ("p>prev renorm-1e4",       renorm(np.where(p_np64 > thr, m_np64, 0.0), 1e4)),
    ]
    NAVG = [("NeighborAvg (as-is)", navg.astype(np.float64)),
            ("NeighborAvg renorm-1e4", renorm(navg, 1e4))]

    SPACES = [("log1p-1e4", lambda A: np.asarray(A, np.float64)),
              ("linear-1e4", lambda A: np.expm1(np.asarray(A, np.float64)))]
    FORMS = [("pooled", mse_pooled), ("balanced(Eq15)", mse_balanced),
             ("per-cell", mse_percell), ("expressed-only", mse_expressed)]

    print("\n" + "=" * 100)
    print("MSE SEARCH — targets: CIFM 0.144, NeighborAvg 0.205 (both from the "
          "SAME definition)")
    print("=" * 100)
    print(f"  {'space':12s}{'form':16s}{'prediction':26s}{'CIFM':>10s}"
          f"{'|d|':>8s}   NeighborAvg (as-is / renorm)")
    hits = []
    for sname, sf in SPACES:
        Tt = sf(truth)
        for fname, ff in FORMS:
            nvals = [ff(Tt, sf(P)) for _, P in NAVG]
            for cname, P in CANDS:
                v = ff(Tt, sf(P))
                dc = abs(v - TARGET["CIFM"])
                dn = min(abs(x - TARGET["NeighborAvg"]) for x in nvals)
                mark = ""
                if dc < 0.02 and dn < 0.03:
                    mark = "  <== BOTH MATCH"
                    hits.append((sname, fname, cname, v, nvals))
                print(f"  {sname:12s}{fname:16s}{cname:26s}{v:>10.4f}{dc:>8.3f}"
                      f"   {nvals[0]:.4f} / {nvals[1]:.4f}{mark}")
        print()

    print("=" * 100)
    if hits:
        print("  DEFINITIONS REPRODUCING BOTH PUBLISHED NUMBERS:")
        for sname, fname, cname, v, nv in hits:
            print(f"    space={sname}  form={fname}  prediction={cname}"
                  f"  -> CIFM {v:.4f}, NeighborAvg {nv[0]:.4f}/{nv[1]:.4f}")
    else:
        print("  NO combination reproduces both. The nearest CIFM cells are:")
        allc = sorted(((abs(ff(sf(truth), sf(P)) - TARGET["CIFM"]), sn, fn, cn,
                        ff(sf(truth), sf(P)))
                       for sn, sf in SPACES for fn, ff in FORMS
                       for cn, P in CANDS))[:6]
        for d, sn, fn, cn, v in allc:
            print(f"    {v:.4f} (|d|={d:.3f})  space={sn} form={fn} pred={cn}")
        print("\n  Note which candidates get close: if only the sparsity-matched")
        print("  ones (topk-matched / p>prev) reach ~0.144, then their reported MSE")
        print("  implies a prediction about as sparse as the truth, and the "
              "released\n  gate at p>0.5 does not produce that on this data "
              "(0.22 nonzero vs 0.025).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
