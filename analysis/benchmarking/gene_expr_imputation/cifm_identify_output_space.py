#!/usr/bin/env python
"""
cifm_identify_output_space.py — WHAT does CIFM's expression head actually emit?
=============================================================================
The paper says only "normalize gene counts and conduct log1p-transformation"
(no target_sum; scanpy's default is the MEDIAN library size, while the tutorial
uses 1e4) and never writes out the loss. So the output space is undocumented.
Empirically it is clearly not log1p(1e4-normalised): on the demo data the
prediction's row sum in that space is ~11.8k against a truth of ~1.36k, and it
calls 18.8% of entries nonzero against a true 2.4%.

So: try every plausible transform and see which one makes prediction and truth
agree. The winner identifies the space. Also test the DROPOUT head separately —
CIFM is a two-head zero-inflated decoder (`relu(mask_cell_expression)` for
magnitude, `sigmoid(mask_cell_dropout)` for the zero pattern), so the magnitude
head is a CONDITIONAL magnitude, not a marginal mean. If the dropout head
predicts the zero pattern well while the magnitude head is miscalibrated, that
localises the problem precisely.

Diagnostics
-----------
  1. calibration: row sums of every candidate representation vs truth's
  2. Pearson for each candidate transform (cell-wise and gene-wise)
  3. dropout head AUROC for predicting (truth > 0) — is the model informative
     about WHICH genes are expressed, independent of magnitude?
  4. magnitude head on the truly-expressed entries only (truth > 0), which
     removes the zero-pattern confound entirely

Usage
-----
  python cifm_identify_output_space.py [--n 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"


def pearson(a, b, axis):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if axis == 0:
        a, b = a.T, b.T
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    na = np.sqrt((a ** 2).sum(1)); nb = np.sqrt((b ** 2).sum(1))
    ok = (na > 0) & (nb > 0)
    return float(np.nanmean((a[ok]*b[ok]).sum(1)/(na[ok]*nb[ok]))) if ok.any() else float("nan")


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
    from sklearn.metrics import roc_auc_score
    from torch_geometric.nn import radius_graph

    repo = a.cifm_repo.resolve(); sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFM.from_pretrained(str(repo),
        args=torch.load(repo/"models_cifm"/"args.pt")).to(dev)
    model.channel2ensembl_ids_source = torch.load(repo/"models_cifm"/"channel2ensembl.pt")
    model.eval()

    adata = sc.read_h5ad(repo/"adata.h5ad")
    raw = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    model.channel_matching([[i] for i in adata.var.index.tolist()],
                           model.channel2ensembl_ids_source)

    X   = adata.X.toarray() if hasattr(adata.X,"toarray") else np.asarray(adata.X)
    Craw= raw.toarray()     if hasattr(raw,"toarray")     else np.asarray(raw)
    xy  = np.asarray(adata.obsm["spatial"], float)[:, :2]
    rng = np.random.default_rng(a.seed)
    sel = rng.choice(adata.n_obs, size=min(a.n, adata.n_obs//4), replace=False)
    keep= np.setdiff1d(np.arange(adata.n_obs), sel)
    ctx = adata[keep].copy()

    # ---- get BOTH heads, ungated -------------------------------------------
    Xc = ctx.X.toarray() if hasattr(ctx.X,"toarray") else np.asarray(ctx.X)
    n_ctx, G = Xc.shape; n_q = len(sel)
    with torch.no_grad():
        e = torch.tensor(Xc, dtype=torch.float32, device=dev)
        e = torch.cat([e, torch.zeros(n_q, G, device=dev)], 0)
        c = torch.tensor(np.concatenate([np.asarray(ctx.obsm["spatial"])[:, :2],
                                         xy[sel]], 0), dtype=torch.float32)
        c = torch.cat([c, torch.zeros(c.shape[0],1)],1).to(dev)
        ei = radius_graph(c, r=model.radius_spatial_graph, max_num_neighbors=10000, loop=True)
        mp = torch.arange(n_ctx, n_ctx+n_q, device=dev)
        emb = model.encode(e, c, ei)
        emb[mp] = model.mask_embedding(torch.zeros(1, dtype=torch.int64, device=dev))
        dec = model.mask_cell_decoder(emb, c, ei)[0][mp]
        mag  = model.relu(model.mask_cell_expression(dec)).cpu().numpy()
        pdrop= model.sigmoid(model.mask_cell_dropout(dec)).cpu().numpy()

    truth_log = X[sel]                      # log1p(1e4-normalised)
    truth_lin = np.expm1(truth_log)         # 1e4-normalised counts
    truth_cnt = Craw[sel]                   # raw counts

    print("=" * 80 + "\n1. CALIBRATION (row sums, median over cells)\n" + "=" * 80)
    for nm, V in (("truth log1p(1e4-norm)", truth_log), ("truth 1e4-norm (linear)", truth_lin),
                  ("truth raw counts", truth_cnt), ("magnitude head (as-is)", mag),
                  ("expm1(magnitude head)", np.expm1(mag))):
        print(f"  {nm:26s} {np.median(V.sum(1)):14.1f}   nonzero frac "
              f"{float((V>0).mean()):.4f}")

    print("\n" + "=" * 80 + "\n2. WHICH TRANSFORM MAKES THEM AGREE?\n" + "=" * 80)
    print(f"  {'prediction repr':34s}{'target':24s}{'cell-wise':>11s}{'gene-wise':>11s}")
    cands = [
        ("mag  (as-is)",                 mag,               "log1p(1e4-norm)", truth_log),
        ("log1p(mag)",                   np.log1p(mag),     "log1p(1e4-norm)", truth_log),
        ("mag  (as-is)",                 mag,               "1e4-norm linear", truth_lin),
        ("expm1(mag)",                   np.expm1(mag),     "1e4-norm linear", truth_lin),
        ("unit(expm1(mag))",             unit(np.expm1(mag)),"unit(1e4-norm)",  unit(truth_lin)),
        ("unit(mag)",                    unit(mag),         "unit(1e4-norm)",  unit(truth_lin)),
        ("log1p(unit(expm1 mag)*1e4)",   np.log1p(unit(np.expm1(mag))*1e4), "log1p(1e4-norm)", truth_log),
        ("mag*(1-p_drop)",               mag*(1-pdrop),     "log1p(1e4-norm)", truth_log),
    ]
    best=None
    for nm,P,tn,T in cands:
        cw,gw = pearson(T,P,1), pearson(T,P,0)
        print(f"  {nm:34s}{tn:24s}{cw:>11.4f}{gw:>11.4f}")
        if best is None or cw>best[1]: best=(nm,cw,tn)
    print(f"\n  BEST cell-wise: {best[0]} vs {best[2]}  (r={best[1]:.4f})")

    print("\n" + "=" * 80 + "\n3. DROPOUT HEAD: does it know WHICH genes are expressed?\n" + "=" * 80)
    y = (truth_log > 0).ravel().astype(int)
    for nm, s in (("1 - p_dropout", (1-pdrop).ravel()), ("magnitude head", mag.ravel())):
        try:
            print(f"  AUROC[{nm:15s}] = {roc_auc_score(y, s):.4f}")
        except Exception as ex:  # noqa: BLE001
            print(f"  AUROC[{nm}] failed: {ex}")
    print(f"  (true nonzero rate {y.mean():.4f}; gate would keep "
          f"{float((pdrop<=0.5).mean()):.4f})")

    print("\n" + "=" * 80 + "\n4. MAGNITUDE ONLY, ON TRULY-EXPRESSED ENTRIES\n" + "=" * 80)
    m = truth_log > 0
    tv, pv = truth_log[m], mag[m]
    r = np.corrcoef(tv, pv)[0,1]
    print(f"  entrywise Pearson on the {m.sum()} expressed entries: {r:.4f}")
    print(f"  truth mean {tv.mean():.3f}  pred mean {pv.mean():.3f}  "
          f"ratio {pv.mean()/max(1e-9,tv.mean()):.3f}")
    print("\n  If the dropout head has good AUROC but no transform aligns the")
    print("  magnitudes, the head emits a CONDITIONAL magnitude on an")
    print("  uncalibrated scale, and Pearson against sparse observed counts is")
    print("  simply the wrong way to score it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
