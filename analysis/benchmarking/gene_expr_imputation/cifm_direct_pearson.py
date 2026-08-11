#!/usr/bin/env python
"""
cifm_direct_pearson.py — the tutorial's cell-10 call, at REAL cell locations,
scored directly against the log-normalised expression. No transforms.
=============================================================================
Tutorial cell 10 predicts at RANDOM locations, so it has no ground truth. This
does the identical call but with the true coordinates of real cells, and
compares the returned tensor DIRECTLY to `adata.X` — which after the tutorial's
own cell 4 is `log1p(normalize_total(1e4))`, i.e. the log-normalised expression.

Nothing is done to the prediction: no expm1, no unit-profile renormalisation, no
read-depth rescaling. Prediction and target are both in the space the model
consumes, which is the most likely space for its output (its self-supervised
objective reconstructs a masked cell from that same representation).

Two variants, because the tutorial call passes the FULL adata as context:

  A) LEAKY (literally the tutorial): context = the whole adata, so each target
     cell's OWN expression is in the context. Not a valid benchmark, but a
     crucial upper bound — if the model cannot reproduce a cell it can see, the
     output space or our call is wrong.
  B) HELD-OUT: the target cells are removed from the context first. This is the
     honest measurement.

Controls in both: the train-mean profile (a constant) and a 16-NN average, in
the same space.

Usage
-----
  python cifm_direct_pearson.py [--n 500]
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    p.add_argument("--n", type=int, default=500, help="how many real cells")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import scanpy as sc
    import torch
    from sklearn.neighbors import NearestNeighbors

    repo = args.cifm_repo.resolve()
    sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- exactly the tutorial's setup --------------------------------------
    model = CIFM.from_pretrained(
        str(repo), args=torch.load(repo / "models_cifm" / "args.pt")).to(dev)
    model.channel2ensembl_ids_source = torch.load(
        repo / "models_cifm" / "channel2ensembl.pt")
    model.eval()

    adata = sc.read_h5ad(repo / "adata.h5ad")
    sc.pp.normalize_total(adata, target_sum=1e4)     # tutorial cell 4
    sc.pp.log1p(adata)
    model.channel_matching([[i] for i in adata.var.index.tolist()],
                           model.channel2ensembl_ids_source)   # cell 6

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(adata.n_obs, size=min(args.n, adata.n_obs // 4), replace=False)
    keep = np.setdiff1d(np.arange(adata.n_obs), sel)
    truth = X[sel]                                   # log1p(1e4-normalised)

    print(f"adata {adata.n_obs} x {adata.n_vars} | {len(sel)} target cells")
    print(f"truth: log-normalised expression, row sums (log space) median "
          f"{np.median(truth.sum(1)):.1f}, nonzero frac {float((truth>0).mean()):.4f}")

    # ---- controls, same space ---------------------------------------------
    const = np.repeat(X[keep].mean(0, keepdims=True), len(sel), axis=0)
    _, idx = NearestNeighbors(n_neighbors=16).fit(xy[keep]).kneighbors(xy[sel])
    knn = X[keep][idx].mean(1)

    for label, ctx_idx in (("A  LEAKY (tutorial call, full adata as context)", None),
                           ("B  HELD-OUT (targets removed from context)", keep)):
        ctx = adata if ctx_idx is None else adata[ctx_idx].copy()
        with torch.no_grad():
            pred = model.predict_cells_at_locations(ctx, xy[sel]).cpu().numpy()
        print("\n" + "=" * 78 + f"\n{label}\n" + "=" * 78)
        print(f"  prediction: shape {pred.shape}, nonzero frac "
              f"{float((pred>0).mean()):.4f}, row-sum median "
              f"{np.median(pred.sum(1)):.1f} (truth {np.median(truth.sum(1)):.1f})")
        print(f"  {'method':26s}{'cell-wise r':>13s}{'gene-wise r':>13s}")
        for nm, P in (("CIFM (raw output)", pred),
                      ("CONSTANT (train mean)", const),
                      ("16-NN average", knn)):
            print(f"  {nm:26s}{pearson(truth,P,1):>13.4f}{pearson(truth,P,0):>13.4f}")

    print("\nInterpretation:")
    print("  A is an upper bound — the model can see each target's own expression.")
    print("  If CIFM is weak even in A, the output space or our call is wrong.")
    print("  If A is strong but B is weak, the model needs context it does not")
    print("  have, which is the receptive-field story, not an implementation bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
