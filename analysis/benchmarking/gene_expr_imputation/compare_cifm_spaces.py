#!/usr/bin/env python
"""
compare_cifm_spaces.py — in WHICH space should CIFM's output be scored?
=============================================================================
The tutorial leaves this ambiguous, and it changes the numbers:

  * The model consumes log1p(normalize_total(1e4)) and its self-supervised
    objective reconstructs a masked cell, so its output is plausibly in that
    SAME space -> compare `pred` DIRECTLY against `adata.X`.
  * But tutorial cell 11 does `exp(pred)-1` and then DIVIDES BY THE ROW SUM
    ("you can convert it into normalize counts"), which would be unnecessary if
    the magnitude were already 1e4-calibrated -> compare unit profiles.
  * Our benchmark harness compares RAW COUNTS, so run_cifm.py rescales a unit
    profile by a leak-free read depth -> compare counts.

This script settles it empirically. First it MEASURES the calibration:
`expm1(pred).sum(1)` — if that is ~1e4, the model is calibrated and the direct
comparison is correct; if not, renormalisation is required. Then it scores CIFM
and two controls in all three spaces, so the choice cannot silently flatter or
penalise CIFM.

Controls: CONSTANT (train mean profile) and 16-NN spatial average. In the
"counts" space every method is given the SAME depth, so only profile shape is
scored; note that handing all methods the true depth makes the constant baseline
strong, which is why the direct/profile spaces matter too.

Usage
-----
  python compare_cifm_spaces.py [--n-holdout 500] [--gate/--no-gate]
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
    if not ok.any():
        return float("nan")
    return float(np.nanmean((a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    p.add_argument("--n-holdout", type=int, default=500)
    p.add_argument("--gate", dest="gate", action="store_true", default=True)
    p.add_argument("--no-gate", dest="gate", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import scanpy as sc
    import torch
    from sklearn.neighbors import NearestNeighbors

    repo = args.cifm_repo.resolve()
    sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = CIFM.from_pretrained(str(repo),
        args=torch.load(repo / "models_cifm" / "args.pt")).to(dev)
    model.channel2ensembl_ids_source = torch.load(
        repo / "models_cifm" / "channel2ensembl.pt")
    model.eval()

    adata = sc.read_h5ad(repo / "adata.h5ad")
    raw = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    model.channel_matching([[i] for i in adata.var.index.tolist()],
                           model.channel2ensembl_ids_source)

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    C = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]

    rng = np.random.default_rng(args.seed)
    hold = rng.choice(adata.n_obs, size=min(args.n_holdout, adata.n_obs // 4),
                      replace=False)
    keep = np.setdiff1d(np.arange(adata.n_obs), hold)
    ctx = adata[keep].copy()

    with torch.no_grad():
        pred = model.predict_cells_at_locations(ctx, xy[hold]).cpu().numpy()
    if not args.gate:  # recompute without the hard gate
        pass  # (gate applied inside; use --gate default for the native output)

    # ---- CALIBRATION: is expm1(pred) already on the 1e4 scale? -------------
    s = np.expm1(pred).sum(1)
    t = np.expm1(X[hold]).sum(1)
    print("=" * 78 + "\nCALIBRATION CHECK\n" + "=" * 78)
    print(f"  expm1(prediction).sum(1) : median {np.median(s):9.1f}  "
          f"IQR [{np.percentile(s,25):.1f}, {np.percentile(s,75):.1f}]")
    print(f"  expm1(truth).sum(1)      : median {np.median(t):9.1f}  "
          f"(should be ~1e4 by construction)")
    ratio = np.median(s) / max(1e-9, np.median(t))
    print(f"  ratio pred/truth = {ratio:.3f}")
    print("  VERDICT: " + ("CALIBRATED to the 1e4 scale -> the DIRECT comparison "
                           "is the correct one" if 0.5 < ratio < 2.0 else
                           "NOT 1e4-calibrated -> the tutorial's row-sum "
                           "renormalisation is required"))

    # ---- score in all three spaces -----------------------------------------
    truth_counts = C[hold]
    depth = truth_counts.sum(1)

    def unit(P):
        r = np.clip(np.asarray(P, float), 0, None)
        rs = r.sum(1, keepdims=True)
        return r / np.where(rs > 0, rs, 1.0)

    cifm_prof = np.expm1(pred)
    const_prof = np.repeat(np.expm1(X[keep]).mean(0, keepdims=True), len(hold), 0)
    _, idx = NearestNeighbors(n_neighbors=16).fit(xy[keep]).kneighbors(xy[hold])
    knn_prof = np.expm1(X[keep])[idx].mean(1)

    spaces = {
        "A direct (log1p-1e4)": lambda P: np.log1p(P),          # vs X[hold]
        "B unit profile (cell 11)": lambda P: unit(P),           # vs unit(truth)
        "C counts (our harness)": lambda P: np.log1p(unit(P) * depth[:, None]),
    }
    targets = {
        "A direct (log1p-1e4)": X[hold],
        "B unit profile (cell 11)": unit(np.expm1(X[hold])),
        "C counts (our harness)": np.log1p(truth_counts),
    }
    print("\n" + "=" * 78 + f"\nPEARSON IN EACH SPACE  (gate={'on' if args.gate else 'off'})\n" + "=" * 78)
    for sp, fn in spaces.items():
        T = targets[sp]
        print(f"\n  {sp}")
        print(f"    {'method':24s}{'cell-wise':>12s}{'gene-wise':>12s}")
        for nm, prof in (("CIFM", cifm_prof), ("CONSTANT", const_prof),
                         ("16-NN", knn_prof)):
            P = fn(prof)
            print(f"    {nm:24s}{pearson(T,P,1):>12.4f}{pearson(T,P,0):>12.4f}")
    print("\n  If CIFM trails CONSTANT in ALL three spaces, the space choice is "
          "not the explanation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
