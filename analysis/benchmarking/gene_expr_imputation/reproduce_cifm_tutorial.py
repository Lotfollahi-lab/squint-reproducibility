#!/usr/bin/env python
"""
reproduce_cifm_tutorial.py — reproduce the ONLY reproducible outputs in CIFM's
own tutorial, to prove weights + preprocessing + our call path are correct.
=============================================================================
CIFM's `test.ipynb` (https://huggingface.co/ynyou/CIFM/blob/main/test.ipynb)
contains NO accuracy metrics — no ground truth, no Pearson, no benchmark table.
It only demonstrates the API. So "reproducing the tutorial" means matching the
two outputs it actually recorded:

  1. `channel_matching` report:  "matching 18289 gene channels out of 18289 ;
     unmatched channels: []"
  2. The `model.embed(adata)` tensor — the notebook saved the corner values of
     the embedding matrix. Reproducing those to ~4 decimals proves the
     checkpoint, the normalisation (normalize_total 1e4 + log1p), the
     channel matching and the graph construction are ALL correct, because any
     error in any of them changes these numbers.

The notebook's third output, `predict_cells_at_locations`, used
`np.random.rand(10, 2)` with NO seed, so its printed values are irreproducible
by construction. We instead check its shape and that its statistics are sane.

REFERENCE VALUES, transcribed from the notebook's stored cell output (cell 8):
    row 0 : [-0.4326, -0.8625,  0.1121, ...,  0.4980,  0.3855, -0.1965]
    row 1 : [-0.6833, -0.9950,  0.1927, ..., -0.2064,  0.6193,  0.0387]
    row 2 : [-0.2099, -0.9877,  0.3462, ...,  0.2102,  0.6807, -0.2155]
    ...
    row -3: [-0.0187, -0.8444,  0.3058, ...,  0.1030,  0.8362, -0.1859]
    row -2: [-0.5535, -0.8201,  0.7805, ..., -0.1402,  0.5221, -0.3520]

Usage
-----
  python reproduce_cifm_tutorial.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"

# (row index, first three values, last three values) from the notebook output
REF = [
    (0,  [-0.4326, -0.8625, 0.1121], [0.4980, 0.3855, -0.1965]),
    (1,  [-0.6833, -0.9950, 0.1927], [-0.2064, 0.6193, 0.0387]),
    (2,  [-0.2099, -0.9877, 0.3462], [0.2102, 0.6807, -0.2155]),
    (-3, [-0.0187, -0.8444, 0.3058], [0.1030, 0.8362, -0.1859]),
    (-2, [-0.5535, -0.8201, 0.7805], [-0.1402, 0.5221, -0.3520]),
]
TOL = 5e-3          # the notebook printed 4 decimals


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reproduce CIFM's tutorial outputs.")
    p.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    import scanpy as sc
    import torch

    repo = args.cifm_repo.resolve()
    ad_path = repo / "adata.h5ad"
    if not ad_path.is_file():
        raise SystemExit(f"{ad_path} missing — run: python download_cifm.py "
                         f"--include-demo --dest {repo}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402

    # ---- tutorial cell 2: load model ---------------------------------------
    model = CIFM.from_pretrained(
        str(repo) if (repo / "model.safetensors").is_file() else "ynyou/CIFM",
        args=torch.load(repo / "models_cifm" / "args.pt")).to(device)
    model.channel2ensembl_ids_source = torch.load(
        repo / "models_cifm" / "channel2ensembl.pt")
    model.eval()

    # ---- tutorial cell 4: load + preprocess --------------------------------
    adata = sc.read_h5ad(ad_path)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"adata: {adata.n_obs} x {adata.n_vars}   (tutorial: 24844 x 18289)")

    # ---- tutorial cell 6: channel matching (CHECK 1) -----------------------
    print("\n--- CHECK 1: channel_matching ---")
    tgt = [[i] for i in adata.var.index.tolist()]
    model.channel_matching(tgt, model.channel2ensembl_ids_source)
    print("  tutorial reported: matching 18289 gene channels out of 18289 ; "
          "unmatched channels: []")

    # ---- tutorial cell 8: embed (CHECK 2 — the decisive one) ---------------
    print("\n--- CHECK 2: model.embed(adata) vs the notebook's saved values ---")
    with torch.no_grad():
        emb = model.embed(adata).cpu().numpy()
    print(f"  shape {emb.shape}")
    ok = True
    for idx, head, tail in REF:
        got_h, got_t = emb[idx, :3], emb[idx, -3:]
        dh = np.abs(got_h - np.array(head)).max()
        dt = np.abs(got_t - np.array(tail)).max()
        good = (dh < TOL) and (dt < TOL)
        ok &= good
        print(f"  row {idx:>3}: got [{got_h[0]:+.4f} {got_h[1]:+.4f} {got_h[2]:+.4f}"
              f" ... {got_t[0]:+.4f} {got_t[1]:+.4f} {got_t[2]:+.4f}]  "
              f"max|diff|={max(dh, dt):.4f}  {'MATCH' if good else 'MISMATCH'}")
    print("\n  VERDICT: " + (
        "REPRODUCED — checkpoint, normalisation, channel matching and graph "
        "construction are all correct." if ok else
        "NOT reproduced. Since the notebook's numbers come from the same "
        "weights, a mismatch means our preprocessing or graph differs from "
        "theirs. Investigate before trusting ANY CIFM number."))

    # ---- tutorial cell 10: prediction (shape/sanity only) -----------------
    print("\n--- CHECK 3: predict_cells_at_locations (values irreproducible: "
          "the tutorial's locations are unseeded) ---")
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    rng = np.random.default_rng(0)
    locs = np.stack([rng.uniform(xy[:, 0].min(), xy[:, 0].max(), 10),
                     rng.uniform(xy[:, 1].min(), xy[:, 1].max(), 10)], axis=1)
    with torch.no_grad():
        out = model.predict_cells_at_locations(adata, locs).cpu().numpy()
    print(f"  shape {out.shape} (tutorial: (10, 18289))  finite={np.isfinite(out).all()}")
    print(f"  nonzero fraction {float((out > 0).mean()):.4f}  "
          f"(tutorial's printed rows are also mostly zeros — the dropout gate)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
