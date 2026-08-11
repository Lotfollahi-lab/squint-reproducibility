#!/usr/bin/env python
"""
!! NOTE any "0.144" / "0.205" below is a MISREAD of Fig. 2B's MSE panel.
   The real values are CIFM 0.266 and NeighborAvg 0.280, with large
   overlapping error bars. Corrected 2026-08-11.

cifm_dump_source.py — print the model's ENTIRE source. No inference, no metrics.
=============================================================================
We have been probing this model one method at a time and have twice been wrong
about what it does. The whole package is 12.4 KB:

    cifm.py                  6,525 bytes
    mlp_and_gnn.py           3,264 bytes
    egnn_void_invariant.py   2,644 bytes

so there is no reason not to read all of it.

WHAT WE HAVE READ SO FAR: channel_matching, predict_cells_at_locations,
encode_decode, embed, encode.

WHAT WE HAVE NEVER READ, and why it matters for reproducing their MSE:
  * `forward`  — the TRAINING entry point. The paper's loss (Appdx B.3, Eq. 15)
    is a balanced MSE on `X_dec` and contains NO dropout/zero-inflation term, so
    the quantity their reported "mismatch error" is computed on is most likely
    whatever `forward` produces -- which need not be `encode_decode`'s GATED
    output. Our gated output scores MSE 1.7682 against their published 0.144;
    the ungated magnitude head scores 8.5340. If `forward` produces a third
    thing, that is the number to reproduce.
  * `proj`  — a module on the class that nothing we have read references. If the
    output passes through a projection we are skipping, it would rescale
    everything.
  * `mask_cell_decoder` / `gene_encoder` / `model` internals — whether the
    encoder's message passing genuinely nullifies zero-expression ("void") nodes,
    which is what makes the released full-graph `encode_decode` equivalent to the
    paper's `f_enc(X_unm, C_unm, A(C_unm))` with masked nodes REMOVED (Eq. 2).
  * whether any normalisation, scaling or clamping happens inside the model that
    we are duplicating or missing outside it.

Confirmed non-issues, so nobody re-checks them: args.pt holds only
{hidden_dim 1024, in_dim 18289, num_layer 2, num_mlp_layers_in_module 4,
radius_spatial_graph 20} -- no normalisation or output-scale setting; and all
three layers channel_matching rebuilds have `bias = None`, so its bias=False
rebuild discards nothing (verified: native / matched / matched+bias are
bit-identical at nonzero 0.2211, MSE 1.7682).

Usage
-----
  python cifm_dump_source.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"
FILES = ["cifm.py", "mlp_and_gnn.py", "egnn_void_invariant.py"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    a = ap.parse_args(argv)
    mdir = a.cifm_repo.resolve() / "models_cifm"

    print(f"source root: {mdir}")
    print("\n--- everything under models_cifm/ ---")
    for p in sorted(mdir.rglob("*")):
        if "__pycache__" in str(p):
            continue
        rel = p.relative_to(mdir)
        print(f"  {'DIR ' if p.is_dir() else '    '}{rel}"
              f"{'' if p.is_dir() else f'  ({p.stat().st_size} bytes)'}")

    for name in FILES:
        f = mdir / name
        print("\n" + "=" * 78)
        print(f"### {name}")
        print("=" * 78)
        if not f.is_file():
            print(f"  MISSING: {f}")
            continue
        txt = f.read_text()
        for i, line in enumerate(txt.split("\n"), 1):
            print(f"{i:4d}| {line}")

    # anything else python-ish that we did not list explicitly
    extra = [p for p in sorted(mdir.rglob("*.py"))
             if p.name not in FILES and "__pycache__" not in str(p)]
    for f in extra:
        print("\n" + "=" * 78)
        print(f"### {f.relative_to(mdir)}   (not in the expected file list)")
        print("=" * 78)
        for i, line in enumerate(f.read_text().split("\n"), 1):
            print(f"{i:4d}| {line}")

    print("\n" + "=" * 78)
    print("READ FOR: `forward` (what quantity the Eq. 15 loss is computed on),")
    print("`proj` (an unreferenced module), whether zero-expression nodes are")
    print("truly nullified in message passing, and any internal normalisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
