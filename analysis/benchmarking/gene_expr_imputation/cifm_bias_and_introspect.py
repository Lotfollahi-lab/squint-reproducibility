#!/usr/bin/env python
"""
cifm_bias_and_introspect.py — does `channel_matching` throw away the heads' bias?
=============================================================================
THE HYPOTHESIS. `channel_matching` rebuilds three layers as `bias=False` and
copies only `.weight.data`:

    linear_out1 = nn.Linear(self.hidden_dim, len(target), bias=False)
    linear_out2 = nn.Linear(self.hidden_dim, len(target), bias=False)
    ...
    self.mask_cell_expression.layers[-1] = linear_out1
    self.mask_cell_dropout.layers[-1]    = linear_out2

If the ORIGINAL final layers carry a bias, it is silently discarded. For the
sparsity head that is exactly the failure we observe: dropping a negative bias
raises `sigmoid(...)` everywhere, so the gate keeps far too many genes. Measured
on the demo data, our gate keeps 22.1% of entries against a truth of 2.5% -- an
8.7x over-call at otherwise-correct per-gene magnitudes (3.29 vs 3.13) -- and that
is what makes our MSE 1.7682 against their published 0.144.

WHY THIS WOULD NOT SHOW UP ANYWHERE ELSE WE HAVE LOOKED
  * The authors' own Visium-HD evaluation would NOT call channel_matching: that
    data already uses the model's native vocabulary. We verified the demo var
    order equals the vocabulary order for all 18,289 positions, so skipping the
    call is legitimate here -- and it is the authors' own configuration.
  * `reproduce_cifm_tutorial.py` cannot catch it: the notebook's stored embedding
    values were themselves produced AFTER channel_matching, so both sides carry
    the same dropped bias.
  * `verify_native_equivalence` cannot catch it: both paths share one already
    channel-matched model.
  * Spearman cannot catch it: a constant per-gene shift inside the sigmoid changes
    WHICH genes pass the gate but the surviving magnitudes keep their order, so
    the rank metric barely moves -- consistent with us reproducing Spearman
    (0.2070 vs 0.212) while missing MSE by 12x.

WHAT THIS SCRIPT DOES
  1. INTROSPECT the checkpoint before any channel matching: dump every attribute
     of args.pt, the module tree of the two heads and the gene encoder, and
     report for each whether `bias is None` -- the direct test of the hypothesis.
     Also print the source of `embed`, `encode` and the head classes, and list
     every public method on CIFM in case there is an inference entry point we
     have not found.
  2. COMPARE, on the demo data, three configurations:
       A  native      : NO channel_matching at all (the authors' own setting for
                        this vocabulary)
       B  matched     : channel_matching with the identity target (what we do)
       C  matched+bias: channel_matching, then the original biases restored onto
                        the rebuilt layers
     Same split, same masking, same metric. If A (and C) collapse the nonzero
     fraction toward 0.025 and drop MSE toward 0.144, the discarded bias is the
     entire discrepancy.

Usage
-----
  python cifm_bias_and_introspect.py
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"


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
    ap.add_argument("--mask-frac", type=float, default=0.05)
    ap.add_argument("--max-eval", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import scanpy as sc
    import torch
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.nn import radius_graph

    repo = a.cifm_repo.resolve(); sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args_pt = torch.load(repo / "models_cifm" / "args.pt")
    src_ids = torch.load(repo / "models_cifm" / "channel2ensembl.pt")

    # ---- 1. INTROSPECTION, before any channel matching ---------------------
    print("=" * 84 + "\n1. CHECKPOINT INTROSPECTION (no channel_matching yet)\n"
          + "=" * 84)
    print("\n--- every attribute of args.pt ---")
    if hasattr(args_pt, "__dict__"):
        for kk, vv in sorted(vars(args_pt).items()):
            print(f"  {kk:34s} {vv!r}")
    else:
        print(f"  args.pt is a {type(args_pt)}: {args_pt!r}")

    m0 = CIFM.from_pretrained(str(repo), args=args_pt)
    print(f"\n--- files in models_cifm/ ---")
    for f in sorted((repo / "models_cifm").iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")

    print("\n--- public methods on CIFM ---")
    print("  " + ", ".join(sorted(
        n for n in dir(m0) if not n.startswith("_") and callable(getattr(m0, n, None))
    )))

    print("\n--- THE DECISIVE CHECK: do the rebuilt layers have a bias? ---")
    found = {}
    for name in ("gene_encoder", "mask_cell_expression", "mask_cell_dropout"):
        mod = getattr(m0, name, None)
        if mod is None:
            print(f"  {name}: ABSENT"); continue
        layers = getattr(mod, "layers", None)
        if layers is None:
            print(f"  {name}: has no .layers ({type(mod).__name__})"); continue
        idx = 0 if name == "gene_encoder" else len(layers) - 1
        lay = layers[idx]
        b = getattr(lay, "bias", None)
        found[name] = None if b is None else b.detach().clone()
        print(f"  {name}.layers[{idx}] = {type(lay).__name__} "
              f"{tuple(getattr(lay,'weight',torch.zeros(0)).shape)}  "
              f"bias = {'None' if b is None else f'present, shape {tuple(b.shape)}, mean {b.mean():.4f}, min {b.min():.4f}, max {b.max():.4f}'}")
    if found.get("mask_cell_dropout") is not None:
        b = found["mask_cell_dropout"]
        print(f"\n  *** mask_cell_dropout HAS a bias. channel_matching rebuilds "
              f"that layer with bias=False, so it is DISCARDED. ***")
        print(f"      sigmoid(bias): mean {torch.sigmoid(b).mean():.4f}, "
              f"frac>0.5 {float((torch.sigmoid(b)>0.5).float().mean()):.4f}")
        print(f"      Dropping a mostly-negative bias raises p and opens the gate.")
    elif "mask_cell_dropout" in found:
        print("\n  mask_cell_dropout has NO bias -> channel_matching discards "
              "nothing here, and this hypothesis is DEAD.")

    print("\n--- source of embed / encode (never read before) ---")
    for nm in ("embed", "encode"):
        fn = getattr(m0, nm, None)
        print(f"\n  === {nm} ===")
        try:
            print(inspect.getsource(fn))
        except Exception as ex:  # noqa: BLE001
            print(f"    unavailable: {ex}")

    del m0

    # ---- 2. THREE CONFIGURATIONS ------------------------------------------
    adata = sc.read_h5ad(repo / "adata.h5ad")
    genes = adata.var.index.astype(str).tolist()
    xy = np.asarray(adata.obsm["spatial"], np.float32)[:, :2]
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    X = (adata.X.toarray() if hasattr(adata.X, "toarray")
         else np.asarray(adata.X)).astype(np.float32)

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

    def build(mode):
        mm = CIFM.from_pretrained(str(repo), args=args_pt).to(dev)
        mm.channel2ensembl_ids_source = src_ids
        mm.eval()
        if mode == "native":
            return mm
        # stash the original biases before they are thrown away
        orig = {}
        for name in ("mask_cell_expression", "mask_cell_dropout"):
            lay = getattr(mm, name).layers[-1]
            b = getattr(lay, "bias", None)
            orig[name] = None if b is None else b.detach().clone()
        mm.channel_matching([[g] for g in genes], src_ids)
        if mode == "matched+bias":
            for name, b in orig.items():
                if b is None:
                    continue
                lay = getattr(mm, name).layers[-1]
                new = torch.nn.Linear(lay.in_features, lay.out_features,
                                      bias=True).to(dev)
                new.weight.data.copy_(lay.weight.data)
                # identity target => channel j maps to source channel j
                new.bias.data.copy_(b[:lay.out_features].to(dev))
                getattr(mm, name).layers[-1] = new
        return mm

    def run(mm):
        n_ctx, G = X[ctx].shape; n_q = q.size
        r = float(mm.radius_spatial_graph)
        with torch.no_grad():
            e = torch.tensor(X[ctx], dtype=torch.float32, device=dev)
            e = torch.cat([e, torch.zeros(n_q, G, device=dev)], 0)
            c = torch.tensor(np.concatenate([xy[ctx], xy[q]], 0),
                             dtype=torch.float32)
            c = torch.cat([c, torch.zeros(c.shape[0], 1)], 1).to(dev)
            ei = radius_graph(c, r=r, max_num_neighbors=10000, loop=True)
            mp = torch.arange(n_ctx, n_ctx + n_q, device=dev)
            emb = mm.encode(e, c, ei)
            emb[mp] = mm.mask_embedding(
                torch.zeros(1, dtype=torch.int64, device=dev))
            dec = mm.mask_cell_decoder(emb, c, ei)[0][mp]
            mag = mm.relu(mm.mask_cell_expression(dec))
            pr = mm.sigmoid(mm.mask_cell_dropout(dec))
            g = mag.clone(); g[pr <= 0.5] = 0.0
            out = g.cpu().numpy().astype(np.float64)
            prn = pr.cpu().numpy().astype(np.float64)
        del e, c, ei, emb, dec, mag, pr, g
        if dev == "cuda":
            torch.cuda.empty_cache()
        return out, prn

    print("\n" + "=" * 84)
    print("2. NATIVE vs MATCHED vs MATCHED+BIAS  (demo data, scattered "
          f"{a.mask_frac:.0%}, {q.size} masked cells)")
    print("=" * 84)
    print(f"  truth: nonzero {float((truth>0).mean()):.4f}, "
          f"row sum {np.median(truth.sum(1)):.1f}, "
          f"all-zeros MSE {float(np.mean(truth**2)):.4f}")
    print(f"\n  {'config':16s}{'nonzero':>9s}{'row sum':>10s}{'mean p':>9s}"
          f"{'p>0.5':>8s}{'Spearman':>10s}{'MSE':>9s}")
    for mode in ("native", "matched", "matched+bias"):
        try:
            mm = build(mode)
            pred, pr = run(mm)
            print(f"  {mode:16s}{float((pred>0).mean()):>9.4f}"
                  f"{np.median(pred.sum(1)):>10.1f}{pr.mean():>9.4f}"
                  f"{float((pr>0.5).mean()):>8.4f}"
                  f"{_corr(_rank_rows(truth), _rank_rows(pred)):>10.4f}"
                  f"{float(np.mean((truth-pred)**2)):>9.4f}")
            del mm
        except Exception as ex:  # noqa: BLE001
            print(f"  {mode:16s} FAILED: {type(ex).__name__}: {ex}")
    print("\n  targets: nonzero ~0.025, Spearman ~0.212, MSE ~0.144")
    print("  If 'native' matches those and 'matched' does not, channel_matching's")
    print("  bias=False rebuild is the whole story, and every number we have")
    print("  produced through it is affected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
