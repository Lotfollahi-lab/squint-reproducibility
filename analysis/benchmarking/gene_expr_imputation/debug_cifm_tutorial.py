#!/usr/bin/env python
"""
debug_cifm_tutorial.py — does CIFM work AT ALL, on its OWN data?
=============================================================================
Our mmb imputation run put CIFM at chance (AUROC 0.559, gene-wise Pearson 0.05)
and — the real red flag — gave nearly IDENTICAL numbers with a radius graph
(~0.5 neighbours) and a k-NN graph (16 neighbours). Predictions that ignore the
graph mean the model is not doing spatial inference, so before reporting
anything we must localise the failure.

This script isolates the question: run CIFM exactly as its own `test.ipynb`
does, on its own bundled `adata.h5ad`, and measure whether its predictions carry
real signal. Requires the demo file:

    python download_cifm.py --include-demo

Four stages, each with a verdict:

  1. TUTORIAL REPLAY — load the model, channel_match on the demo's own gene IDs
     (should match 18289/18289), embed, and predict at random locations. Checks
     the plumbing and reports how VARIED the predictions are across locations.
     Near-zero variance across locations => the model is emitting a constant.
  2. GRAPH GEOMETRY — median NN distance and mean degree at r=20 on the demo
     data. If CIFM's own demo has ~0 neighbours at its own radius, then r=20 is
     not meaningful for that data either, which reframes our mmb result.
  3. ACCURACY ON HELD-OUT REAL CELLS — the test the tutorial does NOT do. Mask
     N real cells, predict at their true coordinates from the remaining cells,
     and score against ground truth (cell-wise / gene-wise Pearson on log1p,
     same definitions as our benchmark harness).
  4. CONTROLS — the same score for (a) a CONSTANT prediction (the train mean
     profile) and (b) a 16-NN spatial average. This is the decisive comparison:
        CIFM >> constant   -> model works; our mmb adaptation is at fault
        CIFM ~= constant   -> the model is contributing nothing here either,
                              so our expectation (or the metric) is wrong
     A model that cannot beat its own mean profile on its own data is not being
     driven by the neighbourhood.

Usage
-----
  python debug_cifm_tutorial.py                    # uses ../cifm/adata.h5ad
  python debug_cifm_tutorial.py --n-holdout 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"


def pearson(a: np.ndarray, b: np.ndarray, axis: int) -> float:
    """Mean Pearson r along `axis`, ignoring zero-variance rows/cols."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if axis == 0:                      # gene-wise: correlate across cells
        a, b = a.T, b.T
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    na = np.sqrt((a ** 2).sum(1))
    nb = np.sqrt((b ** 2).sum(1))
    ok = (na > 0) & (nb > 0)
    if not ok.any():
        return float("nan")
    r = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
    return float(np.nanmean(r))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Localise the CIFM failure.")
    p.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    p.add_argument("--adata", type=Path, default=None,
                   help="Defaults to <cifm-repo>/adata.h5ad (needs "
                        "download_cifm.py --include-demo).")
    p.add_argument("--n-holdout", type=int, default=500)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import scanpy as sc
    import torch
    from torch_geometric.nn import radius_graph

    repo = args.cifm_repo.resolve()
    ad_path = args.adata or (repo / "adata.h5ad")
    if not ad_path.is_file():
        raise SystemExit(
            f"{ad_path} not found. Fetch the demo data first:\n"
            f"  python download_cifm.py --include-demo --dest {repo}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402

    # ---- 1. tutorial replay -------------------------------------------------
    print("=" * 78 + "\n1. TUTORIAL REPLAY\n" + "=" * 78)
    model = CIFM.from_pretrained(
        str(repo) if (repo / "model.safetensors").is_file() else "ynyou/CIFM",
        args=torch.load(repo / "models_cifm" / "args.pt")).to(device)
    model.channel2ensembl_ids_source = torch.load(
        repo / "models_cifm" / "channel2ensembl.pt")
    model.eval()
    print(f"  model on {device}, radius_spatial_graph={model.radius_spatial_graph}")

    adata = sc.read_h5ad(ad_path)
    print(f"  demo adata: {adata.n_obs} cells x {adata.n_vars} genes")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)   # exactly the tutorial
    sc.pp.log1p(adata)

    tgt = [[i] for i in adata.var.index.tolist()]  # tutorial: var index == ENSG
    model.channel_matching(tgt, model.channel2ensembl_ids_source)

    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    rng = np.random.default_rng(args.seed)
    locs = np.stack([rng.uniform(xy[:, 0].min(), xy[:, 0].max(), 10),
                     rng.uniform(xy[:, 1].min(), xy[:, 1].max(), 10)], axis=1)
    with torch.no_grad():
        out = model.predict_cells_at_locations(adata, locs).cpu().numpy()
    across = float(np.mean(np.std(out, axis=0)))   # spread ACROSS locations
    within = float(np.mean(np.std(out, axis=1)))
    print(f"  predict_cells_at_locations -> {out.shape}, finite={np.isfinite(out).all()}")
    print(f"  nonzero fraction={float((out > 0).mean()):.4f}")
    print(f"  mean SD across locations (per gene) = {across:.5f}")
    print(f"  mean SD across genes (per location) = {within:.5f}")
    print("  VERDICT: " + ("predictions VARY by location — plumbing OK"
                           if across > 1e-4 else
                           "predictions are ~CONSTANT across locations (!!)"))

    # ---- 2. graph geometry --------------------------------------------------
    print("\n" + "=" * 78 + "\n2. GRAPH GEOMETRY ON CIFM'S OWN DEMO DATA\n" + "=" * 78)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    d, _ = nn.kneighbors(xy)
    med = float(np.median(d[:, 1]))
    sub = xy[: min(2000, len(xy))]
    deg = float(np.mean([len(x) - 1 for x in NearestNeighbors(
        radius=model.radius_spatial_graph).fit(xy).radius_neighbors(
        sub, return_distance=False)]))
    print(f"  median NN distance = {med:.2f}   span = "
          f"{xy[:,0].ptp():.0f} x {xy[:,1].ptp():.0f}")
    print(f"  mean degree @r={model.radius_spatial_graph}: {deg:.2f}")
    print("  VERDICT: " + ("demo data IS dense enough for r=20"
                           if deg >= 1.0 else
                           "even CIFM's OWN demo has ~no neighbours at r=20 (!!)"))

    # ---- 3/4. accuracy on held-out real cells, vs controls ------------------
    print("\n" + "=" * 78 +
          f"\n3-4. HELD-OUT ACCURACY ({args.n_holdout} real cells) + CONTROLS\n"
          + "=" * 78)
    n = adata.n_obs
    hold = rng.choice(n, size=min(args.n_holdout, n // 4), replace=False)
    keep = np.setdiff1d(np.arange(n), hold)
    ctx = adata[keep].copy()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    truth = X[hold]                                   # log1p space

    # CIFM's output is NOT log1p(1e4-normalised): the tutorial converts it with
    # expm1 and then DIVIDES BY THE ROW SUM, which would be unnecessary if the
    # magnitude were already 1e4-calibrated. So we must put predictions through
    # the SAME conversion run_cifm.py uses before comparing:
    #     expm1 -> unit profile -> x read depth -> log1p
    # and compare in log1p(counts) space, which is what the benchmark harness
    # scores. Comparing the raw model output against log1p(normalised) truth --
    # my first version of this script -- penalises CIFM for a scale/shape
    # mismatch while the dense controls are unaffected. That was a measurement
    # bug, not a model failure.
    counts = adata.layers["counts"]
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    truth_counts = counts[hold]
    depth = truth_counts.sum(1)          # same depth for EVERY method: isolates profile shape

    def to_counts(profile_like):
        r = np.clip(np.asarray(profile_like, dtype=np.float64), 0, None)
        rs = r.sum(1, keepdims=True); rs = np.where(rs > 0, rs, 1.0)
        return (r / rs) * depth[:, None]

    # gated (what the released API emits) and ungated (magnitude head only)
    def predict(gate: bool):
        import torch as T
        from torch_geometric.nn import radius_graph as rg
        n_ctx = ctx.n_obs; n_q = len(hold)
        Xc = ctx.X.toarray() if hasattr(ctx.X, "toarray") else np.asarray(ctx.X)
        with T.no_grad():
            e = T.tensor(Xc, dtype=T.float32, device=device)
            e = T.cat([e, T.zeros(n_q, e.shape[1], device=device)], 0)
            c = T.tensor(np.concatenate([np.asarray(ctx.obsm["spatial"])[:, :2],
                                         xy[hold]], 0), dtype=T.float32)
            c = T.cat([c, T.zeros(c.shape[0], 1)], 1).to(device)
            ei = rg(c, r=model.radius_spatial_graph, max_num_neighbors=10000, loop=True)
            mp = T.arange(n_ctx, n_ctx + n_q, device=device)
            emb = model.encode(e, c, ei)
            emb[mp] = model.mask_embedding(T.zeros(1, dtype=T.int64, device=device))
            dec = model.mask_cell_decoder(emb, c, ei)[0][mp]
            ex = model.relu(model.mask_cell_expression(dec))
            if gate:
                dr = model.sigmoid(model.mask_cell_dropout(dec))
                ex = ex.clone(); ex[dr <= 0.5] = 0
            return ex.cpu().numpy()

    pred_g, pred_u = predict(True), predict(False)
    const_profile = np.repeat(np.expm1(X[keep]).mean(0, keepdims=True), len(hold), axis=0)
    nn16 = NearestNeighbors(n_neighbors=16).fit(xy[keep])
    _, idx = nn16.kneighbors(xy[hold])
    knn_profile = counts[keep][idx].mean(1)

    tl = np.log1p(truth_counts)
    print(f"  comparison space: log1p(counts), identical depth for all methods")
    print(f"  gate zeroes {100*(1-(pred_g>0).mean()):.1f}% of predicted entries\n")
    print(f"  {'method':26s}{'cell-wise r':>13s}{'gene-wise r':>13s}")
    res = {}
    for name, prof in (("CIFM (gated, native)", np.expm1(pred_g)),
                       ("CIFM (ungated)", np.expm1(pred_u)),
                       ("CONSTANT (train mean)", const_profile),
                       ("16-NN average", knn_profile)):
        P = np.log1p(to_counts(prof))
        cw, gw = pearson(tl, P, 1), pearson(tl, P, 0)
        res[name] = (cw, gw)
        print(f"  {name:26s}{cw:>13.4f}{gw:>13.4f}")
    res["CIFM"] = max(res["CIFM (gated, native)"], res["CIFM (ungated)"])
    c_cw, c_gw = res["CIFM"]
    k_cw, k_gw = res["CONSTANT (train mean)"]
    print("\n  VERDICT:")
    if not np.isfinite(c_cw):
        print("    CIFM produced degenerate predictions (nan correlation).")
    elif c_cw > k_cw + 0.05:
        print("    CIFM CLEARLY BEATS its own mean profile on its own data")
        print("    -> the model works; the fault is in our mmb ADAPTATION")
        print("       (prime suspects: ortholog channel_matching, or the")
        print("        coordinate scale / graph we feed it).")
    else:
        print("    CIFM does NOT beat a constant prediction even on its OWN data")
        print("    -> our expectation or measurement is wrong, not just the")
        print("       mmb adaptation. Do NOT report the mmb numbers.")
    print(f"    (16-NN reference: cell-wise {k_cw:.4f} vs CIFM {c_cw:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
