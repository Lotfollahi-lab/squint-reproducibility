#!/usr/bin/env python
"""
cifm_panel_ablation.py — does CIFM degrade because of the PANEL, or did the
earlier test confound itself?
=============================================================================
`cifm_channel_integrity.py` reported magnitude r falling 0.811 -> 0.104 when the
demo data was cut to 431 genes, and I concluded "panel size is the root cause".
THAT CONCLUSION IS RETRACTED. The test had four confounds:

  C1 DIFFERENT TARGETS. normalize_total(1e4) was applied AFTER subsetting, so
     full-panel truth spread 1e4 over 18,289 genes and subset truth over 431
     (mean log1p 0.069 vs 0.153). Prediction and truth were compared in two
     different spaces. Worse, the metric used -- pooled ENTRYWISE Pearson -- is
     demonstrably sensitive to per-cell rescaling, i.e. to exactly that
     confound. So the headline number measured the confound at least in part.
  C2 RANDOM GENES != A REAL PANEL. A uniform draw from 18,289 is dominated by
     lowly-expressed genes; real panels (ours, Xenium's) are curated for
     informative, well-expressed markers. Composition, not count, may do the work.
  C3 NO BASELINES ON THE SAME GENES. Without CONSTANT / 16-NN on the identical
     gene set and target, r=0.10 is uninterpretable -- it could be a
     dynamic-range ceiling that every method hits.
  C4 EMPTY CELLS. Subsetting to random genes zeroed some cells outright
     ("Some cells have zero counts"), injecting degenerate context and targets
     that do not exist in our real 431-gene panel.

HOW THIS SCRIPT FIXES THEM
--------------------------
* PRIMARY METRIC IS CELL-WISE PEARSON over the panel's genes -- computed per
  cell then averaged, hence EXACTLY invariant to any per-cell rescaling of
  either vector, which neutralises C1. AUROC is taken from the sparsity head
  `p`, a probability in [0,1] that no normalisation touches, so it is
  comparable across input conditions too. (Pooled entrywise r is still printed,
  labelled, so the retracted number can be seen next to a sound one.)
* THE INPUT-COVERAGE EFFECT IS ISOLATED. Each panel is scored twice: CIFM fed
  ONLY the panel's genes (what run_cifm.py does) and CIFM fed ALL 18,289 with
  the prediction then restricted to the SAME columns. Same truth, same cells,
  same genes scored -- only input coverage differs, so any gap is that effect.
* THREE SAME-SIZE PANELS (random / HVG / top-expressed) separate composition
  from count, plus the ACTUAL realised ortholog panel via --panel-csv.
* CONSTANT and 16-NN controls on every panel against the same truth (C3).
* ONE COMMON CELL SET across all panels: cells must be non-empty in EVERY panel,
  and the query/context split is drawn ONCE before the loop. So cross-panel
  numbers are like-for-like and independent of panel order (C4 + comparability).
* float32 throughout, and the full-input prediction -- which does not depend on
  the panel -- is computed ONCE and merely re-indexed.

READING THE OUTPUT
------------------
  coverage delta ~ 0      -> panel size is NOT the problem
  coverage delta << 0     -> feeding only the panel genuinely costs CIFM
  CIFM full-input also below CONSTANT/16-NN -> not a panel effect at all
  low truth dynamic range -> a ceiling every method shares; read with care

Usage
-----
  python cifm_panel_ablation.py [--panel-csv .../ortholog_mapping.csv] [--n 300]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"


def cellwise_pearson(T, P):
    """Mean per-cell Pearson across genes. Invariant to per-cell rescaling."""
    T = np.asarray(T, np.float64); P = np.asarray(P, np.float64)
    T = T - T.mean(1, keepdims=True); P = P - P.mean(1, keepdims=True)
    nt = np.sqrt((T ** 2).sum(1)); npd = np.sqrt((P ** 2).sum(1))
    ok = (nt > 0) & (npd > 0)
    if not ok.any():
        return float("nan")
    return float(np.nanmean((T[ok] * P[ok]).sum(1) / (nt[ok] * npd[ok])))


def genewise_pearson(T, P):
    return cellwise_pearson(np.asarray(T).T, np.asarray(P).T)


def _rank_rows(A):
    """Average-rank transform per row (ties averaged), vectorised."""
    A = np.asarray(A, np.float64)
    n, g = A.shape
    order = np.argsort(A, axis=1, kind="stable")
    ranks = np.empty_like(A)
    np.put_along_axis(ranks, order,
                      np.broadcast_to(np.arange(1.0, g + 1.0), (n, g)), axis=1)
    # average ties so the transform is a true Spearman rank
    for i in range(n):
        v = A[i][order[i]]
        r = ranks[i][order[i]]
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


def cellwise_spearman(T, P):
    """
    Mean per-cell Spearman. Invariant to ANY monotone per-cell transform, so it
    removes the last confound in the coverage comparison: full-input predictions
    live on the whole-transcriptome scale while truth is panel-normalised, and
    log1p under two different normalisations is not a per-cell affine map (which
    cell-wise Pearson would require). If Pearson and Spearman agree on the sign
    and rough size of the coverage effect, that effect is real.
    """
    return cellwise_pearson(_rank_rows(T), _rank_rows(P))


def entrywise_pooled(T, P, mask=None):
    """The RETRACTED metric, kept only for side-by-side comparison."""
    t = np.asarray(T, np.float64); p = np.asarray(P, np.float64)
    if mask is not None:
        t, p = t[mask], p[mask]
    else:
        t, p = t.ravel(), p.ravel()
    if t.size < 2 or t.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(t, p)[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    ap.add_argument("--panel-csv", type=Path, default=None,
                    help="ortholog_mapping.csv from a real run_cifm.py run; its "
                         "mapped human ENSG IDs become the 'real ortholog' panel.")
    ap.add_argument("--panel-size", type=int, default=431)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import pandas as pd
    import scanpy as sc
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.nn import radius_graph

    repo = a.cifm_repo.resolve(); sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args_pt = torch.load(repo / "models_cifm" / "args.pt")
    src = torch.load(repo / "models_cifm" / "channel2ensembl.pt")

    base = sc.read_h5ad(repo / "adata.h5ad")
    genes = base.var.index.astype(str).to_numpy()
    xy_all = np.asarray(base.obsm["spatial"], np.float32)[:, :2]
    counts_all = (base.X.toarray() if hasattr(base.X, "toarray")
                  else np.asarray(base.X)).astype(np.float32)
    G_full = len(genes)
    print(f"demo data: {base.n_obs} cells x {G_full} genes  "
          f"({counts_all.nbytes/1e9:.2f} GB float32)")

    def fresh(target_genes):
        m = CIFM.from_pretrained(str(repo), args=args_pt).to(dev)
        m.channel2ensembl_ids_source = src
        m.eval()
        m.channel_matching([[g] for g in target_genes], src)
        return m

    def norm_log(C):
        """log1p(normalize_total(1e4)) row-wise, float32."""
        rs = C.sum(1, keepdims=True, dtype=np.float64)
        rs = np.where(rs > 0, rs, 1.0)
        return np.log1p(C / rs * 1e4).astype(np.float32)

    def predict(model, X_ctx, xy_ctx, q_xy):
        n_ctx, G = X_ctx.shape; n_q = len(q_xy)
        with torch.no_grad():
            e = torch.tensor(X_ctx, dtype=torch.float32, device=dev)
            e = torch.cat([e, torch.zeros(n_q, G, device=dev)], 0)
            c = torch.tensor(np.concatenate([xy_ctx, q_xy], 0), dtype=torch.float32)
            c = torch.cat([c, torch.zeros(c.shape[0], 1)], 1).to(dev)
            ei = radius_graph(c, r=model.radius_spatial_graph,
                              max_num_neighbors=10000, loop=True)
            mp = torch.arange(n_ctx, n_ctx + n_q, device=dev)
            emb = model.encode(e, c, ei)
            emb[mp] = model.mask_embedding(
                torch.zeros(1, dtype=torch.int64, device=dev))
            dec = model.mask_cell_decoder(emb, c, ei)[0][mp]
            m = model.relu(model.mask_cell_expression(dec)).cpu().numpy()
            p = model.sigmoid(model.mask_cell_dropout(dec)).cpu().numpy()
        del e, c, ei, emb, dec
        if dev == "cuda":
            torch.cuda.empty_cache()
        return m.astype(np.float32), p.astype(np.float32)

    # ---- panels: same SIZE, different COMPOSITION ---------------------------
    rng = np.random.default_rng(a.seed)
    K = min(a.panel_size, G_full)
    mean_expr = counts_all.mean(0)
    h = base.copy()
    sc.pp.normalize_total(h, target_sum=1e4); sc.pp.log1p(h)
    sc.pp.highly_variable_genes(h, n_top_genes=K)
    hvg = np.where(h.var["highly_variable"].to_numpy())[0]
    del h

    panels = {
        "random":        np.sort(rng.choice(G_full, size=K, replace=False)),
        "HVG":           np.sort(hvg),
        "top-expressed": np.sort(np.argsort(-mean_expr)[:K]),
    }
    if a.panel_csv is not None:
        mp_df = pd.read_csv(a.panel_csv)
        used = mp_df[mp_df["used_by_cifm"].astype(bool)]["human_ensembl_id"].astype(str)
        pos = {g: i for i, g in enumerate(genes)}
        idx = np.array(sorted(pos[g] for g in used.tolist() if g in pos))
        if idx.size:
            panels["real ortholog"] = idx
            print(f"real ortholog panel: {len(used)} mapped IDs, {idx.size} of them "
                  f"present in the demo data")
        else:
            print("real ortholog panel: NONE of the mapped IDs are in the demo data")
    print("panel sizes: " + ", ".join(f"{k}={v.size}" for k, v in panels.items()))

    # ---- ONE common cell set + ONE split, drawn before the loop -------------
    alive = np.ones(base.n_obs, bool)
    for pn, pidx in panels.items():
        a_p = counts_all[:, pidx].sum(1) > 0
        print(f"  cells empty in panel {pn!r}: {int((~a_p).sum())}")
        alive &= a_p
    live = np.where(alive)[0]
    print(f"  cells non-empty in EVERY panel: {live.size}/{base.n_obs} "
          f"(dropped {base.n_obs - live.size})")
    if live.size < a.n * 4:
        raise SystemExit("too few commonly-usable cells")
    sel = live[rng.choice(live.size, size=min(a.n, live.size // 8), replace=False)]
    keep = np.setdiff1d(live, sel)
    print(f"  query cells {sel.size}, context cells {keep.size} "
          f"(identical for every panel)")

    # ---- full-input prediction: panel-independent, so compute ONCE ----------
    print("\ncomputing the FULL-input prediction once (18,289 channels)...")
    Xf = norm_log(counts_all)
    mF_all, pF_all = predict(fresh(genes), Xf[keep], xy_all[keep], xy_all[sel])
    del Xf
    print(f"  done: {mF_all.shape}")

    results = {}
    for pname, pidx in panels.items():
        print("\n" + "=" * 78 + f"\nPANEL: {pname}  ({pidx.size} genes)\n" + "=" * 78)
        panel_counts = counts_all[:, pidx]
        Xp = norm_log(panel_counts)

        truth = Xp[sel]                      # ONE truth, used by every method
        expressed = truth > 0
        y = expressed.ravel().astype(np.int8)
        print(f"  truth: nonzero frac {expressed.mean():.4f}, "
              f"std of expressed entries {truth[expressed].std():.4f}  "
              f"<- dynamic range; a low value caps EVERY method")

        row = {}
        mP, pP = predict(fresh(genes[pidx]), Xp[keep], xy_all[keep], xy_all[sel])
        assert mP.shape == truth.shape, (mP.shape, truth.shape)
        mF, pF = mF_all[:, pidx], pF_all[:, pidx]

        for label, (mm, pp) in (("CIFM panel-input", (mP, pP)),
                                ("CIFM full-input", (mF, pF))):
            pred = mm * pp
            row[label] = (cellwise_pearson(truth, pred),
                          genewise_pearson(truth, pred),
                          roc_auc_score(y, pp.ravel()),
                          entrywise_pooled(truth, mm, expressed),
                          cellwise_spearman(truth, pred))

        const = np.repeat(Xp[keep].mean(0, keepdims=True), sel.size, axis=0)
        _, nb = NearestNeighbors(n_neighbors=16).fit(
            xy_all[keep]).kneighbors(xy_all[sel])
        knn = Xp[keep][nb].mean(1)
        for label, pred in (("CONSTANT (train mean)", const), ("16-NN spatial", knn)):
            row[label] = (cellwise_pearson(truth, pred),
                          genewise_pearson(truth, pred),
                          roc_auc_score(y, pred.ravel()),
                          entrywise_pooled(truth, pred, expressed),
                          cellwise_spearman(truth, pred))

        print(f"\n  {'method':26s}{'cell-wise':>11s}{'spearman':>10s}"
              f"{'gene-wise':>11s}{'AUROC':>9s}{'entry(retr)':>13s}")
        for k, v in row.items():
            print(f"  {k:26s}{v[0]:>11.4f}{v[4]:>10.4f}{v[1]:>11.4f}"
                  f"{v[2]:>9.4f}{v[3]:>13.4f}")
        d = row["CIFM panel-input"][0] - row["CIFM full-input"][0]
        ds = row["CIFM panel-input"][4] - row["CIFM full-input"][4]
        print(f"  COVERAGE EFFECT (panel-input − full-input): "
              f"cell-wise {d:+.4f}   spearman {ds:+.4f}")
        print("    Spearman is the confound-free version; trust it if they differ.")
        results[pname] = row

    print("\n" + "=" * 78 + "\nSUMMARY — cell-wise Pearson (same cells, same truth)\n"
          + "=" * 78)
    print(f"  {'panel':16s}{'CIFM panel':>12s}{'CIFM full':>11s}"
          f"{'CONSTANT':>10s}{'16-NN':>9s}{'covΔ r':>10s}{'covΔ ρ':>10s}")
    for pn, row in results.items():
        print(f"  {pn:16s}{row['CIFM panel-input'][0]:>12.4f}"
              f"{row['CIFM full-input'][0]:>11.4f}"
              f"{row['CONSTANT (train mean)'][0]:>10.4f}"
              f"{row['16-NN spatial'][0]:>9.4f}"
              f"{row['CIFM panel-input'][0]-row['CIFM full-input'][0]:>+10.4f}"
              f"{row['CIFM panel-input'][4]-row['CIFM full-input'][4]:>+10.4f}")
    print("\n  ACROSS panels of equal size -> composition vs count.")
    print("  WITHIN a panel vs controls -> model failure vs dynamic-range ceiling.")
    print("  'coverage' column is the ONLY number that isolates panel size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
