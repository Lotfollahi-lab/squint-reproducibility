#!/usr/bin/env python
"""
cifm_channel_integrity.py — does `channel_matching` do what run_cifm.py assumes?
=============================================================================
run_cifm.py assumes that after `model.channel_matching(target, source)`, column
i of the decoder's output corresponds to `target[i]`. That assumption has NEVER
been tested. What we did test — reproducing `model.embed(adata)` to 4 decimals —
only exercises the INPUT side, and only for the demo data whose 18,289 genes may
simply BE in vocabulary order. Under identity ordering, a broken permutation is
indistinguishable from a correct one.

If the output columns were misaligned, we would expect exactly the symptoms we
see on our 431-gene panel: near-chance sparsity AUROC (0.57 vs 0.877 on the demo
data), low Pearson, and — the tell — predictions that get WORSE as the model is
given more context, because a sharper prediction attributed to the wrong genes is
more wrong than a flat one. Three of our findings would collapse into one bug.

So test it directly, on CIFM's own data where ground truth is known.

  0. SOURCE. Print `channel_matching`, `predict_cells_at_locations` and
     `encode_decode` verbatim. Should have been step one.
  1. PERMUTATION. Shuffle the gene order, re-match, predict, un-shuffle. If
     target order is honoured this must reproduce the unshuffled prediction to
     floating point. This is the decisive test and it is exact — no thresholds.
  2. SUBSET. Restrict the demo data to ~431 genes and re-measure the two
     diagnostics that scored 0.877 / 0.813 on the full panel. If they collapse
     here, on CIFM's OWN data, then subsetting is the cause and our mouse panel
     is a red herring.
  3. UNMAPPED. Our real panel passes `[]` for genes with no human ortholog.
     Check what the model emits in those columns, and whether their presence
     perturbs the MATCHED columns (it must not).

Usage
-----
  python cifm_channel_integrity.py [--n 300] [--panel-size 431]
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"


def pearson_rows(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    na = np.sqrt((a**2).sum(1)); nb = np.sqrt((b**2).sum(1))
    ok = (na > 0) & (nb > 0)
    return float(np.nanmean((a[ok]*b[ok]).sum(1)/(na[ok]*nb[ok]))) if ok.any() else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    ap.add_argument("--n", type=int, default=300, help="query cells")
    ap.add_argument("--panel-size", type=int, default=431)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import scanpy as sc
    import torch
    from sklearn.metrics import roc_auc_score
    from torch_geometric.nn import radius_graph

    repo = a.cifm_repo.resolve(); sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args_pt = torch.load(repo / "models_cifm" / "args.pt")
    src = torch.load(repo / "models_cifm" / "channel2ensembl.pt")

    def fresh(target):
        """A clean model with `target` matched — channel_matching MUTATES state."""
        m = CIFM.from_pretrained(str(repo), args=args_pt).to(dev)
        m.channel2ensembl_ids_source = src
        m.eval()
        m.channel_matching(target, src)
        return m

    # ---- 0. SOURCE ----------------------------------------------------------
    print("=" * 78 + "\n0. SOURCE OF THE FUNCTIONS WE RELY ON\n" + "=" * 78)
    probe = CIFM.from_pretrained(str(repo), args=args_pt)
    for name in ("channel_matching", "predict_cells_at_locations",
                 "encode_decode"):
        fn = getattr(probe, name, None)
        print(f"\n--- {name} " + "-" * (60 - len(name)))
        if fn is None:
            print("  ABSENT from this checkpoint's class")
            continue
        try:
            print(inspect.getsource(fn))
        except (OSError, TypeError) as ex:
            print(f"  source unavailable: {ex}")
    del probe

    base = sc.read_h5ad(repo / "adata.h5ad")
    genes = base.var.index.astype(str).to_numpy()
    xy = np.asarray(base.obsm["spatial"], float)[:, :2]
    print(f"\ndemo data: {base.n_obs} x {base.n_vars}")
    # Is the demo var order identical to the vocabulary order? If yes, the
    # tutorial could never have exercised the permutation path.
    src_flat = [s[0] if isinstance(s, (list, tuple)) and s else s for s in src]
    same_prefix = sum(1 for i, g in enumerate(genes[:len(src_flat)])
                      if str(src_flat[i]) == g)
    print(f"demo var order == vocabulary order for {same_prefix}/"
          f"{min(len(genes), len(src_flat))} leading positions "
          f"-> tutorial {'CANNOT' if same_prefix > 0.9*min(len(genes),len(src_flat)) else 'does'} "
          f"test the permutation path")

    rng = np.random.default_rng(a.seed)
    sel = rng.choice(base.n_obs, size=min(a.n, base.n_obs // 8), replace=False)
    keep = np.setdiff1d(np.arange(base.n_obs), sel)

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
        return m, p

    def norm_log(ad):
        w = ad.copy()
        sc.pp.normalize_total(w, target_sum=1e4); sc.pp.log1p(w)
        return (w.X.toarray() if hasattr(w.X, "toarray") else np.asarray(w.X))

    # ---- 1. PERMUTATION -----------------------------------------------------
    print("\n" + "=" * 78 + "\n1. PERMUTATION TEST (decisive, exact)\n" + "=" * 78)
    X0 = norm_log(base)
    m0, p0 = predict(fresh([[g] for g in genes]), X0[keep], xy[keep], xy[sel])

    perm = rng.permutation(len(genes))
    gp = genes[perm]
    m1, p1 = predict(fresh([[g] for g in gp]), X0[keep][:, perm], xy[keep], xy[sel])
    m1_al = np.empty_like(m1); m1_al[:, perm] = m1
    p1_al = np.empty_like(p1); p1_al[:, perm] = p1

    dm = float(np.abs(m0 - m1_al).max()); dp = float(np.abs(p0 - p1_al).max())
    print(f"  shapes: baseline {m0.shape}, permuted {m1.shape}")
    print(f"  max|magnitude diff| after un-permuting = {dm:.6g}")
    print(f"  max|sparsity  diff| after un-permuting = {dp:.6g}")
    # scale reference: how big is a typical value?
    print(f"  (magnitude scale: mean {m0.mean():.3f}, max {m0.max():.3f})")
    tol = 1e-3
    if dm < tol and dp < tol:
        print("  VERDICT: PASS — target gene order IS honoured on the output.")
    else:
        print("  VERDICT: *** FAIL *** — output columns do NOT follow target order.")
        print("           run_cifm.py's gene assignment is WRONG and every CIFM")
        print("           number we have produced is invalid. Fix before anything else.")
        # is it a pure permutation of columns? then we can find the right map
        print(f"  Is baseline a column-permutation of the permuted run? "
              f"corr(sorted rows) = "
              f"{pearson_rows(np.sort(m0,1), np.sort(m1,1)):.6f}")

    # ---- 2. SUBSET ----------------------------------------------------------
    print("\n" + "=" * 78 +
          f"\n2. SUBSET TO {a.panel_size} GENES, on CIFM's OWN data\n" + "=" * 78)
    truth_full = X0[sel]
    y_full = (truth_full > 0).ravel().astype(int)
    print(f"  FULL 18289: AUROC(p) {roc_auc_score(y_full, p0.ravel()):.4f}   "
          f"entrywise r on expressed "
          f"{np.corrcoef(truth_full[truth_full>0], m0[truth_full>0])[0,1]:.4f}")

    sub = np.sort(rng.choice(len(genes), size=min(a.panel_size, len(genes)),
                             replace=False))
    ad_s = base[:, sub].copy()
    Xs = norm_log(ad_s)
    ms, ps = predict(fresh([[g] for g in genes[sub]]), Xs[keep], xy[keep], xy[sel])
    ts = Xs[sel]
    ys = (ts > 0).ravel().astype(int)
    print(f"  SUBSET {len(sub):5d}: AUROC(p) {roc_auc_score(ys, ps.ravel()):.4f}   "
          f"entrywise r on expressed "
          f"{np.corrcoef(ts[ts>0], ms[ts>0])[0,1]:.4f}")
    print(f"  input log1p magnitude: full mean {X0[keep].mean():.3f} "
          f"vs subset mean {Xs[keep].mean():.3f} "
          f"(1e4 over {len(sub)} genes instead of {len(genes)})")
    print("\n  If AUROC collapses toward ~0.57 HERE, subsetting alone explains our\n"
          "  mouse-panel result and the ortholog map is not the culprit.")

    # ---- 3. UNMAPPED ENTRIES ------------------------------------------------
    print("\n" + "=" * 78 + "\n3. UNMAPPED ENTRIES (our panel passes [])\n" + "=" * 78)
    tgt = [[g] for g in genes[sub]]
    n_blank = max(1, len(tgt) // 5)
    blanks = rng.choice(len(tgt), size=n_blank, replace=False)
    tgt_b = [([] if i in set(blanks.tolist()) else t) for i, t in enumerate(tgt)]
    try:
        mb, pb = predict(fresh(tgt_b), Xs[keep], xy[keep], xy[sel])
        okcols = np.setdiff1d(np.arange(len(tgt)), blanks)
        print(f"  blanked {n_blank}/{len(tgt)} target entries")
        print(f"  blanked columns: magnitude mean {mb[:, blanks].mean():.4f}, "
              f"max {mb[:, blanks].max():.4f}, "
              f"all-zero = {bool(np.all(mb[:, blanks] == 0))}")
        print(f"  MATCHED columns perturbed by the blanks? "
              f"max|diff| = {float(np.abs(mb[:, okcols] - ms[:, okcols]).max()):.6g}")
        print("  (should be ~0: unmatched channels must not change matched ones.\n"
              "   A large value means unmapped genes CONTAMINATE the rest, and\n"
              "   run_cifm.py should drop them before scoring.)")
    except Exception as ex:  # noqa: BLE001
        print(f"  channel_matching rejected empty entries: {type(ex).__name__}: {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
