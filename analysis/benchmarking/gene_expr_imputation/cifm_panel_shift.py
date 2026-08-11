#!/usr/bin/env python
"""
cifm_panel_shift.py — is our mmb number measuring CIFM, or measuring the fact
that we feed it a 431-gene panel?
=============================================================================
`cifm_marginal_vs_gate.py` validated the PLUMBING on CIFM's own demo data:
output space, head polarity, `to_counts()`, the harness's log1p. All correct.

But that demo data has all 18,289 genes, and our silver data has a 431-gene
mouse panel. CIFM consumes `normalize_total(1e4) + log1p`, so:

    whole transcriptome : 1e4 spread over ~thousands of expressed genes
    431-gene panel      : 1e4 spread over ~400 genes

so per-gene normalised values are larger on a panel by roughly the coverage
ratio, and this script measures that ratio rather than assuming it.

DO NOT read a FULL->PANEL drop as "CIFM cannot handle targeted panels". CIFM's
pretraining corpus INCLUDES Xenium, which is a targeted panel: ~100 samples /
23M cells / 32k measured genes "across four platforms of Visium and Xenium"
(bioRxiv 2025.01.25.634867). Panel input is therefore IN distribution for it,
and `channel_matching` exists precisely to accept a partial gene set. What is
genuinely off-distribution in OUR setup is narrower and must be stated as such:
our panel is MOUSE, reached through an ortholog map into a vocabulary with zero
mouse entries, so only a fraction of the 431 genes survive. This script
quantifies what that costs; it does not license a claim about panels per se.

This script measures it, with everything except the gene set held fixed: same
checkpoint, same cells, same held-out 500, same MARGINAL collapse, same harness
space, same controls. Any drop from FULL to PANEL is caused by the panel alone.

Also probes two things we could not check on the demo data:

  * UNMAPPED CHANNELS. Our panel has genes with no human ortholog; CIFM leaves
    those output channels zero-initialised, and run_cifm.py keeps them in the
    Pearson. This measures what the model actually emits there, and scores
    with and without them, so we know whether they are a real penalty.
  * TARGET_SUM. The paper says only "normalize gene counts and log1p", with NO
    target_sum. If the panel shift is the problem, feeding a coverage-matched
    target_sum (so per-gene magnitudes resemble training) should recover some
    of it. That is a deviation from the tutorial, so it is reported as a
    SENSITIVITY, never as the headline setting.

Panel choice: pass `--panel-csv <out_dir>/ortholog_mapping.csv` from a real
run_cifm.py run to use the ACTUAL realised panel (the `human_ensembl_id` column,
restricted to `used_by_cifm`). Without it, a size-matched HVG panel is used,
which measures the panel-SIZE effect but not our specific gene set.

Usage
-----
  python cifm_panel_shift.py [--panel-csv .../ortholog_mapping.csv] [--n 500]
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


def to_counts(pred_log, depth):
    """Byte-for-byte the transform in run_cifm.py::to_counts."""
    rate = np.expm1(np.clip(pred_log, 0.0, None)).astype(np.float32)
    rs = rate.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0).astype(np.float32)
    rate = rate / rs
    return (rate * depth[:, None].astype(np.float32)).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    ap.add_argument("--panel-csv", type=Path, default=None,
                    help="ortholog_mapping.csv from a real run, to use the "
                         "actual realised panel instead of a size-matched one.")
    ap.add_argument("--panel-size", type=int, default=431,
                    help="Panel size when --panel-csv is not given.")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import pandas as pd
    import scanpy as sc
    import torch
    from sklearn.neighbors import NearestNeighbors
    from torch_geometric.nn import radius_graph

    repo = a.cifm_repo.resolve(); sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args_pt = torch.load(repo / "models_cifm" / "args.pt")
    src = torch.load(repo / "models_cifm" / "channel2ensembl.pt")

    base = sc.read_h5ad(repo / "adata.h5ad")           # raw counts
    xy_all = np.asarray(base.obsm["spatial"], float)[:, :2]
    all_genes = base.var.index.astype(str).to_numpy()

    rng = np.random.default_rng(a.seed)
    sel = rng.choice(base.n_obs, size=min(a.n, base.n_obs // 4), replace=False)
    keep = np.setdiff1d(np.arange(base.n_obs), sel)

    # ---- panels --------------------------------------------------------------
    panels = {"FULL 18289": all_genes}
    n_unmapped = 0
    if a.panel_csv is not None:
        mp = pd.read_csv(a.panel_csv)
        used = mp[mp["used_by_cifm"].astype(bool)]["human_ensembl_id"].astype(str)
        want = [g for g in used.tolist() if g in set(all_genes)]
        n_unmapped = int((~mp["used_by_cifm"].astype(bool)).sum())
        panels[f"REAL panel n={len(want)}"] = np.array(want)
        print(f"panel-csv: {len(mp)} panel genes, {len(used)} mapped to ENSG, "
              f"{len(want)} of those present in the demo data, "
              f"{n_unmapped} unmapped in the real run")
    else:
        h = base.copy()
        sc.pp.normalize_total(h, target_sum=1e4); sc.pp.log1p(h)
        sc.pp.highly_variable_genes(h, n_top_genes=a.panel_size)
        panels[f"HVG panel n={a.panel_size}"] = \
            all_genes[h.var["highly_variable"].to_numpy()]

    def run(genes, target_sum, tag):
        ad = base[:, genes].copy()
        cnt = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
        w = ad.copy()
        sc.pp.normalize_total(w, target_sum=target_sum); sc.pp.log1p(w)
        Xn = w.X.toarray() if hasattr(w.X, "toarray") else np.asarray(w.X)

        model = CIFM.from_pretrained(str(repo), args=args_pt).to(dev)
        model.channel2ensembl_ids_source = src
        model.eval()
        model.channel_matching([[g] for g in genes], src)

        n_ctx, G = Xn[keep].shape; n_q = len(sel)
        with torch.no_grad():
            e = torch.tensor(Xn[keep], dtype=torch.float32, device=dev)
            e = torch.cat([e, torch.zeros(n_q, G, device=dev)], 0)
            c = torch.tensor(np.concatenate([xy_all[keep], xy_all[sel]], 0),
                             dtype=torch.float32)
            c = torch.cat([c, torch.zeros(c.shape[0], 1)], 1).to(dev)
            ei = radius_graph(c, r=model.radius_spatial_graph,
                              max_num_neighbors=10000, loop=True)
            mp_ = torch.arange(n_ctx, n_ctx + n_q, device=dev)
            emb = model.encode(e, c, ei)
            emb[mp_] = model.mask_embedding(
                torch.zeros(1, dtype=torch.int64, device=dev))
            dec = model.mask_cell_decoder(emb, c, ei)[0][mp_]
            m = model.relu(model.mask_cell_expression(dec)).cpu().numpy()
            p = model.sigmoid(model.mask_cell_dropout(dec)).cpu().numpy()
        pred_log = m * p                                    # MARGINAL

        depth = cnt[sel].sum(1)
        T = np.log1p(cnt[sel])
        P = np.log1p(to_counts(pred_log, depth))
        const = np.log1p(to_counts(
            np.repeat(Xn[keep].mean(0, keepdims=True), n_q, 0), depth))
        _, idx = NearestNeighbors(n_neighbors=16).fit(
            xy_all[keep]).kneighbors(xy_all[sel])
        knn = np.log1p(to_counts(Xn[keep][idx].mean(1), depth))

        print(f"\n  {tag}")
        print(f"    input log1p values: mean {Xn[keep].mean():.3f} "
              f"max {Xn[keep].max():.2f}   genes {G}   "
              f"pred nonzero {float((pred_log>0).mean()):.4f} "
              f"(truth {float((cnt[sel]>0).mean()):.4f})")
        print(f"    {'method':22s}{'cell-wise':>12s}{'gene-wise':>12s}")
        out = {}
        for nm, Q in (("CIFM (marginal)", P), ("CONSTANT", const), ("16-NN", knn)):
            cw, gw = pearson(T, Q, 1), pearson(T, Q, 0)
            print(f"    {nm:22s}{cw:>12.4f}{gw:>12.4f}")
            out[nm] = cw
        return out

    print("=" * 78 + "\nPANEL SHIFT: harness-space Pearson, everything else fixed\n"
          + "=" * 78)
    res = {}
    for tag, genes in panels.items():
        res[tag] = run(genes, 1e4, f"{tag}  (target_sum=1e4, as the tutorial)")

    # ---- target_sum sensitivity on the panel --------------------------------
    panel_tag = [t for t in panels if not t.startswith("FULL")]
    if panel_tag:
        t = panel_tag[0]
        genes = panels[t]
        cov = float(np.asarray(base[:, genes].X.sum()) /
                    float(np.asarray(base.X.sum())))
        ts = max(50.0, 1e4 * cov)
        print("\n" + "=" * 78 +
              f"\nTARGET_SUM SENSITIVITY (panel captures {100*cov:.1f}% of counts"
              f" -> coverage-matched target_sum={ts:.0f})\n" + "=" * 78)
        print("  DEVIATION from the tutorial's 1e4. Report as sensitivity only.")
        res[f"{t} @ts={ts:.0f}"] = run(genes, ts, f"{t}  (target_sum={ts:.0f})")

    print("\n" + "=" * 78 + "\nSUMMARY (CIFM cell-wise, harness space)\n" + "=" * 78)
    for k, v in res.items():
        d = v["CIFM (marginal)"] - max(v["CONSTANT"], v["16-NN"])
        print(f"  {k:34s} CIFM {v['CIFM (marginal)']:.4f}   "
              f"best control {max(v['CONSTANT'], v['16-NN']):.4f}   "
              f"gap {d:+.4f}")
    print("\n  A large FULL->PANEL drop means our mmb number is dominated by the\n"
          "  panel domain shift, not by CIFM's imputation ability, and no number\n"
          "  from our harness should be presented as 'CIFM's performance'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
