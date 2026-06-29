#!/usr/bin/env python3
"""
Data-side sanity check for a squint_hln AnnData (silver file, gene-set silver,
or a trained run's predicted_adata). Surfaces the failure modes that make the
HLN metrics look bad but that the model/metrics scripts do NOT check:

  1. is X RAW COUNTS?  (normalized-as-counts breaks SQUINT's NB decoder)
  2. is the normalized matrix sitting in .raw / layers while X is normalized?
  3. are negative-control probes still present?  (Negative*/SystemControl*)
  4. LABEL COVERAGE — how many cells actually have a cell-type / niche label?
     (the released CosMx file is ~1.85M cells; the benchmark's annotated
     reference is only 19,718 — if you kept the full file most cells are
     unlabeled and niche NMI/ARI is computed against mostly-missing labels.)
  5. label cardinality (should be ~16 cell types, ~4-5 niches)
  6. spatial coords present.

Usage:
    python diagnose_squint_hln.py --adata /nfs/.../silver/squint_hln/cosmx_human_lymph_node.h5ad
    python diagnose_squint_hln.py --adata /nfs/.../<run>/predicted_adata.h5ad
    python diagnose_squint_hln.py --self-test
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

CONTROL_RE = re.compile(r"^(Negative|SystemControl|NegPrb|FalseCode|Blank)",
                        re.IGNORECASE)
MISSING_TOKENS = {"", "nan", "none", "na", "unassigned", "unknown", "unlabeled",
                  "unlabelled", "filtered", "removed"}
# candidate obs columns (first match wins) — mirrors fetch_cosmx_lymph_node.py
CELLTYPE_CANDIDATES = ["cell_type", "cell_type_annotation", "new_annotation",
                       "cellType", "celltype", "cell_types", "nb_clus", "annotation"]
NICHE_CANDIDATES = ["niche", "niche_annotation", "niche_type", "manual_niche",
                    "spatial_niche", "region", "spatial_cluster"]


def _dense_rows(X, n=2000):
    sub = X[:n] if X.shape[0] > n else X
    return sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)


def _counts_report(name, X):
    arr = _dense_rows(X)
    finite = arr[np.isfinite(arr)]
    is_int = bool(finite.size and np.allclose(finite, np.round(finite)))
    nonneg = bool(finite.size and finite.min() >= 0)
    libsize = np.asarray(X.sum(1)).ravel()
    looks_counts = is_int and nonneg
    print(f"  [{name}] shape={X.shape}  integer={is_int}  nonneg={nonneg}  "
          f"min={float(finite.min()) if finite.size else 'NA':.3g}  "
          f"max={float(finite.max()) if finite.size else 'NA':.4g}")
    print(f"  [{name}] per-cell total: median={np.median(libsize):.1f} "
          f"min={libsize.min():.3g} max={libsize.max():.4g}  "
          f"(raw CosMx totals are integers in the 100s-1000s+)")
    return looks_counts


def _resolve_key(adata, key, candidates):
    """Use `key` if present, else the first matching candidate column."""
    if key in adata.obs:
        return key
    for c in candidates:
        if c in adata.obs:
            return c
    return None


def _label_report(adata, key, candidates):
    resolved = _resolve_key(adata, key, candidates)
    if resolved is None:
        print(f"  LABEL '{key}': not found (tried {candidates}).")
        print(f"      available obs columns: {list(adata.obs.columns)}")
        print(f"      !! the blob's label_names must point at one of these, or "
              f"the blob is built UNLABELED and NMI/ARI are meaningless.")
        return
    if resolved != key:
        print(f"  LABEL '{key}': not found, but '{resolved}' is — USE THAT in the "
              f"blob config (label_names=...={resolved}).")
    key = resolved
    s = adata.obs[key].astype(str).str.strip()
    n = len(s)
    miss = s.str.lower().isin(MISSING_TOKENS) | adata.obs[key].isna().to_numpy()
    n_miss = int(miss.sum())
    valid = s[~miss]
    ncat = valid.nunique()
    print(f"  LABEL '{key}': {ncat} categories; "
          f"{n_miss}/{n} ({100*n_miss/max(1,n):.1f}%) missing/unassigned")
    vc = valid.value_counts().head(8)
    print(f"      top: {dict(vc)}")
    if n_miss > 0.2 * n:
        print(f"      !! WARNING: >20% of cells lack a '{key}' label — NMI/ARI on "
              f"this axis will be dominated by unlabeled cells. The benchmark uses "
              f"the 19,718-cell ANNOTATED ROI, not the full ~1.85M-cell file.")


def diagnose(adata, cell_key="cell_type_annotation", niche_key="niche_annotation"):
    print(f"\n=== squint_hln diagnosis: {adata.n_obs} cells x {adata.n_vars} genes ===")
    print(f"  obs columns ({len(adata.obs.columns)}): {list(adata.obs.columns)}")
    if adata.n_obs > 100_000:
        print(f"  !! NOTE: {adata.n_obs} cells >> the benchmark's 19,718-cell "
              f"annotated ROI — likely the FULL CosMx file (mostly unlabeled).")

    # X counts
    x_counts = _counts_report("X", adata.X)
    if not x_counts:
        print("  !! X is NOT raw integer counts (looks normalized). SQUINT's NB "
              "decoder expects RAW counts — rebuild from a counts X.")
        if getattr(adata, "raw", None) is not None:
            r_counts = _counts_report("raw", adata.raw.X)
            if r_counts:
                print("  -> adata.raw HOLDS raw counts. Re-run fetch (--raw-to-x) / "
                      "preprocess so X becomes counts, then REBUILD the blob.")
    if "norm" in getattr(adata, "layers", {}):
        print("  [layers] 'norm' present (normalized matrix stashed) — expected "
              "after the raw-counts fix.")

    # controls
    ctrl = [g for g in map(str, adata.var_names) if CONTROL_RE.match(g)]
    if ctrl:
        print(f"  !! {len(ctrl)} negative-control probes still present "
              f"(e.g. {ctrl[:5]}) — re-run fetch (--drop-controls) + rebuild.")
    else:
        print("  controls: none detected (good).")

    # labels (auto-resolve to the actual column if the given key is absent)
    _label_report(adata, cell_key, CELLTYPE_CANDIDATES)
    _label_report(adata, niche_key, NICHE_CANDIDATES)

    # spatial
    if "spatial" in adata.obsm:
        sp = np.asarray(adata.obsm["spatial"])
        print(f"  spatial: present {sp.shape}, x∈[{sp[:,0].min():.0f},"
              f"{sp[:,0].max():.0f}] y∈[{sp[:,1].min():.0f},{sp[:,1].max():.0f}]")
    else:
        print("  !! obsm['spatial'] MISSING — no graph can be built.")
    print("=== end ===\n")


def _self_test():
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 500
    counts = rng.poisson(5.0, size=(n, 12)).astype(np.float32)
    a = ad.AnnData(X=(counts / counts.sum(1, keepdims=True)).astype(np.float32))  # normalized X
    a.var_names = [f"G{i}" for i in range(10)] + ["Negative1", "SystemControl1"]
    raw = ad.AnnData(X=counts); raw.var_names = a.var_names
    a.raw = raw
    a.obsm["spatial"] = rng.normal(size=(n, 2))
    ct = rng.integers(0, 16, n).astype(str)
    ni = np.where(rng.random(n) < 0.5, "unassigned", rng.integers(0, 4, n).astype(str))
    a.obs["cell_type_annotation"] = pd.Categorical(ct)
    a.obs["niche_annotation"] = pd.Categorical(ni)
    diagnose(a)
    print("SELF-TEST: ran diagnosis on a synthetic normalized+control+50%-missing-niche AnnData")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adata")
    p.add_argument("--cell-key", default="cell_type_annotation")
    p.add_argument("--niche-key", default="niche_annotation")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.adata:
        p.error("pass --adata <path> (or --self-test)")
    import anndata as ad
    diagnose(ad.read_h5ad(args.adata), args.cell_key, args.niche_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
