#!/usr/bin/env python
"""
compute_label_conditioned_metrics.py — the scIB label-conditioned scores R2 asked for.
=============================================================================
R2-W1b: "MMD ... does not by itself establish preservation of cell-type-specific
biology; label-conditioned integration metrics would be preferable."

Computed on the SAVED latents, no retraining:
  kbet   scib_metrics.kbet_per_label — batch mixing WITHIN each cell type. This is
         the label-conditioned integration metric R2 asked for.
  clisi  cell-type LISI. Bio-conservation mirror of the iLISI in Table 1.
  casw   cell-type silhouette. A second bio-conservation view.
  ilisi  scib_metrics.ilisi_knn. This IS the iLISI Table 1 reports; recomputed as a
         REPRODUCTION GATE, not as a new result.

WHY THIS CALLS scib_metrics DIRECTLY INSTEAD OF compute_benchmarking_metrics
That function cannot compute these in the current environment. It passes raw sparse
obsp matrices to the LISI family:

    scib_metrics.clisi_knn(X=adata.obsp[f"{latent_key}_90knng_distances"], ...)

but scib-metrics 0.5.6 requires `X: NeighborsResults(indices, distances)`. Observed
on the mouse-brain file: kbet raises ValueError ("Length of batches does not match
number of cells"), while clisi and blisi sit inside a bare `except:` that assigns
0.0, so they fail SILENTLY and return a plausible zero. casw is unaffected, since
silhouette_label takes the embedding directly.

One implication, about the published numbers rather than this script: Table 1's iLISI
(niche 0.609, cell 0.739) cannot be reproduced through that path in this venv, since
it would return the 0.0 sentinel. Either the venv's scib-metrics was upgraded after
those runs, or the numbers came from elsewhere. Worth settling before the
camera-ready, and it means the gate below can disagree with 0.609 for reasons that
have nothing to do with the representation.

benchmarking.py is deliberately NOT modified: it produced the published numbers. We
reuse only its graph builder, compute_knn_graph_connectivities_and_distances, so the
neighbour graph is exactly the paper's (same k, same random_state).

WHERE kbet CAN BE COMPUTED AT ALL (measured, see label_batch_feasibility.py)
scib-metrics skips any label confined to one batch (_kbet.py:151) and then averages
with np.nanmean, so an annotation that does not straddle batches yields NaN rather
than a low score.

  mouse brain  NOT COMPUTABLE. `cell_type` (49 classes) is the UNION of two
               per-section vocabularies -- 23 CL-ontology names in batch 82
               ("astrocyte", "microglial cell") and 26 descriptive names in batch 15
               ("Astrocytes", "Microglia") -- with ZERO overlap. Every label is
               single-batch, all 49 are skipped, kbet = NaN. The niche columns are
               worse: Sub_molecular_tissue_region (63) exists only in batch 15 and
               ccf_region_name (167) only in batch 82. This is a property of the
               annotations, not of this script or of the representation.
  eczema       COMPUTABLE. `new_annotation` (21 classes) is one shared vocabulary
               across the three patient sections; 18 classes appear in all three,
               20 in at least two, 100% of cells scorable. Use --cell-type-key
               new_annotation: the silver also carries a stray 40-class `cell_type`
               that the benchmark deliberately does not score.
  NSCLC        COMPUTABLE, with --drop-label-nan. All 10 cell types and all 12
               niches appear in both sections, but `cell_type` has 8,980 NaN.

SELF-LOOP CONVENTION
scib-metrics' own builders return k neighbours INCLUDING the query point (column 0 is
self at distance 0), so NeighborsResults is built that way here. scanpy-style graphs
exclude self, hence --include-self (default) prepends it. If the gate disagrees with
a published anchor, try --no-include-self before concluding the representation is
wrong: this convention shifts every LISI value.

USAGE
  # 1. look first; computes nothing
  python compute_label_conditioned_metrics.py --inspect --adata .../predicted_adata.h5ad

  # 2. identify which obsm key reproduces which published anchor (seed 0 is enough)
  python compute_label_conditioned_metrics.py --adata .../seed0/predicted_adata.h5ad \\
      --expect-ilisi neighborhood_latent=0.609,cell_latent=0.739 --out /tmp/ident.csv

  # 3. then the real run: five seeds, the two identified keys
  python compute_label_conditioned_metrics.py --method SQUINT \\
      --latent-keys neighborhood_latent,cell_latent \\
      --adata .../seed0/... --adata .../seed1/... --out .../lcm_SQUINT.csv

Needs the squint venv (the one that can import vqniche).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

METRICS = ["ilisi", "kbet", "clisi", "casw"]
K_LISI, K_KBET = 90, 50          # the k's benchmarking.py uses for these metrics
LABEL_HINTS = ("cell_type", "cell_types", "celltype", "annotation", "niche",
               "region", "domain", "leiden", "cluster")
BATCH_HINTS = ("batch", "section", "sample", "donor", "slide", "assay")


def inspect(path: Path) -> None:
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    print(f"\n=== {path} ===\n  n_obs={a.n_obs}  n_vars={a.n_vars}")
    print("\n  obsm keys (candidates for --latent-keys):")
    for k in a.obsm.keys():
        try:
            print(f"    {k:32s} {a.obsm[k].shape}")
        except Exception:                                          # noqa: BLE001
            print(f"    {k:32s} ?")
    print("\n  obs columns that look like labels or batches:")
    for c in a.obs.columns:
        lc = c.lower()
        tag = ("LABEL" if any(h in lc for h in LABEL_HINTS) else
               "BATCH" if any(h in lc for h in BATCH_HINTS) else None)
        if tag:
            col = a.obs[c]
            print(f"    [{tag}] {c:30s} {col.nunique(dropna=True):6d} classes, "
                  f"{int(col.isna().sum())} NaN")
    print("\n  A label column with NaNs is unusable here: kbet and clisi need a label")
    print("  for every cell, and dropping cells would change the scored cell set.")


def knn_arrays(D, k: int, include_self: bool):
    """Sparse kNN distance matrix -> (indices, distances) of shape (n, k[+1])."""
    from scipy.sparse import csr_matrix
    D = csr_matrix(D)
    n = D.shape[0]
    width = k + 1 if include_self else k
    idx = np.zeros((n, width), dtype=np.int64)
    dst = np.zeros((n, width), dtype=np.float64)
    short = 0
    for i in range(n):
        s, e = D.indptr[i], D.indptr[i + 1]
        cols, vals = D.indices[s:e], D.data[s:e]
        keep = cols != i                              # drop any stored self-edge
        cols, vals = cols[keep], vals[keep]
        order = np.argsort(vals, kind="stable")[:k]
        c, v = cols[order], vals[order]
        if c.size < k:                                # pad by repeating the last
            short += 1
            if c.size == 0:
                c, v = np.array([i]), np.array([0.0])
            c = np.concatenate([c, np.full(k - c.size, c[-1])])
            v = np.concatenate([v, np.full(k - v.size, v[-1])])
        if include_self:
            idx[i] = np.concatenate([[i], c]); dst[i] = np.concatenate([[0.0], v])
        else:
            idx[i] = c; dst[i] = v
    if short:
        print(f"      note: {short}/{n} rows had fewer than {k} neighbours and were "
              f"padded; consider --fully-connected if that fraction is large")
    return idx, dst


def self_test() -> int:
    """
    Validate knn_arrays against sklearn on synthetic data. This is the only
    genuinely new logic here and the likeliest place for an off-by-one, so it is
    checked in the target environment before any metric number is trusted.
    """
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5)); k = 4
    d, i = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    # a scanpy-style graph: k neighbours per row, self EXCLUDED
    rows = np.repeat(np.arange(60), k)
    D = csr_matrix((d[:, 1:].ravel(), (rows, i[:, 1:].ravel())), shape=(60, 60))

    ok = True
    idx, dst = knn_arrays(D, k, include_self=True)
    checks = [
        (f"include_self shape == (60,{k+1})", idx.shape == (60, k + 1)),
        ("column 0 is the query point", bool((idx[:, 0] == np.arange(60)).all())),
        ("column 0 distance is 0", bool((dst[:, 0] == 0).all())),
        ("neighbours match sklearn", bool((idx[:, 1:] == i[:, 1:]).all())),
        ("distances ascending", bool((np.diff(dst, axis=1) >= -1e-12).all())),
    ]
    idx2, _ = knn_arrays(D, k, include_self=False)
    checks += [
        (f"no_include_self shape == (60,{k})", idx2.shape == (60, k)),
        ("no_include_self matches sklearn", bool((idx2 == i[:, 1:]).all())),
    ]
    lil = D.tolil(); lil[7, :] = 0
    D2 = lil.tocsr(); D2.eliminate_zeros()
    try:
        o, _ = knn_arrays(D2, k, include_self=True)
        checks.append(("empty row padded, no crash", o.shape == (60, k + 1)))
    except Exception as ex:                                        # noqa: BLE001
        checks.append((f"empty row padded, no crash ({ex})", False))

    print("=== knn_arrays self-test ===")
    for name, good in checks:
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
    print("\n  " + ("all checks passed; the NeighborsResults conversion is sound."
                    if ok else
                    "*** A CHECK FAILED. Do not trust any LISI/kbet number until "
                    "this is fixed: the conversion feeds every one of them. ***"))
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adata", type=Path, action="append",
                   help="Repeat once per seed. Not needed with --self-test.")
    p.add_argument("--self-test", action="store_true",
                   help="Validate the sparse->NeighborsResults conversion against "
                        "sklearn and exit. Run this once before trusting metrics.")
    p.add_argument("--inspect", action="store_true",
                   help="Print available keys and exit without computing.")
    p.add_argument("--method", default="SQUINT")
    p.add_argument("--latent-keys",
                   default=("cell_code_indices,cell_latent,cell_emb,"
                            "neighborhood_code_indices,neighborhood_latent,"
                            "neighborhood_emb"),
                   help="Comma list. The default is every candidate in the "
                        "mouse-brain file, for the identification run.")
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--batch-key", default="adata_batch_id")
    p.add_argument("--drop-label-nan", action="store_true",
                   help="Drop cells with no cell-type label instead of refusing to "
                        "run. Needed on CosMx NSCLC, whose cell_type has 8,980 NaN. "
                        "Standard scIB practice, but it shrinks the scored cell set, "
                        "so the recomputed ilisi is then NOT comparable to Table 1 "
                        "(which scores every cell). n_cells and n_dropped record it.")
    p.add_argument("--include-self", dest="include_self", action="store_true",
                   default=True, help="Prepend the query point as neighbour 0, "
                                      "matching scib-metrics' own builders.")
    p.add_argument("--no-include-self", dest="include_self", action="store_false")
    p.add_argument("--fully-connected", action="store_true")
    p.add_argument("--expect-ilisi", default=None,
                   help="Arms the gate. Bare float, or key=value pairs such as "
                        "'neighborhood_latent=0.609,cell_latent=0.739'. Keys with no "
                        "expectation are reported ungated, which is how you identify "
                        "which obsm key Table 1 used.")
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=None,
                   help="CSV to write. Default label_conditioned_metrics_<method>.csv "
                        "in the CWD. Never overwrites an existing file (see --force).")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    if not a.adata:
        p.error("--adata is required (or use --self-test)")

    if a.inspect:
        for f in a.adata:
            inspect(f)
        return 0

    # Guard the output BEFORE computing: a name clash should cost a second.
    out = a.out or Path(f"label_conditioned_metrics_{a.method}.csv")
    if out.exists() and not a.force:
        raise SystemExit(f"{out} already exists. This script never overwrites "
                         f"results. Use a different --out, or --force.")

    import anndata as ad
    import pandas as pd
    try:
        import scib_metrics
        from scib_metrics.nearest_neighbors import NeighborsResults
        from vqniche.metrics.utils import (
            compute_knn_graph_connectivities_and_distances as build_knng)
    except ImportError as ex:                                      # noqa: BLE001
        raise SystemExit(f"cannot import scib_metrics or vqniche ({ex}). Activate "
                         f"the squint venv, i.e. the one that produced Table 1.")
    print(f"scib_metrics {scib_metrics.__version__}; calling it directly because "
          f"benchmarking.py passes raw sparse matrices where 0.5.x wants "
          f"NeighborsResults")

    latent_keys = [s.strip() for s in a.latent_keys.split(",") if s.strip()]
    rows = []
    for f in a.adata:
        seed = next((q.split("seed")[-1] for q in f.parts[::-1] if "seed" in q), "0")
        adata = ad.read_h5ad(f)
        n_dropped = 0
        for nm, col in (("cell-type-key", a.cell_type_key),
                        ("batch-key", a.batch_key)):
            if col not in adata.obs.columns:
                raise SystemExit(f"{f}\n  --{nm} {col!r} not in obs. Run --inspect.")
            n_nan = int(adata.obs[col].isna().sum())
            if not n_nan:
                continue
            # Unlabelled cells may be dropped; a cell with no BATCH cannot be
            # placed at all, so that stays fatal.
            if nm == "cell-type-key" and a.drop_label_nan:
                keep = ~adata.obs[col].isna().to_numpy()
                n_dropped = n_nan
                adata = adata[keep].copy()
                print(f"  --drop-label-nan: dropped {n_nan} cells with no "
                      f"{col!r}; {adata.n_obs} remain. The recomputed ilisi is "
                      f"therefore on a SUBSET and is not a Table 1 reproduction.")
                continue
            raise SystemExit(
                f"{f}\n  --{nm} {col!r} has {n_nan} NaN. kbet and clisi need a "
                f"label for every cell; pick a complete column (--inspect), or "
                f"pass --drop-label-nan to score the labelled subset.")
        labels = np.asarray(adata.obs[a.cell_type_key].astype(str))
        batches = np.asarray(adata.obs[a.batch_key].astype(str))

        for lk in latent_keys:
            if lk not in adata.obsm:
                raise SystemExit(f"{f}\n  --latent-keys {lk!r} not in obsm.")
            print(f"\n--- seed {seed} | {lk} ---")
            row = {"method": a.method, "seed": seed, "latent_key": lk,
                   "cell_type_key": a.cell_type_key, "batch_key": a.batch_key,
                   "include_self": a.include_self, "n_cells": int(adata.n_obs),
                   "n_dropped_no_label": n_dropped, "n_labels": int(len(
                       np.unique(labels))), "n_batches": int(len(
                           np.unique(batches))),
                   "scib_version": scib_metrics.__version__, "path": str(f),
                   "error": ""}
            row.update({m: float("nan") for m in METRICS})
            try:
                X = np.asarray(adata.obsm[lk], dtype=np.float64)
                sd = int(seed) if str(seed).isdigit() else 0
                for k in (K_LISI, K_KBET):
                    if f"{lk}_{k}knng_distances" not in adata.obsp:
                        print(f"      building {k}-NN graph on {lk} ...")
                        build_knng(adata=adata, feature_key=lk,
                                   knng_key=f"{lk}_{k}knng",
                                   fully_connected=a.fully_connected,
                                   n_neighbors=k, random_state=sd)
                nr = {k: NeighborsResults(*(lambda t: (t[0], t[1]))(
                          knn_arrays(adata.obsp[f"{lk}_{k}knng_distances"], k,
                                     a.include_self)))
                      for k in (K_LISI, K_KBET)}

                row["ilisi"] = float(scib_metrics.ilisi_knn(nr[K_LISI], batches))
                row["clisi"] = float(scib_metrics.clisi_knn(nr[K_LISI], labels))
                row["kbet"] = float(scib_metrics.kbet_per_label(
                    nr[K_KBET], batches=batches, labels=labels))
                row["casw"] = float(scib_metrics.silhouette_label(X, labels))
                print("    " + "  ".join(f"{m}={row[m]:.4f}" for m in METRICS))
            except Exception as ex:                                # noqa: BLE001
                # A sweep over CANDIDATE representations: integer code indices give
                # degenerate neighbour graphs, so one failing candidate must not end
                # the run. The failure is printed and stored, never swallowed.
                row["error"] = f"{type(ex).__name__}: {ex}"
                print(f"    FAILED -> {row['error'][:170]}")
            rows.append(row)
            for k in (K_LISI, K_KBET):        # free graphs before the next key
                adata.obsp.pop(f"{lk}_{k}knng_distances", None)
                adata.obsp.pop(f"{lk}_{k}knng_connectivities", None)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 84 + "\nMEAN +/- SD ACROSS SEEDS\n" + "=" * 84)
    print(f"  {'latent':26s}" + "".join(f"{m:>17s}" for m in METRICS))
    for lk, g in df.groupby("latent_key"):
        cells = "".join(f"{g[m].mean():>9.4f}+/-{g[m].std(ddof=1):<7.4f}"
                        for m in METRICS)
        print(f"  {lk:26s}{cells}" +
              ("   [all FAILED]" if g[METRICS].isna().all().all() else ""))

    ok = True
    if a.expect_ilisi:
        exp = ({k.strip(): float(v) for k, v in
                (kv.split("=", 1) for kv in a.expect_ilisi.split(","))}
               if "=" in a.expect_ilisi
               else {lk: float(a.expect_ilisi) for lk in latent_keys})
        print("\n" + "=" * 84 +
              "\nREPRODUCTION GATE (recomputed ilisi vs published Table 1)\n" +
              "=" * 84)
        for lk, g in df.groupby("latent_key"):
            got = g["ilisi"].mean()
            if lk not in exp:
                print(f"  {lk:26s} ilisi {got:.4f}   (ungated; compare yourself)")
                continue
            d = abs(got - exp[lk])
            ok &= d <= a.tol
            print(f"  {lk:26s} ilisi {got:.4f} vs {exp[lk]:.4f}  |diff| {d:.4f}  "
                  f"{'PASS' if d <= a.tol else 'FAIL'}")
        if not ok:
            print("\n  Gate failed. Before concluding the representation is wrong,")
            print("  try --no-include-self: the self-loop convention shifts every")
            print("  LISI value. And note the docstring caveat, that the published")
            print("  iLISI may not be reproducible in this venv at all.")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} appeared while computing; refusing to overwrite.")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
