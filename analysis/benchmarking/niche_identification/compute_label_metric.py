#!/usr/bin/env python
"""
compute_label_metric.py — ONE label-conditioned metric, one method, one dataset.
=============================================================================
For R2-W1b. Split to one metric per process so every (metric, dataset, method) can
run as its own LSF job: the metrics have wildly different costs (basw needs no graph,
kbet_per_label builds a diffusion map per cell type) and bundling them meant the
199k-cell NSCLC jobs died on the slowest one and lost the cheap ones with it.

Output is LONG, one row per (metric, seed, latent_key), with the number in `value`.
That keeps the schema identical across metrics so the csvs concatenate.

HOW THESE ARE KEPT COMPARABLE TO TABLE 1
iLISI and MMD are not reimplemented here. squint/examples/compute_inference_metrics.py
provides compute_ilisi and compute_mmd_comparable, every benchmark runner calls them,
and run_pca_leiden.py:1046 records that the batch-integration CSV uses "same helpers +
same defaults". This module imports the same two, and builds its graphs the same way
compute_ilisi does -- pynndescent NNDescent straight on the embedding at k=90, query
point included (see paper_graph). An earlier version went through scanpy's
sc.pp.neighbors wrapper instead, which excludes self and returns connectivities, so
the self column had to be re-prepended by hand: a needless second difference from the
published numbers, now removed.

THE METRICS, and why each is or is not label-conditioned
  basw        scib_metrics.silhouette_batch. Silhouette w.r.t. BATCH within each cell
              type, averaged over cell types. Label-conditioned. No graph.
  bras        scib_metrics.bras. Same machinery with cosine distance and mean-distance
              -to-all-other-batches; introduced to fix documented erratic behaviour of
              Batch ASW (Rautenstrauch & Ohler, Nat Biotechnol 2025). No graph.
  kbet_label  scib_metrics.kbet_per_label. Batch mixing within each cell type through
              a DIFFUSION map. The strictest reading of R2's request. An earlier note
              here claimed it is inherently undefined on a quantized representation,
              because the per-label subgraph fragments and _kbet.py falls back to
              `score = 0`. That fragmentation was caused by the row-order tie-breaking
              described below, not by quantization, so the claim is RETRACTED and the
              metric is computed normally.
  kbet_strat  scib_metrics.kbet (the PLAIN one) computed within each cell type. This is
              Buttner et al.'s original chi-squared test on kNN neighbourhoods. The
              diffusion step that breaks kbet_label is scIB's addition, not part of the
              original metric, so this is the label-conditioned kBET that survives ties.
  cilisi      per-cell-type iLISI, normalised to [0,1] (Rautenstrauch & Ohler 2025;
              carmonalab/scIntegrationMetrics). Emits TWO rows, `cilisi` weighted over
              cells and `cilisi_means` as the mean of per-type means, matching the two
              the R package reports. NOT the same as clisi: clisi feeds CELL-TYPE
              labels to LISI and asks whether cell types stay separated
              (bio-conservation), while cilisi asks whether BATCHES mix inside each
              cell type (integration). One letter apart, opposite questions.
  graph_conn  scib_metrics.graph_connectivity. Per cell type, the fraction of its cells
              in the largest connected component. Label-conditioned. Its earlier
              collapse on quantized representations had the same row-order cause.
  cmmd        the paper's OWN compute_mmd_comparable, computed WITHIN each cell type
              and averaged. The most direct answer to R2-W1b, which objected that
              "MMD does not by itself establish preservation of cell-type-specific
              biology": same MMD, same median-heuristic bandwidth, same defaults, now
              conditioned on cell type. Emits cmmd and cmmd_means. A DISTANCE, so
              LOWER IS BETTER, opposite to every other metric here.
  ilisi/mmd   the two UNCONDITIONED metrics of Table 1, computed with the paper's own
              helpers as the anchor the conditioned versions are read against, not as
              new results.
  clisi/casw  bio-conservation, included only so one runner covers the full scIB set.

SKIP RULE, applied by scIB and the R package alike: a cell type needs >=10 cells and
>=2 batches. Where no cell type spans batches every label is skipped, and the metric
is undefined rather than bad. That is the mouse brain, whose 49 cell types are two
disjoint per-section vocabularies: basw raises "No objects to concatenate",
kbet_label returns NaN from np.nanmean of an empty slice. Both are recorded in
`error` / left NaN, never reported as zero. n_types_scored is always written so the
denominator is visible.

WHAT IS SCORED: exactly the four representations the paper's own integration metrics
use -- cell_emb, neighborhood_emb, cell_latent, neighborhood_latent. The raw
code-index arrays are refused (see resolve_key).

USAGE
  python compute_label_metric.py --metric cilisi --method SQUINT \\
      --cell-type-key new_annotation --latent-keys cell_emb,cell_latent \\
      --adata .../seed0/predicted_adata.h5ad --out .../lm_cilisi_<ds>_SQUINT.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from compute_label_conditioned_metrics import load_minimal

# =============================================================================
# WHY EVERY NEIGHBOUR GRAPH HERE IS BUILT ON RANDOMLY PERMUTED ROWS
#
# The predicted_adata files are section CONCATENATIONS, so row order IS batch:
# measured on eczema, batch 0 occupies rows 0-17831, batch 1 rows 17832-35215,
# batch 2 rows 35216-53654 -- exactly three contiguous runs.
#
# SQUINT's representations contain huge groups of EXACTLY identical points, because
# *_emb is a codebook lookup: the largest tied group in cell_emb holds 1,669 cells at
# one point. kNN implementations break distance ties by row index, so a tied query
# returns its index-adjacent neighbours, which are its SAME-BATCH neighbours. Measured
# directly: that tied group's true batch composition is 654/356/659, near-perfectly
# mixed, yet sklearn returns 90 neighbours all from batch 0.
#
# Any neighbourhood metric computed that way measures the concatenation order of the
# file rather than the embedding, and it does so in the direction that makes a
# discrete representation look catastrophically unmixed. It produced CiLISI 0.14 on a
# representation whose published iLISI is 0.498, which is what exposed it.
#
# Permuting rows once, up front, makes tie-breaking uniform over the tied group,
# which is the correct behaviour: among cells at an identical point, no cell is
# nearer than any other, so the neighbourhood should be a uniform sample of them.
# The permutation is seeded, so runs are reproducible.
# =============================================================================
SHUFFLE_SEED = 0

METRICS = ["basw", "bras", "kbet_label", "kbet_strat", "cilisi", "cmmd",
           "graph_conn", "ilisi", "mmd", "clisi", "casw"]
NEEDS_GLOBAL_GRAPH = {"kbet_label": 50, "graph_conn": 90, "ilisi": 90, "clisi": 90}
NEEDS_SUBSET = {"kbet_strat", "cilisi", "cmmd"}
K_SUBSET = 90
# MMD is a DISTANCE: lower is better, opposite to every other metric here.
LOWER_IS_BETTER = {"mmd", "cmmd"}


def paper_helpers():
    """
    The paper's OWN iLISI and MMD implementations, imported rather than reimplemented,
    so these numbers are produced by the same code that produced Table 1.

    squint/examples/compute_inference_metrics.py, reached the same way
    run_pca_leiden.py:133 _import_metric_helpers reaches it. Every benchmark runner
    calls these two, and run_pca_leiden.py:1046 records that the batch-integration CSV
    uses "same helpers + same defaults as compute_inference_metrics.py".
    """
    import sys
    for p in ("/nfs/team361/sb75/squint/examples",
              "/Users/sebastian.birk/workspace/squint_claude_project/squint/examples"):
        if Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)
    from compute_inference_metrics import compute_ilisi, compute_mmd_comparable
    return compute_ilisi, compute_mmd_comparable


def paper_graph(emb, k):
    """
    The kNN graph exactly as compute_inference_metrics.compute_ilisi builds it:

        knn = NNDescent(emb, n_neighbors=min(n_neighbors, len(emb) - 1))
        indices, distances = knn.neighbor_graph
        NeighborsResults(indices=indices, distances=distances)

    pynndescent DIRECTLY on the embedding at k=90, with the query point included as
    neighbour 0. Deliberately not scanpy's sc.pp.neighbors wrapper, which was used
    here previously: it excludes self and returns connectivities, so the self column
    had to be re-prepended by hand, and that is a needless second source of difference
    from the published numbers.
    """
    from pynndescent import NNDescent
    from scib_metrics.nearest_neighbors import NeighborsResults
    kk = int(min(k, len(emb) - 1))
    idx, dist = NNDescent(emb, n_neighbors=kk).neighbor_graph
    return NeighborsResults(indices=idx, distances=dist)


def dataset_of(path: Path) -> str:
    parts = path.parts
    if "artifacts" in parts:
        i = parts.index("artifacts")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "?"


def seed_of(path: Path) -> str:
    for q in path.parts[::-1]:
        m = re.search(r"seed[_-]?(\d+)", q)
        if m:
            return m.group(1)
    return "0"


CODE_INDEX_KEYS = ("cell_code_indices", "neighborhood_code_indices")


def resolve_key(adata, spec):
    """
    Read a representation, and REFUSE the raw code-index arrays.

    Scored representations are exactly the four the paper's own integration metrics
    use -- cell_emb, neighborhood_emb, cell_latent, neighborhood_latent -- read
    straight from obsm. See any run's metrics/batch_integration_metrics.csv, whose
    emb_key column holds cell_emb / neighborhood_emb / cell_latent /
    neighborhood_latent and never a code-index key.

    obsm['cell_code_indices'] is refused rather than handled, because it is not a
    metric space. It is (n, 2): column 0 the coarse RVQ code (K=30), column 1 the
    residual code (K=90). Euclidean distance over that asks for distance in a space
    whose two axes are unrelated categorical variables on different scales, where
    "code 5 vs code 7" is a difference of 2 and means nothing. Every neighbourhood
    metric computed on it earlier was meaningless for that reason alone, before the
    tie-breaking problem above is even considered. *_emb is the faithful vector form
    of the same discrete representation -- it is the codebook lookup, constant within
    each code -- so nothing is lost by scoring it instead.
    """
    if spec in CODE_INDEX_KEYS or spec.startswith(CODE_INDEX_KEYS):
        raise ValueError(
            f"{spec!r} is a raw RVQ code-index array and is deliberately not scored. "
            f"Its two integer columns are unrelated categoricals on different scales, "
            f"so no distance defined on it is meaningful. Score the quantized vectors "
            f"instead: cell_emb / neighborhood_emb, which is what the paper's own "
            f"iLISI and MMD are computed on.")
    return np.asarray(adata.obsm[spec], dtype=np.float64), spec


def scorable(labels, batches):
    """The cell types scIB and scIntegrationMetrics both agree are usable."""
    out = []
    for g in np.unique(labels):
        m = labels == g
        if int(m.sum()) >= 10 and len(np.unique(batches[m])) >= 2:
            out.append(g)
    return out


def per_type_scores(X, labels, batches, which):
    """
    The label-conditioned metrics: subset to one cell type, apply the paper's own
    machinery inside it, then aggregate over cell types.

      cilisi      iLISI within the subset, via paper_graph (NNDescent k=90) and
                  scib ilisi_knn -- the same construction compute_ilisi uses globally.
      kbet_strat  plain scib kbet within the subset, i.e. Buttner's chi-squared test
                  on the same graph.
      cmmd        the PAPER'S OWN compute_mmd_comparable within the subset. This is the
                  most direct answer to R2-W1b, which objected that "MMD does not by
                  itself establish preservation of cell-type-specific biology": it is
                  the same MMD, same median-heuristic bandwidth, same defaults,
                  conditioned on cell type. NOTE it is a DISTANCE, so lower is better.

    Returns (cell-weighted mean, mean of per-type means, n types scored).
    """
    import scib_metrics

    _, compute_mmd = paper_helpers()
    rng = np.random.default_rng(SHUFFLE_SEED)
    vals, sizes = [], []
    for g in scorable(labels, batches):
        m = labels == g
        Xg, bg = X[m], batches[m]
        # Permute within the subset too: subsetting a section-ordered file preserves
        # the ordering, so the row-order tie-breaking artefact survives subsetting.
        q = rng.permutation(Xg.shape[0])
        Xg, bg = Xg[q], bg[q]
        if which == "cmmd":
            v = compute_mmd(Xg, bg, rng=np.random.default_rng(SHUFFLE_SEED))
            if v is None:            # <2 non-empty batches; scorable() should prevent
                continue
            vals.append(float(v))
        else:
            nr = paper_graph(Xg, K_SUBSET)
            vals.append(float(scib_metrics.ilisi_knn(nr, bg)) if which == "cilisi"
                        else float(scib_metrics.kbet(nr, bg)[0]))
        sizes.append(int(m.sum()))
    if not vals:
        raise ValueError("no cell type has >=10 cells and >=2 batches, so this "
                         "metric is undefined on this dataset")
    w = np.asarray(sizes, float) / float(sum(sizes))
    return float(np.dot(vals, w)), float(np.mean(vals)), len(vals)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", required=True, choices=METRICS)
    p.add_argument("--adata", type=Path, action="append", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--latent-keys", required=True)
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--batch-key", default="adata_batch_id")
    p.add_argument("--drop-label-nan", action="store_true")
    p.add_argument("--fully-connected", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    if a.out.exists() and not a.force:
        raise SystemExit(f"{a.out} exists. Use --force or a different --out.")

    import pandas as pd
    import scib_metrics

    want = [s.strip() for s in a.latent_keys.split(",") if s.strip()]
    rows = []
    for f in a.adata:
        import h5py
        with h5py.File(f, "r") as h:
            have = list(h["obsm"].keys()) if "obsm" in h else []
        # A spec may be 'cell_code_indices[level_0]'; the obsm key is the part before
        # the bracket (see resolve_key).
        def base_of(spec):
            return re.sub(r"\[level_\d+\]$", "", spec)
        keys = [k for k in want if base_of(k) in have]
        base = {"metric": a.metric, "method": a.method, "dataset": dataset_of(f),
                "seed": seed_of(f), "cell_type_key": a.cell_type_key,
                "batch_key": a.batch_key, "scib_version": scib_metrics.__version__,
                "path": str(f)}
        for k in [k for k in want if base_of(k) not in have]:
            rows.append({**base, "latent_key": k, "value": float("nan"),
                         "error": f"obsm key {k!r} not in file (have {have})"})
        if not keys:
            continue

        adata = load_minimal(f, keys, (a.cell_type_key, a.batch_key))
        n_dropped = 0
        nan_mask = adata.obs[a.cell_type_key].isna()
        if nan_mask.any():
            if not a.drop_label_nan:
                raise SystemExit(f"{f}\n  {a.cell_type_key!r} has "
                                 f"{int(nan_mask.sum())} NaN; pass --drop-label-nan.")
            n_dropped = int(nan_mask.sum())
            adata = adata[~nan_mask.to_numpy()].copy()
        # See the SHUFFLE_SEED note at the top: this is not cosmetic. Without it every
        # graph-based metric on a tied representation reports the file's section
        # ordering instead of the embedding.
        perm = np.random.default_rng(SHUFFLE_SEED).permutation(adata.n_obs)
        adata = adata[perm].copy()
        labels = np.asarray(adata.obs[a.cell_type_key].astype(str))
        batches = np.asarray(adata.obs[a.batch_key].astype(str))
        n_ok = len(scorable(labels, batches))
        print(f"\n--- {a.metric} | {a.method} | seed {base['seed']} | "
              f"{adata.n_obs} cells | {n_ok}/{len(np.unique(labels))} cell types "
              f"scorable ---", flush=True)
        if not n_ok:
            print("    no cell type spans batches; every label-conditioned metric is"
                  " undefined here", flush=True)

        for k in keys:
            row = {**base, "latent_key": k, "n_cells": int(adata.n_obs),
                   "n_dropped_no_label": n_dropped, "n_types_scorable": n_ok,
                   "n_labels_total": int(len(np.unique(labels))),
                   "n_batches": int(len(np.unique(batches))),
                   "value": float("nan"), "n_types_scored": -1, "error": ""}
            try:
                X, _ = resolve_key(adata, k)
                if a.metric in ("basw", "bras"):
                    fn = (scib_metrics.silhouette_batch if a.metric == "basw"
                          else scib_metrics.bras)
                    row["value"] = float(fn(X, labels=labels, batch=batches))
                    row["n_types_scored"] = n_ok
                elif a.metric == "casw":
                    row["value"] = float(scib_metrics.silhouette_label(X, labels))
                elif a.metric in NEEDS_SUBSET:
                    v, vm, nt = per_type_scores(X, labels, batches, a.metric)
                    row["value"], row["n_types_scored"] = v, nt
                    if a.metric in ("cilisi", "cmmd"):
                        rows.append({**row, "metric": f"{a.metric}_means",
                                     "value": vm})
                elif a.metric == "mmd":
                    # The paper's own MMD, unconditioned: the Table 1 anchor that the
                    # conditioned cmmd should be read against. Same helper, same
                    # defaults (n_sub=2000, n_sigma=1000).
                    _, compute_mmd = paper_helpers()
                    v = compute_mmd(X, batches,
                                    rng=np.random.default_rng(SHUFFLE_SEED))
                    row["value"] = float("nan") if v is None else float(v)
                else:
                    # Global-graph metrics, on the graph the PAPER builds: NNDescent
                    # straight on the embedding (see paper_graph). k=90 for the LISI
                    # family and graph connectivity, k=50 for kbet, matching
                    # benchmarking.py.
                    nr = paper_graph(X, NEEDS_GLOBAL_GRAPH[a.metric])
                    if a.metric == "kbet_label":
                        row["value"] = float(scib_metrics.kbet_per_label(
                            nr, batches=batches, labels=labels))
                        row["n_types_scored"] = n_ok
                    elif a.metric == "graph_conn":
                        row["value"] = float(scib_metrics.graph_connectivity(
                            nr, labels))
                    elif a.metric == "ilisi":
                        row["value"] = float(scib_metrics.ilisi_knn(nr, batches))
                    else:
                        row["value"] = float(scib_metrics.clisi_knn(nr, labels))
                print(f"    {k:28s} {a.metric} = {row['value']:.4f}", flush=True)
            except Exception as ex:                                # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {ex}"
                print(f"    {k:28s} FAILED -> {row['error'][:130]}", flush=True)
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        print("\n" + "=" * 74 + "\nMEAN +/- SD ACROSS SEEDS\n" + "=" * 74)
        for (mt, k), g in df.groupby(["metric", "latent_key"]):
            ok = g["value"].dropna()
            cell = (f"{ok.mean():.4f} +/- {ok.std(ddof=1):.4f}" if len(ok) > 1
                    else f"{ok.mean():.4f}" if len(ok) == 1 else "n/a")
            print(f"  {mt:14s}{k:30s}{cell:>22s}   ({len(ok)}/{len(g)} seeds)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
