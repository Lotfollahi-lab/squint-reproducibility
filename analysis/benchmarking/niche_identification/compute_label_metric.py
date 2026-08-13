#!/usr/bin/env python
"""
compute_label_metric.py — ONE label-conditioned metric, one method, one dataset.
=============================================================================
For R2-W1b. One metric per process, so every (metric, dataset, method) runs as its own
LSF job. cilisi is much cheaper than the metrics it replaced, but the split is kept
because a single failure then costs one number instead of all of them, which is how the
199k-cell NSCLC runs used to lose a whole batch of metrics at once.

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

THE METRICS
  cilisi      per-cell-type iLISI, normalised to [0,1] (Rautenstrauch & Ohler 2025;
              carmonalab/scIntegrationMetrics). The label-conditioned integration
              metric R2-W1b asked for: it subsets to one annotated group and asks
              whether BATCHES mix inside it. Emits TWO rows, `cilisi` weighted over
              cells and `cilisi_means` as the mean of per-group means, matching the
              two the R package reports.
              Mind the direction: it asks whether BATCHES mix inside a group, not
              whether the groups themselves stay separated.
  ilisi/mmd   the two UNCONDITIONED metrics of Table 1, via the paper's own helpers,
              as the anchor cilisi is read against rather than as new results. MMD is
              a DISTANCE, so lower is better; cilisi and ilisi are higher-is-better.

ONE FINDING WORTH KEEPING, because it cost real time to establish and applies to any
metric added here later. Any metric that SELECTS NEIGHBOURS from a graph is corrupted
on a quantized representation unless rows are permuted first; see the note below. When
that happens the symptom is not an error but a plausible, badly wrong number, and the
natural conclusion to draw from it, that a quantized representation cannot support such
a metric, is false. Permute first, then judge.

SKIP RULE, applied by scIB and scIntegrationMetrics alike: a group needs >=10 cells
and >=2 batches. Where no group spans batches the metric is undefined rather than bad,
and cilisi raises "no cell type has >=10 cells and >=2 batches". That is the mouse
brain, whose 49 cell types are two disjoint per-section vocabularies. The failure is
recorded in `error` with the value left NaN, never reported as zero, and
n_types_scored is always written so the denominator is visible.

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

METRICS = ["cilisi", "ilisi", "mmd"]
NEEDS_GLOBAL_GRAPH = {"ilisi": 90}
NEEDS_SUBSET = {"cilisi"}
K_SUBSET = 90
# MMD is a DISTANCE: lower is better, unlike cilisi and ilisi.
LOWER_IS_BETTER = {"mmd"}


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


# --------------------------------------------------------------------------
# Inlined from the retired compute_label_conditioned_metrics.py so this module
# stands alone. That file computed the metric set we no longer report and has
# been deleted.
# --------------------------------------------------------------------------
def _read_elem():
    try:
        from anndata.io import read_elem                           # anndata >= 0.11
    except ImportError:                                            # pragma: no cover
        from anndata.experimental import read_elem
    return read_elem

def load_minimal(path: Path, obsm_keys, obs_cols):
    """
    obs plus the requested obsm arrays, read through h5py. Never touches uns or X.

    Not ad.read_h5ad, for two independent reasons:
      * Several baseline predicted_adata.h5ad files carry uns['log1p']['base'] = None,
        written by an older anndata, and 0.11.4 raises IORegistryError ("No read
        method registered for IOSpec(encoding_type='null')") while parsing uns. It
        fails on the whole file even though nothing here needs uns.
      * X is never used. Every metric reads obsm and obs, and sc.pp.neighbors takes
        use_rep=<obsm key>. Skipping X turns a multi-GB read into a few hundred MB,
        which is what lets the 199k-cell NSCLC file run inside a 64 GB job.
    """
    import anndata as ad
    import h5py
    read_elem = _read_elem()
    with h5py.File(path, "r") as h:
        if "obs" not in h:
            raise SystemExit(f"{path}\n  no obs group. This file is a stub: the "
                             f"novae baseline saved predicted_adata.h5ad without "
                             f"obs or obsm at every timestamp, so its metrics "
                             f"cannot be recomputed from disk.")
        obs = read_elem(h["obs"])
        for c in obs_cols:
            if c not in obs.columns:
                raise SystemExit(f"{path}\n  obs column {c!r} absent. Run --inspect.")
        if "obsm" not in h:
            raise SystemExit(f"{path}\n  no obsm group, so there is no "
                             f"representation to score (looked for "
                             f"{list(obsm_keys)}).")
        have = list(h["obsm"].keys())
        missing = [k for k in obsm_keys if k not in have]
        if missing:
            raise SystemExit(f"{path}\n  --latent-keys {missing} not in obsm. "
                             f"Present: {have}")
        obsm = {k: np.asarray(read_elem(h["obsm"][k])) for k in obsm_keys}
    return ad.AnnData(obs=obs, obsm=obsm)


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


def per_type_scores(X, labels, batches, which="cilisi"):
    """
    The label-conditioned metrics: subset to one cell type, apply the paper's own
    machinery inside it, then aggregate over cell types.

    cilisi: iLISI within the subset, via scib ilisi_knn on an EXACT k=90 kNN.

    Exact rather than paper_graph's NNDescent, deliberately. The paper uses NNDescent
    for the GLOBAL graph because approximate search is a necessity at 50k-200k cells;
    inside a group of a few thousand it is cheap to be exact, and NNDescent's
    approximation error is largest at small n. Measured on eczema cell_emb, seed 0:
    the stored five-seed run recorded 0.6473, exact reproduces 0.6450 and NNDescent
    gives 0.6521, so exact is the closer match. All three sit far inside the 0.0160
    seed-to-seed sd, so the choice does not move any reported figure; exact is used
    because it is both more accurate here and what the reported runs used.
    paper_graph is still used for the global ilisi, where matching the published
    recipe is the point.

    Returns (cell-weighted mean, mean of per-type means, n types scored).
    """
    import scib_metrics
    from scib_metrics.nearest_neighbors import NeighborsResults
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(SHUFFLE_SEED)
    vals, sizes = [], []
    for g in scorable(labels, batches):
        m = labels == g
        Xg, bg = X[m], batches[m]
        # Permute within the subset too: subsetting a section-ordered file preserves
        # the ordering, so the row-order tie-breaking artefact survives subsetting.
        q = rng.permutation(Xg.shape[0])
        Xg, bg = Xg[q], bg[q]
        k = int(min(K_SUBSET, int(m.sum()) - 1))
        d, i = NearestNeighbors(n_neighbors=k + 1).fit(Xg).kneighbors(Xg)
        vals.append(float(scib_metrics.ilisi_knn(
            NeighborsResults(indices=i, distances=d), bg)))
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
                if a.metric in NEEDS_SUBSET:
                    v, vm, nt = per_type_scores(X, labels, batches, a.metric)
                    row["value"], row["n_types_scored"] = v, nt
                    rows.append({**row, "metric": f"{a.metric}_means",
                                 "value": vm})
                elif a.metric == "mmd":
                    # The paper's own MMD, unconditioned: the Table 1 anchor. Same
                    # helper, same defaults (n_sub=2000, n_sigma=1000).
                    _, compute_mmd = paper_helpers()
                    v = compute_mmd(X, batches,
                                    rng=np.random.default_rng(SHUFFLE_SEED))
                    row["value"] = float("nan") if v is None else float(v)
                else:
                    # ilisi, on the graph the PAPER builds: NNDescent straight on the
                    # embedding at k=90 (see paper_graph).
                    nr = paper_graph(X, NEEDS_GLOBAL_GRAPH[a.metric])
                    row["value"] = float(scib_metrics.ilisi_knn(nr, batches))
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
