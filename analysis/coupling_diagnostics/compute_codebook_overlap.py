#!/usr/bin/env python
"""
compute_codebook_overlap.py
===========================
Codebook *specialization* matrix: how much cell-type information lives in the
niche codes, how much niche information lives in the cell codes, and -- the
piece that makes those numbers interpretable -- how much the two EXPERT
ANNOTATIONS already share with each other.

Motivation (MLCB reviewer response)
-----------------------------------
Reviewers asked us to "directly quantify how much cell-type information remains
in the niche codes and how much niche information remains in the cell codes."
The four code-vs-label cells of that question are ALREADY computed by
`squint/examples/compute_inference_metrics.py` (it scores every code view
against every label key) and land in each run's
`metrics/niche_identification_metrics.csv`.

What is NOT computed anywhere is the **reference ceiling**: NMI/ARI between the
two ground-truth annotations themselves. That matters because cell type and
niche are biologically correlated (e.g. cortical layers are simultaneously a
cell-type gradient and a spatial domain), so a non-zero cross term is expected
even for a perfectly specialised codebook. Without the ceiling, an off-diagonal
NMI of 0.38 is uninterpretable; with it, the claim becomes falsifiable:

    each codebook is closest to its OWN annotation, and neither cross term
    exceeds what the two annotations already share with each other.

MATCHED CELL SETS (important)
-----------------------------
The annotations cover DIFFERENT cell subsets: on the mouse brain `cell_type` is
valid for 86,822 cells while the niche annotation covers 85,704. Scoring each
pair on its own native subset -- what the metrics pipeline does -- means the
cross terms and the ceiling would sit on three different cell sets and would
not be strictly comparable.

This script therefore emits every pair twice, tagged by `cellset`:

  * `native`  -- each pair on its own validity mask. Reproduces the pipeline's
                 `metrics/niche_identification_metrics.csv` EXACTLY; use it to
                 verify the script against the published Table 1 numbers.
  * `common`  -- every pair on ONE shared mask: cells with a valid cell-type
                 label AND a valid niche label (and in the requested split).
                 **This is the set the reported matrix uses**, so all four
                 code x label cells, the label x label ceiling and the
                 code x code overlap describe the same cells.

The `niche` side is always the aggregated niche quantity used everywhere else in
the paper (cell-count-weighted over whichever region columns a dataset
populates), never the individual region columns -- those are kept in the
per-seed CSV for transparency but excluded from the printed matrix.

Other conventions are mirrored EXACTLY from `compute_inference_metrics.py` so
the numbers stay comparable to the published Table 1 values:
  * the same DEFAULT_{CELL,NICHE}_LABEL_KEYS search lists;
  * the same label cleaning (drop NaN and the literal strings "nan"/"None");
  * the same code views: `level_0` (coarse macro cluster) and `composite`
    (`pd.factorize` over the full per-level tuple);
  * the same cell-count-weighted aggregation of the niche label columns.

We additionally report AMI (adjusted mutual information) alongside NMI, because
NMI is biased upward when the two groupings have very different numbers of
clusters -- relevant for the `composite` views (~2000 leaf codes) and for
annotations with 49/63/167 classes. AMI corrects for chance; if the
specialization pattern holds under both, it is not an artefact of cluster count.

Usage
-----
  # multiseed sweep (default): one predicted_adata per seed -> mean +/- sd
  python compute_codebook_overlap.py --variant s57_v19_ --dataset mmb0-1b_smb1-1b_1p

  # a single run
  python compute_codebook_overlap.py --predicted-adata /path/to/predicted_adata.h5ad

Outputs (to --out, default <run parent>/codebook_overlap):
  codebook_overlap_per_seed.csv   one row per (seed, split, cellset, x, y)
  codebook_overlap_summary.csv    mean/sd/n_seeds per (split, cellset, x, y)
and a printed specialization matrix (cellset=common).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None
try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None
try:
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )
except ImportError:  # pragma: no cover
    normalized_mutual_info_score = None


# --------------------------------------------------------------------------
# Defaults (kept identical to compute_inference_metrics.py / diagnose_coupling)
# --------------------------------------------------------------------------
DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"
DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"
DEFAULT_VARIANT_PREFIX = "s57_v19_"     # the headline reference model

DEFAULT_CELL_LABEL_KEYS = ["cell_type", "cell_types", "annotation", "new_annotation"]
DEFAULT_NICHE_LABEL_KEYS = ["niche", "Sub_molecular_tissue_region", "ccf_region_name",
                            "spatial_cluster", "niche_type"]

# obs/obsm keys written by run_squint.py's predict()
CELL_CODE_KEY = "cell_code_index"           # -> obsm['cell_code_indices'] when L>1
NICHE_CODE_KEY = "neighborhood_code_index"  # -> obsm['neighborhood_code_indices']

NICHE_AGG_KEY = "niche"   # name of the aggregated niche quantity


# --------------------------------------------------------------------------
# Helpers mirrored from compute_inference_metrics.py
# --------------------------------------------------------------------------
def _clean_labels(adata, label_key: str) -> Optional[np.ndarray]:
    """
    Return an object array of string labels, invalid entries as None.

    Mirrors compute_inference_metrics.py: drops real NaN plus the literal
    strings "nan" and "None" (h5ad round-trips smuggle both through).
    """
    if label_key not in adata.obs.columns:
        return None
    labels = adata.obs[label_key]
    labels_str = labels.astype("object").map(
        lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
    )
    arr = np.asarray(labels_str.to_numpy(), dtype=object).copy()
    bad = np.array([(v is None) or (v == "nan") or (v == "None") for v in arr])
    arr[bad] = None
    return arr


def _flatten_codes(adata, obs_key: str) -> List[Tuple[np.ndarray, str]]:
    """
    Pull integer cluster id(s) per cell. Identical semantics to
    compute_inference_metrics._flatten_codes: single-level -> obs (1 view);
    multi-level -> obsm, reported as [level_0] and [composite].
    """
    if obs_key in adata.obs.columns:
        return [(adata.obs[obs_key].to_numpy().astype(int), obs_key)]

    obsm_key = obs_key.replace("_index", "_indices")
    if obsm_key in adata.obsm:
        idx_2d = np.asarray(adata.obsm[obsm_key]).astype(int)
        if idx_2d.ndim == 1:
            return [(idx_2d, obsm_key)]
        out: List[Tuple[np.ndarray, str]] = []
        out.append((idx_2d[:, 0].astype(int), f"{obsm_key}[level_0]"))
        composite, _uniq = pd.factorize(list(map(tuple, idx_2d)))
        out.append((composite.astype(int), f"{obsm_key}[composite]"))
        return out
    return []


def _split_mask(adata, split: str) -> np.ndarray:
    """all -> every cell; otherwise match obs['data_split'] (train/test)."""
    n = adata.n_obs
    if split == "all":
        return np.ones(n, dtype=bool)
    if "data_split" not in adata.obs.columns:
        return np.zeros(n, dtype=bool)
    return (adata.obs["data_split"].astype(str).to_numpy() == split)


def _merge_niche_columns(labels: Dict[str, np.ndarray],
                         cols: Sequence[str]) -> np.ndarray:
    """
    Build ONE niche label vector by taking, per cell, the value from the first
    column that is valid there.

    Only used with --niche-mode merged. NOTE: when a dataset's sections use
    disjoint region vocabularies (mouse brain), a merged column also encodes
    section identity, which is why the paper's default is the weighted
    aggregation instead.
    """
    n = len(next(iter(labels.values())))
    out = np.full(n, None, dtype=object)
    for col in cols:
        vals = labels[col]
        take = np.array([(out[i] is None) and (vals[i] is not None) for i in range(n)])
        out[take] = vals[take]
    return out


def _score(x: np.ndarray, y: np.ndarray) -> dict:
    """NMI / ARI / AMI on two aligned label vectors."""
    return {
        "NMI": float(normalized_mutual_info_score(x, y)),
        "ARI": float(adjusted_rand_score(x, y)),
        "AMI": float(adjusted_mutual_info_score(x, y)),
        "n_cells": int(len(x)),
        "n_x_clusters": int(len(np.unique(x))),
        "n_y_clusters": int(len(np.unique(y))),
    }


def _pair_row(seed, split, cellset, x_kind, x_key, y_kind, y_key,
              x_vals, y_vals, keep) -> Optional[dict]:
    """Score one pair on the cells in `keep` (already fully masked)."""
    n = int(keep.sum())
    if n == 0:
        return None
    row = {"seed": seed, "split": split, "cellset": cellset,
           "x_kind": x_kind, "x_key": x_key,
           "y_kind": y_kind, "y_key": y_key}
    row.update(_score(np.asarray(x_vals)[keep], np.asarray(y_vals)[keep]))
    return row


def _weighted_niche_aggregate(rows: List[dict],
                              niche_cols: Sequence[str]) -> List[dict]:
    """
    Collapse the several niche region columns into one comparable `niche` row
    per (seed, split, cellset, x_key), cell-count weighted.

    Mirrors compute_inference_metrics.py: each per-column row restricts to the
    cells where THAT column is valid, so n varies and the weighting is what
    makes the collapsed number comparable across sections.

    Verified to reproduce the pipeline's synthesized "niche" rows to <1e-12.
    """
    if pd is None or not rows:
        return []
    agg_cols = [c for c in niche_cols if c != NICHE_AGG_KEY]
    if not agg_cols:
        # dataset has a real `niche` column already -- nothing to aggregate
        return []
    df = pd.DataFrame(rows)
    sub = df[(df["y_kind"] == "label") & (df["y_key"].isin(set(agg_cols)))]
    out: List[dict] = []
    if sub.empty:
        return out
    for (seed, split, cellset, x_kind, x_key), grp in sub.groupby(
            ["seed", "split", "cellset", "x_kind", "x_key"], sort=False):
        n_tot = int(grp["n_cells"].sum())
        if n_tot <= 0:
            continue
        agg = {"seed": seed, "split": split, "cellset": cellset,
               "x_kind": x_kind, "x_key": x_key,
               "y_kind": "label", "y_key": NICHE_AGG_KEY}
        for m in ("NMI", "ARI", "AMI"):
            agg[m] = float((grp[m] * grp["n_cells"]).sum() / n_tot)
        agg["n_cells"] = n_tot
        agg["n_x_clusters"] = int(grp["n_x_clusters"].max())
        agg["n_y_clusters"] = np.nan   # aggregate: no single class count
        agg["sources"] = ";".join(map(str, grp["y_key"].tolist()))
        out.append(agg)
    return out


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
def analyse_one(adata, seed: str, splits: Sequence[str],
                cell_label_keys: Sequence[str],
                niche_label_keys: Sequence[str],
                niche_mode: str = "weighted",
                verbose: bool = True) -> List[dict]:
    """All pairs for one predicted_adata, on both the native and common cell sets."""
    rows: List[dict] = []

    cell_views = _flatten_codes(adata, CELL_CODE_KEY)
    niche_views = _flatten_codes(adata, NICHE_CODE_KEY)
    if not cell_views and not niche_views:
        print(f"  [{seed}] no code indices in obs/obsm; skipping", file=sys.stderr)
        return rows

    present_cell = [k for k in cell_label_keys if k in adata.obs.columns]
    present_niche = [k for k in niche_label_keys if k in adata.obs.columns]
    if not present_cell or not present_niche:
        print(f"  [{seed}] need >=1 cell-type and >=1 niche label column "
              f"(found cell={present_cell}, niche={present_niche}); skipping",
              file=sys.stderr)
        return rows

    labels = {k: _clean_labels(adata, k) for k in set(present_cell) | set(present_niche)}

    if niche_mode == "merged" and len(present_niche) > 1:
        labels[NICHE_AGG_KEY] = _merge_niche_columns(labels, present_niche)
        niche_cols: List[str] = [NICHE_AGG_KEY]
        if verbose:
            print(f"  [{seed}] merged niche columns {present_niche} -> "
                  f"single '{NICHE_AGG_KEY}' vector")
    else:
        niche_cols = list(present_niche)

    valid = {k: np.array([v is not None for v in labels[k]]) for k in labels}
    cell_primary = present_cell[0]

    for split in splits:
        smask = _split_mask(adata, split)
        if smask.sum() == 0:
            continue

        # Cells usable for BOTH axes: valid cell type AND valid niche (any column).
        niche_any = np.zeros(adata.n_obs, dtype=bool)
        for k in niche_cols:
            niche_any |= valid[k]
        common = smask & valid[cell_primary] & niche_any

        for cellset, base in (("native", smask), ("common", common)):
            if base.sum() == 0:
                continue

            # ---- (1) code x label ---------------------------------------
            for views, branch in ((cell_views, "cell"), (niche_views, "niche")):
                for codes, cname in views:
                    for lkey in present_cell + niche_cols:
                        r = _pair_row(seed, split, cellset, "code", cname,
                                      "label", lkey, codes, labels[lkey],
                                      base & valid[lkey])
                        if r:
                            r["branch"] = branch
                            rows.append(r)

            # ---- (2) label x label: THE REFERENCE CEILING ----------------
            for ckey in present_cell:
                for nkey in niche_cols:
                    if nkey == ckey:
                        continue
                    r = _pair_row(seed, split, cellset, "label", ckey,
                                  "label", nkey, labels[ckey], labels[nkey],
                                  base & valid[ckey] & valid[nkey])
                    if r:
                        r["branch"] = "reference"
                        rows.append(r)

            # ---- (3) code x code: direct codebook overlap ----------------
            for (c_codes, c_name) in cell_views:
                for (n_codes, n_name) in niche_views:
                    # compare like with like (level_0 vs level_0, etc.)
                    if c_name.split("[")[-1] != n_name.split("[")[-1]:
                        continue
                    r = _pair_row(seed, split, cellset, "code", c_name,
                                  "code", n_name, c_codes, n_codes, base)
                    if r:
                        r["branch"] = "code-code"
                        rows.append(r)

    rows.extend(_weighted_niche_aggregate(rows, niche_cols))

    if verbose:
        for r in rows:
            if r["split"] != "all" or r["cellset"] != "common":
                continue
            print(f"  [{seed}] {r['x_key']:<38s} vs {r['y_key']:<28s} "
                  f"NMI={r['NMI']:.4f} ARI={r['ARI']:.4f} AMI={r['AMI']:.4f} "
                  f"(n={r['n_cells']})")
    return rows


# --------------------------------------------------------------------------
# Run discovery (mirrors diagnose_coupling.py)
# --------------------------------------------------------------------------
def _read_run_dir_col(csv_path: str) -> List[str]:
    if pd is None:
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    col = next((c for c in ("run_dir", "run_directory", "rundir", "dir")
                if c in df.columns), None)
    if col is None:
        return []
    return [str(x) for x in df[col].dropna().tolist()]


def _latest_ts_dir(parent: str) -> Optional[str]:
    subs = [p for p in glob.glob(os.path.join(parent, "*")) if os.path.isdir(p)]
    if not subs:
        return None
    return sorted(subs, key=lambda p: os.path.basename(p))[-1]


def _resolve_adata_paths(args) -> List[Tuple[str, str]]:
    """Return [(label, predicted_adata_path)]; multiseed -> one entry per seed."""
    if args.predicted_adata:
        return [("single", args.predicted_adata)]
    if args.run_dir:
        return [("single", os.path.join(args.run_dir, "predicted_adata.h5ad"))]

    base = os.path.join(args.artifacts_root, args.dataset)
    ms = sorted(d for d in glob.glob(os.path.join(base, f"{args.variant}*__multiseed"))
                if os.path.isdir(d))
    if ms:
        sweep = ms[0]
        ts = (os.path.join(sweep, args.timestamp)
              if args.timestamp and args.timestamp != "latest"
              else _latest_ts_dir(sweep))
        if ts and os.path.isfile(os.path.join(ts, "seed_run_index.csv")):
            out = []
            for i, rd in enumerate(_read_run_dir_col(
                    os.path.join(ts, "seed_run_index.csv"))):
                p = os.path.join(rd, "predicted_adata.h5ad")
                if os.path.isfile(p):
                    out.append((f"seed{i}", p))
            if out:
                return out
            print(f"[overlap] multiseed index found but no predicted_adata.h5ad "
                  f"under its run_dirs ({ts}); falling back.", file=sys.stderr)

    var_dirs = sorted(d for d in glob.glob(os.path.join(base, f"{args.variant}*"))
                      if os.path.isdir(d) and not d.endswith("__multiseed"))
    if not var_dirs:
        raise SystemExit(f"[overlap] no run dirs matching {args.variant!r} under {base}")
    out = []
    for d in sorted(x for x in glob.glob(os.path.join(var_dirs[0], "*"))
                    if os.path.isdir(x)):
        p = os.path.join(d, "predicted_adata.h5ad")
        if os.path.isfile(p):
            out.append((os.path.basename(d), p))
    if not out:
        raise SystemExit(f"[overlap] no predicted_adata.h5ad under {var_dirs[0]}")
    return out


def _load(path: str):
    """Load obs/obsm only where possible (backed mode avoids pulling X over NFS)."""
    try:
        return ad.read_h5ad(path, backed="r")
    except Exception:
        return ad.read_h5ad(path)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _print_matrix(summary, metric: str, cell_label: str,
                  split: str = "all", cellset: str = "common") -> None:
    """Print the specialization matrix + gaps for one metric on one cell set."""
    def get(xk, yk):
        m = summary[(summary["split"] == split) & (summary["cellset"] == cellset)
                    & (summary["x_key"] == xk) & (summary["y_key"] == yk)]
        if m.empty:
            return None
        r = m.iloc[0]
        sd = r.get(f"{metric}_sd", float("nan"))
        return (r[f"{metric}_mean"], (sd if sd == sd else None), r.get("n_cells"))

    def fmt(v):
        if v is None:
            return "n/a"
        mean, sd, _ = v
        return f"{mean:.3f}+/-{sd:.3f}" if sd is not None else f"{mean:.3f}"

    cc = "cell_code_indices[level_0]"
    nc = "neighborhood_code_indices[level_0]"
    print(f"\n--- {metric} (split={split}, cellset={cellset}) ---")
    print(f"{'':<26}{'vs ' + cell_label:>20}{'vs niche':>20}")
    for label, key in (("cell codes (L1)", cc), ("niche codes (L1)", nc)):
        print(f"{label:<26}{fmt(get(key, cell_label)):>20}{fmt(get(key, NICHE_AGG_KEY)):>20}")
    ceil = get(cell_label, NICHE_AGG_KEY)
    print(f"{'REFERENCE: annotations':<26}{'--':>20}{fmt(ceil):>20}"
          f"   <- ceiling: what the two expert annotations share")
    print(f"{'code-code overlap':<26}{fmt(get(cc, nc)):>20}")

    ns = {v[2] for v in (get(cc, cell_label), get(nc, cell_label),
                         get(cc, NICHE_AGG_KEY), get(nc, NICHE_AGG_KEY), ceil)
          if v is not None and v[2] is not None}
    if ns:
        same = "IDENTICAL" if len(ns) == 1 else "DIFFER"
        print(f"  n_cells across matrix entries: "
              f"{sorted(int(n) for n in ns)}  -> {same}")

    d_cell = (get(cc, cell_label), get(nc, cell_label))
    d_niche = (get(nc, NICHE_AGG_KEY), get(cc, NICHE_AGG_KEY))
    if all(v is not None for v in d_cell):
        print(f"  -> {cell_label} axis: cell codes lead niche codes by "
              f"{d_cell[0][0] - d_cell[1][0]:+.3f}")
    if all(v is not None for v in d_niche):
        print(f"  -> niche axis    : niche codes lead cell codes by "
              f"{d_niche[0][0] - d_niche[1][0]:+.3f}")
    if ceil is not None:
        for nm, v in (("niche codes vs " + cell_label, get(nc, cell_label)),
                      ("cell codes vs niche", get(cc, NICHE_AGG_KEY))):
            if v is not None:
                rel = "BELOW" if v[0] < ceil[0] else "ABOVE"
                print(f"  -> cross term ({nm}) is {rel} the annotation ceiling "
                      f"({v[0]:.3f} vs {ceil[0]:.3f})")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Codebook specialization matrix (code x label, label x label "
                    "ceiling, code x code) from saved predicted_adata.h5ad.")
    ap.add_argument("--variant", default=DEFAULT_VARIANT_PREFIX)
    ap.add_argument("--timestamp", default="latest")
    ap.add_argument("--predicted-adata", default=None,
                    help="Score exactly this file instead of discovering runs.")
    ap.add_argument("--run-dir", default=None,
                    help="Run directory containing predicted_adata.h5ad.")
    ap.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--splits", default="all",
                    help="Comma-separated splits (default 'all'; 'all,train' "
                         "mirrors the pipeline's two rows).")
    ap.add_argument("--cellset", default="common", choices=["common", "native"],
                    help="Cell set for the PRINTED matrix. 'common' (default) "
                         "puts every entry on the same cells; 'native' "
                         "reproduces the pipeline CSV. Both are always written.")
    ap.add_argument("--niche-mode", default="weighted", choices=["weighted", "merged"],
                    help="How to form the single niche quantity from a dataset's "
                         "region columns. 'weighted' (default) = cell-count-"
                         "weighted average, identical to the paper. 'merged' = "
                         "one concatenated label vector (NB: also encodes section "
                         "identity when sections use disjoint vocabularies).")
    ap.add_argument("--cell-label-keys", default=",".join(DEFAULT_CELL_LABEL_KEYS))
    ap.add_argument("--niche-label-keys", default=",".join(DEFAULT_NICHE_LABEL_KEYS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    missing = [n for n, m in (("pandas", pd), ("anndata", ad),
                              ("scikit-learn", normalized_mutual_info_score))
               if m is None]
    if missing:
        raise SystemExit(f"[overlap] missing required package(s): {', '.join(missing)}. "
                         f"Activate the squint venv "
                         f"(source /nfs/team361/sb75/.venvs/squint/bin/activate).")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    cell_keys = [s.strip() for s in args.cell_label_keys.split(",") if s.strip()]
    niche_keys = [s.strip() for s in args.niche_label_keys.split(",") if s.strip()]

    paths = _resolve_adata_paths(args)
    print(f"[overlap] {len(paths)} run(s) to score")

    all_rows: List[dict] = []
    for label, path in paths:
        print(f"[overlap] {label}: {path}")
        adata = _load(path)
        all_rows.extend(analyse_one(adata, label, splits, cell_keys, niche_keys,
                                    niche_mode=args.niche_mode))
        try:
            adata.file.close()
        except Exception:
            pass

    if not all_rows:
        raise SystemExit("[overlap] no pairs scored -- check label keys / code keys.")

    per_seed = pd.DataFrame(all_rows)

    grp_cols = ["split", "cellset", "x_kind", "x_key", "y_kind", "y_key"]
    agg = {m: ["mean", "std"] for m in ("NMI", "ARI", "AMI")}
    agg["n_cells"] = ["mean"]
    agg["seed"] = ["count"]
    summary = per_seed.groupby(grp_cols, dropna=False).agg(agg).reset_index()
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c
                       for c in summary.columns]
    summary = summary.rename(columns={"seed_count": "n_seeds",
                                      "n_cells_mean": "n_cells"})

    out_dir = args.out
    if out_dir is None:
        if args.predicted_adata or args.run_dir:
            out_dir = os.path.join(os.getcwd(), "codebook_overlap")
        else:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(paths[0][1])),
                                   "codebook_overlap")
    os.makedirs(out_dir, exist_ok=True)
    per_seed.to_csv(os.path.join(out_dir, "codebook_overlap_per_seed.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "codebook_overlap_summary.csv"), index=False)

    present_cell = [k for k in cell_keys
                    if ((summary["x_key"] == k) | (summary["y_key"] == k)).any()]
    cell_label = present_cell[0] if present_cell else "cell_type"

    print("\n" + "=" * 78)
    print("CODEBOOK SPECIALIZATION MATRIX  "
          f"({len(paths)} seed(s), variant={args.variant}, dataset={args.dataset})")
    print("=" * 78)
    for metric in ("NMI", "ARI", "AMI"):
        _print_matrix(summary, metric, cell_label, cellset=args.cellset)

    print(f"\n[overlap] wrote {out_dir}/codebook_overlap_{{per_seed,summary}}.csv")
    print("[overlap] SANITY CHECK: the cellset='native' code-vs-label rows must "
          "match this run's metrics/niche_identification_metrics.csv exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
