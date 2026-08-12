#!/usr/bin/env python
"""
compute_basw.py — batch ASW only, for R2-W1b, on every representation.
=============================================================================
R2-W1b asked for label-conditioned integration metrics. scib_metrics.silhouette_batch
is one: verified in _silhouette.py, it loops over cell types, subsets X to each, and
computes the silhouette with respect to BATCH inside that one cell type, then averages
the per-cell-type means. Its docstring: "ASW with respect to batch ids within each
label". A metric measuring only global batch mixing would not take `labels` at all.

WHY THIS EXISTS SEPARATELY FROM compute_label_conditioned_metrics.py
That script also computes kbet, clisi, casw, ilisi and graph_conn, all of which need
a 90-NN and a 50-NN graph, and kbet additionally runs a diffusion map per cell type.
That is the slow part, and it is what pushed the 199k-cell NSCLC jobs into
TERM_MEMLIMIT at 128 GB. basw needs NO graph: it reads obsm directly. So this runs in
a fraction of the time and memory, which is the point when the only question is
R2-W1b.

It also scores ALL SIX SQUINT representations in one job, since each is cheap:
  cell_code_indices / neighborhood_code_indices   the integer codes (Table 1's iLISI)
  cell_emb / neighborhood_emb                     the quantized vectors those index
  cell_latent / neighborhood_latent               pre-quantization, continuous
This matters because kbet is undefined on the discrete ones (94% of neighbour
distances are exactly 0, so its diffusion step hits scib's `score = 0` fallback for
19 of 21 eczema labels) whereas basw is defined on all six. So basw is what lets the
response answer R2-W1b on the paper's OWN discrete representation.

ONE CAVEAT, recorded so nobody has to rediscover it. scIB scores basw as
mean(1 - |silhouette|), so two cells at an identical point give a silhouette of 0,
which reads as PERFECT mixing. A codebook therefore has a mechanical pull toward 1.
Measured on eczema seed 0 that pull does not dominate: the discrete encodings score
BELOW the continuous latent (0.823 and 0.872 against 0.933), the opposite of what
tie-inflation alone would give. Usable on codes, but not tie-proof.

n_labels_scored IS PART OF THE OUTPUT, because basw averages only over cell types
that contain more than one batch. Where no cell type spans batches, scib's loop
appends nothing and pd.concat([]) raises "No objects to concatenate": that is the
mouse brain, whose 49 cell types are two disjoint per-section vocabularies. The
error is recorded per row and basw left NaN, rather than being reported as a score.

USAGE
  python compute_basw.py --method SQUINT --latent-keys cell_emb,cell_latent \\
      --cell-type-key new_annotation --adata .../seed0/predicted_adata.h5ad \\
      --out .../basw_xhs1000-3b_1p_SQUINT.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# Same directory; reused so the h5py reader and its uns/log1p workaround live in
# exactly one place.
from compute_label_conditioned_metrics import _read_elem, load_minimal


def scored_labels(labels: np.ndarray, batches: np.ndarray):
    """
    Replicate scib's own skip rule so the csv can record how many cell types the
    score is an average over. _silhouette.py drops a label when it holds a single
    batch, or as many batches as cells.
    """
    used, skipped = [], []
    for g in np.unique(labels):
        m = labels == g
        nb = len(np.unique(batches[m]))
        (used if (nb > 1 and nb != int(m.sum())) else skipped).append(str(g))
    return used, skipped


def available_obsm(path: Path):
    import h5py
    with h5py.File(path, "r") as h:
        return list(h["obsm"].keys()) if "obsm" in h else []


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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--adata", type=Path, action="append", required=True,
                   help="Repeat once per seed.")
    p.add_argument("--method", required=True)
    p.add_argument("--latent-keys", required=True,
                   help="Comma list. Keys absent from a file are recorded as an "
                        "error row rather than aborting the run.")
    p.add_argument("--cell-type-key", default="cell_type")
    p.add_argument("--batch-key", default="adata_batch_id")
    p.add_argument("--drop-label-nan", action="store_true",
                   help="Drop cells with no cell-type label. Needed on CosMx NSCLC "
                        "(8,980 NaN in cell_type).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    if a.out.exists() and not a.force:
        raise SystemExit(f"{a.out} exists. Use --force or a different --out.")

    import pandas as pd
    try:
        import scib_metrics
    except ImportError as ex:                                      # noqa: BLE001
        raise SystemExit(f"cannot import scib_metrics ({ex}). Activate the squint "
                         f"venv.")
    _read_elem()          # fail early if anndata's reader is not importable
    want = [s.strip() for s in a.latent_keys.split(",") if s.strip()]

    rows = []
    for f in a.adata:
        have = available_obsm(f)
        keys = [k for k in want if k in have]
        missing = [k for k in want if k not in have]
        base = {"method": a.method, "dataset": dataset_of(f), "seed": seed_of(f),
                "cell_type_key": a.cell_type_key, "batch_key": a.batch_key,
                "scib_version": scib_metrics.__version__, "path": str(f)}
        for k in missing:
            print(f"  {f.name}: obsm {k!r} absent (present: {have})")
            rows.append({**base, "latent_key": k, "basw": float("nan"),
                         "error": f"obsm key {k!r} not in file"})
        if not keys:
            continue

        adata = load_minimal(f, keys, (a.cell_type_key, a.batch_key))
        n_dropped = 0
        nan_mask = adata.obs[a.cell_type_key].isna()
        if nan_mask.any():
            if not a.drop_label_nan:
                raise SystemExit(
                    f"{f}\n  {a.cell_type_key!r} has {int(nan_mask.sum())} NaN. "
                    f"Pass --drop-label-nan to score the labelled subset.")
            n_dropped = int(nan_mask.sum())
            adata = adata[~nan_mask.to_numpy()].copy()
            print(f"  --drop-label-nan: dropped {n_dropped}; {adata.n_obs} remain")
        if adata.obs[a.batch_key].isna().any():
            raise SystemExit(f"{f}\n  {a.batch_key!r} has NaN; a cell with no batch "
                             f"cannot be placed.")

        labels = np.asarray(adata.obs[a.cell_type_key].astype(str))
        batches = np.asarray(adata.obs[a.batch_key].astype(str))
        used, skipped = scored_labels(labels, batches)
        print(f"\n--- {a.method} | seed {base['seed']} | {adata.n_obs} cells | "
              f"{len(used)}/{len(used) + len(skipped)} cell types scorable ---")
        if skipped:
            print(f"    skipped (single batch): {', '.join(skipped[:8])}"
                  + (f" ... (+{len(skipped) - 8})" if len(skipped) > 8 else ""))
        if not used:
            print("    NO cell type spans batches, so basw is undefined here. This "
                  "is\n    the mouse-brain case: cell_type is two disjoint "
                  "per-section vocabularies.")

        for k in keys:
            row = {**base, "latent_key": k, "n_cells": int(adata.n_obs),
                   "n_dropped_no_label": n_dropped,
                   "n_labels_total": len(used) + len(skipped),
                   "n_labels_scored": len(used), "n_batches": len(set(batches)),
                   "basw": float("nan"), "error": ""}
            try:
                X = np.asarray(adata.obsm[k], dtype=np.float64)
                row["basw"] = float(scib_metrics.silhouette_batch(
                    X, labels=labels, batch=batches))
                print(f"    {k:28s} basw = {row['basw']:.4f}")
            except Exception as ex:                                # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {ex}"
                print(f"    {k:28s} FAILED -> {row['error'][:120]}")
            rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) and "latent_key" in df:
        print("\n" + "=" * 74 + "\nMEAN +/- SD ACROSS SEEDS\n" + "=" * 74)
        for k, g in df.groupby("latent_key"):
            ok = g["basw"].dropna()
            cell = (f"{ok.mean():.4f} +/- {ok.std(ddof=1):.4f}" if len(ok) > 1
                    else f"{ok.mean():.4f}" if len(ok) == 1 else "n/a")
            print(f"  {k:30s}{cell:>22s}   ({len(ok)}/{len(g)} seeds)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
