#!/usr/bin/env python3
"""
Recompute niche-identification (and cell-type) NMI/ARI for existing
baseline runs WITHOUT re-training.

Use case
--------
A baseline run completed but its `per_seed_niche_identification.csv`
is empty because the niche label-key wasn't in the runtime defaults
at the time. This was the case for spatch_*_1p before `spatial_cluster`
/ `annotation` were added to `DEFAULT_NICHE_LABEL_KEYS` /
`DEFAULT_CELL_LABEL_KEYS` in `cell_type_identification/run_pca_leiden.py`.
Instead of re-running every baseline (training + Leiden + UMAP + metrics),
this script:

  1. Iterates <artifacts_root>/<dataset_tag>/baseline-*/<latest_TS>/
  2. Loads `predicted_adata.h5ad` (carries seed[0]'s obs + Leiden /
     Novae cluster column).
  3. Recomputes NMI/ARI against the (now-correct) cell + niche label
     keys.
  4. Writes / appends:
       <run_dir>/seeds/seed_0/metrics/niche_identification_metrics.csv
       <run_dir>/metrics/per_seed_niche_identification.csv   (seed=0 rows
                                                              merged with any
                                                              non-empty existing
                                                              seeds 1..N rows)

Limitations
-----------
- Only seed[0] is recoverable from disk. Seeds 1..N-1 have their
  per-seed cluster assignments thrown away after each iteration in
  the runners (only seed[0] is persisted to `predicted_adata.h5ad`).
- For methods with a CHEAP-to-retrain Leiden step (BANKSY,
  neigh-expr-pca) you could rerun all seeds in ~1 minute; for deep
  methods (CellCharter, GraphST, Novae, NicheCompass) each seed
  needs a full model fit. The full re-run path is the one in
  `submit_niche_id_baselines.sh`; this script is the "I just want
  seed[0] numbers right now" shortcut.

Compatibility
-------------
- Cluster column auto-detection: tries `leiden` first (used by
  Banksy / CellCharter / GraphST / NicheCompass / neigh-expr-pca),
  then `novae_domains_*`, then any obs column matching `--cluster-col`.
- Label columns: `--cell-label-keys` / `--niche-label-keys` default
  to the spatch-aware lists from run_pca_leiden.py.

Usage
-----
  # Recompute seed[0] NMI for every baseline run on spatch_ov_1p:
  python analysis/benchmarking/recompute_niche_metrics.py \\
      --dataset-tag spatch_ov_1p

  # Limit to specific baselines:
  python analysis/benchmarking/recompute_niche_metrics.py \\
      --dataset-tag spatch_ov_1p \\
      --baselines baseline-banksy,baseline-cellcharter

  # Override label keys:
  python analysis/benchmarking/recompute_niche_metrics.py \\
      --dataset-tag spatch_ov_1p \\
      --cell-label-keys annotation \\
      --niche-label-keys spatial_cluster
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd


# Reuse the helpers from the existing runner so the recompute is
# bit-identical to what a fresh run would produce.
sys.path.insert(0, str(Path(__file__).resolve().parent / "cell_type_identification"))
from run_pca_leiden import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_CELL_LABEL_KEYS,
    DEFAULT_NICHE_LABEL_KEYS,
    _compute_niche_identification,
    _import_metric_helpers,
)


def _detect_cluster_col(adata: ad.AnnData, explicit: Optional[str] = None) -> str:
    """Return the obs column that holds per-cell cluster assignments.

    Preference: explicit override > `leiden` > `novae_domains_<N>` (any
    matching column; pick the highest N). Errors with a helpful column
    listing if nothing matches.
    """
    if explicit:
        if explicit not in adata.obs.columns:
            raise SystemExit(
                f"--cluster-col {explicit!r} not in obs. Available: "
                f"{sorted(adata.obs.columns.tolist())}"
            )
        return explicit
    if "leiden" in adata.obs.columns:
        return "leiden"
    novae_cols = sorted(
        (c for c in adata.obs.columns if re.match(r"^novae_domains_\d+$", c)),
        key=lambda c: int(c.split("_")[-1]),
        reverse=True,
    )
    if novae_cols:
        return novae_cols[0]
    raise SystemExit(
        f"No cluster column found in obs. Tried 'leiden' and "
        f"'novae_domains_<N>'. Available: "
        f"{sorted(adata.obs.columns.tolist())}. Pass --cluster-col."
    )


def _latest_ts_dir(variant_dir: Path) -> Optional[Path]:
    """Return the most recent timestamp subdir under `variant_dir`
    that has a `predicted_adata.h5ad`."""
    if not variant_dir.is_dir():
        return None
    ts_dirs = sorted(
        (p for p in variant_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    for ts in ts_dirs:
        if (ts / "predicted_adata.h5ad").is_file():
            return ts
    return None


def _recompute_for_run(
        run_dir: Path,
        cell_keys: List[str],
        niche_keys: List[str],
        cluster_col_override: Optional[str],
    ) -> Tuple[int, pd.DataFrame]:
    """Recompute seed[0] NMI/ARI for one baseline run dir. Returns
    `(n_rows_written, niche_df)`. Writes the per-seed CSV under
    `<run_dir>/seeds/seed_0/metrics/niche_identification_metrics.csv`.
    """
    adata_path = run_dir / "predicted_adata.h5ad"
    if not adata_path.is_file():
        print(f"  [{run_dir.name}] no predicted_adata.h5ad, skipping.")
        return 0, pd.DataFrame()
    adata = ad.read_h5ad(adata_path)
    cluster_col = _detect_cluster_col(adata, cluster_col_override)
    print(f"  cluster column: {cluster_col!r}")
    print(f"  obs columns:    {sorted(adata.obs.columns.tolist())}")

    # Drop label-key candidates absent from obs up-front so the
    # downstream call only reports successes (the helper would skip
    # missing ones anyway, but pre-filtering keeps logs clean).
    cell_keys_present  = [k for k in cell_keys  if k in adata.obs.columns]
    niche_keys_present = [k for k in niche_keys if k in adata.obs.columns]
    if not cell_keys_present and not niche_keys_present:
        print(f"  none of --cell-label-keys / --niche-label-keys are in "
              f"this run's obs; nothing to recompute.")
        return 0, pd.DataFrame()
    print(f"  cell  keys present: {cell_keys_present}")
    print(f"  niche keys present: {niche_keys_present}")

    compute_nmi_ari, _, _ = _import_metric_helpers()
    niche_df = _compute_niche_identification(
        adata=adata,
        leiden_key=cluster_col,
        cell_label_keys=cell_keys_present,
        niche_label_keys=niche_keys_present,
        compute_nmi_ari=compute_nmi_ari,
    )
    if niche_df.empty:
        print(f"  recompute produced 0 rows.")
        return 0, niche_df

    # Write per-seed CSV.
    seed0_metrics = run_dir / "seeds" / "seed_0" / "metrics"
    seed0_metrics.mkdir(parents=True, exist_ok=True)
    per_seed_csv = seed0_metrics / "niche_identification_metrics.csv"
    out = niche_df.drop(columns=["seed"], errors="ignore")
    out.to_csv(per_seed_csv, index=False)
    print(f"  -> {per_seed_csv}  ({len(out)} rows)")
    return len(out), niche_df


def _rebuild_top_level_per_seed(
        run_dir: Path,
        seed0_df: pd.DataFrame,
    ) -> None:
    """Rebuild `<run_dir>/metrics/per_seed_niche_identification.csv`:
    keep any non-seed-0 rows that already exist (multi-seed runs), then
    splice in the freshly-computed seed=0 rows, replacing any prior
    seed=0 rows that came from the now-stale defaults.
    """
    if seed0_df.empty:
        return
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    top_csv = metrics_dir / "per_seed_niche_identification.csv"

    new_seed0 = seed0_df.copy()
    if "seed" not in new_seed0.columns:
        new_seed0.insert(0, "seed", 0)

    if top_csv.is_file():
        existing = pd.read_csv(top_csv)
        if "seed" in existing.columns:
            existing_non_seed0 = existing[existing["seed"] != 0].copy()
        else:
            existing_non_seed0 = pd.DataFrame()  # no seed column -> all stale
    else:
        existing_non_seed0 = pd.DataFrame()

    out = pd.concat([new_seed0, existing_non_seed0], ignore_index=True)
    out.to_csv(top_csv, index=False)
    print(f"  -> {top_csv}  ({len(out)} rows total; "
          f"{len(new_seed0)} fresh seed=0, "
          f"{len(existing_non_seed0)} preserved from seeds >0)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--artifacts-root", type=Path,
                   default=DEFAULT_ARTIFACTS_ROOT,
                   help=f"Default: {DEFAULT_ARTIFACTS_ROOT}")
    p.add_argument("--dataset-tag", type=str, required=True,
                   help="e.g. spatch_ov_1p")
    p.add_argument("--baselines", type=str, default="",
                   help="Comma-separated subset of baseline dir names "
                        "to process (e.g. baseline-banksy,baseline-novae). "
                        "Default: every baseline-* directory under "
                        "<artifacts_root>/<dataset_tag>/ that has a "
                        "predicted_adata.h5ad.")
    p.add_argument("--cell-label-keys", type=str,
                   default=",".join(DEFAULT_CELL_LABEL_KEYS),
                   help=f"Default: {DEFAULT_CELL_LABEL_KEYS}")
    p.add_argument("--niche-label-keys", type=str,
                   default=",".join(DEFAULT_NICHE_LABEL_KEYS),
                   help=f"Default: {DEFAULT_NICHE_LABEL_KEYS}")
    p.add_argument("--cluster-col", type=str, default=None,
                   help="Override the cluster obs-column (default: "
                        "auto-detect 'leiden' or 'novae_domains_<N>').")
    args = p.parse_args()

    cell_keys  = [k.strip() for k in args.cell_label_keys.split(",")  if k.strip()]
    niche_keys = [k.strip() for k in args.niche_label_keys.split(",") if k.strip()]
    dataset_root = args.artifacts_root / args.dataset_tag
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    baselines_filter = {b.strip() for b in args.baselines.split(",") if b.strip()}
    candidate_dirs = sorted(
        p for p in dataset_root.iterdir()
        if p.is_dir() and p.name.startswith("baseline-")
    )
    if baselines_filter:
        candidate_dirs = [p for p in candidate_dirs if p.name in baselines_filter]
    if not candidate_dirs:
        raise SystemExit(
            f"No matching baseline-* dirs under {dataset_root} "
            f"(filter: {sorted(baselines_filter) or 'all'})."
        )

    print(f"Dataset tag:     {args.dataset_tag}")
    print(f"Artifacts root:  {args.artifacts_root}")
    print(f"Baselines:       {[p.name for p in candidate_dirs]}")
    print(f"Cell  label keys: {cell_keys}")
    print(f"Niche label keys: {niche_keys}")
    print()

    total_written = 0
    for variant_dir in candidate_dirs:
        run_dir = _latest_ts_dir(variant_dir)
        print(f"--- {variant_dir.name}")
        if run_dir is None:
            print(f"  no <TS>/predicted_adata.h5ad found; skipping.")
            continue
        print(f"  run dir: {run_dir}")
        n_rows, niche_df = _recompute_for_run(
            run_dir=run_dir,
            cell_keys=cell_keys,
            niche_keys=niche_keys,
            cluster_col_override=args.cluster_col,
        )
        if n_rows > 0:
            _rebuild_top_level_per_seed(run_dir=run_dir, seed0_df=niche_df)
            total_written += n_rows
        print()

    print(f"DONE. Wrote {total_written} NMI/ARI rows across "
          f"{len(candidate_dirs)} baselines.")


if __name__ == "__main__":
    sys.exit(main())
