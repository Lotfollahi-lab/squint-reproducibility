"""
scVI-on-X_nbr baseline (NBR-level Pearson) for the gene-expression
imputation benchmark with the SQUINT region-holdout split.

Stock scVI, but trained on NEIGHBORHOOD-AGGREGATED counts instead of
per-cell counts. Concretely:

  working_adata.X = adata.layers["X_nbr"]   (1-hop spatial MEAN, FLOAT —
                                             same target SQUINT's nbr
                                             branch trains against)

scVI's input/output is then `X_nbr` rather than `X`. Its NB decoder
learns a niche-aware reconstruction (the latent ends up niche-aware
*because the input is niche-aggregated*, not because of any custom
spatial encoder). This is the cleanest "did SQUINT's dual-VQ
architecture help over a plain-vanilla generative model with the same
target?" comparator: held architecture/loss/training procedure fixed
(scVI), only the input differs.

X_nbr is computed via `aggregate_1hop_neighbor_features(...,
return_mean=True)` — bit-for-bit identical to SQUINT's niche-target
construction (see `vqniche.utils.loss_utils`). The float values are
passed unrounded to scVI; NB log-likelihood extends continuously to
non-negative reals via `lgamma`, so there is no need to discretise.

Trained on `data_split == "train"` cells; inference produces predicted
neighborhood expression on the FULL adata. The predicted layer is
stored as `X_hat_nbr` (niche branch — same key SQUINT uses).

Output layout:
  <out_dir>/predicted_adata.h5ad             (X_hat_nbr layer + X_nbr target + data_split)
  <out_dir>/metrics/per_seed_pearson_reconstruction.csv
  <out_dir>/metrics/pearson_reconstruction_metrics.csv
  <out_dir>/user_specified_config.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from _holdout_utils import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_DATASET_TAG,
    DEFAULT_HOLDOUT_REGIONS,
    DEFAULT_SILVER_DIR,
    apply_holdout_regions,
    build_pearson_dataframe,
    compute_X_nbr,
    load_silver_concat,
    spatial_knn_per_batch,
    write_pearson_outputs,
)


DEFAULT_VARIANT_TAG = "baseline-scvi-nbr+region-holdout"


# ---------------------------------------------------------------------------
# scVI training + inference (on X_nbr targets)
# ---------------------------------------------------------------------------

def train_one_seed(
        adata_full: ad.AnnData,
        seed: int,
        batch_key: str,
        max_epochs: Optional[int],
        n_latent: int,
        accelerator: str,
    ) -> ad.AnnData:
    """Train scVI on `adata_full[data_split == "train"]` with X_nbr as
    BOTH the input and the target, then predict per-cell reconstructed
    nbr expression on the FULL adata. Writes
    `adata_full.layers["X_hat_nbr"]` in place and returns the same adata.

    SQUINT-matched target: the input/target is the FLOAT 1-hop spatial
    mean of the per-cell counts (same `aggregate_1hop_neighbor_features
    (..., return_mean=True)` SQUINT uses for its niche branch — see
    `vqniche.utils.loss_utils.aggregate_1hop_neighbor_features`). The
    NB likelihood handles real-valued non-negative targets via
    continuous `lgamma` extension, so no rounding is needed; we just
    pass the float through. scvi-tools warns about non-integer inputs
    but proceeds. Suppress the warning by relying on `setup_anndata`'s
    permissive mode; if your scvi version raises instead, the offending
    check is `scvi.data._utils._check_nonnegative_integers` and can be
    monkey-patched to no-op in this script.
    """
    import scvi
    import torch
    import warnings

    scvi.settings.seed = int(seed)
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # X_nbr must be present already (caller computed it via compute_X_nbr).
    if "X_nbr" not in adata_full.layers:
        raise SystemExit(
            "adata.layers['X_nbr'] missing — call compute_X_nbr() first."
        )
    X_nbr = adata_full.layers["X_nbr"]
    if sp.issparse(X_nbr):
        X_nbr = X_nbr.toarray()
    X_nbr = np.maximum(np.asarray(X_nbr, dtype=np.float32), 0.0)

    train_mask = (adata_full.obs["data_split"].to_numpy() == "train")
    n_train = int(train_mask.sum())
    n_test  = int(adata_full.n_obs - n_train)
    print(f"  train cells: {n_train}   held-out cells: {n_test}")

    # Build a working AnnData where .X is the FLOAT X_nbr — same target
    # SQUINT trains its niche branch against. scVI's NB likelihood
    # extends continuously to non-integer targets via lgamma; the int
    # check inside scvi-tools is a soft warning, not an error.
    working = ad.AnnData(
        X=X_nbr,
        obs=adata_full.obs.copy(),
        var=adata_full.var.copy(),
        obsm=dict(adata_full.obsm),
    )
    working.layers["counts"] = X_nbr.copy()
    train_working = working[train_mask].copy()

    print(f"  setup_anndata(layer='counts', batch_key={batch_key!r})  "
          "[scVI on nbr-aggregated input]")
    scvi.model.SCVI.setup_anndata(train_working, layer="counts",
                                   batch_key=batch_key)

    print(f"  training scVI (seed={seed}, n_latent={n_latent}, "
          f"max_epochs={max_epochs})...")
    t0 = time.time()
    vae = scvi.model.SCVI(train_working, n_latent=int(n_latent))
    vae.train(
        max_epochs=max_epochs,
        accelerator=accelerator,
        check_val_every_n_epoch=1,
    )
    print(f"  training took {time.time() - t0:.1f}s")

    # Inference on the FULL working adata (train + held-out). We pull
    # the length-1-library rate (softmax over genes) and multiply by
    # each cell's observed library size on the X_nbr scale to recover
    # the NB mean — same shape and scale as the X_nbr target.
    #
    # NOTE: older scvi-tools accepted `library_size="observed"` as a
    # shortcut for this; newer versions removed the special string
    # ("unsupported operand type(s) for *=: 'Tensor' and 'str'").
    # Doing the multiplication ourselves works on every version.
    print("  inference on full adata (train + held-out)...")
    full_working = working.copy()
    scvi.model.SCVI.setup_anndata(full_working, layer="counts",
                                   batch_key=batch_key)
    rate = vae.get_normalized_expression(
        adata=full_working,
        return_numpy=True,
    )
    rate = np.asarray(rate, dtype=np.float32)
    # Per-cell library size on the X_nbr scale (row-sum of the
    # neighborhood-aggregated input).
    lib = X_nbr.sum(axis=1).astype(np.float32)
    X_hat_nbr = rate * lib[:, None]
    if X_hat_nbr.shape != (adata_full.n_obs, adata_full.n_vars):
        raise RuntimeError(
            f"scVI X_hat_nbr shape {X_hat_nbr.shape} != "
            f"({adata_full.n_obs}, {adata_full.n_vars})"
        )

    adata_full.layers["X_hat_nbr"] = X_hat_nbr
    print(f"  X_hat_nbr populated (shape={X_hat_nbr.shape}, "
          f"min={X_hat_nbr.min():.3f}, mean={X_hat_nbr.mean():.3f}, "
          f"max={X_hat_nbr.max():.3f})")
    return adata_full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--silver-dir", type=str, default=DEFAULT_SILVER_DIR)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--variant-tag", type=str, default=DEFAULT_VARIANT_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--batch-key", type=str, default="adata_batch_id")
    p.add_argument("--n-spatial-neighs", type=int, default=10,
                   help="kNN graph size for X_nbr aggregation. Default 10 "
                        "(matches sibling nbr baselines).")
    p.add_argument("--use-default-holdout-regions", action="store_true", default=True)
    p.add_argument("--no-default-holdout-regions",
                   dest="use_default_holdout_regions", action="store_false")
    # scVI training hparams.
    p.add_argument("--scvi-max-epochs", type=int, default=None,
                   help="None = scvi-tools' own heuristic (~400 for ~100k cells).")
    p.add_argument("--scvi-n-latent", type=int, default=10)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--accelerator", type=str, default="auto")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds parsed to empty list.")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = (
            args.artifacts_root / args.dataset_tag / args.variant_tag / ts
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {args.out_dir}")
    print(f"Seeds   : {seeds}")

    # ---- 1. Load silver -------------------------------------------------
    print("\n=== Loading silver ===")
    adata = load_silver_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"Concatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from obs.")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    # ---- 2. Held-out region split ---------------------------------------
    print("\n=== Held-out region split ===")
    regions = DEFAULT_HOLDOUT_REGIONS if args.use_default_holdout_regions else None
    if regions is None:
        raise SystemExit("Custom regions not configured; pass --use-default-holdout-regions.")
    apply_holdout_regions(adata, batch_key=args.batch_key, regions=regions)

    # ---- 3. Spatial graph + X_nbr target --------------------------------
    print("\n=== Spatial graph + X_nbr target ===")
    spatial_knn_per_batch(adata, n_neighs=args.n_spatial_neighs,
                          batch_key=args.batch_key)
    compute_X_nbr(adata, normalize="mean")

    # ---- 4. Per-seed train + infer + Pearson ----------------------------
    per_seed_frames: List[pd.DataFrame] = []
    seed0_adata: Optional[ad.AnnData] = None
    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        adata_s = adata.copy()
        train_one_seed(
            adata_s,
            seed=seed,
            batch_key=args.batch_key,
            max_epochs=args.scvi_max_epochs,
            n_latent=args.scvi_n_latent,
            accelerator=args.accelerator,
        )
        df = build_pearson_dataframe(adata_s, seed=seed, log1p=True, n_hvg=50)
        per_seed_frames.append(df)
        if s_idx == 0:
            seed0_adata = adata_s

        for split in ("all", "train", "test"):
            row = df[
                (df["split"]       == split) &
                (df["branch"]      == "niche") &
                (df["axis"]        == "cell_wise") &
                (df["transform"]   == "log1p") &
                (df["gene_subset"] == "all")
            ]
            if not row.empty:
                v = float(row["pearson_mean"].iloc[0])
                print(f"  Pearson niche cell_wise log1p all (split={split:<5s}) = {v:.4f}")

    per_seed = (
        pd.concat(per_seed_frames, ignore_index=True)
        if per_seed_frames else pd.DataFrame()
    )

    # ---- 5. Write CSVs --------------------------------------------------
    print("\n=== Writing outputs ===")
    write_pearson_outputs(args.out_dir, per_seed)

    # ---- 6. Save seed-0 predicted adata --------------------------------
    if seed0_adata is not None:
        out_h5ad = args.out_dir / "predicted_adata.h5ad"
        sib = (Path(__file__).resolve().parent.parent
               / "cell_type_identification")
        if str(sib) not in sys.path:
            sys.path.insert(0, str(sib))
        try:
            from run_pca_leiden import _sanitize_for_h5ad  # type: ignore
            _sanitize_for_h5ad(seed0_adata)
        except Exception as exc:
            print(f"  (sanitizer not available: {exc})")
        seed0_adata.write_h5ad(out_h5ad)
        print(f"  -> {out_h5ad}  (seed[0] snapshot)")

    # ---- 7. Console summary --------------------------------------------
    print()
    print("=" * 78)
    print(f"SUMMARY (mean ± std across {len(seeds)} seeds, "
          "cell_wise × log1p × all)")
    print("=" * 78)
    canonical = per_seed[
        (per_seed["branch"]      == "niche") &
        (per_seed["axis"]        == "cell_wise") &
        (per_seed["transform"]   == "log1p") &
        (per_seed["gene_subset"] == "all")
    ]
    for split in ("all", "train", "test"):
        sub = canonical[canonical["split"] == split]
        if sub.empty: continue
        vals = sub["pearson_mean"].astype(float)
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        print(f"  niche Pearson (split={split:<5s}) = "
              f"{vals.mean():.4f} ± {std:.4f}  "
              f"(min={vals.min():.4f}, max={vals.max():.4f})")
    print(f"  (full {len(per_seed)}-row variant table written to "
          f"per_seed_pearson_reconstruction.csv)")

    # ---- 8. Stub config -------------------------------------------------
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "scVI on X_nbr targets (region-holdout)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "scVI-nbr"},
        "scvi_nbr": {
            "max_epochs":       args.scvi_max_epochs,
            "n_latent":         args.scvi_n_latent,
            "accelerator":      args.accelerator,
            "n_spatial_neighs": args.n_spatial_neighs,
            "batch_key":        args.batch_key,
            "seeds":            seeds,
            "holdout_regions":  DEFAULT_HOLDOUT_REGIONS,
        },
    }
    with open(args.out_dir / "user_specified_config.yaml", "w") as f:
        yaml.safe_dump(stub_cfg, f, sort_keys=False)
    print(f"\n  -> {args.out_dir / 'user_specified_config.yaml'}")

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  Run dir : {args.out_dir}")
    print(f"  Variant : {args.variant_tag}")
    print(f"  Seeds   : {len(seeds)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
