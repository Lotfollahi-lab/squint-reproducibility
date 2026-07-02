"""
MLP-regression baseline for the gene-expression IMPUTATION benchmark
(SQUINT region-holdout split).

This is a faithful re-implementation of GeST's own "MLP" imputation baseline
(Hao et al., MLCB 2025, bioRxiv 2025.04.09.648072): a plain multilayer
perceptron that maps a cell's SPATIAL COORDINATES to its gene-expression
profile, "trained on cells' absolute spatial coordinates and gene expressions
from the uncropped areas" — i.e. a learned spatial interpolator. The held-out
region is predicted purely from its coordinates (the model never sees held-out
cells' expression). Coordinates are z-scored PER SECTION using the TRAIN cells'
statistics, so multiple sections share one MLP on a common scale.

It is the parametric counterpart to run_knn_spatial.py (the non-parametric
neighbor-averaging floor) and the natural learned comparator for GeST /
SQUINT-stage-2 on this task.

Mirrors run_scvi.py exactly except for the model: the shared `_holdout_utils`
handles silver loading, the held-out split, the neighborhood layers, the full
metric panel (Pearson / Spearman / RMSE / AUROC), and the CSVs.

Output (per the harness):
  <out_dir>/predicted_adata_seed{N}.h5ad
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
    add_neighborhood_layers,
    apply_holdout_regions,
    build_pearson_dataframe,
    load_silver_concat,
    write_pearson_outputs, write_predicted_adata,
)


DEFAULT_VARIANT_TAG = "baseline-mlp-spatial+region-holdout"


# ---------------------------------------------------------------------------
# Feature construction + MLP train/predict
# ---------------------------------------------------------------------------

def _coord_features(
        coords: np.ndarray,
        batch: np.ndarray,
        is_train: np.ndarray,
    ) -> np.ndarray:
    """Per-section z-scored (x, y) coordinates. Mean/std are computed from the
    section's TRAIN cells only (so the held-out region uses train-derived
    normalization); falls back to all cells if a section has no train cells."""
    F = np.asarray(coords, dtype=np.float32).copy()
    for b in pd.unique(batch):
        sec = np.where(batch == b)[0]
        ref = sec[is_train[sec]]
        if ref.size == 0:
            ref = sec
        mu = F[ref].mean(axis=0)
        sd = F[ref].std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        F[sec] = (F[sec] - mu) / sd
    return F


def train_one_seed(
        adata_full: ad.AnnData,
        seed: int,
        batch_key: str,
        epochs: int,
        hidden: int,
        n_layers: int,
        lr: float,
        batch_size: int,
        accelerator: str,
        spatial_key: str = "spatial",
    ) -> ad.AnnData:
    """Train coords->log1p(expression) MLP on TRAIN cells, predict every cell,
    write `adata_full.layers['X_hat']` (count scale = expm1 of the prediction).
    Returns the same adata."""
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    if spatial_key not in adata_full.obsm:
        raise SystemExit(f"obsm[{spatial_key!r}] missing.")
    use_cuda = (accelerator in ("auto", "gpu")) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    coords = np.asarray(adata_full.obsm[spatial_key], dtype=np.float64)
    batch = adata_full.obs[batch_key].to_numpy()
    is_train = (adata_full.obs["data_split"].to_numpy() == "train")
    n_train = int(is_train.sum())
    print(f"  train cells: {n_train}   held-out cells: {adata_full.n_obs - n_train}"
          f"   device: {device}")

    F = _coord_features(coords, batch, is_train)                  # (n, 2), z-scored
    X = adata_full.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    T = np.log1p(np.clip(np.asarray(X, dtype=np.float32), 0, None))  # (n, g) log1p target
    n_genes = T.shape[1]

    # --- model: 2 -> hidden -> ... -> n_genes (ReLU MLP) -----------------
    layers: List[nn.Module] = [nn.Linear(2, hidden), nn.ReLU()]
    for _ in range(max(0, n_layers - 1)):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, n_genes))
    model = nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    F_t = torch.from_numpy(F).float()
    T_t = torch.from_numpy(T).float()
    train_idx = np.where(is_train)[0]
    if train_idx.size == 0:
        raise SystemExit("no train cells — cannot fit the MLP.")

    print(f"  training MLP (seed={seed}, hidden={hidden}, n_layers={n_layers}, "
          f"epochs={epochs}, bs={batch_size}, lr={lr})...")
    t0 = time.time()
    model.train()
    rng = np.random.default_rng(int(seed))
    for ep in range(int(epochs)):
        perm = rng.permutation(train_idx)
        ep_loss, n_b = 0.0, 0
        for start in range(0, perm.size, batch_size):
            bidx = perm[start:start + batch_size]
            xb = F_t[bidx].to(device)
            yb = T_t[bidx].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_b += 1
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep + 1:>4d}/{epochs}  train MSE={ep_loss / max(n_b, 1):.4f}")
    print(f"  training took {time.time() - t0:.1f}s")

    # --- predict every cell ---------------------------------------------
    model.eval()
    preds = np.empty((adata_full.n_obs, n_genes), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, adata_full.n_obs, 8192):
            xb = F_t[start:start + 8192].to(device)
            preds[start:start + 8192] = model(xb).cpu().numpy()
    X_hat = np.expm1(np.clip(preds, 0, None)).astype(np.float32)   # back to count scale
    if X_hat.shape != (adata_full.n_obs, adata_full.n_vars):
        raise RuntimeError(f"X_hat shape {X_hat.shape} != "
                           f"({adata_full.n_obs}, {adata_full.n_vars})")
    adata_full.layers["X_hat"] = X_hat
    print(f"  X_hat populated (shape={X_hat.shape}, min={X_hat.min():.3f}, "
          f"mean={X_hat.mean():.3f}, max={X_hat.max():.3f})")
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
    p.add_argument("--nbr-neighs", type=int, default=16,
                   help="Spatial kNN neighbors for the niche-branch aggregation "
                        "(X_nbr / X_hat_nbr), matching SQUINT's native niche graph.")
    p.add_argument("--mlp-epochs", type=int, default=300)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-layers", type=int, default=3,
                   help="Number of hidden layers (each `--mlp-hidden` wide).")
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-batch-size", type=int, default=4096)
    p.add_argument("--use-default-holdout-regions", action="store_true", default=True)
    p.add_argument("--no-default-holdout-regions",
                   dest="use_default_holdout_regions", action="store_false")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--accelerator", type=str, default="auto",
                   help="'auto' (gpu if available) | 'gpu' | 'cpu'.")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds parsed to empty list.")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = args.artifacts_root / args.dataset_tag / args.variant_tag / ts
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

    # ---- 3. Per-seed train + infer + Pearson ----------------------------
    per_seed_frames: List[pd.DataFrame] = []
    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        adata_s = adata.copy()
        train_one_seed(
            adata_s, seed=seed, batch_key=args.batch_key,
            epochs=args.mlp_epochs, hidden=args.mlp_hidden, n_layers=args.mlp_layers,
            lr=args.mlp_lr, batch_size=args.mlp_batch_size, accelerator=args.accelerator,
        )
        add_neighborhood_layers(adata_s, batch_key=args.batch_key, n_neighs=args.nbr_neighs)
        df = build_pearson_dataframe(adata_s, seed=seed, log1p=True, n_hvg=50)
        per_seed_frames.append(df)
        write_predicted_adata(adata_s, args.out_dir, seed)
        for split in ("all", "train", "test"):
            row = df[(df["split"] == split) & (df["branch"] == "cell") &
                     (df["axis"] == "cell_wise") & (df["transform"] == "log1p") &
                     (df["gene_subset"] == "all")]
            if not row.empty:
                print(f"  Pearson cell_wise log1p all (split={split:<5s}) = "
                      f"{float(row['pearson_mean'].iloc[0]):.4f}")

    per_seed = (pd.concat(per_seed_frames, ignore_index=True)
                if per_seed_frames else pd.DataFrame())

    # ---- 4. Write CSVs --------------------------------------------------
    print("\n=== Writing outputs ===")
    write_pearson_outputs(args.out_dir, per_seed)

    # ---- 5. Console summary --------------------------------------------
    print("\n" + "=" * 78)
    print(f"SUMMARY (mean ± std across {len(seeds)} seeds, cell_wise × log1p × all)")
    print("=" * 78)
    canonical = per_seed[(per_seed["branch"] == "cell") &
                         (per_seed["axis"] == "cell_wise") &
                         (per_seed["transform"] == "log1p") &
                         (per_seed["gene_subset"] == "all")]
    for split in ("all", "train", "test"):
        sub = canonical[canonical["split"] == split]
        if sub.empty:
            continue
        vals = sub["pearson_mean"].astype(float)
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        print(f"  cell Pearson (split={split:<5s}) = {vals.mean():.4f} ± {std:.4f}")

    # ---- 6. Stub config -------------------------------------------------
    import yaml
    stub_cfg = {
        "experiment": {"name": args.variant_tag,
                       "description": "MLP coords->expression imputation baseline "
                                      "(GeST's MLP baseline; region-holdout)."},
        "dataset": {"dataset_name": args.dataset_tag, "dataset_tag": args.dataset_tag,
                    "root_data_dir": str(Path(args.silver_dir).parent.parent)},
        "model": {"model_name": "mlp-spatial"},
        "mlp": {"epochs": args.mlp_epochs, "hidden": args.mlp_hidden,
                "n_layers": args.mlp_layers, "lr": args.mlp_lr,
                "batch_size": args.mlp_batch_size, "nbr_neighs": args.nbr_neighs,
                "batch_key": args.batch_key, "seeds": seeds,
                "holdout_regions": DEFAULT_HOLDOUT_REGIONS},
    }
    with open(args.out_dir / "user_specified_config.yaml", "w") as f:
        yaml.safe_dump(stub_cfg, f, sort_keys=False)
    print(f"\n  -> {args.out_dir / 'user_specified_config.yaml'}")

    print("\n" + "=" * 78)
    print(f"DONE\n  Run dir : {args.out_dir}\n  Variant : {args.variant_tag}\n"
          f"  Seeds   : {len(seeds)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
