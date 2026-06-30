"""
GeST baseline for the gene-expression imputation benchmark (SQUINT region-holdout
split). GeST (Hao et al., MLCB 2025) has no public code/weights, so this is a
faithful reimplementation of its "unseen cell generation" task (predict a
held-out region's per-cell expression from the surrounding cells). See
gest/ for the components (meta-cell tokenizer, SPE, spatial-attention
transformer, hierarchical loss, diagonal serialization).

Per-dataset training (like the paper's GP/MLP baselines): fit on the train
(non-held-out) cells, predict every cell from its nearest OBSERVED (train)
neighbors. Held-out cells' own expression is never seen.

Mirrors run_scvi.py exactly except for the model: the shared `_holdout_utils`
handles silver loading, the held-out split, Pearson, and the CSVs.

Output (per the harness):
  <out_dir>/predicted_adata.h5ad
  <out_dir>/metrics/per_seed_pearson_reconstruction.csv
  <out_dir>/metrics/pearson_reconstruction_metrics.csv
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

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from _holdout_utils import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATASET_TAG, DEFAULT_HOLDOUT_REGIONS,
    DEFAULT_SILVER_DIR, add_neighborhood_layers, apply_holdout_regions,
    build_pearson_dataframe, load_silver_concat, write_pearson_outputs,
    _to_dense_2d,
)
from gest import MetaCellVocab                              # noqa: E402
from gest.model import GeST, GeSTConfig                     # noqa: E402
from gest.train import train_gest, predict_all              # noqa: E402

DEFAULT_VARIANT_TAG = "gest-imputed+region-holdout"


def _test_spearman(adata: ad.AnnData) -> Optional[float]:
    """Mean gene-wise Spearman on held-out cells (paper's rho metric, for a
    sanity cross-check vs Table 1 ~0.3 on MERFISH). Returns None if scipy absent."""
    try:
        from scipy.stats import spearmanr
    except Exception:
        return None
    test = (adata.obs["data_split"].to_numpy() == "test")
    if test.sum() < 3:
        return None
    X = _to_dense_2d(adata.X)[test]
    Xh = _to_dense_2d(adata.layers["X_hat"])[test]
    rs = []
    for g in range(X.shape[1]):
        if X[:, g].std() > 0 and Xh[:, g].std() > 0:
            rs.append(spearmanr(X[:, g], Xh[:, g]).correlation)
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.mean(rs)) if rs else None


def train_one_seed(adata_full: ad.AnnData, seed: int, batch_key: str,
                   args) -> ad.AnnData:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = (args.device if args.device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_mask = (adata_full.obs["data_split"].to_numpy() == "train")
    print(f"  train cells: {int(train_mask.sum())}   "
          f"held-out: {int(adata_full.n_obs - train_mask.sum())}   device={device}")

    X = _to_dense_2d(adata_full.X)
    coords = np.asarray(adata_full.obsm["spatial"], dtype=np.float64)[:, :2]
    section = adata_full.obs[batch_key].astype("category").cat.codes.to_numpy()

    # meta-cell vocabulary fit on TRAIN cells only (Eq. 7)
    print(f"  fitting meta-cell vocab (K={args.n_meta}) on train cells...")
    vocab = MetaCellVocab.fit(X[train_mask], n_meta=args.n_meta, n_pca=args.n_pca,
                              log1p=True, seed=seed)

    cfg = GeSTConfig(d_model=args.d_model, n_layers=args.n_layers,
                     n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout)
    model = GeST(n_genes=X.shape[1], cfg=cfg).to(device)
    model.set_vocab(vocab.C_expr, vocab.level_labels, vocab.level_sizes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GeST params: {n_params/1e6:.2f}M  (d={args.d_model}, "
          f"L={args.n_layers}, H={args.n_heads})")

    t0 = time.time()
    train_gest(X, coords, section, train_mask, model, vocab,
               window=args.window, seq_n=args.seq_n, steps=args.steps,
               batch_size=args.batch_size, lr=args.lr,
               weight_decay=args.weight_decay, device=device, seed=seed)
    print(f"  training took {time.time()-t0:.1f}s")

    print(f"  predicting all cells from observed neighbors (k={args.neighbors_k}, "
          f"mode={args.decode})...")
    X_hat = predict_all(X, coords, section, train_mask, model, vocab,
                        neighbors_k=args.neighbors_k, device=device,
                        batch_size=args.pred_batch, mode=args.decode)
    adata_full.layers["X_hat"] = X_hat
    print(f"  X_hat populated (shape={X_hat.shape}, min={X_hat.min():.3f}, "
          f"mean={X_hat.mean():.3f}, max={X_hat.max():.3f})")
    return adata_full


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--silver-dir", type=str, default=DEFAULT_SILVER_DIR)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--variant-tag", type=str, default=DEFAULT_VARIANT_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--batch-key", type=str, default="adata_batch_id")
    p.add_argument("--use-default-holdout-regions", action="store_true", default=True)
    p.add_argument("--no-default-holdout-regions",
                   dest="use_default_holdout_regions", action="store_false")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--device", type=str, default="auto")
    # GeST hparams
    p.add_argument("--n-meta", type=int, default=500, help="K meta cells (Eq. 7).")
    p.add_argument("--n-pca", type=int, default=50)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq-n", type=int, default=64, help="cells per serialized crop.")
    p.add_argument("--window", type=float, default=40.0,
                   help="crop side in median-NN-distance units (scale-invariant).")
    p.add_argument("--neighbors-k", type=int, default=30,
                   help="observed neighbors per held-out cell at inference.")
    p.add_argument("--decode", type=str, default="weighted",
                   choices=["weighted", "picking"])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pred-batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--smoke", action="store_true", help="tiny/fast plumbing run.")
    args = p.parse_args()

    if args.smoke:
        args.n_meta = min(args.n_meta, 64)
        args.d_model, args.n_layers, args.n_heads, args.d_ff = 32, 2, 4, 64
        args.seq_n, args.steps, args.batch_size = 16, 30, 4
        args.seeds = args.seeds.split(",")[0]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = args.artifacts_root / args.dataset_tag / args.variant_tag / ts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {args.out_dir}\nSeeds   : {seeds}")

    print("\n=== Loading silver ===")
    adata = load_silver_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    print("\n=== Held-out region split ===")
    regions = DEFAULT_HOLDOUT_REGIONS if args.use_default_holdout_regions else None
    if regions is None:
        raise SystemExit("Pass --use-default-holdout-regions.")
    apply_holdout_regions(adata, batch_key=args.batch_key, regions=regions)

    per_seed_frames: List[pd.DataFrame] = []
    seed0_adata: Optional[ad.AnnData] = None
    for s_idx, seed in enumerate(seeds):
        print("\n" + "=" * 78 + f"\nSEED {seed}  ({s_idx+1}/{len(seeds)})\n" + "=" * 78)
        adata_s = adata.copy()
        train_one_seed(adata_s, seed=seed, batch_key=args.batch_key, args=args)
        add_neighborhood_layers(adata_s, batch_key=args.batch_key)
        df = build_pearson_dataframe(adata_s, seed=seed, log1p=True, n_hvg=50)
        per_seed_frames.append(df)
        if s_idx == 0:
            seed0_adata = adata_s
        rho = _test_spearman(adata_s)
        for split in ("all", "train", "test"):
            row = df[(df.split == split) & (df.branch == "cell")
                     & (df.axis == "gene_wise") & (df.transform == "raw")
                     & (df.gene_subset == "all")]
            if not row.empty:
                print(f"  Pearson gene_wise raw (split={split:<5s}) = "
                      f"{float(row['pearson_mean'].iloc[0]):.4f}")
        if rho is not None:
            print(f"  [cross-check] test gene-wise Spearman = {rho:.4f}  "
                  f"(paper Table 1 MERFISH rho ~0.24-0.30)")

    per_seed = pd.concat(per_seed_frames, ignore_index=True) if per_seed_frames else pd.DataFrame()
    print("\n=== Writing outputs ===")
    write_pearson_outputs(args.out_dir, per_seed)

    if seed0_adata is not None:
        out_h5ad = args.out_dir / "predicted_adata.h5ad"
        sib = Path(__file__).resolve().parent.parent / "cell_type_identification"
        if str(sib) not in sys.path:
            sys.path.insert(0, str(sib))
        try:
            from run_pca_leiden import _sanitize_for_h5ad  # type: ignore
            _sanitize_for_h5ad(seed0_adata)
        except Exception as exc:
            print(f"  (sanitizer not available: {exc})")
        seed0_adata.write_h5ad(out_h5ad)
        print(f"  -> {out_h5ad}")

    print("\n" + "=" * 78 + f"\nDONE  variant={args.variant_tag}  seeds={len(seeds)}\n" + "=" * 78)


if __name__ == "__main__":
    main()
