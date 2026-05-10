"""
Vanilla VQ-VAE (NBR-only) baseline for the gene-expression imputation
benchmark with the SQUINT region-holdout split.

Same minimal architecture as `run_vanilla_vq_cell.py` (encoder MLP +
single VQ + NB decoder), but trained to reconstruct the
NEIGHBORHOOD-aggregated expression target `X_nbr` (1-hop spatial mean
of `adata.X`) rather than per-cell counts. The encoder still consumes
PER-CELL counts as input — the difference is entirely on the target
side. This isolates the "is the cell→neighborhood mapping learnable
with a small VQ bottleneck?" question without any of SQUINT's
spatial/adversarial machinery.

Stripped-down spec:
  Encoder      MLP cell encoder (hidden=256, depth=2, ReLU)
  Quantiser    single VectorQuantize layer (codebook_size=30, 1 head, EMA)
  Decoder      NB decoder predicting per-gene rate (per-cell library
               taken from row-sum of X_nbr, so the rate is on the same
               scale as the target)
  Loss         NB nll on adata.layers["X_nbr"] (nbr-level)
               + standard VQ commit loss (alpha=0.25)
  ABSENT       no GNN, no cell-recon head, no FiLM, no adversarial,
               no adjacency, no masking

Output:
  <out_dir>/predicted_adata.h5ad             (X_hat_nbr layer + X_nbr target + data_split)
  <out_dir>/metrics/per_seed_pearson_reconstruction.csv
  <out_dir>/metrics/pearson_reconstruction_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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


DEFAULT_VARIANT_TAG = "vanilla-vq-nbr+region-holdout"


# ---------------------------------------------------------------------------
# Model — identical structure to run_vanilla_vq_cell, only the *target*
# of the NB likelihood differs.
# ---------------------------------------------------------------------------

def _build_model(n_genes: int, latent_dim: int, hidden_dim: int,
                 codebook_size: int, vq_decay: float):
    import torch
    from torch import nn
    try:
        from vector_quantize_pytorch import VectorQuantize
    except ImportError as exc:
        raise SystemExit(
            "Vanilla VQ-VAE requires `vector_quantize_pytorch`. Install "
            f"with `pip install vector-quantize-pytorch`. Original: {exc}"
        )

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_genes, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # Input: per-cell counts (cell-level). log1p for scale.
            return self.net(torch.log1p(x))

    class NBDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rate = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_genes),
            )
            self.log_theta = nn.Parameter(torch.zeros(n_genes))

        def forward(self, z: "torch.Tensor", lib: "torch.Tensor"
                    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            rate = torch.softmax(self.rate(z), dim=-1)
            mu = lib.unsqueeze(-1) * rate
            theta = torch.exp(self.log_theta).clamp(min=1e-4, max=1e4)
            return mu, theta

    class VanillaVQVAENbr(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.vq = VectorQuantize(
                dim=latent_dim,
                codebook_size=codebook_size,
                decay=vq_decay,
                use_cosine_sim=True,
                commitment_weight=0.25,
                kmeans_init=True,
                kmeans_iters=10,
                threshold_ema_dead_code=2,
                eps=1e-5,
            )
            self.decoder = NBDecoder()

        def forward(self, x_cell: "torch.Tensor", lib_nbr: "torch.Tensor"):
            z_e = self.encoder(x_cell)
            z_q, indices, commit_loss = self.vq(z_e)
            mu, theta = self.decoder(z_q, lib_nbr)
            return mu, theta, commit_loss, indices

    return VanillaVQVAENbr()


def _nb_nll(x, mu, theta):
    import torch
    eps = 1e-8
    log_theta = torch.log(theta + eps)
    log_mu = torch.log(mu + eps)
    log_theta_mu = torch.log(theta + mu + eps)
    nll = (
        torch.lgamma(theta + eps)
        + torch.lgamma(x + 1.0)
        - torch.lgamma(x + theta + eps)
        + (theta + x) * log_theta_mu
        - x * log_mu
        - theta * log_theta
    )
    return nll.sum(dim=-1)


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------

def train_one_seed(
        adata: ad.AnnData,
        seed: int,
        epochs: int,
        batch_size: int,
        lr: float,
        latent_dim: int,
        hidden_dim: int,
        codebook_size: int,
        vq_decay: float,
        device: str,
    ) -> ad.AnnData:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    # Cell-level input (encoder consumes per-cell counts).
    X_cell = adata.X
    if sp.issparse(X_cell):
        X_cell = X_cell.toarray()
    X_cell = X_cell.astype(np.float32)

    # Nbr-level target (1-hop neighborhood mean — assumed already in layers).
    if "X_nbr" not in adata.layers:
        raise SystemExit(
            "adata.layers['X_nbr'] missing — call compute_X_nbr() before training."
        )
    X_nbr = adata.layers["X_nbr"]
    if sp.issparse(X_nbr):
        X_nbr = X_nbr.toarray()
    X_nbr = X_nbr.astype(np.float32)
    # NB targets must be non-negative integers in spirit; the nbr-mean
    # produces non-negative reals which is OK for NB likelihood (NB
    # supports continuous extension via lgamma).

    train_mask = (adata.obs["data_split"].to_numpy() == "train")
    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(~train_mask)[0]
    print(f"  train cells: {train_idx.size}   held-out cells: {test_idx.size}")

    # Library scale taken from the row-sum of the nbr target so the NB
    # rate × library matches the target's magnitude.
    lib_nbr = X_nbr.sum(axis=1).astype(np.float32)
    lib_nbr = np.maximum(lib_nbr, 1.0)

    train_ds = TensorDataset(
        torch.from_numpy(X_cell[train_idx]),
        torch.from_numpy(X_nbr[train_idx]),
        torch.from_numpy(lib_nbr[train_idx]),
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )

    n_genes = X_cell.shape[1]
    model = _build_model(
        n_genes=n_genes, latent_dim=latent_dim, hidden_dim=hidden_dim,
        codebook_size=codebook_size, vq_decay=vq_decay,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"  training (epochs={epochs}, batch_size={batch_size}, lr={lr}, "
          f"latent={latent_dim}, hidden={hidden_dim}, K={codebook_size})...")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        nll_sum = 0.0; commit_sum = 0.0; n_used = 0
        for xc, xn, lb in train_dl:
            xc = xc.to(device); xn = xn.to(device); lb = lb.to(device)
            mu, theta, commit_loss, _ = model(xc, lb)
            nll = _nb_nll(xn, mu, theta).mean()
            loss = nll + commit_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            nll_sum += float(nll.item()) * xc.shape[0]
            commit_sum += float(commit_loss.item()) * xc.shape[0]
            n_used += xc.shape[0]
        wall = time.time() - t0
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"    epoch {epoch+1:>3d}/{epochs}: "
                  f"NB-nll-nbr={nll_sum/max(n_used,1):.3f}  "
                  f"commit={commit_sum/max(n_used,1):.4f}  "
                  f"({wall:.1f}s)")

    # Inference on all cells.
    print("  inference (all cells)...")
    model.eval()
    X_hat_nbr = np.zeros_like(X_nbr)
    with torch.no_grad():
        for start in range(0, X_cell.shape[0], batch_size):
            stop = min(start + batch_size, X_cell.shape[0])
            xc = torch.from_numpy(X_cell[start:stop]).to(device)
            lb = torch.from_numpy(lib_nbr[start:stop]).to(device)
            mu, _theta, _cl, _idx = model(xc, lb)
            X_hat_nbr[start:stop] = mu.detach().cpu().numpy()

    adata.layers["X_hat_nbr"] = X_hat_nbr
    print(f"  X_hat_nbr populated (shape={X_hat_nbr.shape}, "
          f"min={X_hat_nbr.min():.3f}, mean={X_hat_nbr.mean():.3f}, "
          f"max={X_hat_nbr.max():.3f})")
    return adata


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
    # Holdout (defaults baked in, like cell variant).
    p.add_argument("--use-default-holdout-regions", action="store_true", default=True)
    p.add_argument("--no-default-holdout-regions",
                   dest="use_default_holdout_regions", action="store_false")
    # Training.
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--codebook-size", type=int, default=30)
    p.add_argument("--vq-decay", type=float, default=0.8)
    # Repro / device.
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--device", type=str, default="auto")
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

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"Run dir : {args.out_dir}")
    print(f"Seeds   : {seeds}")
    print(f"Device  : {device}")

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

    # ---- 3. Build spatial graph + X_nbr target --------------------------
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
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            codebook_size=args.codebook_size,
            vq_decay=args.vq_decay,
            device=device,
        )
        df = build_pearson_dataframe(adata_s, seed=seed, log1p=True, n_hvg=50)
        per_seed_frames.append(df)
        if s_idx == 0:
            seed0_adata = adata_s

        # Print quick summary — canonical (cell_wise × log1p × all) row only.
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
                       "description": "Vanilla VQ-VAE nbr-only baseline (region-holdout)."},
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "VanillaVQVAE-nbr"},
        "vanilla_vq": {
            "epochs":         args.epochs,
            "batch_size":     args.batch_size,
            "lr":             args.lr,
            "latent_dim":     args.latent_dim,
            "hidden_dim":     args.hidden_dim,
            "codebook_size":  args.codebook_size,
            "vq_decay":       args.vq_decay,
            "n_spatial_neighs": args.n_spatial_neighs,
            "seeds":          seeds,
            "device":         device,
            "batch_key":      args.batch_key,
            "holdout_regions": DEFAULT_HOLDOUT_REGIONS,
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
