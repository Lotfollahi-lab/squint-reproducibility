"""
Vanilla VQ-VAE (CELL-only) baseline for the gene-expression imputation
benchmark with the SQUINT region-holdout split.

Strips SQUINT down to the simplest possible VQ generative model:

  Encoder      MLP cell encoder (hidden=256, depth=2, ReLU)
  Quantiser    single VectorQuantize layer (codebook_size=30, 1 head, EMA)
  Decoder      NB decoder (per-gene rate × per-cell library size; theta
               learned per gene)
  Loss         NB negative-log-likelihood on adata.X (cell-level)
               + standard VQ commitment loss (alpha=0.25)
  ABSENT       no GNN, no nbr-recon head, no FiLM, no adversarial,
               no adjacency, no masking, no FiLM/covariate decoder

Trained on `data_split == "train"` cells (the SQUINT region-holdout
split). At inference, runs the full encoder-quantiser-decoder pipeline
on every cell (train AND test) and saves predicted `X_hat` plus the
`data_split` column. Pearson on `X_hat` vs `adata.X` is then reported
per split (full / train / test).

Output layout:
  <out_dir>/predicted_adata.h5ad                              (with X_hat layer)
  <out_dir>/metrics/per_seed_pearson_reconstruction.csv
  <out_dir>/metrics/pearson_reconstruction_metrics.csv

Run:
  python run_vanilla_vq_cell.py \\
      --silver-dir /nfs/team361/sb75/.../silver/mmb0-1b_smb1-1b_1p \\
      --epochs 80 --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Local imports.
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
    load_silver_concat,
    write_pearson_outputs,
)


DEFAULT_VARIANT_TAG = "vanilla-vq-cell+region-holdout"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_model(n_genes: int, latent_dim: int, hidden_dim: int,
                 codebook_size: int, vq_decay: float):
    """Construct the vanilla VQ-VAE: MLP encoder + VQ + NB decoder."""
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
            # log1p input — same scale convention as scVI / SQUINT cell encoder.
            return self.net(torch.log1p(x))

    class NBDecoder(nn.Module):
        """Negative-binomial decoder: predicts mean = library_size * softmax(rate),
        with a per-gene dispersion `theta` (learned, gene-specific)."""

        def __init__(self) -> None:
            super().__init__()
            self.rate = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_genes),
            )
            # Per-gene log-dispersion (theta). Initialised at 1.0.
            self.log_theta = nn.Parameter(torch.zeros(n_genes))

        def forward(self, z: "torch.Tensor", lib: "torch.Tensor"
                    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            # px_rate per gene: library_size * softmax(rate).
            rate = torch.softmax(self.rate(z), dim=-1)
            mu = lib.unsqueeze(-1) * rate
            theta = torch.exp(self.log_theta).clamp(min=1e-4, max=1e4)
            return mu, theta

    class VanillaVQVAECell(nn.Module):
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

        def forward(self, x: "torch.Tensor", lib: "torch.Tensor"):
            z_e = self.encoder(x)
            z_q, indices, commit_loss = self.vq(z_e)
            mu, theta = self.decoder(z_q, lib)
            return mu, theta, commit_loss, indices

    return VanillaVQVAECell()


def _nb_nll(x: "torch.Tensor", mu: "torch.Tensor", theta: "torch.Tensor"
            ) -> "torch.Tensor":
    """Negative log-likelihood of NB(mu, theta) on observed counts x.
    `theta` is per-gene dispersion (broadcasted across cells)."""
    import torch
    eps = 1e-8
    # Broadcast theta (n_genes,) across cells.
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
    return nll.sum(dim=-1)  # sum over genes per cell


# ---------------------------------------------------------------------------
# Training / inference loop
# ---------------------------------------------------------------------------

def _to_dense_torch(X, device) -> "torch.Tensor":
    import torch
    if sp.issparse(X):
        X = X.toarray()
    return torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)


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
    """Train the vanilla VQ-VAE on `data_split == 'train'` cells, then
    run inference on EVERY cell and write `X_hat` into adata.layers.
    Returns the same `adata` for chaining."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    # Convert counts to dense tensor on CPU first; keep test cells out of training.
    X_dense = adata.X
    if sp.issparse(X_dense):
        X_dense = X_dense.toarray()
    X_dense = X_dense.astype(np.float32)

    train_mask = (adata.obs["data_split"].to_numpy() == "train")
    train_idx = np.where(train_mask)[0]
    test_idx  = np.where(~train_mask)[0]
    print(f"  train cells: {train_idx.size}   held-out cells: {test_idx.size}")

    # Library size per cell (sum of raw counts) — used as the NB scale factor.
    lib = X_dense.sum(axis=1).astype(np.float32)
    # Avoid 0-library for any cell (would make NB mu collapse).
    lib = np.maximum(lib, 1.0)

    train_ds = TensorDataset(
        torch.from_numpy(X_dense[train_idx]),
        torch.from_numpy(lib[train_idx]),
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0,
    )

    n_genes = X_dense.shape[1]
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
        nll_sum = 0.0
        commit_sum = 0.0
        n_used = 0
        for xb, lb in train_dl:
            xb = xb.to(device); lb = lb.to(device)
            mu, theta, commit_loss, _ = model(xb, lb)
            nll = _nb_nll(xb, mu, theta).mean()
            loss = nll + commit_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            nll_sum += float(nll.item()) * xb.shape[0]
            commit_sum += float(commit_loss.item()) * xb.shape[0]
            n_used += xb.shape[0]
        wall = time.time() - t0
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"    epoch {epoch+1:>3d}/{epochs}: "
                  f"NB-nll={nll_sum/max(n_used,1):.3f}  "
                  f"commit={commit_sum/max(n_used,1):.4f}  "
                  f"({wall:.1f}s)")

    # Inference on all cells, in batches to stay GPU-friendly.
    print("  inference (all cells)...")
    model.eval()
    X_hat = np.zeros_like(X_dense)
    with torch.no_grad():
        for start in range(0, X_dense.shape[0], batch_size):
            stop = min(start + batch_size, X_dense.shape[0])
            xb = torch.from_numpy(X_dense[start:stop]).to(device)
            lb = torch.from_numpy(lib[start:stop]).to(device)
            mu, _theta, _cl, _idx = model(xb, lb)
            X_hat[start:stop] = mu.detach().cpu().numpy()

    adata.layers["X_hat"] = X_hat
    print(f"  X_hat populated (shape={X_hat.shape}, "
          f"min={X_hat.min():.3f}, mean={X_hat.mean():.3f}, max={X_hat.max():.3f})")
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
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override run dir. Default: "
                        "<artifacts_root>/<dataset_tag>/<variant_tag>/<TS>/")
    p.add_argument("--batch-key", type=str, default="adata_batch_id")
    # Holdout-region spec (uses the SQUINT default geometry).
    # Override via JSON file if you want a custom region — kept simple here.
    p.add_argument("--use-default-holdout-regions", action="store_true",
                   default=True,
                   help="Use the SQUINT _patch_holdout_regions geometry "
                        "(default). Disable with --no-default-holdout-regions "
                        "if you want to write your own regions dict in code.")
    p.add_argument("--no-default-holdout-regions", dest="use_default_holdout_regions",
                   action="store_false")
    # Training hparams.
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--codebook-size", type=int, default=30,
                   help="VQ codebook size. Match SQUINT cell branch "
                        "level-1 (30) for fair comparison.")
    p.add_argument("--vq-decay", type=float, default=0.8)
    # Repro / device.
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' picks cuda:0 if torch.cuda.is_available() else cpu.")
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

    # Resolve device.
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

    # ---- 1. Load + concat silver ----------------------------------------
    print("\n=== Loading silver ===")
    adata = load_silver_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"Concatenated AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from obs.")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")

    # ---- 2. Apply held-out-region split ---------------------------------
    print("\n=== Held-out region split ===")
    regions = DEFAULT_HOLDOUT_REGIONS if args.use_default_holdout_regions else None
    if regions is None:
        raise SystemExit(
            "No regions specified. Either pass --use-default-holdout-regions or "
            "extend this script to load a custom regions JSON."
        )
    apply_holdout_regions(adata, batch_key=args.batch_key, regions=regions)

    # ---- 3. Per-seed train + infer + Pearson ----------------------------
    per_seed_frames: List[pd.DataFrame] = []
    seed0_adata: Optional[ad.AnnData] = None
    for s_idx, seed in enumerate(seeds):
        print()
        print("=" * 78)
        print(f"SEED {seed}  ({s_idx + 1}/{len(seeds)})")
        print("=" * 78)
        # Fresh copy per seed so X_hat from seed_{i-1} doesn't leak.
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
                (df["split"]      == split) &
                (df["branch"]     == "cell") &
                (df["axis"]       == "cell_wise") &
                (df["transform"]  == "log1p") &
                (df["gene_subset"] == "all")
            ]
            if not row.empty:
                v = float(row["pearson_mean"].iloc[0])
                print(f"  Pearson cell_wise log1p all (split={split:<5s}) = {v:.4f}")

    per_seed = (
        pd.concat(per_seed_frames, ignore_index=True)
        if per_seed_frames else pd.DataFrame()
    )

    # ---- 4. Write CSVs --------------------------------------------------
    print("\n=== Writing outputs ===")
    write_pearson_outputs(args.out_dir, per_seed)

    # ---- 5. Save seed-0 predicted adata --------------------------------
    if seed0_adata is not None:
        out_h5ad = args.out_dir / "predicted_adata.h5ad"
        # Strip ArrowStringArray columns for h5ad compatibility (same trick
        # used everywhere in this repo).
        import sys as _sys
        sib = (Path(__file__).resolve().parent.parent
               / "cell_type_identification")
        if str(sib) not in _sys.path:
            _sys.path.insert(0, str(sib))
        try:
            from run_pca_leiden import _sanitize_for_h5ad  # type: ignore
            _sanitize_for_h5ad(seed0_adata)
        except Exception as exc:
            print(f"  (sanitizer not available: {exc} — writing raw)")
        seed0_adata.write_h5ad(out_h5ad)
        print(f"  -> {out_h5ad}  (seed[0] snapshot)")

    # ---- 6. Console summary --------------------------------------------
    print()
    print("=" * 78)
    print(f"SUMMARY (mean ± std across {len(seeds)} seeds, "
          "cell_wise × log1p × all)")
    print("=" * 78)
    canonical = per_seed[
        (per_seed["branch"]      == "cell") &
        (per_seed["axis"]        == "cell_wise") &
        (per_seed["transform"]   == "log1p") &
        (per_seed["gene_subset"] == "all")
    ]
    for split in ("all", "train", "test"):
        sub = canonical[canonical["split"] == split]
        if sub.empty:
            continue
        vals = sub["pearson_mean"].astype(float)
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        print(f"  cell Pearson  (split={split:<5s}) = "
              f"{vals.mean():.4f} ± {std:.4f}  "
              f"(min={vals.min():.4f}, max={vals.max():.4f})")
    print(f"  (full {len(per_seed)}-row variant table written to "
          f"per_seed_pearson_reconstruction.csv)")

    # ---- 7. Stub config -------------------------------------------------
    import yaml
    stub_cfg = {
        "experiment": {
            "name": args.variant_tag,
            "description": "Vanilla VQ-VAE cell-only baseline (region-holdout).",
        },
        "dataset": {
            "dataset_name": args.dataset_tag,
            "dataset_tag":  args.dataset_tag,
            "root_data_dir": str(Path(args.silver_dir).parent.parent),
        },
        "model": {"model_name": "VanillaVQVAE-cell"},
        "vanilla_vq": {
            "epochs":         args.epochs,
            "batch_size":     args.batch_size,
            "lr":             args.lr,
            "latent_dim":     args.latent_dim,
            "hidden_dim":     args.hidden_dim,
            "codebook_size":  args.codebook_size,
            "vq_decay":       args.vq_decay,
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
