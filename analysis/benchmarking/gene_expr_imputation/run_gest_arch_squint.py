"""
Ablation arm: GeST ARCHITECTURE on SQUINT's FROZEN discrete codes.

Isolates "architecture vs codes/decoder". This arm reuses the EXACT SQUINT codes
+ frozen stage-1 NB decoder that the "SQUINT (imputed)" bar uses, and differs
ONLY in the prediction transformer: instead of SQUINT's stage-2 graph-native
masked-code transformer, it uses the GeST decoder-only transformer + the
Spatial-Attention suffix mask + diagonal serialization + 2D sinusoidal SPE
(``gest/squint_codes.py``, reusing ``gest/model.py``'s blocks).

Pipeline (mirrors ``run_gest.py``, but predicts codes not expression):
  1. Load the FROZEN ``predicted_adata.h5ad`` (true X + true SQUINT code stacks +
     batch ids + spatial). This single adata drives everything so the cells +
     codes + decoder all align by construction (no silver<->codes join).
  2. apply_holdout_regions -> obs['data_split'] in {train, test} (same geometry
     as SQUINT / the other imputation baselines).
  3. Read the 4 true codes (cell L0/L1 + niche L0/L1) for training targets.
  4. Train GeSTSquintCodes on TRAIN cells; predict each cell's 4 codes from its
     nearest OBSERVED (train) neighbors.
  5. Decode the PREDICTED cell codes -> X_hat through the frozen stage-1 decoder
     (reusing ``examples/stage2_decode_pearson.py``'s decode_cell_xhat path).
  6. SELF-CHECK: decode the TRUE codes the same way and confirm it reproduces the
     stored ``layers['X_hat']`` (cell-wise Pearson ~1.0) -- the same guard the
     SQUINT-imputed script uses; if it fails the decode path is misconfigured.
  7. build_pearson_dataframe + write_pearson_outputs to variant
     'squint-gestarch+region-holdout'. Also reports test gene-wise Spearman.

Output (per the shared harness):
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
    apply_holdout_regions, build_pearson_dataframe, write_pearson_outputs,
    _to_dense_2d,
)
from gest.model import GeSTConfig                                   # noqa: E402
from gest.squint_codes import GeSTSquintCodes                       # noqa: E402
from gest.squint_codes_np import (                                  # noqa: E402
    CodeStackSpec, read_squint_codes, build_target_codes_array,
    per_target_sizes, target_names,
)
from gest.squint_codes_train import train_gest_codes, predict_codes_all  # noqa: E402

DEFAULT_VARIANT_TAG = "squint-gestarch+region-holdout"


# ---------------------------------------------------------------------------
# Decode helpers: reuse the SQUINT-imputed decode path verbatim.
# ---------------------------------------------------------------------------
def _import_decode_helpers():
    """Import decode_cell_xhat / _cov_index / _find_ckpt / _pearson_pairwise from
    squint/examples/stage2_decode_pearson.py so the decode is bit-identical to
    the SQUINT-imputed bar. Falls back to a SQUINT_EXAMPLES env override."""
    import importlib.util
    import os

    cand_dirs = []
    env = os.environ.get("SQUINT_EXAMPLES_DIR")
    if env:
        cand_dirs.append(Path(env))
    # repo layout: .../squint-reproducibility/... and a sibling .../squint/examples
    here = _THIS
    for up in here.parents:
        cand = up / "squint" / "examples"
        if cand.is_dir():
            cand_dirs.append(cand)
    for d in cand_dirs:
        f = d / "stage2_decode_pearson.py"
        if f.is_file():
            spec = importlib.util.spec_from_file_location("stage2_decode_pearson", f)
            mod = importlib.util.module_from_spec(spec)
            # the stage-1 src must be importable for the run, but importing the
            # module itself only needs numpy (torch is imported inside main()).
            src = d.parent / "src"
            if src.is_dir() and str(src) not in sys.path:
                sys.path.insert(0, str(src))
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "could not locate squint/examples/stage2_decode_pearson.py; set "
        "SQUINT_EXAMPLES_DIR to its directory.")


def _test_spearman(adata: ad.AnnData) -> Optional[float]:
    """Mean gene-wise Spearman on held-out cells (cross-check vs the GeST arm)."""
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
                   args, decode_mod) -> ad.AnnData:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = (args.device if args.device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_mask = (adata_full.obs["data_split"].to_numpy() == "train")
    print(f"  train cells: {int(train_mask.sum())}   "
          f"held-out: {int(adata_full.n_obs - train_mask.sum())}   device={device}")

    coords = np.asarray(adata_full.obsm["spatial"], dtype=np.float64)[:, :2]
    section = adata_full.obs[batch_key].astype("category").cat.codes.to_numpy()

    # ---- true SQUINT codes (training targets + decode) --------------------
    codes_cell, codes_niche, sizes_cell, sizes_niche = read_squint_codes(adata_full)
    codes_stack = build_target_codes_array(codes_cell, codes_niche)   # (n, T)
    sizes = per_target_sizes(sizes_cell, sizes_niche)
    names = target_names(sizes_cell, sizes_niche)
    spec = CodeStackSpec(names=names, sizes=sizes)
    print(f"  SQUINT code targets: {list(zip(names, sizes))}  (T={spec.n_targets})")

    cfg = GeSTConfig(d_model=args.d_model, n_layers=args.n_layers,
                     n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout)
    model = GeSTSquintCodes(spec, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GeST-codes params: {n_params/1e6:.2f}M  (d={args.d_model}, "
          f"L={args.n_layers}, H={args.n_heads})")

    t0 = time.time()
    train_gest_codes(codes_stack, coords, section, train_mask, model,
                     window=args.window, seq_n=args.seq_n, steps=args.steps,
                     batch_size=args.batch_size, lr=args.lr,
                     weight_decay=args.weight_decay, device=device, seed=seed,
                     label_smoothing=args.label_smoothing)
    print(f"  training took {time.time()-t0:.1f}s")

    print(f"  predicting codes from observed neighbors (k={args.neighbors_k})...")
    pred_stack = predict_codes_all(codes_stack, coords, section, train_mask, model,
                                   neighbors_k=args.neighbors_k, device=device,
                                   batch_size=args.pred_batch)   # (n, T)

    # held-out code accuracy (sanity: per-target argmax accuracy on test cells)
    test_mask = ~train_mask
    if test_mask.any():
        for t, nm in enumerate(names):
            acc = float((pred_stack[test_mask, t] == codes_stack[test_mask, t]).mean())
            print(f"    [code acc test] {nm:<10s} = {acc:.4f}")

    # ---- decode PREDICTED cell codes -> X_hat via the frozen stage-1 decoder
    cell_cols = spec.cell_levels                       # indices of cell-branch codes
    pred_cell = pred_stack[:, cell_cols].astype(np.int64)
    true_cell = codes_stack[:, cell_cols].astype(np.int64)

    X = _to_dense_2d(adata_full.X)
    cov_all, n_cov = decode_mod._cov_index(adata_full)
    run_dir = Path(args.predicted_adata).resolve().parent
    ckpt = decode_mod._find_ckpt(str(run_dir), args.stage1_ckpt)
    print(f"  loading stage-1 decoder from {ckpt}")
    from vqniche.models import VQNiche_Dual
    s1 = VQNiche_Dual.load_from_checkpoint(ckpt, map_location=device)
    s1.eval().to(device)
    ccn = getattr(s1.encoder, "ccn_mode", None)
    if ccn not in (None, "film_scale", "film", "film_cont"):
        raise NotImplementedError(
            f"ccn_mode='{ccn}' alters code selection PRE-VQ; decode-from-indices "
            f"supports None/film_scale/film/film_cont (matches stage2_decode_pearson).")

    def _decode(idx_cell_np, rows):
        ic = torch.as_tensor(idx_cell_np[rows], dtype=torch.long, device=device)
        rd = torch.as_tensor(X[rows].sum(axis=1), dtype=torch.float32, device=device)
        cov = (torch.as_tensor(cov_all[rows], dtype=torch.long, device=device)
               if s1.decoder_covariate_dim > 0 else None)
        with torch.no_grad():
            xhat = decode_mod.decode_cell_xhat(s1, ic, None, cov, rd, torch)
        return xhat.detach().cpu().numpy().astype(np.float32)

    # SELF-CHECK: true codes must reproduce stored layers['X_hat'] (~1.0).
    if "X_hat" in adata_full.layers:
        n = X.shape[0]
        samp = (np.arange(n) if n <= 20000
                else np.random.default_rng(0).choice(n, 20000, replace=False))
        xhat_true = _decode(true_cell, samp)
        stored = _to_dense_2d(adata_full.layers["X_hat"])[samp]
        cw = decode_mod._pearson_pairwise(xhat_true, stored, axis=1)
        cw = cw[np.isfinite(cw)]
        sc = float(np.median(cw)) if cw.size else float("nan")
        print(f"  SELF-CHECK true-code X_hat vs stored X_hat: cell-wise Pearson "
              f"median = {sc:.4f} (expect ~1.0)")
        if not (sc >= args.selfcheck_min):
            print(f"  *** WARNING: self-check {sc:.4f} < {args.selfcheck_min}: decode "
                  f"path unreliable (covariate/FiLM/read-depth). Fix before trusting.")
    else:
        print("  (no stored layers['X_hat']; skipping decode self-check)")

    # decode the PREDICTED codes for ALL cells -> X_hat
    X_hat = np.zeros_like(X, dtype=np.float32)
    for start in range(0, X.shape[0], args.pred_batch):
        rows = np.arange(start, min(start + args.pred_batch, X.shape[0]))
        X_hat[rows] = _decode(pred_cell, rows)
    adata_full.layers["X_hat"] = X_hat
    print(f"  X_hat populated (shape={X_hat.shape}, min={X_hat.min():.3f}, "
          f"mean={X_hat.mean():.3f}, max={X_hat.max():.3f})")
    return adata_full


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predicted-adata", required=True,
                   help="Frozen stage-1 predicted_adata.h5ad (true X + codes + X_hat).")
    p.add_argument("--stage1-ckpt", default=None,
                   help="Stage-1 .ckpt (default: auto-find under the predicted_adata dir).")
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
    p.add_argument("--selfcheck-min", type=float, default=0.99,
                   help="Min cell-wise Pearson for the true-code self-check (warn below).")
    # GeST-arch hparams (mirror run_gest.py defaults)
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
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pred-batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true", help="tiny/fast plumbing run.")
    args = p.parse_args()

    if args.smoke:
        args.d_model, args.n_layers, args.n_heads, args.d_ff = 32, 2, 4, 64
        args.seq_n, args.steps, args.batch_size = 16, 30, 4
        args.seeds = args.seeds.split(",")[0]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = args.artifacts_root / args.dataset_tag / args.variant_tag / ts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {args.out_dir}\nSeeds   : {seeds}")

    decode_mod = _import_decode_helpers()

    print("\n=== Loading frozen predicted_adata (true X + codes) ===")
    adata = ad.read_h5ad(args.predicted_adata)
    print(f"AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    if args.batch_key not in adata.obs:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from predicted_adata.obs")
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
        train_one_seed(adata_s, seed=seed, batch_key=args.batch_key, args=args,
                       decode_mod=decode_mod)
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
            print(f"  [cross-check] test gene-wise Spearman = {rho:.4f}")

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
