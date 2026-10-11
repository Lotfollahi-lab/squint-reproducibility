#!/usr/bin/env python
"""
Re-score a saved ``predicted_adata.h5ad`` with the FULL metric panel
(Pearson + Spearman + MSE/RMSE + zero/nonzero AUROC + a marker-gene subset)
WITHOUT retraining.

Use this when a run's ``metrics/per_seed_pearson_reconstruction.csv`` predates
the panel metrics (i.e. only has ``pearson_mean``), so the imputation plot's
non-Pearson panels come up empty. The imputation runners save
``<run_dir>/predicted_adata.h5ad`` with the true ``X`` + ``layers['X_hat']`` +
``obs['data_split']`` + cell-type labels, so ``build_pearson_dataframe`` can be
re-run on it directly — only numpy/sklearn, no GPU, no model reload.

IMPORTANT — SEED COVERAGE: the runners save only SEED 0's adata. So:
  * single-seed / --smoke runs  -> this fully reconstructs the CSV.
  * multi-seed runs (--seeds 0,1,2,3,4) -> this recovers SEED 0 ONLY. The other
    seeds' X_hat were never saved, so their new metrics cannot be recomputed
    here. RE-RUN the job to restore all seeds' error bars (the metric code now
    lives in build_pearson_dataframe, so a plain re-run writes the full panel).

Usage:
  # point at a run dir (writes <run_dir>/metrics/*.csv)
  python rescore_imputation.py --run-dir \
    /nfs/.../artifacts/mmb0-1b_smb1-1b_1p/squint-gestarch+region-holdout/20260630_111942

  # or an explicit adata + out dir
  python rescore_imputation.py --adata /path/predicted_adata.h5ad --out-dir /path

Leak-free parity re-scoring (opt-in; mirrors stage2_decode_pearson
--nbr-observed-fill). Saved h5ads already carry X_nbr / X_hat_nbr, so pass
--recompute-nbr to rebuild them; a NEW --out-dir is required so the original
metrics are never overwritten:
  python rescore_imputation.py --run-dir $ART/$D/gest-imputed+region-holdout/<TS> \
    --out-dir $ART/$D/gest-imputed+region-holdout__nbrtruth/<TS> \
    --nbr-observed-fill truth --recompute-nbr \
    --holdout-from $ART/$D/<SQUINT inductive variant>/<TS>_seed0/predicted_adata.h5ad
  * --nbr-observed-fill truth: observed neighbours enter X_hat_nbr with their
    TRUE counts (X_nbr unchanged); heldout_only: X_nbr and X_hat_nbr both
    aggregate over held-out neighbours plus self only.
  * --holdout-from: copy obs['data_split'] from another predicted_adata by
    obs_names (e.g. SQUINT's), asserting the same number of held-out cells AND
    an identical per-cell split (otherwise the baseline must be re-run under
    the reference split; --allow-split-change scores anyway and records the
    count in a holdout_split_changed column).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
from _holdout_utils import (  # noqa: E402
    add_neighborhood_layers, build_pearson_dataframe, write_pearson_outputs,
)


def _resolve_seed(out_dir: Path, cli_seed):
    """Pick the seed label for the saved adata. The runner saves seeds[0]'s
    adata; recover that label from the existing CSV when possible."""
    if cli_seed is not None:
        return int(cli_seed), None
    csv = out_dir / "metrics" / "per_seed_pearson_reconstruction.csv"
    if csv.is_file():
        try:
            seeds = sorted(pd.read_csv(csv)["seed"].dropna().astype(int).unique().tolist())
        except Exception:  # noqa: BLE001
            seeds = []
        if len(seeds) == 1:
            return seeds[0], None
        if len(seeds) > 1:
            return seeds[0], (
                f"existing CSV has {len(seeds)} seeds {seeds} but only seed "
                f"{seeds[0]}'s adata was saved — re-scoring recovers SEED "
                f"{seeds[0]} ONLY. Re-run the job for all-seed error bars.")
    return 0, None


def _read_obs(path: Path) -> pd.DataFrame:
    """obs of an h5ad without loading X / layers (anndata's element reader on
    the h5py file; falls back to a backed read)."""
    read_elem = None
    try:
        from anndata.io import read_elem  # type: ignore[attr-defined]  (>=0.11)
    except Exception:  # noqa: BLE001
        try:
            from anndata.experimental import read_elem  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            read_elem = None
    if read_elem is not None:
        try:
            import h5py
            with h5py.File(path, "r") as f:
                return read_elem(f["obs"])
        except Exception as exc:  # noqa: BLE001
            print(f"  (element-wise obs read failed: {exc!r}; using a backed read)")
    a = ad.read_h5ad(path, backed="r")
    try:
        return a.obs.copy()
    finally:
        try:
            a.file.close()
        except Exception:  # noqa: BLE001
            pass


def _load_holdout_ref(path: Path) -> pd.Series:
    """Boolean held-out mask (obs['data_split'] == 'test') of a reference
    predicted_adata, indexed by obs_names."""
    obs = _read_obs(path)
    if "data_split" not in obs.columns:
        raise SystemExit(f"--holdout-from: obs['data_split'] missing in {path}")
    ref = pd.Series(obs["data_split"].astype(str).to_numpy() == "test",
                    index=obs.index.astype(str))
    if not ref.index.is_unique:
        raise SystemExit(f"--holdout-from: obs_names of {path} are not unique.")
    print(f"--holdout-from {path}: n_obs={ref.size}, n_test={int(ref.sum())}")
    return ref


def _apply_holdout_from(adata, ref: pd.Series, src: Path,
                        allow_change: bool = False) -> int:
    """Overwrite obs['data_split'] with the reference held-out mask, matched by
    obs_names. Every obs_name must be in the reference and the matched n_test
    must equal the reference n_test. The per-cell split must also be identical
    (a cell that changes split was fit on / used as a neighbour input under
    the method's own split, so held-out information could reach its score);
    a changed or uncheckable split raises unless ``allow_change``. Returns the
    number of cells that changed split (-1 = the adata had no data_split)."""
    names = adata.obs_names.astype(str)
    missing = ~names.isin(ref.index)
    if missing.any():
        raise SystemExit(
            f"--holdout-from: {int(missing.sum())}/{adata.n_obs} obs_names are not in "
            f"{src} (e.g. {list(names[missing][:3])} vs {list(ref.index[:3])}).")
    new_test = ref.reindex(names).to_numpy(dtype=bool)
    n_ref, n_new = int(ref.sum()), int(new_test.sum())
    if n_new != n_ref:
        raise SystemExit(f"--holdout-from: n_test mismatch: {n_new} matched held-out "
                         f"cells vs {n_ref} in {src}.")
    if "data_split" in adata.obs.columns:
        old_test = adata.obs["data_split"].astype(str).to_numpy() == "test"
        n_changed = int((old_test != new_test).sum())
        print(f"  [holdout-from] n_test {int(old_test.sum())} -> {n_new}; "
              f"{n_changed} cell(s) change split")
    else:
        n_changed = -1
        print("  [holdout-from] adata has no obs['data_split']; its split cannot "
              "be checked against the reference")
    if n_changed != 0:
        what = (f"{n_changed} cell(s) were predicted under a different split than"
                if n_changed > 0 else "the prediction's own split is unknown vs")
        if not allow_change:
            raise SystemExit(
                f"--holdout-from: {what} {src}. Re-run the method under the "
                "reference split, or pass --allow-split-change to score anyway "
                "(recorded in the holdout_split_changed column).")
        print(f"  WARNING: {what} {src}; scoring anyway (--allow-split-change).",
              file=sys.stderr)
    split = np.full(adata.n_obs, "train", dtype=object)
    split[new_test] = "test"
    adata.obs["data_split"] = pd.Categorical(split, categories=["train", "test"])
    return n_changed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Run dir holding predicted_adata.h5ad + metrics/ "
                        "(CSVs are rewritten into <run-dir>/metrics/).")
    p.add_argument("--adata", type=Path, default=None,
                   help="Explicit predicted_adata.h5ad (needs --out-dir).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where metrics/ is written (defaults to --run-dir).")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed label for the saved adata (default: infer from the "
                        "existing CSV, else 0).")
    p.add_argument("--n-hvg", type=int, default=50)
    p.add_argument("--cell-type-key", type=str, default=None,
                   help="obs column for the marker subset (default: auto-detect).")
    p.add_argument("--batch-key", type=str, default="adata_batch_id",
                   help="obs batch/section column for the spatial graph.")
    p.add_argument("--nbr-neighs", type=int, default=16,
                   help="Spatial kNN neighbors for the niche-branch aggregation "
                        "(X_nbr / X_hat_nbr). Default 16 (matches SQUINT's native "
                        "niche graph); was 10.")
    p.add_argument("--no-nbr", action="store_true",
                   help="Skip adding the neighborhood branch (X_hat_nbr/X_nbr).")
    p.add_argument("--nbr-observed-fill", choices=["pred", "truth", "heldout_only"],
                   default="pred",
                   help="How OBSERVED neighbours enter the niche branch: 'pred' "
                        "(default) = the method's own (LOO) prediction; 'truth' = "
                        "their true counts; 'heldout_only' = X_nbr and X_hat_nbr "
                        "both aggregate held-out neighbours plus self only. "
                        "Non-default needs --recompute-nbr on saved h5ads.")
    p.add_argument("--recompute-nbr", action="store_true",
                   help="Drop and rebuild existing X_nbr / X_hat_nbr layers "
                        "(replaces a native X_hat_nbr by the aggregated X_hat).")
    p.add_argument("--holdout-from", type=Path, default=None,
                   help="predicted_adata.h5ad whose obs['data_split'] is copied "
                        "by obs_names before scoring (asserts equal n_test and "
                        "an identical per-cell split).")
    p.add_argument("--allow-split-change", action="store_true",
                   help="With --holdout-from: score even if cells change split "
                        "(or the adata has no data_split); adds a "
                        "holdout_split_changed provenance column.")
    args = p.parse_args(argv)
    if args.allow_split_change and args.holdout_from is None:
        p.error("--allow-split-change needs --holdout-from.")

    # Opt-in leak-free knobs: never write their metrics over the original ones.
    nondefault = (args.nbr_observed_fill != "pred" or args.recompute_nbr
                  or args.holdout_from is not None)
    if nondefault:
        if args.no_nbr and (args.nbr_observed_fill != "pred" or args.recompute_nbr):
            p.error("--no-nbr conflicts with --nbr-observed-fill / --recompute-nbr.")
        src_dir = args.run_dir if args.run_dir is not None else (
            args.adata.parent if args.adata is not None else None)
        if args.out_dir is None or (
                src_dir is not None
                and args.out_dir.resolve() == Path(src_dir).resolve()):
            p.error("--nbr-observed-fill / --recompute-nbr / --holdout-from need a "
                    "NEW --out-dir (not the source run dir), so the original "
                    "metrics are not overwritten.")
    holdout_ref = (_load_holdout_ref(args.holdout_from)
                   if args.holdout_from is not None else None)

    def _uns_to_layers(adata) -> None:
        """SQUINT's predict saves X_hat / X_hat_nbr / X_nbr to adata.uns (torch
        tensors); build_pearson_dataframe reads layers. Copy any (n_obs, n_vars)
        uns tensor into the matching layer so SQUINT is scored by the SAME code
        (and metric definitions) as the baselines, using its NATIVE niche
        prediction (X_hat_nbr) — no re-aggregation. No-op for baseline adatas
        (they already have the layers)."""
        import numpy as _np
        for k in ("X_hat", "X_hat_nbr", "X_nbr"):
            if k in adata.layers or k not in adata.uns:
                continue
            v = adata.uns[k]
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else _np.asarray(v)
            if getattr(arr, "shape", None) == adata.shape:      # per-cell matrix only
                adata.layers[k] = arr
                print(f"  [uns->layers] copied uns['{k}'] -> layers['{k}'] {arr.shape}")

    def _score_one(adata_path: Path, seed: int) -> pd.DataFrame:
        print(f"Loading {adata_path}  (seed={seed})")
        adata = ad.read_h5ad(adata_path)
        n_split_changed = None
        if holdout_ref is not None:
            n_split_changed = _apply_holdout_from(
                adata, holdout_ref, args.holdout_from,
                allow_change=args.allow_split_change)
        if "data_split" not in adata.obs.columns:
            raise SystemExit(
                f"obs['data_split'] missing in {adata_path} — not holdout-split.")
        _uns_to_layers(adata)                                    # SQUINT uns -> layers
        if "X_hat" not in adata.layers:
            raise SystemExit(f"layers['X_hat'] (or uns['X_hat']) missing in {adata_path}.")
        print(f"  n_obs={adata.n_obs}, n_vars={adata.n_vars}")
        if not args.no_nbr:
            # No-op when X_hat_nbr + X_nbr are already present (SQUINT native
            # niche, or a prior run) — only aggregates for cell-only methods.
            # --recompute-nbr rebuilds them (e.g. with --nbr-observed-fill).
            add_neighborhood_layers(adata, batch_key=args.batch_key,
                                    n_neighs=args.nbr_neighs,
                                    observed_fill=args.nbr_observed_fill,
                                    force=args.recompute_nbr)
        df = build_pearson_dataframe(
            adata, seed=seed, log1p=True, n_hvg=args.n_hvg,
            cell_type_key=args.cell_type_key, verbose=True)
        if nondefault and not df.empty:
            df["nbr_observed_fill"] = args.nbr_observed_fill   # provenance column
        if args.allow_split_change and n_split_changed is not None and not df.empty:
            df["holdout_split_changed"] = n_split_changed      # -1 = unknown
        return df

    # Resolve which adata(s) to score. --run-dir now prefers the per-seed files
    # (predicted_adata_seed{N}.h5ad, one per seed) so ALL seeds are re-scored and
    # concatenated into one CSV; falls back to the legacy single predicted_adata.h5ad.
    frames = []
    if args.adata is not None:
        out_dir = args.out_dir or args.adata.parent
        # Seed label from the SOURCE metrics when writing to a new --out-dir.
        seed, warn = _resolve_seed(args.adata.parent if nondefault else out_dir, args.seed)
        if warn:
            print(f"  WARNING: {warn}", file=sys.stderr)
        frames.append(_score_one(args.adata, seed))
    elif args.run_dir is not None:
        out_dir = args.out_dir or args.run_dir
        per_seed_files = sorted(args.run_dir.glob("predicted_adata_seed*.h5ad"))
        if per_seed_files:
            print(f"Found {len(per_seed_files)} per-seed adata(s) in {args.run_dir}")
            for f in per_seed_files:
                try:
                    sd = int(f.stem.split("seed")[-1])
                except ValueError:
                    sd = 0
                frames.append(_score_one(f, sd))
        else:
            legacy = args.run_dir / "predicted_adata.h5ad"
            if not legacy.is_file():
                raise SystemExit(
                    f"no predicted_adata_seed*.h5ad or predicted_adata.h5ad in "
                    f"{args.run_dir}")
            seed, warn = _resolve_seed(args.run_dir if nondefault else out_dir, args.seed)
            if warn:
                print(f"  WARNING: {warn}", file=sys.stderr)
            print(f"  (legacy single-file fallback: {legacy})")
            frames.append(_score_one(legacy, seed))
    else:
        raise SystemExit("Pass --run-dir or --adata.")

    frames = [f for f in frames if f is not None and not f.empty]
    per_seed = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if per_seed.empty:
        raise SystemExit("build_pearson_dataframe returned no rows.")
    print(f"  scored seeds: {sorted(per_seed['seed'].dropna().astype(int).unique().tolist())}")

    have = [c for c in ("spearman_mean", "mse_mean", "rmse_mean",
                        "auroc_zero", "auprc_zero") if c in per_seed.columns]
    print(f"  panel columns now present: {have}")
    print(f"  rows: {len(per_seed)}  | axes: {sorted(per_seed['axis'].unique())}"
          f"  | subsets: {sorted(per_seed['gene_subset'].unique())}")
    write_pearson_outputs(out_dir, per_seed)
    print("DONE — re-plot with: python analysis/benchmarking/plots/"
          "plot_imputation_benchmark.py --metric panel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
