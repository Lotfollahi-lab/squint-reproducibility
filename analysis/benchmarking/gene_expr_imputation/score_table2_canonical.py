#!/usr/bin/env python
"""
Canonical Table 2 scorer: gene-expression reconstruction + imputation on ONE
neighbourhood definition for every method (mouse brain region holdout,
coord-aligned silver). Read-only with respect to every existing artifact: all
outputs go to a NEW --out-dir (default <artifacts-root>/table2_canonical/<ts>/).
Run in the squint venv on a CPU node (submit_score_table2_canonical.sh).

Canonical frame
  The coord-aligned silver is loaded with load_silver_concat(.., batch_key), the
  batch column is cast to category and apply_holdout_regions marks the held-out
  boxes (the runners' exact sequence). add_neighborhood_layers then builds the
  directed kNN(--nbr-k)+self graph per section (spatial_knn_per_batch) and the
  target X_nbr = row mean of raw X over that graph (compute_X_nbr): the code
  path every baseline runner used. Every method's rows are mapped onto the
  silver rows by obs_names (SQUINT: by section + obs_name without the ad.concat
  '-<k>' suffix, because its predicted_adata suffixes sections by sorted batch
  id and load_silver_concat by sorted file name; last resort: 1:1 nearest
  coordinate per section) and must carry an identical data_split and X.

Canonical scoring (split 'test'; build_pearson_dataframe's row logic through
_branch_pearson_rows, with the marker genes selected from the silver train
cells exactly as build_pearson_dataframe selects them)
  Reconstruction, cell: every method's stored X_hat.
  Reconstruction, neighbourhood (target = canonical X_nbr):
    scVI / Vanilla VQ-VAE: native X_hat_nbr of their *-nbr runs;
    NicheCompass: X_hat_nbr rows rescaled to the canonical X_nbr row sums
      (author decision 2; the unscaled rows, and a variant that rescales only
      the GP-predicted genes and keeps run_nichecompass's constant cohort-mean
      fill of the other genes, go to the _reference CSV);
    SQUINT: per-cell niche-decoder outputs of the PREDICT-TIME codes (stored
      obsm['neighborhood_emb'] decoded in the replayed predict batches, seed
      mode), averaged over the canonical graph (author decision 1).
  Imputation, cell: SQUINT (MC) / SQUINT (No MC) from the RE-DECODED arms'
    <seed>/metrics/heldout_predictions.npz (stage2_decode_pearson --save-pred,
    see submit_score_table2_canonical.sh for the exact command; the paper arms
    carry no npz); GeST and kNN (spatial) from their stored X_hat.
  Imputation, neighbourhood (target = canonical X_nbr): canonical-graph mean of
    F with F[held-out] = the method's held-out prediction and F[observed] = the
    TRUE counts X for every method (parity arm, author decision 3). The
    'ownfill' variant (SQUINT observed = stored stage-1 X_hat; GeST / kNN
    observed = their own leave-one-out X_hat) is written for comparison.

Self-checks (PASS / FAIL / WARN / INFO lines, collected in selfcheck_report.txt;
a FAIL never stops the run, it is flagged loudly and the exit code is 1)
  S0  the test-only scorer used for the canonical rows == build_pearson_dataframe
      test rows on the same stored data
  S1  every baseline seed h5ad re-scored as stored with build_pearson_dataframe
      == its runner per-seed CSV (cell + niche, split test); identical
      obs_names / data_split / X vs the silver; stored X_nbr and obsp graph ==
      the canonical ones (#rows differing reported)
  S2  each SQUINT predicted_adata re-scored as stored == the multiseed per-seed
      CSV; identical data_split / X after the row mapping; the canonical-order
      re-score of the stored X_hat equals it too (only row order changes)
  S3  replay of SQUINT's predict pass (predict-time ckpt from uns, else best
      val_loss; predict batch size and subgraphs as in the farm-verified
      validate_table2.sh), gated on the deterministic part, the decoders on the
      STORED embeddings: obsm['cell_emb'] reproduces the stored X_hat, and
      obsm['neighborhood_emb'] aggregated over the predict subgraph reproduces
      the stored X_hat_nbr on the rows whose in-neighbours are all seeds of
      their batch (--replay-tol, max row rel-L2). The full forward re-quantises
      the codes on this device and is INFO only (code-equality fractions, rows
      above tol)
  S4  each re-decoded SQUINT imputation arm: heldout_predictions.npz cell rows
      re-scored == the arm's per-seed CSV cell rows; neighbourhood re-scored
      with the decode's own convention (observed = stored X_hat or truth per
      the npz provenance; graph = stage2_decode_pearson's sklearn kNN+self
      built on the predicted_adata coordinates per section) == the CSV niche
      rows. Decode provenance, each a FAIL (the canonical rows use the npz
      x_hat only): this seed's predicted_adata; the published stage-2 codes
      (--stage2-codes-dir); K = --mc-decode-samples (MC) / 1 hard (No MC);
      smooth_neighs 0; read depth 'neighbor' with 16 neighbours, and the npz
      read_depth == the leak-free depth recomputed here, x_hat rows summing
      to it and not to the true library; npz x_true == X[global_idx] (row
      order); true-code self-check >= --selfcheck-min; decode ckpt == the
      predict-time ckpt (a WARN with
      --allow-ckpt-mismatch). Link to the published numbers: re-decoded CSV
      cell rows == the paper arm CSV cell rows (--asstored-*-arm-dir; rel tol
      --tol-redecode-mc for MC, --tol-redecode for No MC, as the decode's own
      --compare-csv gate). A difference explained by a different decode
      checkpoint is a WARN (re-decode drift, delta reported), otherwise FAIL;
      the FAIL names a pre-fdca41a true-depth paper arm when its CSV mtime
      predates the read-depth fix commits.
  S5  (--existing-tables-dir) as-stored panel rows == the existing panel CSVs

Outputs (in --out-dir)
  reconstruction_benchmark_panel_{cell,niche}.csv   canonical (fair) panels
  imputation_benchmark_panel_{cell,niche}.csv       canonical (parity) panels
  imputation_benchmark_panel_niche_ownfill.csv      observed = own fill
  reconstruction_benchmark_panel_niche_reference.csv
        NicheCompass (unscaled), NicheCompass (GP genes rescaled) and
        SQUINT (native graph, as stored)
  asstored/{reconstruction,imputation}_benchmark_panel_{cell,niche}.csv
        the stored numbers through the same panel mapping (current Table 2;
        SQUINT imputation rows from the PAPER arms' CSVs)
  redecoded/imputation_benchmark_panel_{cell,niche}.csv
        the re-decoded SQUINT arms' own CSV rows (summary 'drift' line =
        redecoded - asstored, i.e. re-decode drift apart from the canonical change)
  per_seed/<set>/<panel file>/<method>.csv   per-seed frames behind each panel
  summary_means.csv, selfcheck_report.txt, run_config.json
The panel CSVs carry the existing columns
(metric,panel,axis,transform,gene_subset,branch,method,split,seed,value), the
published method labels and the panel codes of plot_imputation_benchmark's
_GRID_ROWS (rows built with its _load_csv_filtered, the loop of
render_metric_grid without the figure), so
  python analysis/benchmarking/compute_table2_significance.py --tables-dir <out-dir>
works unchanged (and --tables-dir <out-dir>/asstored gives the current stars).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import re
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")   # the plot module imports pyplot

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

_THIS = Path(__file__).resolve().parent
_BENCH = _THIS.parent
_PLOTS = _BENCH / "plots"
for _p in (_THIS, _BENCH, _PLOTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from _holdout_utils import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT, DEFAULT_DATASET_TAG, DEFAULT_HOLDOUT_REGIONS,
    _MARKER_LABEL_KEYS, _branch_pearson_rows, _neighbor_mean,
    _select_marker_indices, _to_dense_2d, add_neighborhood_layers,
    apply_holdout_regions, build_pearson_dataframe, load_silver_concat,
)
from plot_imputation_benchmark import _GRID_ROWS, _load_csv_filtered  # noqa: E402
from compute_table2_significance import COLUMNS as TABLE2_COLUMNS  # noqa: E402


# ---------------------------------------------------------------------------
# Paper defaults (Table 2, mouse brain, coord-aligned re-runs)
# ---------------------------------------------------------------------------

DEFAULT_SILVER_DIR = ("/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver/"
                      "mmb0-1b_smb1-1b_1p_coord_aligned")
_SQUINT_SPINE = ("dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16"
                 "+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc"
                 "+diversity-w10+filmscale+crossmnn-wt10-k1")
DEFAULT_SQUINT_SEEDS = ["20260629_023326_seed0", "20260629_024131_seed1",
                        "20260629_025158_seed2", "20260629_025456_seed3",
                        "20260629_025518_seed4"]
DEFAULT_MULTISEED_TS = "20260628_230209"
DEFAULT_STAGE2_ROOT = "stage2-ablation"
DEFAULT_MC_ARM = "decode-temp05-nonoise-mc1000-nichecell"
DEFAULT_NOMC_ARM = "decode-temp05-nonoise-nichecell"
DEFAULT_CODES_ARM = "decode-temp05-nonoise"      # trained stage-2 arm holding the codes
CODES_NAME = "stage2_predicted_codes.npz"
PAPER_READ_DEPTH_NEIGHS = 16                     # submit_stage2_mc_sweep.sh READ_DEPTH_NEIGHS
NPZ_NAME = "heldout_predictions.npz"
CSV_NAME = "per_seed_pearson_reconstruction.csv"

# baseline key (submit_imputation_recon.sh method names) -> (default variant
# folder, roles). A role is (block, branch, published label).
BASELINES: "OrderedDict[str, Tuple[str, List[Tuple[str, str, str]]]]" = OrderedDict([
    ("scvi",         ("baseline-scvi+region-holdout+coord-aligned",
                      [("recon", "cell", "scVI")])),
    ("scvi-nbr",     ("baseline-scvi-nbr+region-holdout+coord-aligned",
                      [("recon", "niche", "scVI")])),
    ("nichecompass", ("baseline-nichecompass+region-holdout+coord-aligned",
                      [("recon", "cell", "NicheCompass"),
                       ("recon", "niche", "NicheCompass")])),
    ("vanilla-cell", ("vanilla-vq-cell+region-holdout+coord-aligned",
                      [("recon", "cell", "Vanilla VQ-VAE")])),
    ("vanilla-nbr",  ("vanilla-vq-nbr+region-holdout+coord-aligned",
                      [("recon", "niche", "Vanilla VQ-VAE")])),
    ("gest",         ("gest-imputed+region-holdout+coord-aligned",
                      [("imp", "cell", "GeST"), ("imp", "niche", "GeST")])),
    ("knn",          ("baseline-knn-spatial+region-holdout+coord-aligned",
                      [("imp", "cell", "kNN (spatial)"),
                       ("imp", "niche", "kNN (spatial)")])),
])

RECON_LABELS = ["scVI", "NicheCompass", "Vanilla VQ-VAE", "SQUINT"]
IMP_LABELS = ["SQUINT (MC)", "GeST", "SQUINT (No MC)", "kNN (spatial)"]
NC_UNSCALED = "NicheCompass (unscaled)"
NC_GPONLY = "NicheCompass (GP genes rescaled)"
SQ_NATIVE = "SQUINT (native graph, as stored)"
SQ_ARMS = ["SQUINT (MC)", "SQUINT (No MC)"]

# (set, stem, branch, labels): one long panel CSV each.
# set '' = canonical (out-dir root), 'asstored' = stored numbers,
# 'redecoded' = the re-decoded SQUINT arms' own CSV rows.
PANEL_FILES = [
    ("", "reconstruction_benchmark_panel_cell", "cell", RECON_LABELS),
    ("", "reconstruction_benchmark_panel_niche", "niche", RECON_LABELS),
    ("", "imputation_benchmark_panel_cell", "cell", IMP_LABELS),
    ("", "imputation_benchmark_panel_niche", "niche", IMP_LABELS),
    ("", "imputation_benchmark_panel_niche_ownfill", "niche", IMP_LABELS),
    ("", "reconstruction_benchmark_panel_niche_reference", "niche",
     [NC_UNSCALED, NC_GPONLY, SQ_NATIVE]),
    ("asstored", "reconstruction_benchmark_panel_cell", "cell", RECON_LABELS),
    ("asstored", "reconstruction_benchmark_panel_niche", "niche", RECON_LABELS),
    ("asstored", "imputation_benchmark_panel_cell", "cell", IMP_LABELS),
    ("asstored", "imputation_benchmark_panel_niche", "niche", IMP_LABELS),
    ("redecoded", "imputation_benchmark_panel_cell", "cell", SQ_ARMS),
    ("redecoded", "imputation_benchmark_panel_niche", "niche", SQ_ARMS),
]
MAIN_STEMS = ["reconstruction_benchmark_panel_cell",
              "reconstruction_benchmark_panel_niche",
              "imputation_benchmark_panel_cell",
              "imputation_benchmark_panel_niche"]
PANEL_COLUMNS = ["metric", "panel", "axis", "transform", "gene_subset", "branch",
                 "method", "split", "seed", "value"]

_KEY_COLS = ["branch", "split", "axis", "transform", "gene_subset"]
_VAL_COLS = ["pearson_mean", "pearson_median", "spearman_mean", "spearman_median",
             "mse_mean", "mse_median", "rmse_mean", "auroc_zero", "auprc_zero",
             "n_cells", "n_genes"]
_SUFFIX = re.compile(r"-\d+$")          # ad.concat(index_unique='-') suffix


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class Report:
    """PASS / FAIL / WARN / SKIP / INFO lines, echoed and kept for the report."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.n = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0, "INFO": 0}

    def _emit(self, status: str, cid: str, msg: str) -> None:
        line = f"[{status}] {cid}: {msg}"
        if status == "FAIL":
            line = "!!! " + line
        print(line, flush=True)
        self.lines.append(line)
        self.n[status] = self.n.get(status, 0) + 1

    def check(self, cid: str, ok: bool, msg: str, warn_only: bool = False) -> bool:
        self._emit("PASS" if ok else ("WARN" if warn_only else "FAIL"), cid, msg)
        return bool(ok)

    def fail(self, cid: str, msg: str) -> None:
        self._emit("FAIL", cid, msg)

    def skip(self, cid: str, msg: str) -> None:
        self._emit("SKIP", cid, msg)

    def info(self, cid: str, msg: str) -> None:
        self._emit("INFO", cid, msg)

    def section(self, title: str) -> None:
        bar = "=" * 100
        print(f"\n{bar}\n{title}\n{bar}", flush=True)
        self.lines.append(f"\n== {title} ==")

    def guarded(self, cid: str, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception:  # noqa: BLE001
            self.fail(cid, "exception:\n" + traceback.format_exc())
            return None

    def counts(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.n.items())


class Store:
    """Per-seed frames keyed (set, stem, label)."""

    def __init__(self) -> None:
        self.frames: Dict[Tuple[str, str, str], List[pd.DataFrame]] = defaultdict(list)

    def add(self, set_name: str, stem: str, label: str, df: Optional[pd.DataFrame],
            branch: str, seed: int) -> None:
        if df is None or df.empty:
            return
        d = df[(df["branch"] == branch) & (df["split"] == "test")].copy()
        if d.empty:
            return
        d["seed"] = int(seed)
        self.frames[(set_name, stem, label)].append(d)

    def get(self, set_name: str, stem: str, label: str) -> List[pd.DataFrame]:
        return self.frames.get((set_name, stem, label), [])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _dense32(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _get_mat(a: ad.AnnData, k: str) -> Optional[np.ndarray]:
    """layers[k], else an (n_obs, n_vars) uns[k] (older SQUINT predicts)."""
    if k in a.layers:
        return _to_dense_2d(a.layers[k])
    if k in a.uns:
        v = _dense32(a.uns[k])
        if v is not None and v.shape == tuple(a.shape):
            return v
    return None


def _norm_sec(v) -> np.ndarray:
    out = []
    for x in np.asarray(v):
        try:
            out.append(str(int(float(x))))
        except (TypeError, ValueError):
            out.append(str(x))
    return np.asarray(out, dtype=object)


def _seed_of(name: str) -> int:
    """Seed from a '..._seedN' / 'predicted_adata_seedN' name (last match, as
    plot_imputation_benchmark._load_csv_filtered reads it)."""
    mm = re.findall(r"seed_?(\d+)", str(name))
    if not mm:
        raise ValueError(f"no seed number in {name!r}")
    return int(mm[-1])


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def _test_mask(obs: pd.DataFrame) -> np.ndarray:
    return obs["data_split"].astype(str).to_numpy() == "test"


def _order_cols(df: pd.DataFrame) -> pd.DataFrame:
    """build_pearson_dataframe's column order."""
    lead = ["seed", "branch", "split", "axis", "transform", "gene_subset",
            "pearson_mean", "pearson_median"]
    panel = ["spearman_mean", "spearman_median", "mse_mean", "mse_median",
             "rmse_mean", "auroc_zero", "auprc_zero"]
    tail = ["n_cells", "n_genes"]
    ordered = [c for c in lead if c in df.columns]
    ordered += [c for c in panel if c in df.columns]
    ordered += [c for c in tail if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    return df[ordered]


def _marker_idx_bpd(X: np.ndarray, obs: pd.DataFrame,
                    cell_type_key: Optional[str] = None):
    """Marker genes exactly as build_pearson_dataframe selects them (truth on
    the train cells, first present label key)."""
    split_col = obs["data_split"].astype("object").to_numpy()
    lab_key = cell_type_key or next(
        (k for k in _MARKER_LABEL_KEYS if k in obs.columns), None)
    if lab_key is None or lab_key not in obs.columns:
        return None, None
    labels_all = obs[lab_key].to_numpy()
    train_mask = split_col != "test"
    if train_mask.sum() == 0:
        train_mask = np.ones(len(obs), dtype=bool)
    tl = np.log1p(np.clip(X[train_mask], 0, None))
    return _select_marker_indices(tl, labels_all[train_mask]), lab_key


def _score_test(target: np.ndarray, pred: np.ndarray, test: np.ndarray,
                marker_idx, seed: int, branch: str, n_hvg: int) -> pd.DataFrame:
    """build_pearson_dataframe's rows for ONE branch and split 'test' (same
    _branch_pearson_rows call; float32 dense inputs as it uses)."""
    rows = _branch_pearson_rows(
        branch=branch, target_full=_to_dense_2d(target), pred_full=_to_dense_2d(pred),
        cell_mask=np.asarray(test, dtype=bool), split_label="test", log1p=True,
        n_hvg=n_hvg, marker_idx=marker_idx)
    for r in rows:
        r["seed"] = int(seed)
    return _order_cols(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _compare_frames(new: pd.DataFrame, ref: pd.DataFrame, branch: str,
                    split: str = "test", rel: bool = False):
    """(max |diff| over shared metric cells, #rows compared, #key rows only on
    one side). NaN vs NaN counts as equal; NaN vs a number as inf. rel=True:
    |a - b| / max(|a|, |b|, 1) (for float summation-order differences)."""
    def sel(d):
        if d is None or not set(_KEY_COLS) <= set(d.columns):
            return pd.DataFrame(columns=_KEY_COLS)
        d = d[(d["branch"] == branch) & (d["split"] == split)].copy()
        d["gene_subset"] = d["gene_subset"].astype(str)
        return d
    a, b = sel(new), sel(ref)
    if a.empty and b.empty:
        return None
    m = a.merge(b, on=_KEY_COLS, how="outer", suffixes=("_a", "_b"), indicator=True)
    n_only = int((m["_merge"] != "both").sum())
    both = m[m["_merge"] == "both"]
    dmax = 0.0
    for c in _VAL_COLS:
        ca, cb = c + "_a", c + "_b"
        if ca not in both.columns or cb not in both.columns:
            continue
        va = both[ca].to_numpy(dtype=float)
        vb = both[cb].to_numpy(dtype=float)
        na, nb = np.isnan(va), np.isnan(vb)
        if (na != nb).any():
            dmax = float("inf")
        ok = ~(na | nb)
        if ok.any():
            d = np.abs(va[ok] - vb[ok])
            if rel:
                d = d / np.maximum(np.maximum(np.abs(va[ok]), np.abs(vb[ok])), 1.0)
            dmax = max(dmax, float(d.max()))
    return dmax, int(len(both)), n_only


def _check_frames(R: Report, cid: str, new: pd.DataFrame, ref: pd.DataFrame,
                  branches, tol: float, what: str, rel: bool = False) -> None:
    for br in branches:
        res = _compare_frames(new, ref, br, rel=rel)
        if res is None:
            R.skip(f"{cid}/{br}", f"no {br} rows on either side ({what})")
            continue
        dmax, n, n_only = res
        R.check(f"{cid}/{br}", n > 0 and n_only == 0 and dmax <= tol,
                f"{what}: max {'rel ' if rel else ''}|diff|={dmax:.3e} over {n} rows, "
                f"{n_only} unmatched row(s) (tol {tol:g})")


def _coord_map(m_coords, m_sec, s_coords, s_sec, tol: float):
    """1:1 nearest-coordinate match within each section (validate_table2.sh's
    join). Returns (idx or None, max distance)."""
    from scipy.spatial import cKDTree
    idx = np.full(len(s_sec), -1, dtype=np.int64)
    dmax = 0.0
    for s in np.unique(s_sec):
        ia, ib = np.where(s_sec == s)[0], np.where(m_sec == s)[0]
        if ib.size != ia.size:
            return None, float("inf")
        d, j = cKDTree(m_coords[ib]).query(s_coords[ia], k=1)
        idx[ia] = ib[j]
        dmax = max(dmax, float(d.max()) if d.size else 0.0)
    if (idx < 0).any() or np.unique(idx).size != idx.size or dmax > tol:
        return None, dmax
    return idx, dmax


def _row_map(m_names, m_sec, s_names, s_sec, m_coords=None, s_coords=None,
             coord_tol: float = 0.05):
    """For every canonical (silver) row, the row index in the method's adata.
    1) exact obs_names (accepted only if every pair is in the same section);
    2) (section, obs_name without the ad.concat '-<k>' suffix), which survives
       a different section order in the two concats (SQUINT's predicted_adata
       suffixes by sorted batch id, load_silver_concat by sorted file name);
    3) 1:1 nearest coordinate within the section (<= coord_tol).
    The caller then asserts identical data_split and X on the mapped rows."""
    m_names = np.asarray(m_names).astype(str)
    s_names = np.asarray(s_names).astype(str)
    mi, si = pd.Index(m_names), pd.Index(s_names)
    info = {"same_name_set": bool(len(mi) == len(si) and mi.is_unique and si.is_unique
                                  and set(m_names) == set(s_names))}
    if info["same_name_set"]:
        idx = mi.get_indexer(si)
        if (idx >= 0).all() and (m_sec[idx] == s_sec).all():
            return idx, "obs_names", info
        info["exact_names_section_mismatch"] = int((m_sec[idx] != s_sec).sum())
    mk = pd.Index([f"{s}|{_SUFFIX.sub('', x)}" for s, x in zip(m_sec, m_names)])
    sk = pd.Index([f"{s}|{_SUFFIX.sub('', x)}" for s, x in zip(s_sec, s_names)])
    if len(mk) == len(sk) and mk.is_unique and sk.is_unique:
        idx = mk.get_indexer(sk)
        if (idx >= 0).all():
            return idx, "section+base_name", info
    if m_coords is not None and s_coords is not None and len(m_names) == len(s_names):
        idx, dmax = _coord_map(m_coords, m_sec, s_coords, s_sec, coord_tol)
        info["coord_max_dist"] = dmax
        if idx is not None:
            return idx, "section+nearest_coord", info
    return None, "unmapped", info


def _gene_map(m_var, s_var) -> Optional[np.ndarray]:
    m_var = pd.Index(np.asarray(m_var).astype(str))
    s_var = pd.Index(np.asarray(s_var).astype(str))
    if len(m_var) != len(s_var) or not m_var.is_unique:
        return None
    c = m_var.get_indexer(s_var)
    return c if (c >= 0).all() else None


def _align(M: Optional[np.ndarray], r: np.ndarray, c: np.ndarray):
    if M is None:
        return None
    ident_r = np.array_equal(r, np.arange(r.size))
    ident_c = np.array_equal(c, np.arange(c.size))
    if ident_r and ident_c:
        return M
    out = M if ident_r else M[r]
    return out if ident_c else out[:, c]


def _graph_rows_differing(A1, A2) -> np.ndarray:
    """Row indices where two sparse graphs differ (pattern or value)."""
    D = (sp.csr_matrix(A1, dtype=np.float32) - sp.csr_matrix(A2, dtype=np.float32)).tocsr()
    D.eliminate_zeros()
    return np.unique(D.nonzero()[0])


def _rowrel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-8)


def _ckpt_param_diff(a: str, b: str) -> Optional[float]:
    """max |param diff| between two Lightning checkpoints (None if unreadable)."""
    try:
        import torch

        def _load(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except TypeError:                       # torch < 1.13: no weights_only
                return torch.load(p, map_location="cpu")
        da, db = _load(a)["state_dict"], _load(b)["state_dict"]
        mx = 0.0
        for k, v in da.items():
            if k in db and db[k].shape == v.shape and v.is_floating_point():
                mx = max(mx, float((v.float() - db[k].float()).abs().max()))
        return mx
    except Exception:  # noqa: BLE001
        return None


def _import_s2d(examples_dir: Optional[Path]):
    """stage2_decode_pearson (squint examples) for the decode's own scoring
    and kNN helpers, plus its uns-checkpoint helpers."""
    if examples_dir is None:
        import squint  # noqa: F401
        examples_dir = Path(sys.modules["squint"].__file__).resolve().parents[2] / "examples"
    p = Path(examples_dir) / "stage2_decode_pearson.py"
    if not p.is_file():
        raise SystemExit(f"stage2_decode_pearson.py not found at {p} (--squint-examples-dir)")
    spec = importlib.util.spec_from_file_location("stage2_decode_pearson", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, p


def _resolve_run_dir(path: Path, what: str) -> Path:
    """Run dir = the folder holding predicted_adata_seed*.h5ad. Given a variant
    folder, its single timestamp sub-folder (error if none or several)."""
    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f"{what}: {path} is not a directory")
    if sorted(path.glob("predicted_adata_seed*.h5ad")):
        return path
    subs = sorted(d for d in path.iterdir()
                  if d.is_dir() and sorted(d.glob("predicted_adata_seed*.h5ad")))
    if len(subs) != 1:
        raise SystemExit(
            f"{what}: expected exactly ONE timestamp folder with "
            f"predicted_adata_seed*.h5ad under {path}, found {len(subs)}: "
            f"{[d.name for d in subs]}. Pass the run dir via --baseline-dirs.")
    return subs[0]


# ---------------------------------------------------------------------------
# Canonical frame
# ---------------------------------------------------------------------------

def build_canonical(args, R: Report) -> dict:
    R.section("Canonical frame: silver, held-out split, directed kNN+self graph, X_nbr")
    t0 = time.time()
    excl = os.environ.get("SQUINT_EXCLUDE_BATCHES", "")
    if excl.strip():
        R.fail("canon/env", f"SQUINT_EXCLUDE_BATCHES={excl!r} is set; the silver "
                            "loader would drop sections. Unset it.")
    silver = load_silver_concat(Path(args.silver_dir), batch_key=args.batch_key)
    if args.batch_key not in silver.obs.columns:
        raise SystemExit(f"--batch-key={args.batch_key!r} missing from the silver obs")
    silver.obs[args.batch_key] = silver.obs[args.batch_key].astype("category")
    apply_holdout_regions(silver, batch_key=args.batch_key, regions=DEFAULT_HOLDOUT_REGIONS)
    for k in ("X_hat", "X_hat_nbr", "X_nbr"):
        if k in silver.layers:
            raise SystemExit(f"silver unexpectedly carries layers[{k!r}]")
    if "spatial_connectivities" in silver.obsp:
        # (ad.concat drops obsp, so this only triggers for a single-file silver)
        R.info("canon/graph", "silver carries obsp['spatial_connectivities']; rebuilt")
        del silver.obsp["spatial_connectivities"]
    add_neighborhood_layers(silver, batch_key=args.batch_key, n_neighs=args.nbr_k)
    if "X_nbr" not in silver.layers or "spatial_connectivities" not in silver.obsp:
        raise SystemExit("add_neighborhood_layers did not build the graph / X_nbr "
                         "(obsm['spatial'] missing in the silver?)")

    sec = _norm_sec(silver.obs[args.batch_key].to_numpy())
    # spatial_knn_per_batch block-diagonalises per-section graphs in order of
    # first appearance: valid only if every section is one contiguous block.
    first = list(dict.fromkeys(sec.tolist()))
    blocks = [s for i, s in enumerate(sec) if i == 0 or sec[i - 1] != s]
    R.check("canon/graph-blocks", blocks == first,
            f"sections {first} are contiguous row blocks in first-appearance order "
            f"(block_diag assumption of spatial_knn_per_batch)")
    A = sp.csr_matrix(silver.obsp["spatial_connectivities"], dtype=np.float32)
    deg = np.diff(A.indptr)
    R.check("canon/graph-degree", bool((deg == args.nbr_k + 1).all()),
            f"entries per row: unique {np.unique(deg)[:6].tolist()} (expect "
            f"{args.nbr_k + 1} = k + self); self-loops {int((A.diagonal() > 0).sum())}"
            f"/{A.shape[0]}; asymmetric entries {int((A != A.T).nnz)}; values "
            f"{np.unique(A.data)[:4].tolist()}")
    # no cross-section edges
    rr, cc = A.nonzero()
    R.check("canon/graph-sections", bool((sec[rr] == sec[cc]).all()),
            "no edge crosses sections")
    test = _test_mask(silver.obs)
    X = _to_dense_2d(silver.X)
    X_nbr = _to_dense_2d(silver.layers["X_nbr"])
    marker_idx, lab_key = _marker_idx_bpd(X, silver.obs)
    per_sec = {s: (int((sec == s).sum()), int(((sec == s) & test).sum())) for s in first}
    R.info("canon/frame", f"n_obs={silver.n_obs} n_vars={silver.n_vars} test={int(test.sum())}; "
                          f"per section (n, test) {per_sec}; markers "
                          f"{0 if marker_idx is None else marker_idx.size} genes from "
                          f"obs[{lab_key!r}] (train cells); built in {time.time() - t0:.0f}s")
    C = dict(names=np.asarray(silver.obs_names).astype(str), sec=sec, coords=_xy(silver),
             split=silver.obs["data_split"].astype(str).to_numpy(), test=test,
             X=X, X_nbr=X_nbr, A=A, var=np.asarray(silver.var_names).astype(str),
             marker_idx=marker_idx, lab_key=lab_key, n=silver.n_obs)
    del silver
    gc.collect()
    return C


def _xy(a: ad.AnnData) -> Optional[np.ndarray]:
    if "spatial" not in a.obsm:
        return None
    return np.asarray(a.obsm["spatial"], dtype=np.float64)[:, :2]


def _map_and_check(R: Report, cid: str, C: dict, names, sec, var, split, X,
                   expect_exact: bool, tol: float, coords=None):
    """Row/gene mapping onto the canonical frame plus the identity checks
    (data_split, X; coordinates reported). Returns (r, c) or (None, None)."""
    r, how, info = _row_map(names, sec, C["names"], C["sec"], coords, C["coords"])
    if r is None:
        R.fail(f"{cid}/rows", f"rows could not be mapped onto the silver ({info})")
        return None, None
    if expect_exact:
        R.check(f"{cid}/rows", how == "obs_names",
                f"mapped by {how}; identical obs_name set={info['same_name_set']}")
    else:
        R.info(f"{cid}/rows", f"mapped by {how}; identical obs_name set="
                              f"{info['same_name_set']} {info}")
    if coords is not None and C["coords"] is not None:
        dxy = np.linalg.norm(coords[r] - C["coords"], axis=1)
        R.info(f"{cid}/coords", f"mapped pairs: max xy distance {float(dxy.max()):.3g}, "
                                f"{int((dxy > 0).sum())} pair(s) not bit-identical")
    c = _gene_map(var, C["var"])
    if c is None:
        R.fail(f"{cid}/genes", "var_names are not the silver's gene set")
        return None, None
    R.check(f"{cid}/genes", True, f"same {c.size} genes; identical order="
                                  f"{bool(np.array_equal(c, np.arange(c.size)))}")
    sp_m = np.asarray(split).astype(str)[r]
    n_bad = int((sp_m != C["split"]).sum())
    R.check(f"{cid}/data_split", n_bad == 0,
            f"{n_bad} row(s) with a different data_split (test: method "
            f"{int((sp_m == 'test').sum())} vs canonical {int(C['test'].sum())})")
    Xa = _align(X, r, c)
    dx = float(np.abs(Xa - C["X"]).max())
    R.check(f"{cid}/X", dx <= tol, f"max |X - X_silver| = {dx:.3g}")
    return r, c


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def process_baseline(key: str, run_dir: Path, C: dict, args, R: Report, S: Store) -> None:
    R.section(f"[baseline {key}] {run_dir}")
    roles = BASELINES[key][1]
    csv_path = run_dir / "metrics" / CSV_NAME
    CS = pd.read_csv(csv_path) if csv_path.is_file() else None
    if CS is None:
        R.fail(f"S1/{key}", f"runner per-seed CSV missing: {csv_path}")
    files = sorted(run_dir.glob("predicted_adata_seed*.h5ad"), key=lambda f: _seed_of(f.stem))
    seeds = [_seed_of(f.stem) for f in files]
    R.check(f"S1/{key}/seeds", len(files) == 5,
            f"{len(files)} seed h5ad(s) {seeds}; CSV seeds "
            f"{sorted(CS['seed'].unique().tolist()) if CS is not None else None}",
            warn_only=True)
    for i, f in enumerate(files):
        R.guarded(f"S1/{key}/seed{seeds[i]}", _process_baseline_seed,
                  key, roles, f, seeds[i], i, CS, C, args, R, S)
        gc.collect()


def _process_baseline_seed(key, roles, f: Path, seed: int, i: int, CS, C, args, R, S):
    t0 = time.time()
    cid = f"{key}/seed{seed}"
    B = ad.read_h5ad(f)
    X = _to_dense_2d(B.X)
    Xh, Xhn, Xn = _get_mat(B, "X_hat"), _get_mat(B, "X_hat_nbr"), _get_mat(B, "X_nbr")
    test_b = _test_mask(B.obs)
    branches = [b for b, m in (("cell", Xh), ("niche", Xhn)) if m is not None]
    # ---- S0 / S1: re-score as stored ------------------------------------
    mk_b, _ = _marker_idx_bpd(X, B.obs)
    fast = []
    if Xh is not None:
        fast.append(_score_test(X, Xh, test_b, mk_b, seed, "cell", args.n_hvg))
    if Xhn is not None:
        if Xn is None:
            R.fail(f"S1/{cid}", "X_hat_nbr without X_nbr")
        else:
            fast.append(_score_test(Xn, Xhn, test_b, mk_b, seed, "niche", args.n_hvg))
    fast = pd.concat(fast, ignore_index=True) if fast else pd.DataFrame()
    literal = (not args.fast_selfchecks) or i == 0
    if literal:
        lit = build_pearson_dataframe(B, seed=seed, log1p=True, n_hvg=args.n_hvg)
        lit = lit[lit["split"] == "test"]
        _check_frames(R, f"S0/{cid}", fast, lit, branches, args.tol,
                      "test-only scorer vs build_pearson_dataframe")
        stored = lit
    else:
        stored = fast
    if CS is not None:
        ref = CS[CS["seed"] == seed]
        if ref.empty:
            R.fail(f"S1/{cid}", f"seed {seed} not in the runner CSV")
        else:
            _check_frames(R, f"S1/{cid}", stored, ref, branches, args.tol,
                          ("build_pearson_dataframe" if literal else "test-only scorer")
                          + " on the stored h5ad vs runner CSV")
    # ---- identity vs the silver -------------------------------------------
    r, c = _map_and_check(R, f"S1/{cid}", C, B.obs_names, _norm_sec(B.obs[args.batch_key]),
                          B.var_names, B.obs["data_split"], X, True, 0.0, coords=_xy(B))
    if r is None:
        return
    if Xn is not None:
        d = np.abs(_align(Xn, r, c) - C["X_nbr"])
        n_rows = int((d.max(axis=1) > args.tol).sum())
        R.check(f"S1/{cid}/X_nbr", n_rows == 0,
                f"stored X_nbr vs canonical: {n_rows} row(s) differ (> {args.tol:g}; "
                f"{int((d[C['test']].max(axis=1) > args.tol).sum())} test), max |diff| "
                f"{float(d.max()):.3g}")
    if "spatial_connectivities" in B.obsp:
        Ab = sp.csr_matrix(B.obsp["spatial_connectivities"], dtype=np.float32)
        if not np.array_equal(r, np.arange(r.size)):
            Ab = Ab[r][:, r]
        rows = _graph_rows_differing(Ab, C["A"])
        R.check(f"S1/{cid}/graph", rows.size == 0,
                f"stored obsp graph vs canonical: {rows.size} row(s) differ "
                f"({int(C['test'][rows].sum()) if rows.size else 0} test)")
    else:
        R.skip(f"S1/{cid}/graph", "no obsp['spatial_connectivities'] in the h5ad")
    # ---- canonical + as-stored frames -------------------------------------
    Xh_c, Xhn_c = _align(Xh, r, c), _align(Xhn, r, c)
    T, M, k = C["test"], C["marker_idx"], args.n_hvg
    for block, branch, label in roles:
        if block == "recon" and branch == "cell":
            stem = "reconstruction_benchmark_panel_cell"
            S.add("", stem, label, _score_test(C["X"], Xh_c, T, M, seed, "cell", k), "cell", seed)
            S.add("asstored", stem, label, stored, "cell", seed)
        elif block == "recon" and branch == "niche":
            stem = "reconstruction_benchmark_panel_niche"
            if Xhn_c is None:
                R.fail(f"canon/{cid}", "no X_hat_nbr for a neighbourhood role")
                continue
            S.add("asstored", stem, label, stored, "niche", seed)
            if key == "nichecompass":
                # decision 2: rescale each row to the canonical X_nbr library
                rs = C["X_nbr"].sum(1) / np.maximum(Xhn_c.sum(1), 1e-8)
                Xhn_rs = Xhn_c * rs[:, None]
                S.add("", stem, label, _score_test(C["X_nbr"], Xhn_rs, T, M, seed, "niche", k),
                      "niche", seed)
                S.add("", "reconstruction_benchmark_panel_niche_reference", NC_UNSCALED,
                      _score_test(C["X_nbr"], Xhn_c, T, M, seed, "niche", k), "niche", seed)
                # run_nichecompass fills genes outside every GP source mask with
                # a constant (cohort per-gene mean of X); the row rescale above
                # also scales those. Variant: rescale only the GP-predicted genes
                # to the X_nbr library on those genes, constant fill untouched.
                const = np.ptp(Xhn_c, axis=0) == 0
                gp = ~const
                Xhn_gp = Xhn_c.copy()
                rs_gp = C["X_nbr"][:, gp].sum(1) / np.maximum(Xhn_c[:, gp].sum(1), 1e-8)
                Xhn_gp[:, gp] *= rs_gp[:, None]
                S.add("", "reconstruction_benchmark_panel_niche_reference", NC_GPONLY,
                      _score_test(C["X_nbr"], Xhn_gp, T, M, seed, "niche", k), "niche", seed)
                share = (Xhn_c[T][:, const].sum(1)
                         / np.maximum(Xhn_c[T].sum(1), 1e-8)) if const.any() else np.zeros(1)
                ratio = Xhn_c[T].sum(1) / np.maximum(C["X_nbr"][T].sum(1), 1e-8)
                indeg = np.asarray(C["A"].sum(axis=0)).ravel()[T]
                rho = pd.Series(ratio).corr(pd.Series(indeg), method="spearman")
                R.info(f"canon/{cid}/nc-scale",
                       f"X_hat_nbr/X_nbr row-sum ratio (test): mean {ratio.mean():.4f} "
                       f"CV {ratio.std() / max(ratio.mean(), 1e-12):.4f}; Spearman(ratio, "
                       f"in-degree) {rho:.3f}; rescaled to the X_nbr row sums. Constant "
                       f"(non-GP, cohort-mean filled) genes: {int(const.sum())}/{const.size}, "
                       f"mean share of the X_hat_nbr row sum (test) {float(share.mean()):.4f}"
                       + (f"; these are rescaled too (decision 2 as applied), see "
                          f"{NC_GPONLY!r} in the _reference CSV for the GP-only rescale"
                          if const.any() else ""))
                del Xhn_gp
            else:
                S.add("", stem, label, _score_test(C["X_nbr"], Xhn_c, T, M, seed, "niche", k),
                      "niche", seed)
        elif block == "imp" and branch == "cell":
            stem = "imputation_benchmark_panel_cell"
            S.add("", stem, label, _score_test(C["X"], Xh_c, T, M, seed, "cell", k), "cell", seed)
            S.add("asstored", stem, label, stored, "cell", seed)
        elif block == "imp" and branch == "niche":
            stem = "imputation_benchmark_panel_niche"
            F = np.where(T[:, None], Xh_c, C["X"])               # parity: observed = truth
            fair = _score_test(C["X_nbr"], _neighbor_mean(C["A"], F), T, M, seed, "niche", k)
            own = _score_test(C["X_nbr"], _neighbor_mean(C["A"], Xh_c), T, M, seed, "niche", k)
            S.add("", stem, label, fair, "niche", seed)
            S.add("", "imputation_benchmark_panel_niche_ownfill", label, own, "niche", seed)
            S.add("asstored", stem, label, stored, "niche", seed)
            # own fill on the canonical graph == the runner's stored niche rows
            _check_frames(R, f"S1/{cid}/ownfill", own, stored, ["niche"], args.tol,
                          "canonical own-fill niche vs stored niche (same graph)")
    R.info(f"S1/{cid}", f"done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# SQUINT: replay of the stored predict pass (validate_table2.sh logic)
# ---------------------------------------------------------------------------

def _find_ckpts(rd: Path):
    cdir = rd / "checkpoints"
    if not cdir.is_dir():
        cdir = rd / "files" / "checkpoints"
    files = sorted(str(p) for p in cdir.glob("*.ckpt")) if cdir.is_dir() else []
    last = [f for f in files if "last" in os.path.basename(f).lower()]
    best, bv = None, np.inf
    for f in files:   # mirrors find_best_checkpoint(mode=min, val_loss)
        try:
            v = float(os.path.basename(f).split("val_loss=")[1].split(".ckpt")[0])
        except (IndexError, ValueError):
            continue
        if v < bv:
            bv, best = v, f
    return best, (last[0] if last else None), files


def _predict_bs(rd: Path):
    for fn in ("user_specified_config.yaml", "config.yaml"):
        p = rd / fn
        if p.is_file():
            m = re.search(r"predict_batch_size:\s*(\d+)", p.read_text())
            if m:
                return int(m.group(1)), str(p)
    return None, None


def squint_replay(A: ad.AnnData, X: np.ndarray, run_dir: Path, s2d, args,
                  R: Report, cid: str) -> dict:
    """Replay SQUINT's predict pass from the stored predicted_adata: the
    predict-time checkpoint, the PyG 1-hop subgraph of each predict batch (from
    uns['edge_index']) and the predict batch size that reproduces the stored
    X_hat. Gated on the deterministic part only, the decoders applied to the
    STORED embeddings (the predict-time codes): obsm['cell_emb'] -> stored
    X_hat (validate_table2.sh's exact-replay check) and obsm['neighborhood_emb']
    aggregated over the predict subgraph -> stored X_hat_nbr. The full forward
    re-quantises the codes on this device (near-tie codes can flip vs the GPU
    predict pass) and is reported as INFO only. Returns the per-cell niche
    output of the predict-time codes (seed mode; the decision-1 input) in the
    adata's row order."""
    import torch
    torch.set_num_threads(int(args.threads))
    import squint  # noqa: F401
    from squint.models import Squint_Dual
    from squint.utils.loss_utils import aggregate_1hop_neighbor_features
    for _n, _m in list(sys.modules.items()):   # old ckpts may pickle 'vqniche.*'
        if _n == "squint" or _n.startswith("squint."):
            sys.modules.setdefault("vqniche" + _n[len("squint"):], _m)
    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        R.info(cid, "cuda requested but unavailable; replaying on cpu")
        dev = "cpu"

    N, G = A.n_obs, A.n_vars
    lib = X.sum(1)
    sec_int = A.obs["adata_batch_id"].astype(int).to_numpy()
    uniq = np.array(sorted(np.unique(sec_int)))
    sec_dense = np.searchsorted(uniq, sec_int)        # = stage2_decode_pearson._cov_index
    Xh_st, Xhn_st, Xn_st = _get_mat(A, "X_hat"), _get_mat(A, "X_hat_nbr"), _get_mat(A, "X_nbr")
    if Xh_st is None or Xhn_st is None or Xn_st is None:
        raise RuntimeError("stored X_hat / X_hat_nbr / X_nbr missing")
    codes_c = np.asarray(A.obsm["cell_code_indices"]).astype(np.int64).reshape(N, -1)
    codes_n = np.asarray(A.obsm["neighborhood_code_indices"]).astype(np.int64).reshape(N, -1)
    cell_emb = np.asarray(A.obsm["cell_emb"], np.float32)
    if "neighborhood_emb" not in A.obsm:
        raise RuntimeError("obsm['neighborhood_emb'] missing: no predict-time niche codes")
    nb_emb = np.asarray(A.obsm["neighborhood_emb"], np.float32)   # z_q_niche (seed rows)
    if "edge_index" not in A.uns:
        raise RuntimeError("uns['edge_index'] missing: cannot rebuild the predict subgraphs")
    EI = np.asarray(A.uns["edge_index"]).astype(np.int64)
    EI = EI if EI.shape[0] == 2 else EI.T
    src, dst = EI[0], EI[1]

    # checkpoint: the one predict() recorded in uns, else the best val_loss one
    meta, meta_key = s2d._stage1_meta(A)
    stored = s2d._meta_str(meta.get("ckpt"))
    meta_rd = s2d._meta_str(meta.get("run_dir"))
    ck_uns = s2d._resolve_stored_ckpt(stored, str(run_dir), meta_rd)
    best, last, files = _find_ckpts(run_dir)
    ckpt = ck_uns or best or last
    if ckpt is None:
        raise RuntimeError(f"no checkpoint under {run_dir}")
    R.info(f"{cid}/ckpt", f"replay ckpt {os.path.basename(ckpt)} "
                          f"({'uns[' + repr(meta_key) + '][ckpt]' if ck_uns else 'best val_loss (uns ckpt not resolvable: ' + repr(stored) + ')'}); "
                          f"best val_loss {best and os.path.basename(best)}; last "
                          f"{last and os.path.basename(last)}; uns ckpt == best: "
                          f"{bool(ck_uns and best and s2d._same_ckpt(ck_uns, best))}")
    model = Squint_Dual.load_from_checkpoint(ckpt, map_location=dev)
    model.eval().to(dev)

    # PyG NeighborLoader, 1 hop, fan-out -1: seeds + all in-neighbours
    order = np.argsort(dst, kind="stable")
    src_s, dst_s = src[order], dst[order]
    indptr = np.searchsorted(dst_s, np.arange(N + 1))

    def subgraph(seeds):
        seeds = np.asarray(seeds, np.int64)
        st, en = indptr[seeds], indptr[seeds + 1]
        lens = en - st
        csum = np.cumsum(lens)
        idx = np.repeat(st - (csum - lens), lens) + np.arange(int(csum[-1]))
        nb, tg = src_s[idx], np.repeat(seeds, lens)
        nodes = np.concatenate([seeds, np.setdiff1d(np.unique(nb), seeds)])
        lmap = np.full(N, -1, np.int64)
        lmap[nodes] = np.arange(nodes.size)
        return nodes, np.stack([lmap[nb], lmap[tg]])

    def dec_cell(nodes):
        with torch.no_grad():
            z = torch.as_tensor(cell_emb[nodes], device=dev)
            if model.decoder_covariate_dim > 0:
                ids = torch.as_tensor(sec_dense[nodes], dtype=torch.long, device=dev)
                Em = (model.batch_embedding_cell
                      if getattr(model, "decoupled_decoder_covariate", False)
                      else model.batch_embedding)
                z = torch.cat([z, Em(ids)], dim=-1)
            o = model.attribute_decoder_cell(
                x=z, read_depth=torch.as_tensor(lib[nodes], device=dev), conditions=None)
        return o.float().cpu().numpy()

    def dec_niche(zn, ids, rdep):
        """attribute_decoder_niche as the forward calls it: z_q_niche + the
        niche covariate, read depth = the nodes' own library."""
        z = zn
        if model.decoder_covariate_dim > 0:
            Em = (model.batch_embedding_niche
                  if getattr(model, "decoupled_decoder_covariate", False)
                  else model.batch_embedding)
            z = torch.cat([z, Em(ids)], dim=-1)
        return model.attribute_decoder_niche(x=z, read_depth=rdep, conditions=None)

    def full_batches(batches):
        """Full forward per predict batch, plus the niche decoder on the
        predict-time codes: the seeds' stored z_q_niche, the non-seed
        in-neighbours' replayed (truncated-subgraph) z_q_niche, as the predict
        pass fed its layer norm, which normalises over the whole batch tensor.
        closed = seeds whose in-neighbours are all seeds of their batch."""
        Xh, Xhn, Xn, SM, SMp, Xhnp = (np.zeros((N, G), np.float32) for _ in range(6))
        ic = np.zeros_like(codes_c)
        inn = np.zeros_like(codes_n)
        closed = np.zeros(N, dtype=bool)
        for seeds in batches:
            nodes, ei = subgraph(seeds)
            Bn = len(seeds)
            x = torch.as_tensor(X[nodes], device=dev)
            eit = torch.as_tensor(ei, dtype=torch.long, device=dev)
            ids = torch.as_tensor(sec_dense[nodes], dtype=torch.long, device=dev)
            with torch.no_grad():
                outs = model(batch_x=x, batch_edge_index=eit, read_depth=x.sum(dim=-1),
                             adata_batch_ids=ids)
                zqn, idc, idn, xc, xn = outs[3], outs[4], outs[5], outs[6], outs[7]
                agg = aggregate_1hop_neighbor_features(
                    X=torch.cat([x, xn], dim=-1), edge_index=eit, return_mean=True,
                    batch_size=Bn)
                if zqn.shape[1] != nb_emb.shape[1]:
                    raise RuntimeError(f"z_q_niche dim {zqn.shape[1]} != obsm"
                                       f"['neighborhood_emb'] dim {nb_emb.shape[1]}")
                zp = zqn.clone()
                zp[:Bn] = torch.as_tensor(nb_emb[seeds], device=dev, dtype=zp.dtype)
                xp = dec_niche(zp, ids, x.sum(dim=-1))
                aggp = aggregate_1hop_neighbor_features(
                    X=xp, edge_index=eit, return_mean=True, batch_size=Bn)
            Xh[seeds] = xc[:Bn].float().cpu().numpy()
            SM[seeds] = xn[:Bn].float().cpu().numpy()    # niche output, full 1-hop (seed mode)
            SMp[seeds] = xp[:Bn].float().cpu().numpy()   # same, predict-time (stored) codes
            Xn[seeds] = agg[:, :G].float().cpu().numpy()
            Xhn[seeds] = agg[:, G:].float().cpu().numpy()
            Xhnp[seeds] = aggp.float().cpu().numpy()
            ic[seeds] = idc[:Bn].reshape(Bn, -1).cpu().numpy()
            inn[seeds] = idn[:Bn].reshape(Bn, -1).cpu().numpy()
            opn = np.zeros(Bn, dtype=bool)
            opn[ei[1][ei[0] >= Bn]] = True
            closed[seeds] = ~opn
        return Xh, Xhn, Xn, SM, SMp, Xhnp, ic, inn, closed

    # predict batch size: the config's, verified by a cell-decoder replay
    # (validate_table2.sh: stop the search below 1e-4, accept below its tol)
    B_cfg, cfg_path = _predict_bs(run_dir)
    best_B, best_err, best_out = None, np.inf, None
    for Bsz in dict.fromkeys([b for b in (B_cfg, 65536, 32768, 16384, 8192, 4096) if b]):
        out = np.zeros((N, G), np.float32)
        for st in range(0, N, Bsz):
            seeds = np.arange(st, min(N, st + Bsz))
            nodes, _ = subgraph(seeds)
            out[seeds] = dec_cell(nodes)[: len(seeds)]
        e_try = _rowrel(out, Xh_st)
        err = float(e_try.max())
        R.info(f"{cid}/batch", f"cell-decoder replay B={Bsz}: max row rel-L2 vs stored X_hat {err:.2e}")
        if err < best_err:
            best_err, best_B, best_out = err, Bsz, e_try
        if err < min(1e-4, args.replay_tol):
            break
    R.info(f"{cid}/batch", f"predict_batch_size in config {B_cfg} ({cfg_path}); using B={best_B}")
    tol = args.replay_tol
    # S3 gate 1 (deterministic): cell decoder on the stored cell_emb
    ok_h = R.check(f"{cid}/X_hat", best_err <= tol,
                   f"cell decoder on the stored obsm['cell_emb'] (predict batches B={best_B}) "
                   f"vs stored X_hat: max row rel-L2 {best_err:.2e}, "
                   f"{int((best_out > tol).sum())}/{N} row(s) above tol (tol {tol:g}, "
                   f"validate_table2.sh's exact-replay check)")
    batches = [np.arange(st, min(N, st + best_B)) for st in range(0, N, best_B)]
    Xh1, Xhn1, Xn1, SM1, SMp, Xhnp, ic1, in1, closed = full_batches(batches)
    # S3 gate 2 (deterministic): niche decoder on the stored neighborhood_emb,
    # aggregated over the predict subgraph, on the rows whose aggregate only
    # holds stored-code outputs (no non-seed in-neighbour)
    e_np = _rowrel(Xhnp, Xhn_st)
    n_cl = int(closed.sum())
    e_cl = float(e_np[closed].max()) if n_cl else float("inf")
    e_op = e_np[~closed]
    ok_hn = R.check(f"{cid}/X_hat_nbr", n_cl > 0 and e_cl <= tol,
                    f"niche decoder on the stored obsm['neighborhood_emb'], aggregated over "
                    f"the predict subgraph, vs stored X_hat_nbr on the {n_cl}/{N} row(s) whose "
                    f"in-neighbours are all seeds of their batch: max row rel-L2 {e_cl:.2e} "
                    f"(tol {tol:g}); other rows (a non-seed in-neighbour enters with its "
                    f"replayed code): {int((e_op > tol).sum())}/{e_op.size} above tol, max "
                    f"{float(e_op.max()) if e_op.size else 0.0:.2e}")
    # full forward: NOT gated (codes re-quantised on this device)
    eq_c = (ic1 == codes_c).all(1)
    eq_n = (in1 == codes_n).all(1)
    e_h = _rowrel(Xh1, Xh_st)
    e_hn = _rowrel(Xhn1, Xhn_st)
    R.info(f"{cid}/full-forward",
           f"not gated (re-quantises the codes on {dev}; near-tie codes can flip vs the "
           f"predict pass; the canonical rows use the stored codes): codes equal stored: "
           f"cell {float(eq_c.mean()):.6f} ({int((~eq_c).sum())} row(s) differ), niche "
           f"{float(eq_n.mean()):.6f} ({int((~eq_n).sum())} row(s) differ); X_hat "
           f"{int((e_h > tol).sum())}/{N} row(s) above tol (max row rel-L2 "
           f"{float(e_h.max()):.2e}); X_hat_nbr {int((e_hn > tol).sum())}/{N} row(s) above "
           f"tol (max {float(e_hn.max()):.2e}); X_nbr max |diff| "
           f"{float(np.abs(Xn1 - Xn_st).max()):.3g}")
    e_sm = _rowrel(SM1, SMp)
    R.info(f"{cid}/niche-out",
           f"decision-1 per-cell niche outputs (stored codes) vs full-forward ones: "
           f"{int(eq_n.sum())} row(s) with equal niche codes, max row rel-L2 "
           f"{float(e_sm[eq_n].max()) if eq_n.any() else 0.0:.2e}; {int((~eq_n).sum())} "
           f"row(s) with a flipped niche code, max row rel-L2 "
           f"{float(e_sm[~eq_n].max()) if (~eq_n).any() else 0.0:.2e}")
    del model, Xh1, Xhn1, Xn1, SM1, Xhnp
    gc.collect()
    return dict(SM=SMp, ckpt=ckpt, B=best_B, verified=bool(ok_h and ok_hn))


# ---------------------------------------------------------------------------
# SQUINT per seed: S2, S3, canonical reconstruction, imputation arms (S4)
# ---------------------------------------------------------------------------

def add_asstored_arms(seed_run: str, args, R: Report, S: Store) -> None:
    """As-stored SQUINT (MC) / (No MC) imputation rows = the PAPER arms'
    per-seed CSVs (current Table 2). Needs no predicted_adata or silver
    mapping, so main runs it guarded and before process_squint_seed."""
    seed = _seed_of(seed_run)
    for label, d in zip(SQ_ARMS, (args.asstored_mc_arm_dir, args.asstored_nomc_arm_dir)):
        p = Path(d) / seed_run / "metrics" / CSV_NAME
        if not p.is_file():
            R.fail(f"asstored/{label}/seed{seed}", f"published arm per-seed CSV missing: {p}")
            continue
        PS = pd.read_csv(p)
        if "seed" in PS.columns and not (PS["seed"] == seed).all():
            R.info(f"asstored/{label}/seed{seed}", f"CSV seed column "
                   f"{sorted(PS['seed'].unique().tolist())} != folder seed {seed}; using "
                   f"the folder seed")
        S.add("asstored", "imputation_benchmark_panel_cell", label, PS, "cell", seed)
        S.add("asstored", "imputation_benchmark_panel_niche", label, PS, "niche", seed)


def _squint_asstored_recon(A, X, Xh_st, Xhn_st, Xn_st, test_sq, seed, seed_run, ms,
                           cid, args, R: Report, S: Store) -> None:
    """S0 / S2: re-score the stored predicted_adata as stored (as-stored
    SQUINT reconstruction rows + the native-graph reference)."""
    mk_sq, _ = _marker_idx_bpd(X, A.obs)
    fast = pd.concat([_score_test(X, Xh_st, test_sq, mk_sq, seed, "cell", args.n_hvg),
                      _score_test(Xn_st, Xhn_st, test_sq, mk_sq, seed, "niche", args.n_hvg)],
                     ignore_index=True)
    literal = (not args.fast_selfchecks) or seed_run == args.squint_seeds[0]
    if literal:
        lit = build_pearson_dataframe(A, seed=seed, log1p=True, n_hvg=args.n_hvg)
        lit = lit[lit["split"] == "test"]
        _check_frames(R, f"S0/{cid}", fast, lit, ["cell", "niche"], args.tol,
                      "test-only scorer vs build_pearson_dataframe")
        stored = lit
    else:
        stored = fast
    if ms is None:
        R.fail(f"S2/{cid}", f"seed {seed} not in the multiseed CSV")
    else:
        _check_frames(R, f"S2/{cid}", stored, ms, ["cell", "niche"], args.tol,
                      "stored predicted_adata re-scored vs multiseed CSV")
    S.add("asstored", "reconstruction_benchmark_panel_cell", "SQUINT", stored, "cell", seed)
    S.add("asstored", "reconstruction_benchmark_panel_niche", "SQUINT", stored, "niche", seed)
    S.add("", "reconstruction_benchmark_panel_niche_reference", SQ_NATIVE, stored, "niche", seed)


def process_squint_seed(seed_run: str, C: dict, args, R: Report, S: Store, s2d, MS) -> None:
    seed = _seed_of(seed_run)
    rd = args.artifacts_root / args.dataset_tag / args.squint_variant / seed_run
    p = rd / "predicted_adata.h5ad"
    R.section(f"[SQUINT seed {seed}] {p}")
    t0 = time.time()
    cid = f"squint/seed{seed}"
    A = ad.read_h5ad(p)
    X = _to_dense_2d(A.X)
    Xh_st, Xhn_st, Xn_st = _get_mat(A, "X_hat"), _get_mat(A, "X_hat_nbr"), _get_mat(A, "X_nbr")
    for k, M in (("X_hat", Xh_st), ("X_hat_nbr", Xhn_st), ("X_nbr", Xn_st)):
        if M is None:
            R.fail(f"S2/{cid}", f"stored {k} missing (layers and uns)")
        elif k not in A.layers:
            A.layers[k] = M                           # rescore_imputation's uns->layers
    test_sq = _test_mask(A.obs)
    ms = MS[MS["seed"] == seed] if MS is not None else None
    if ms is not None and ms.empty:
        ms = None
    # ---- S0 / S2: re-score as stored (guarded: S3 / S4 below never depend on it)
    R.guarded(f"S2/{cid}", _squint_asstored_recon, A, X, Xh_st, Xhn_st, Xn_st, test_sq,
              seed, seed_run, ms, cid, args, R, S)
    # ---- mapping onto the silver (a failure skips only the canonical rows) --
    mp = R.guarded(f"S2/{cid}/map", lambda: _map_and_check(
        R, f"S2/{cid}", C, A.obs_names, _norm_sec(A.obs["adata_batch_id"]), A.var_names,
        A.obs["data_split"], X, False, 0.0, coords=_xy(A)))
    r, c = mp if mp is not None else (None, None)
    T, Mk, k = C["test"], C["marker_idx"], args.n_hvg
    inv_r = Xh_c = None
    if r is None:
        R.fail(f"canon/{cid}", "no row mapping: SQUINT canonical rows skipped for this seed "
                               "(S3 / S4 self-checks still run)")
    else:
        inv_r = np.empty(r.size, dtype=np.int64)
        inv_r[r] = np.arange(r.size)
        Xh_c = _align(Xh_st, r, c)
        cell = _score_test(C["X"], Xh_c, T, Mk, seed, "cell", k)
        S.add("", "reconstruction_benchmark_panel_cell", "SQUINT", cell, "cell", seed)
        if ms is not None:
            _check_frames(R, f"S2/{cid}/canon-order", cell, ms, ["cell"], args.tol_order,
                          "canonical-order re-score of the stored X_hat vs multiseed CSV",
                          rel=True)
    # ---- S3 replay -> per-cell niche outputs of the STORED (predict-time)
    # codes, averaged on the canonical graph (decision 1) ----------------------
    rep = R.guarded(f"S3/{cid}", squint_replay, A, X, rd, s2d, args, R, f"S3/{cid}")
    replay_ckpt = None
    if rep is not None:
        if r is not None:
            SM_c = _align(rep["SM"], r, c)
            niche = _score_test(C["X_nbr"], _neighbor_mean(C["A"], SM_c), T, Mk, seed,
                                "niche", k)
            S.add("", "reconstruction_benchmark_panel_niche", "SQUINT", niche, "niche", seed)
            del SM_c
        if rep["verified"]:
            replay_ckpt = rep["ckpt"]     # reproduces the stored X_hat: the predict-time ckpt
        del rep
    else:
        R.fail(f"canon/{cid}", "replay failed: SQUINT canonical neighbourhood rows missing")
    gc.collect()
    # ---- S4 imputation arms ------------------------------------------------
    ctx = dict(A=A, X=X, Xh_st=Xh_st, test=test_sq, path=p, rd=rd, r=r, c=c, inv_r=inv_r,
               Xh_c=Xh_c, graphs={}, marker=None, replay_ckpt=replay_ckpt, ckpt_cmp={})
    for label, arm_dir, paper_dir in zip(
            SQ_ARMS, (args.mc_arm_dir, args.nomc_arm_dir),
            (args.asstored_mc_arm_dir, args.asstored_nomc_arm_dir)):
        R.guarded(f"S4/{label}/seed{seed}", _process_arm, label, Path(arm_dir),
                  Path(paper_dir), seed_run, seed, ctx, C, args, R, S, s2d)
        gc.collect()
    R.info(cid, f"done in {time.time() - t0:.0f}s")


def _npz_scalar(Z, k, default=None):
    if k not in Z.files:
        return default
    v = Z[k]
    return v.item() if getattr(v, "ndim", 1) == 0 else default


def _as_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _same_weights(ctx, a, b):
    """(True / False / None if unreadable, max |param diff|) for two ckpt
    paths; realpath-equal counts as identical. Cached per seed in ctx."""
    pa, pb = os.path.realpath(str(a)), os.path.realpath(str(b))
    if pa == pb:
        return True, 0.0
    if (pa, pb) not in ctx["ckpt_cmp"]:
        ctx["ckpt_cmp"][(pa, pb)] = _ckpt_param_diff(pa, pb)
    mx = ctx["ckpt_cmp"][(pa, pb)]
    return (None if mx is None else mx == 0.0), mx


def _paper_ckpt(paper_csv: pd.DataFrame, rd: Path, s2d):
    """(policy, ckpt path or None) the paper arm decoded with: its CSV
    ckpt_policy column, else 'last' (the decode default before that column
    existed), resolved as stage2_decode_pearson resolves it."""
    pol = "last"
    if "ckpt_policy" in paper_csv.columns:
        vals = paper_csv["ckpt_policy"].dropna().astype(str).unique().tolist()
        pol = vals[0] if len(vals) == 1 else f"ambiguous {vals}"
    try:
        if pol == "last":
            return pol, s2d._find_ckpt(str(rd), None)
        if pol == "best":
            return pol, s2d._best_ckpt(str(rd))
    except Exception:  # noqa: BLE001
        pass
    return pol, None


def _paper_depth_note(paper_p: Path, hard: bool, s2d) -> str:
    """The decode's --compare-csv history note for a failed vs-paper link: the
    paper arm CSV mtime vs the read-depth fix commits
    (stage2_decode_pearson._RD_COMMITS_UTC). Before 6b1389a every held-out
    decode used the TRUE library size; before fdca41a the hard (No MC) cell
    decode still did. The paper CSV carries no read-depth column, so the mtime
    is the only (indicative) evidence."""
    commits = getattr(s2d, "_RD_COMMITS_UTC", None)
    if not commits:
        return ""
    try:
        t = datetime.fromtimestamp(os.path.getmtime(paper_p), tz=timezone.utc)
    except OSError:
        return ""
    before = [sha for sha, tt in commits if t < datetime(*tt, tzinfo=timezone.utc)]
    note = (f" Paper CSV mtime {t.isoformat()}: before read-depth fix commit(s) "
            f"{before or 'none'} of {[sha for sha, _ in commits]} (indicative only, the "
            f"farm checkout may have differed).")
    if "6b1389a" in before or ("fdca41a" in before and hard):
        note += (" LIKELY CAUSE: the paper arm decoded with each held-out cell's TRUE "
                 "library size ("
                 + ("every decode before 6b1389a" if "6b1389a" in before else
                    "the hard No MC cell decode before fdca41a")
                 + "), which a leak-free re-decode does not reproduce; the as-stored "
                   "rows of this arm (the summary 'delta' reference) are then the "
                   "leaky true-depth ones.")
    return note


def _process_arm(label, arm_dir: Path, paper_dir: Path, seed_run, seed, ctx, C, args, R, S,
                 s2d):
    cid = f"S4/{label}/seed{seed}"
    npz_p = arm_dir / seed_run / "metrics" / NPZ_NAME
    csv_p = arm_dir / seed_run / "metrics" / CSV_NAME
    paper_p = paper_dir / seed_run / "metrics" / CSV_NAME
    CS = pd.read_csv(csv_p) if csv_p.is_file() else None
    if CS is None:
        R.fail(cid, f"re-decoded arm per-seed CSV missing: {csv_p}")
    else:
        if "seed" in CS.columns and not (CS["seed"] == seed).all():
            R.info(cid, f"CSV seed column {sorted(CS['seed'].unique().tolist())} != folder "
                        f"seed {seed}; using the folder seed")
        S.add("redecoded", "imputation_benchmark_panel_cell", label, CS, "cell", seed)
        S.add("redecoded", "imputation_benchmark_panel_niche", label, CS, "niche", seed)
    if not npz_p.is_file():
        R.fail(cid, f"{npz_p} missing: re-run this arm with stage2_decode_pearson "
                    f"--save-pred (new OUT_ROOT / ARM_SUFFIX) and pass its folder; "
                    f"{label} canonical imputation rows skipped for this seed")
        return
    A, X, Xh_st, test_sq = ctx["A"], ctx["X"], ctx["Xh_st"], ctx["test"]
    Z = np.load(npz_p)
    for kk in ("global_idx", "x_hat"):
        if kk not in Z.files:
            R.fail(cid, f"{npz_p} has no {kk!r} (keys {Z.files})")
            return
    gidx = Z["global_idx"].astype(np.int64)
    xh = Z["x_hat"].astype(np.float32)
    meta = {kk: _npz_scalar(Z, kk) for kk in Z.files if getattr(Z[kk], "ndim", 1) == 0}
    R.info(cid, f"npz {npz_p}: {gidx.size} rows; provenance {meta}")
    # ---- provenance --------------------------------------------------------
    # The canonical rows below use the npz x_hat only, so every property that
    # decides whether x_hat is the fair, published prediction is a FAIL.
    pa = meta.get("predicted_adata")
    R.check(f"{cid}/predicted_adata",
            bool(pa) and os.path.realpath(str(pa)) == os.path.realpath(str(ctx["path"])),
            f"npz decoded from {pa!r} (this seed's predicted_adata: {ctx['path']})")
    codes, codes_exp = meta.get("stage2_codes"), args.stage2_codes_dir / seed_run / CODES_NAME
    R.check(f"{cid}/stage2_codes",
            bool(codes) and os.path.realpath(str(codes)) == os.path.realpath(str(codes_exp)),
            f"npz decoded codes {codes!r} (expect the published stage-2 codes {codes_exp})")
    exp_mc = label == SQ_ARMS[0]
    K_exp, eff_exp = (int(args.mc_decode_samples), "mc") if exp_mc else (1, "hard")
    K, eff = meta.get("decode_samples"), meta.get("decode_effective")
    R.check(f"{cid}/decode_samples",
            _as_float(K) == K_exp and str(eff) == eff_exp,
            f"decode_samples={K}, decode_effective={eff!r} (expect {K_exp}, {eff_exp!r} "
            f"for {label})")
    sm = meta.get("smooth_neighs")
    R.check(f"{cid}/smooth_neighs", _as_float(sm) == 0,
            f"smooth_neighs={sm} (expect 0: x_hat not spatially smoothed)")
    rdm, rdk = meta.get("read_depth_mode"), meta.get("read_depth_neighs")
    R.check(f"{cid}/read_depth", rdm == "neighbor"
            and _as_float(rdk) == PAPER_READ_DEPTH_NEIGHS,
            f"read_depth_mode={rdm!r}, read_depth_neighs={rdk} (leak-free paper setting: "
            f"'neighbor', {PAPER_READ_DEPTH_NEIGHS}; both re-decode tools default to the "
            f"leaky 'true')")
    # the depth actually used: npz read_depth == the leak-free depth recomputed
    # here, x_hat rows sum to it (decoder = softmax * depth) and not to the
    # held-out cells' own library sizes
    if "read_depth" not in Z.files:
        R.fail(f"{cid}/read_depth_values", "npz has no 'read_depth' array")
    else:
        rd_z = Z["read_depth"].astype(np.float64)
        L_all = np.asarray(X.sum(axis=1), dtype=np.float32).ravel()
        L_h = L_all[gidx].astype(np.float64)
        rel_sum = float(np.max(np.abs(xh.sum(1, dtype=np.float64) - rd_z)
                               / np.maximum(rd_z, 1e-12)))
        rel_true = float(np.median(np.abs(rd_z - L_h) / np.maximum(L_h, 1e-12)))
        rd_re = s2d._neighbor_read_depth(
            np.asarray(A.obsm["spatial"]), np.asarray(A.obs["adata_batch_id"].values), gidx,
            L_all, PAPER_READ_DEPTH_NEIGHS).astype(np.float64)
        rel_re = float(np.max(np.abs(rd_re - rd_z) / np.maximum(np.abs(rd_re), 1e-12)))
        R.check(f"{cid}/read_depth_values",
                rel_re <= 1e-5 and rel_sum <= 1e-3 and rel_true > 1e-3,
                f"npz read_depth vs leak-free depth recomputed here ({PAPER_READ_DEPTH_NEIGHS} "
                f"nearest observed cells per section): max rel diff {rel_re:.2e} (tol 1e-5); "
                f"x_hat row sum vs read_depth: max rel diff {rel_sum:.2e} (tol 1e-3); "
                f"read_depth vs own library size: median rel diff {rel_true:.3f} (must be "
                f"> 1e-3, ~0 = leaky true depth)")
    scm = _as_float(meta.get("selfcheck_cellwise_pearson_median"))
    R.check(f"{cid}/selfcheck", scm >= args.selfcheck_min,
            f"decode true-code self-check cell-wise Pearson median {scm:.4f} "
            f"(min {args.selfcheck_min:g})")
    if "obs_names" in Z.files:
        ok = np.array_equal(np.asarray(Z["obs_names"]).astype(str),
                            np.asarray(A.obs_names[gidx]).astype(str))
        R.check(f"{cid}/obs_names", ok, "npz obs_names == predicted_adata.obs_names[global_idx]")
    if "var_names" in Z.files:
        ok = np.array_equal(np.asarray(Z["var_names"]).astype(str),
                            np.asarray(A.var_names).astype(str))
        R.check(f"{cid}/var_names", ok, "npz var_names == predicted_adata.var_names")
    want = np.where(test_sq)[0]
    cov_ok = R.check(f"{cid}/coverage", gidx.size == np.unique(gidx).size and
                     np.array_equal(np.unique(gidx), want),
                     f"global_idx == the {want.size} held-out rows of the predicted_adata")
    if "x_true" in Z.files and cov_ok:
        xt = Z["x_true"]
        R.check(f"{cid}/x_true", xt.shape == (gidx.size, X.shape[1])
                and np.array_equal(xt.astype(np.float32), X[gidx]),
                "npz x_true == predicted_adata X[global_idx] (x_hat rows are these "
                "held-out cells, in this order)")
    if meta.get("niche_from_cell") is not None:
        R.check(f"{cid}/niche_from_cell", bool(meta.get("niche_from_cell")),
                f"niche_from_cell={meta.get('niche_from_cell')}", warn_only=True)
    if meta.get("nbr_neighs") is not None:
        R.check(f"{cid}/nbr_neighs", int(meta.get("nbr_neighs")) == args.nbr_k,
                f"nbr_neighs={meta.get('nbr_neighs')} (canonical k={args.nbr_k})",
                warn_only=True)
    # decode checkpoint vs the predict-time one that wrote the codes and X_hat:
    # the S3-verified replay ckpt, else the uns path recorded in the npz.
    # A FAIL unless --allow-ckpt-mismatch (then a WARN).
    allow = bool(args.allow_ckpt_mismatch)
    hint = ("; accepted as a WARN (--allow-ckpt-mismatch)" if allow else
            "; re-decode with --stage1-ckpt-policy adata, or pass --allow-ckpt-mismatch "
            "to accept the paper arms' decode checkpoint")
    ck_dec = meta.get("stage1_ckpt_realpath") or meta.get("stage1_ckpt")
    ck_ref, ref_src = ctx["replay_ckpt"], "S3-verified replay ckpt"
    if not ck_ref and meta.get("stage1_ckpt_predict_time"):
        ck_ref = s2d._resolve_stored_ckpt(str(meta.get("stage1_ckpt_predict_time")),
                                          str(ctx["rd"]))
        ref_src = "npz stage1_ckpt_predict_time"
    if ck_dec and ck_ref:
        same_w, mx = _same_weights(ctx, ck_dec, ck_ref)
        R.check(f"{cid}/ckpt", same_w is True,
                f"decode ckpt {os.path.basename(str(ck_dec))} (policy "
                f"{meta.get('ckpt_policy')!r}) vs {ref_src} {os.path.basename(str(ck_ref))}: "
                f"max |param diff| {'unreadable' if mx is None else format(mx, '.3e')}"
                + ("" if same_w is True else hint), warn_only=allow)
    else:
        R.check(f"{cid}/ckpt", False, f"decode ckpt {ck_dec!r}; predict-time reference "
                                      f"unknown (S3 not verified and no uns ckpt in the "
                                      f"npz)" + hint, warn_only=allow)
    # ---- S4: re-score with the decode's own convention ----------------------
    if "stage2_npz_row" in Z.files:
        o = Z["stage2_npz_row"].astype(np.int64)
        g_sc = np.empty_like(gidx)
        g_sc[o] = gidx
        x_sc = np.empty_like(xh)
        x_sc[o] = xh
    else:
        R.info(cid, "no stage2_npz_row: scoring in ascending row order (float sums may "
                    "differ in the last digits)")
        g_sc, x_sc = gidx, xh
    if ctx["marker"] is None:
        lab = next((kk for kk in ("cell_type", "cell_types", "new_annotation", "annotation")
                    if kk in A.obs.columns), None)
        mk = None
        if lab is not None:
            labs = A.obs[lab].to_numpy()
            tr = A.obs["data_split"].to_numpy() != "test"
            if tr.sum() == 0:
                tr = np.ones(A.n_obs, dtype=bool)
            mk = s2d._marker_idx(np.log1p(np.clip(X[tr], 0, None)), labs[tr])
        ctx["marker"] = (mk,)
    mk_dec = ctx["marker"][0]
    dec_cell = pd.DataFrame(s2d._pearson_rows(X[g_sc], x_sc, branch="cell", split="test",
                                              seed=seed, marker_idx=mk_dec))
    if CS is not None:
        _check_frames(R, f"{cid}/decode-cell", dec_cell, CS, ["cell"], args.tol,
                      "npz x_hat re-scored (decode convention) vs arm CSV")
    fill = str(meta.get("nbr_observed_fill") or "stage1")
    obs_x = str(meta.get("observed_xhat") or "stored")
    nfc = meta.get("niche_from_cell")
    if nfc is not None and not bool(nfc):
        R.skip(f"{cid}/decode-niche", "arm used the native niche decoder; the per-cell niche "
                                      "decode is not in the npz")
    elif fill == "stage1" and obs_x.startswith("redecode"):
        R.skip(f"{cid}/decode-niche", "observed rows were redecoded by the arm; not in the npz")
    else:
        kk = int(meta.get("nbr_neighs") or args.nbr_k)
        if kk not in ctx["graphs"]:
            coords = np.asarray(A.obsm["spatial"])
            batch = np.asarray(A.obs["adata_batch_id"].values)
            ctx["graphs"][kk] = s2d._spatial_knn_selfloop(coords, batch, kk)
            if kk == args.nbr_k and ctx["r"] is not None:
                Ad = ctx["graphs"][kk]
                r = ctx["r"]
                Ad_s = Ad if np.array_equal(r, np.arange(r.size)) else Ad[r][:, r]
                rows = _graph_rows_differing(Ad_s, C["A"])
                Xn_dec = s2d._nbr_mean(Ad, X)
                dX = np.abs(_align(Xn_dec, r, ctx["c"]) - C["X_nbr"])[C["test"]].max(axis=1)
                R.info(f"S4/seed{seed}/graph",
                       f"decode graph (sklearn kNN+self on predicted_adata coords) vs canonical "
                       f"(squidpy on silver coords): {rows.size} row(s) differ "
                       f"({int(C['test'][rows].sum()) if rows.size else 0} test); decode X_nbr "
                       f"target differs on {int((dX > 1e-4).sum())} test row(s) (max |diff| "
                       f"{float(dX.max()):.3g})")
        Ad = ctx["graphs"][kk]
        held = np.zeros(A.n_obs, dtype=bool)
        held[gidx] = True
        A_agg = s2d._restrict_cols(Ad, held) if fill == "heldout_only" else Ad
        Xn_d = s2d._nbr_mean(A_agg, X)
        full = (X if fill == "truth" else Xh_st).copy()
        full[g_sc] = x_sc
        Xhn_d = s2d._nbr_mean(A_agg, full)
        dec_niche = pd.DataFrame(s2d._pearson_rows(Xn_d[g_sc], Xhn_d[g_sc], branch="niche",
                                                   split="test", seed=seed, marker_idx=mk_dec))
        if CS is not None:
            _check_frames(R, f"{cid}/decode-niche", dec_niche, CS, ["niche"], args.tol,
                          f"neighbourhood re-scored (decode convention: fill={fill!r}, "
                          f"observed_xhat={obs_x!r}, k={kk}) vs arm CSV")
        del Xn_d, Xhn_d, full
    # ---- link to the published arm: re-decoded CSV cell rows vs paper CSV -------
    PS = pd.read_csv(paper_p) if paper_p.is_file() else None
    if CS is None or PS is None:
        R.fail(f"{cid}/vs-paper", f"cannot link the npz to the published arm: "
                                  f"{csv_p if CS is None else paper_p} missing")
    elif os.path.realpath(str(arm_dir)) == os.path.realpath(str(paper_dir)):
        R.info(f"{cid}/vs-paper", f"re-decoded arm == paper arm folder {paper_dir}")
    else:
        # per-arm tolerance, as the decode's own --compare-csv gate (MC
        # re-decodes drift at its 1e-3 level: multinomial sampling, GPU order)
        tol_rd = args.tol_redecode_mc if exp_mc else args.tol_redecode
        res = _compare_frames(CS, PS, "cell", rel=True)
        dmax, n, n_only = res if res is not None else (float("inf"), 0, 0)
        what = (f"re-decoded arm CSV cell rows vs paper arm CSV {paper_p}: max rel |diff| "
                f"{dmax:.3e} over {n} rows, {n_only} unmatched (tol rel {tol_rd:g}, "
                f"{'MC' if exp_mc else 'No MC'} arm)")
        if meta.get("compare_pass") is not None:
            what += (f"; the decode's own --compare-csv gate: "
                     f"{'PASS' if bool(meta.get('compare_pass')) else 'FAIL'} vs "
                     f"{meta.get('compare_csv')} (tol {meta.get('compare_tol')})")
        if n > 0 and n_only == 0 and dmax <= tol_rd:
            R.check(f"{cid}/vs-paper", True, what + ": the npz x_hat is the published "
                                                    "arm's prediction")
        else:
            what += "." + _paper_depth_note(paper_p, not exp_mc, s2d)
            pol_p, ck_p = _paper_ckpt(PS, ctx["rd"], s2d)
            pol_d = str(meta.get("ckpt_policy"))
            same_w, mx = (_same_weights(ctx, ck_dec, ck_p) if ck_dec and ck_p
                          else (None, None))
            drift = same_w is False or (same_w is None and pol_p != pol_d)
            R.check(f"{cid}/vs-paper", False,
                    what + f" Decode ckpt: paper policy {pol_p!r} -> "
                    f"{ck_p and os.path.basename(str(ck_p))}, re-decode policy {pol_d!r} -> "
                    f"{ck_dec and os.path.basename(str(ck_dec))}, max |param diff| "
                    f"{'unknown' if mx is None else format(mx, '.3e')}. "
                    + ("RE-DECODE DRIFT from the different checkpoint: the canonical "
                       f"{label} rows of this seed are NOT the published arm's prediction "
                       "and the summary canonical - asstored delta includes this drift "
                       "(see its 'drift' line)" if drift else
                       "Same checkpoint but different numbers: the re-decode does not "
                       "reproduce the published arm (other flags, codes or read depth "
                       "differ; see the paper CSV mtime note above)"),
                    warn_only=drift)
    # ---- canonical imputation rows ----------------------------------------------
    if ctx["r"] is None:
        R.skip(f"{cid}/canon-rows", "no silver row mapping for this seed (FAIL above): "
                                    "canonical rows skipped")
        return
    s_rows = ctx["inv_r"][gidx]
    T = C["test"]
    ok = s_rows.size == int(T.sum()) and np.array_equal(np.sort(s_rows), np.where(T)[0])
    if not R.check(f"{cid}/canon-rows", ok, "held-out rows map onto exactly the canonical "
                                            "test rows"):
        return
    xh_c = xh if np.array_equal(ctx["c"], np.arange(ctx["c"].size)) else xh[:, ctx["c"]]
    Mk, k = C["marker_idx"], args.n_hvg
    P = C["X"].copy()
    P[s_rows] = xh_c                                   # parity: observed = truth
    cell = _score_test(C["X"], P, T, Mk, seed, "cell", k)
    fair = _score_test(C["X_nbr"], _neighbor_mean(C["A"], P), T, Mk, seed, "niche", k)
    O = ctx["Xh_c"].copy()
    O[s_rows] = xh_c                                   # own fill = stage-1 X_hat
    own = _score_test(C["X_nbr"], _neighbor_mean(C["A"], O), T, Mk, seed, "niche", k)
    S.add("", "imputation_benchmark_panel_cell", label, cell, "cell", seed)
    S.add("", "imputation_benchmark_panel_niche", label, fair, "niche", seed)
    S.add("", "imputation_benchmark_panel_niche_ownfill", label, own, "niche", seed)
    if CS is not None:
        _check_frames(R, f"{cid}/canon-order", cell, CS, ["cell"], args.tol_order,
                      "canonical-order cell re-score vs arm CSV (row order + harness only)",
                      rel=True)


# ---------------------------------------------------------------------------
# Panels, summary, existing-table comparison
# ---------------------------------------------------------------------------

def build_panel_long(resolved: "OrderedDict[str, Optional[Path]]", branch: str,
                     split: str = "test") -> pd.DataFrame:
    """render_metric_grid's long-row loop (plot_imputation_benchmark), without
    the figure: panels in _GRID_ROWS order, methods in `resolved` order."""
    long_rows: List[dict] = []
    panels = [p for row in _GRID_ROWS for p in row]
    for suffix, axis, transform, gene_subset, value_col, _title, _hib in panels:
        for label, csv in resolved.items():
            if csv is None:
                continue
            df = _load_csv_filtered(csv, axis, transform, gene_subset, value_col,
                                    branch=branch)
            if df.empty:
                continue
            sub = df[df["split"] == split]
            v = sub["value"].astype(float).to_numpy()
            if v.size == 0:
                continue
            for vv, sd in zip(v, sub["seed"].astype(int)):
                long_rows.append({
                    "metric": value_col, "panel": suffix, "axis": axis,
                    "transform": transform, "gene_subset": gene_subset,
                    "branch": branch, "method": label, "split": split,
                    "seed": int(sd), "value": float(vv),
                })
    return pd.DataFrame(long_rows, columns=PANEL_COLUMNS)


def write_panels(S: Store, out_dir: Path, R: Report) -> Dict[Tuple[str, str], pd.DataFrame]:
    R.section("Panel CSVs")
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for set_name, stem, branch, labels in PANEL_FILES:
        resolved: "OrderedDict[str, Optional[Path]]" = OrderedDict()
        for label in labels:
            fr = S.get(set_name, stem, label)
            if not fr:
                msg = f"{set_name or 'canonical'}/{stem}: no rows for {label!r}"
                if set_name == "" and stem in MAIN_STEMS:
                    R.fail("panels", msg + " (canonical table incomplete)")
                else:
                    R.check("panels", False, msg, warn_only=True)
                resolved[label] = None
                continue
            df = pd.concat(fr, ignore_index=True).sort_values("seed", kind="stable")
            seeds = sorted(df["seed"].unique().tolist())
            if len(seeds) != 5:
                R.check("panels", False, f"{set_name or 'canonical'}/{stem}/{label}: seeds "
                                         f"{seeds}", warn_only=True)
            pdir = out_dir / "per_seed" / (set_name or "canonical") / stem
            pdir.mkdir(parents=True, exist_ok=True)
            path = pdir / f"{_slug(label)}.csv"
            _order_cols(df).to_csv(path, index=False)
            resolved[label] = path
        long = build_panel_long(resolved, branch)
        target = (out_dir / set_name if set_name else out_dir)
        target.mkdir(parents=True, exist_ok=True)
        long.to_csv(target / f"{stem}.csv", index=False)
        out[(set_name, stem)] = long
        R.info("panels", f"wrote {target / (stem + '.csv')}: {len(long)} rows, methods "
                         f"{list(dict.fromkeys(long['method'].tolist()))}")
    return out


def summarise(panels: Dict[Tuple[str, str], pd.DataFrame], out_dir: Path, R: Report) -> None:
    R.section("Summary: 5-seed means of the Table 2 columns (split test)")
    rows = []
    for (set_name, stem), long in panels.items():
        for col, code in TABLE2_COLUMNS:
            d = long[long["panel"] == code]
            for meth, g in d.groupby("method", sort=False):
                v = g["value"].astype(float)
                rows.append({"set": set_name or "canonical", "file": stem, "method": meth,
                             "column": col, "panel": code, "mean": float(v.mean()),
                             "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                             "n_seeds": int(v.size)})
    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "summary_means.csv", index=False)
    if summ.empty:
        return
    cols = [c for c, _ in TABLE2_COLUMNS]
    hdr = f"{'method':<34s}{'set':<11s}" + "".join(f"{c:>10s}" for c in cols)
    for stem in MAIN_STEMS + ["imputation_benchmark_panel_niche_ownfill",
                              "reconstruction_benchmark_panel_niche_reference"]:
        d = summ[summ["file"] == stem]
        if d.empty:
            continue
        print(f"\n-- {stem} --\n{hdr}")
        R.lines.append(f"\n-- {stem} --\n{hdr}")
        for meth in dict.fromkeys(d["method"].tolist()):
            for st in ("canonical", "asstored", "redecoded"):
                e = d[(d["method"] == meth) & (d["set"] == st)]
                if e.empty:
                    continue
                m = e.set_index("column")["mean"]
                line = f"{meth:<34s}{st:<11s}" + "".join(
                    f"{m.get(c, float('nan')):10.4f}" for c in cols)
                print(line)
                R.lines.append(line)
            mean = {st: d[(d["method"] == meth) & (d["set"] == st)].set_index("column")["mean"]
                    for st in ("canonical", "asstored", "redecoded")}
            # delta = canonical - asstored; drift = redecoded - asstored (the part
            # of delta that comes from the re-decode, not the canonical scoring)
            for tag, a, b in (("delta", mean["canonical"], mean["asstored"]),
                              ("drift", mean["redecoded"], mean["asstored"])):
                if a.empty or b.empty:
                    continue
                line = f"{meth:<34s}{tag:<11s}" + "".join(
                    f"{a.get(c, float('nan')) - b.get(c, float('nan')):+10.4f}" for c in cols)
                print(line)
                R.lines.append(line)


def compare_existing(panels, existing_dir: Path, R: Report, tol: float) -> None:
    R.section(f"S5: as-stored panel rows vs existing panel CSVs in {existing_dir}")
    for stem in MAIN_STEMS:
        f = existing_dir / f"{stem}.csv"
        new = panels.get(("asstored", stem))
        if not f.is_file() or new is None:
            R.skip(f"S5/{stem}", f"missing {'existing ' + str(f) if not f.is_file() else 'as-stored panel'}")
            continue
        E = pd.read_csv(f)
        key = ["panel", "method", "seed"]
        m = new.merge(E[key + ["value"]], on=key, how="outer", suffixes=("_new", "_old"),
                      indicator=True)
        for meth in dict.fromkeys(m["method"].tolist()):
            g = m[m["method"] == meth]
            both = g[g["_merge"] == "both"]
            d = (both["value_new"] - both["value_old"]).abs()
            dmax = float(d.max()) if len(d) else float("nan")
            n_new = int((g["_merge"] == "left_only").sum())
            n_old = int((g["_merge"] == "right_only").sum())
            R.check(f"S5/{stem}/{meth}", len(both) > 0 and n_new == 0 and n_old == 0
                    and dmax <= tol,
                    f"{len(both)} rows matched, max|diff|={dmax:.3e}; only as-stored {n_new}, "
                    f"only existing {n_old}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", type=str, default=DEFAULT_DATASET_TAG)
    p.add_argument("--silver-dir", type=str, default=DEFAULT_SILVER_DIR,
                   help="Coord-aligned silver dir the baselines were re-run on.")
    p.add_argument("--batch-key", type=str, default="adata_batch_id")
    p.add_argument("--squint-variant", type=str, default=None,
                   help="SQUINT region-holdout stage-1 variant folder (default: the "
                        "filmscale + cross-MNN dec-w32 spine +region-holdout+<dataset>).")
    p.add_argument("--squint-seeds", nargs="+", default=list(DEFAULT_SQUINT_SEEDS),
                   help="The 5 SQUINT seed run folders inside --squint-variant.")
    p.add_argument("--squint-multiseed-ts", type=str, default=DEFAULT_MULTISEED_TS,
                   help="Timestamp of <variant>__multiseed/<ts>/metrics/" + CSV_NAME)
    p.add_argument("--mc-arm-dir", type=Path, default=None,
                   help="RE-DECODED SQUINT (MC) arm: folder with <seed>/metrics/" + NPZ_NAME
                        + " (and the arm's per-seed CSV). Default <artifacts>/<dataset>/"
                        + DEFAULT_STAGE2_ROOT + "/" + DEFAULT_MC_ARM + " (the paper arm, "
                        "which has no npz: canonical SQUINT imputation rows then FAIL).")
    p.add_argument("--nomc-arm-dir", type=Path, default=None,
                   help="RE-DECODED SQUINT (No MC) arm folder. Default <artifacts>/<dataset>/"
                        + DEFAULT_STAGE2_ROOT + "/" + DEFAULT_NOMC_ARM)
    p.add_argument("--asstored-mc-arm-dir", type=Path, default=None,
                   help="PAPER SQUINT (MC) arm (as-stored rows, vs-paper link). Default "
                        "<artifacts>/<dataset>/" + DEFAULT_STAGE2_ROOT + "/" + DEFAULT_MC_ARM)
    p.add_argument("--asstored-nomc-arm-dir", type=Path, default=None,
                   help="PAPER SQUINT (No MC) arm. Default <artifacts>/<dataset>/"
                        + DEFAULT_STAGE2_ROOT + "/" + DEFAULT_NOMC_ARM)
    p.add_argument("--stage2-codes-dir", type=Path, default=None,
                   help="Trained stage-2 arm whose <seed>/" + CODES_NAME + " the re-decodes "
                        "must have decoded. Default <artifacts>/<dataset>/"
                        + DEFAULT_STAGE2_ROOT + "/" + DEFAULT_CODES_ARM)
    p.add_argument("--mc-decode-samples", type=int, default=1000,
                   help="Expected --decode-samples K of the SQUINT (MC) arm (No MC: 1, hard).")
    p.add_argument("--selfcheck-min", type=float, default=0.99,
                   help="Min decode true-code self-check (cell-wise Pearson median) in the npz.")
    p.add_argument("--allow-ckpt-mismatch", action="store_true",
                   help="Accept a decode checkpoint whose weights differ from the predict-time "
                        "one (the paper arms' last.ckpt for seeds 2 and 4) as a WARN instead "
                        "of a FAIL.")
    p.add_argument("--baseline-dirs", nargs="*", default=[], metavar="KEY=PATH",
                   help="Override a baseline run (KEY in " + ", ".join(BASELINES) + "; PATH "
                        "= a run dir with predicted_adata_seed*.h5ad, or a variant folder "
                        "holding exactly one timestamp run). Default: the "
                        "+coord-aligned variant folders under <artifacts>/<dataset>/.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="NEW output folder (default <artifacts>/table2_canonical/<ts>). "
                        "Must not exist or be empty; never inside <artifacts>/<dataset>.")
    p.add_argument("--existing-tables-dir", type=Path, default=None,
                   help="Optional folder with the existing panel CSVs (e.g. the paper's "
                        "tables/table2_coord_aligned) for S5.")
    p.add_argument("--nbr-k", type=int, default=16,
                   help="Neighbours of the canonical directed kNN+self graph.")
    p.add_argument("--n-hvg", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu", help="Device for the S3 replay.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--tol", type=float, default=1e-6,
                   help="Max |diff| for S0 / S1 / S2 / S4 (same data, same harness).")
    p.add_argument("--tol-order", type=float, default=1e-5,
                   help="Max relative |diff| (|a-b| / max(|a|,|b|,1)) when only row / "
                        "gene order (float summation order) differs: canonical-order "
                        "re-scores vs the stored CSVs.")
    p.add_argument("--tol-redecode", type=float, default=1e-4,
                   help="Max relative |diff| between the re-decoded and the paper SQUINT "
                        "(No MC) arm CSV cell rows (same flags and checkpoint; GPU float "
                        "order only).")
    p.add_argument("--tol-redecode-mc", type=float, default=1e-3,
                   help="Same for the SQUINT (MC) arm (= stage2_decode_pearson's "
                        "--compare-csv default for K > 1: multinomial sampling drifts "
                        "on another GPU).")
    p.add_argument("--replay-tol", type=float, default=1e-3,
                   help="Max row relative L2 of the S3 decoder replays on the stored "
                        "embeddings vs stored X_hat / X_hat_nbr (validate_table2.sh "
                        "accepts its exact replay below 1e-3).")
    p.add_argument("--fast-selfchecks", action="store_true",
                   help="Literal build_pearson_dataframe only for the first seed of each "
                        "run (S0 proves the test-only scorer equal); other seeds use the "
                        "test-only scorer for S1 / S2. Default: literal for every seed.")
    p.add_argument("--squint-examples-dir", type=Path, default=None,
                   help="squint/examples (for stage2_decode_pearson helpers). Default: "
                        "derived from the installed squint package.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    ds = args.artifacts_root / args.dataset_tag
    if args.squint_variant is None:
        args.squint_variant = f"{_SQUINT_SPINE}+region-holdout+{args.dataset_tag}"
    if args.mc_arm_dir is None:
        args.mc_arm_dir = ds / DEFAULT_STAGE2_ROOT / DEFAULT_MC_ARM
    if args.nomc_arm_dir is None:
        args.nomc_arm_dir = ds / DEFAULT_STAGE2_ROOT / DEFAULT_NOMC_ARM
    if args.asstored_mc_arm_dir is None:
        args.asstored_mc_arm_dir = ds / DEFAULT_STAGE2_ROOT / DEFAULT_MC_ARM
    if args.asstored_nomc_arm_dir is None:
        args.asstored_nomc_arm_dir = ds / DEFAULT_STAGE2_ROOT / DEFAULT_NOMC_ARM
    if args.stage2_codes_dir is None:
        args.stage2_codes_dir = ds / DEFAULT_STAGE2_ROOT / DEFAULT_CODES_ARM
    if len(args.squint_seeds) != 5:
        print(f"WARNING: {len(args.squint_seeds)} SQUINT seed folders (paper: 5)", flush=True)
    for s in args.squint_seeds:
        if not re.search(r"seed_?\d+", s):
            raise SystemExit(f"--squint-seeds entry {s!r} has no '_seed<N>' suffix")
    if len({_seed_of(s) for s in args.squint_seeds}) != len(args.squint_seeds):
        raise SystemExit(f"--squint-seeds has repeated seed numbers: {args.squint_seeds}")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(int(args.threads))
    except Exception:  # noqa: BLE001  (optional)
        pass

    # ---- resolve inputs (config errors stop here, before any heavy work) -------
    over = {}
    for kv in args.baseline_dirs:
        if "=" not in kv:
            raise SystemExit(f"--baseline-dirs entry {kv!r} is not KEY=PATH")
        kk, vv = kv.split("=", 1)
        if kk not in BASELINES:
            raise SystemExit(f"--baseline-dirs key {kk!r} not in {list(BASELINES)}")
        over[kk] = Path(vv)
    base_runs: "OrderedDict[str, Path]" = OrderedDict()
    for kk, (tag, _roles) in BASELINES.items():
        base_runs[kk] = _resolve_run_dir(over.get(kk, ds / tag), f"baseline {kk}")
    sq_dirs = [ds / args.squint_variant / s for s in args.squint_seeds]
    for d in sq_dirs:
        if not (d / "predicted_adata.h5ad").is_file():
            raise SystemExit(f"missing SQUINT predicted_adata: {d / 'predicted_adata.h5ad'}")
    ms_csv = (ds / (args.squint_variant + "__multiseed") / args.squint_multiseed_ts
              / "metrics" / CSV_NAME)
    if not Path(args.silver_dir).is_dir():
        raise SystemExit(f"--silver-dir {args.silver_dir} is not a directory")
    s2d, s2d_path = _import_s2d(args.squint_examples_dir)

    # ---- output folder: NEW, outside every input folder ----------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or (args.artifacts_root / "table2_canonical" / ts)).absolute()
    guards = [ds, Path(args.silver_dir), args.mc_arm_dir, args.nomc_arm_dir,
              args.asstored_mc_arm_dir, args.asstored_nomc_arm_dir, args.stage2_codes_dir,
              *base_runs.values(), *sq_dirs]
    # both the given and the symlink-resolved spellings (realpath resolves the
    # existing prefix of a not-yet-created out dir)
    outs = {out_dir, Path(os.path.realpath(out_dir))}
    for g in guards:
        for gg in {Path(g).absolute(), Path(os.path.realpath(g))}:
            for o in outs:
                if o == gg or gg in o.parents:
                    raise SystemExit(f"refusing to write inside an input folder: "
                                     f"{out_dir} ({o}) is in {g} ({gg})")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir {out_dir} exists and is not empty (never overwritten)")
    out_dir.mkdir(parents=True, exist_ok=True)

    R = Report()
    R.section("Inputs")
    cfg ={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(out_dir=str(out_dir), baseline_runs={k: str(v) for k, v in base_runs.items()},
               squint_runs=[str(d) for d in sq_dirs], multiseed_csv=str(ms_csv),
               stage2_decode_pearson=str(s2d_path), host=platform.node(),
               python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
               anndata=ad.__version__, started=ts)
    with open(out_dir / "run_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    for kk, v in base_runs.items():
        R.info("inputs", f"baseline {kk:<13s} {v}")
    R.info("inputs", f"SQUINT variant {args.squint_variant}; seeds {args.squint_seeds}")
    R.info("inputs", f"multiseed CSV {ms_csv}")
    R.info("inputs", f"re-decoded MC arm {args.mc_arm_dir}; No-MC arm {args.nomc_arm_dir}")
    R.info("inputs", f"paper (as-stored) MC arm {args.asstored_mc_arm_dir}; No-MC arm "
                     f"{args.asstored_nomc_arm_dir}; stage-2 codes {args.stage2_codes_dir}; "
                     f"expected MC K {args.mc_decode_samples}; allow ckpt mismatch "
                     f"{bool(args.allow_ckpt_mismatch)}")
    R.info("inputs", f"stage2_decode_pearson helpers from {s2d_path}")
    R.info("inputs", f"out dir {out_dir}")
    for label, arm in (("SQUINT (MC)", args.mc_arm_dir), ("SQUINT (No MC)", args.nomc_arm_dir)):
        miss = [s for s in args.squint_seeds
                if not (Path(arm) / s / "metrics" / NPZ_NAME).is_file()]
        R.check(f"S4/{label}/preflight", not miss,
                f"{NPZ_NAME} present for {len(args.squint_seeds) - len(miss)}/"
                f"{len(args.squint_seeds)} seeds under {arm}"
                + (f"; missing {miss} (FAIL per seed below)" if miss else ""),
                warn_only=True)
    MS = pd.read_csv(ms_csv) if ms_csv.is_file() else None
    if MS is None:
        R.fail("S2/inputs", f"multiseed CSV missing: {ms_csv}")

    C = build_canonical(args, R)
    S = Store()
    for kk, run in base_runs.items():
        process_baseline(kk, run, C, args, R, S)
    R.section("As-stored SQUINT imputation rows (paper arms)")
    for s in args.squint_seeds:
        R.guarded(f"asstored/{s}", add_asstored_arms, s, args, R, S)
    for s in args.squint_seeds:
        R.guarded(f"squint/{s}", process_squint_seed, s, C, args, R, S, s2d, MS)
        gc.collect()

    panels = write_panels(S, out_dir, R)
    if args.existing_tables_dir is not None:
        compare_existing(panels, Path(args.existing_tables_dir), R, args.tol)
    summarise(panels, out_dir, R)

    n_fail, n_warn = R.n.get("FAIL", 0), R.n.get("WARN", 0)
    R.section("Self-check verdict")
    if n_fail:
        verdict = (f"!!! {n_fail} SELF-CHECK FAILURE(S): canonical CSVs were written but "
                   f"are NOT validated; read the FAIL lines ({R.counts()})")
    elif n_warn:
        verdict = (f"ALL SELF-CHECKS PASSED, BUT {n_warn} WARN(S) TO READ (accepted "
                   f"deviations, e.g. --allow-ckpt-mismatch, re-decode drift, missing "
                   f"seeds) ({R.counts()})")
    else:
        verdict = f"ALL SELF-CHECKS PASSED ({R.counts()})"
    print(verdict, flush=True)
    R.lines.append(verdict)
    with open(out_dir / "selfcheck_report.txt", "w") as fh:
        fh.write(f"score_table2_canonical.py  started {ts}  runtime "
                 f"{time.time() - t_start:.0f}s\nout_dir {out_dir}\n")
        fh.write(json.dumps(cfg, indent=2) + "\n")
        fh.write("\n".join(R.lines) + "\n")
    print(f"\nwrote {out_dir}  [total {time.time() - t_start:.0f}s]\n"
          f"significance: python {_BENCH / 'compute_table2_significance.py'} "
          f"--tables-dir {out_dir}", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
