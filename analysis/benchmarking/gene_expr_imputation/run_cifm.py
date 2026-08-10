#!/usr/bin/env python
"""
run_cifm.py — CIFM as a held-out-region gene-expression imputation baseline.
=============================================================================
CIFM ("Building Foundation Models to Characterize Cellular Interactions via
Geometric Self-Supervised Learning on Spatial Genomics", bioRxiv 2025.01.25.
634867) is a 100M-parameter geometric GNN pretrained on 23M cells whose
self-supervised objective is *exactly* our imputation task: infer a cell's
expression from its surrounding microenvironment. Reviewers asked for it as a
Table 2 comparator, so this runner evaluates the RELEASED checkpoint
(`ynyou/CIFM`) under the IDENTICAL protocol used for SQUINT / GeST / kNN.

Nothing here is re-implemented: we call the released weights. What this file
does is (a) put our data in the format CIFM expects, (b) keep the split, the
leak-freedom and the metric panel bit-identical to the other baselines.

PROTOCOL PARITY — every one of these mirrors GeST exactly
---------------------------------------------------------
1. SPLIT. `apply_holdout_regions(adata, batch_key, DEFAULT_HOLDOUT_REGIONS)` —
   the same two fixed per-section rectangles SQUINT itself uses
   (`run_squint.py::_patch_holdout_regions`). Seed-independent, applied ONCE
   before the seed loop. We print the per-batch held-out counts so they can be
   diffed against the GeST log — the fastest proof the split is identical.
2. GENE IDs. Our panel is 431 MOUSE SYMBOLS and CIFM's vocabulary is 18,289
   HUMAN Ensembl IDs (`channel2ensembl.pt`; verified 0 ENSMUSG entries), so a
   mouse→human ortholog map is required — exactly as for the paper's other
   human-only foundation baselines. We call the SAME helper with the SAME
   arguments as `run_scgpt.py:272-276`:
       add_human_ortholog_ensembl_ids(adata, species=species, gene_col_in=None,
           gene_col_out="human_ensembl_id", fallback_to_input=False,
           verbose=True, inplace=True)
   (`_nicheformer_embedding.add_human_ortholog_ensembl_ids`: mygene → mouse
   Ensembl → Ensembl REST homology → mygene human DB → NCBI HomoloGene.) This
   is the same mapping `submit_all_benchmarks.sh:337-339` applies to scGPT,
   scGPT-spatial and Geneformer for `mmb0-1b_smb1-1b_1p`. The realised map is
   written to `<out_dir>/ortholog_mapping.csv` so the run is auditable and can
   be pinned/diffed later (the helper hits live APIs and is NOT cached).
3. METRICS. `add_neighborhood_layers(..., n_neighs=16)` then
   `build_pearson_dataframe(..., log1p=True, n_hvg=50)` then
   `write_pearson_outputs` — the shared implementation, so Pearson/Spearman/
   RMSE/AUROC/AP, the HVG-50 set and the marker set are computed identically.
4. OUTPUT CONTRACT. `layers["X_hat"]` = (n_obs, n_vars) float32 on the RAW
   COUNT scale for ALL cells (train and test), same row/var order as loaded.

CIFM-SPECIFIC HANDLING (the parts that needed a decision)
---------------------------------------------------------
* INPUT FORMAT. Per the official `test.ipynb`, CIFM consumes
  `normalize_total(target_sum=1e4)` + `log1p` of raw counts, and
  `obsm['spatial']` **in micrometres** (it builds `radius_graph(r=20)`).
  We normalise a working copy only, and print a coordinate diagnostic (median
  nearest-neighbour distance + the mean neighbour count at r=20) so a unit
  mismatch cannot pass silently. `--coord-scale` rescales if needed.
* LEAK-FREE READ DEPTH. CIFM emits a log-normalised profile, NOT counts, so it
  needs a depth to become count-scale. We port `_neighbor_read_depth` verbatim
  from `squint/examples/stage2_decode_pearson.py:415-448`: a held-out cell's
  depth is the mean library size of its k=16 nearest OBSERVED cells in the same
  section; train cells keep their own. This is the same rule SQUINT (MC) uses,
  and it never reads a held-out cell's own counts. Using the cell's own depth
  would silently corrupt RMSE / AUROC / AP while leaving Pearson plausible.
* PREDICTING TRAIN CELLS, LEAK-FREE. The harness needs `X_hat` everywhere
  (train rows feed the train/all splits and the niche aggregation at test
  cells). We predict train cells in chunks with the chunk itself removed from
  the context, mirroring GeST's `cand = cand[cand != ti]`
  (`gest/train.py:193`). Test cells never appear in any context.
* PER-SECTION GRAPHS. Every forward pass is restricted to one section, so no
  edge ever crosses sections — matching our convention everywhere else.
* DROPOUT GATE. CIFM's `encode_decode` hard-gates its output
  (`expressions_dec[dropouts_dec <= 0.5] = 0`). GeST and the kNN floor are
  ungated continuous means, and the harness derives AUROC/AP from the
  magnitude of `X_hat`, so gating would both disadvantage CIFM and break
  comparability. DEFAULT IS UNGATED; `--apply-dropout-gate` reproduces CIFM's
  native behaviour. Whichever is used is recorded in the config stub.
* SEEDS. CIFM is a frozen checkpoint with deterministic inference, so all seeds
  give identical numbers. We follow the established convention for the
  deterministic baseline (`run_knn_spatial.py:214-217`): run once and replicate
  the rows for schema parity, and say so. Do NOT present this as 5 replicates.

Usage
-----
  python run_cifm.py --use-default-holdout-regions --seeds 0,1,2,3,4 \
      --cifm-repo /nfs/team361/sb75/models/CIFM

Requires (own venv): torch, torch-geometric (+scatter/sparse/cluster), e3nn,
scanpy, mygene, huggingface_hub, and the `models_cifm/` package from the CIFM
repo on PYTHONPATH (`--cifm-repo`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    write_predicted_adata,
)

# the ortholog helper lives with the cell-type baselines
_CT_DIR = _THIS.parent / "cell_type_identification"
if str(_CT_DIR) not in sys.path:
    sys.path.insert(0, str(_CT_DIR))

DEFAULT_VARIANT_TAG = "baseline-cifm+region-holdout"
DEFAULT_READ_DEPTH_NEIGHS = 16          # == SQUINT's --read-depth-neighs
CIFM_HF_REPO = "ynyou/CIFM"


# ---------------------------------------------------------------------------
# Leak-free read depth.
#
# We do NOT re-implement this: we IMPORT SQUINT's own
# `stage2_decode_pearson._neighbor_read_depth`, so the depth rule applied to
# CIFM cannot drift from the one used to produce the SQUINT (MC) row. That
# module is safe to import (its only top-level statement is a sys.path insert).
# The local copy below is a byte-faithful fallback used only if the squint
# checkout is not importable, and it prints a loud warning when that happens.
# ---------------------------------------------------------------------------
def _neighbor_read_depth_fallback(coords, batch, gidx, L, k=DEFAULT_READ_DEPTH_NEIGHS):
    """Verbatim copy of squint/examples/stage2_decode_pearson.py:415-448."""
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    held = np.zeros(n, dtype=bool); held[gidx] = True
    L = np.asarray(L, dtype=np.float32)
    rd = np.empty(gidx.size, dtype=np.float32)
    pos = {int(g): i for i, g in enumerate(gidx)}
    global_obs_mean = float(L[~held].mean()) if (~held).any() else float(L.mean())

    for b in np.unique(batch):
        in_b = (batch == b)
        obs_b = np.where(in_b & ~held)[0]
        hold_b = np.where(in_b & held)[0]
        if hold_b.size == 0:
            continue
        if obs_b.size == 0:
            for g in hold_b:
                rd[pos[int(g)]] = global_obs_mean
            continue
        kk = int(min(k, obs_b.size))
        nn = NearestNeighbors(n_neighbors=kk).fit(coords[obs_b])
        _, nbr = nn.kneighbors(coords[hold_b])
        depths = L[obs_b][nbr].mean(axis=1)
        for j, g in enumerate(hold_b):
            rd[pos[int(g)]] = float(depths[j])
    return rd


def resolve_neighbor_read_depth(squint_examples: Optional[Path]):
    """Prefer SQUINT's canonical implementation; fall back to the local copy."""
    cands = [squint_examples] if squint_examples else []
    cands += [_THIS.parents[3] / "squint" / "examples",
              Path("/nfs/team361/sb75/squint/examples")]
    for c in cands:
        if c is None or not Path(c).is_dir():
            continue
        try:
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            from stage2_decode_pearson import _neighbor_read_depth as fn
            print(f"  read-depth: using SQUINT's canonical "
                  f"_neighbor_read_depth from {c}")
            return fn
        except Exception as e:  # noqa: BLE001
            print(f"  read-depth: could not import from {c} ({type(e).__name__}: {e})")
    print("  read-depth: WARNING — falling back to the local verbatim copy; "
          "verify it still matches stage2_decode_pearson.py")
    return _neighbor_read_depth_fallback


# ---------------------------------------------------------------------------
# Gene-ID mapping — same helper + same arguments as run_scgpt.py
# ---------------------------------------------------------------------------
def build_channel_map(
        adata: ad.AnnData,
        species: str,
        out_csv: Optional[Path] = None,
        ortholog_csv: Optional[Path] = None,
    ) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Populate ``var['human_ensembl_id']`` with the SAME call run_scgpt.py makes,
    then convert it to CIFM's ``channel2ensembl_ids_target`` format:
    one list per gene channel, EMPTY for genes with no human ortholog (which
    `channel_matching` then leaves zero-initialised).

    If ``ortholog_csv`` is given, the mapping is READ from that file instead of
    being re-derived. The helper queries mygene.info and the Ensembl REST API
    live and is not cached, so pinning the map (a) makes the run reproducible
    and (b) allows compute nodes with no outbound internet. Precompute it on a
    login node with ``--write-ortholog-csv ... --ortholog-only``.
    """
    if ortholog_csv is not None:
        m = pd.read_csv(ortholog_csv)
        for col in ("gene", "human_ensembl_id"):
            if col not in m.columns:
                raise SystemExit(f"--ortholog-csv {ortholog_csv} lacks a "
                                 f"{col!r} column (need gene, human_ensembl_id)")
        lut = dict(zip(m["gene"].astype(str), m["human_ensembl_id"].astype(str)))
        genes = adata.var_names.astype(str).to_numpy()
        missing = [g for g in genes if g not in lut]
        if missing:
            raise SystemExit(
                f"--ortholog-csv is missing {len(missing)} of {len(genes)} genes "
                f"in this panel (e.g. {missing[:5]}). Regenerate it for this "
                f"dataset with --write-ortholog-csv --ortholog-only.")
        raw = np.array([lut[g] for g in genes], dtype=object)
        adata.var["human_ensembl_id"] = raw
        print(f"  ortholog map: loaded from {ortholog_csv} (no network call)")
    else:
        from _nicheformer_embedding import add_human_ortholog_ensembl_ids

        add_human_ortholog_ensembl_ids(
            adata, species=species, gene_col_in=None,
            gene_col_out="human_ensembl_id", fallback_to_input=False,
            verbose=True, inplace=True,
        )
        raw = adata.var["human_ensembl_id"].astype(str).to_numpy()

    target: List[List[str]] = []
    for v in raw:
        v = (v or "").strip()
        # fallback_to_input=False leaves unmapped genes as NaN/None/empty;
        # only keep well-formed human Ensembl gene IDs.
        target.append([v] if v.startswith("ENSG") else [])

    n_mapped = sum(1 for t in target if t)
    stats = {"n_genes": len(target), "n_mapped": n_mapped,
             "n_unmapped": len(target) - n_mapped}
    print(f"  ortholog map: {n_mapped}/{len(target)} genes -> human ENSG "
          f"({100.0*n_mapped/max(1,len(target)):.1f}%)")

    if out_csv is not None:
        pd.DataFrame({
            "gene": adata.var_names.astype(str),
            "human_ensembl_id": raw,
            "used_by_cifm": [bool(t) for t in target],
        }).to_csv(out_csv, index=False)
        print(f"  wrote realised mapping -> {out_csv}")
    return target, stats


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_cifm(cifm_repo: Path, device: str):
    """Load the released checkpoint exactly as the official test.ipynb does."""
    import torch

    repo = Path(cifm_repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402

    args_model = torch.load(repo / "models_cifm" / "args.pt")
    model = CIFM.from_pretrained(CIFM_HF_REPO, args=args_model).to(device)
    model.channel2ensembl_ids_source = torch.load(
        repo / "models_cifm" / "channel2ensembl.pt")
    model.eval()
    print(f"  CIFM loaded (radius_spatial_graph="
          f"{getattr(model, 'radius_spatial_graph', '?')}, device={device})")
    return model


# ---------------------------------------------------------------------------
# Coordinate diagnostic — a unit mismatch must not pass silently
# ---------------------------------------------------------------------------
def coord_diagnostic(coords: np.ndarray, batch: np.ndarray, radius: float) -> None:
    from sklearn.neighbors import NearestNeighbors

    print(f"\n=== Coordinate diagnostic (CIFM expects micrometres, r={radius}) ===")
    for b in np.unique(batch):
        c = coords[batch == b]
        nn = NearestNeighbors(n_neighbors=2).fit(c)
        d, _ = nn.kneighbors(c)
        med = float(np.median(d[:, 1]))
        span = (float(c[:, 0].ptp()), float(c[:, 1].ptp()))
        n_within = NearestNeighbors(radius=radius).fit(c).radius_neighbors(
            c[: min(500, len(c))], return_distance=False)
        mean_deg = float(np.mean([len(x) - 1 for x in n_within]))
        flag = "" if 1.0 <= med <= 100.0 else "   <-- SUSPICIOUS: not micrometre-like"
        print(f"  batch {b}: n={len(c)}  median NN dist={med:.2f}  "
              f"xy span={span[0]:.0f}x{span[1]:.0f}  mean deg @r={radius}: "
              f"{mean_deg:.1f}{flag}")
    print("  (a mean degree near 0 or in the thousands means the unit is wrong)")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _predict_chunk(model, ctx_X, ctx_xy, q_xy, device, apply_gate: bool):
    """
    Faithful re-implementation of CIFM.predict_cells_at_locations that also
    exposes the continuous dropout head (needed so we can choose NOT to gate).
    ctx_X: (n_ctx, G) normalised+log1p context expression. Returns (n_q, G).
    """
    import torch
    from torch_geometric.nn import radius_graph

    n_ctx, G = ctx_X.shape
    n_q = q_xy.shape[0]
    with torch.no_grad():
        expr = torch.tensor(ctx_X, dtype=torch.float32, device=device)
        expr = torch.cat([expr, torch.zeros(n_q, G, device=device)], dim=0)

        xy = np.concatenate([ctx_xy, q_xy], axis=0)
        coords = torch.tensor(xy, dtype=torch.float32)
        coords = torch.cat([coords, torch.zeros(coords.shape[0], 1)], dim=1).to(device)

        edge_index = radius_graph(coords, r=model.radius_spatial_graph,
                                  max_num_neighbors=10000, loop=True)
        mapping = torch.arange(n_ctx, n_ctx + n_q, device=device)

        emb = model.encode(expr, coords, edge_index)
        emb[mapping] = model.mask_embedding(
            torch.zeros(1, dtype=torch.int64, device=device))
        emb_dec = model.mask_cell_decoder(emb, coords, edge_index)[0][mapping]

        pred = model.relu(model.mask_cell_expression(emb_dec))
        if apply_gate:
            drop = model.sigmoid(model.mask_cell_dropout(emb_dec))
            pred = pred.clone()
            pred[drop <= 0.5] = 0.0
        return pred.detach().cpu().numpy().astype(np.float32)


def predict_all_cells(
        model, adata: ad.AnnData, batch_key: str, device: str,
        apply_gate: bool, train_chunk: int, coord_scale: float,
    ) -> np.ndarray:
    """
    Log-space predictions for EVERY cell, leak-free, section by section.

    * test cells   : context = all TRAIN cells of that section.
    * train cells  : context = all TRAIN cells of that section MINUS the chunk
                     being predicted (mirrors GeST dropping the query itself).
    """
    import scanpy as sc

    # CIFM's expected input format: normalize_total(1e4) + log1p of raw counts.
    work = ad.AnnData(
        X=adata.X.copy(),
        obs=adata.obs[[batch_key, "data_split"]].copy(),
        var=adata.var[[]].copy(),
    )
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    Xn = np.asarray(work.X.todense() if hasattr(work.X, "todense") else work.X,
                    dtype=np.float32)

    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2] * coord_scale
    batch = adata.obs[batch_key].to_numpy()
    is_train = (adata.obs["data_split"].to_numpy() == "train")

    G = Xn.shape[1]
    out = np.zeros((adata.n_obs, G), dtype=np.float32)

    for b in np.unique(batch):
        m = (batch == b)
        tr_idx = np.where(m & is_train)[0]
        te_idx = np.where(m & ~is_train)[0]
        print(f"\n  -- section {b}: {tr_idx.size} train (context), "
              f"{te_idx.size} held out --")

        # ---- held-out cells: full train context, single pass -------------
        if te_idx.size:
            t0 = time.time()
            out[te_idx] = _predict_chunk(model, Xn[tr_idx], xy[tr_idx],
                                         xy[te_idx], device, apply_gate)
            print(f"     test  : {te_idx.size} cells in {time.time()-t0:.1f}s")

        # ---- train cells: chunked, chunk excluded from its own context ----
        n_chunks = int(np.ceil(tr_idx.size / max(1, train_chunk)))
        t0 = time.time()
        for ci in range(n_chunks):
            q = tr_idx[ci * train_chunk:(ci + 1) * train_chunk]
            if q.size == 0:
                continue
            ctx = np.setdiff1d(tr_idx, q, assume_unique=False)
            if ctx.size == 0:
                print("     WARNING: empty context for a train chunk; skipping")
                continue
            out[q] = _predict_chunk(model, Xn[ctx], xy[ctx], xy[q],
                                    device, apply_gate)
        print(f"     train : {tr_idx.size} cells in {n_chunks} chunk(s), "
              f"{time.time()-t0:.1f}s")
    return out


def to_counts(pred_log: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    CIFM predicts log1p(1e4-normalised) values. Convert to the RAW COUNT scale
    the harness expects: expm1 -> renormalise to a unit profile -> x read depth
    (leak-free for held-out cells). Mirrors how SQUINT's NB rate is scaled by
    library size (`run_scvi.py:129-133`, `stage2_decode_pearson.py`).
    """
    rate = np.expm1(np.clip(pred_log, 0.0, None)).astype(np.float32)
    rs = rate.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0).astype(np.float32)
    rate = rate / rs
    return (rate * depth[:, None].astype(np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="CIFM held-out-region imputation baseline (protocol-matched "
                    "to SQUINT / GeST).")
    # --- shared harness flags (copied verbatim from run_gest.py) ---
    p.add_argument("--silver-dir", type=Path, default=DEFAULT_SILVER_DIR)
    p.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--dataset-tag", default=DEFAULT_DATASET_TAG)
    p.add_argument("--variant-tag", default=DEFAULT_VARIANT_TAG)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--batch-key", default="adata_batch_id")
    p.add_argument("--nbr-neighs", type=int, default=16)
    p.add_argument("--use-default-holdout-regions", action="store_true", default=True)
    p.add_argument("--no-default-holdout-regions", dest="use_default_holdout_regions",
                   action="store_false")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    # --- CIFM-specific ---
    p.add_argument("--cifm-repo", type=Path, required=True,
                   help="Local clone of the ynyou/CIFM HF repo (must contain "
                        "models_cifm/{cifm.py,args.pt,channel2ensembl.pt}).")
    p.add_argument("--species", default="mouse",
                   help="Source species of adata.var_names, for the ortholog map.")
    p.add_argument("--ortholog-csv", type=Path, default=None,
                   help="Read a PINNED mouse->human map (cols: gene, "
                        "human_ensembl_id) instead of querying mygene/Ensembl. "
                        "Use on compute nodes without internet, and for exact "
                        "reproducibility (the helper is not cached).")
    p.add_argument("--write-ortholog-csv", type=Path, default=None,
                   help="Also write the realised map here (in addition to "
                        "<out_dir>/ortholog_mapping.csv).")
    p.add_argument("--ortholog-only", action="store_true",
                   help="Resolve + write the ortholog map, then exit without "
                        "loading CIFM. Run this on a login node with internet.")
    p.add_argument("--apply-dropout-gate", action="store_true",
                   help="Reproduce CIFM's native hard gate (expr[p_drop<=0.5]=0). "
                        "OFF by default: GeST/kNN are ungated and the harness "
                        "derives AUROC/AP from prediction magnitude.")
    p.add_argument("--read-depth-neighs", type=int, default=DEFAULT_READ_DEPTH_NEIGHS)
    p.add_argument("--squint-examples", type=Path, default=None,
                   help="Path to squint/examples, so we can import SQUINT's own "
                        "_neighbor_read_depth instead of a copy (recommended).")
    p.add_argument("--train-chunk", type=int, default=4096,
                   help="Train cells predicted per forward pass (each chunk is "
                        "removed from its own context to stay leak-free).")
    p.add_argument("--coord-scale", type=float, default=1.0,
                   help="Multiply obsm['spatial'] by this to reach micrometres.")
    args = p.parse_args(argv)

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip() != ""]

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = args.artifacts_root / args.dataset_tag / args.variant_tag / ts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {args.out_dir}\nSeeds   : {seeds}\nDevice  : {device}")

    print("\n=== Loading silver ===")
    adata = load_silver_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
    if args.smoke:
        sub = np.random.default_rng(0).choice(adata.n_obs,
                                             size=min(4000, adata.n_obs),
                                             replace=False)
        adata = adata[np.sort(sub)].copy()
        print(f"  SMOKE: subset to n_obs={adata.n_obs}")

    print("\n=== Held-out region split (identical to SQUINT / GeST) ===")
    regions = DEFAULT_HOLDOUT_REGIONS if args.use_default_holdout_regions else None
    if regions is None:
        raise SystemExit("Pass --use-default-holdout-regions.")
    apply_holdout_regions(adata, batch_key=args.batch_key, regions=regions)
    n_test = int((adata.obs["data_split"].to_numpy() == "test").sum())
    print(f"  held out {n_test}/{adata.n_obs} cells "
          f"({100.0*n_test/adata.n_obs:.2f}%) — diff this against the GeST log")

    print("\n=== Gene IDs: mouse -> human orthologs "
          "(same helper + args as run_scgpt.py) ===")
    _map_csv = args.out_dir / "ortholog_mapping.csv"
    channel2ensembl_target, map_stats = build_channel_map(
        adata, species=args.species, out_csv=_map_csv,
        ortholog_csv=args.ortholog_csv)
    if args.write_ortholog_csv:
        Path(args.write_ortholog_csv).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_map_csv, args.write_ortholog_csv)
        print(f"  pinned map also written -> {args.write_ortholog_csv}")
    if map_stats["n_mapped"] == 0:
        raise SystemExit(
            "No gene mapped to a human Ensembl ID — CIFM would be all-zero. "
            "Check --species and that mygene / Ensembl REST are reachable.")
    if args.ortholog_only:
        print("\n--ortholog-only: map resolved and written; exiting before "
              "loading CIFM. Pass --ortholog-csv <that file> on the compute node.")
        return 0

    print("\n=== Loading CIFM ===")
    model = load_cifm(args.cifm_repo, device)
    model.channel_matching(channel2ensembl_target, model.channel2ensembl_ids_source)

    coord_diagnostic(
        np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2] * args.coord_scale,
        adata.obs[args.batch_key].to_numpy(),
        float(getattr(model, "radius_spatial_graph", 20.0)))

    print("\n=== CIFM inference (leak-free, per section) ===")
    pred_log = predict_all_cells(
        model, adata, batch_key=args.batch_key, device=device,
        apply_gate=args.apply_dropout_gate, train_chunk=args.train_chunk,
        coord_scale=args.coord_scale)

    print("\n=== Leak-free read depth (k=%d, observed neighbours only) ==="
          % args.read_depth_neighs)
    X = adata.X
    L_all = np.asarray(X.sum(axis=1), dtype=np.float32).ravel()
    depth = L_all.copy()
    neighbor_read_depth = resolve_neighbor_read_depth(args.squint_examples)
    test_idx = np.where(adata.obs["data_split"].to_numpy() == "test")[0]
    if test_idx.size:
        depth[test_idx] = neighbor_read_depth(
            np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2],
            adata.obs[args.batch_key].to_numpy(), test_idx, L_all,
            k=args.read_depth_neighs)
        print(f"  held-out depth: mean={depth[test_idx].mean():.1f} "
              f"(their true mean would have been {L_all[test_idx].mean():.1f} "
              f"— not used)")

    X_hat = to_counts(pred_log, depth)
    if X_hat.shape != (adata.n_obs, adata.n_vars):
        raise RuntimeError(f"X_hat shape {X_hat.shape} != "
                           f"({adata.n_obs}, {adata.n_vars})")
    adata.layers["X_hat"] = X_hat

    print("\n=== Scoring (shared harness) ===")
    add_neighborhood_layers(adata, batch_key=args.batch_key,
                            n_neighs=args.nbr_neighs)
    base = build_pearson_dataframe(adata, seed=seeds[0], log1p=True, n_hvg=50)

    # CIFM is a frozen checkpoint: inference is deterministic, so replicate the
    # rows for schema parity exactly as the deterministic kNN floor does.
    frames = []
    for s in seeds:
        d = base.copy()
        d["seed"] = s
        frames.append(d)
    per_seed = pd.concat(frames, ignore_index=True)
    if len(seeds) > 1:
        print(f"  NOTE: CIFM inference is deterministic — the {len(seeds)} seed "
              f"rows are IDENTICAL replicates (schema parity only), not "
              f"independent runs. Report as such.")

    write_predicted_adata(adata, args.out_dir, seeds[0])
    write_pearson_outputs(args.out_dir, per_seed)

    (args.out_dir / "user_specified_config.yaml").write_text(
        "# CIFM imputation baseline — provenance\n"
        + json.dumps({
            "hf_repo": CIFM_HF_REPO,
            "cifm_repo": str(args.cifm_repo),
            "species": args.species,
            "ortholog_helper": "_nicheformer_embedding.add_human_ortholog_ensembl_ids"
                               " (species=%s, gene_col_in=None, "
                               "fallback_to_input=False) — same as run_scgpt.py"
                               % args.species,
            "genes_mapped": map_stats,
            "input_format": "normalize_total(1e4)+log1p; spatial in micrometres",
            "radius_spatial_graph": float(getattr(model, "radius_spatial_graph", -1)),
            "apply_dropout_gate": bool(args.apply_dropout_gate),
            "read_depth_mode": "neighbor(observed only)",
            "read_depth_neighs": args.read_depth_neighs,
            "train_chunk": args.train_chunk,
            "coord_scale": args.coord_scale,
            "deterministic_seed_replicates": len(seeds) > 1,
            "holdout": "DEFAULT_HOLDOUT_REGIONS (identical to SQUINT/GeST)",
            "nbr_neighs": args.nbr_neighs,
        }, indent=2) + "\n")
    print(f"\nDone -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
