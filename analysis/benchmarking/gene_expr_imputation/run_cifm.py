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

We call the released weights, but the FORWARD PATH IS RE-IMPLEMENTED:
`_predict_chunk` reproduces `encode_decode` so that (a) both decoder heads can be
exposed for `--output`, and (b) the context can be restricted per section to stay
leak-free -- neither of which `predict_cells_at_locations` permits. A
re-implementation can drift, so `verify_native_equivalence` runs both paths on a
small problem at startup and RAISES unless they agree to 1e-4. Everything else
this file does is (a) put our data in the format CIFM expects and (b) leave the
split, the cell set, the gene panel and the metric panel exactly as the other
baselines have them.

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
   written to `<out_dir>/ortholog_mapping.csv` on every run, so the map is
   auditable and can be reused via --ortholog-csv (the helper hits live APIs and
   is NOT cached, so pinning is the only way to guarantee two runs share a map).
3. METRICS. `add_neighborhood_layers(..., n_neighs=16)` then
   `build_pearson_dataframe(..., log1p=True, n_hvg=50)` then
   `write_pearson_outputs` — the shared implementation, so Pearson/Spearman/
   RMSE/AUROC/AP, the HVG-50 set and the marker set are computed identically.
4. OUTPUT CONTRACT. `layers["X_hat"]` = (n_obs, n_vars) float32 on the RAW
   COUNT scale for ALL cells (train and test), same row/var order as loaded.

FIDELITY TO THE PUBLISHED METHOD
--------------------------------
Everything below is sourced from the paper (You et al., "Building Foundation
Models to Characterize Cellular Interactions via Geometric Self-Supervised
Learning on Spatial Genomics", MLGenX 2025 workshop version) or from the released
code, with the location given. Earlier revisions of this docstring carried
several speculative explanations for CIFM's low scores -- receptive-field limits,
panel size, the ortholog map -- which were either unverified or refuted. They
have been deleted rather than corrected; do not reintroduce them.

WHAT WE MATCH
  * PREPROCESSING (Appdx B.1). Raw counts, "coordinates measured in
    micrometers", then "normalize gene counts and conduct log1p-transformation".
    We apply normalize_total(target_sum=1e4)+log1p. NOTE the paper does NOT state
    a target_sum; 1e4 comes from the official test.ipynb, the only concrete
    guidance available.
  * GRAPH RADIUS (Appdx B.1). "the radius threshold r_thres set to 20um for
    Visium-HD, Xenium-V1 and Xenium-Prime and 150um for Visium-Spatial". Our data
    is single-cell resolution, so r=20um (the checkpoint's own
    radius_spatial_graph) is correct. This is why `--coord-scale dataset` matters:
    it applies the published per-assay unit conversion to reach TRUE micrometres
    (median NN 74.72 x 0.194 = 14.50um for batch 15, 10.99 x 1.0 = 10.99um for
    batch 82), both comfortably inside r=20um.
  * EVALUATION GEOMETRY (Appdx B.1, Fig. 2A). Their in-sample evaluation is a
    CONTIGUOUS REGIONAL split: train if x <= x_thres, test if x > x_thres and
    y < y_thres, with x_thres = x_min + 0.6*(x_max-x_min) and
    y_thres = y_min + 0.5*(y_max-y_min), a ~3:1:1 cell ratio. Their test region is
    therefore a contiguous rectangle spanning ~40% x 50% of the slide bbox --
    LARGER than our 25% x 25% held-out rectangles. Contiguous-region in-painting
    is thus the authors' own protocol, not a mismatch we introduced.
  * SINGLE-SHOT INFERENCE. `predict_cells_at_locations` -> `encode_decode` masks
    all query cells at once and never feeds a prediction back. That is what their
    benchmark uses, so `--infill-steps 1` is the DEFAULT. (The autoregressive
    procedure in their Sec. 5 / Fig. 4 is a perturbation-response *simulation*
    utility, not their expression-inference benchmark. `--infill-steps 12` exists
    to test parity with SQUINT's 12-step MaskGIT decoder; it is a DEVIATION.)
  * THE HARD GATE IS THEIR INFERENCE. `encode_decode` ends with
    `expressions_dec[dropouts_dec <= 0.5] = 0`, so `--output gate` is the
    authors' released procedure and is the DEFAULT here. Their loss (Appdx B.3,
    Eq. 15) is a *balanced MSE* on the reconstruction -- equally weighting
    entries where X>0 and where X=0 -- and contains NO dropout term at all, so
    the second head is an implementation detail absent from the paper. Treat
    `--output marginal` (m*p) and `--output magnitude` (m alone) as OUR
    deviations, documented in the config stub and the variant tag.

WHAT WE DELIBERATELY DO NOT COPY -- THE BENCHMARK SETUP IS SQUINT'S/GeST'S
  Appdx B.1 also removes mitochondrial genes, drops "void cells (those without
  detected gene expression)" and keeps only cells annotated "in tissue". We do
  NOT apply any of that. The held-out split, the cell set, the gene panel and the
  metric panel must stay bit-identical to SQUINT / GeST / the kNN floor, so that
  the comparison is between METHODS and not between preprocessing pipelines.
  Faithfulness here means the model is CALLED the way its authors call it; it does
  not extend to re-filtering the benchmark data underneath it.

WHAT WE CANNOT MATCH, AND MUST DISCLOSE
  * SPECIES. Appdx B.1 filters "Human" under Species; the pretraining corpus is
    human-only (Fig. 1A: Visium-HD 18,070 genes / Xenium-Prime 5,091 /
    Visium-Spatial 32,978 / Xenium-V1 1,814; ~100 slides, 23,139,655 cells,
    32,986 genes). Cross-species transfer is never evaluated by the authors, so
    our mouse panel reached through a human ortholog map is outside their tested
    envelope.
  * SMALL PANELS NEED FINETUNING, PER THE AUTHORS. Sec. 3, p.5: CIFM
    "underperforms in the Xenium-V1 samples ... possibly due to the huge
    discrepancies in the gene measurement scale: around 17K genes on average in
    Visium-HD, 300 genes in Xenium-V1, and 5K genes in Xenium-Prime. This can be
    remedied with further finetuning: we further finetune CIFM on Xenium-V1,
    which results in the best correlation across all 38 slides." Our panel is 431
    genes -- the Xenium-V1 regime. No training/finetuning code is released
    (HF repo: checkpoint + test.ipynb only), so zero-shot is the only option the
    release supports. State this with any number.
  * THEIR CORRELATION METRIC IS SPEARMAN. Sec. 3: "correlation assesses whether
    the model ranks gene expression correctly"; Fig. 2B/2C axes read "Spearman
    Correlation r". Their published in-sample Visium-HD figures (read off
    Fig. 2B) are Spearman ~0.212 for CIFM vs ~0.17 for NeighborAvg, and MSE 0.144
    vs 0.205 -- i.e. a ~0.04 margin over a naive neighbourhood mean on their BEST
    platform. Their baseline set is UnifRnd / BernRnd / NeighborAvg, the last
    being "a naive neighborhood average approach that computes the mean
    expressions of the neighboring cells" -- our kNN floor. Report Spearman
    alongside Pearson when comparing to their published numbers.
  * COUNT ROUND TRIP. They score in the normalised log space directly and never
    return to counts. Our shared harness requires raw-count `X_hat`, so
    `to_counts` adds expm1 -> renormalise -> x(neighbour read depth). That step is
    our harness's requirement, not part of their evaluation.
  * PRETRAINING MASK. Appdx B.3: "we remove 5% of the nodes for masking",
    randomly and uniformly. So pretraining masks a scattered 5%, while both their
    evaluation and ours use a contiguous region.

VOID-INVARIANCE: OUR FULL-GRAPH ENCODER *IS* THE PAPER'S `Rmv` (proven)
-----------------------------------------------------------------------
Appdx B.3 encodes with the masked nodes REMOVED, `f_enc(X_unm, C_unm, A(C_unm))`,
while the released `encode_decode` runs the encoder over the FULL graph with the
masked rows set to zero. These are not merely similar -- they are BITWISE
IDENTICAL for the unmasked nodes, verified two ways (2026-08-11):

  * From source. `MLPBiasFree` maps 0 -> 0 exactly (bias-free Linear -> ReLU ->
    LayerNorm(elementwise_affine=False), and LayerNorm(0) = 0/sqrt(0+eps) = 0),
    so `gene_encoder(0) = 0`. In `EGNNLayer.message`,
    `inner_prod = mean(h_i*h_j)` is exactly 0 if either node is void, and
    `innerprod_embedding(0) = 0` multiplies the ENTIRE concatenated message
    including `dist_embedding(dists)` -- so `msg = mlp_msg(0) = 0`. Void nodes
    also stay void through every layer, and `pred(0) = 0`.
  * Numerically. A standalone re-implementation compared full-graph-with-zeroed-
    rows against masked-nodes-removed (12 unmasked + 5 masked, 89 extra edges):
    max|difference| on the unmasked embeddings = 0.000e+00.

Ablating the mechanisms shows what actually carries the invariance: removing the
Eq. 11 intensity gate breaks it (rel. err 1.7), replacing the void-excluding
coordinate denominator (`counts[inner_prod==0] = 0`) with a plain degree breaks it
mildly (2.9e-2), while sum-vs-mean pooling makes NO difference (1.5e-13) -- because
`mlp_upd` opens with a scale-invariant LayerNorm. So the paper's stated mechanism
(3) is not load-bearing and the coordinate denominator, which it does not mention,
is. None of this changes what we do; it means the radius-mode encoder needs no
correction.

WHAT THE PAPER'S LOSS CANNOT BE
-------------------------------
`expressions_dec[dropouts_dec <= 0.5] = 0` is a hard threshold with identically
zero gradient into `mask_cell_dropout`. If Eq. 15 were computed on the GATED
output, that head could never have been trained -- yet it demonstrably is
(AUROC 0.877 for truth>0, against 0.461 for the magnitude head). So the gate is
not inside the objective, Eq. 15's `X_dec` is the UNGATED
`relu(mask_cell_expression(.))`, and Eq. 15 as printed is an incomplete statement
of the real objective: a zero-inflation / BCE term is missing from the paper.
This is also consistent with `forward()` being an empty stub and `proj`
(hidden->1) being referenced nowhere -- the release is a training class stripped
to inference. NOTE neither candidate reproduces their MSE locally (gated 1.7682,
ungated 8.5340, published 0.144), so an unaccounted step still sits between
`X_dec` and their reported number.

A PAPER-FAITHFUL MECHANISM FOR CONTIGUOUS-REGION DEGRADATION
------------------------------------------------------------
Every masked node receives the SAME `mask_embedding` vector `e`. For a cell deep
inside a large contiguous hole, nearly all of its decoder edges therefore carry
`inner_prod = ||e||^2/d`, a positive constant, and only `dists` varies. The
decoder output collapses toward a near-constant profile, and a fixed 0.5 threshold
on a near-constant `p` keeps essentially the SAME gene set for every interior
cell -- the model's marginal prior, learned where ~17k genes are detected per
cell and hence far denser than a 431-gene panel. This is a property of the
released design, not of our code. Testable if ever needed: interior held-out
cells should have near-identical predicted profiles, and the kept-gene fraction
should rise with distance from the region boundary. It does NOT explain the
over-call under a scattered mask on the demo data.

CIFM-SPECIFIC HANDLING (the parts that needed a decision)
---------------------------------------------------------
* INPUT FORMAT. Per the official `test.ipynb`, CIFM consumes
  `normalize_total(target_sum=1e4)` + `log1p` of raw counts, and
  `obsm['spatial']` **in micrometres** (it builds `radius_graph(r=20)`).
  We normalise a working copy only. The two sections are stored in DIFFERENT
  units (measured on the full data: median NN 74.72 for batch 15 vs 10.99 for
  batch 82). `--coord-scale dataset` (the DEFAULT) applies the PUBLISHED
  per-assay conversion, which yields TRUE micrometres — 74.72 x 0.194 = 14.50 um
  and 10.99 x 1.0 = 10.99 um. Both sit well inside r=20 um, so every cell gets a
  handful of neighbours: the single-cell spacing regime CIFM was trained on. It
  does NOT force a target NN distance — that is `--coord-scale auto`, a deviation
  which density-matches the sections and so erases a real difference in cell
  density (it picks 0.1338 for batch 15 vs the published 0.194, 31% off).
  SQUINT/GeST are unaffected either way: their graphs are k-NN (rank-based),
  hence scale-free.
* LEAK-FREE READ DEPTH. CIFM emits a log-normalised profile, NOT counts, so it
  needs a depth to become count-scale. We port `_neighbor_read_depth` verbatim
  from `squint/examples/stage2_decode_pearson.py:415-448`: a held-out cell's
  depth is the mean library size of its k=16 nearest OBSERVED cells in the same
  section; train cells keep their own. It never reads a held-out cell's own
  counts. Using the cell's own depth would silently corrupt RMSE / AUROC / AP
  while leaving Pearson plausible.
  PARITY CONFIRMED (author, 2026-08-11): the paper's SQUINT imputed numbers were
  produced with READ_DEPTH_MODE=neighbor, i.e. the leak-free rule, matching CIFM,
  GeST and the kNN floor. Note this is NOT the script default --
  `submit_stage2_mc_sweep.sh` defaults to READ_DEPTH_MODE="true", which
  `stage2_decode_pearson.py`'s own help calls "leaks the target's depth -- valid
  only for the reconstruction sanity-check". So a future re-run WITHOUT an explicit
  override would silently produce a leaky SQUINT number that is not comparable
  with these baselines. It matters because the harness scores log1p(counts), and a
  per-cell scale factor does not cancel out of Pearson in log space (nor from
  RMSE/AUROC/AP). Always pass READ_DEPTH_MODE=neighbor explicitly, and confirm via
  the echoed "[decode] [read-depth]" log line.
* PREDICTING TRAIN CELLS, LEAK-FREE. The harness needs `X_hat` everywhere
  (train rows feed the train/all splits and the niche aggregation at test
  cells). We predict train cells in chunks with the chunk itself removed from
  the context, mirroring GeST's `cand = cand[cand != ti]`
  (`gest/train.py:193`). Test cells never appear in any context.
* PER-SECTION GRAPHS. Every forward pass is restricted to one section, so no
  edge ever crosses sections — matching our convention everywhere else.
* TWO HEADS -> ONE PREDICTION (`--output`). CIFM's decoder is FACTORISED and
  zero-inflated: `relu(mask_cell_expression)` is a CONDITIONAL magnitude `m`,
  and `sigmoid(mask_cell_dropout)` is P(EXPRESSED) `p` — note the polarity, the
  head's name is misleading. CIFM's own `encode_decode` keeps entries with
  p > 0.5 (`expressions_dec[dropouts_dec <= 0.5] = 0`), and measured on CIFM's
  own demo data `p` scores AUROC 0.877 / AP 0.221 for (truth > 0) against a
  0.024 prevalence, while `m` alone scores AUROC 0.461 — i.e. chance.
  So the sparsity information lives ENTIRELY in `p`:
      magnitude  m          99.8% dense against a 2.4%-dense truth. Because
                            `to_counts()` row-normalises, that splits each
                            cell's depth across ~18.3k entries instead of ~440
                            and dilutes exactly the genes Pearson rewards.
      gate       m*(p>0.5)  CIFM's native inference.
      marginal   m*p        E[expression] under a zero-inflated likelihood.
  DEFAULT IS `gate`: it is verbatim what the released `encode_decode` returns,
  so it is the faithful choice. `marginal` (m*p) is the zero-inflated expectation
  and scored better on CIFM's demo data (harness-space cell-wise Pearson 0.3954
  vs 0.2705), but it is OUR deviation, not the authors'. `magnitude` discards the
  sparsity head entirely and is simply wrong; it exists only to reproduce
  superseded numbers. The choice is recorded in the config stub and the variant
  tag, so runs made under different settings can never be averaged together.
  log1p(1e4-normalised) — on truly-expressed entries it matches truth with
  entrywise Pearson 0.813 and a mean ratio of 1.023 — so no recalibration
  transform is applied or needed. See `cifm_marginal_vs_gate.py`.
* SEEDS. Every requested seed runs a FULL inference pass (no row copying), so
  the per-seed CSV is genuinely produced per seed. But CIFM is a frozen
  checkpoint with deterministic inference (model.eval(), no sampling), so those
  passes return IDENTICAL numbers, and we deliberately do not inject artificial
  per-seed noise to manufacture a spread. Report CIFM as a deterministic
  baseline, like the kNN floor — NOT as N independent replicates. The runner
  prints how many distinct metric rows were actually produced, so this is
  visible in the log.

Usage
-----
  python run_cifm.py --use-default-holdout-regions --seeds 0,1,2,3,4 \
      --cifm-repo <repo>/analysis/benchmarking/cifm   # (the default)

Requires (own venv): torch, torch-geometric (+scatter/sparse/cluster), e3nn,
scanpy, mygene, huggingface_hub, and the `models_cifm/` package from the CIFM
repo on PYTHONPATH (`--cifm-repo`).
"""

from __future__ import annotations

import argparse
import math
import json
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
# The checkpoint lives beside the other benchmarked models
# (analysis/benchmarking/cifm; gitignored, like geneformer / scGPT / uce_model).
DEFAULT_CIFM_REPO = _THIS.parent / "cifm"
DEFAULT_READ_DEPTH_NEIGHS = 16          # == SQUINT's --read-depth-neighs
CIFM_HF_REPO = "ynyou/CIFM"

# ---------------------------------------------------------------------------
# Coordinate units, per section, FROM THE ORIGINAL DATASET PUBLICATIONS.
#
# CIFM's graph is a FIXED-radius graph (r=20, in micrometres), so unlike our own
# k-NN graph it is NOT scale-free: wrong units => an empty or saturated graph and
# meaningless predictions. Our silver files carry the SOURCE coordinates verbatim
# (`extract_utils.py` copies `x`/`y` for STARmap+ and `center_x`/`center_y` for
# MERFISH with no rescaling), so the conversion has to be applied here.
#
#   batch 15 -- STARmap PLUS mouse CNS. Shi et al., Nature 622:552-561 (2023),
#       doi:10.1038/s41586-023-06569-5: imaged "at a voxel size of
#       194 x 194 x 345 nm^3", i.e. xy coordinates are 0.194 um VOXELS.
#       Cross-check on our full data (n=42136): median NN 74.72 voxels
#       x 0.194 = 14.5 um (a sensible cell spacing) and the section spans
#       ~34.5k voxels x 0.194 = 6.7 mm (a sensible mouse CNS section). Taken as
#       micrometres instead, spacing would be 75 um -- implausibly sparse.
#   batch 82 -- MERFISH whole mouse brain. Zhang et al., Nature (2023),
#       doi:10.1038/s41586-023-06808-9. `center_x`/`center_y` are already
#       MICROMETRES (the Vizgen/MERFISH cell-metadata convention). Cross-check
#       on our full data (n=44686): median NN 10.99 um and a ~5.3 x 5.8 mm
#       section -- both already correct, hence factor 1.0.
#
# NOTE these are dataset-specific. For any other dataset either add an entry or
# use --coord-scale auto (density matching), which is a fallback, NOT equivalent:
# auto forces both sections to the SAME cell density, erasing a real difference:
# on the full data it chose 0.1338 for batch 15 (31% below the published 0.194)
# and 0.9097 for batch 82 (9% below 1.0). The two sections genuinely differ --
# 14.5 um vs 11.0 um spacing -- and the published factors preserve that.
DEFAULT_COORD_SCALES = {15: 0.194, 82: 1.0}


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
    live and is not cached, so pointing this at a previous run's
    ``ortholog_mapping.csv`` is the way to guarantee two runs share one map.
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
                f"in this panel (e.g. {missing[:5]}) — it was built for a "
                f"different dataset. Drop the flag to resolve the map freshly.")
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
    # Prefer the LOCAL weights so the compute node needs no internet; fall back
    # to the Hub only if download_cifm.py has not been run.
    src = str(repo) if (repo / "model.safetensors").is_file() else CIFM_HF_REPO
    if src == CIFM_HF_REPO:
        print("  NOTE: model.safetensors not found locally -> fetching from the "
              "Hub (this node needs internet). Run download_cifm.py to avoid it.")
    model = CIFM.from_pretrained(src, args=args_model).to(device)
    model.channel2ensembl_ids_source = torch.load(
        repo / "models_cifm" / "channel2ensembl.pt")
    model.eval()
    print(f"  CIFM loaded (radius_spatial_graph="
          f"{getattr(model, 'radius_spatial_graph', '?')}, device={device})")
    return model


# ---------------------------------------------------------------------------
# Coordinate diagnostic — a unit mismatch must not pass silently
# ---------------------------------------------------------------------------
def resolve_coord_scales(coords: np.ndarray, batch: np.ndarray, radius: float,
                         coord_scale: str, target_nn: float) -> Dict:
    """
    Per-section coordinate scale factors.

    CIFM's graph is a FIXED-radius graph (r=20 in its training units, nominally
    micrometres), so unlike our own k-NN graph it is NOT scale-free: get the unit
    wrong and the graph is either empty (no context reaches the masked cell, so
    predictions are meaningless) or fully connected.

    The two sections are stored in DIFFERENT units — measured on the FULL data,
    median nearest-neighbour distance is 74.72 for batch 15 (voxels) and 10.99
    for batch 82 (already micrometres). ``dataset`` (the DEFAULT) applies the
    PUBLISHED per-assay conversion from DEFAULT_COORD_SCALES, which puts both in
    true micrometres: 74.72 x 0.194 = 14.50 um and 10.99 x 1.0 = 10.99 um. Both
    sit comfortably under CIFM's r=20 um, so each cell gets a handful of
    neighbours — the single-cell spacing regime CIFM was trained on. NOTE this is
    NOT a rescaling to ``target_nn``; that is ``auto``, which is a deviation
    (it density-matches the sections and so erases a real biological difference
    in cell density). ``auto`` chose 0.1338 for batch 15 against the published
    0.194, i.e. 31% off. SQUINT/GeST are unaffected either way: their graphs are
    k-NN (rank-based) hence scale-free.

    Pass a number to apply one global factor instead.
    """
    from sklearn.neighbors import NearestNeighbors

    print(f"\n=== Coordinate scaling (CIFM r={radius}; target median NN "
          f"= {target_nn} when auto; mode={coord_scale}) ===")
    scales: Dict = {}
    for b in np.unique(batch):
        c = coords[batch == b]
        nn = NearestNeighbors(n_neighbors=2).fit(c)
        d, _ = nn.kneighbors(c)
        med = float(np.median(d[:, 1]))
        if coord_scale == "dataset":
            if b not in DEFAULT_COORD_SCALES:
                raise SystemExit(
                    f"--coord-scale dataset: no published unit conversion known "
                    f"for section {b!r}. Add it to DEFAULT_COORD_SCALES (with the "
                    f"citation) or pass --coord-scale auto / a number.")
            f = float(DEFAULT_COORD_SCALES[b])
        elif coord_scale == "auto":
            f = (target_nn / med) if med > 0 else 1.0
        else:
            f = float(coord_scale)
        scales[b] = f
        span = (float(np.ptp(c[:, 0])) * f, float(np.ptp(c[:, 1])) * f)
        cs = c * f
        nb = NearestNeighbors(radius=radius).fit(cs).radius_neighbors(
            cs[: min(500, len(cs))], return_distance=False)
        deg = float(np.mean([len(x) - 1 for x in nb]))
        flag = "" if 1.0 <= deg <= 200.0 else "   <-- CHECK: empty or saturated graph"
        print(f"  section {b}: n={len(c)}  median NN={med:.2f} -> x{f:.4g} "
              f"(NN becomes {med*f:.2f})  span={span[0]:.0f}x{span[1]:.0f}  "
              f"mean deg @r={radius}: {deg:.1f}{flag}")
    print("  (mean degree ~0 => no context reaches the masked cell; "
          "~1000s => everything is a neighbour)")
    return scales


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _as_context(pred_log: np.ndarray) -> np.ndarray:
    """
    Re-normalise a PREDICTED profile so it can be fed back in as context.

    Required for iterative in-painting. The model consumes
    `normalize_total(1e4) + log1p`, but its raw output is NOT on that scale --
    on CIFM's demo data the marginal's log-space row sum is ~15.3k against a
    truth of ~1.36k. Feeding that back unchanged would inject context ~11x too
    large and poison every later step. So: expm1 -> renormalise to 1e4 -> log1p,
    which is exactly the transform the observed context went through.
    """
    lin = np.expm1(np.clip(pred_log, 0.0, None))
    rs = lin.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    return np.log1p(lin / rs * 1e4).astype(np.float32)


def _infill_iterative(model, Xn, xy, tr_idx, te_idx, device, output: str,
                      graph: str, knn_k: int, steps: int, schedule: str,
                      rank: str):
    """
    MaskGIT-style iterative in-painting for CIFM, mirroring SQUINT's decoder.

    WHY THIS EXISTS. SQUINT's stage-2 prior fills the held-out region with
    `steps=12` confidence-scheduled iterations, and committed cells become
    context for the next one (`vqniche/stage2/decode.py`: "large contiguous
    holes are filled from the outside in"). CIFM's released API
    (`predict_cells_at_locations`) is SINGLE-SHOT: every held-out cell is masked
    at once and no prediction is ever fed back, so its reach is strictly one
    4-hop/80 um receptive field. Scoring a 12-iteration method against a
    1-iteration one on a 1.3-1.7 mm hole is OUR asymmetry, not CIFM's, and
    CIFM's paper does multi-cell inference autoregressively. This restores
    parity. `--infill-steps 1` is the released-API behaviour.

    Schedule is `_schedule_ratio` copied from `vqniche/stage2/decode.py:33-38`
    (cosine: the fraction REMAINING masked after `step` is cos(0.5*pi*step/steps)),
    so the two methods commit on the same curve.

    Ranking -- which cells to commit first. SQUINT ranks by the summed log-prob
    of the chosen discrete codes. CIFM has no categorical head, so:
      'confidence' (default, closest to MaskGIT) = mean_g log max(p_g, 1-p_g),
          the model's own certainty in its sparsity calls -- the same head that
          scores AUROC 0.877 on CIFM's demo data.
      'distance'   = ascending distance to the nearest context cell, i.e. fill
          strictly from the boundary inward. Independent of head calibration,
          which matters here because that head only reaches AUROC ~0.57 on our
          ortholog-mapped panel.
    Both are deterministic; no Gumbel noise (SQUINT's noise_anneal trick would
    add per-seed variance CIFM does not otherwise have).
    """
    from scipy.spatial import cKDTree

    n_te = int(te_idx.size)
    G = Xn.shape[1]
    pred = np.zeros((n_te, G), dtype=np.float32)
    committed = np.zeros(n_te, dtype=bool)
    ctx_X, ctx_xy = Xn[tr_idx], xy[tr_idx]

    for step in range(1, steps + 1):
        todo = np.where(~committed)[0]
        if todo.size == 0:
            break
        p_step, conf = _predict_chunk(model, ctx_X, ctx_xy, xy[te_idx[todo]],
                                      device, output, graph=graph, knn_k=knn_k,
                                      return_conf=True)
        pred[todo] = p_step

        if step == steps:
            committed[todo] = True
            break

        # fraction that should REMAIN masked after this step (SQUINT's curve)
        remain = (math.cos(0.5 * math.pi * step / steps) if schedule == "cosine"
                  else max(0.0, 1.0 - step / steps))
        n_reveal = int(todo.size - math.floor(remain * n_te))
        if n_reveal <= 0:
            continue
        n_reveal = min(n_reveal, todo.size)

        if rank == "distance":
            d, _ = cKDTree(ctx_xy).query(xy[te_idx[todo]], k=1)
            order = np.argsort(d, kind="stable")            # nearest first
        else:
            order = np.argsort(-conf, kind="stable")        # most confident first
        take = todo[order[:n_reveal]]
        committed[take] = True
        ctx_X = np.concatenate([ctx_X, _as_context(pred[take])], axis=0)
        ctx_xy = np.concatenate([ctx_xy, xy[te_idx[take]]], axis=0)
        print(f"       step {step:2d}/{steps}: committed {take.size} "
              f"({int(committed.sum())}/{n_te}), context {ctx_X.shape[0]}")
    return pred


def _predict_chunk(model, ctx_X, ctx_xy, q_xy, device, output: str,
                   graph: str = "radius", knn_k: int = 16,
                   return_conf: bool = False):
    """
    Faithful re-implementation of CIFM.predict_cells_at_locations that also
    exposes the continuous dropout head (needed so we can choose NOT to gate).
    ctx_X: (n_ctx, G) normalised+log1p context expression. Returns (n_q, G).
    """
    import torch
    from torch_geometric.nn import knn_graph, radius_graph

    n_ctx, G = ctx_X.shape
    n_q = q_xy.shape[0]
    with torch.no_grad():
        expr = torch.tensor(ctx_X, dtype=torch.float32, device=device)
        expr = torch.cat([expr, torch.zeros(n_q, G, device=device)], dim=0)

        xy = np.concatenate([ctx_xy, q_xy], axis=0)
        coords = torch.tensor(xy, dtype=torch.float32)
        coords = torch.cat([coords, torch.zeros(coords.shape[0], 1)], dim=1).to(device)

        # CIFM natively uses a FIXED-RADIUS graph. `--graph knn` swaps in the
        # same k-NN connectivity SQUINT / GeST / the kNN floor use, so every
        # method sees a receptive field of the same size (k=16) — the
        # apples-to-apples comparison. NOTE this is off-distribution for CIFM
        # (it was pretrained on radius graphs only) and must be disclosed.
        # The coordinate VALUES still matter either way: CIFM's EGNN consumes
        # relative positions, so the published unit conversion is applied in
        # both modes.
        if graph == "knn":
            # k+1: torch_cluster calls knn(x, x, k if loop else k+1), so with
            # loop=True the self-edge eats one slot. k=16 alone would give 15
            # spatial neighbours + self, not the harness's 16 + self.
            #
            # !! KNOWN DEFECT, kNN MODE ONLY (verified 2026-08-11 against the
            # released source). A k-NN graph has a FIXED neighbour budget, so the
            # zero-expression query cells concatenated below DISPLACE real
            # observed neighbours from each context cell's k slots -- and then
            # contribute nothing, because CIFM's encoder is exactly
            # void-invariant (proven below). Context cells near the held-out
            # region are therefore genuinely STARVED of context, worse the denser
            # the masked block. Under the paper's own formulation
            # (`A(C_unm)`, Appdx B.1/B.3) every context cell would get k OBSERVED
            # neighbours. RADIUS MODE IS UNAFFECTED: a radius graph has no budget,
            # so adding void nodes adds only edges that carry exactly zero.
            # => `--graph radius` (the default) is the faithful arm. Disclose this
            # whenever a `--graph knn` number is reported.
            edge_index = knn_graph(coords, k=knn_k + 1, loop=True)
        else:
            edge_index = radius_graph(coords, r=model.radius_spatial_graph,
                                      max_num_neighbors=10000, loop=True)
        mapping = torch.arange(n_ctx, n_ctx + n_q, device=device)

        emb = model.encode(expr, coords, edge_index)
        emb[mapping] = model.mask_embedding(
            torch.zeros(1, dtype=torch.int64, device=device))
        emb_dec = model.mask_cell_decoder(emb, coords, edge_index)[0][mapping]

        # Two heads. `m` is a CONDITIONAL magnitude; `p` is P(EXPRESSED) —
        # confirmed both by CIFM's own gate direction and empirically
        # (AUROC 0.877 for truth>0 vs 0.461 for `m`). See the module docstring.
        m = model.relu(model.mask_cell_expression(emb_dec))
        p = None
        if output == "magnitude":
            pred = m                                    # pre-2026-08 default
        else:
            p = model.sigmoid(model.mask_cell_dropout(emb_dec))
            if output == "gate":                        # CIFM's native inference
                pred = m.clone()
                pred[p <= 0.5] = 0.0
            elif output == "marginal":                  # E[expr], the default
                pred = m * p
            else:
                raise ValueError(f"unknown --output {output!r}")
        out = pred.detach().cpu().numpy().astype(np.float32)
        if not return_conf:
            return out
        # Per-cell confidence for the iterative committer: mean_g log max(p,1-p).
        # With --output magnitude the sparsity head is unused, so fall back to a
        # constant (the caller's 'distance' ranking is the meaningful one there).
        if p is None:
            conf = np.zeros(out.shape[0], dtype=np.float32)
        else:
            import torch as _t
            conf = (_t.log(_t.maximum(p, 1.0 - p).clamp_min(1e-9))
                    .mean(dim=1).detach().cpu().numpy().astype(np.float32))
        return out, conf


def verify_native_equivalence(model, Xn, xy, tr_idx, device, tol: float = 1e-4,
                             n_ctx: int = 300, n_q: int = 8,
                             run_output: str = "gate",
                             run_graph: str = "radius") -> bool:
    """
    Prove our forward path equals the authors' own `predict_cells_at_locations`.

    `_predict_chunk` RE-IMPLEMENTS CIFM's inference rather than calling it -- we
    need the two heads exposed (for --output) and per-section leak-free context,
    neither of which their API allows. A re-implementation can silently diverge,
    and nothing else in this file would catch it: the tutorial reproduction
    exercises `embed()`, not the decoder.

    So run both on the same small problem and require agreement. Uses
    `--output gate`, since that is exactly what `encode_decode` returns
    (`expressions_dec[dropouts_dec <= 0.5] = 0`). Raises on mismatch -- a
    divergence here invalidates every number the run would produce.
    """
    import anndata as _ad
    import torch
    from scipy.sparse import csr_matrix

    if tr_idx.size < n_ctx + n_q:
        print(f"  (section has {tr_idx.size} train cells, need "
              f"{n_ctx + n_q}; native-equivalence check deferred to a later "
              f"section)")
        return False
    ctx = tr_idx[:n_ctx]
    qry = tr_idx[n_ctx:n_ctx + n_q]

    # their API: an AnnData of context + an array of query locations
    ad_ctx = _ad.AnnData(X=csr_matrix(Xn[ctx].astype(np.float32)))
    ad_ctx.obsm["spatial"] = xy[ctx].astype(np.float32)
    with torch.no_grad():
        native = model.predict_cells_at_locations(
            ad_ctx, xy[qry].astype(np.float32)).cpu().numpy()
    ours = _predict_chunk(model, Xn[ctx], xy[ctx], xy[qry], device,
                          output="gate", graph="radius")

    if native.shape != ours.shape:
        raise RuntimeError(
            f"native-equivalence check: shape {native.shape} vs {ours.shape}")
    d = float(np.abs(native - ours).max())
    scale = float(np.abs(native).max())
    print(f"  native-equivalence check: max|diff| = {d:.3g} "
          f"(value scale {scale:.3g}, nonzero frac "
          f"{float((native > 0).mean()):.4f})")
    nz = float((native > 0).mean())
    if nz == 0.0:
        raise RuntimeError(
            "native-equivalence check is VACUOUS: the reference prediction is "
            "entirely zero, so 'max|diff|=0' proves nothing. The gate zeroed "
            "every probe entry, which usually means the radius graph is empty -- "
            "check --coord-scale and the printed mean degree before trusting "
            "anything downstream.")
    if not np.isfinite(d) or d > tol:
        raise RuntimeError(
            f"_predict_chunk DIVERGES from CIFM.predict_cells_at_locations: "
            f"max|diff|={d:.6g} > tol={tol}. Our re-implementation of "
            f"encode_decode is not faithful -- fix before trusting any number.")
    print(f"  -> verified IDENTICAL to the released inference for "
          f"output=gate, graph=radius (nonzero frac {nz:.4f}).")
    if run_output != "gate" or run_graph != "radius":
        print(f"     CAVEAT: this run uses output={run_output}, graph={run_graph}. "
              f"The released API only exposes gate+radius, so those settings "
              f"cannot be checked against it. Entries with p<=0.5 (which "
              f"--output magnitude/marginal do use) and the k-NN edge "
              f"construction are therefore NOT covered by this check.")


def predict_all_cells(
        model, adata: ad.AnnData, batch_key: str, device: str,
        output: str, train_chunk: int, coord_scales: Dict,
        train_chunk_order: str = "random",
        infill_steps: int = 1, infill_schedule: str = "cosine",
        infill_rank: str = "confidence",
        graph: str = "radius", knn_k: int = 16, seed: int = 0,
        verify_native: bool = True,
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

    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2].copy()
    batch = adata.obs[batch_key].to_numpy()
    for _b, _f in coord_scales.items():          # per-section scaling
        xy[batch == _b] *= _f
    is_train = (adata.obs["data_split"].to_numpy() == "train")

    G = Xn.shape[1]
    out = np.zeros((adata.n_obs, G), dtype=np.float32)

    _verified = False
    for b in np.unique(batch):
        m = (batch == b)
        tr_idx = np.where(m & is_train)[0]
        te_idx = np.where(m & ~is_train)[0]
        print(f"\n  -- section {b}: {tr_idx.size} train (context), "
              f"{te_idx.size} held out --")
        if verify_native and not _verified:
            _verified = verify_native_equivalence(
                model, Xn, xy, tr_idx, device,
                run_output=output, run_graph=graph)

        # ---- held-out cells: full train context, single pass -------------
        if te_idx.size:
            t0 = time.time()
            if infill_steps > 1:
                out[te_idx] = _infill_iterative(
                    model, Xn, xy, tr_idx, te_idx, device, output,
                    graph=graph, knn_k=knn_k, steps=infill_steps,
                    schedule=infill_schedule, rank=infill_rank)
            else:
                out[te_idx] = _predict_chunk(model, Xn[tr_idx], xy[tr_idx],
                                             xy[te_idx], device, output,
                                             graph=graph, knn_k=knn_k)
            print(f"     test  : {te_idx.size} cells in {time.time()-t0:.1f}s")

        # ---- train cells: chunked, chunk excluded from its own context ----
        # At least TWO chunks: with one chunk the chunk IS the whole train set,
        # so its complement is empty and the train predictions would silently be
        # left at zero (this bit the smoke run, where n_train < train_chunk).
        if tr_idx.size >= 2:
            n_chunks = max(2, int(np.ceil(tr_idx.size / max(1, train_chunk))))
            # `tr_idx` is in AnnData row order, and spatial data is very often
            # stored in FOV/tile order — so `array_split` alone would carve out
            # SPATIALLY CONTIGUOUS chunks, making every train cell a hole-filling
            # case too. That destroys the train split's value as a control: it is
            # supposed to be the easy, context-rich condition against which the
            # held-out REGION is compared, and it is also the only setting close
            # to CIFM's pretraining regime (a scattered mask). We therefore
            # permute before splitting, which yields spatially scattered chunks.
            #
            # This does NOT manufacture per-seed variance: the permutation is
            # drawn from a seed-derived generator, so a given seed always yields
            # the same partition, and CIFM's forward pass stays deterministic
            # (frozen weights, model.eval(), no sampling). Two seeds do now
            # differ in WHICH cells share a chunk, which is genuine protocol
            # variation rather than injected noise — but CIFM should still be
            # reported as a deterministic baseline, since the effect is tiny
            # next to the between-method gaps. `--train-chunk-order index`
            # restores the old contiguous behaviour for diagnosis.
            if train_chunk_order == "random":
                perm = np.random.default_rng(seed).permutation(tr_idx.size)
                chunks = np.array_split(tr_idx[perm], n_chunks)
            else:
                chunks = np.array_split(tr_idx, n_chunks)
        else:
            # A lone train cell has an empty complement, so it cannot be
            # predicted leak-free; leaving it at zero would silently enter the
            # metrics as a NaN-dropped row while n_cells still counted it.
            if tr_idx.size == 1:
                print(f"     WARNING: section {b} has 1 train cell; it cannot be "
                      f"predicted leak-free and is left unpredicted")
            n_chunks, chunks = (0, [])
        t0 = time.time()
        for q in chunks:
            if q.size == 0:
                continue
            ctx = np.setdiff1d(tr_idx, q, assume_unique=False)
            if ctx.size == 0:
                print("     WARNING: empty context for a train chunk; skipping")
                continue
            out[q] = _predict_chunk(model, Xn[ctx], xy[ctx], xy[q],
                                    device, output, graph=graph, knn_k=knn_k)
        print(f"     train : {tr_idx.size} cells in {n_chunks} chunk(s), "
              f"{time.time()-t0:.1f}s")
    if verify_native and not _verified:
        print("\n  *** WARNING: the native-equivalence check NEVER RAN -- no "
              "section had enough train cells. The forward path is UNVERIFIED "
              "for this run. ***")
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
    p.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO,
                   help=f"Local copy of the ynyou/CIFM repo (must contain "
                        f"models_cifm/{{cifm.py,args.pt,channel2ensembl.pt}} and, "
                        f"for offline runs, model.safetensors). Default: "
                        f"{DEFAULT_CIFM_REPO} — populate it with download_cifm.py.")
    p.add_argument("--species", default="mouse",
                   help="Source species of adata.var_names, for the ortholog map.")
    p.add_argument("--ortholog-csv", type=Path, default=None,
                   help="Optional: reuse a previous run's "
                        "<out_dir>/ortholog_mapping.csv instead of re-querying "
                        "mygene/Ensembl, so repeated runs share an identical map "
                        "(the helper hits live APIs and is not cached). Not "
                        "required — every run writes its own map.")
    p.add_argument("--infill-steps", type=int, default=1,
                   help="Iterative in-painting steps for the HELD-OUT region. "
                        "1 (default) = CIFM's released single-shot "
                        "predict_cells_at_locations. 12 matches SQUINT's "
                        "stage-2 decoder (vqniche/stage2 DecodeConfig.steps=12), "
                        "which fills contiguous holes outside-in and feeds "
                        "committed cells back as context — scoring 12 "
                        "iterations against 1 is OUR asymmetry, not CIFM's. "
                        "Report both.")
    p.add_argument("--infill-schedule", default="cosine",
                   choices=["cosine", "linear"],
                   help="Commit schedule, copied from vqniche/stage2/decode.py.")
    p.add_argument("--infill-rank", default="confidence",
                   choices=["confidence", "distance"],
                   help="Which masked cells to commit first. 'confidence' = "
                        "mean_g log max(p,1-p) from CIFM's sparsity head "
                        "(closest to MaskGIT). 'distance' = nearest-to-context "
                        "first, i.e. strictly boundary-inward; independent of "
                        "that head's calibration, which is poor on our panel "
                        "(AUROC ~0.57 vs 0.877 on CIFM's own data).")
    p.add_argument("--train-chunk-order", default="random",
                   choices=["random", "index"],
                   help="How train cells are partitioned into chunks (each "
                        "chunk is excluded from its own context). 'random' "
                        "(DEFAULT) permutes with a seed-derived RNG first, so "
                        "chunks are spatially SCATTERED — the train split is "
                        "then a genuine context-rich control and the closest "
                        "match to CIFM's pretraining mask. 'index' splits in "
                        "AnnData row order, which on FOV/tile-ordered data "
                        "carves out contiguous blocks and turns train cells "
                        "into hole-filling cases as well; kept for diagnosis.")
    p.add_argument("--output", default="gate",
                   choices=["gate", "marginal", "magnitude"],
                   help="How to collapse CIFM's two decoder heads into one "
                        "prediction. 'gate' (DEFAULT) = m*(p>0.5), which is "
                        "verbatim what the released encode_decode does "
                        "(`expressions_dec[dropouts_dec<=0.5]=0`) and therefore "
                        "the FAITHFUL choice. 'marginal' = m*p, the zero-inflated "
                        "expectation: a DEVIATION, though it measured better on "
                        "CIFM's own demo data (harness-space cell-wise Pearson "
                        "0.3954 vs 0.2705 for the gate). 'magnitude' = m alone, "
                        "which discards the sparsity head entirely and is simply "
                        "wrong; kept only to reproduce superseded numbers. The "
                        "paper's loss (Appdx B.3 Eq. 15) is a balanced MSE with "
                        "no dropout term, so the second head is undocumented "
                        "there -- the released code is the only authority, and it "
                        "gates.")
    p.add_argument("--graph", default="radius", choices=["radius", "knn"],
                   help="'radius' (default) = CIFM's native fixed-radius graph. "
                        "'knn' swaps in the same k-NN connectivity SQUINT/GeST "
                        "use, matching every method's receptive field; "
                        "off-distribution for CIFM, so disclose it.")
    p.add_argument("--knn-k", type=int, default=16,
                   help="k for --graph knn (16 = the graph SQUINT/GeST use).")
    p.add_argument("--read-depth-neighs", type=int, default=DEFAULT_READ_DEPTH_NEIGHS)
    p.add_argument("--squint-examples", type=Path, default=None,
                   help="Path to squint/examples, so we can import SQUINT's own "
                        "_neighbor_read_depth instead of a copy (recommended).")
    p.add_argument("--train-chunk", type=int, default=4096,
                   help="Train cells predicted per forward pass (each chunk is "
                        "removed from its own context to stay leak-free).")
    p.add_argument("--coord-scale", default="dataset",
                   help="'dataset' (default) applies the PUBLISHED per-section "
                        "unit conversion from DEFAULT_COORD_SCALES (STARmap+ "
                        "0.194 um/voxel, Shi et al. 2023; MERFISH already um, "
                        "Zhang et al. 2023). 'auto' instead density-matches each "
                        "section to --coord-target-nn (fallback for datasets with "
                        "no known conversion). Or a number for one global factor.")
    p.add_argument("--coord-target-nn", type=float, default=10.0,
                   help="Target median NN distance in CIFM's units when "
                        "--coord-scale auto (10 um is typical cell spacing).")
    args = p.parse_args(argv)

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip() != ""]

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode the settings that change the NUMBERS into the variant tag, so
    # `plot_imputation_benchmark.py` can never average runs made under different
    # graphs or head-collapses into one bar. Only done when the tag is the
    # default — an explicit --variant-tag is respected verbatim.
    if args.variant_tag == DEFAULT_VARIANT_TAG:
        args.variant_tag = f"{DEFAULT_VARIANT_TAG}+{args.graph}"
        if args.output != "gate":
            args.variant_tag += f"+out-{args.output}"
        if args.train_chunk_order != "random":
            args.variant_tag += f"+tco-{args.train_chunk_order}"
        if args.infill_steps > 1:
            args.variant_tag += f"+infill{args.infill_steps}-{args.infill_rank}"
            if args.infill_schedule != "cosine":
                args.variant_tag += f"-{args.infill_schedule}"
        if args.graph == "knn" and args.knn_k != 16:
            args.variant_tag += f"+k{args.knn_k}"
        if args.train_chunk != 4096:
            args.variant_tag += f"+tc{args.train_chunk}"
        if args.coord_scale != "dataset":
            args.variant_tag += f"+cs-{args.coord_scale}"
        print(f"Variant : {args.variant_tag}  (auto-derived from --graph/"
              f"--output/--coord-scale)")

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = args.artifacts_root / args.dataset_tag / args.variant_tag / ts
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {args.out_dir}\nSeeds   : {seeds}\nDevice  : {device}")

    print("\n=== Loading silver ===")
    adata = load_silver_concat(Path(args.silver_dir), batch_key=args.batch_key)
    print(f"AnnData: n_obs={adata.n_obs}, n_vars={adata.n_vars}")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype("category")
    # Scales are measured on the FULL data: --smoke thins the section ~20x,
    # which inflates nearest-neighbour distances ~sqrt(20)x and would otherwise
    # produce wrong factors.
    _full_xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)[:, :2]
    _full_batch = adata.obs[args.batch_key].to_numpy()

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
    channel2ensembl_target, map_stats = build_channel_map(
        adata, species=args.species,
        out_csv=args.out_dir / "ortholog_mapping.csv",
        ortholog_csv=args.ortholog_csv)
    if map_stats["n_mapped"] == 0:
        raise SystemExit(
            "No gene mapped to a human Ensembl ID — CIFM would be all-zero. "
            "Check --species and that mygene / Ensembl REST are reachable.")

    print("\n=== Loading CIFM ===")
    model = load_cifm(args.cifm_repo, device)
    # channel_matching only PRINTS its match count and returns None, so a total
    # failure (e.g. version-suffixed Ensembl IDs, which its exact `in` test cannot
    # match) would leave every weight zero -> m=0, p=sigmoid(0)=0.5 -> an all-zero
    # X_hat that no shape or finiteness check would catch. Re-derive the count.
    _src_ids = {e for ents in model.channel2ensembl_ids_source
                for e in (ents if isinstance(ents, (list, tuple)) else [ents])}
    _n_match = sum(1 for t in channel2ensembl_target
                   if t and any(e in _src_ids for e in t))
    print(f"  vocabulary match: {_n_match}/{len(channel2ensembl_target)} panel "
          f"genes found in CIFM's channel vocabulary")
    if _n_match == 0:
        raise SystemExit(
            "channel_matching would match ZERO channels: every weight stays "
            "zero and the run would emit an all-zero X_hat. Check the ortholog "
            "IDs against channel2ensembl.pt (version suffixes such as "
            "ENSG00000141510.15 will not match its exact-equality test).")
    model.channel_matching(channel2ensembl_target, model.channel2ensembl_ids_source)

    coord_scales = resolve_coord_scales(
        _full_xy, _full_batch,
        radius=float(getattr(model, "radius_spatial_graph", 20.0)),
        coord_scale=args.coord_scale, target_nn=args.coord_target_nn)

    # Read depth is seed-independent (coords + observed library sizes only),
    # so resolve it once and reuse it for every seed.
    print("\n=== Leak-free read depth (k=%d, observed neighbours only) ==="
          % args.read_depth_neighs)
    L_all = np.asarray(adata.X.sum(axis=1), dtype=np.float32).ravel()
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

    per_seed_frames = []
    for s_idx, seed in enumerate(seeds):
        print("\n" + "=" * 78 +
              f"\nSEED {seed}  ({s_idx + 1}/{len(seeds)})\n" + "=" * 78)
        adata_s = adata.copy()

        pred_log = predict_all_cells(
            model, adata_s, batch_key=args.batch_key, device=device,
            output=args.output, train_chunk=args.train_chunk,
            train_chunk_order=args.train_chunk_order,
            infill_steps=args.infill_steps,
            infill_schedule=args.infill_schedule,
            infill_rank=args.infill_rank,
            coord_scales=coord_scales, graph=args.graph, knn_k=args.knn_k,
            seed=seed)

        X_hat = to_counts(pred_log, depth)
        if not np.isfinite(X_hat).all():
            n_bad = int((~np.isfinite(X_hat)).sum())
            raise RuntimeError(
                f"X_hat contains {n_bad} non-finite values. float32 expm1 "
                f"overflows above ~88.7, so an extreme magnitude-head output "
                f"turns the row sum to inf and the row to NaN. Do not report "
                f"these numbers.")
        _zero_rows = int((X_hat.sum(axis=1) == 0).sum())
        if _zero_rows:
            print(f"  NOTE {_zero_rows} all-zero X_hat rows; the harness drops "
                  f"them as NaN while still reporting the full n_cells")
        if X_hat.shape != (adata_s.n_obs, adata_s.n_vars):
            raise RuntimeError(f"X_hat shape {X_hat.shape} != "
                               f"({adata_s.n_obs}, {adata_s.n_vars})")
        adata_s.layers["X_hat"] = X_hat

        add_neighborhood_layers(adata_s, batch_key=args.batch_key,
                                n_neighs=args.nbr_neighs)
        per_seed_frames.append(
            build_pearson_dataframe(adata_s, seed=seed, log1p=True, n_hvg=50))
        write_predicted_adata(adata_s, args.out_dir, seed)
        del adata_s

    per_seed = pd.concat(per_seed_frames, ignore_index=True)
    print("\n=== Writing outputs ===")
    _nuniq = per_seed.drop(columns=["seed"]).drop_duplicates().shape[0]
    _nrows = per_seed.shape[0] // max(1, len(seeds))
    print(
        f"\nSeeds run: {len(seeds)}. CIFM's forward pass is deterministic "
        f"(frozen weights, model.eval(), no sampling), so TEST-cell predictions "
        f"do not vary with the seed. "
        + ("Train-cell predictions DO vary: --train-chunk-order random "
           "permutes the chunk partition per seed, so train/all rows differ "
           "across seeds by protocol, not by model noise."
           if args.train_chunk_order == "random" else
           "--train-chunk-order index makes the train partition seed-independent "
           "too, so all rows are identical across seeds.")
        + f" Distinct metric rows: {_nuniq}/{_nrows}."
        + " Report CIFM as a deterministic baseline (like the kNN floor), NOT as"
        + f" {len(seeds)} independent replicates.")
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
            "graph": args.graph,
            "knn_k": args.knn_k,
            "output": args.output,
            "train_chunk_order": args.train_chunk_order,
            "infill_steps": int(args.infill_steps),
            "infill_schedule": args.infill_schedule,
            "infill_rank": args.infill_rank,
            "read_depth_mode": "neighbor(observed only)",
            "read_depth_neighs": args.read_depth_neighs,
            "train_chunk": args.train_chunk,
            "coord_scale": args.coord_scale,
            "coord_target_nn": args.coord_target_nn,
            "per_section_scales": {str(k): float(v) for k, v in coord_scales.items()},
            "deterministic_seed_replicates": len(seeds) > 1,
            "holdout": "DEFAULT_HOLDOUT_REGIONS (identical to SQUINT/GeST)",
            "nbr_neighs": args.nbr_neighs,
        }, indent=2) + "\n")
    print(f"\nDone -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
