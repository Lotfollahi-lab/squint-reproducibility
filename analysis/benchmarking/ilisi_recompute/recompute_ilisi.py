#!/usr/bin/env python
"""
recompute_ilisi.py: corrected global iLISI for Table 1, Table 3 and Table S1.
=============================================================================

WHY
Every published iLISI (SQUINT and every benchmark runner) comes from
squint/examples/compute_inference_metrics.py compute_ilisi (l.528-549), and that
helper has two faults:

  1. TIE-BREAKING. NNDescent is built on the rows in FILE order. predicted_adata
     files are section concatenations, so row order is batch, and SQUINT's
     quantized embeddings (cell_emb / neighborhood_emb, a codebook lookup) hold
     large groups of exactly identical rows. Which tied cells become neighbours
     can then follow row order, i.e. section, and understate iLISI. How much of
     this the paper's estimator (NNDescent) shows on each dataset is measured
     here (file-order replay vs permuted) and stated in the merge report.
       affected     every quantized SQUINT value scored in file order: Table 1
                    Mouse Brain (86,822 cells) and Eczema (53,655), and every
                    quantized Table 3 / Table S1 column (all Mouse Brain).
       CONTROL      NSCLC SQUINT (199,672 cells > ilisi_max_cells = 100,000) was
                    scored on default_rng(0).choice(n, 100000), which is already
                    a shuffled order. The recompute must reproduce its published
                    0.948 (niche) / 0.967 (cell); if it moves by more than one
                    seed s.d., stop and investigate before using any number.
       unaffected   continuous embeddings (baselines, *_latent, s57_v33) have no
                    exact ties.
  2. SILENT FALLBACK. When `from scib_metrics import ilisi_knn` fails in a
     method's environment, compute_ilisi returns the MEAN of an UNSCALED,
     uniformly weighted inverse Simpson index over all k neighbours, self
     included (range [1, n_batches]), instead of scib's SCALED MEDIAN (range
     [0, 1]). Four Table 1 cells are that other statistic: GraphST 1.448 (Mouse
     Brain) and 2.041 (Eczema), scGPT 1.562 and scGPT-spatial 1.567 (Mouse
     Brain). Any published value >= 1 is a fallback value. Rescaling them is not
     a fix (mean vs median, uniform vs Gaussian kernel, self included): they are
     recomputed here.

WHAT IS COMPUTED: the paper's protocol, changed only where it was wrong
  graph   pynndescent NNDescent on the embedding, euclidean, k = 90, query point
          included, exactly as compute_ilisi and compute_label_metric.paper_graph
          build it (plus an explicit random_state, so runs are reproducible).
  score   scib_metrics.ilisi_knn with its defaults: perplexity floor(90/3) = 30,
          scaled, i.e. (median per-cell LISI - 1) / (n_batches - 1).
  cells   all cells, except SQUINT runs with more than 100,000 cells, which use
          the same default_rng(0).choice(n, 100000, replace=False) subset as
          compute_inference_metrics (run_squint.py passes no --seed, so seed 0 for
          every training seed). Baselines were never capped
          (run_pca_leiden._compute_batch_integration), so they are not capped here.
  batch   obs['adata_batch_id'] as str, every method.
  FIX 1   rows are randomly permuted before the neighbour search, repeated over
          --n-perm permutation seeds (default 3; mean and s.d. kept).
  FIX 2   scib_metrics ONLY. If it cannot be imported the script exits; it never
          falls back. Every value is asserted to be a scalar in [0, 1].
Per task it also records what is needed to trust the number:
  * a file-order REPLAY (the published protocol, unpermuted), which should
    reproduce the published value up to NNDescent noise; the gap between replay
    and permuted IS the tie artefact as the paper's estimator sees it. The merge
    report states explicitly whether that gap exists (fault 1 reproduced with
    NNDescent) or not;
  * the fallback statistic on that replay graph, which must reproduce the four
    >= 1 cells and so proves which embedding produced them;
  * for SQUINT runs, the MMD recomputed with the producer's own call
    (compute_inference_metrics.compute_mmd_comparable, rng default_rng(0)). MMD
    is deterministic given the embedding, row order and batch labels, so a match
    to the per-run MMD proves that the re-scored file is the one behind the
    published row;
  * tie diagnostics: fraction of distinct rows, largest tie group, fraction of
    cells whose tie group is larger than k (their whole neighbourhood is ties),
    and the number of unfilled (-1) NNDescent neighbours;
  * sha1 of the embedding, which shows whether "seed-invariant" methods really
    saved identical embeddings for every seed (GraphST was never given its seed);
  * with --knn exact|both, an exact kNN with the query point in column 0 and
    every distance tie broken INDEPENDENTLY PER QUERY ROW, uniformly at random
    (seeded). It is a robustness check reported next to NNDescent; the primary
    value stays NNDescent unless merge --final-estimator exact is passed.

HOW: one script, six subcommands
  manifest   every per-seed row behind Table 1 (3 datasets x niche and cell-type
             blocks x methods x 5 seeds, plus SQUINT *_latent controls), Table 3
             and Table S1 (Mouse Brain variants x 5 seeds x niche and cell), found
             with the same run discovery, emb keys and batch key that produced the
             published numbers (the plot scripts' and ablation report's own
             functions are imported and reused). The existing per-seed NMI, ARI,
             MMD, RT and published iLISI are attached from the same files, and
             the five-seed means of ALL of them are checked against the printed
             tables. Writes manifest.csv + manifest_coverage.txt; --dry-run
             writes nothing (but still checks that --out-dir may be written).
  plan       the LSF job list (one job per (dataset, method or variant)).
             --sanity overrides the manifest's choice for the continuous sanity
             rows without rebuilding it.
  compute    one job: one result csv per task, skipped when present (resumable);
             --force recomputes, and after a successful recompute moves the old
             csv to results/superseded/.
  merge      per-seed csv, five-seed summary with Welch's t-tests against the
             reference row, Table 1 drop-in csvs for compute_table1_significance.py,
             tidy per-seed frames for the Fig S3 b,d panels, and a diff report of
             every paper-visible iLISI (old -> new) including the numbers quoted
             in the text and in the author response. Every merge writes a NEW
             folder merged/<UTC timestamp>/.
  figures    re-renders the Table 1 benchmark panels (Fig S3 b,d) from a merged
             folder's tidy frames with the plot modules' own make_figure, into a
             new <merged>/fig_panels/.
  selftest   (or --selftest) synthetic data with heavy exact ties stored batch by
             batch, scored against an oracle built by sampling random
             tie-breaking neighbourhoods directly and scoring them with the same
             scib call: (1) the exact kNN puts self in column 0 and breaks ties
             independently per row; (2) exact kNN and NNDescent on permuted rows
             recover the oracle, for a balanced and an imbalanced (30/70)
             composition; (3) the mechanism of fault 1 (ties broken by row index
             on batch-sorted rows) understates iLISI; (4) scib's scaled median
             differs from the fallback mean. The NNDescent file-order gap is
             printed next to the mechanism gap but not asserted: whether the
             paper's estimator inherits the row-order bias is measured on the real
             data by the merge report. Needs only numpy, scib_metrics and
             pynndescent (scikit-learn optional). Exit code 1 if a check fails.

Writes only under --out-dir (default <artifacts>/ilisi_recompute, which must be a
subfolder of the artifacts root and either new or created by this script). No
existing file is ever overwritten, and no bytecode is written next to the
imported repo modules (sys.dont_write_bytecode).

USAGE (cluster; submit_recompute_ilisi.sh wraps the LSF part). Heavy steps
(selftest, compute) run inside LSF jobs, never on the submission host:
  source /nfs/team361/sb75/.venvs/squint/bin/activate
  cd /nfs/team361/sb75/squint-reproducibility
  S=analysis/benchmarking/ilisi_recompute/recompute_ilisi.py
  W=analysis/benchmarking/ilisi_recompute/submit_recompute_ilisi.sh
  SELFTEST=1 bash $W                    # selftest as an interactive LSF job
  DRY_RUN=1 bash $W                     # coverage report + job plan, nothing written
  ONLY=chl59.squint bash $W             # NSCLC control first
  bash $W                               # everything else
  python $S merge                       # light: csv reads only
  python $S figures                     # Fig S3 b,d panels from the newest merge
  # one task by hand, inside LSF:
  bsub -I -q normal -n 4 -M 32000 -R 'select[mem>32000] rusage[mem=32000]' \\
    "JAX_PLATFORMS=cpu NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=4 python $S compute --job <task_id>"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# The producer modules imported below live in the repo (and in squint/examples);
# importing them must not write __pycache__/ next to them: outputs go only under
# --out-dir.
sys.dont_write_bytecode = True

try:                  # compute, merge and selftest need numpy; --help, plan and
    import numpy as np  # the printed-table constants do not, so they work anywhere
except ImportError:     # pragma: no cover
    np = None


# =============================================================================
# Protocol constants (compute_inference_metrics.py defaults, as run_squint.py
# and run_pca_leiden.py call it)
# =============================================================================
DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"
DEFAULT_OUT_NAME = "ilisi_recompute"
SCRIPT_REL = "analysis/benchmarking/ilisi_recompute/recompute_ilisi.py"
REPO = Path(__file__).resolve().parents[3]

K = 90                    # --ilisi-n-neighbors
ILISI_MAX_CELLS = 100000  # --ilisi-max-cells (SQUINT path only)
SUBSAMPLE_SEED = 0        # --seed; run_squint.py never passes one
BATCH_KEY = "adata_batch_id"
SEEDS = (0, 1, 2, 3, 4)
DEFAULT_N_PERM = 3
PERM_SEED_STRIDE = 100    # permutation seed j of training seed s is s*100 + j
EXACT_CHUNK = 256
SCIB_VERSION_USED = "0.5.6"   # what the CiLISI runs (Table S3) recorded
OUT_MARKER = ".ilisi_recompute_dir"
PER_SEED_FILES = ("per_seed_niche_identification.csv",
                  "per_seed_batch_integration.csv")
EMB_KEYS = {"niche": "neighborhood_emb", "cell": "cell_emb"}
LATENT_KEYS = {"niche": "neighborhood_latent", "cell": "cell_latent"}


@dataclass(frozen=True)
class Dataset:
    tag: str             # artifacts/<tag>
    short: str           # suffix of the paper's tables/*_benchmark_<short>.csv
    label: str
    squint_variant: str  # sweep dir is <squint_variant>__multiseed
    niche_label: str     # label_key the Table 1 niche NMI/ARI used
    cell_label: str      # label_key the Table 1 cell-type NMI/ARI used
    n_obs: int
    n_batches: int


_HUMAN_SQUINT = ("dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16"
                 "+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc"
                 "+diversity-w10+filmscale+crossmnn-wt10-k1+")
DATASETS = (
    Dataset("mmb0-1b_smb1-1b_1p", "mmb0-1b_smb1", "Mouse Brain",
            "s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p",
            "niche", "cell_type", 86822, 2),
    Dataset("chl59-2b_1p", "chl59", "CosMx NSCLC", _HUMAN_SQUINT + "chl59-2b_1p",
            "niche", "cell_type", 199672, 2),
    Dataset("xhs1000-3b_1p", "xhs1000", "Xenium Eczema",
            _HUMAN_SQUINT + "xhs1000-3b_1p",
            "niche_type", "new_annotation", 53655, 3),
)
DS_BY_TAG = {d.tag: d for d in DATASETS}
ABLATION_DATASET = "mmb0-1b_smb1-1b_1p"

# Table 1 blocks: (block, branch, plot module that built the csv, baselines).
# The baseline lists equal the plot modules' DEFAULT_BASELINES (checked at run time).
T1_BLOCKS = (
    ("niche", "niche", "plot_niche_identification_benchmark", (
        ("baseline-nichecompass", "NicheCompass"),
        ("baseline-cellcharter", "CellCharter"),
        ("baseline-banksy", "BANKSY"),
        ("baseline-graphst", "GraphST"),
        ("baseline-novae", "Novae"))),
    ("cell_type", "cell", "plot_cell_type_identification_benchmark", (
        ("baseline-nicheformer", "Nicheformer"),
        ("baseline-scvi", "scVI"),
        ("baseline-uce", "UCE"),
        ("baseline-geneformer", "Geneformer"),
        ("baseline-scgpt", "scGPT"),
        ("baseline-scgpt-spatial", "scGPT-spatial"))),
)
T1_CSV = {"niche": ("niche_identification_benchmark",
                    ["method", "seed", "MMD", "Niche ARI", "Niche NMI",
                     "Runtime (s)", "iLISI"]),
          "cell_type": ("cell_type_identification_benchmark",
                        ["method", "seed", "Cell-type ARI", "Cell-type NMI", "MMD",
                         "Runtime (s)", "iLISI"])}

# Embedding computed once and reused by every seed (only Leiden varies), so a
# shared top-level predicted_adata.h5ad is valid for every seed. GraphST is here
# because run_graphst.py calls GraphST.GraphST(adata, device=...) without
# random_seed and its published iLISI is bit-identical across seeds; the emb_sha1
# recorded per task confirms or refutes this.
SEED_INVARIANT = {"baseline-banksy", "baseline-graphst", "baseline-nicheformer",
                  "baseline-uce", "baseline-geneformer", "baseline-scgpt",
                  "baseline-scgpt-spatial"}
# Known reasons why an embedding is not on disk. The decision itself is taken from
# the h5py probe (no obs / obsm group or key), so a valid rerun (plan C-4/D5:
# Novae with the fixed writer) is scored like any other continuous baseline; the
# name only supplies the reason text, and decides alone only with --no-probe.
NOT_REEVALUABLE = {
    "baseline-novae": (
        "Novae's predicted_adata.h5ad has no obs/obsm (write_h5ad crashed), so it "
        "cannot be re-scored. Its published value is a scib-path score (< 1) on a "
        "continuous embedding, which neither fault affects, so it is carried "
        "unchanged."),
}
_NOT_ON_DISK = ("no obs group", "no obsm group", "obsm ")
NOT_RUN = {
    ("chl59-2b_1p", "baseline-graphst"): (
        "GraphST never produced output on CosMx NSCLC (PASTE pairwise_align CUDA "
        "OOM); Table 1 prints n/a."),
}

# Table 1 as printed in mlcb2026_paper/main.tex (tab:benchmark): five-seed means,
# NMI/ARI/iLISI to 3 d.p., MMD x1e3 to 2 d.p., RT in minutes. Used as the
# provenance gate (the discovered files must reproduce these) and as the "old"
# side of the diff report.
_T1_PRINTED = """
mmb0-1b_smb1 niche     SQUINT        0.702 0.410 1.05  0.609 16
mmb0-1b_smb1 niche     NicheCompass  0.694 0.373 8.90  0.005 50
mmb0-1b_smb1 niche     CellCharter   0.643 0.326 28.23 0.017 21
mmb0-1b_smb1 niche     BANKSY        0.602 0.306 1.36  0.769 19
mmb0-1b_smb1 niche     GraphST       0.517 0.211 46.83 1.448 42
mmb0-1b_smb1 niche     Novae         0.506 0.185 4.89  0.801 41
chl59        niche     SQUINT        0.397 0.172 0.52  0.948 33
chl59        niche     NicheCompass  0.423 0.246 2.02  0.418 210
chl59        niche     CellCharter   0.414 0.247 12.45 0.521 82
chl59        niche     BANKSY        0.228 0.114 0.78  0.762 202
chl59        niche     GraphST       n/a   n/a   n/a   n/a   n/a
chl59        niche     Novae         0.124 0.108 18.56 0.704 141
xhs1000      niche     SQUINT        0.514 0.207 3.76  0.496 25
xhs1000      niche     NicheCompass  0.580 0.278 8.27  0.042 54
xhs1000      niche     CellCharter   0.545 0.258 33.50 0.133 15
xhs1000      niche     BANKSY        0.480 0.217 6.01  0.305 16
xhs1000      niche     GraphST       0.372 0.184 17.08 2.041 23
xhs1000      niche     Novae         0.358 0.301 14.30 0.278 25
mmb0-1b_smb1 cell_type SQUINT        0.494 0.204 2.27  0.739 16
mmb0-1b_smb1 cell_type Nicheformer   0.529 0.298 36.32 0.015 27
mmb0-1b_smb1 cell_type scVI          0.501 0.237 17.67 0.289 21
mmb0-1b_smb1 cell_type UCE           0.474 0.276 30.00 0.039 70
mmb0-1b_smb1 cell_type Geneformer    0.383 0.196 43.50 0.099 19
mmb0-1b_smb1 cell_type scGPT         0.243 0.103 19.17 1.562 25
mmb0-1b_smb1 cell_type scGPT-spatial 0.222 0.117 22.77 1.567 32
chl59        cell_type SQUINT        0.604 0.339 0.61  0.967 33
chl59        cell_type Nicheformer   0.407 0.222 7.64  0.791 92
chl59        cell_type scVI          0.594 0.378 5.86  0.809 90
chl59        cell_type UCE           0.336 0.312 6.21  0.737 211
chl59        cell_type Geneformer    0.503 0.340 8.43  0.752 78
chl59        cell_type scGPT         0.487 0.301 6.32  0.774 102
chl59        cell_type scGPT-spatial 0.496 0.337 7.69  0.755 49
xhs1000      cell_type SQUINT        0.593 0.322 8.93  0.501 25
xhs1000      cell_type Nicheformer   0.393 0.158 18.48 0.541 18
xhs1000      cell_type scVI          0.662 0.426 18.47 0.487 12
xhs1000      cell_type UCE           0.599 0.376 11.23 0.451 43
xhs1000      cell_type Geneformer    0.553 0.302 17.05 0.481 13
xhs1000      cell_type scGPT         0.546 0.301 19.21 0.501 8
xhs1000      cell_type scGPT-spatial 0.559 0.349 16.42 0.463 6
"""


def _parse_t1_printed():
    out = {}
    for line in _T1_PRINTED.strip().splitlines():
        ds, block, method, nmi, ari, mmd, ilisi, rt = line.split()
        out[(ds, block, method)] = {"nmi": nmi, "ari": ari, "mmd": mmd,
                                    "ilisi": ilisi, "rt": rt}
    return out


PRINTED_T1 = _parse_t1_printed()

# Tables 3 and S1 as printed (main.tex tab:ablations, supplementary_material.tex
# tab:ablations_capacity), all Mouse Brain. A block is (name, plot_ablations.AXES
# key, columns); a column is (paper header, variant prefix, shaded reference,
# printed niche iLISI, printed cell iLISI), the printed values carrying their
# superscript stars exactly as typeset. The Discretization block has no AXES
# entry: its columns come from compare_discrete_vs_continuous.py (DISC_COLUMNS).
T3_LAYOUT = (
    ("Adjacency", "axis_1_adjacency", (
        ("w/", "s57_v19_", True, "0.609", "0.739"),
        ("w/o", "s57_v1_", False, "0.413***", "0.737"))),
    ("Contrastive", "axis_2_contrastive", (
        ("Cross", "s57_v19_", True, "0.609", "0.739"),
        ("Within", "s57_v20_", False, "0.469**", "0.489***"),
        ("None", "s57_v21_", False, "0.672**", "0.739"))),
    ("Decoder cov.", "axis_3_decoder_cov", (
        ("w/", "s57_v19_", True, "0.609", "0.739"),
        ("w/o", "s57_v2_", False, "0.057***", "0.030***"))),
    ("Coupling", "axis_10_coupling_mechanism", (
        ("FiLM", "s57_v19_", True, "0.609", "0.739"),
        ("Decpl", "s57_v25_", False, "0.313***", "0.740"),
        ("Trunk", "s57_v22_", False, "0.330**", "0.688**"),
        ("X-st", "s57_v23_", False, "0.285***", "0.749"),
        ("Aff", "s57_v24_", False, "0.355**", "0.713*"))),
    ("Discretization", "discretization", (
        ("VQ (codes)", "s57_v19_", True, "0.609", "0.739"),
        ("VQ (Leiden)", "s57_v19_", False, "0.036***", "0.387***"),
        ("VAE (Leiden)", "s57_v33_", False, "0.052***", "0.319***"))),
    ("Residual VQ (K)", "axis_11_residual_vq_levels", (
        ("(30, 90)", "s57_v19_", True, "0.609", "0.739"),
        ("2700", "s57_v30_", False, "0.572", "0.556***"),
        ("120", "s57_v31_", False, "0.648", "0.391***"),
        ("30", "s57_v32_", False, "0.532", "0.079***"))),
)
S1_LAYOUT = (
    ("GNN depth", "axis_4_gnn_layers", (
        ("1L", "s57_v19_", True, "0.609", "0.739"),
        ("2L", "s57_v3_", False, "0.551", "0.737"))),
    ("Neighbours", "axis_5_neighbors", (
        ("16", "s57_v19_", True, "0.609", "0.739"),
        ("8", "s57_v4_", False, "0.651", "0.734"),
        ("16/8", "s57_v5_", False, "0.598", "0.745"),
        ("24", "s57_v6_", False, "0.589", "0.751"))),
    ("Cell cb. K(1)", "axis_8_cell_codebook_L0", (
        ("(30,30)", "s57_v8_", True, "0.634", "0.709"),
        ("(10,30)", "s57_v13_", False, "0.574*", "0.569**"),
        ("(90,30)", "s57_v14_", False, "0.605", "0.734"),
        ("(300,30)", "s57_v15_", False, "0.558**", "0.640**"))),
    ("Cell cb. K(2)", "axis_6_codebook_size", (
        ("(30, 90)", "s57_v19_", True, "0.609", "0.739"),
        ("(30,10)", "s57_v7_", False, "0.627", "0.614***"),
        ("(30,30)", "s57_v8_", False, "0.634", "0.709"),
        ("(30,300)", "s57_v9_", False, "0.620", "0.737"))),
    ("Niche cb. K(1)", "axis_9_niche_codebook_L0", (
        ("(30,30)", "s57_v11_", True, "0.623", "0.751"),
        ("(10,30)", "s57_v16_", False, "0.572", "0.747"),
        ("(90,30)", "s57_v17_", False, "0.636", "0.739"),
        ("(300,30)", "s57_v18_", False, "0.617", "0.741"))),
    ("Niche cb. K(2)", "axis_7_niche_codebook_size", (
        ("(30, 90)", "s57_v19_", True, "0.609", "0.739"),
        ("(30,10)", "s57_v10_", False, "0.584", "0.755"),
        ("(30,30)", "s57_v11_", False, "0.623", "0.751"),
        ("(30,300)", "s57_v12_", False, "0.572", "0.737"))),
)
# Every metric row of Tables 3 and S1 as printed (main.tex l.793-801,
# supplementary_material.tex l.739-747), columns in the order of the layouts
# above, stars as typeset. NMI/ARI/MMD are the provenance gate for the values
# kept as they are; the iLISI rows must equal the layouts (asserted below).
_ABL_PRINTED = {
    "T3": """
niche NMI   0.702 0.350*** 0.702 0.704 0.702 0.702 0.702 0.702 0.705 0.691** 0.702 0.692*** 0.702 0.715*** 0.717*** 0.702 0.704 0.682*** 0.707*
niche ARI   0.410 0.149*** 0.410 0.415 0.413 0.410 0.415 0.410 0.419 0.403 0.413 0.407 0.410 0.396* 0.391* 0.410 0.300*** 0.339*** 0.422*
niche MMD   1.05 4.50** 1.05 1.51* 1.04 1.05 9.96*** 1.05 0.98 0.99 0.89 1.21 1.05 19.79*** 16.45*** 1.05 1.06 1.13 1.07
niche iLISI 0.609 0.413*** 0.609 0.469** 0.672** 0.609 0.057*** 0.609 0.313*** 0.330** 0.285*** 0.355** 0.609 0.036*** 0.052*** 0.609 0.572 0.648 0.532
cell NMI    0.494 0.493 0.494 0.497 0.424*** 0.494 0.525*** 0.494 0.493 0.486 0.496 0.489 0.494 0.509** 0.514*** 0.494 0.387*** 0.445*** 0.479**
cell ARI    0.204 0.207 0.204 0.216 0.165*** 0.204 0.252*** 0.204 0.202 0.208 0.202 0.218 0.204 0.243*** 0.245** 0.204 0.006*** 0.080*** 0.199
cell MMD    2.27 2.08 2.27 2.79 2.21 2.27 46.30*** 2.27 2.21 0.96** 2.35 1.11** 2.27 3.12 9.15*** 2.27 2.54 1.79 1.79
cell iLISI  0.739 0.737 0.739 0.489*** 0.739 0.739 0.030*** 0.739 0.740 0.688** 0.749 0.713* 0.739 0.387*** 0.319*** 0.739 0.556*** 0.391*** 0.079***
""",
    "S1": """
niche NMI   0.702 0.675** 0.702 0.661*** 0.685** 0.710** 0.702 0.703 0.702 0.701 0.702 0.704 0.702 0.702 0.699 0.600*** 0.689** 0.701 0.702 0.701 0.699 0.703
niche ARI   0.410 0.379** 0.410 0.383*** 0.404 0.412 0.410 0.414 0.414 0.413 0.410 0.412 0.410 0.410 0.404 0.246*** 0.360*** 0.410 0.410 0.411 0.404 0.412
niche MMD   1.05 1.52 1.05 0.97 1.09 1.11 1.01 1.13 1.46** 2.53*** 1.05 1.02 1.01 0.99 0.98 0.96 1.10 1.02 1.05 0.96 0.98 1.03
niche iLISI 0.609 0.551 0.609 0.651 0.598 0.589 0.634 0.574* 0.605 0.558** 0.609 0.627 0.634 0.620 0.623 0.572 0.636 0.617 0.609 0.584 0.623 0.572
cell NMI    0.494 0.491 0.494 0.490 0.494 0.493 0.490 0.446*** 0.458*** 0.434*** 0.494 0.490 0.490 0.493 0.490 0.492 0.493 0.495* 0.494 0.491 0.490 0.490
cell ARI    0.204 0.201 0.204 0.198 0.205 0.201 0.200 0.268*** 0.097*** 0.040*** 0.204 0.201 0.200 0.201 0.200 0.203 0.203 0.209* 0.204 0.200 0.200 0.203
cell MMD    2.27 2.51 2.27 2.44 2.20 2.34 2.12 1.73* 2.52 3.04** 2.27 1.64* 2.12 2.43 2.43 2.26 2.02* 2.53 2.27 2.19 2.43 2.49
cell iLISI  0.739 0.737 0.739 0.734 0.745 0.751 0.709 0.569** 0.734 0.640** 0.739 0.614*** 0.709 0.737 0.751 0.747 0.739 0.741 0.739 0.755 0.751 0.737
""",
}


def _parse_abl_printed():
    """{(table, block, column, branch): {nmi, ari, mmd, ilisi}} (with stars)."""
    out = {}
    for table, layout in (("T3", T3_LAYOUT), ("S1", S1_LAYOUT)):
        cols = [(block, c[0]) for block, _ax, cs in layout for c in cs]
        for line in _ABL_PRINTED[table].strip().splitlines():
            branch, metric, *vals = line.split()
            if len(vals) != len(cols):
                raise RuntimeError(f"{table} {branch} {metric}: {len(vals)} printed "
                                   f"values for {len(cols)} columns")
            for (block, col), v in zip(cols, vals):
                out.setdefault((table, block, col, branch), {})[metric.lower()] = v
        for block, _ax, cs in layout:          # the iLISI rows must equal the layouts
            for col, _p, _r, pn, pc in cs:
                for branch, want in (("niche", pn), ("cell", pc)):
                    got = out[(table, block, col, branch)]["ilisi"]
                    if got != want:
                        raise RuntimeError(f"{table}/{block}/{col}/{branch}: printed "
                                           f"iLISI {got} vs layout {want}")
    return out


PRINTED_ABL = _parse_abl_printed()

# Discretization column -> (condition in discretization_per_seed.csv, obsm keys).
# VQ (codes) is s57_v19 z_q (quantized, affected); VQ (Leiden) is s57_v19's
# pre-quantization latent and VAE (Leiden) is s57_v33's ContinuousVQ passthrough
# embedding (both continuous, unaffected).
DISC_COLUMNS = {
    "VQ (codes)": ("SQUINT (codes)", EMB_KEYS),
    "VQ (Leiden)": ("SQUINT (Leiden)", LATENT_KEYS),
    "VAE (Leiden)": ("Continuous (Leiden)", EMB_KEYS),
}

# Numbers quoted in the text and in the submitted author response, with the
# table cells they are computed from. Cells are (table, ds_short, block, column,
# branch); `how` says how the new string is formed.
TEXT_QUOTES = (
    ("0.739", "Table 1 Mouse Brain cell-type SQUINT; author response l.25 and l.61; "
     "Table 3 cell-iLISI default", "one",
     (("T1", "mmb0-1b_smb1", "cell_type", "SQUINT", "cell"),)),
    ("0.609 / 0.61", "Table 1 Mouse Brain niche SQUINT; main text ablations "
     "('0.61 -> 0.04-0.05', '0.61 vs <=0.36'); author response l.49, l.61, l.64; "
     "Table S1 caption", "one",
     (("T1", "mmb0-1b_smb1", "niche", "SQUINT", "niche"),)),
    ("0.74 -> 0.49", "main text ablations (within-section contrastive, cell iLISI)",
     "arrow", (("T3", "mmb0-1b_smb1", "Contrastive", "Cross", "cell"),
               ("T3", "mmb0-1b_smb1", "Contrastive", "Within", "cell"))),
    ("0.313", "author response l.49 (decoupling drops niche iLISI from 0.609)",
     "one", (("T3", "mmb0-1b_smb1", "Coupling", "Decpl", "niche"),)),
    ("<=0.36", "main text ablations (FiLM niche iLISI 0.61 vs <=0.36 for the other "
     "couplings)", "max",
     (("T3", "mmb0-1b_smb1", "Coupling", "Decpl", "niche"),
      ("T3", "mmb0-1b_smb1", "Coupling", "Trunk", "niche"),
      ("T3", "mmb0-1b_smb1", "Coupling", "X-st", "niche"),
      ("T3", "mmb0-1b_smb1", "Coupling", "Aff", "niche"))),
    ("0.057", "author response l.61 (niche iLISI without the decoder covariate)",
     "one", (("T3", "mmb0-1b_smb1", "Decoder cov.", "w/o", "niche"),)),
    ("0.030", "author response l.25 and l.61 (cell iLISI without the decoder "
     "covariate)", "one",
     (("T3", "mmb0-1b_smb1", "Decoder cov.", "w/o", "cell"),)),
    ("0.04-0.05", "main text ablations (VQ / VAE Leiden niche iLISI; continuous "
     "latents, expected unchanged)", "range",
     (("T3", "mmb0-1b_smb1", "Discretization", "VQ (Leiden)", "niche"),
      ("T3", "mmb0-1b_smb1", "Discretization", "VAE (Leiden)", "niche"))),
    ("0.651", "Table S1 caption (k=8 niche iLISI)", "one",
     (("S1", "mmb0-1b_smb1", "Neighbours", "8", "niche"),)),
    ("0.589", "author response l.64; Table S1 caption (k=24 niche iLISI)", "one",
     (("S1", "mmb0-1b_smb1", "Neighbours", "24", "niche"),)),
)

MANIFEST_FIELDS = [
    "row_id", "table", "dataset_tag", "ds_short", "dataset_label", "block", "branch",
    "column", "method", "role", "is_ref", "ref_column", "paper_visible", "seed",
    "status", "seed_status", "variant_dir", "ts_dir", "ts_selection", "run_dir",
    "adata_path", "adata_scope", "obsm_key", "batch_key", "emb_kind",
    "emb_seed_invariant", "ilisi_max_cells", "subsample_seed", "cellset", "n_obs",
    "emb_dim", "probe_error", "nmi", "ari", "mmd", "runtime_s", "runtime_ts",
    "mmd_producer", "ilisi_published", "run_dir_ilisi", "printed_ilisi",
    "printed_stars", "printed_nmi", "printed_ari", "printed_mmd", "printed_rt",
    "printed_provenance_ok", "printed_other_ok", "printed_other_detail",
    "fallback_suspected", "tie_affected", "control", "unchanged_expected",
    "sanity_tier", "compute", "reason", "task_id", "job_id",
]
RESULT_FIELDS = [
    "task_id", "job_id", "protocol", "estimator", "statistic", "perm_seed",
    "nnd_random_state", "value", "seconds", "n_missing_nbrs", "n_cells",
    "n_cells_file", "cellset", "n_batches", "k", "perplexity", "adata_path",
    "obsm_key", "batch_key", "emb_dim", "emb_dtype", "emb_sha1", "n_unique_rows",
    "frac_unique_rows", "max_tie_group", "frac_cells_tied", "frac_cells_tie_gt_k",
    "scib_version", "pynndescent_version", "numpy_version", "jax_version", "python",
    "host", "lsf_job_id", "git_sha", "started_utc",
]
# Roles whose MMD came from compute_inference_metrics (SQUINT runs), so the
# producer's MMD call can be replayed exactly as a provenance check.
SQUINT_ROLES = ("squint", "ablation", "latent_control")
MMD_N_SUB, MMD_N_SIGMA = 2000, 1000   # compute_inference_metrics defaults
MMD_TOL = 1e-6


# =============================================================================
# Small helpers
# =============================================================================
def _need_numpy():
    if np is None:
        raise SystemExit("numpy is required for this step. Activate the cluster "
                         "venv: source /nfs/team361/sb75/.venvs/squint/bin/activate")


def _fnum(x):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _finite(x):
    return math.isfinite(_fnum(x))


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _mean(vals):
    v = [_fnum(x) for x in vals if _finite(x)]
    return sum(v) / len(v) if v else float("nan")


def _sd(vals):
    v = [_fnum(x) for x in vals if _finite(x)]
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _f3(x):
    return f"{_fnum(x):.3f}" if _finite(x) else ""


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _h8(s):
    return hashlib.md5(str(s).encode()).hexdigest()[:8]


def _split_printed(s):
    """'0.413***' -> ('0.413', '***')."""
    m = re.match(r"^([0-9.]+|n/a)(\**)$", s.strip())
    if not m:
        raise ValueError(f"cannot parse printed value {s!r}")
    return m.group(1), m.group(2)


def _cell(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return repr(v) if math.isfinite(v) else ""
    if v is None:
        return ""
    return v


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _seed_int(v):
    return int(float(v)) if _finite(v) else 0


def _write_new(path, write_fn):
    """
    Create `path` and never replace an existing file. The content goes to a temp
    file next to it, which is then hard-linked into place: os.link refuses an
    existing target, and a crash mid-write never leaves a half file behind that
    would later look finished.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write_fn(tmp)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise
        except OSError:          # a filesystem without hard links
            with open(tmp, "rb") as src, open(path, "xb") as dst:
                dst.write(src.read())
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_new_csv(path, rows, fields):
    def w(p):
        with open(p, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            wr.writeheader()
            for r in rows:
                wr.writerow({k: _cell(r.get(k, "")) for k in fields})
    _write_new(path, w)


def _write_new_text(path, text):
    _write_new(path, lambda p: Path(p).write_text(text))


def _write_new_df(path, df):
    _write_new(path, lambda p: df.to_csv(p, index=False))


def _supersede(path):
    """Move `path` aside into superseded/ (kept, never deleted)."""
    path = Path(path)
    dest_dir = path.parent / "superseded"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{path.stem}.{_stamp()}.{os.getpid()}{path.suffix}"
    os.rename(path, dest)
    return dest


def _resolve_dirs(a):
    """(artifacts root, out dir); the out dir must be a subfolder of the root."""
    art = Path(os.path.abspath(a.artifacts_root))
    out = (Path(os.path.abspath(a.out_dir)) if a.out_dir
           else art / DEFAULT_OUT_NAME)
    if out == art or art not in out.parents:
        raise SystemExit(f"--out-dir {out} must be a subfolder of the artifacts root "
                         f"{art}; nothing is written anywhere else.")
    return art, out


def _out_dir_problem(out):
    """Why `out` may not be written into (None if it may); creates nothing."""
    if out.exists():
        if not out.is_dir():
            return f"{out} exists and is not a folder."
        if (out / OUT_MARKER).is_file():
            return None
        if any(out.iterdir()):
            return (f"{out} exists and was not created by this script. Choose a new "
                    f"--out-dir (OUT=...); existing folders are not written into.")
    return None


def _claim_out_dir(out):
    """Create the out dir, or accept it only if this script created it."""
    problem = _out_dir_problem(out)
    if problem:
        raise SystemExit(problem)
    if (out / OUT_MARKER).is_file():
        return
    out.mkdir(parents=True, exist_ok=True)
    if not (out / OUT_MARKER).exists():
        _write_new_text(out / OUT_MARKER,
                        f"created by {Path(__file__).name} at {_stamp()} UTC\n")


_MOD_DIRS = {
    "compute_label_metric": "analysis/benchmarking/niche_identification",
    "compute_table1_significance": "analysis/benchmarking",
    "plot_niche_identification_benchmark": "analysis/benchmarking/plots",
    "plot_cell_type_identification_benchmark": "analysis/benchmarking/plots",
    "plot_ablations": "analysis/ablations/plots",
    "summarize_ablation_multiseed": "analysis/ablations",
    "run_ablation_report": "analysis/ablations",
}


def _repo_module(name):
    """
    Import one of the modules that PRODUCED the published numbers, so discovery,
    filtering and significance here run through the very same functions rather
    than a reimplementation that could drift.
    """
    import importlib
    d = str(REPO / _MOD_DIRS[name])
    if d not in sys.path:
        sys.path.insert(0, d)
    if name.startswith("plot_") or name == "run_ablation_report":
        os.environ.setdefault("MPLBACKEND", "Agg")
    return importlib.import_module(name)


# =============================================================================
# Manifest: discovery
# =============================================================================
def _ts_candidates(variant_dir):
    """
    TS dirs under `variant_dir` holding a per-seed metrics csv, newest first. The
    first is what the plot scripts' _find_latest_metrics_dir and
    summarize_ablation_multiseed._resolve_metrics_dir select.
    """
    variant_dir = Path(variant_dir)
    if not variant_dir.is_dir():
        return []
    out = []
    for ts in sorted((p for p in variant_dir.iterdir() if p.is_dir()),
                     key=lambda p: p.name, reverse=True):
        m = ts / "metrics"
        if m.is_dir() and any((m / f).is_file() for f in PER_SEED_FILES):
            out.append(ts)
    return out


def _select_ts(cands, ok):
    """
    The plot rule (latest TS with per-seed csvs) when that run reproduces the
    printed table; otherwise the newest run that does, because a run made after
    the table was built must not silently replace the one behind it. If none
    reproduces it, the latest is kept and flagged.
    """
    def safe(c):
        try:
            return bool(ok(c))
        except Exception:                                   # noqa: BLE001
            return False
    if not cands:
        return None, "missing"
    if safe(cands[0]):
        return cands[0], "latest"
    for c in cands[1:]:
        if safe(c):
            return c, "older_ts_matches_printed"
    return cands[0], "latest_UNMATCHED_printed"


def _avg_dups(pairs, what, notes=None):
    """
    {key: mean of its values} for (key, value) pairs. Duplicate per-seed rows are
    AVERAGED, as the producers do (the Table 1 plot's pivot_table(aggfunc=mean),
    run_ablation_report's _metric_array), never last-row-wins; a NOTE is added
    when duplicates differ by more than 1e-9.
    """
    acc = {}
    for k, v in pairs:
        acc.setdefault(k, []).append(v)
    out = {}
    for k, vs in acc.items():
        fin = [x for x in vs if math.isfinite(x)]
        out[k] = sum(fin) / len(fin) if fin else float("nan")
        if notes is not None and len(fin) > 1 and max(fin) - min(fin) > 1e-9:
            notes.append(f"NOTE {what} {k}: {len(fin)} duplicate rows differ "
                         f"({min(fin)!r} .. {max(fin)!r}); averaged")
    return out


def _batch_scores(metrics_dir, notes=None):
    """{(seed, emb_key, metric): score} from per_seed_batch_integration.csv."""
    f = Path(metrics_dir) / "per_seed_batch_integration.csv"
    if not f.is_file():
        return {}
    return _avg_dups((((_seed_int(r.get("seed")), r.get("emb_key", ""),
                        r.get("metric", "")), _fnum(r.get("score")))
                      for r in _read_csv(f)), f, notes)


def _seed_run_dirs(ts_dir):
    """
    ({seed: run_dir}, {seed: status}) of a multiseed sweep. A seed is included
    when seed_run_index.csv gives it a run_dir: the aggregator writes a run_dir
    for exactly the seeds it aggregated, including seeds a user added through
    --seeds although their stamp says e.g. 'metrics_failed'. The status is
    recorded, not filtered on. Without the index, every seed_runs/ stamp is
    used; rows are only ever built for seeds present in the per-seed metrics.
    """
    ts_dir = Path(ts_dir)
    out, status = {}, {}
    idx = ts_dir / "seed_run_index.csv"
    if idx.is_file():
        for r in _read_csv(idx):
            s = _seed_int(r.get("seed"))
            status[s] = (r.get("status") or "").strip()
            if (r.get("run_dir") or "").strip():
                out[s] = Path(r["run_dir"].strip())
    if not out:
        for f in sorted((ts_dir / "seed_runs").glob("seed_*_run_dir.txt")):
            m = re.match(r"seed_(\d+)_run_dir\.txt$", f.name)
            if not m:
                continue
            st = f.with_name(f"seed_{m.group(1)}_status.txt")
            status[int(m.group(1))] = st.read_text().strip() if st.is_file() else ""
            rd = f.read_text().strip()
            if rd:
                out[int(m.group(1))] = Path(rd)
    return out, status


def _run_dir_ilisi(run_dir, emb_key):
    """The iLISI the run itself wrote (cross-check of the sweep aggregate)."""
    if not run_dir:
        return float("nan")
    f = Path(run_dir) / "metrics" / "batch_integration_metrics.csv"
    if not f.is_file():
        return float("nan")
    for r in _read_csv(f):
        if r.get("emb_key") == emb_key and r.get("metric") == "iLISI":
            return _fnum(r.get("score"))
    return float("nan")


def _baseline_adata(ts_dir, seed):
    """Per-seed snapshot if written (runs from 2026-07-01 on), else the shared one."""
    p = Path(ts_dir) / "seeds" / f"seed_{seed}" / "predicted_adata.h5ad"
    if p.is_file():
        return p, "per_seed"
    p = Path(ts_dir) / "predicted_adata.h5ad"
    if p.is_file():
        return p, "shared_toplevel"
    return None, "none"


def _probe(path, key, batch_key, cache, enabled):
    """Header-only h5py check that the row can be scored; never reads arrays."""
    out = {"n_obs": -1, "emb_dim": -1, "probe_error": ""}
    if not enabled or not path:
        return out
    ck = (str(path), key, batch_key)
    if ck in cache:
        return cache[ck]
    try:
        import h5py
        if not Path(path).is_file():
            out["probe_error"] = "adata file missing"
        else:
            with h5py.File(path, "r") as h:
                if "obs" not in h:
                    out["probe_error"] = "no obs group (stub file)"
                elif batch_key not in h["obs"]:
                    out["probe_error"] = f"obs column {batch_key!r} absent"
                elif "obsm" not in h:
                    out["probe_error"] = "no obsm group"
                elif key not in h["obsm"]:
                    have = list(h["obsm"].keys())
                    out["probe_error"] = f"obsm {key!r} absent (have {have})"
                else:
                    ds = h["obsm"][key]
                    if not isinstance(ds, h5py.Dataset) or len(ds.shape) != 2:
                        out["probe_error"] = f"obsm {key!r} is not a dense 2-D array"
                    else:
                        out["n_obs"], out["emb_dim"] = int(ds.shape[0]), int(ds.shape[1])
    except Exception as ex:                                 # noqa: BLE001
        out["probe_error"] = f"{type(ex).__name__}: {ex}"
    cache[ck] = out
    return out


@contextmanager
def _pinned_metrics_dir(mod, metrics_dir):
    """
    Make a plot module's load_method_metrics read one specific TS. It resolves
    the TS through the module global _find_latest_metrics_dir at call time, so
    pinning that global reuses the plot's own filtering code unchanged.
    """
    orig = mod._find_latest_metrics_dir
    mod._find_latest_metrics_dir = lambda _variant_dir: Path(metrics_dir)
    try:
        yield
    finally:
        mod._find_latest_metrics_dir = orig


_T1_METRIC_NAMES = {"Niche NMI": "nmi", "Niche ARI": "ari", "Cell-type NMI": "nmi",
                    "Cell-type ARI": "ari", "iLISI": "ilisi", "MMD": "mmd"}


def _t1_per_seed(mod, block, variant_dir, ts_dir, method, label_key, notes=None):
    """
    {seed: {nmi, ari, ilisi, mmd}} through the plot module's load_method_metrics,
    duplicate (seed, metric) rows averaged exactly as make_figure's
    pivot_table(aggfunc=mean) does (e.g. NSCLC SQUINT seed 3 has two split=all
    niche rows; the published table holds their mean).
    """
    with _pinned_metrics_dir(mod, Path(ts_dir) / "metrics"):
        if block == "niche":
            df = mod.load_method_metrics(Path(variant_dir), method,
                                         niche_label_key=label_key)
        else:
            df = mod.load_method_metrics(Path(variant_dir), method,
                                         cell_label_key=label_key)
    out = {}
    if df is None or df.empty:
        return out
    pairs = []
    for _, r in df.iterrows():
        name = _T1_METRIC_NAMES.get(str(r["metric"]))
        if name:
            pairs.append(((int(r["seed"]), name), float(r["value"])))
    for (s, name), v in _avg_dups(pairs, f"{method} {Path(ts_dir).name}",
                                  notes).items():
        out.setdefault(s, {})[name] = v
    return out


def _t1_emb_key(mod, block, metrics_dir):
    """The emb_key the plot module picks (neighborhood_emb / cell_emb / baseline)."""
    f = Path(metrics_dir) / "per_seed_batch_integration.csv"
    if not f.is_file():
        return ""
    keys = sorted({r.get("emb_key", "") for r in _read_csv(f)} - {""})
    pick = (mod._pick_batch_emb_key if block == "niche"
            else mod._pick_batch_emb_key_cell)
    return pick(keys) or ""


def _t1_runtimes(mod, variant_dir, ts_dir, method, notes):
    """
    ({seed: runtime_s}, TS it came from). The plot's load_method_runtime_rows
    reads the newest TS with a per_seed_runtimes.csv; when the table was built,
    that was the metrics TS or an older one, never a newer one. So: the metrics
    TS itself, else the newest TS not newer than it; the plot rule (latest) only
    as a last resort, with a NOTE.
    """
    vdir, ts_dir = Path(variant_dir), Path(ts_dir)
    cands = sorted((p for p in vdir.iterdir() if p.is_dir()
                    and (p / "metrics" / "per_seed_runtimes.csv").is_file()),
                   key=lambda p: p.name, reverse=True) if vdir.is_dir() else []
    pick = next((p for p in cands if p.name <= ts_dir.name), None)
    if pick is None:
        rows = mod.load_method_runtime_rows(vdir, method)
        if rows:
            notes.append(f"NOTE {vdir.name}: runtimes only in a TS newer than the "
                         f"metrics TS {ts_dir.name}; using the plot rule (latest)")
            return (_avg_dups(((int(r["seed"]), float(r["value"])) for r in rows),
                              f"runtime {vdir.name}", notes),
                    cands[0].name if cands else "")
        return {}, ""
    pairs = []
    for r in _read_csv(pick / "metrics" / "per_seed_runtimes.csv"):
        v = _fnum(r.get("runtime_seconds"))
        if math.isfinite(v):
            pairs.append((_seed_int(r.get("seed")), v))
    return _avg_dups(pairs, f"runtime {vdir.name}/{pick.name}", notes), pick.name


# Paper formatting of each kept metric: (scale, decimals).
_FMT = {"nmi": (1.0, 3), "ari": (1.0, 3), "mmd": (1e3, 2), "rt": (1.0 / 60.0, 0),
        "ilisi": (1.0, 3)}


def _other_check(values, printed):
    """
    Does each kept metric's five-seed mean format to its printed value?
    values: {metric: [per-seed values]}; printed: {metric: '0.410***'}.
    Returns (True / False / '' if nothing to check, 'metric got vs printed; ...').
    """
    bad, n = [], 0
    for key, vals in values.items():
        want = printed.get(key, "")
        if not want or want.startswith("n/a"):
            continue
        want = _split_printed(want)[0]
        m = _mean(vals)
        scale, nd = _FMT[key]
        got = f"{m * scale:.{nd}f}" if math.isfinite(m) else "nan"
        n += 1
        if got != want:
            bad.append(f"{key} {got} vs printed {want}")
    if not n:
        return "", ""
    return (not bad), "; ".join(bad)


def _t1_matches(per_seed, printed):
    """Do the five-seed means format to the printed NMI / ARI / MMD / iLISI?"""
    if not per_seed:
        return False
    for key, scale, nd in (("nmi", 1.0, 3), ("ari", 1.0, 3), ("mmd", 1e3, 2),
                           ("ilisi", 1.0, 3)):
        want = printed.get(key, "")
        if not want or want == "n/a":
            continue
        m = _mean([v.get(key) for v in per_seed.values()])
        if not math.isfinite(m) or f"{m * scale:.{nd}f}" != want:
            return False
    return True


def _new_row(**kw):
    r = {k: "" for k in MANIFEST_FIELDS}
    r.update(kw)
    return r


def _t1_rows(art, notes):
    rows = []
    for ds in DATASETS:
        for block, branch, modname, baselines in T1_BLOCKS:
            mod = _repo_module(modname)
            mine = {d: lab for d, lab in baselines}
            theirs = dict(getattr(mod, "DEFAULT_BASELINES", {}))
            if theirs and theirs != mine:
                notes.append(f"NOTE {modname}.DEFAULT_BASELINES differs from this "
                             f"script's Table 1 list: {theirs} vs {mine}")
            label_key = ds.niche_label if block == "niche" else ds.cell_label
            entries = [(ds.squint_variant + "__multiseed", "SQUINT", "squint")]
            entries += [(d, lab, "baseline") for d, lab in baselines]
            for vname, method, role in entries:
                vdir = art / ds.tag / vname
                pr = PRINTED_T1.get((ds.short, block, method), {})
                base = dict(table="T1", dataset_tag=ds.tag, ds_short=ds.short,
                            dataset_label=ds.label, block=block, branch=branch,
                            column=method, method=method, role=role,
                            is_ref=(role == "squint"), ref_column="SQUINT",
                            paper_visible=True, variant_dir=str(vdir),
                            batch_key=BATCH_KEY, printed_ilisi=pr.get("ilisi", ""),
                            printed_stars="", printed_nmi=pr.get("nmi", ""),
                            printed_ari=pr.get("ari", ""),
                            printed_mmd=pr.get("mmd", ""),
                            printed_rt=pr.get("rt", ""))
                if (ds.tag, vname) in NOT_RUN:
                    for s in SEEDS:
                        rows.append(_new_row(**base, seed=s, status="not_run",
                                             reason=NOT_RUN[(ds.tag, vname)]))
                    continue
                cands = _ts_candidates(vdir)
                ts, sel = _select_ts(cands, lambda t: _t1_matches(
                    _t1_per_seed(mod, block, vdir, t, method, label_key), pr))
                if ts is None:
                    for s in SEEDS:
                        rows.append(_new_row(**base, seed=s, status="missing",
                                             reason=f"no TS with per-seed csvs "
                                                    f"under {vdir}"))
                    continue
                if sel == "latest":
                    plot_pick = mod._find_latest_metrics_dir(vdir)
                    if plot_pick is not None and Path(plot_pick) != ts / "metrics":
                        notes.append(f"NOTE {ds.short}/{method}: plot module picks "
                                     f"{plot_pick}, this script {ts}/metrics")
                per_seed = _t1_per_seed(mod, block, vdir, ts, method, label_key,
                                        notes)
                if not per_seed:
                    for s in SEEDS:
                        rows.append(_new_row(**base, seed=s, status="missing",
                                             ts_dir=str(ts), ts_selection=sel,
                                             reason=f"no per-seed metrics for label "
                                                    f"{label_key!r} in {ts}"))
                    continue
                emb_key = _t1_emb_key(mod, block, ts / "metrics")
                rts, rt_ts = _t1_runtimes(mod, vdir, ts, method, notes)
                bsc = _batch_scores(ts / "metrics", notes)
                prov = _t1_matches(per_seed, pr)
                # Every kept metric, RT included, against the printed table.
                other_ok, other_detail = _other_check(
                    {"nmi": [v.get("nmi") for v in per_seed.values()],
                     "ari": [v.get("ari") for v in per_seed.values()],
                     "mmd": [v.get("mmd") for v in per_seed.values()],
                     "rt": [rts.get(s) for s in per_seed]}, pr)
                if other_ok is False:
                    notes.append(f"WARN {ds.short}/{block}/{method}: kept metrics do "
                                 f"not reproduce the printed table: {other_detail}")
                runs, run_status = (_seed_run_dirs(ts) if role == "squint"
                                    else ({}, {}))
                if role == "squint" and len(set(map(str, runs.values()))) < len(runs):
                    notes.append(f"WARN {ds.short}/{method}: run-dir collision in "
                                 f"{ts}/seed_run_index.csv")
                if sorted(per_seed) != list(SEEDS):
                    notes.append(f"NOTE {ds.short}/{block}/{method}: seeds "
                                 f"{sorted(per_seed)} in {ts}")
                for s in sorted(per_seed):
                    m = per_seed[s]
                    r = _new_row(**base, seed=s, status="ok", ts_dir=str(ts),
                                 ts_selection=sel, obsm_key=emb_key,
                                 nmi=m.get("nmi", float("nan")),
                                 ari=m.get("ari", float("nan")),
                                 mmd=m.get("mmd", float("nan")),
                                 runtime_s=rts.get(s, float("nan")),
                                 runtime_ts=rt_ts,
                                 ilisi_published=m.get("ilisi", float("nan")),
                                 printed_provenance_ok=prov,
                                 printed_other_ok=other_ok,
                                 printed_other_detail=other_detail)
                    if role == "squint":
                        rd = runs.get(s)
                        r.update(run_dir=str(rd) if rd else "",
                                 seed_status=run_status.get(s, ""),
                                 adata_path=(str(rd / "predicted_adata.h5ad")
                                             if rd else ""),
                                 adata_scope="per_run" if rd else "none",
                                 emb_kind="quantized", emb_seed_invariant=False,
                                 ilisi_max_cells=ILISI_MAX_CELLS,
                                 subsample_seed=SUBSAMPLE_SEED,
                                 mmd_producer=bsc.get((s, emb_key, "MMD"),
                                                      float("nan")),
                                 run_dir_ilisi=_run_dir_ilisi(rd, emb_key))
                        rows.append(r)
                        # The continuous pre-quantization latent of the same run:
                        # not in the paper, but it makes the code-resolution effect
                        # visible next to the quantized value (camera-ready C-1).
                        lk = LATENT_KEYS[branch]
                        rows.append(dict(
                            r, column="SQUINT (latent)", method="SQUINT (latent)",
                            role="latent_control", is_ref=False, paper_visible=False,
                            obsm_key=lk, emb_kind="continuous",
                            nmi=float("nan"), ari=float("nan"),
                            mmd=bsc.get((s, lk, "MMD"), float("nan")),
                            mmd_producer=bsc.get((s, lk, "MMD"), float("nan")),
                            runtime_s=float("nan"), runtime_ts="",
                            ilisi_published=bsc.get((s, lk, "iLISI"), float("nan")),
                            printed_ilisi="", printed_nmi="", printed_ari="",
                            printed_mmd="", printed_rt="", printed_provenance_ok="",
                            printed_other_ok="", printed_other_detail="",
                            run_dir_ilisi=_run_dir_ilisi(rd, lk)))
                    else:
                        ap, scope = _baseline_adata(ts, s)
                        r.update(adata_path=str(ap) if ap else "", adata_scope=scope,
                                 emb_kind="continuous",
                                 emb_seed_invariant=vname in SEED_INVARIANT,
                                 ilisi_max_cells=0, subsample_seed="")
                        rows.append(r)
    return rows


def _check_layout(pa, notes):
    """The paper's column layout must match plot_ablations.AXES."""
    axes = {a.key: a for a in pa.AXES}
    for table, layout in (("T3", T3_LAYOUT), ("S1", S1_LAYOUT)):
        for block, axis_key, cols in layout:
            if axis_key == "discretization":
                continue
            ax = axes.get(axis_key)
            if ax is None:
                notes.append(f"WARN {table}/{block}: axis {axis_key} not in "
                             f"plot_ablations.AXES")
                continue
            theirs = {e.prefix for e in ax.entries}
            mine = {c[1] for c in cols}
            ref_theirs = next((e.prefix for e in ax.entries if e.is_default), None)
            ref_mine = next(c[1] for c in cols if c[2])
            if theirs != mine or ref_theirs != ref_mine:
                notes.append(f"WARN {table}/{block}: layout {sorted(mine)} ref "
                             f"{ref_mine} vs AXES {sorted(theirs)} ref {ref_theirs}")


def _ablation_ts(prefix, art, ds_tag, sm, checks):
    """Sweep TS for one variant prefix: summarize_ablation_multiseed's choice,
    validated against the printed iLISI of every column that uses it."""
    base = art / ds_tag
    sweeps = sorted(d for d in base.glob(f"{prefix}*__multiseed") if d.is_dir())
    cands = []
    md = sm._resolve_metrics_dir(prefix, str(art), ds_tag)
    if md is not None:
        cands.append(Path(md).parent)
    for sw in sweeps:
        cands += [c for c in _ts_candidates(sw) if c not in cands]

    def ok(ts):
        b = _batch_scores(ts / "metrics")
        for nkey, ckey, pn, pc in checks:
            for key, want in ((nkey, pn), (ckey, pc)):
                vals = [v for (s, k, m), v in b.items() if k == key and m == "iLISI"]
                if _f3(_mean(vals)) != want:
                    return False
        return True
    ts, sel = _select_ts(cands, ok)
    return ts, sel, len(sweeps)


def _niche_id(metrics_dir, rar, notes=None):
    """{(seed, code_key, label_key): (NMI, ARI)} after the report's L=1 key fix;
    duplicate rows averaged (run_ablation_report._metric_array uses them all)."""
    import pandas as pd
    f = Path(metrics_dir) / "per_seed_niche_identification.csv"
    if not f.is_file():
        return {}, set()
    d = pd.read_csv(f)
    labels = (set(d["label_key"].astype(str).unique())
              if "label_key" in d.columns else set())
    if "split" not in d.columns:
        d["split"] = "all"
    d = rar._canonicalize_code_keys(d)
    d = d[d["split"] == "all"]
    pairs = []
    for _, r in d.iterrows():
        k = (int(r.get("seed", 0)), str(r["code_key"]), str(r["label_key"]))
        pairs += [((k, "NMI"), float(r["NMI"])), ((k, "ARI"), float(r["ARI"]))]
    avg = _avg_dups(pairs, f, notes)
    out = {k: (avg.get((k, "NMI"), float("nan")), avg.get((k, "ARI"), float("nan")))
           for (k, _m) in avg}
    return out, labels


def _disc_expected():
    """{(condition, branch): printed iLISI} of Table 3's Discretization block."""
    cols = next(c for _b, ax, c in T3_LAYOUT if ax == "discretization")
    out = {}
    for col, _p, _r, pn, pc in cols:
        cond = DISC_COLUMNS[col][0]
        out[(cond, "niche")] = _split_printed(pn)[0]
        out[(cond, "cell")] = _split_printed(pc)[0]
    return out


def _load_disc(p):
    rows = _read_csv(p)
    if not rows or not {"condition", "branch", "metric", "value"} <= set(rows[0]):
        return {}
    return _avg_dups((((r["condition"].strip(), r["branch"].strip().lower(),
                        r["metric"].strip(), _seed_int(r.get("seed_idx"))),
                       _fnum(r.get("value"))) for r in rows), p)


def _disc_candidates(art, ds_tag, rar, notes):
    """
    Every discretization_per_seed.csv that could be Table 3's source, most likely
    first: the default OUTBASE of submit_compare_discrete_vs_continuous.sh
    (<ds>/discretization/, which run_ablation_report's lookup never searches),
    the ablations dir, run_ablation_report's own pick, then every older
    */comparison_vs_discrete/ copy (newest first).
    """
    base = art / ds_tag
    c = [base / "discretization" / "discretization_per_seed.csv",
         base / "ablations" / "discretization_per_seed.csv"]
    try:
        p = rar._resolve_discretization(base / "ablations", None)
        if p is not None:
            c.append(Path(p))
    except Exception as ex:                                 # noqa: BLE001
        notes.append(f"NOTE run_ablation_report discretization lookup failed: {ex}")
    old = []
    for root in (base, art):
        for pat in ("*/comparison_vs_discrete/discretization_per_seed.csv",
                    "*/*/comparison_vs_discrete/discretization_per_seed.csv"):
            old += list(root.glob(pat))
    c += sorted(set(old), key=lambda q: q.stat().st_mtime, reverse=True)
    out = []
    for q in c:
        if q.is_file() and q.name == "discretization_per_seed.csv" and q not in out:
            out.append(q)
    return out


def _discretization(art, ds_tag, rar, explicit, notes):
    """
    ({(condition, branch, metric, seed_idx): value}, path) of the
    discretization_per_seed.csv behind Table 3's Discretization block. A
    candidate qualifies only if it holds all three DISC_COLUMNS conditions AND
    its five-seed iLISI means reproduce the printed 0.609 / 0.036 / 0.052 (niche)
    and 0.739 / 0.387 / 0.319 (cell), the same test _select_ts applies to
    timestamps. --discretization-csv (DISC_CSV= in the wrapper) is validated the
    same way.
    """
    want = _disc_expected()
    conds = {c for c, _b in want}
    if explicit:
        cands = [Path(explicit)]
        if not cands[0].is_file():
            notes.append(f"WARN --discretization-csv {explicit} not found")
            cands = []
    else:
        cands = _disc_candidates(art, ds_tag, rar, notes)
    has_conds = []
    for p in cands:
        d = _load_disc(p)
        have = {k[0] for k in d}
        if not conds <= have:
            notes.append(f"NOTE discretization candidate {p}: conditions "
                         f"{sorted(have)} lack {sorted(conds - have)}; skipped")
            continue
        has_conds.append((p, d))
        got = {(c, b): _f3(_mean([v for (c_, b_, m, _i), v in d.items()
                                  if c_ == c and b_ == b and m == "iLISI"]))
               for c, b in want}
        bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
        if not bad:
            notes.append(f"NOTE Discretization values from {p} (reproduces the "
                         f"printed iLISI)")
            return d, p
        notes.append(f"NOTE discretization candidate {p} does not reproduce the "
                     f"printed iLISI: {bad}")
    if has_conds:
        p, d = has_conds[0]
        notes.append(f"WARN DISCRETIZATION: no candidate reproduces the printed "
                     f"iLISI; using {p} anyway (flagged in PROVENANCE). Pass the "
                     f"right file with DISC_CSV= / --discretization-csv.")
        return d, p
    notes.append("WARN DISCRETIZATION: no discretization_per_seed.csv with the "
                 f"conditions {sorted(conds)} was found (tried: "
                 f"{[str(p) for p in cands] or 'none'}). The Discretization "
                 "columns fall back to the sweeps' per-seed iLISI/MMD and the Leiden "
                 "columns' NMI/ARI stay blank. Pass DISC_CSV= / --discretization-csv.")
    return {}, None


def _disc_seed_map(disc, v19_bsc, notes):
    """seed_idx (worker index) -> training seed, matched on the per-seed niche
    iLISI of 'SQUINT (codes)' (= s57_v19 neighborhood_emb); identity otherwise."""
    idxs = sorted({k[3] for k in disc if k[0] == "SQUINT (codes)" and k[1] == "niche"
                   and k[2] == "iLISI"})
    ref = {s: v for (s, k, m), v in v19_bsc.items()
           if k == "neighborhood_emb" and m == "iLISI"}
    mapping = {}
    for i in idxs:
        v = disc[("SQUINT (codes)", "niche", "iLISI", i)]
        hits = [s for s, rv in ref.items() if abs(rv - v) < 1e-9]
        mapping[i] = hits[0] if len(hits) == 1 else None
    if idxs and all(v is not None for v in mapping.values()) \
            and len(set(mapping.values())) == len(mapping):
        return mapping
    notes.append("NOTE discretization seed_idx could not be matched to training "
                 "seeds by value; assuming seed_idx == seed.")
    return {i: i for i in idxs}


def _ablation_rows(art, disc_arg, notes):
    pa = _repo_module("plot_ablations")
    sm = _repo_module("summarize_ablation_multiseed")
    rar = _repo_module("run_ablation_report")
    ds = DS_BY_TAG[ABLATION_DATASET]
    _check_layout(pa, notes)

    # Every (prefix, obsm keys) and the printed iLISI each must reproduce. The
    # Discretization columns come from a different producer
    # (compare_discrete_vs_continuous.py), so they only decide the sweep of a
    # prefix that appears nowhere else (s57_v33).
    checks, disc_checks = {}, {}
    for _table, layout in (("T3", T3_LAYOUT), ("S1", S1_LAYOUT)):
        for _block, axis_key, cols in layout:
            for col, prefix, _ref, pn, pc in cols:
                is_disc = axis_key == "discretization"
                keys = DISC_COLUMNS[col][1] if is_disc else EMB_KEYS
                item = (keys["niche"], keys["cell"], _split_printed(pn)[0],
                        _split_printed(pc)[0])
                lst = (disc_checks if is_disc else checks).setdefault(prefix, [])
                clash = [x for x in lst if x[:2] == item[:2] and x != item]
                if clash:
                    notes.append(f"WARN printed values for {prefix} {item[:2]} differ "
                                 f"between columns: {clash[0][2:]} vs {item[2:]}")
                if item not in lst:
                    lst.append(item)
    for prefix, lst in disc_checks.items():
        checks.setdefault(prefix, list(lst))

    sweep = {}
    for prefix, lst in sorted(checks.items()):
        ts, sel, n_sw = _ablation_ts(prefix, art, ds.tag, sm, lst)
        sweep[prefix] = (ts, sel)
        if n_sw > 1:
            notes.append(f"NOTE {prefix}: {n_sw} __multiseed sweeps match; using "
                         f"{ts.parent.name if ts else 'none'}")
        if sel not in ("latest",):
            notes.append(f"{'WARN' if 'UNMATCHED' in sel or sel == 'missing' else 'NOTE'}"
                         f" {prefix}: TS selection {sel} ({ts})")
    bsc = {p: (_batch_scores(ts / "metrics", notes) if ts else {})
           for p, (ts, _s) in sweep.items()}
    runs, run_status = {}, {}
    for p, (ts, _s) in sweep.items():
        runs[p], run_status[p] = _seed_run_dirs(ts) if ts else ({}, {})
    nid = {p: (_niche_id(ts / "metrics", rar, notes) if ts else ({}, set()))
           for p, (ts, _s) in sweep.items()}
    for p, rd in runs.items():
        if len(set(map(str, rd.values()))) < len(rd):
            notes.append(f"WARN {p}: run-dir collision in seed_run_index.csv")
    disc, disc_path = _discretization(art, ds.tag, rar, disc_arg, notes)
    seed_map = _disc_seed_map(disc, bsc.get("s57_v19_", {}), notes) if disc else {}
    inv_seed_map = {v: k for k, v in seed_map.items()}

    rows = []
    for table, layout in (("T3", T3_LAYOUT), ("S1", S1_LAYOUT)):
        for block, axis_key, cols in layout:
            ref_col, ref_prefix = next((c[0], c[1]) for c in cols if c[2])
            ref_labels = nid.get(ref_prefix, ({}, set()))[1]
            label = {
                "cell": next((x for x in pa.DEFAULT_CELL_LABEL_KEYS
                              if x in ref_labels), None),
                "niche": next((x for x in pa.DEFAULT_NICHE_LABEL_KEYS
                               if x in ref_labels), None)}
            code = {"cell": pa.CELL_CODE_KEY, "niche": pa.NICHE_CODE_KEY}
            for col, prefix, is_ref, pn, pc in cols:
                is_disc = axis_key == "discretization"
                cond, keys = (DISC_COLUMNS[col] if is_disc else (None, EMB_KEYS))
                ts, sel = sweep[prefix]
                for branch, printed in (("niche", pn), ("cell", pc)):
                    pv, pstars = _split_printed(printed)
                    pr = PRINTED_ABL[(table, block, col, branch)]
                    key = keys[branch]
                    vname = ts.parent.name if ts else ""
                    kind = ("continuous" if key.endswith("_latent")
                            or "continuous" in vname else "quantized")
                    method = prefix.rstrip("_") + (":latent" if key.endswith("_latent")
                                                   else "")
                    base = dict(table=table, dataset_tag=ds.tag, ds_short=ds.short,
                                dataset_label=ds.label, block=block, branch=branch,
                                column=col, method=method, role="ablation",
                                is_ref=is_ref, ref_column=ref_col, paper_visible=True,
                                variant_dir=str(ts.parent) if ts else "",
                                ts_dir=str(ts) if ts else "", ts_selection=sel,
                                obsm_key=key, batch_key=BATCH_KEY, emb_kind=kind,
                                emb_seed_invariant=False,
                                ilisi_max_cells=ILISI_MAX_CELLS,
                                subsample_seed=SUBSAMPLE_SEED,
                                printed_ilisi=pv, printed_stars=pstars,
                                printed_nmi=pr.get("nmi", ""),
                                printed_ari=pr.get("ari", ""),
                                printed_mmd=pr.get("mmd", ""))
                    if ts is None:
                        for s in SEEDS:
                            rows.append(_new_row(**base, seed=s, status="missing",
                                                 reason=f"no __multiseed sweep for "
                                                        f"{prefix}"))
                        continue
                    b = bsc[prefix]
                    if is_disc and disc:
                        seeds = sorted(seed_map.get(i, i) for (c_, br, m, i) in disc
                                       if c_ == cond and br == branch
                                       and m == "iLISI")
                        why = (f"condition {cond!r} / {branch} has no per-seed iLISI "
                               f"in {disc_path}")
                    else:
                        seeds = sorted({s for (s, k, m) in b
                                        if k == key and m == "iLISI"})
                        why = (f"no per-seed iLISI for {key} in {ts}/metrics/"
                               f"per_seed_batch_integration.csv")
                    if not seeds:
                        for s in SEEDS:
                            rows.append(_new_row(**base, seed=s, status="missing",
                                                 reason=why))
                        continue
                    nd = nid[prefix][0]
                    block_rows = []
                    for s in seeds:
                        if is_disc and disc:
                            i = inv_seed_map.get(s, s)
                            vals = {name: disc.get((cond, branch, m, i), float("nan"))
                                    for name, m in (("nmi", "NMI"), ("ari", "ARI"),
                                                    ("mmd", "MMD"),
                                                    ("ilisi_published", "iLISI"))}
                        else:
                            na = nd.get((s, code[branch], label[branch] or ""),
                                        (float("nan"), float("nan")))
                            if is_disc and col != "VQ (codes)":
                                # Leiden partitions, not codes: only the
                                # discretization csv holds their NMI/ARI.
                                na = (float("nan"), float("nan"))
                            vals = dict(nmi=na[0], ari=na[1],
                                        mmd=b.get((s, key, "MMD"), float("nan")),
                                        ilisi_published=b.get((s, key, "iLISI"),
                                                              float("nan")))
                        rd = runs[prefix].get(s)
                        block_rows.append(_new_row(
                            **base, **vals, seed=s, status="ok",
                            seed_status=run_status[prefix].get(s, ""),
                            run_dir=str(rd) if rd else "",
                            adata_path=str(rd / "predicted_adata.h5ad") if rd else "",
                            adata_scope="per_run" if rd else "none",
                            runtime_s=float("nan"),
                            mmd_producer=b.get((s, key, "MMD"), float("nan")),
                            run_dir_ilisi=_run_dir_ilisi(rd, key)))
                    prov = _f3(_mean([r["ilisi_published"] for r in block_rows])) == pv
                    other_ok, other_detail = _other_check(
                        {m: [r[m] for r in block_rows] for m in ("nmi", "ari", "mmd")},
                        pr)
                    if other_ok is False:
                        notes.append(f"WARN {table}/{block}/{col}/{branch}: kept "
                                     f"metrics do not reproduce the printed table: "
                                     f"{other_detail}")
                    for r in block_rows:
                        r["printed_provenance_ok"] = prov
                        r["printed_other_ok"] = other_ok
                        r["printed_other_detail"] = other_detail
                    rows += block_rows
    return rows


def _job_label(r):
    if r["role"] == "baseline":
        return Path(r["variant_dir"]).name.replace("baseline-", "")
    m = re.match(r"^(s\d+_v\d+)_", Path(r["variant_dir"]).name)
    return m.group(1) if m else "squint"


def _finalize(rows, sanity, probe_cache, probe_on):
    """Flags, compute decision and task / job ids for every manifest row."""
    pending = []
    for r in rows:
        r["row_id"] = "|".join(str(r[k]) for k in ("table", "ds_short", "block",
                                                     "column", "branch", "seed"))
        for k in ("fallback_suspected", "tie_affected", "control",
                  "unchanged_expected", "compute"):
            r[k] = False
        if r["status"] != "ok":
            r["reason"] = r["reason"] or r["status"]
            continue
        pr = _probe(r["adata_path"], r["obsm_key"], r["batch_key"], probe_cache,
                    probe_on)
        r.update(n_obs=pr["n_obs"], emb_dim=pr["emb_dim"],
                 probe_error=pr["probe_error"])
        n_obs = pr["n_obs"] if pr["n_obs"] > 0 else DS_BY_TAG[r["dataset_tag"]].n_obs
        maxc = int(_fnum(r["ilisi_max_cells"])) if _finite(r["ilisi_max_cells"]) else 0
        subsampled = maxc > 0 and n_obs > maxc
        r["cellset"] = f"sub{maxc}" if subsampled else "all"
        pub = _fnum(r["ilisi_published"])
        r["fallback_suspected"] = math.isfinite(pub) and pub >= 1.0
        r["tie_affected"] = r["emb_kind"] == "quantized" and not subsampled
        r["control"] = r["emb_kind"] == "quantized" and subsampled
        r["unchanged_expected"] = (r["emb_kind"] == "continuous"
                                   and not r["fallback_suspected"])
        if r["adata_path"]:
            lab = _job_label(r)
            r["job_id"] = f"{r['ds_short']}.{lab}"
            r["task_id"] = (f"{r['ds_short']}.{lab}.{r['obsm_key']}.s{r['seed']}."
                            f"{r['cellset']}.{_h8(r['adata_path'])}")
        method_dir = Path(r["variant_dir"]).name
        unaffected = not (r["fallback_suspected"] or r["tie_affected"]
                          or r["control"])
        # Decided from the file (probe), so a valid rerun is scored; the name
        # only decides when probing is off.
        not_on_disk = ((r["probe_error"].startswith(_NOT_ON_DISK) if probe_on
                        else method_dir in NOT_REEVALUABLE)
                       or (not r["adata_path"] and method_dir in NOT_REEVALUABLE))
        r["sanity_tier"] = ""
        if unaffected and r["role"] == "baseline" and not_on_disk:
            r["status"] = "not_reevaluable"
            r["reason"] = NOT_REEVALUABLE.get(method_dir, (
                f"embedding not on disk ({r['probe_error'] or 'not probed'}); the "
                f"published value is a scib-path score on a continuous embedding, "
                f"which neither fault affects, so it is carried unchanged"))
        elif not r["adata_path"]:
            r["reason"] = "no predicted_adata.h5ad resolved for this seed"
        elif r["probe_error"]:
            r["reason"] = f"adata unusable: {r['probe_error']}"
        elif r["adata_scope"] == "shared_toplevel" and not r["emb_seed_invariant"]:
            r["reason"] = ("only a shared top-level adata exists and this method's "
                           "embedding is trained per seed, so it is not this seed's "
                           "embedding")
        elif r["fallback_suspected"]:
            r["compute"], r["reason"] = True, "fallback value (>= 1): recompute"
        elif r["control"]:
            r["compute"], r["reason"] = True, ("NSCLC control (published on a "
                                               "shuffled 100k subset)")
        elif r["tie_affected"]:
            r["compute"], r["reason"] = True, "quantized, scored in file order"
        elif r["role"] == "latent_control":
            r["compute"], r["reason"] = True, "SQUINT continuous-latent control"
        else:
            pending.append(r)
    groups = {}
    for r in pending:
        groups.setdefault((r["dataset_tag"], r["variant_dir"], r["obsm_key"]),
                          []).append(r)
    for grp in groups.values():
        first = min(int(r["seed"]) for r in grp)
        for r in grp:
            # The tier is recorded so plan / compute --sanity can change the
            # choice later without rebuilding the manifest.
            r["sanity_tier"] = "seed0" if int(r["seed"]) == first else "other"
            pick = _sanity_pick(r["sanity_tier"], sanity)
            r["compute"] = pick
            r["reason"] = (f"continuous, unaffected: sanity recompute "
                           f"(--sanity {sanity})" if pick else
                           f"continuous, unaffected: published kept "
                           f"(--sanity {sanity})")
    return rows


def _sanity_pick(tier, mode):
    return bool(tier) and (mode == "all" or (mode == "seed0" and tier == "seed0"))


def _effective_compute(r, sanity=None):
    """The manifest's compute decision, with the continuous sanity rows re-decided
    when plan / compute are given --sanity."""
    tier = (r.get("sanity_tier") or "").strip()
    if sanity and tier:
        return _sanity_pick(tier, sanity)
    return _truthy(r["compute"])


def _manifest_sanity(rows):
    """The --sanity mode a manifest was built with (from its sanity rows)."""
    tiered = [r for r in rows if (r.get("sanity_tier") or "").strip()]
    if not tiered:
        return "n/a"
    picked = {r["sanity_tier"].strip() for r in tiered if _truthy(r["compute"])}
    return "all" if "other" in picked else ("seed0" if picked else "none")


def _coverage_report(rows, notes, out_path, sanity):
    L = []
    t = {}
    for r in rows:
        t.setdefault(r["table"], 0)
        t[r["table"]] += 1
    tasks = {r["task_id"] for r in rows if r["compute"]}
    jobs = {r["job_id"] for r in rows if r["compute"]}
    L.append("=" * 78)
    L.append(f"MANIFEST  {out_path or '(dry run: nothing written)'}")
    L.append("=" * 78)
    L.append(f"rows {len(rows)}  " + "  ".join(f"{k} {v}" for k, v in sorted(t.items()))
             + f"   paper-visible {sum(1 for r in rows if r['paper_visible'])}")
    L.append(f"sanity mode (continuous, unaffected rows): {sanity}  (plan / compute "
             f"--sanity, SANITY= in the wrapper, overrides it without a rebuild)")
    L.append(f"compute: {sum(1 for r in rows if r['compute'])} rows -> "
             f"{len(tasks)} unique tasks in {len(jobs)} jobs")
    L.append("")
    L.append(f"{'table':<5} {'dataset':<13} {'block':<16} {'column':<16} {'br':<5} "
             f"{'seeds':>5} {'cmp':>3} {'pub iLISI':>9} {'printed':>8} {'ok':>3} "
             f"{'ts':<10} flags")
    seen = {}
    for r in rows:
        k = (r["table"], r["ds_short"], r["block"], r["column"], r["branch"])
        seen.setdefault(k, []).append(r)
    for k, grp in seen.items():
        r0 = grp[0]
        flags = sorted({f for r in grp for f in ("fallback_suspected", "tie_affected",
                                                  "control") if r[f]})
        if r0["status"] != "ok":
            flags.append(r0["status"])
        ok = r0["printed_provenance_ok"]
        L.append(f"{k[0]:<5} {k[1]:<13} {k[2][:16]:<16} {k[3][:16]:<16} {k[4]:<5} "
                 f"{len(grp):>5} {sum(1 for r in grp if r['compute']):>3} "
                 f"{_f3(_mean([r['ilisi_published'] for r in grp])):>9} "
                 f"{r0['printed_ilisi'] or '-':>8} "
                 f"{('yes' if _truthy(ok) else 'NO') if ok != '' else '-':>3} "
                 f"{(r0['ts_selection'] or '-')[:10]:<10} {','.join(flags)}")
    excl = {}
    for r in rows:
        if not r["compute"]:
            excl.setdefault(r["reason"], []).append(r)
    L.append("")
    L.append("NOT RECOMPUTED (reason: rows)")
    for reason, grp in sorted(excl.items(), key=lambda x: -len(x[1])):
        who = sorted({f"{r['ds_short']}/{r['column']}" for r in grp})
        L.append(f"  {len(grp):>4}  {reason}")
        L.append(f"        {', '.join(who[:12])}{' ...' if len(who) > 12 else ''}")
    missing = [r for r in rows if r["status"] == "missing"
               or (r["status"] == "ok" and (r["probe_error"] or not r["adata_path"]))]
    if missing:
        L.append("")
        L.append("MISSING / UNUSABLE PATHS")
        for r in missing[:60]:
            L.append(f"  {r['row_id']}: {r['adata_path'] or r['variant_dir']}  "
                     f"{r['probe_error'] or r['reason']}")
    unmatched = sorted({f"{r['table']}/{r['ds_short']}/{r['block']}/{r['column']}"
                        for r in rows if r["paper_visible"]
                        and r["printed_provenance_ok"] is False})
    L.append("")
    L.append("PROVENANCE: groups whose files do NOT reproduce the printed iLISI "
             "(Table 3/S1) or NMI/ARI/MMD/iLISI (Table 1): "
             + (", ".join(unmatched) if unmatched else "none"))
    other = {}
    for r in rows:
        if r["paper_visible"] and r["printed_other_ok"] is False:
            other.setdefault(f"{r['table']}/{r['ds_short']}/{r['block']}/"
                             f"{r['column']}/{r['branch']}", r["printed_other_detail"])
    L.append("PROVENANCE: groups whose KEPT metrics (NMI/ARI/MMD, Table 1 also RT) do "
             "NOT reproduce the printed table: " + ("none" if not other else ""))
    for k, v in sorted(other.items()):
        L.append(f"  {k}: {v}")
    must = [r for r in rows if r["status"] == "ok" and not r["compute"]
            and (r["tie_affected"] or r["fallback_suspected"] or r["control"])]
    L.append("AFFECTED ROWS THAT CANNOT BE RECOMPUTED: "
             + (", ".join(r["row_id"] for r in must) if must else "none"))
    # The sweep aggregate must equal what each run wrote itself; a difference
    # means the run was re-predicted or re-scored after aggregation.
    drift = [r["row_id"] for r in rows if _finite(r["run_dir_ilisi"])
             and _finite(r["ilisi_published"])
             and abs(_fnum(r["run_dir_ilisi"]) - _fnum(r["ilisi_published"])) > 1e-9]
    L.append("RUN DIRS WHOSE OWN batch_integration_metrics.csv DIFFERS FROM THE SWEEP: "
             + (", ".join(drift[:20]) + (" ..." if len(drift) > 20 else "")
                if drift else "none"))
    if notes:
        L.append("")
        L.append("NOTES")
        L += [f"  {n}" for n in notes]
    return "\n".join(L) + "\n"


def _plan_lines(rows, out_dir, grain, include_done=False, sanity=None):
    jobs = {}
    seen = set()
    for r in rows:
        if not _effective_compute(r, sanity) or r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        job = r["job_id"] if grain == "method" else r["task_id"]
        done = (Path(out_dir) / "results" / f"{r['task_id']}.csv").exists()
        j = jobs.setdefault(job, {"ds": r["dataset_tag"], "n": 0, "todo": 0,
                                  "max_n": 0})
        j["n"] += 1
        j["todo"] += 0 if done else 1
        n_obs = int(_fnum(r["n_obs"])) if _finite(r["n_obs"]) else -1
        if n_obs <= 0:
            n_obs = DS_BY_TAG[r["dataset_tag"]].n_obs
        maxc = int(_fnum(r["ilisi_max_cells"])) if _finite(r["ilisi_max_cells"]) else 0
        j["max_n"] = max(j["max_n"], min(n_obs, maxc) if maxc else n_obs)
    out = []
    for job, j in sorted(jobs.items()):
        if j["todo"] or include_done:
            out.append(f"PLAN\t{job}\t{j['ds']}\t{j['n']}\t{j['todo']}\t{j['max_n']}")
    return out


def cmd_manifest(a):
    art, out = _resolve_dirs(a)
    notes = []
    probe_cache = {}
    rows = _t1_rows(art, notes)
    rows += _ablation_rows(art, a.discretization_csv, notes)
    rows = _finalize(rows, a.sanity, probe_cache, not a.no_probe)
    mpath = out / "manifest.csv"
    if a.dry_run:
        print(_coverage_report(rows, notes, None, a.sanity))
        print("\n".join(_plan_lines(rows, out, a.grain)))
        # The real run would refuse this out dir: say so now, not after the plan
        # has been read as good.
        problem = _out_dir_problem(out)
        if problem:
            print(f"!! {problem}", file=sys.stderr)
            return 2
        if mpath.exists():
            print(f"NOTE {mpath} already exists: the real run uses it as is "
                  f"(rebuild with manifest --force).", file=sys.stderr)
        return 0
    if mpath.exists() and not a.force:
        raise SystemExit(f"{mpath} exists. Use it (plan / compute / merge), or pass "
                         f"--force to move it to superseded/ and rebuild.")
    _claim_out_dir(out)
    for f in (mpath, out / "manifest_coverage.txt"):
        if f.exists():
            print(f"superseded {f} -> {_supersede(f)}")
    _write_new_csv(mpath, rows, MANIFEST_FIELDS)
    report = _coverage_report(rows, notes, mpath, a.sanity)
    _write_new_text(out / "manifest_coverage.txt", report)
    print(report)
    print(f"wrote {mpath}  ({len(rows)} rows)")
    print(f"wrote {out / 'manifest_coverage.txt'}")
    return 0


def cmd_plan(a):
    _art, out = _resolve_dirs(a)
    rows = _read_manifest(out)
    built = _manifest_sanity(rows)
    if a.sanity and a.sanity != built:
        print(f"# sanity rows: --sanity {a.sanity} overrides the manifest's {built}",
              file=sys.stderr)
    lines = _plan_lines(rows, out, a.grain, include_done=a.all, sanity=a.sanity)
    print("\n".join(lines) if lines else "# nothing to do: every task has a result")
    return 0


def _read_manifest(out):
    f = Path(out) / "manifest.csv"
    if not f.is_file():
        raise SystemExit(f"{f} not found; run the manifest step first.")
    return _read_csv(f)


# =============================================================================
# Compute
# =============================================================================
def _require_scib():
    """scib_metrics or nothing: the silent fallback is exactly fault 2."""
    try:
        import scib_metrics
        from scib_metrics import ilisi_knn                      # noqa: F401
        from scib_metrics.nearest_neighbors import NeighborsResults  # noqa: F401
    except Exception as ex:                                 # noqa: BLE001
        raise SystemExit(
            f"scib_metrics is required and could not be imported "
            f"({type(ex).__name__}: {ex}).\nThis script never falls back to the "
            f"unscaled inverse-Simpson mean that produced the out-of-range Table 1 "
            f"cells. Activate /nfs/team361/sb75/.venvs/squint (scib-metrics "
            f"{SCIB_VERSION_USED}).")
    return scib_metrics.__version__


def _versions():
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:                                   # noqa: BLE001
            return ""
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                       # noqa: BLE001
        sha = ""
    return {"scib_version": ver("scib_metrics"), "pynndescent_version": ver("pynndescent"),
            "numpy_version": ver("numpy"), "jax_version": ver("jax"),
            "python": platform.python_version(), "host": socket.gethostname(),
            "lsf_job_id": os.environ.get("LSB_JOBID", ""), "git_sha": sha}


def _read_h5_elem(node, log=print):
    """anndata's read_elem (as compute_label_metric.load_minimal uses it), with a
    plain-h5py fallback for categoricals and arrays (logged when used)."""
    try:
        return _repo_module("compute_label_metric")._read_elem()(node)
    except Exception as ex:                                 # noqa: BLE001
        import h5py
        log(f"    WARN read_elem failed on {node.name} ({type(ex).__name__}: {ex}); "
            f"reading it with plain h5py")
        if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
            cats = np.array([c.decode() if isinstance(c, bytes) else c
                             for c in node["categories"][()]], dtype=object)
            codes = node["codes"][()]
            return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
        if isinstance(node, h5py.Dataset):
            v = node[()]
            if getattr(v, "dtype", None) is not None and v.dtype.kind in ("S", "O"):
                v = np.array([x.decode() if isinstance(x, bytes) else x for x in v],
                             dtype=object)
            return v
        raise


_MISSING_LABELS = {"None", "nan", "NaN", "<NA>", ""}


def load_embedding(path, obsm_key, batch_key, n_batches=None, log=print):
    """
    One obsm array + obs[batch_key] as str, via h5py; X/uns never read. The array
    keeps its stored dtype when it is float32/float64 (as compute_inference_metrics
    reads it, which the MMD provenance replay needs); the kNN runs on a float32
    copy. Missing batch labels, or a batch count other than the dataset's, raise:
    either would change B in scib's (median - 1) / (B - 1) scaling.
    """
    import h5py
    clm = _repo_module("compute_label_metric")
    if obsm_key in clm.CODE_INDEX_KEYS or obsm_key.startswith(clm.CODE_INDEX_KEYS):
        raise ValueError(f"{obsm_key!r} is a code-index array, not a metric space")
    with h5py.File(path, "r") as h:
        if "obs" not in h:
            raise RuntimeError(f"{path}: no obs group (stub file), cannot be re-scored")
        if batch_key not in h["obs"]:
            raise RuntimeError(f"{path}: obs column {batch_key!r} absent")
        if "obsm" not in h or obsm_key not in h["obsm"]:
            raise RuntimeError(f"{path}: obsm {obsm_key!r} absent")
        X = np.asarray(_read_h5_elem(h["obsm"][obsm_key], log))
        b = _read_h5_elem(h["obs"][batch_key], log)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float32)
    batch = np.asarray(np.asarray(b, dtype=object).astype(str))
    if X.ndim != 2 or X.shape[0] != batch.shape[0]:
        raise RuntimeError(f"{path}: obsm {obsm_key!r} shape {X.shape} vs "
                           f"{batch.shape[0]} cells")
    if not np.isfinite(X).all():
        raise RuntimeError(f"{path}: obsm {obsm_key!r} has non-finite values")
    labels = np.unique(batch)
    bad = sorted(set(labels.tolist()) & _MISSING_LABELS)
    if bad:
        raise RuntimeError(f"{path}: obs[{batch_key!r}] has missing labels {bad} "
                           f"({int(np.isin(batch, bad).sum())} cells)")
    if n_batches is not None and labels.size != n_batches:
        raise RuntimeError(f"{path}: obs[{batch_key!r}] has {labels.size} batches "
                           f"{labels.tolist()[:10]}, the dataset has {n_batches}")
    return X, batch


def tie_stats(X, k):
    """Exact duplicate structure as the neighbour search sees it (float32)."""
    Xc = np.ascontiguousarray(X, dtype=np.float32) + np.float32(0.0)  # -0.0 -> 0.0
    _, inv, counts = np.unique(Xc, axis=0, return_inverse=True, return_counts=True)
    size = counts[np.asarray(inv).ravel()]
    n = Xc.shape[0]
    return {"n_unique_rows": int(counts.size),
            "frac_unique_rows": float(counts.size / n),
            "max_tie_group": int(counts.max()),
            "frac_cells_tied": float(np.mean(size > 1)),
            "frac_cells_tie_gt_k": float(np.mean(size > k))}


def nndescent_graph(X, k, random_state):
    """
    The paper's graph: NNDescent(emb, n_neighbors=min(k, n-1)).neighbor_graph,
    euclidean, query point included, as compute_inference_metrics.compute_ilisi
    and compute_label_metric.paper_graph build it. The only addition is an
    explicit random_state so a run can be repeated; it seeds the RP-tree
    initialisation and changes nothing else.
    """
    from pynndescent import NNDescent
    kk = int(min(k, X.shape[0] - 1))
    idx, dist = NNDescent(X, n_neighbors=kk, random_state=random_state).neighbor_graph
    return np.asarray(idx), np.asarray(dist)


def n_missing_nbrs(idx):
    """Unfilled NNDescent slots (-1). scib would read them as labels[-1], i.e. the
    last cell's batch, so they are counted for every graph and gated in merge."""
    return int(np.count_nonzero(np.asarray(idx) < 0))


def exact_graph(X, k, chunk=EXACT_CHUNK, rng=None, ties="random"):
    """
    Exact euclidean kNN, query point in column 0 of every row.

    ties="random" (the estimator used for results): every distance tie is broken
    INDEPENDENTLY FOR EACH QUERY ROW, uniformly at random. Each candidate gets
    the sort key (float32 distance bits << 32) | (random 32 bits >= 1), so the
    kk smallest keys are the nearest points with ties resolved by a fresh draw
    per (query, candidate) pair; self gets key 0 and is therefore always taken,
    first. Two members of one tie group thus get independent neighbour sets (a
    shared order would hand every member of the group the same neighbours, one
    draw per group instead of one per cell). The result does not depend on row
    order. `rng` seeds the draws (default_rng(0) if None).

    ties="index": ties broken by row index, self not forced. Not an estimator:
    only the selftest uses it, to show the mechanism of fault 1 (on batch-sorted
    rows every tied cell then takes the same, same-batch, neighbours).

    Distances are computed between the DISTINCT rows only and gathered back, so
    identical rows get bit-identical distances (a BLAS kernel could otherwise
    round two copies of the same vector differently and break a tie by noise).
    Tie-free inputs skip the random keys: self is forced first, and the rare
    exact distance tie between distinct points is resolved arbitrarily.
    """
    if ties not in ("random", "index"):
        raise ValueError(f"ties={ties!r}")
    rng = rng if rng is not None else np.random.default_rng(0)
    Xc = np.ascontiguousarray(X, dtype=np.float32) + np.float32(0.0)  # -0.0 -> 0.0
    n = Xc.shape[0]
    kk = int(min(k, n - 1))
    U, inv = np.unique(Xc, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    tie_free = U.shape[0] == n
    U64 = U.astype(np.float64)
    sq = np.einsum("ij,ij->i", U64, U64)
    idx_out = np.empty((n, kk), dtype=np.int64)
    dist_out = np.empty((n, kk), dtype=np.float32)
    for s in range(0, n, chunk):
        q = np.arange(s, min(n, s + chunk))
        m = q.size
        rows = np.arange(m)
        uq = inv[q]
        du = sq[uq][:, None] + sq[None, :] - 2.0 * (U64[uq] @ U64.T)
        np.maximum(du, 0.0, out=du)
        du[rows, uq] = 0.0
        d = du.astype(np.float32)[:, inv]                  # (m, n) squared
        del du
        d += np.float32(0.0)                                # -0.0 -> +0.0
        if ties == "index":
            kth = np.partition(d, kk - 1, axis=1)[:, kk - 1:kk]
            lt = d < kth
            need = kk - lt.sum(axis=1, keepdims=True)
            eq = d == kth
            take = lt | (eq & (np.cumsum(eq, axis=1, dtype=np.int32) <= need))
            r_, c_ = np.nonzero(take)                       # row-major, index order
            cols = c_.reshape(m, kk)
            dd = d[r_, c_].reshape(m, kk)
            order = np.lexsort((cols, dd), axis=1)          # distance, then index
            cols = np.take_along_axis(cols, order, axis=1)
        elif tie_free:
            d[rows, q] = -1.0                               # self strictly first
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
            dp = np.take_along_axis(d, part, axis=1)
            cols = np.take_along_axis(part, np.argsort(dp, axis=1, kind="stable"),
                                      axis=1)
            d[rows, q] = 0.0
        else:
            # Non-negative float32 bit patterns order like the values.
            key = d.view(np.uint32).astype(np.uint64)
            key <<= np.uint64(32)
            key |= rng.integers(1, 2 ** 32, size=(m, n), dtype=np.uint64)
            key[rows, q] = 0                                # self strictly first
            part = np.argpartition(key, kk - 1, axis=1)[:, :kk]
            kp = np.take_along_axis(key, part, axis=1)
            del key
            cols = np.take_along_axis(part, np.argsort(kp, axis=1), axis=1)
        idx_out[q] = cols
        dist_out[q] = np.sqrt(np.take_along_axis(d, cols, axis=1))
    return idx_out, dist_out


def scib_ilisi(idx, dist, batch):
    """scib_metrics.ilisi_knn with its defaults; must be a scalar in [0, 1]."""
    from scib_metrics import ilisi_knn
    from scib_metrics.nearest_neighbors import NeighborsResults
    raw = ilisi_knn(NeighborsResults(indices=idx, distances=dist), batch)
    if np.ndim(raw) != 0:
        raise RuntimeError(f"scib_metrics.ilisi_knn returned shape {np.shape(raw)}; "
                           f"expected the scaled median (a scalar). Refusing to "
                           f"average it into something else.")
    v = float(raw)
    if not math.isfinite(v) or v < -1e-6 or v > 1.0 + 1e-6:
        raise ValueError(f"scib iLISI {v} is outside [0, 1]")
    return v


def fallback_inline_mean(knn_indices, batch_labels):
    """
    The silent fallback, reproduced in behaviour to explain the four >= 1 Table 1
    cells: compute_inference_metrics._ilisi_inline (uniform weights over all k
    neighbours, self included, unscaled inverse Simpson in [1, n_batches]) and
    then np.mean over cells (compute_ilisi l.549). NOT an iLISI; never reported
    as one.
    """
    nbr = batch_labels[knn_indices]
    n, k = nbr.shape
    _u, dense = np.unique(nbr, return_inverse=True)
    dense = np.asarray(dense).reshape(n, k)
    counts = np.zeros((n, int(_u.size)), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(n), k), dense.ravel()), 1.0)
    p = counts / float(k)
    return float(np.mean(1.0 / np.sum(p ** 2, axis=1)))


def _producer_mmd(X, batch, log):
    """
    compute_inference_metrics.compute_mmd_comparable exactly as the SQUINT
    metrics step calls it (n_sub 2000, n_sigma 1000, rng default_rng(--seed=0),
    no pair cap) on the full embedding in its stored dtype. MMD is deterministic
    given embedding, row order and batch labels, so matching the per-run MMD
    proves the re-scored file is the published one. None if the helper is not
    importable (the check is then reported as skipped).
    """
    try:
        _ilisi_fn, mmd_fn = _repo_module("compute_label_metric").paper_helpers()
    except Exception as ex:                                 # noqa: BLE001
        log(f"    WARN producer MMD unavailable ({type(ex).__name__}: {ex}); "
            f"provenance check skipped")
        return None
    v = mmd_fn(X, batch, n_sub=MMD_N_SUB, n_sigma=MMD_N_SIGMA,
               rng=np.random.default_rng(SUBSAMPLE_SEED), max_pairs=None)
    return None if v is None else float(v)


def run_task(t, a, versions, log=print):
    """All protocols for one (adata, obsm key, cell set, training seed)."""
    started = _stamp()
    ds = DS_BY_TAG.get(t["dataset_tag"])
    X, batch = load_embedding(t["adata_path"], t["obsm_key"], t["batch_key"],
                              n_batches=ds.n_batches if ds else None, log=log)
    n_file = X.shape[0]
    emb_dtype = str(X.dtype)
    sha = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
    mmd = None
    t_mmd = time.time()
    if (t.get("role") in SQUINT_ROLES and _finite(t.get("mmd_producer"))
            and not a.no_mmd_check):
        mmd = _producer_mmd(X, batch, log)
    maxc = int(_fnum(t["ilisi_max_cells"])) if _finite(t["ilisi_max_cells"]) else 0
    if maxc and n_file > maxc:
        # compute_inference_metrics l.925-931, verbatim: rng(--seed=0).choice.
        base = np.random.default_rng(int(_fnum(t["subsample_seed"]))).choice(
            n_file, maxc, replace=False)
        cellset = f"rng{int(_fnum(t['subsample_seed']))}_choice_{maxc}"
    else:
        base = np.arange(n_file)
        cellset = "all"
    # pynndescent works in float32 whatever the input, so this loses nothing.
    Xb, bb = np.ascontiguousarray(X[base], dtype=np.float32), batch[base]
    del X
    ties = tie_stats(Xb, K)
    tied = ties["n_unique_rows"] < Xb.shape[0]
    n_b = int(np.unique(bb).size)
    if ds and n_b != ds.n_batches:
        raise RuntimeError(f"the scored cell set ({cellset}) holds {n_b} batches, the "
                           f"dataset has {ds.n_batches}")
    seed = int(_fnum(t["seed"]))
    perm_seeds = [seed * PERM_SEED_STRIDE + j for j in range(a.n_perm)]
    use_nnd = a.knn in ("nndescent", "both")
    use_exact = a.knn in ("exact", "both")
    log(f"  {t['task_id']}\n    {t['adata_path']}  [{t['obsm_key']}]  "
        f"{Xb.shape[0]}/{n_file} cells ({cellset}), dim {Xb.shape[1]}, {emb_dtype}, "
        f"{n_b} batches, {ties['n_unique_rows']} distinct rows "
        f"(largest tie group {ties['max_tie_group']}, "
        f"{ties['frac_cells_tie_gt_k']:.1%} of cells in groups > k)")
    common = dict(task_id=t["task_id"], job_id=t["job_id"], n_cells=int(Xb.shape[0]),
                  n_cells_file=int(n_file), cellset=cellset, n_batches=n_b, k=K,
                  perplexity=int(math.floor(min(K, Xb.shape[0] - 1) / 3)),
                  adata_path=t["adata_path"], obsm_key=t["obsm_key"],
                  batch_key=t["batch_key"], emb_dim=int(Xb.shape[1]),
                  emb_dtype=emb_dtype, emb_sha1=sha, started_utc=started, **ties,
                  **versions)
    out = []

    def rec(protocol, estimator, statistic, ps, rs, value, t0, n_miss=0):
        out.append(dict(common, protocol=protocol, estimator=estimator,
                        statistic=statistic, perm_seed=ps, nnd_random_state=rs,
                        value=float(value), seconds=round(time.time() - t0, 2),
                        n_missing_nbrs=int(n_miss)))
        log(f"    {protocol:<10} {estimator:<9} {statistic:<20} perm {ps:>4}  "
            f"{value:.6g}  ({time.time() - t0:.0f}s)"
            + (f"  [{n_miss} unfilled (-1) neighbours]" if n_miss else ""))

    if mmd is not None:
        rec("provenance", "producer", "mmd", -1, -1, mmd, t_mmd)
        want = _fnum(t.get("mmd_producer"))
        log(f"    MMD replay {mmd:.9g} vs per-run {want:.9g}: "
            f"{'match' if abs(mmd - want) <= MMD_TOL else 'MISMATCH'}")
    if not a.no_replay:
        # The published protocol (file order, or the rng(0).choice order for a
        # capped SQUINT run); its gap to the permuted value is the tie artefact
        # as the paper's estimator sees it.
        t0 = time.time()
        idx, dist = nndescent_graph(Xb, K, perm_seeds[0])
        nm = n_missing_nbrs(idx)
        rec("fileorder", "nndescent", "scib_ilisi", -1, perm_seeds[0],
            scib_ilisi(idx, dist, bb), t0, nm)
        t0 = time.time()
        rec("fileorder", "nndescent", "fallback_inline_mean", -1, perm_seeds[0],
            fallback_inline_mean(idx, bb), t0, nm)
        del idx, dist
    for j, ps in enumerate(perm_seeds):
        p = np.random.default_rng(ps).permutation(Xb.shape[0])
        Xp, bp = Xb[p], bb[p]
        if use_nnd:
            t0 = time.time()
            idx, dist = nndescent_graph(Xp, K, ps)
            rec("permuted", "nndescent", "scib_ilisi", ps, ps,
                scib_ilisi(idx, dist, bp), t0, n_missing_nbrs(idx))
            del idx, dist
        # The exact graph with per-row random ties does not depend on row order;
        # with ties, each repeat is a fresh tie draw (seeded by ps); without ties
        # one graph is enough.
        if use_exact and (tied or j == 0):
            t0 = time.time()
            idx, dist = exact_graph(Xp, K, a.exact_chunk,
                                    rng=np.random.default_rng(ps))
            rec("permuted", "exact", "scib_ilisi", ps, -1, scib_ilisi(idx, dist, bp),
                t0)
            del idx, dist
    return out


def cmd_compute(a):
    _need_numpy()
    scib_v = _require_scib()
    if a.require_scib_version and scib_v != a.require_scib_version:
        raise SystemExit(f"scib_metrics {scib_v} != required {a.require_scib_version}")
    if scib_v != SCIB_VERSION_USED:
        print(f"WARN scib_metrics {scib_v}; the CiLISI runs used {SCIB_VERSION_USED}",
              flush=True)
    import pynndescent                                          # noqa: F401  (fail early)
    _art, out = _resolve_dirs(a)
    rows = _read_manifest(out)
    want = [r for r in rows if _effective_compute(r, a.sanity)
            and (a.job == "ALL" or a.job in (r["job_id"], r["task_id"]))]
    tasks, seen = [], set()
    for r in want:
        if r["task_id"] not in seen:
            seen.add(r["task_id"])
            tasks.append(r)
    if not tasks:
        raise SystemExit(f"no compute task matches --job {a.job!r} in {out}/manifest.csv")
    versions = _versions()
    res_dir = out / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    print(f"job {a.job}: {len(tasks)} task(s)  knn={a.knn}  n_perm={a.n_perm}  "
          f"scib_metrics {scib_v}  pynndescent {versions['pynndescent_version']}",
          flush=True)
    n_ok = n_skip = n_fail = 0
    for t in tasks:
        f = res_dir / f"{t['task_id']}.csv"
        if f.exists() and not a.force:
            n_skip += 1
            print(f"  skip {t['task_id']} (result exists)", flush=True)
            continue
        try:
            out_rows = run_task(t, a, versions,
                                log=lambda s: print(s, flush=True))
        except Exception as ex:                             # noqa: BLE001
            n_fail += 1
            msg = (f"{t['task_id']}\n{t['adata_path']} [{t['obsm_key']}]\n"
                   f"{traceback.format_exc()}")
            print(f"  FAILED {t['task_id']}: {type(ex).__name__}: {ex}", flush=True)
            _write_new_text(res_dir / "failed" / f"{t['task_id']}.{_stamp()}."
                            f"{os.getpid()}.txt", msg)
            continue
        if f.exists() and not a.force:
            # Another job finished the same task meanwhile: keep its result.
            n_skip += 1
            print(f"  {t['task_id']}: written by another job meanwhile; kept",
                  flush=True)
            continue
        if f.exists():
            print(f"  superseded {f.name} -> {_supersede(f)}", flush=True)
        try:
            _write_new_csv(f, out_rows, RESULT_FIELDS)
        except FileExistsError:
            n_skip += 1
            print(f"  {t['task_id']}: written by another job meanwhile; kept",
                  flush=True)
            continue
        n_ok += 1
    print(f"done: {n_ok} computed, {n_skip} skipped (existing), {n_fail} failed",
          flush=True)
    return 1 if n_fail else 0


# =============================================================================
# Merge
# =============================================================================
_AGG_COLUMNS = ("task_id", "ilisi_nnd", "ilisi_nnd_sd", "n_perm_nnd", "ilisi_exact",
                "ilisi_exact_sd", "n_perm_exact", "ilisi_replay", "fallback_replay",
                "mmd_recheck", "n_missing_nbrs_max", "n_cells_scored",
                "n_batches_scored", "cellset_scored", "frac_unique_rows",
                "max_tie_group", "frac_cells_tie_gt_k", "emb_dtype", "emb_sha1",
                "scib_version", "pynndescent_version")
_CHECK_COLUMNS = ("replay_tol", "replay_ok", "mmd_ok", "fallback_ok",
                  "recompute_provenance")
REPLAY_TOL_FLOOR = 0.02
FALLBACK_TOL = 0.01


def _aggregate_results(res):
    import pandas as pd
    rows = []
    for tid, g in res.groupby("task_id", sort=False):
        def pick(protocol, estimator, statistic):
            m = ((g["protocol"] == protocol) & (g["estimator"] == estimator)
                 & (g["statistic"] == statistic))
            return pd.to_numeric(g.loc[m, "value"], errors="coerce").dropna()
        pn = pick("permuted", "nndescent", "scib_ilisi")
        pe = pick("permuted", "exact", "scib_ilisi")
        rn = pick("fileorder", "nndescent", "scib_ilisi")
        rf = pick("fileorder", "nndescent", "fallback_inline_mean")
        mm = pick("provenance", "producer", "mmd")
        f0 = g.iloc[0]
        miss = (pd.to_numeric(g["n_missing_nbrs"], errors="coerce")
                if "n_missing_nbrs" in g.columns else pd.Series(dtype=float))

        def mean(s):
            return float(s.mean()) if len(s) else float("nan")

        def sd(s):
            return float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        rows.append({
            "task_id": tid, "ilisi_nnd": mean(pn), "ilisi_nnd_sd": sd(pn),
            "n_perm_nnd": int(len(pn)), "ilisi_exact": mean(pe),
            "ilisi_exact_sd": sd(pe), "n_perm_exact": int(len(pe)),
            "ilisi_replay": mean(rn), "fallback_replay": mean(rf),
            "mmd_recheck": mean(mm),
            "n_missing_nbrs_max": float(miss.max()) if miss.notna().any()
            else float("nan"),
            "n_cells_scored": f0["n_cells"], "n_batches_scored": f0["n_batches"],
            "cellset_scored": f0["cellset"],
            "frac_unique_rows": f0["frac_unique_rows"],
            "max_tie_group": f0["max_tie_group"],
            "frac_cells_tie_gt_k": f0["frac_cells_tie_gt_k"],
            "emb_dtype": f0.get("emb_dtype", ""),
            "emb_sha1": f0["emb_sha1"], "scib_version": f0["scib_version"],
            "pynndescent_version": f0["pynndescent_version"]})
    return pd.DataFrame(rows)


_GKEY = ["table", "dataset_tag", "block", "column", "branch"]


def _provenance_checks(ps):
    """
    Per-row evidence that a recomputed value scores the embedding behind the
    published row:
      replay_ok    the file-order replay (published protocol) is within
                   max(0.02, seed s.d. of the published values of that cell) of
                   the published value;
      mmd_ok       the producer's MMD call on the file reproduces the per-run MMD
                   to 1e-6 (SQUINT runs; deterministic, so this proves embedding,
                   row order and batch labels at once);
      fallback_ok  for the >= 1 cells, the fallback statistic on the replay graph
                   reproduces the published value to 0.01.
    recompute_provenance summarises them (the MMD check outranks the replay one:
    a replay miss with a matching MMD is NNDescent's tie handling, not a wrong
    file).
    """
    sd_pub = ps.groupby(_GKEY)["ilisi_published"].transform(
        lambda s: s.std(ddof=1) if s.notna().sum() > 1 else float("nan"))
    tol, rep_ok, mmd_ok, fb_ok, prov = [], [], [], [], []
    for (_i, r), sdv in zip(ps.iterrows(), sd_pub):
        t = max(REPLAY_TOL_FLOOR, sdv if _finite(sdv) else 0.0)
        pub, rep = _fnum(r["ilisi_published"]), _fnum(r["ilisi_replay"])
        ro = ""
        if not r["fallback_suspected"] and math.isfinite(pub) and math.isfinite(rep):
            ro = abs(rep - pub) <= t
        mo = ""
        mp, mr = _fnum(r.get("mmd_producer")), _fnum(r.get("mmd_recheck"))
        if math.isfinite(mp) and math.isfinite(mr):
            mo = abs(mr - mp) <= MMD_TOL
        fo = ""
        fr = _fnum(r["fallback_replay"])
        if r["fallback_suspected"] and math.isfinite(pub) and math.isfinite(fr):
            fo = abs(fr - pub) < FALLBACK_TOL
        if not _finite(r["ilisi_recomputed"]):
            pv = ""
        elif r["fallback_suspected"]:
            pv = {True: "verified_fallback_replay",
                  False: "UNVERIFIED_fallback_replay"}.get(fo, "unchecked")
        elif mo is True:
            pv = "verified_mmd"
        elif mo is False:
            pv = "UNVERIFIED_mmd_mismatch"
        elif ro is True:
            pv = "verified_replay"
        elif ro is False:
            pv = "UNVERIFIED_replay_mismatch"
        else:
            pv = "unchecked"
        tol.append(t)
        rep_ok.append(ro)
        mmd_ok.append(mo)
        fb_ok.append(fo)
        prov.append(pv)
    ps["replay_tol"], ps["replay_ok"], ps["mmd_ok"] = tol, rep_ok, mmd_ok
    ps["fallback_ok"], ps["recompute_provenance"] = fb_ok, prov
    return ps


def _final_values(ps, policy):
    """
    ilisi_final + its source, per row (see the --policy help). Under --policy
    all, a cell's published values are replaced only when EVERY seed of that
    cell has a recompute, so a five-seed mean never mixes recomputed and
    published values (with --sanity seed0 only one seed of a continuous cell is
    recomputed, so such cells stay published).
    """
    complete = {}
    if policy == "all":
        for key, g in ps.groupby(_GKEY):
            ok = g[g["status"] == "ok"]
            complete[key] = bool(len(ok)) and bool(ok["ilisi_recomputed"].notna().all())
    picked = []
    for _, r in ps.iterrows():
        rec = r["ilisi_recomputed"]
        affected = (r["fallback_suspected"] or r["tie_affected"] or r["control"]
                    or r["role"] == "latent_control")
        src = ("recomputed_UNVERIFIED"
               if str(r["recompute_provenance"]).startswith("UNVERIFIED")
               else "recomputed")
        if r["status"] in ("not_run", "missing"):
            picked.append((float("nan"), r["status"]))
        elif r["status"] == "not_reevaluable":
            picked.append((r["ilisi_published"], "published_not_reevaluable"))
        elif affected and math.isfinite(rec):
            picked.append((rec, src))
        elif affected:
            # Never carry a value known to be wrong: a fallback or tie-biased number
            # without its recompute is left blank and listed in the gates.
            picked.append((float("nan"), "MISSING_RECOMPUTE"))
        elif (policy == "all" and complete.get(tuple(r[k] for k in _GKEY))
              and math.isfinite(rec)):
            picked.append((rec, src))
        else:
            picked.append((r["ilisi_published"], "published"))
    ps["ilisi_final"] = [v for v, _s in picked]
    ps["ilisi_final_source"] = [s for _v, s in picked]
    return ps


_SUMMARY_METRICS = (("NMI", "nmi"), ("ARI", "ari"), ("MMD", "mmd"), ("RT_s", "runtime_s"),
                    ("iLISI_published", "ilisi_published"),
                    ("iLISI_recomputed", "ilisi_recomputed"),
                    ("iLISI_exact", "ilisi_exact"), ("iLISI_replay", "ilisi_replay"),
                    ("iLISI_final", "ilisi_final"))
_PRINTED_COL = {"NMI": "printed_nmi", "ARI": "printed_ari", "MMD": "printed_mmd",
                "RT_s": "printed_rt", "iLISI_published": "printed_ilisi",
                "iLISI_final": "printed_ilisi"}


def _printed_parts(r, metric):
    """(printed value, printed stars) of a summary metric, '' when not printed."""
    col = _PRINTED_COL.get(metric)
    s = str(r.get(col, "") or "").strip() if col else ""
    if not s:
        return "", ""
    v, st = _split_printed(s)
    if col == "printed_ilisi":
        st = str(r.get("printed_stars", "") or "")
    return v, st


def _display(metric, mean):
    if not math.isfinite(mean):
        return ""
    if metric == "MMD":
        return f"{mean * 1e3:.2f}"
    if metric == "RT_s":
        return f"{mean / 60.0:.0f}"
    return f"{mean:.3f}"


def _summary(ps):
    """Five-seed mean / s.d. per paper cell and metric, Welch vs the reference."""
    import pandas as pd
    t1 = _repo_module("compute_table1_significance")
    sm = _repo_module("summarize_ablation_multiseed")
    groups = {}
    for _, r in ps.iterrows():
        groups.setdefault(tuple(r[k] for k in _GKEY), []).append(r)

    def vals(grp, col):
        return [float(r[col]) for r in grp if _finite(r[col])]
    out = []
    for key, grp in groups.items():
        r0 = grp[0]
        ref = groups.get((key[0], key[1], key[2], r0["ref_column"], key[4]), [])
        for metric, col in _SUMMARY_METRICS:
            v = vals(grp, col)
            if not v and metric not in ("iLISI_final", "iLISI_published"):
                continue
            rv = vals(ref, col)
            if r0["is_ref"] or not v or not rv:
                p, st = float("nan"), ""
            elif key[0] == "T1":
                p = t1.welch_p(v, rv)
                st = t1.stars(p)
                p = float("nan") if p is None else float(p)
            else:
                p = sm._pvalue(np.asarray(rv), np.asarray(v), "ttest")
                st = sm._stars(p)
            m = _mean(v)
            pv, pst = _printed_parts(r0, metric)
            out.append({
                "table": key[0], "dataset_tag": key[1], "ds_short": r0["ds_short"],
                "dataset_label": r0["dataset_label"], "block": key[2],
                "column": key[3], "branch": key[4], "method": r0["method"],
                "role": r0["role"], "is_ref": bool(r0["is_ref"]),
                "ref_column": r0["ref_column"], "paper_visible": bool(r0["paper_visible"]),
                "metric": metric, "n": len(v), "mean": m, "sd": _sd(v),
                "p_vs_ref": p, "stars": st, "display": _display(metric, m),
                "printed": pv, "printed_stars": pst,
                "matches_printed": (_display(metric, m) == pv) if pv and pv != "n/a"
                else ""})
    return pd.DataFrame(out)


def _star_str(st):
    return st if st in ("*", "**", "***") else ""


def _reason(r):
    if r["status"] == "not_run":
        return "not run (n/a)"
    if r["status"] == "missing":
        return "MISSING: run not found on disk"
    if r["status"] == "not_reevaluable":
        return "carried (no embedding on disk)"
    if r["fallback_suspected"]:
        return "FALLBACK statistic recomputed with scib"
    if r["control"]:
        return "NSCLC control (shuffled subset)"
    if r["tie_affected"]:
        return "quantized ties, file order"
    return "continuous, unaffected"


def _merge_dest(out):
    dest = out / "merged" / _stamp()
    if dest.exists():                     # two merges in the same second
        dest = dest.with_name(f"{dest.name}_{os.getpid()}")
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def cmd_merge(a):
    _need_numpy()
    import pandas as pd
    _art, out = _resolve_dirs(a)
    if not (out / "manifest.csv").is_file():
        raise SystemExit(f"{out}/manifest.csv not found")
    # Everything as text first: printed values must keep their typeset form
    # ("0.410", "0.030", "n/a"), which a float parse would turn into 0.41 / NaN.
    man = pd.read_csv(out / "manifest.csv", dtype=str, keep_default_na=False)
    for c in ("seed", "n_obs", "emb_dim", "nmi", "ari", "mmd", "runtime_s",
              "mmd_producer", "ilisi_published", "run_dir_ilisi", "ilisi_max_cells"):
        if c not in man.columns:
            man[c] = ""
        man[c] = pd.to_numeric(man[c], errors="coerce")
    man["seed"] = man["seed"].astype(int)
    for c in ("is_ref", "paper_visible", "fallback_suspected", "tie_affected",
              "control", "unchanged_expected", "compute", "emb_seed_invariant"):
        man[c] = man[c].map(_truthy)
    files = sorted((out / "results").glob("*.csv"))
    if files:
        res = pd.concat([pd.read_csv(f, dtype={"task_id": str, "job_id": str,
                                               "emb_sha1": str, "cellset": str,
                                               "emb_dtype": str})
                         for f in files], ignore_index=True)
        agg = _aggregate_results(res)
    else:
        agg = pd.DataFrame(columns=["task_id"])
    ps = man.merge(agg, on="task_id", how="left")
    for c in _AGG_COLUMNS:              # no results yet: keep the schema anyway
        if c not in ps.columns:
            ps[c] = float("nan")
    use = "ilisi_exact" if a.final_estimator == "exact" else "ilisi_nnd"
    ps["ilisi_recomputed"] = pd.to_numeric(ps[use], errors="coerce")
    ps["ilisi_recomputed_sd"] = pd.to_numeric(ps[use + "_sd"], errors="coerce")
    ps = _provenance_checks(ps)
    ps = _final_values(ps, a.policy)

    dest = _merge_dest(out)
    keep = [c for c in MANIFEST_FIELDS if c in ps.columns]
    extra = (["ilisi_recomputed", "ilisi_recomputed_sd", "ilisi_final",
              "ilisi_final_source"] + list(_CHECK_COLUMNS)
             + [c for c in _AGG_COLUMNS if c != "task_id"])
    ps_out = ps[keep + extra]
    _write_new_df(dest / "per_seed.csv", ps_out)
    summ = _summary(ps)
    _write_new_df(dest / "summary.csv", summ)
    diff = _diff_rows(ps, summ)
    _write_new_df(dest / "ilisi_diff.csv", pd.DataFrame(diff))
    _write_t1_tables(ps, dest / "t1_tables")
    _write_fig_tidy(ps, dest / "fig_tidy")
    report = _report(ps, summ, diff, a, len(files))
    _write_new_text(dest / "ilisi_diff_report.txt", report)
    print(report)
    print(f"wrote {dest}/per_seed.csv, summary.csv, ilisi_diff.csv, "
          f"ilisi_diff_report.txt, t1_tables/, fig_tidy/")
    print(f"Table 1 stars from the new csvs:\n  python analysis/benchmarking/"
          f"compute_table1_significance.py --tables-dir {dest / 't1_tables'}")
    print(f"Fig S3 b,d panels from the new values:\n  python {SCRIPT_REL} figures "
          f"--merged {dest}")
    return 0


def _t1_visible(ps):
    return ps[(ps["table"] == "T1") & ps["paper_visible"]
              & ~ps["status"].isin(["not_run", "missing"])]


def _write_t1_tables(ps, dest):
    """Drop-in copies of mlcb2026_paper/tables/*_identification_benchmark_<ds>.csv
    with iLISI replaced by ilisi_final (same columns, same row order)."""
    dest.mkdir(parents=True, exist_ok=False)
    sub = _t1_visible(ps)
    for ds in DATASETS:
        for block, (stem, cols) in T1_CSV.items():
            g = sub[(sub["dataset_tag"] == ds.tag) & (sub["block"] == block)]
            if g.empty:
                continue
            nm = "Niche" if block == "niche" else "Cell-type"
            rows = [{"method": r["method"], "seed": int(r["seed"]),
                     f"{nm} NMI": float(r["nmi"]), f"{nm} ARI": float(r["ari"]),
                     "MMD": float(r["mmd"]), "Runtime (s)": float(r["runtime_s"]),
                     "iLISI": float(r["ilisi_final"])}
                    for _, r in g.sort_values(["method", "seed"]).iterrows()]
            _write_new_csv(dest / f"{stem}_{ds.short}.csv", rows, cols)


def _write_fig_tidy(ps, dest):
    """
    The tidy (method, seed, metric, value) frame each Table 1 benchmark figure
    (Fig S3 b,d for Mouse Brain) is drawn from, in the schema the plot modules'
    make_figure takes (their load_method_metrics + load_method_runtime_rows
    output), with iLISI = ilisi_final. `figures` renders them.
    """
    dest.mkdir(parents=True, exist_ok=False)
    sub = _t1_visible(ps)
    for ds in DATASETS:
        for block, (stem, _cols) in T1_CSV.items():
            g = sub[(sub["dataset_tag"] == ds.tag) & (sub["block"] == block)]
            if g.empty:
                continue
            nm = "Niche" if block == "niche" else "Cell-type"
            rows = []
            for _, r in g.sort_values(["method", "seed"]).iterrows():
                for metric, v in ((f"{nm} NMI", r["nmi"]), (f"{nm} ARI", r["ari"]),
                                  ("iLISI", r["ilisi_final"]), ("MMD", r["mmd"]),
                                  ("Runtime (s)", r["runtime_s"])):
                    if _finite(v):
                        rows.append({"method": r["method"], "seed": int(r["seed"]),
                                     "metric": metric, "value": float(v)})
            _write_new_csv(dest / f"{stem}_{ds.short}_tidy.csv", rows,
                           ["method", "seed", "metric", "value"])


def _diff_rows(ps, summ):
    """One row per paper-visible iLISI cell: printed -> new."""
    rows = []
    s = summ[summ["paper_visible"]]
    fin = s[s["metric"] == "iLISI_final"].set_index(_GKEY)
    pub = s[s["metric"] == "iLISI_published"].set_index(_GKEY)
    for key, r in fin.iterrows():
        g = ps[(ps["table"] == key[0]) & (ps["dataset_tag"] == key[1])
               & (ps["block"] == key[2]) & (ps["column"] == key[3])
               & (ps["branch"] == key[4])]
        r0 = g.iloc[0]
        old = r["printed"] + (r["printed_stars"] if key[0] != "T1" else "")
        new = r["display"] + (_star_str(r["stars"]) if key[0] != "T1" else "")
        pm = pub.loc[key]["mean"] if key in pub.index else float("nan")
        rows.append({
            "table": key[0], "dataset": r0["dataset_label"], "block": key[2],
            "column": key[3], "branch": key[4], "old_printed": old, "new": new,
            "changed": old != new, "published_mean": pm, "new_mean": r["mean"],
            "delta": (r["mean"] - pm) if _finite(pm) and _finite(r["mean"])
            else float("nan"),
            "new_p_vs_ref": r["p_vs_ref"], "reason": _reason(r0),
            "final_source": ",".join(sorted(set(g["ilisi_final_source"]))),
            "provenance": ",".join(sorted(set(g["recompute_provenance"]) - {""})),
            "perm_sd_mean": _mean(g["ilisi_recomputed_sd"]),
            "exact_mean": _mean(g["ilisi_exact"]),
            "replay_mean": _mean(g["ilisi_replay"]),
            "fallback_replay_mean": _mean(g["fallback_replay"])})
    return rows


def _report(ps, summ, diff, a, n_files):
    L = []
    W = L.append
    W("=" * 78)
    W("CORRECTED GLOBAL iLISI: printed -> new")
    W(f"policy={a.policy}  estimator={a.final_estimator}  result files={n_files}  "
      f"scib_metrics={','.join(sorted(set(ps['scib_version'].dropna().astype(str))))}")
    W("=" * 78)
    if a.final_estimator == "exact":
        W("NOTE --final-estimator exact: the values below are the exact kNN with "
          "per-row random tie-breaking, NOT the paper's NNDescent estimator. Say so "
          "wherever they are used.")
    W("")
    W("GATES")
    # 1. NSCLC control
    for block in ("niche", "cell_type"):
        g = ps[(ps["table"] == "T1") & (ps["dataset_tag"] == "chl59-2b_1p")
               & (ps["block"] == block) & (ps["role"] == "squint")]
        pub, new = list(g["ilisi_published"]), list(g["ilisi_recomputed"])
        mp, sp, mn = _mean(pub), _sd(pub), _mean(new)
        if not _finite(mn):
            W(f"  [NOT RUN] NSCLC control ({block}): no recomputed values")
            continue
        tol = max(sp if _finite(sp) else 0.0, 0.002)
        status = "PASS" if abs(mn - mp) <= tol else "STOP"
        W(f"  [{status}] NSCLC control ({block}): published {mp:.4f} +/- {sp:.4f}, "
          f"recomputed {mn:.4f}; |delta| {abs(mn - mp):.4f} vs 1 s.d. "
          f"(floor 0.002) {tol:.4f}")
        if status == "STOP":
            W("         STOP: a pipeline fault, not the tie artefact. Investigate "
              "before using any recomputed value.")
    # 2. provenance of the re-scored files: producer MMD replay (SQUINT runs)
    g = ps[ps["mmd_ok"].isin([True, False])].drop_duplicates("task_id")
    if g.empty:
        W("  [NOT RUN] MMD provenance replay: no SQUINT task has one (helper not "
          "importable, --no-mmd-check, or no results yet)")
    else:
        bad = g[g["mmd_ok"] == False]                      # noqa: E712
        W(f"  [{'OK' if bad.empty else 'CHECK'}] MMD provenance replay (SQUINT "
          f"files; producer call, tol {MMD_TOL:g}): {len(g) - len(bad)}/{len(g)} "
          f"tasks reproduce the per-run MMD"
          + ("" if bad.empty else "; NOT: " + ", ".join(bad["task_id"].head(10))))
    # 3. file-order replay reproduces the published value
    g = ps[ps["replay_ok"].isin([True, False])]
    for kind in ("quantized", "continuous"):
        gg = g[g["emb_kind"] == kind]
        if gg.empty:
            continue
        d = (gg["ilisi_replay"] - gg["ilisi_published"]).abs()
        bad = gg[gg["replay_ok"] == False]                 # noqa: E712
        W(f"  [{'OK' if bad.empty else 'CHECK'}] file-order replay vs published "
          f"({kind}, {len(gg)} rows; tol max({REPLAY_TOL_FLOOR}, cell s.d.)): median "
          f"|d| {d.median():.4f}, max {d.max():.4f}; {len(bad)} outside")
        for _, r in bad.drop_duplicates("task_id").head(10).iterrows():
            why = ("embedding verified by MMD: NNDescent tie handling depends on "
                   "random_state" if str(r["mmd_ok"]) == "True" else
                   "provenance UNVERIFIED (re-predicted or wrong file?)")
            W(f"         {r['task_id']}: published {r['ilisi_published']:.4f} replay "
              f"{r['ilisi_replay']:.4f} (tol {r['replay_tol']:.4f}); {why}")
    unv = ps[ps["ilisi_final_source"] == "recomputed_UNVERIFIED"]
    W(f"  [{'OK' if unv.empty else 'CHECK'}] recomputed values with unverified "
      f"provenance (ilisi_final_source = recomputed_UNVERIFIED): {len(unv)}"
      + ("" if unv.empty else ": " + ", ".join(unv["row_id"].head(12))))
    # 4. fault 1 with the paper's estimator: permuted minus file order
    g = ps[ps["tie_affected"] & ps["ilisi_replay"].notna()
           & ps["ilisi_nnd"].notna()]
    if g.empty:
        W("  [NOT RUN] fault 1 (tie artefact) with NNDescent: no tie-affected "
          "results yet")
    for (ds, key), gg in g.groupby(["dataset_label", "obsm_key"]):
        d = gg["ilisi_nnd"] - gg["ilisi_replay"]
        psd = _mean(gg["ilisi_nnd_sd"])
        noise = max(0.01, 2.0 * psd if _finite(psd) else 0.0)
        state = ("REPRODUCED" if d.mean() > noise else
                 "REVERSED (file order scores HIGHER)" if d.mean() < -noise else
                 "NOT REPRODUCED")
        W(f"  [FINDING] fault 1 {state} with the paper's estimator (NNDescent) on "
          f"{ds} {key}: permuted - file order = {d.mean():+.4f} (min {d.min():+.4f}, "
          f"max {d.max():+.4f}, {len(gg)} rows; noise bound {noise:.4f})")
        if state != "REPRODUCED":
            W("         The published values are then NOT understated by row order "
              "under NNDescent; any change vs published comes from NNDescent noise "
              "or provenance (see the replay and MMD gates), not from fault 1. Do "
              "not present the exact-kNN values as a correction of the NNDescent "
              "ones unless --final-estimator exact is chosen deliberately.")
        ex = gg["ilisi_exact"].dropna()
        if len(ex):
            W(f"         exact kNN (per-row random ties) on the same rows: mean "
              f"{ex.mean():.4f} vs NNDescent permuted {gg['ilisi_nnd'].mean():.4f}")
    # 5. fallback reproduction
    g = ps[ps["fallback_suspected"] & (ps["table"] == "T1")]
    for (ds, m), gg in g.groupby(["dataset_label", "method"]):
        fr, pb = _mean(gg["fallback_replay"]), _mean(gg["ilisi_published"])
        ok = _finite(fr) and abs(fr - pb) < FALLBACK_TOL
        W(f"  [{'OK' if ok else 'CHECK'}] fallback replay {ds} {m}: published "
          f"{pb:.4f}, fallback statistic on the replay graph {_f3(fr) or 'n/a'}; "
          f"scib recompute {_f3(_mean(gg['ilisi_recomputed'])) or 'n/a'}")
    # 6. estimator agreement
    g = ps[ps["ilisi_exact"].notna() & ps["ilisi_nnd"].notna()]
    if not g.empty:
        d = (g["ilisi_exact"] - g["ilisi_nnd"]).abs()
        bad = g[d > 0.02]
        W(f"  [{'OK' if bad.empty else 'CHECK'}] exact (per-row random ties) vs "
          f"NNDescent, permuted rows: median |d| {d.median():.4f}, max "
          f"{d.max():.4f}; {len(bad)} rows > 0.02")
        for _, r in bad.drop_duplicates("task_id").head(10).iterrows():
            W(f"         {r['task_id']}: nnd {r['ilisi_nnd']:.4f} exact "
              f"{r['ilisi_exact']:.4f}")
    # 7. unfilled NNDescent neighbours
    g = ps[ps["n_missing_nbrs_max"].notna()].drop_duplicates("task_id")
    bad = g[g["n_missing_nbrs_max"] > 0]
    if not g.empty:
        W(f"  [{'OK' if bad.empty else 'CHECK'}] NNDescent graphs with unfilled (-1) "
          f"neighbours: {len(bad)} task(s)"
          + ("" if bad.empty else ": " + ", ".join(bad["task_id"].head(8))
             + " (scib reads -1 as the last cell's batch)"))
    # 8. seed invariance
    g = ps[(ps["role"] == "baseline") & ps["emb_seed_invariant"] & ps["emb_sha1"].notna()
           & (ps["adata_scope"] == "per_seed")]
    for (ds, m), gg in g.groupby(["dataset_label", "method"]):
        n = gg["emb_sha1"].nunique()
        W(f"  [INFO] {ds} {m}: {n} distinct embedding(s) across "
          f"{gg['seed'].nunique()} seeds ({'seed-invariant' if n == 1 else 'VARIES'})")
    # 9. kind vs measured ties
    g = ps[ps["frac_unique_rows"].notna()].drop_duplicates("task_id")
    fu = pd_num(g["frac_unique_rows"])
    odd = g[((g["emb_kind"] == "continuous") & (fu < 0.99))
            | ((g["emb_kind"] == "quantized") & (fu > 0.9))]
    W(f"  [{'OK' if odd.empty else 'CHECK'}] emb_kind vs measured ties: "
      f"{len(odd)} task(s) disagree"
      + ("" if odd.empty else ": " + ", ".join(odd["task_id"].head(8))))
    # 10. coverage
    miss = ps[ps["ilisi_final_source"] == "MISSING_RECOMPUTE"]
    W(f"  [{'OK' if miss.empty else 'CHECK'}] affected rows without a recompute: "
      f"{len(miss)}" + ("" if miss.empty else ": " + ", ".join(miss["row_id"].head(12))))
    gone = ps[(ps["status"] == "missing") & ps["paper_visible"]]
    W(f"  [{'OK' if gone.empty else 'CHECK'}] paper-visible rows whose run was not "
      f"found: {len(gone)}"
      + ("" if gone.empty else ": " + ", ".join(gone["row_id"].head(12))))
    # 11. provenance of the selection and of every kept metric
    bad = sorted({f"{r['table']}/{r['ds_short']}/{r['block']}/{r['column']}"
                  for _, r in ps.iterrows() if r["paper_visible"]
                  and str(r["printed_provenance_ok"]) == "False"})
    W(f"  [{'OK' if not bad else 'CHECK'}] files reproduce the printed iLISI "
      f"(and Table 1 NMI/ARI/MMD): " + ("all" if not bad else "NOT for " + ", ".join(bad)))
    s = summ[summ["paper_visible"] & summ["metric"].isin(["NMI", "ARI", "MMD", "RT_s"])
             & (summ["matches_printed"] == False)]                 # noqa: E712
    W(f"  [{'OK' if s.empty else 'CHECK'}] kept metrics (NMI/ARI/MMD, Table 1 RT) "
      f"reproduce every printed value: "
      + ("all" if s.empty else f"NOT for {len(s)} cell(s)"))
    for _, r in s.head(15).iterrows():
        W(f"         {r['table']}/{r['ds_short']}/{r['block']}/{r['column']}/"
          f"{r['branch']} {r['metric']}: {r['display']} vs printed {r['printed']}")
    # 12. printed stars reproduced by the published per-seed values
    s = summ[(summ["metric"] == "iLISI_published") & (summ["table"] != "T1")
             & ~summ["is_ref"]]
    mism = [f"{r['table']}/{r['block']}/{r['column']}/{r['branch']} printed "
            f"'{r['printed_stars']}' vs '{_star_str(r['stars'])}'"
            for _, r in s.iterrows() if r["printed_stars"] != _star_str(r["stars"])]
    W(f"  [{'OK' if not mism else 'CHECK'}] Table 3/S1 iLISI stars recomputed from "
      f"the published per-seed values match the printed ones"
      + ("" if not mism else ": " + "; ".join(mism[:8])))
    W("")
    for table, title in (("T1", "TABLE 1 iLISI (five-seed means; iLISI carries no "
                                "stars in Table 1)"),
                         ("T3", "TABLE 3 iLISI (stars: Welch vs the shaded column)"),
                         ("S1", "TABLE S1 iLISI (stars: Welch vs the shaded column)")):
        W(title)
        for d in diff:
            if d["table"] != table:
                continue
            flag = "*" if d["changed"] else " "
            extra = ""
            if _finite(d["exact_mean"]):
                extra += f" exact {d['exact_mean']:.3f}"
            if _finite(d["perm_sd_mean"]):
                extra += f" perm-sd {d['perm_sd_mean']:.3f}"
            if "UNVERIFIED" in d["provenance"]:
                extra += " PROVENANCE-UNVERIFIED"
            W(f" {flag} {d['dataset'][:13]:<13} {d['block'][:15]:<15} "
              f"{d['column'][:14]:<14} {d['branch']:<5} {d['old_printed']:>9} -> "
              f"{d['new'] or 'n/a':<9} [{d['reason']}]{extra}")
        W("")
    W("TEXT AND AUTHOR-RESPONSE QUOTES")
    fin = summ[summ["metric"] == "iLISI_final"].set_index(
        ["table", "ds_short", "block", "column", "branch"])

    def val(c):
        return float(fin.loc[c]["mean"]) if c in fin.index else float("nan")

    def stars(c):
        return _star_str(fin.loc[c]["stars"]) if c in fin.index else ""
    for quote, where, how, cells in TEXT_QUOTES:
        v = [val(c) for c in cells]
        if not all(math.isfinite(x) for x in v):
            new = "n/a (not computed)"
        elif how == "one":
            new = f"{v[0]:.3f}"
        elif how == "arrow":
            new = f"{v[0]:.2f} -> {v[1]:.2f}"
        elif how == "max":
            new = f"<={max(v):.2f}"
        else:
            new = f"{min(v):.2f}-{max(v):.2f}"
        W(f"  {quote:<14} -> {new:<16} {where}")
    W("")
    W("CLAIMS THAT DEPEND ON THE NEW iLISI (holds / FAILS / n/a)")
    mb = "mmb0-1b_smb1"

    def claim(text, cells, test):
        v = [val(c) for c in cells]
        if not all(math.isfinite(x) for x in v):
            W(f"  [n/a  ] {text}")
            return
        ok, detail = test(v, [stars(c) for c in cells])
        W(f"  [{'holds' if ok else 'FAILS'}] {text}: {detail}")

    T3 = lambda blk, col, br: ("T3", mb, blk, col, br)          # noqa: E731
    S1 = lambda blk, col, br: ("S1", mb, blk, col, br)          # noqa: E731
    claim("main text: FiLM gives the strongest niche mixing (iLISI 0.61 vs <=0.36)",
          [T3("Coupling", c, "niche") for c in ("FiLM", "Decpl", "Trunk", "X-st",
                                                 "Aff")],
          lambda v, s: (v[0] > max(v[1:]),
                        f"FiLM {v[0]:.3f} vs max other {max(v[1:]):.3f}"))
    claim("main text: removing the adjacency loss weakens niche iLISI",
          [T3("Adjacency", "w/", "niche"), T3("Adjacency", "w/o", "niche")],
          lambda v, s: (v[1] < v[0], f"w/o {v[1]:.3f}{s[1]} vs w/ {v[0]:.3f}"))
    claim("main text: within-section contrastive collapses cell iLISI (0.74 -> 0.49)",
          [T3("Contrastive", "Cross", "cell"), T3("Contrastive", "Within", "cell")],
          lambda v, s: (v[1] < v[0] and bool(s[1]),
                        f"Within {v[1]:.3f}{s[1]} vs Cross {v[0]:.3f}"))
    claim("main text: removing the contrastive preserves integration (None not "
          "significantly below Cross, niche and cell)",
          [T3("Contrastive", "Cross", "niche"), T3("Contrastive", "None", "niche"),
           T3("Contrastive", "Cross", "cell"), T3("Contrastive", "None", "cell")],
          lambda v, s: ((v[1] >= v[0] or not s[1]) and (v[3] >= v[2] or not s[3]),
                        f"niche None {v[1]:.3f}{s[1]} vs {v[0]:.3f}; cell None "
                        f"{v[3]:.3f}{s[3]} vs {v[2]:.3f}"))
    claim("main text + author response: removing the decoder covariate collapses "
          "cell iLISI (0.739 -> 0.030) and niche iLISI (0.609 -> 0.057)",
          [T3("Decoder cov.", "w/", "cell"), T3("Decoder cov.", "w/o", "cell"),
           T3("Decoder cov.", "w/", "niche"), T3("Decoder cov.", "w/o", "niche")],
          lambda v, s: (v[1] < 0.5 * v[0] and bool(s[1]) and v[3] < 0.5 * v[2]
                        and bool(s[3]),
                        f"cell {v[1]:.3f}{s[1]} vs {v[0]:.3f}; niche {v[3]:.3f}{s[3]} "
                        f"vs {v[2]:.3f}"))
    claim("main text: Leiden on a continuous latent is far less integrated (niche "
          "iLISI 0.61 -> 0.04-0.05)",
          [T3("Discretization", "VQ (codes)", "niche"),
           T3("Discretization", "VQ (Leiden)", "niche"),
           T3("Discretization", "VAE (Leiden)", "niche")],
          lambda v, s: (v[0] > 2 * max(v[1:]),
                        f"codes {v[0]:.3f} vs Leiden {v[1]:.3f}{s[1]} / "
                        f"{v[2]:.3f}{s[2]}"))
    claim("author response R3-WC: decoupling drops niche iLISI (0.609 -> 0.313)",
          [T3("Coupling", "FiLM", "niche"), T3("Coupling", "Decpl", "niche")],
          lambda v, s: (v[1] < v[0] and bool(s[1]),
                        f"Decpl {v[1]:.3f}{s[1]} vs FiLM {v[0]:.3f}"))
    claim("Table S1 caption + author response R4-W4: niche iLISI moves opposite to "
          "niche NMI (k=8 above, k=24 below k=16)",
          [S1("Neighbours", "16", "niche"), S1("Neighbours", "8", "niche"),
           S1("Neighbours", "24", "niche")],
          lambda v, s: (v[1] > v[0] > v[2],
                        f"k=8 {v[1]:.3f}, k=16 {v[0]:.3f}, k=24 {v[2]:.3f}"))
    nb = summ[(summ["table"] == "S1") & (summ["block"] == "Neighbours")
              & (summ["metric"] == "iLISI_final") & ~summ["is_ref"]]
    sig = [f"{r['column']}/{r['branch']}{_star_str(r['stars'])}" for _, r in nb.iterrows()
           if _star_str(r["stars"])]
    W("  [" + ("holds" if not sig else "FAILS") + "] Table S1 caption: no iLISI "
      "difference in the Neighbours block is significant"
      + ("" if not sig else ": " + ", ".join(sig)))
    for ds_short, block, m in (("mmb0-1b_smb1", "niche", "GraphST"),
                               ("xhs1000", "niche", "GraphST"),
                               ("mmb0-1b_smb1", "cell_type", "scGPT"),
                               ("mmb0-1b_smb1", "cell_type", "scGPT-spatial")):
        br = "niche" if block == "niche" else "cell"
        x, sq = val(("T1", ds_short, block, m, br)), val(("T1", ds_short, block,
                                                          "SQUINT", br))
        if math.isfinite(x) and math.isfinite(sq):
            W(f"  [{'holds' if x < sq else 'FAILS'}] {m} below SQUINT ({ds_short}, "
              f"{block}; the paper's 'GraphST's far higher iLISI' is then an "
              f"artefact): {x:.3f} vs {sq:.3f}")
    W("")
    W("DEPENDENT OUTPUTS TO REGENERATE")
    W("  Table 1 / 3 / S1 iLISI rows and their stars (summary.csv, t1_tables/); the "
      "quotes above in the main text, the Table S1 caption and the response.")
    W("  Fig S3 b,d (supplement: 'iLISI ... is reported in Fig S3'): the panels "
      "read iLISI per seed; fig_tidy/ holds the corrected frames, `figures "
      "--merged <this folder>` re-renders them with the plot modules' make_figure.")
    W("")
    W("NOTES")
    W("  Novae: when its embedding is not on disk, its published values (scib path, "
      "continuous) are carried unchanged; a valid rerun file is scored instead.")
    W("  GraphST CosMx NSCLC: never ran (PASTE OOM); stays n/a.")
    W("  The naive rescale (x-1)/(B-1) of the fallback cells is NOT the scib statistic "
      "(mean vs median, uniform vs Gaussian kernel, self included); only the scib "
      "recompute above is a valid replacement.")
    W("  Table 1 bold/underline and stars cover NMI/ARI/MMD only, so they do not move; "
      "re-run compute_table1_significance.py on t1_tables/ to confirm.")
    return "\n".join(L) + "\n"


def pd_num(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


def _newest_merge(out):
    d = sorted((p for p in (out / "merged").glob("*") if p.is_dir()
                and (p / "fig_tidy").is_dir()), key=lambda p: p.name)
    if not d:
        raise SystemExit(f"no merged/<ts>/fig_tidy under {out}; run merge first")
    return d[-1]


def cmd_figures(a):
    """Fig S3 b,d (and the other datasets' benchmark panels) from a merge's
    fig_tidy/ frames, drawn by the plot modules' own make_figure."""
    _need_numpy()
    import pandas as pd
    _art, out = _resolve_dirs(a)
    merged = Path(os.path.abspath(a.merged)) if a.merged else _newest_merge(out)
    if out not in merged.parents:
        raise SystemExit(f"--merged {merged} is not under {out}")
    dest = merged / "fig_panels"
    if dest.exists():
        raise SystemExit(f"{dest} exists; figures are never overwritten (merge again "
                         f"for a new folder)")
    dest.mkdir(parents=False)
    mods = {"niche": "plot_niche_identification_benchmark",
            "cell_type": "plot_cell_type_identification_benchmark"}
    n = 0
    for block, (stem, _cols) in T1_CSV.items():
        mod = _repo_module(mods[block])
        for f in sorted((merged / "fig_tidy").glob(f"{stem}_*_tidy.csv")):
            df = pd.read_csv(f)
            if df.empty:
                continue
            if hasattr(mod, "_apply_nature_style"):
                mod._apply_nature_style()
            base = dest / f.name.replace("_tidy.csv", "")
            print(f"rendering {f.name} -> {base}.*", flush=True)
            mod.make_figure(df, base)
            n += 1
    print(f"{n} panel set(s) in {dest}")
    return 0 if n else 1


# =============================================================================
# Selftest
# =============================================================================
def _synthetic_tied(rng, n_per_batch, n_codes, d, pure_frac):
    """
    Two batches stored batch by batch (all A rows, then all B rows), every cell an
    exact copy of one of `n_codes` well-separated code vectors. With pure_frac = 0
    codes are drawn independently of batch (perfect mixing); otherwise that
    fraction of each batch sits in codes only its own batch uses.
    """
    codes = (rng.normal(size=(n_codes, d)) * 10.0).astype(np.float32)
    batch = np.repeat(np.array(["A", "B"]), n_per_batch)
    n = batch.size
    if pure_frac <= 0:
        assign = rng.integers(0, n_codes, size=n)
    else:
        n_pure = n_codes // 4
        pure = {"A": np.arange(0, n_pure), "B": np.arange(n_pure, 2 * n_pure)}
        shared = np.arange(2 * n_pure, n_codes)
        assign = np.empty(n, dtype=np.int64)
        for b in ("A", "B"):
            m = np.flatnonzero(batch == b)
            is_pure = rng.random(m.size) < pure_frac
            assign[m] = np.where(is_pure, rng.choice(pure[b], m.size),
                                 rng.choice(shared, m.size))
    return codes[assign], batch, assign


def _synthetic_imbalanced(rng, n_groups, group_size, frac_a, d):
    """
    `n_groups` large tie groups, each `frac_a` batch A and the rest batch B, rows
    stored batch by batch. At p ~ 0.3 LISI is steep in p (unlike p ~ 0.5), so an
    estimator whose tie draws are shared by a whole group shows up here.
    """
    codes = (rng.normal(size=(n_groups, d)) * 10.0).astype(np.float32)
    n_a = int(round(frac_a * group_size))
    assign = np.concatenate([np.repeat(np.arange(n_groups), n_a),
                             np.repeat(np.arange(n_groups), group_size - n_a)])
    batch = np.array(["A"] * (n_groups * n_a) + ["B"] * (n_groups * (group_size - n_a)))
    return codes[assign], batch, assign


def _oracle_ilisi(assign, batch, k, rng, n_draws=3):
    """
    Scaled iLISI under INDEPENDENT per-cell uniform tie-breaking, scored by the
    same scib call as everything else, so it assumes nothing about how scib
    weights neighbours or treats the self column. Every tie group must exceed
    k: a cell's ideal neighbourhood is then itself (column 0) plus k-1 members
    of its own group drawn uniformly without replacement, all at distance 0.
    Returns (mean, s.d.) over `n_draws` graphs.
    """
    n = assign.size
    kk = int(min(k, n - 1))
    groups = [np.flatnonzero(assign == g) for g in np.unique(assign)]
    if min(m.size for m in groups) <= kk:
        raise ValueError("the oracle needs every tie group larger than k")
    vals = []
    for _ in range(n_draws):
        idx = np.empty((n, kk), dtype=np.int64)
        idx[:, 0] = np.arange(n)
        for m in groups:
            for j in range(m.size):
                pos = rng.choice(m.size - 1, kk - 1, replace=False)
                pos[pos >= j] += 1                          # skip self
                idx[m[j], 1:] = m[pos]
        vals.append(scib_ilisi(idx, np.zeros((n, kk), dtype=np.float32), batch))
    return float(np.mean(vals)), float(np.std(vals))


def cmd_selftest(a):
    _need_numpy()
    _require_scib()
    import pynndescent                                          # noqa: F401
    rng = np.random.default_rng(a.seed)
    hard, soft = [], []

    def check(name, ok, detail, is_hard=True):
        (hard if is_hard else soft).append((name, bool(ok), detail))
        tag = ("PASS" if ok else "FAIL") if is_hard else "INFO"
        print(f"  [{tag}] {name}: {detail}", flush=True)

    def nnd(X, b, rs):
        idx, dist = nndescent_graph(X, K, rs)
        return scib_ilisi(idx, dist, b), n_missing_nbrs(idx)

    print("PART 1: perfectly mixed codes, rows stored batch by batch", flush=True)
    X, batch, assign = _synthetic_tied(rng, a.n_per_batch, 12, 8, 0.0)
    ties = tie_stats(X, K)
    check("tie diagnostics", ties["n_unique_rows"] == 12
          and ties["frac_cells_tie_gt_k"] == 1.0,
          f"{ties['n_unique_rows']} distinct rows, largest group "
          f"{ties['max_tie_group']}, {ties['frac_cells_tie_gt_k']:.0%} of cells in "
          f"groups > k")
    oracle, osd = _oracle_ilisi(assign, batch, K, rng)
    print(f"  oracle (independent random tie-breaking, scored by scib): {oracle:.4f} "
          f"(s.d. over draws {osd:.4f})", flush=True)
    p = rng.permutation(X.shape[0])
    gi, _gd = exact_graph(X, K, rng=np.random.default_rng(1))
    check("exact kNN puts the query point in column 0 of every row",
          bool(np.all(gi[:, 0] == np.arange(X.shape[0]))),
          f"{int(np.sum(gi[:, 0] == np.arange(X.shape[0])))}/{X.shape[0]} rows")
    big = np.flatnonzero(assign == np.bincount(assign).argmax())
    distinct = len({tuple(np.sort(r)) for r in gi[big, 1:]}) / big.size
    check("exact kNN breaks ties independently per row (members of one tie group "
          "get different neighbour sets)", distinct >= 0.99,
          f"{distinct:.3f} of {big.size} members have a distinct set")
    e_file = scib_ilisi(gi, _gd, batch)
    e_perm = scib_ilisi(*exact_graph(X[p], K, rng=np.random.default_rng(2)), batch[p])
    check("exact kNN, file order, recovers the oracle (row-order invariant)",
          abs(e_file - oracle) <= 0.04, f"{e_file:.4f} vs {oracle:.4f}")
    check("exact kNN, permuted rows, recovers the oracle",
          abs(e_perm - oracle) <= 0.04, f"{e_perm:.4f} vs {oracle:.4f}")
    i_file = scib_ilisi(*exact_graph(X, K, ties="index"), batch)
    check("fault 1 mechanism: ties broken by row index on batch-sorted rows "
          "understate iLISI", i_file <= oracle - 0.5,
          f"{i_file:.4f} vs oracle {oracle:.4f} (gap {i_file - oracle:+.4f})")
    n_file, m_file = nnd(X, batch, 0)
    n_perm, m_perm = nnd(X[p], batch[p], 0)
    check("NNDescent (the paper's graph), permuted rows, recovers the oracle",
          abs(n_perm - oracle) <= 0.08,
          f"{n_perm:.4f} vs {oracle:.4f}" + (f" [{m_perm} unfilled neighbours]"
                                             if m_perm else ""))
    check("NNDescent, file order (does the paper's estimator inherit fault 1?)",
          True, f"{n_file:.4f}, gap to permuted {n_file - n_perm:+.4f}, vs the "
          f"mechanism gap {i_file - oracle:+.4f}"
          + (f" [{m_file} unfilled neighbours]" if m_file else "")
          + ". Not asserted; on real data the merge report measures it", is_hard=False)

    print("PART 2: imbalanced shared codes (30% A / 70% B), 4 large tie groups, "
          "rows stored batch by batch", flush=True)
    X3, b3, a3 = _synthetic_imbalanced(rng, 4, 1500, 0.3, 8)
    oracle3, osd3 = _oracle_ilisi(a3, b3, K, rng)
    print(f"  oracle: {oracle3:.4f} (s.d. over draws {osd3:.4f})", flush=True)
    p3 = rng.permutation(X3.shape[0])
    e3 = scib_ilisi(*exact_graph(X3[p3], K, rng=np.random.default_rng(3)), b3[p3])
    check("exact kNN, permuted rows, recovers the oracle at p ~ 0.3",
          abs(e3 - oracle3) <= 0.04, f"{e3:.4f} vs {oracle3:.4f}")
    n3p, m3p = nnd(X3[p3], b3[p3], 0)
    check("NNDescent, permuted rows, recovers the oracle at p ~ 0.3",
          abs(n3p - oracle3) <= 0.08,
          f"{n3p:.4f} vs {oracle3:.4f}" + (f" [{m3p} unfilled neighbours]"
                                            if m3p else ""))
    n3f, m3f = nnd(X3, b3, 0)
    i3 = scib_ilisi(*exact_graph(X3, K, ties="index"), b3)
    check("NNDescent, file order, at p ~ 0.3", True,
          f"{n3f:.4f}, gap to permuted {n3f - n3p:+.4f}, vs the mechanism gap "
          f"{i3 - oracle3:+.4f}" + (f" [{m3f} unfilled neighbours]" if m3f else ""),
          is_hard=False)

    print("PART 3: 60% of each batch in batch-pure codes: scaled median vs fallback "
          "mean", flush=True)
    X2, b2, _a2 = _synthetic_tied(rng, a.n_per_batch, 12, 8, 0.6)
    p2 = rng.permutation(X2.shape[0])
    idx, dist = exact_graph(X2[p2], K, rng=np.random.default_rng(4))
    scaled = scib_ilisi(idx, dist, b2[p2])
    fb = fallback_inline_mean(idx, b2[p2])
    naive = (fb - 1.0) / (2 - 1)
    check("scib value is the scaled median in [0, 1]", -1e-6 <= scaled <= 0.1,
          f"{scaled:.4f} (most cells sit in batch-pure codes, so the median cell is "
          f"unmixed)")
    check("fallback value is an unscaled mean in [1, n_batches]",
          1.0 - 1e-9 <= fb <= 2.0 + 1e-9, f"{fb:.4f}")
    check("rescaling the fallback does not give the scib value",
          abs(naive - scaled) > 0.2,
          f"(fallback - 1)/(B - 1) = {naive:.4f} vs scib {scaled:.4f}")

    print("PART 4: exact_graph against scikit-learn on continuous data", flush=True)
    try:
        from sklearn.neighbors import NearestNeighbors
        Xc = rng.normal(size=(3000, 6)).astype(np.float32)
        i1, d1 = exact_graph(Xc, K)
        d2, i2 = NearestNeighbors(n_neighbors=K).fit(Xc).kneighbors(Xc)
        same = np.mean([set(x) == set(y) for x, y in zip(i1, i2)])
        check("exact_graph equals sklearn on tie-free data", same >= 0.999
              and np.allclose(np.sort(d1, axis=1), np.sort(d2, axis=1), atol=1e-3)
              and bool(np.all(i1[:, 0] == np.arange(Xc.shape[0]))),
              f"{same:.4f} of rows have identical neighbour sets; self in column 0")
    except ImportError:
        print("  (scikit-learn not installed: skipped)", flush=True)

    n_fail = sum(1 for _n, ok, _d in hard if not ok)
    print(f"\nselftest: {len(hard) - n_fail}/{len(hard)} hard checks passed; "
          f"{len(soft)} informational line(s)", flush=True)
    return 1 if n_fail else 0


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--selftest":        # same thing, flag spelling
        argv[0] = "selftest"
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
        sp.add_argument("--out-dir", default=None,
                        help=f"default <artifacts-root>/{DEFAULT_OUT_NAME}; must be a "
                             f"subfolder of the artifacts root")

    m = sub.add_parser("manifest", help="enumerate rows; write manifest.csv")
    common(m)
    m.add_argument("--dry-run", action="store_true", help="print, write nothing")
    m.add_argument("--force", action="store_true",
                   help="supersede an existing manifest (moved, never deleted)")
    m.add_argument("--no-probe", action="store_true",
                   help="skip the h5py header checks of every adata")
    m.add_argument("--sanity", choices=("none", "seed0", "all"), default="seed0",
                   help="recompute continuous, unaffected rows too: none, the lowest "
                        "seed per (dataset, method, key) (default), or all. Recorded "
                        "per row (sanity_tier), so plan / compute --sanity can change "
                        "it later without a rebuild")
    m.add_argument("--discretization-csv", default=None,
                   help="discretization_per_seed.csv behind Table 3's Discretization "
                        "block (default: <ds>/discretization/, the ablations dir, "
                        "run_ablation_report's lookup, older comparison_vs_discrete/ "
                        "copies; the first that holds the three conditions and "
                        "reproduces the printed iLISI wins)")
    m.add_argument("--grain", choices=("method", "task"), default="method",
                   help="job grain for the PLAN lines printed by --dry-run")

    pl = sub.add_parser("plan", help="print the LSF job list")
    common(pl)
    pl.add_argument("--grain", choices=("method", "task"), default="method",
                    help="one job per (dataset, method/variant), or per task")
    pl.add_argument("--all", action="store_true", help="also list finished jobs")
    pl.add_argument("--sanity", choices=("none", "seed0", "all"), default=None,
                    help="override the manifest's choice for the continuous sanity "
                         "rows (default: as built)")

    c = sub.add_parser("compute", help="run one job (resumable)")
    common(c)
    c.add_argument("--job", required=True,
                   help="a job_id or task_id from the manifest, or ALL")
    c.add_argument("--knn", choices=("nndescent", "exact", "both"), default="nndescent",
                   help="nndescent = the paper's estimator; exact = exact kNN, self "
                        "in column 0, ties broken independently per row at random "
                        "(seeded); both")
    c.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    c.add_argument("--no-replay", action="store_true",
                   help="skip the file-order replay (and the fallback statistic)")
    c.add_argument("--exact-chunk", type=int, default=EXACT_CHUNK)
    c.add_argument("--require-scib-version", default=None)
    c.add_argument("--sanity", choices=("none", "seed0", "all"), default=None,
                   help="as for plan; pass the same value to both")
    c.add_argument("--no-mmd-check", action="store_true",
                   help="skip the producer-MMD provenance replay on SQUINT files")
    c.add_argument("--force", action="store_true",
                   help="recompute tasks that have a result; after a successful "
                        "recompute the old csv is moved to results/superseded/")

    mg = sub.add_parser("merge", help="per-seed + summary csvs and the diff report")
    common(mg)
    mg.add_argument("--policy", choices=("affected", "all"), default="affected",
                    help="affected: replace only tie-affected, fallback and control "
                         "rows (every other published value is a valid scib score on "
                         "a continuous embedding and is kept); all: also replace a "
                         "continuous cell, but only when every seed of it has a "
                         "recompute (never a mixed five-seed mean)")
    mg.add_argument("--final-estimator", choices=("nndescent", "exact"),
                    default="nndescent",
                    help="nndescent (default) = the paper's estimator on permuted "
                         "rows; exact = the exact-kNN robustness check, to be named "
                         "as such wherever its values are used")

    fg = sub.add_parser("figures", help="re-render the Table 1 benchmark panels "
                                        "(Fig S3 b,d) from a merge's fig_tidy/")
    common(fg)
    fg.add_argument("--merged", default=None,
                    help="a merged/<ts> folder (default: the newest with fig_tidy/)")

    s = sub.add_parser("selftest", help="synthetic checks of the estimators and of "
                                        "both faults")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-per-batch", type=int, default=3000)

    a = p.parse_args(argv)
    if getattr(a, "n_perm", 1) < 1:
        p.error("--n-perm must be >= 1")
    return {"manifest": cmd_manifest, "plan": cmd_plan, "compute": cmd_compute,
            "merge": cmd_merge, "figures": cmd_figures,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
