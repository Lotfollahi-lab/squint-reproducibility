#!/usr/bin/env bash
# submit_all_benchmarks.sh
# -----------------------------------------------------------------------------
# Submit one LSF job per benchmarking method (cell-type identification +
# niche identification) on a given dataset. Defaults to mmb0-1b_smb1-1b_1p
# (1 MERFISH + 1 STARmap, mouse ~431 genes, coord-aligned silver dir);
# other datasets via the DATASET_TAG argument.
#
# Mirrors the structure of squint/examples/submit_dataset_sweep.sh:
# one bsub per method, each with its own venv, log files, and resource
# class. Per-method config (script path, venv, resource class, dataset-
# specific extra args) is defined in the METHODS table below.
#
# Usage:
#   bash submit_all_benchmarks.sh [DATASET_TAG] [OPTIONS]
#
# Arguments:
#   DATASET_TAG  — dataset key (default: mmb0-1b_smb1-1b_1p).
#                  Used to:
#                    * find the silver dir (resolved per-dataset in the
#                      CASE block below — e.g. mmb0-1b_smb1-1b_1p uses
#                      the `_coord_aligned` silver dir on lustre).
#                    * tag artefacts under $ARTIFACTS_ROOT/<DATASET_TAG>/
#                    * choose dataset-specific extra args (see CASE block).
#
# Environment overrides (any can be set on the command line):
#   SILVER_ROOT       — silver dir parent (default: /nfs/team361/sb75/DATASETS/silver)
#   ARTIFACTS_ROOT    — artefact root      (default: /nfs/team361/sb75/squint-reproducibility/artifacts)
#   REPO              — repo root (auto-detected from this script)
#   LOG_ROOT          — LSF log dir parent (default: $ARTIFACTS_ROOT/logs)
#   VENV_ROOT         — venv parent dir    (default: /nfs/team361/sb75/.venvs)
#   LSF_GROUP         — bsub -G            (default: s10396)
#   LSF_QUEUE         — bsub -q            (default: training-parallel)
#   LSF_WALL          — bsub -W            (default: 24:00)
#   LSF_GPU           — bsub -gpu spec     (default: mode=exclusive_process:num=1:block=yes)
#   ONLY              — comma-separated subset of method keys to submit
#                       (e.g. ONLY="scvi,pca-leiden" submits just those two)
#   EXCLUDE           — comma-separated keys to skip
#   DRY_RUN           — "1" to print bsub commands without submitting
#
# Examples:
#   # Submit all methods on mmb0-1b_smb1-1b_1p (the default):
#   bash submit_all_benchmarks.sh
#
#   # Submit on a different dataset:
#   bash submit_all_benchmarks.sh chl59-8b_1p
#
#   # Just one method, dry-run to inspect:
#   DRY_RUN=1 ONLY=nicheformer bash submit_all_benchmarks.sh
#
#   # Skip the heavy foundation models:
#   EXCLUDE=geneformer,scgpt,scgpt-spatial,uce bash submit_all_benchmarks.sh
#
# Logs:
#   stdout/err -> $LOG_ROOT/<dataset_tag>/all-benchmarks/<method>.{out,err}
# Per-job artefacts go to <ARTIFACTS_ROOT>/<dataset_tag>/<variant>/<TS>/
# as configured by each runner script's own --variant-tag default.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Args -----------------------------------------------------------
DATASET_TAG="${1:-mmb0-1b_smb1-1b_1p}"

# --- Paths ----------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO="${REPO:-"$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)"}"
RUNNER="$SCRIPT_DIR/_run_one_benchmark.sh"
if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: per-job runner not found at $RUNNER" >&2
    exit 1
fi

# Per-dataset silver root + leaf-dir resolution. Each dataset lives on
# a different filesystem in our cluster setup, and some have a non-
# standard leaf dirname (e.g. `_coord_aligned`) under the silver root.
# Override SILVER_DIR directly to point anywhere.
case "$DATASET_TAG" in
    chl59-8b_1p)
        SILVER_ROOT_DEFAULT="/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver"
        SILVER_LEAF_DEFAULT="$DATASET_TAG"
        ;;
    squint_hln)
        # CosMx human lymph node (NanoString), single section, manual
        # niches (spatial-niche-benchmark). Standard silver leaf on team361
        # (== the `*` default, made explicit so it groups with the
        # dataset-args case below).
        SILVER_ROOT_DEFAULT="/nfs/team361/sb75/DATASETS/silver"
        SILVER_LEAF_DEFAULT="$DATASET_TAG"
        ;;
    smb1-20b_1p)
        # STARmap+ mouse-CNS, 20 sections, native STARmap gene panel (the
        # STARmap-only counterpart to mmb20). Standard silver leaf on
        # team361 (== the `*` default, made explicit so it groups with the
        # dataset-args case below).
        SILVER_ROOT_DEFAULT="/nfs/team361/sb75/DATASETS/silver"
        SILVER_LEAF_DEFAULT="$DATASET_TAG"
        ;;
    spatch_ov_1p|spatch_hcc_1p|spatch_coad_1p)
        # SPATCH pan-cancer spatial subsets (ovarian / HCC / colon adeno),
        # human, multi-section (dataset_id_*.h5ad). Standard team361 silver
        # leaf (== the `*` default, explicit so it groups with the
        # dataset-args case below).
        SILVER_ROOT_DEFAULT="/nfs/team361/sb75/DATASETS/silver"
        SILVER_LEAF_DEFAULT="$DATASET_TAG"
        ;;
    xhs1000-3b_1p)
        # Xenium human skin, 3 sections (batch 11/19/32), ~4948 genes. Silver
        # dir populated by analysis/data_preparation/prep_xhs_3b.py (stamped
        # uns['batch'] + unique obs_names). Standard team361 silver leaf.
        SILVER_ROOT_DEFAULT="/nfs/team361/sb75/DATASETS/silver"
        SILVER_LEAF_DEFAULT="$DATASET_TAG"
        ;;
    mmb0-1b_smb1-1b_1p)
        # 1 MERFISH + 1 STARmap mouse-brain silver. The on-disk leaf
        # is `<tag>_coord_aligned` (post xy-alignment pass); the
        # DATASET_TAG kept as the user-facing key so artefact dirs +
        # benchmark CSV columns stay short and stable.
        SILVER_ROOT_DEFAULT="/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver"
        SILVER_LEAF_DEFAULT="${DATASET_TAG}_coord_aligned"
        ;;
    *)
        SILVER_ROOT_DEFAULT="/nfs/team361/sb75/DATASETS/silver"
        SILVER_LEAF_DEFAULT="$DATASET_TAG"
        ;;
esac
SILVER_ROOT="${SILVER_ROOT:-$SILVER_ROOT_DEFAULT}"
SILVER_DIR="${SILVER_DIR:-$SILVER_ROOT/$SILVER_LEAF_DEFAULT}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts}"
LOG_ROOT="${LOG_ROOT:-$ARTIFACTS_ROOT/logs}"
LOG_DIR="$LOG_ROOT/$DATASET_TAG/all-benchmarks"
# Only create the log dir on a real submission — DRY_RUN should be
# inspectable on any machine (e.g. laptop) without the NFS mount.
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$LOG_DIR"
fi

VENV_ROOT="${VENV_ROOT:-/nfs/team361/sb75/.venvs}"

# --- Foundation-model artefact paths (override per-dataset/install) ----
# Edit these if your release artefacts live elsewhere. They're only
# referenced by methods that need them; the per-method extra-args block
# below interpolates them.
# Nicheformer release layout on disk is `<...>/nicheformer/data/`
# containing `nicheformer.ckpt` directly + `model_means/<tech>_mean_script.npy`
# + `model_means/model.h5ad`. The runner searches `<model_dir>/...` and
# `<model_dir>/model_means/...`, so we point it at the `data/` subdir.
NICHEFORMER_MODEL_DIR="${NICHEFORMER_MODEL_DIR:-$REPO/analysis/benchmarking/nicheformer/data}"
# Geneformer release ships as `geneformer-v2-104M/Geneformer-V2-104M/`
# (the outer dir is the HuggingFace repo name, the inner is the model
# itself). The runner needs the inner one — that's where `config.json`,
# `pytorch_model.bin`, `vocab.json` etc. live.
GENEFORMER_MODEL_DIR="${GENEFORMER_MODEL_DIR:-$REPO/analysis/benchmarking/geneformer/geneformer-v2-104M/Geneformer-V2-104M}"
# scGPT release is unpacked as `scGPT/scGPT_human/` (no `checkpoints/`
# intermediary). Don't add that subdir back unless your local layout
# really has one — the runner reads files directly from the path you
# pass.
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-$REPO/analysis/benchmarking/scGPT/scGPT_human}"
SCGPT_SPATIAL_MODEL_DIR="${SCGPT_SPATIAL_MODEL_DIR:-$REPO/analysis/benchmarking/scGPT-spatial/checkpoints/scGPT_spatial_v1}"
UCE_MODEL_LOC="${UCE_MODEL_LOC:-$REPO/analysis/benchmarking/uce_model/33l_8ep_1024t_1280.torch}"
UCE_MODEL_FILES_DIR="${UCE_MODEL_FILES_DIR:-$REPO/analysis/benchmarking/uce_model}"
# NicheCompass auxiliary inputs. The orthologs CSV ships from the
# nichecompass release as `human_mouse_gene_orthologs.csv` (not the
# shorter `gene_orthologs.csv` the upstream docs use as a placeholder
# name); the mebocost dir holds the metabolite-enzyme-sensor GP TSVs
# + the cached `combined_gp_dict.pkl`.
NICHECOMPASS_ORTHOLOGS_CSV="${NICHECOMPASS_ORTHOLOGS_CSV:-$REPO/analysis/benchmarking/nichecompass/human_mouse_gene_orthologs.csv}"
NICHECOMPASS_MEBOCOST_DIR="${NICHECOMPASS_MEBOCOST_DIR:-$REPO/analysis/benchmarking/nichecompass/metabolite_enzyme_sensor_gps}"

# --- LSF defaults ---------------------------------------------------
LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
# CPU-only methods (cpu_small: pca-leiden, harmony, banksy, neigh-expr-pca) must
# NOT go to the GPU queue: that queue's esub rejects jobs that don't request a
# GPU ("you need to select a system which has gpus, eg -gpu"). Route them to a
# CPU queue instead. (This only surfaces when running submit_all_benchmarks.sh
# directly; the wrappers force UNIFORM_RESOURCE=gpu_high_memory so everything
# gets a GPU.) Override if your CPU queue is named differently.
CPU_QUEUE="${CPU_QUEUE:-normal}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
# Default wall-time bumped 24h -> 96h (4 days). The 24h ceiling was too
# tight for the heaviest baselines on the largest datasets — Geneformer
# / scGPT FM extraction alone can take 12-18h on chl59 / spatch_1p, and
# stacking that on top of a per-seed Leiden binary search occasionally
# overran 24h. 96h gives every baseline plenty of headroom regardless of
# dataset scale. Override via `--wall` on the wrappers or the env var.
LSF_WALL="${LSF_WALL:-96:00}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"
EXCLUDE="${EXCLUDE:-}"

# UNIFORM_RESOURCE: when set to a known resource class (cpu_small /
# gpu_standard / gpu_high_memory), every submitted job uses THAT class
# regardless of what the METHODS table specifies. Intended for fair
# runtime comparisons across methods — all jobs land on identically-
# sized nodes, so any runtime delta is attributable to the method
# rather than to compute heterogeneity. The wrappers
# `submit_cell_type_baselines.sh` / `submit_niche_id_baselines.sh`
# set `UNIFORM_RESOURCE=gpu_high_memory` by default for exactly this
# reason. Leave unset to honour per-method classes from METHODS.
UNIFORM_RESOURCE="${UNIFORM_RESOURCE:-}"

# Per-resource-class LSF args.
#
# Memory tuning history:
#   - gpu_high_memory was originally 256 GB. GraphST on the mmb-smb
#     graph (~100k cells) peaked at 263.7 GB — just over the cap —
#     and bsub killed it with TERM_MEMLIMIT. Bumped to 384 GB
#     (= 50% headroom over GraphST's observed peak) so every method
#     finishes cleanly. The lighter methods (BANKSY, scvi) don't
#     actually use 384 GB; the bump just gives them a wider safety
#     margin and lets `UNIFORM_RESOURCE=gpu_high_memory` keep
#     producing apples-to-apples runtime numbers across methods.
#   - gpu_xtreme_memory (768 GB) added for runs where even
#     gpu_high_memory's 384 GB is insufficient. First hit on
#     spatch_ov_1p, where GraphST's PASTE alignment + adjacency matrix
#     blew through 384 GB. Intended for rerunning specific heavy
#     baselines on the bigger spatial datasets; not the default for the
#     wrappers (which still pin `UNIFORM_RESOURCE=gpu_high_memory` so
#     the apples-to-apples comparison stays sane on smaller datasets).
resource_args() {
    case "$1" in
        cpu_small)
            echo "-n 4 -M 64000 -R select[mem>64000] -R rusage[mem=64000] -R span[ptile=4]"
            ;;
        gpu_standard)
            echo "-n 6 -M 128000 -R select[mem>128000] -R rusage[mem=128000] -R span[ptile=6] -gpu $LSF_GPU"
            ;;
        gpu_high_memory)
            echo "-n 8 -M 384000 -R select[mem>384000] -R rusage[mem=384000] -R span[ptile=8] -gpu $LSF_GPU"
            ;;
        gpu_xtreme_memory)
            echo "-n 12 -M 768000 -R select[mem>768000] -R rusage[mem=768000] -R span[ptile=12] -gpu $LSF_GPU"
            ;;
        *)
            echo "ERROR: unknown resource class $1" >&2
            exit 1
            ;;
    esac
}

# Queue per resource class: GPU classes go to the GPU queue (LSF_QUEUE); the
# CPU class (cpu_small) goes to CPU_QUEUE — GPU queues reject non-GPU jobs.
resource_queue() {
    case "$1" in
        cpu_small) echo "$CPU_QUEUE" ;;
        *)         echo "$LSF_QUEUE" ;;
    esac
}

# Validate UNIFORM_RESOURCE up-front so a typo fails before any bsub
# fires (otherwise the first METHOD silently inherits the bogus class
# and `resource_args` errors out per-job).
if [[ -n "$UNIFORM_RESOURCE" ]]; then
    case "$UNIFORM_RESOURCE" in
        cpu_small|gpu_standard|gpu_high_memory|gpu_xtreme_memory) ;;
        *)
            echo "ERROR: UNIFORM_RESOURCE=$UNIFORM_RESOURCE is not a known class." >&2
            echo "       Valid: cpu_small | gpu_standard | gpu_high_memory | gpu_xtreme_memory" >&2
            exit 1
            ;;
    esac
fi

# --- Dataset-specific extra args ----------------------------------------
# Each method needs DIFFERENT gene-ID handling depending on species /
# technology / what's in adata.var_names. The per-method flags here are
# the result of auditing each foundation-model script's gene-ID code path.
# See `submit_all_benchmarks.sh` comments at the top of each block for
# the audit notes. Add a new CASE entry for other datasets as needed.
#
# LABEL_KEY_ARGS: per-dataset ground-truth label columns, appended to
# COMMON_ARGS and forwarded to EVERY runner (all 14 accept
# --cell-label-keys / --niche-label-keys; matched labels are
# canonicalised to cell_type / niche downstream). Empty by default so
# mmb / chl59 keep using the runner defaults (cell_type / niche /
# annotation / spatial_cluster). Only datasets whose silver obs columns
# fall outside those defaults need to set it (e.g. squint_hln).
LABEL_KEY_ARGS=""
case "$DATASET_TAG" in
    chl59-8b_1p)
        # CosMx Lung, human, ~946 genes, 8 batches.
        # Confirmed layout (from the user's adata inspection):
        #   - var_names = HGNC SYMBOLS  (e.g. AATK, ABL1, ...)
        #   - var has a pre-computed `ensembl_id` column (ENSG...)
        #     so Geneformer/Nicheformer can SKIP the mygene lookup
        #     and read Ensembl IDs straight from var
        # Train/test split: Lung13 + Lung5_Rep3 are held out — their
        # h5ad files are SKIPPED by _load_concat (via SQUINT_EXCLUDE_BATCHES
        # below) so neither training nor metrics see them.
        SPECIES="human"
        NICHEFORMER_TECHNOLOGY="cosmx"
        # Held-out batches, propagated to _load_concat via env var.
        # Multiple tokens are comma-separated. Matched as substring
        # against the filename (e.g. `Lung13+SMI+Flat+data.tar.h5ad`).
        HOLDOUT_BATCHES="Lung13,Lung5_Rep3"
        # scGPT / scGPT-spatial: vocab is HGNC; data is HGNC → no flags.
        SCGPT_GENE_FLAGS=""
        SCGPT_SPATIAL_GENE_FLAGS=""
        # Geneformer: read ENSG directly from var["ensembl_id"] — the
        # silver h5ads already carry the conversion, so the mygene
        # lookup is unnecessary (slower + a network dependency).
        GENEFORMER_GENE_FLAGS="--ensembl-id-col ensembl_id"
        # Nicheformer: same — pass the pre-mapped var column.
        # (--species is set separately by the row template via
        # `--species $SPECIES`, so don't repeat it here.)
        NICHEFORMER_GENE_FLAGS="--gene-col ensembl_id"
        UCE_SPECIES_FLAGS="--uce-species human"
        NICHECOMPASS_SPECIES="human"
        ;;
    mmb0-1b_smb1-1b_1p)
        # MERFISH + STARmap mouse brain, mouse, ~431 genes.
        # var_names = MOUSE SYMBOLS — every foundation model needs the
        # mouse → human ortholog mapping.
        SPECIES="mouse"
        NICHEFORMER_TECHNOLOGY="merfish"
        HOLDOUT_BATCHES=""   # no per-section holdout on mmb-smb
        SCGPT_GENE_FLAGS="--map-via-human-orthologs"
        SCGPT_SPATIAL_GENE_FLAGS="--map-via-human-orthologs"
        GENEFORMER_GENE_FLAGS="--map-via-human-orthologs"
        NICHEFORMER_GENE_FLAGS="--auto-map-symbols"
        UCE_SPECIES_FLAGS="--uce-species mouse"
        NICHECOMPASS_SPECIES="mouse"
        ;;
    squint_hln)
        # CosMx human lymph node (NanoString), single section, manual
        # niches. var_names = HGNC SYMBOLS; the silver h5ad carries NO
        # pre-computed `ensembl_id` column (unlike chl59), so the
        # ENSG-based foundation models map symbols -> human ENSG via
        # mygene (`--auto-map-symbols`, species=human) — needs internet
        # on the compute node for the first mygene call.
        SPECIES="human"
        NICHEFORMER_TECHNOLOGY="cosmx"
        HOLDOUT_BATCHES=""          # single section -> no train/test holdout
        # scGPT / scGPT-spatial: vocab is HGNC, data is HGNC -> no flags.
        SCGPT_GENE_FLAGS=""
        SCGPT_SPATIAL_GENE_FLAGS=""
        # Geneformer V2 (human ENSG vocab): map HGNC symbols -> human
        # ENSG via mygene. (--auto-map-symbols routes to species=human
        # here, NOT the mouse->ENSMUSG path.)
        GENEFORMER_GENE_FLAGS="--auto-map-symbols"
        NICHEFORMER_GENE_FLAGS="--auto-map-symbols"
        UCE_SPECIES_FLAGS="--uce-species human"
        NICHECOMPASS_SPECIES="human"
        # Ground-truth labels live in the manual-annotation columns of
        # the silver h5ad (== what the SQUINT run + blob use), which are
        # NOT in the runner defaults — pass them explicitly so NMI/ARI
        # are computed against the right columns.
        LABEL_KEY_ARGS="--cell-label-keys cell_type_annotation --niche-label-keys niche_annotation"
        ;;
    smb1-20b_1p)
        # STARmap+ mouse-CNS, 20 sections, native STARmap gene panel
        # (STARmap-only counterpart to mmb20). MOUSE -> same FM gene-ID
        # handling as mmb0-1b_smb1-1b_1p (mouse symbols -> human-ortholog
        # ENSG). MULTI-section (20 batches) -> iLISI/MMD batch integration
        # IS meaningful here (unlike single-section squint_hln).
        SPECIES="mouse"
        # nicheformer has no STARmap-specific mean; reuse the `merfish`
        # technology mean as the in-situ proxy — the same choice already
        # used for the STARmap section in mmb0-1b_smb1-1b_1p.
        NICHEFORMER_TECHNOLOGY="merfish"
        HOLDOUT_BATCHES=""   # train/eval on all 20 sections (no holdout)
        SCGPT_GENE_FLAGS="--map-via-human-orthologs"
        SCGPT_SPATIAL_GENE_FLAGS="--map-via-human-orthologs"
        GENEFORMER_GENE_FLAGS="--map-via-human-orthologs"
        NICHEFORMER_GENE_FLAGS="--auto-map-symbols"
        UCE_SPECIES_FLAGS="--uce-species mouse"
        NICHECOMPASS_SPECIES="mouse"
        # Ground-truth labels: STARmap sections carry cell_type / niche /
        # Sub_molecular_tissue_region / ccf_region_name — ALL already in the
        # runner defaults, so no explicit override needed (same as the mmb
        # case). Leave LABEL_KEY_ARGS empty.
        ;;
    spatch_ov_1p|spatch_hcc_1p|spatch_coad_1p)
        # SPATCH pan-cancer spatial subsets (ovarian / HCC / colon adeno).
        # HUMAN, multi-section (dataset_id_*.h5ad) -> iLISI/MMD batch
        # integration IS meaningful here.
        SPECIES="human"
        # !! PLATFORM-DEPENDENT (verify per subset). The SPATCH extraction
        # supports BOTH Xenium (var_names = ENSG + an `ensembl_id` var col)
        # AND CosMx (var_names = HGNC symbols). The geneformer/nicheformer
        # flag below is ROBUST to both: `_ensure_ensembl_ids` auto-detects
        # ENSG var_names BEFORE falling back to mygene symbol->ENSG mapping.
        # nicheformer technology defaults to the common Xenium case; switch
        # to `cosmx` (or another) per subset if needed.
        NICHEFORMER_TECHNOLOGY="xenium"
        HOLDOUT_BATCHES=""
        # scGPT vocab is HGNC SYMBOLS. If a subset's var_names are ENSG
        # (Xenium), scGPT/scGPT-spatial will under-match genes -> consider
        # excluding them or pre-mapping ENSG->symbol for that subset.
        SCGPT_GENE_FLAGS=""
        SCGPT_SPATIAL_GENE_FLAGS=""
        GENEFORMER_GENE_FLAGS="--auto-map-symbols"
        NICHEFORMER_GENE_FLAGS="--auto-map-symbols"
        UCE_SPECIES_FLAGS="--uce-species human"
        NICHECOMPASS_SPECIES="human"
        # spatch blob labels (== what the SQUINT run uses): cell=annotation,
        # niche=spatial_cluster. Pass explicitly so the right columns are
        # scored even if a section also carries a stray `cell_type` column.
        LABEL_KEY_ARGS="--cell-label-keys annotation --niche-label-keys spatial_cluster"
        ;;
    xhs1000-3b_1p)
        # Xenium human skin, 3 sections (batch 11/19/32 — DIFFERENT patients),
        # ~4948 genes, var_names = HGNC symbols. MULTI-section -> iLISI/MMD
        # batch integration IS meaningful. Labels (== what the SQUINT run uses):
        # cell=new_annotation, niche=niche_type (NOT in the runner defaults, so
        # pass explicitly — also note the silver carries a stray `cell_type`
        # column we do NOT want scored). scGPT vocab is HGNC = data -> no flags;
        # geneformer/nicheformer map HGNC symbols -> human ENSG via mygene
        # (--auto-map-symbols; needs internet on the node for the first call).
        SPECIES="human"
        NICHEFORMER_TECHNOLOGY="xenium"
        HOLDOUT_BATCHES=""
        SCGPT_GENE_FLAGS=""
        SCGPT_SPATIAL_GENE_FLAGS=""
        GENEFORMER_GENE_FLAGS="--auto-map-symbols"
        NICHEFORMER_GENE_FLAGS="--auto-map-symbols"
        UCE_SPECIES_FLAGS="--uce-species human"
        NICHECOMPASS_SPECIES="human"
        LABEL_KEY_ARGS="--cell-label-keys new_annotation --niche-label-keys niche_type"
        ;;
    *)
        echo "WARNING: unknown DATASET_TAG=$DATASET_TAG; falling back to mouse defaults" >&2
        SPECIES="mouse"
        NICHEFORMER_TECHNOLOGY="merfish"
        HOLDOUT_BATCHES=""
        SCGPT_GENE_FLAGS="--map-via-human-orthologs"
        SCGPT_SPATIAL_GENE_FLAGS="--map-via-human-orthologs"
        GENEFORMER_GENE_FLAGS="--map-via-human-orthologs"
        NICHEFORMER_GENE_FLAGS="--auto-map-symbols"
        UCE_SPECIES_FLAGS="--uce-species mouse"
        NICHECOMPASS_SPECIES="mouse"
        ;;
esac

COMMON_ARGS="--silver-dir $SILVER_DIR --dataset-tag $DATASET_TAG --artifacts-root $ARTIFACTS_ROOT $LABEL_KEY_ARGS"

# --- Method registry ----------------------------------------------------
# Each row: key | script | venv | resource_class | extra_args
# `extra_args` may reference any shell var above (interpolated below).
METHODS=(
    # Cell-type identification ────────────────────────────────────────
    "pca-leiden|analysis/benchmarking/cell_type_identification/run_pca_leiden.py|squint|cpu_small|"
    # harmony depends on `harmonypy`, which is installed in the banksy
    # venv (not in the stock squint env). Routing it to `banksy` keeps
    # both methods on the same numba/scanpy pin set that's already
    # validated for the niche-id `banksy` baseline.
    "harmony|analysis/benchmarking/cell_type_identification/run_harmony.py|banksy|cpu_small|"
    # scvi imports `scvi-tools` which lives in the cellcharter venv
    # (NOT the squint one — squint has scvi-tools at an older pin
    # incompatible with the run_scvi.py call signatures).
    "scvi|analysis/benchmarking/cell_type_identification/run_scvi.py|cellcharter|gpu_standard|"
    "geneformer|analysis/benchmarking/cell_type_identification/run_geneformer.py|geneformer|gpu_high_memory|--model-dir $GENEFORMER_MODEL_DIR $GENEFORMER_GENE_FLAGS"
    "nicheformer|analysis/benchmarking/cell_type_identification/run_nicheformer.py|nicheformer|gpu_high_memory|--model-dir $NICHEFORMER_MODEL_DIR --species $SPECIES --technology $NICHEFORMER_TECHNOLOGY $NICHEFORMER_GENE_FLAGS"
    # scgpt-spatial venv satisfies BOTH the stock-scGPT runner and the
    # scGPT-spatial one (flash-attn-enabled wheels). The old standalone
    # `scgpt` venv was removed.
    "scgpt|analysis/benchmarking/cell_type_identification/run_scgpt.py|scgpt-spatial|gpu_high_memory|--model-dir $SCGPT_MODEL_DIR $SCGPT_GENE_FLAGS"
    "scgpt-spatial|analysis/benchmarking/cell_type_identification/run_scgpt_spatial.py|scgpt-spatial|gpu_high_memory|--model-dir $SCGPT_SPATIAL_MODEL_DIR $SCGPT_SPATIAL_GENE_FLAGS"
    "uce|analysis/benchmarking/cell_type_identification/run_uce.py|tissuejepa|gpu_high_memory|--model-loc $UCE_MODEL_LOC --model-files-dir $UCE_MODEL_FILES_DIR $UCE_SPECIES_FLAGS"

    # Niche identification ─────────────────────────────────────────────
    # banksy has its own venv (banksy is sensitive to numba / scanpy
    # versions; pinning it separately keeps the stock squint env stable).
    "banksy|analysis/benchmarking/niche_identification/run_banksy.py|banksy|cpu_small|"
    "cellcharter|analysis/benchmarking/niche_identification/run_cellcharter.py|cellcharter|gpu_standard|"
    "graphst|analysis/benchmarking/niche_identification/run_graphst.py|graphst|gpu_standard|--paste-use-gpu"
    "novae|analysis/benchmarking/niche_identification/run_novae.py|novae|gpu_standard|"
    "nichecompass|analysis/benchmarking/niche_identification/run_nichecompass.py|nichecompass|gpu_standard|--species $NICHECOMPASS_SPECIES --gene-orthologs-csv $NICHECOMPASS_ORTHOLOGS_CSV --mebocost-dir $NICHECOMPASS_MEBOCOST_DIR"
    "neigh-expr-pca|analysis/benchmarking/niche_identification/run_neigh_expr_pca.py|squint|cpu_small|"
)

# --- Filter via ONLY / EXCLUDE ----------------------------------------
parse_csv() { tr ',' '\n' <<< "$1" | sed '/^$/d'; }

if [[ -n "$ONLY" ]]; then
    ONLY_KEYS="$(parse_csv "$ONLY")"
fi
if [[ -n "$EXCLUDE" ]]; then
    EXCLUDE_KEYS="$(parse_csv "$EXCLUDE")"
fi

want_method() {
    local k="$1"
    if [[ -n "${ONLY_KEYS:-}" ]]; then
        grep -qFx "$k" <<< "$ONLY_KEYS" || return 1
    fi
    if [[ -n "${EXCLUDE_KEYS:-}" ]]; then
        if grep -qFx "$k" <<< "$EXCLUDE_KEYS"; then return 1; fi
    fi
    return 0
}

# --- Summary --------------------------------------------------------
echo "================================================================"
echo "Dataset tag      : $DATASET_TAG"
echo "Silver dir       : $SILVER_DIR"
echo "Repo             : $REPO"
echo "Log dir          : $LOG_DIR"
echo "Venv root        : $VENV_ROOT"
echo "Species defaults : $SPECIES"
echo "Nicheformer tech : $NICHEFORMER_TECHNOLOGY"
[[ -n "${ONLY:-}" ]]             && echo "ONLY            : $ONLY"
[[ -n "${EXCLUDE:-}" ]]          && echo "EXCLUDE         : $EXCLUDE"
[[ -n "${UNIFORM_RESOURCE:-}" ]] && echo "UNIFORM_RESOURCE: $UNIFORM_RESOURCE  (forces every method onto this class)"
echo "DRY_RUN         : $DRY_RUN"
echo "================================================================"

# --- Submit one bsub per method ---------------------------------------
n_submitted=0
n_skipped=0
for entry in "${METHODS[@]}"; do
    IFS='|' read -r KEY SCRIPT_REL VENV_NAME RESOURCE EXTRA <<< "$entry"

    if ! want_method "$KEY"; then
        echo "[ skip ] $KEY"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    SCRIPT_PATH="$REPO/$SCRIPT_REL"
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo "[ MISS ] $KEY  (script not found: $SCRIPT_PATH)" >&2
        n_skipped=$((n_skipped + 1))
        continue
    fi

    VENV_PATH="$VENV_ROOT/$VENV_NAME"
    # Skip the venv existence check on DRY_RUN — the dry-run is meant
    # to be inspectable on machines (e.g. laptop) that don't have the
    # NFS-mounted venvs. On real submission, missing venvs DO skip.
    if [[ "$DRY_RUN" != "1" && ! -f "$VENV_PATH/bin/activate" ]]; then
        echo "[ MISS ] $KEY  (venv not found: $VENV_PATH)" >&2
        n_skipped=$((n_skipped + 1))
        continue
    fi

    # UNIFORM_RESOURCE (when set) overrides per-method RESOURCE so every
    # job lands on identically-sized compute — required for the runtime
    # comparison the user is benchmarking against. Falls back to the
    # METHODS-table value when unset.
    if [[ -n "$UNIFORM_RESOURCE" ]]; then
        EFFECTIVE_RESOURCE="$UNIFORM_RESOURCE"
    else
        EFFECTIVE_RESOURCE="$RESOURCE"
    fi
    RES="$(resource_args "$EFFECTIVE_RESOURCE")"
    JOB_NAME="bench-${DATASET_TAG}-${KEY}"
    LOG_OUT="$LOG_DIR/${KEY}.out"
    LOG_ERR="$LOG_DIR/${KEY}.err"

    # The script invocation is COMMON_ARGS + EXTRA (interpolated). Use
    # `eval` so variable references inside EXTRA expand correctly.
    eval "FULL_ARGS=( $COMMON_ARGS $EXTRA )"

    BSUB_CMD=(
        bsub
        -G "$LSF_GROUP"
        -q "$(resource_queue "$EFFECTIVE_RESOURCE")"
        -W "$LSF_WALL"
        -J "$JOB_NAME"
        -o "$LOG_OUT"
        -e "$LOG_ERR"
    )
    # shellcheck disable=SC2206
    BSUB_CMD+=( $RES )

    # Env vars to propagate into the LSF job. `--env KEY=VAL` slots are
    # parsed by `_run_one_benchmark.sh` before the `--` separator and
    # exported before activating the venv / running the script. This
    # is the robust way to thread settings (e.g. SQUINT_EXCLUDE_BATCHES
    # for the train/test holdout) regardless of LSF's env-passthrough
    # configuration.
    RUNNER_ENV_ARGS=()
    if [[ -n "${HOLDOUT_BATCHES:-}" ]]; then
        RUNNER_ENV_ARGS+=( --env "SQUINT_EXCLUDE_BATCHES=$HOLDOUT_BATCHES" )
    fi
    # Optional: rapids-singlecell GPU-accelerated Leiden. The wrappers
    # set these env vars when --rapids-leiden is passed:
    #   * SQUINT_LEIDEN_BACKEND=rapids
    #     -> arms the inline path (used when rapids is importable in
    #        the baseline's own venv).
    #   * SQUINT_LEIDEN_RAPIDS_ENV_SETUP="<bash cmd>"
    #     -> arms the subprocess path (rapids in a separate conda env;
    #        helper spawns bash -lc "<cmd> && python worker.py").
    # When both are set the helper prefers subprocess (env-setup
    # cmd wins). See run_pca_leiden.py for the full dispatcher.
    if [[ -n "${SQUINT_LEIDEN_BACKEND:-}" ]]; then
        RUNNER_ENV_ARGS+=( --env "SQUINT_LEIDEN_BACKEND=$SQUINT_LEIDEN_BACKEND" )
    fi
    if [[ -n "${SQUINT_LEIDEN_RAPIDS_ENV_SETUP:-}" ]]; then
        RUNNER_ENV_ARGS+=( --env "SQUINT_LEIDEN_RAPIDS_ENV_SETUP=$SQUINT_LEIDEN_RAPIDS_ENV_SETUP" )
    fi

    if [[ -n "$UNIFORM_RESOURCE" && "$EFFECTIVE_RESOURCE" != "$RESOURCE" ]]; then
        echo "[ run  ] $KEY  (resource=$EFFECTIVE_RESOURCE  [forced via UNIFORM_RESOURCE; method default was $RESOURCE], venv=$VENV_NAME)"
    else
        echo "[ run  ] $KEY  (resource=$EFFECTIVE_RESOURCE, venv=$VENV_NAME)"
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '          '
        printf '%q ' "${BSUB_CMD[@]}"
        printf '%q %q %q ' bash "$RUNNER" "$SCRIPT_PATH"
        printf '%q ' "$VENV_PATH"
        if [[ ${#RUNNER_ENV_ARGS[@]} -gt 0 ]]; then
            printf '%q ' "${RUNNER_ENV_ARGS[@]}"
        fi
        printf -- '-- '
        printf '%q ' "${FULL_ARGS[@]}"
        printf '\n'
    else
        "${BSUB_CMD[@]}" \
            bash "$RUNNER" "$SCRIPT_PATH" "$VENV_PATH" \
                "${RUNNER_ENV_ARGS[@]}" \
                -- "${FULL_ARGS[@]}"
    fi
    n_submitted=$((n_submitted + 1))
done

echo "================================================================"
echo "Submitted: $n_submitted   Skipped: $n_skipped"
echo "Watch with: bjobs -aw"
echo "Logs      : $LOG_DIR/<method>.{out,err}"
echo "================================================================"
