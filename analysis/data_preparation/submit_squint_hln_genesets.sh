#!/usr/bin/env bash
# submit_squint_hln_genesets.sh
# -----------------------------------------------------------------------------
# End-to-end orchestrator for the squint_hln GENE-FILTERING comparison, matching
# the niche-benchmark paper's feature-selection axis (all detected genes /
# top-2000 HVG / top-2000 SVG). For each gene set it chains, with LSF -w
# dependencies:
#
#   [optional] preprocess  ->  build blob  ->  submit_multi_seed (5-seed train)
#
#   preprocess : analysis/data_preparation/preprocess_squint_hln.py  (CPU; drops
#                negative controls, filter_genes, writes silver/squint_hln_<tag>/)
#   build blob : run_squint.py --build-blob --build-blob-dataset squint_hln_<tag> (CPU)
#   train      : squint/examples/submit_multi_seed.sh <ref-variant+_<tag>> <seeds> (GPU)
#
# The reference model is byte-identical across gene sets (the FiLM-scale
# squint_hln reference); only the blob differs. So this isolates the effect of
# the gene set on SQUINT's cell/niche resolution (NMI/ARI vs the manual
# cell_type_annotation / niche_annotation labels).
#
# Usage:
#   bash submit_squint_hln_genesets.sh [GENE_SETS] [SEEDS] [--preprocess] [--skip-train] [--skip-build]
#     GENE_SETS  comma list of {all,hvg,svg}     (default: all,hvg,svg)
#     SEEDS      comma list for the multiseed sweep (default: 0,1,2,3,4)
#     --preprocess  also run the preprocessing job first (chained before builds)
#     --skip-build  assume blobs already built; only submit training
#     --skip-train  build blobs (+ preprocess) only; print the train commands
#
# Env overrides:
#   VENV_PATH    /nfs/team361/sb75/.venvs/squint
#   SQUINT_REPO  <auto: ../../../squint relative to this script>
#   REPRO_REPO   <auto: this repo root>
#   SILVER_ROOT  /nfs/team361/sb75/DATASETS/silver
#   INPUT_H5AD   <SILVER_ROOT>/squint_hln/cosmx_human_lymph_node.h5ad
#   N_TOP        2000          (HVG/SVG count)
#   CPU_QUEUE    normal        CPU_GROUP team361   CPU_MEM_MB 128000  CPU_WALL 8:00
#   GPU_QUEUE    gpu-lotfollahi (forwarded to submit_multi_seed.sh as LSF_QUEUE)
#   DRY_RUN      0  (1 = print all bsub commands, submit nothing)
# -----------------------------------------------------------------------------
set -euo pipefail

GENE_SETS="all,hvg,svg"
SEEDS="0,1,2,3,4"
DO_PREPROCESS=0
SKIP_BUILD=0
SKIP_TRAIN=0
pos=()
for a in "$@"; do
    case "$a" in
        --preprocess) DO_PREPROCESS=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        --skip-train) SKIP_TRAIN=1 ;;
        -h|--help) sed -n '3,45p' "$0"; exit 0 ;;
        *) pos+=("$a") ;;
    esac
done
[[ ${#pos[@]} -ge 1 ]] && GENE_SETS="${pos[0]}"
[[ ${#pos[@]} -ge 2 ]] && SEEDS="${pos[1]}"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPRO_REPO="${REPRO_REPO:-"$( cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd )"}"
SQUINT_REPO="${SQUINT_REPO:-"$( cd -- "$REPRO_REPO/../squint" &> /dev/null && pwd )"}"
VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
SILVER_ROOT="${SILVER_ROOT:-/nfs/team361/sb75/DATASETS/silver}"
INPUT_H5AD="${INPUT_H5AD:-$SILVER_ROOT/squint_hln/cosmx_human_lymph_node.h5ad}"
N_TOP="${N_TOP:-2000}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs/squint_hln_genesets}"

CPU_QUEUE="${CPU_QUEUE:-normal}"; CPU_GROUP="${CPU_GROUP:-team361}"
CPU_MEM_MB="${CPU_MEM_MB:-128000}"; CPU_CORES="${CPU_CORES:-4}"; CPU_WALL="${CPU_WALL:-8:00}"
GPU_QUEUE="${GPU_QUEUE:-gpu-lotfollahi}"
DRY_RUN="${DRY_RUN:-0}"

REF_PREFIX="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+filmscale+"

# gene-set tag -> silver dataset name (case, not assoc array, for bash 3.2 compat)
dataset_of() {
    case "$1" in
        all) echo "squint_hln_allgenes" ;;
        hvg) echo "squint_hln_hvg2k" ;;
        svg) echo "squint_hln_svg2k" ;;
        *)   echo "" ;;
    esac
}

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: VENV_PATH=$VENV_PATH is not a venv (no bin/activate)." >&2; exit 1
fi
mkdir -p "$LOG_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"

echo "=========================================================="
echo "squint_hln gene-set sweep   gene_sets=$GENE_SETS  seeds=$SEEDS"
echo "  preprocess=$DO_PREPROCESS  skip_build=$SKIP_BUILD  skip_train=$SKIP_TRAIN"
echo "  SQUINT_REPO=$SQUINT_REPO"
echo "  SILVER_ROOT=$SILVER_ROOT   N_TOP=$N_TOP"
echo "  CPU: -q $CPU_QUEUE -G $CPU_GROUP -M $CPU_MEM_MB    GPU train queue: $GPU_QUEUE"
echo "=========================================================="

# ---- optional preprocessing (one CPU job, produces all requested silver dirs)
PREP_JOB=""
if [[ "$DO_PREPROCESS" == "1" ]]; then
    PREP_JOB="hln-prep-$TS"
    PREP_CMD="source '$VENV_PATH/bin/activate'; cd '$REPRO_REPO'; \
python analysis/data_preparation/preprocess_squint_hln.py \
--input '$INPUT_H5AD' --silver-root '$SILVER_ROOT' --gene-sets '$GENE_SETS' --n-top $N_TOP"
    BSUB=( bsub -G "$CPU_GROUP" -q "$CPU_QUEUE" -n "$CPU_CORES" -M "$CPU_MEM_MB"
           -R "select[mem>$CPU_MEM_MB] rusage[mem=$CPU_MEM_MB] span[hosts=1]"
           -W "$CPU_WALL" -J "$PREP_JOB"
           -o "$LOG_ROOT/preprocess.out" -e "$LOG_ROOT/preprocess.err" )
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${BSUB[@]}"; printf 'bash -lc %q\n' "$PREP_CMD"
    else
        "${BSUB[@]}" bash -lc "$PREP_CMD"
    fi
fi

IFS=',' read -ra SETS <<< "$GENE_SETS"
for gs in "${SETS[@]}"; do
    gs="$(echo "$gs" | xargs)"; [[ -z "$gs" ]] && continue
    ds="$(dataset_of "$gs")"
    if [[ -z "$ds" ]]; then echo "skip unknown gene set '$gs'"; continue; fi
    variant="${REF_PREFIX}${ds}"

    # ---- build blob (depends on preprocess if requested) ----
    BUILD_JOB="hln-build-${ds}-$TS"
    if [[ "$SKIP_BUILD" != "1" ]]; then
        BUILD_CMD="source '$VENV_PATH/bin/activate'; cd '$SQUINT_REPO'; \
python examples/run_squint.py --build-blob --build-blob-dataset '$ds'"
        BSUB=( bsub -G "$CPU_GROUP" -q "$CPU_QUEUE" -n "$CPU_CORES" -M "$CPU_MEM_MB"
               -R "select[mem>$CPU_MEM_MB] rusage[mem=$CPU_MEM_MB] span[hosts=1]"
               -W "$CPU_WALL" -J "$BUILD_JOB"
               -o "$LOG_ROOT/build_${ds}.out" -e "$LOG_ROOT/build_${ds}.err" )
        [[ -n "$PREP_JOB" ]] && BSUB+=( -w "ended(\"$PREP_JOB\")" )
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '%q ' "${BSUB[@]}"; printf 'bash -lc %q\n' "$BUILD_CMD"
        else
            "${BSUB[@]}" bash -lc "$BUILD_CMD"
        fi
    fi

    # ---- training: a small dependent job that runs submit_multi_seed.sh once
    #      the blob is built (which in turn submits the 5 GPU seed jobs) -------
    if [[ "$SKIP_TRAIN" != "1" ]]; then
        TRAIN_LAUNCH_CMD="source '$VENV_PATH/bin/activate'; cd '$SQUINT_REPO'; \
LSF_QUEUE='$GPU_QUEUE' bash examples/submit_multi_seed.sh '$variant' '$SEEDS'"
        TBSUB=( bsub -G "$CPU_GROUP" -q "$CPU_QUEUE" -n 1 -M 4000
                -R "select[mem>4000] rusage[mem=4000]" -W "0:30"
                -J "hln-trainlaunch-${ds}-$TS"
                -o "$LOG_ROOT/trainlaunch_${ds}.out" -e "$LOG_ROOT/trainlaunch_${ds}.err" )
        [[ "$SKIP_BUILD" != "1" ]] && TBSUB+=( -w "ended(\"$BUILD_JOB\")" )
        if [[ "$DRY_RUN" == "1" ]]; then
            printf '%q ' "${TBSUB[@]}"; printf 'bash -lc %q\n' "$TRAIN_LAUNCH_CMD"
        else
            "${TBSUB[@]}" bash -lc "$TRAIN_LAUNCH_CMD"
        fi
        echo "  variant: $variant"
    fi
done

echo "----------------------------------------------------------"
echo "Submitted. Chain per gene set: [preprocess] -> build blob -> submit_multi_seed (5 GPU seeds)."
echo "Logs under: $LOG_ROOT"
[[ "$DRY_RUN" == "1" ]] && echo "(DRY_RUN: nothing was actually submitted)"
