#!/usr/bin/env bash
# submit_compare_discrete_vs_continuous.sh
# -----------------------------------------------------------------------------
# Submit the discrete-vs-continuous run comparison to LSF.
#
# Compares a discrete dual-VQ run against its continuous-latent twin on the
# FAIR footing (k-means clustering of the per-cell embeddings vs ground-truth
# cell-type / niche labels, identical for both), plus reconstruction + batch-
# integration metrics from each run's metrics/*.csv. CPU-only (reads the two
# predicted_adata.h5ad files; no GPU, no model reload).
#
# Usage:
#   bash submit_compare_discrete_vs_continuous.sh [<args forwarded to the .py>]
#
# With NO args it compares the two default runs baked into the .py
# (the s49_v23 discrete run @20260513_223846 and the s53_v1 continuous run
# @20260627_074514). Forward overrides verbatim, e.g.:
#   bash submit_compare_discrete_vs_continuous.sh \
#        --continuous-run /path/to/<timestamp> --discrete-run /path/to/<timestamp>
#   bash submit_compare_discrete_vs_continuous.sh --out-dir /path/out
#
# Env overrides:
#   VENV_PATH   /nfs/team361/sb75/.venvs/squint
#   LOG_ROOT    /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   LSF_GROUP   team361
#   LSF_QUEUE   normal
#   LSF_CORES   4
#   LSF_MEM_MB  128000
#   LSF_WALL    2:00
#   DRY_RUN     0
# -----------------------------------------------------------------------------
set -euo pipefail

PY_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/compare_discrete_vs_continuous.py"

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_QUEUE="${LSF_QUEUE:-normal}"
LSF_CORES="${LSF_CORES:-4}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_WALL="${LSF_WALL:-2:00}"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/compare_discrete_vs_continuous/${STAMP}"
JOB_NAME="cmp_dvc_${STAMP}"

WORKER=$(cat <<EOF
set -euo pipefail
source "$VENV_PATH/bin/activate"
echo "[cmp] host=\$(hostname) python=\$(which python)"
python "$PY_SCRIPT" ${PY_ARGS[@]+"${PY_ARGS[@]}"}
EOF
)

echo "Submitting discrete-vs-continuous comparison:"
echo "  script : $PY_SCRIPT"
echo "  py args: ${PY_ARGS[*]:-<defaults: s49_v23 @20260513_223846 vs s53_v1 @20260627_074514>}"
echo "  queue  : $LSF_QUEUE   group: $LSF_GROUP   cores: $LSF_CORES   mem: ${LSF_MEM_MB}MB   wall: $LSF_WALL"
echo "  logs   : $LOG_DIR/{out,err}.log"

BSUB_ARGS=(
    -J "$JOB_NAME"
    -G "$LSF_GROUP"
    -q "$LSF_QUEUE"
    -n "$LSF_CORES"
    -M "$LSF_MEM_MB"
    -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB] span[hosts=1]"
    -W "$LSF_WALL"
    -o "$LOG_DIR/out.log"
    -e "$LOG_DIR/err.log"
)

if [[ "$DRY_RUN" == "1" ]]; then
    echo "----- DRY RUN: bsub command -----"
    printf 'bsub'; printf ' %q' "${BSUB_ARGS[@]}"; printf ' bash -lc <WORKER>\n'
    echo "----- WORKER -----"; echo "$WORKER"
    exit 0
fi

mkdir -p "$LOG_DIR"
bsub "${BSUB_ARGS[@]}" bash -lc "$WORKER"
echo "Submitted. Tail logs with:  tail -f $LOG_DIR/out.log"
