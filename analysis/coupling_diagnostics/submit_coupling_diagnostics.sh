#!/usr/bin/env bash
# submit_coupling_diagnostics.sh
# -----------------------------------------------------------------------------
# Submit the cell/niche coupling diagnostics to LSF (CPU-only).
#
# Measures, on an already-trained DECOUPLED model's saved embeddings (no
# retraining, no GPU), which coupling lever could beat the decoupled baseline:
#   1. redundancy (CKA/CCA between cell & niche)        -> disentanglement (#2)?
#   2. cross-predictability (each branch leaks other's label?)
#   3. composition premise (neighbour cell-types predict niche?) -> cell->niche (#1)?
# It only READS predicted_adata.h5ad across the variant's multiseed seeds, so
# this is a CPU job (no model reload).
#
# Usage:
#   bash submit_coupling_diagnostics.sh [VARIANT] [-- <extra args to the .py>]
#
#   VARIANT  Optional. A variant key (full slug or prefix). Defaults to the
#            DECOUPLED reference s55_v3_ (the baseline the coupling sweep ties).
#   Anything after a literal `--` is forwarded to diagnose_coupling.py
#   (e.g. --timestamp 20260601_120000, --predicted-adata FILE, --out DIR).
#
# Examples:
#   bash submit_coupling_diagnostics.sh                      # s55_v3 decoupled, all seeds
#   bash submit_coupling_diagnostics.sh s56_v1_              # diagnose the coupled model too
#   bash submit_coupling_diagnostics.sh -- --predicted-adata /path/predicted_adata.h5ad
#
# Env overrides:
#   VENV_PATH    /nfs/team361/sb75/.venvs/squint
#   LOG_ROOT     /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   LSF_GROUP    team361          (CPU cost group)
#   LSF_QUEUE    normal           (CPU queue)
#   LSF_CORES    4
#   LSF_MEM_MB   64000            (predicted_adata can be several GB)
#   LSF_WALL     2:00
#   DRY_RUN      0                set to 1 to print the bsub and exit
# -----------------------------------------------------------------------------
set -euo pipefail

DEFAULT_VARIANT="s55_v3_"

# Split args at the first literal `--`: before = VARIANT, after = py passthrough.
VARIANT=""
PY_ARGS=()
seen_dd=0
for a in "$@"; do
    if [[ "$seen_dd" -eq 0 && "$a" == "--" ]]; then seen_dd=1; continue; fi
    if [[ "$seen_dd" -eq 1 ]]; then PY_ARGS+=("$a");
    elif [[ -z "$VARIANT" ]]; then VARIANT="$a"; fi
done
VARIANT="${VARIANT:-$DEFAULT_VARIANT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_REPO="${REPRO_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY_SCRIPT="$SCRIPT_DIR/diagnose_coupling.py"

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_QUEUE="${LSF_QUEUE:-normal}"
LSF_CORES="${LSF_CORES:-4}"
LSF_MEM_MB="${LSF_MEM_MB:-64000}"
LSF_WALL="${LSF_WALL:-2:00}"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/coupling_diagnostics/${STAMP}"
JOB_NAME="coupdiag_${STAMP}"

# diagnose_coupling.py is self-contained (anndata/numpy/pandas + optional
# sklearn — no vqniche/torch import).
WORKER=$(cat <<EOF
set -euo pipefail
source "$VENV_PATH/bin/activate"
echo "[coupdiag] host=\$(hostname) python=\$(which python)"
echo "[coupdiag] variant=$VARIANT"
python "$PY_SCRIPT" --variant "$VARIANT" ${PY_ARGS[@]+"${PY_ARGS[@]}"}
EOF
)

echo "Submitting coupling diagnostics:"
echo "  variant   : $VARIANT"
echo "  py args   : ${PY_ARGS[*]:-<none>}"
echo "  script    : $PY_SCRIPT"
echo "  queue     : $LSF_QUEUE   group: $LSF_GROUP   cores: $LSF_CORES   mem: ${LSF_MEM_MB}MB   wall: $LSF_WALL"
echo "  logs      : $LOG_DIR/{out,err}.log"

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
