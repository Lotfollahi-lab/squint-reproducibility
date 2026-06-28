#!/usr/bin/env bash
# submit_analyze_run.sh
# -----------------------------------------------------------------------------
# Submit the UNIFIED post-inference analysis (analyze_run.py) to LSF (CPU-only).
# Generates ALL per-run plots — code_index_plots / umap_plots / svg_plots /
# codebook_usage_plots — for one run or every seed of a variant. It only READS
# predicted_adata.h5ad (no GPU / model reload).
#
# All CLI args are forwarded verbatim to analyze_run.py, e.g.:
#   bash submit_analyze_run.sh --variant <KEY> --all-seeds
#   bash submit_analyze_run.sh --predicted-adata /path/predicted_adata.h5ad
#   bash submit_analyze_run.sh --variant <KEY> --timestamp 20260601_120000 --only umap,codebook
#
# Env overrides:
#   VENV_PATH    /nfs/team361/sb75/.venvs/squint
#   LOG_ROOT     /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   LSF_GROUP    team361      LSF_QUEUE normal   LSF_CORES 4
#   LSF_MEM_MB   96000        LSF_WALL 6:00      DRY_RUN 0
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/analyze_run.py"

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_QUEUE="${LSF_QUEUE:-normal}"
LSF_CORES="${LSF_CORES:-4}"
LSF_MEM_MB="${LSF_MEM_MB:-96000}"   # 5 seeds x 4 plot types, several-GB adatas
LSF_WALL="${LSF_WALL:-6:00}"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/analyze_run/${STAMP}"
JOB_NAME="analyze_${STAMP}"

# analyze_run.py shells out to the example plotters (squint/examples/) and the
# codebook tool with the SAME python, so activating the env covers all four.
WORKER=$(cat <<EOF
set -euo pipefail
source "$VENV_PATH/bin/activate"
echo "[analyze] host=\$(hostname) python=\$(which python)"
python "$PY_SCRIPT" $*
EOF
)

echo "Submitting unified analysis:"
echo "  script : $PY_SCRIPT"
echo "  args   : $*"
echo "  queue  : $LSF_QUEUE   group: $LSF_GROUP   cores: $LSF_CORES   mem: ${LSF_MEM_MB}MB   wall: $LSF_WALL"
echo "  logs   : $LOG_DIR/{out,err}.log"

BSUB_ARGS=(
    -J "$JOB_NAME" -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$LSF_CORES" -M "$LSF_MEM_MB"
    -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB] span[hosts=1]"
    -W "$LSF_WALL" -o "$LOG_DIR/out.log" -e "$LOG_DIR/err.log"
)

if [[ "$DRY_RUN" == "1" ]]; then
    echo "----- DRY RUN -----"; printf 'bsub'; printf ' %q' "${BSUB_ARGS[@]}"
    printf ' bash -lc <WORKER>\n'; echo "----- WORKER -----"; echo "$WORKER"; exit 0
fi

mkdir -p "$LOG_DIR"
bsub "${BSUB_ARGS[@]}" bash -lc "$WORKER"
echo "Submitted. Tail logs with:  tail -f $LOG_DIR/out.log"
