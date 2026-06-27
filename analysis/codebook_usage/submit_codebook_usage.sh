#!/usr/bin/env bash
# submit_codebook_usage.sh
# -----------------------------------------------------------------------------
# Submit the codebook-usage report (addresses the "codebook usage" reviewer comment) to LSF.
#
# Reports active-code fraction + perplexity per (L, K) level for both VQ
# branches of a trained SQUINT run, to confirm the EMA dead-code reinit
# prevents codebook collapse. It only READS the run's predicted_adata.h5ad,
# so this is a CPU-only job (no GPU, no model reload).
#
# Usage:
#   bash submit_codebook_usage.sh [VARIANT] [-- <extra args to the .py>]
#
#   VARIANT  Optional. A registered variant key. Defaults to the s49_v23
#            mouse-brain winner the reviewer asked about. The script reports
#            on that variant's LATEST run unless you pass --timestamp.
#
#   Anything after a literal `--` is forwarded verbatim to
#   report_codebook_usage.py (e.g. --timestamp 20260601_120000,
#   --dataset mmb0-1b_smb1-1b_1p, --run-dir DIR, --predicted-adata FILE,
#   --out-dir DIR, --no-plot).
#
# Examples:
#   bash submit_codebook_usage.sh                       # default variant, latest run
#   bash submit_codebook_usage.sh "<VARIANT>" -- --timestamp 20260601_120000
#   bash submit_codebook_usage.sh -- --predicted-adata /path/predicted_adata.h5ad
#
# Env overrides:
#   VENV_PATH    /nfs/team361/sb75/.venvs/squint
#   REPRO_REPO   <auto: this script's repo root>
#   LOG_ROOT     /nfs/team361/sb75/squint-reproducibility/artifacts/logs
#   LSF_GROUP    team361          (CPU cost group)
#   LSF_QUEUE    normal           (CPU queue)
#   LSF_CORES    2
#   LSF_MEM_MB   64000            (predicted_adata can be several GB)
#   LSF_WALL     1:00
#   DRY_RUN      0                set to 1 to print the bsub and exit
# -----------------------------------------------------------------------------
set -euo pipefail

DEFAULT_VARIANT="s49_v23_dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+contrastWB-w10-k5+mmb0-1b_smb1-1b_1p"

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

# Resolve this script's location -> repo root (analysis/codebook_usage/ -> repo).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_REPO="${REPRO_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY_SCRIPT="$SCRIPT_DIR/report_codebook_usage.py"

VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_QUEUE="${LSF_QUEUE:-normal}"
LSF_CORES="${LSF_CORES:-2}"
LSF_MEM_MB="${LSF_MEM_MB:-64000}"
LSF_WALL="${LSF_WALL:-1:00}"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/codebook_usage/${STAMP}"
JOB_NAME="cbu_${STAMP}"

# The worker command: activate the env, run the report. report_codebook_usage.py
# is self-contained (anndata/numpy/pandas/matplotlib only — no vqniche import).
WORKER=$(cat <<EOF
set -euo pipefail
source "$VENV_PATH/bin/activate"
echo "[cbu] host=\$(hostname) python=\$(which python)"
echo "[cbu] variant=$VARIANT"
python "$PY_SCRIPT" --variant "$VARIANT" ${PY_ARGS[@]+"${PY_ARGS[@]}"}
EOF
)

echo "Submitting codebook-usage report:"
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
