#!/usr/bin/env bash
# submit_compare_discrete_vs_continuous.sh
# -----------------------------------------------------------------------------
# PARALLEL discrete-vs-continuous discretization comparison: submit ONE LSF job
# PER SEED (each loads one discrete + one continuous predicted_adata.h5ad and
# runs the Leiden clustering) + a dependent AGGREGATOR job that concatenates the
# per-seed results and writes the combined summary CSV + figure. This replaces
# the old single sequential job (Leiden over all seeds was the bottleneck).
#
# 3 folds x 2 branches, per seed:
#   1) SQUINT discrete codes (L0, used directly),
#   2) SQUINT Leiden  (headline emb, Leiden-to-K=30),
#   3) Continuous Leiden (s57_v33 emb, Leiden-to-K=30).
# CPU-only (no GPU, no model reload).
#
# Usage (no args = paper defaults; --out-dir recommended):
#   bash submit_compare_discrete_vs_continuous.sh --out-dir <BASE>
#   bash submit_compare_discrete_vs_continuous.sh --out-dir <BASE> \
#       --discrete-runs   <s57_v19 __multiseed dir> \
#       --continuous-runs <s57_v33 __multiseed dir>
# --discrete-runs / --continuous-runs each take a SINGLE path (a __multiseed
# sweep dir / seed_run_index.csv / variant parent); they default (in the .py) to
# the s57_v19 (discrete) / s57_v33 (continuous) __multiseed sweeps. Any other
# flags (--match, --leiden-seed, --test, --error, --cell/niche-label-key) are
# forwarded to both the workers and the aggregator.
#
# Outputs: per-seed CSVs in <BASE>/perseed/seed_<i>/; final
# discretization_summary.csv + discretization_comparison.{svg,png,pdf} in <BASE>.
#
# Env overrides: VENV_PATH, LOG_ROOT, LSF_GROUP(team361), LSF_QUEUE(normal),
#   LSF_CORES(8), LSF_MEM_MB(128000), LSF_WALL(2:00), AGG_MEM_MB(32000),
#   AGG_WALL(1:00), OUT_BASE, DRY_RUN.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/compare_discrete_vs_continuous.py"
VENV_PATH="${VENV_PATH:-/nfs/team361/sb75/.venvs/squint}"
LOG_ROOT="${LOG_ROOT:-/nfs/team361/sb75/squint-reproducibility/artifacts/logs}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_QUEUE="${LSF_QUEUE:-normal}"
LSF_CORES="${LSF_CORES:-8}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_WALL="${LSF_WALL:-2:00}"
AGG_MEM_MB="${AGG_MEM_MB:-32000}"
AGG_WALL="${AGG_WALL:-1:00}"
DRY_RUN="${DRY_RUN:-0}"

# ---- parse args: pull out discrete/continuous/out-dir; the rest = COMMON ----
DISC=""; CONT=""; OUTBASE="${OUT_BASE:-}"; COMMON=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --discrete-runs)   DISC="$2"; shift 2;;
    --continuous-runs) CONT="$2"; shift 2;;
    --out-dir)         OUTBASE="$2"; shift 2;;
    *) COMMON+=("$1"); shift;;
  esac
done
OUTBASE="${OUTBASE:-/nfs/team361/sb75/squint-reproducibility/artifacts/mmb0-1b_smb1-1b_1p/discretization}"
COMMON_STR=""; ((${#COMMON[@]})) && COMMON_STR="$(printf ' %q' "${COMMON[@]}")"

LISTARGS=()
[[ -n "$DISC" ]] && LISTARGS+=(--discrete-runs "$DISC")
[[ -n "$CONT" ]] && LISTARGS+=(--continuous-runs "$CONT")

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/compare_discrete_vs_continuous/${STAMP}"

# ---- resolve per-seed run pairs on the login node (needs venv for pandas) ----
echo "[cmp] resolving per-seed run pairs..."
PAIRS=()
while IFS= read -r _line; do
    [[ -n "$_line" ]] && PAIRS+=("$_line")
done < <( { source "$VENV_PATH/bin/activate" 2>/dev/null || true; } ; \
    python "$PY_SCRIPT" --list-pairs ${LISTARGS[@]+"${LISTARGS[@]}"} 2>/dev/null \
    | grep '^PAIR' || true )
if [[ ${#PAIRS[@]} -eq 0 ]]; then
    echo "ERROR: --list-pairs returned no pairs. Check that the discrete/continuous" >&2
    echo "       __multiseed sweeps exist (see --discrete-runs/--continuous-runs)." >&2
    exit 1
fi
echo "[cmp] ${#PAIRS[@]} seed pair(s)  |  out-base: $OUTBASE  |  common: ${COMMON[*]:-<none>}"
[[ "$DRY_RUN" == "1" ]] || mkdir -p "$LOG_DIR"

# ---- one worker per seed ----
WORKER_IDS=()
for line in "${PAIRS[@]}"; do
    IFS=$'\t' read -r _ idx disc cont <<< "$line"
    seeddir="$OUTBASE/perseed/seed_${idx}"
    jname="cmp_dvc_${STAMP}_s${idx}"
    [[ "$DRY_RUN" == "1" ]] || mkdir -p "$seeddir"
    JOB="set -euo pipefail
source $(printf %q "$VENV_PATH")/bin/activate
python $(printf %q "$PY_SCRIPT") --discrete-runs $(printf %q "$disc") --continuous-runs $(printf %q "$cont") --no-plot --out-dir $(printf %q "$seeddir")${COMMON_STR}"
    BSUB=( bsub -J "$jname" -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$LSF_CORES" -M "$LSF_MEM_MB"
           -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB] span[hosts=1]" -W "$LSF_WALL"
           -o "$LOG_DIR/seed_${idx}.out" -e "$LOG_DIR/seed_${idx}.err" )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "--- worker seed $idx ($disc | $cont) ---"
        printf '%q ' "${BSUB[@]}"; printf 'bash -lc %q\n\n' "$JOB"
    else
        out="$("${BSUB[@]}" bash -lc "$JOB")"; echo "$out"
        id="$(echo "$out" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' | head -1)"
        [[ -n "$id" ]] && WORKER_IDS+=("$id")
    fi
done

# ---- dependent aggregator (concatenate per-seed CSVs -> summary + figure) ----
AGG="set -euo pipefail
source $(printf %q "$VENV_PATH")/bin/activate
python $(printf %q "$PY_SCRIPT") --aggregate $(printf %q "$OUTBASE/perseed/*") --out-dir $(printf %q "$OUTBASE")${COMMON_STR}"
AGG_BSUB=( bsub -J "cmp_dvc_${STAMP}_agg" -G "$LSF_GROUP" -q "$LSF_QUEUE" -n 2 -M "$AGG_MEM_MB"
           -R "select[mem>$AGG_MEM_MB] rusage[mem=$AGG_MEM_MB] span[hosts=1]" -W "$AGG_WALL"
           -o "$LOG_DIR/aggregate.out" -e "$LOG_DIR/aggregate.err" )
if [[ "$DRY_RUN" == "1" ]]; then
    echo "--- aggregator (waits for all workers) ---"
    printf '%q ' "${AGG_BSUB[@]}"; printf '[-w done(<worker-ids>)] bash -lc %q\n' "$AGG"
    exit 0
fi
DEP=""
if [[ ${#WORKER_IDS[@]} -gt 0 ]]; then
    DEP="$(printf 'done(%s)&&' "${WORKER_IDS[@]}")"; DEP="${DEP%&&}"
    AGG_BSUB+=( -w "$DEP" )
fi
"${AGG_BSUB[@]}" bash -lc "$AGG"
echo "Submitted ${#WORKER_IDS[@]} per-seed worker(s) + 1 aggregator."
echo "  dependency: ${DEP:-<none>}"
echo "  final: $OUTBASE/discretization_summary.csv (+ discretization_comparison.*) after the aggregator runs."
echo "  logs:  $LOG_DIR/"
