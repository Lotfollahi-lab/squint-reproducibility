#!/bin/bash
# =============================================================================
# submit_recompute_ilisi.sh: corrected global iLISI as LSF jobs, one per
# (dataset, method or variant), plus one dependent merge job.
#
# WHY: see the docstring of recompute_ilisi.py. In short, the iLISI in Tables 1,
# 3 and S1 was computed (1) on rows in file order, which can understate SQUINT's
# quantized embeddings because exactly tied cells pick same-section neighbours,
# and (2) for four Table 1 cells (GraphST Mouse Brain / Eczema, scGPT and
# scGPT-spatial Mouse Brain) with an unscaled fallback statistic, because
# scib_metrics was not importable in those methods' environments. This
# recomputes the affected values in ONE environment that has scib_metrics, with
# rows permuted, and keeps NMI / ARI / MMD / RT exactly as published.
#
# STEPS
#   0. hard check that scib_metrics and pynndescent import in $VENV (there is no
#      fallback, by design: the fallback is fault 2)
#   1. manifest, once, on this host (cheap: csv reads + h5py header probes):
#      $OUT/manifest.csv and $OUT/manifest_coverage.txt
#   2. one bsub per job that still has unfinished tasks and is not already
#      pending/running in LSF (CPU only; no GPU is needed: NNDescent, the exact
#      kNN and scib's jax LISI all run on CPU)
#   3. one merge job that waits for all of them (SUBMIT_MERGE=1), else run the
#      merge command printed at the end
#
# OUTPUT: only under $OUT (default $REPO/artifacts/ilisi_recompute, a NEW
# subfolder created on first use; the Python refuses any folder it did not
# create, in DRY_RUN too). Nothing is ever overwritten: finished tasks are
# skipped; with FORCE=1 they are recomputed and, after a successful recompute,
# the old result is moved to results/superseded/; every merge writes a new
# merged/<UTC timestamp>/ folder; LSF logs are appended (-o). No __pycache__ is
# written into the repo (PYTHONDONTWRITEBYTECODE=1).
#
# RESOURCES per dataset, following submit_label_metrics.sh's global-graph class:
#   chl59 with full 199,672-cell graphs (baselines)  long    256 GB  (128 GB hit
#                                                             TERM_MEMLIMIT for the
#                                                             global iLISI before)
#   chl59 SQUINT (100k subset), mouse brain          normal   96 GB
#   eczema                                           normal   64 GB
# Override with LSF_QUEUE= / LSF_WALL= / LSF_MEM=; check a queue's ceiling with
#   bqueues -l <queue> | grep -i runlimit
#
# KNOBS
#   SANITY=none|seed0|all  continuous, unaffected rows (sanity recomputes only).
#                          Building the manifest: default seed0. Once it exists,
#                          SANITY= is passed to plan and compute as an override
#                          (no rebuild needed); unset = as built. SANITY=none skips
#                          e.g. the 256 GB NSCLC baseline sanity jobs.
#   DISC_CSV=<path>        discretization_per_seed.csv behind Table 3's
#                          Discretization block (default: searched and validated
#                          against the printed iLISI; see manifest_coverage.txt)
#   ONLY=<substring>       only job ids containing it (e.g. chl59.squint)
#   DATASETS="<tags>"      only these dataset tags (every job of the dataset,
#                          sanity jobs included)
#   KNN=nndescent|exact|both, N_PERM=3, GRAIN=method|task, FORCE=1, DRY_RUN=1,
#   SUBMIT_MERGE=0, SELFTEST=1 (runs the selftest as an interactive LSF job and
#   exits; nothing is written)
#
# USAGE (from an LSF submission host)
#   SELFTEST=1 bash analysis/benchmarking/ilisi_recompute/submit_recompute_ilisi.sh
#   DRY_RUN=1 bash analysis/benchmarking/ilisi_recompute/submit_recompute_ilisi.sh
#   ONLY=chl59.squint bash .../submit_recompute_ilisi.sh   # NSCLC control first (SQUINT only)
#   bash .../submit_recompute_ilisi.sh                     # everything else
#   SANITY=none bash .../submit_recompute_ilisi.sh         # ... without sanity jobs
#   KNN=nndescent N_PERM=5 bash .../submit_recompute_ilisi.sh
#   FORCE=1 ONLY=<job> bash .../submit_recompute_ilisi.sh  # recompute finished tasks
# =============================================================================
set -uo pipefail

REPO="${REPO:-/nfs/team361/sb75/squint-reproducibility}"
VENV="${VENV:-/nfs/team361/sb75/.venvs/squint}"
ART="${ART:-$REPO/artifacts}"
OUT="${OUT:-$ART/ilisi_recompute}"
# nndescent = the paper's estimator (always the primary value); both adds the
# exact kNN with per-row random tie-breaking as a robustness check. It is cheap
# on quantized embeddings (distances between distinct rows only) and computed
# once per task on tie-free ones.
KNN="${KNN:-both}"
N_PERM="${N_PERM:-3}"
SANITY="${SANITY:-}"         # empty: seed0 when building, as built afterwards
DISC_CSV="${DISC_CSV:-}"
GRAIN="${GRAIN:-method}"     # method: one job per (dataset, method); task: per task
DATASETS="${DATASETS:-}"     # restrict to these dataset tags (space separated)
ONLY="${ONLY:-}"             # restrict to job ids containing this string
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
SELFTEST="${SELFTEST:-0}"
SUBMIT_MERGE="${SUBMIT_MERGE:-1}"
JOB_PREFIX="${JOB_PREFIX:-ilisi}"
CORES="${CORES:-4}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_EXTRA="${LSF_EXTRA:-}"
LSF_QUEUE_OVERRIDE="${LSF_QUEUE:-}"
LSF_WALL_OVERRIDE="${LSF_WALL:-}"
LSF_MEM_OVERRIDE="${LSF_MEM:-}"

SCRIPT="analysis/benchmarking/ilisi_recompute/recompute_ilisi.py"
PY="$VENV/bin/python"
LOG="$OUT/logs"
DIRS="--artifacts-root $ART --out-dir $OUT"
SANITY_ARG=""; [[ -n "$SANITY" ]] && SANITY_ARG="--sanity $SANITY"
DISC_ARG=""; [[ -n "$DISC_CSV" ]] && DISC_ARG="--discretization-csv $DISC_CSV"
# Thread counts match the reservation (numba would otherwise use every core of
# the node); JAX on CPU only, so it does not probe for a GPU; no bytecode.
ENVS="JAX_PLATFORMS=cpu NUMBA_NUM_THREADS=$CORES OMP_NUM_THREADS=$CORES \
OPENBLAS_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES PYTHONDONTWRITEBYTECODE=1"
export PYTHONDONTWRITEBYTECODE=1

case "$SANITY" in ""|none|seed0|all) ;; *) echo "!! SANITY=$SANITY: none|seed0|all"; exit 2 ;; esac
# Outputs go ONLY to a subfolder of the artifacts root.
case "$OUT/" in
    "$ART"/?*/) ;;
    *) echo "!! OUT=$OUT must be a subfolder of $ART; nothing written."; exit 2 ;;
esac
cd "$REPO" || { echo "!! no repo at $REPO"; exit 2; }

# ---- 0. scib_metrics or nothing ---------------------------------------------
if ! "$PY" - <<'PY'
import scib_metrics, pynndescent
from scib_metrics import ilisi_knn                        # noqa: F401
from scib_metrics.nearest_neighbors import NeighborsResults  # noqa: F401
print(f"== scib_metrics {scib_metrics.__version__}, "
      f"pynndescent {pynndescent.__version__}")
PY
then
    echo "!! scib_metrics / pynndescent not importable in $VENV: refusing to submit."
    exit 2
fi

# ---- selftest: an interactive LSF job, never the submission host -------------
if [[ "$SELFTEST" == "1" ]]; then
    ST_CMD="source $VENV/bin/activate && cd $REPO && export $ENVS && \
python -u $SCRIPT selftest"
    ST=(bsub -I -G "$LSF_GROUP" -q "${LSF_QUEUE_OVERRIDE:-normal}"
        -W "${LSF_WALL_OVERRIDE:-1:00}" -n "$CORES" -M "${LSF_MEM_OVERRIDE:-16000}"
        -R "select[mem>${LSF_MEM_OVERRIDE:-16000}]"
        -R "rusage[mem=${LSF_MEM_OVERRIDE:-16000}]" -R "span[hosts=1]")
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "would run: ${ST[*]} $LSF_EXTRA \"$ST_CMD\""
        exit 0
    fi
    # shellcheck disable=SC2086
    "${ST[@]}" $LSF_EXTRA "$ST_CMD"
    exit $?
fi

# ---- 1. manifest + job plan -------------------------------------------------
PLAN_ALL=""; [[ "$FORCE" == "1" ]] && PLAN_ALL="--all"
if [[ -f "$OUT/manifest.csv" ]]; then
    echo "== manifest exists: $OUT/manifest.csv"
    echo "   sanity rows: ${SANITY:-as built} (override with SANITY=none|seed0|all)"
    echo "   (rebuild: $PY $SCRIPT manifest $DIRS --force --sanity ${SANITY:-seed0}${DISC_ARG:+ $DISC_ARG})"
    # shellcheck disable=SC2086
    PLAN=$("$PY" "$SCRIPT" plan $DIRS --grain "$GRAIN" $PLAN_ALL $SANITY_ARG) || exit 2
elif [[ "$DRY_RUN" == "1" ]]; then
    # shellcheck disable=SC2086
    PLAN=$("$PY" "$SCRIPT" manifest $DIRS --dry-run --sanity "${SANITY:-seed0}" \
               --grain "$GRAIN" $DISC_ARG | tee /dev/stderr \
           | { grep '^PLAN' || true; }) || exit 2
else
    # shellcheck disable=SC2086
    "$PY" "$SCRIPT" manifest $DIRS --sanity "${SANITY:-seed0}" $DISC_ARG || exit 2
    # shellcheck disable=SC2086
    PLAN=$("$PY" "$SCRIPT" plan $DIRS --grain "$GRAIN" $PLAN_ALL $SANITY_ARG) || exit 2
fi
[[ "$DRY_RUN" == "1" ]] || mkdir -p "$LOG"

resources () {   # resources <dataset> <largest cell set in the job> -> QUEUE MEM WALL
    local DS=$1 N=$2
    QUEUE=normal; WALL=12:00
    case "$DS" in
        chl59-2b_1p)
            if (( N > 100000 )); then QUEUE=long; MEM=256000; else MEM=96000; fi ;;
        mmb0-1b_smb1-1b_1p) MEM=96000 ;;
        *)                  MEM=64000 ;;
    esac
    [[ -n "$LSF_QUEUE_OVERRIDE" ]] && QUEUE="$LSF_QUEUE_OVERRIDE"
    [[ -n "$LSF_WALL_OVERRIDE" ]] && WALL="$LSF_WALL_OVERRIDE"
    [[ -n "$LSF_MEM_OVERRIDE" ]] && MEM="$LSF_MEM_OVERRIDE"
}

queued () {   # queued <job name>: pending or running in LSF already?
    command -v bjobs >/dev/null 2>&1 || return 1
    bjobs -noheader -J "$1" 2>/dev/null | awk '{print $3}' \
        | grep -qE '^(PEND|RUN|PSUSP|USUSP|SSUSP)$'
}

# ---- 2. one job per (dataset, method / variant) -----------------------------
NSUB=0; NFILT=0; NQUEUED=0
XF=""; [[ "$FORCE" == "1" ]] && XF=" --force"
echo
printf '%-44s %-20s %6s %5s %-7s %7s\n' JOB DATASET TASKS TODO QUEUE MEM_MB
while IFS=$'\t' read -r TAG JOB DS NT NTODO MAXN; do
    [[ "$TAG" == "PLAN" ]] || continue
    if [[ -n "$DATASETS" && " $DATASETS " != *" $DS "* ]] \
       || [[ -n "$ONLY" && "$JOB" != *"$ONLY"* ]]; then
        NFILT=$((NFILT+1)); continue
    fi
    resources "$DS" "$MAXN"
    if queued "${JOB_PREFIX}_${JOB}"; then
        printf '%-44s %-20s %6s %5s %s\n' "$JOB" "$DS" "$NT" "$NTODO" \
               "already queued in LSF: not resubmitted"
        NQUEUED=$((NQUEUED+1)); continue
    fi
    printf '%-44s %-20s %6s %5s %-7s %7s\n' "$JOB" "$DS" "$NT" "$NTODO" "$QUEUE" "$MEM"
    CMD="source $VENV/bin/activate && cd $REPO && export $ENVS && python -u \
$SCRIPT compute $DIRS --job $JOB --knn $KNN --n-perm $N_PERM $SANITY_ARG$XF"
    BSUB=(bsub -G "$LSF_GROUP" -q "$QUEUE" -J "${JOB_PREFIX}_${JOB}" -W "$WALL"
          -n "$CORES" -M "$MEM" -R "select[mem>$MEM]" -R "rusage[mem=$MEM]"
          -R "span[hosts=1]")
    NSUB=$((NSUB+1))
    if [[ "$DRY_RUN" == "1" ]]; then
        ((NSUB == 1)) && echo "   e.g. ${BSUB[*]} $LSF_EXTRA -o $LOG/$JOB.out" \
                              "-e $LOG/$JOB.err \"$CMD\""
        continue
    fi
    # shellcheck disable=SC2086
    "${BSUB[@]}" $LSF_EXTRA -o "$LOG/$JOB.out" -e "$LOG/$JOB.err" "$CMD" \
        || echo "!! bsub failed for $JOB"
done <<< "$PLAN"
echo
echo "== $NSUB job(s) $([[ "$DRY_RUN" == "1" ]] && echo 'would be ')submitted;" \
     "$NQUEUED already queued; $NFILT filtered out by DATASETS/ONLY"

# ---- 3. merge ---------------------------------------------------------------
MERGE="$PY $SCRIPT merge $DIRS"
if [[ "$DRY_RUN" != "1" && "$SUBMIT_MERGE" == "1" && "$NSUB" -gt 0 ]]; then
    # merge_<prefix> does not match the ${JOB_PREFIX}_* wildcard it waits on.
    # Every merge writes its own merged/<timestamp>/, so a second one is harmless.
    bsub -G "$LSF_GROUP" -q normal -J "merge_${JOB_PREFIX}" \
         -w "ended(${JOB_PREFIX}_*)" -W 2:00 -n 1 -M 16000 \
         -R "select[mem>16000]" -R "rusage[mem=16000]" $LSF_EXTRA \
         -o "$LOG/merge.out" -e "$LOG/merge.err" \
         "source $VENV/bin/activate && cd $REPO && export PYTHONDONTWRITEBYTECODE=1 \
&& python -u $SCRIPT merge $DIRS" \
        && echo "== merge job queued (waits for ${JOB_PREFIX}_*); report in" \
                "$OUT/merged/<timestamp>/ilisi_diff_report.txt"
fi
echo
echo "merge (any time; partial results are reported as such):"
echo "  cd $REPO && $MERGE"
echo "Fig S3 b,d panels from the newest merge:"
echo "  cd $REPO && $PY $SCRIPT figures $DIRS"
echo "check the NSCLC control gate in the report BEFORE using any recomputed value."
