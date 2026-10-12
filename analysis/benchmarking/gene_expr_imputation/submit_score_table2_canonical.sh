#!/usr/bin/env bash
# =============================================================================
# submit_score_table2_canonical.sh
# -----------------------------------------------------------------------------
# Submit score_table2_canonical.py (one CPU bsub, squint venv): re-score every
# Table 2 method on ONE canonical neighbourhood definition (directed kNN16+self
# per section on the coord-aligned silver, X_nbr = row mean of raw X), with
# SQUINT's neighbourhood reconstruction on that graph, NicheCompass rescaled to
# the X_nbr library, and the imputation parity arm (observed neighbours = true
# counts for every method), plus the S0-S5 self-checks. READ-ONLY w.r.t. every
# existing artifact: all outputs go to a NEW $OUT_DIR.
#
# Usage (fresh shell on a farm head node):
#   bash analysis/benchmarking/gene_expr_imputation/submit_score_table2_canonical.sh
#   DRY_RUN=1 bash .../submit_score_table2_canonical.sh        # print the bsub only
#   REDECODE_ROOT=<OUT_ROOT of the re-decode> bash .../submit_score_table2_canonical.sh
#   MC_ARM_DIR=<dir> NOMC_ARM_DIR=<dir> bash .../submit_score_table2_canonical.sh
#
# SQUINT imputation needs the held-out predictions, which the paper arms do not
# store: re-decode both arms with the PAPER flags plus --save-pred into a NEW
# root (codes are read from the paper stage2-ablation/decode-temp05-nonoise;
# nothing is written there). From the squint repo on a farm head node:
#   cd /nfs/team361/sb75/squint
#   ART=/nfs/team361/sb75/squint-reproducibility/artifacts DATASET=mmb0-1b_smb1-1b_1p
#   OUT_ROOT=$ART/$DATASET/stage2-ablation-savepred-last COMPARE_ROOT=$ART/$DATASET/stage2-ablation \
#     ONLY_NICHE_FROM_CELL=1 NICHE_FROM_CELL_KS="0 1000" READ_DEPTH_MODE=neighbor \
#     DECODE_EXTRA="--save-pred --stage1-ckpt-policy last" bash examples/submit_stage2_mc_sweep.sh
# (COMPARE_ROOT only reads the paper arm CSVs: the decode's own --compare-csv
# gate and its paper-CSV-mtime note land in its log and in the npz, and this
# scorer quotes that verdict in its vs-paper line.)
# READ_DEPTH_MODE=neighbor is required: submit_stage2_mc_sweep.sh and
# stage2_decode_pearson.py both default to the leaky 'true' read depth, which
# this scorer FAILs. These are the paper flags and the paper decode checkpoint
# (last.ckpt; for seeds 2 and 4 it differs from the predict-time checkpoint
# that wrote the codes): score it with ALLOW_CKPT_MISMATCH=1 (a WARN instead
# of a FAIL). The vs-paper check (re-decoded vs paper arm CSV cell rows; rel
# tol 1e-3 MC, 1e-4 No MC) is then expected to PASS, with one known exception
# that is NOT established either way: the nichecell arms (6b591df), the
# read-depth flag (6b1389a) and the hard-decode depth fix (fdca41a) all landed
# on 2026-07-02, so if the paper No MC arm predates fdca41a its hard cell
# decode used each held-out cell's TRUE library size. Its vs-paper check then
# FAILs (the message reports the paper CSV mtime against those commits and
# names this cause) and its as-stored rows, the summary 'delta' reference, are
# the leaky true-depth ones. Alternative, decode checkpoint == predict-time
# checkpoint:
#   DECODE_EXTRA="--save-pred --stage1-ckpt-policy adata" with
#   OUT_ROOT=$ART/$DATASET/stage2-ablation-savepred-adata (same COMPARE_ROOT;
#   its gate is then expected to FAIL for seeds 2 and 4); seeds 2 and 4 then
#   differ from the paper arms, reported as re-decode drift (WARN per seed and
#   a 'drift' line per method in the summary).
# Then point REDECODE_ROOT at that OUT_ROOT (or MC_ARM_DIR / NOMC_ARM_DIR at
# the two arm folders). Without the npz the canonical SQUINT imputation rows
# are skipped and flagged FAIL (everything else still runs). The as-stored
# SQUINT imputation rows always come from the paper arms (ASSTORED_*_ARM_DIR).
# Full farm runbook: README of this repository, "Canonical Table 2 scoring".
#
# Runtime: a few hours on 8 cores (literal build_pearson_dataframe on all 40
# stored h5ads + 5 SQUINT predict replays); FAST_SELFCHECKS=1 roughly halves it.
# A self-check FAIL does not stop the run; the job then exits 1.
#
# Every variable is defined here (env overrides allowed). Outputs:
#   $OUT_DIR/{reconstruction,imputation}_benchmark_panel_{cell,niche}.csv
#   $OUT_DIR/imputation_benchmark_panel_niche_ownfill.csv
#   $OUT_DIR/reconstruction_benchmark_panel_niche_reference.csv
#   $OUT_DIR/asstored/*.csv (current Table 2), redecoded/*.csv (re-decoded
#   SQUINT arms' own rows), per_seed/, summary_means.csv,
#   $OUT_DIR/selfcheck_report.txt, run_config.json
# Then:  python analysis/benchmarking/compute_table2_significance.py --tables-dir $OUT_DIR
# =============================================================================
set -euo pipefail

# ---- code + environment -----------------------------------------------------
VENV="${VENV:-/nfs/team361/sb75/.venvs/squint}"
REPRO="${REPRO:-/nfs/team361/sb75/squint-reproducibility}"
SQUINT_REPO="${SQUINT_REPO:-/nfs/team361/sb75/squint}"
SCRIPT="${SCRIPT:-$REPRO/analysis/benchmarking/gene_expr_imputation/score_table2_canonical.py}"

# ---- data -------------------------------------------------------------------
ART="${ART:-/nfs/team361/sb75/squint-reproducibility/artifacts}"
DATASET="${DATASET:-mmb0-1b_smb1-1b_1p}"
SILVER_DIR="${SILVER_DIR:-/lustre/scratch126/cellgen/lotfollahi/DATASETS/silver/mmb0-1b_smb1-1b_1p_coord_aligned}"
SQUINT_SPINE="${SQUINT_SPINE:-dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1}"
SQUINT_VARIANT="${SQUINT_VARIANT:-${SQUINT_SPINE}+region-holdout+${DATASET}}"
SQUINT_SEEDS="${SQUINT_SEEDS:-20260629_023326_seed0 20260629_024131_seed1 20260629_025158_seed2 20260629_025456_seed3 20260629_025518_seed4}"
SQUINT_MULTISEED_TS="${SQUINT_MULTISEED_TS:-20260628_230209}"
# re-decoded arms (with heldout_predictions.npz); default = the paper arms (no npz)
REDECODE_ROOT="${REDECODE_ROOT:-}"
_ARM_ROOT="${REDECODE_ROOT:-$ART/$DATASET/stage2-ablation}"
MC_ARM_DIR="${MC_ARM_DIR:-$_ARM_ROOT/decode-temp05-nonoise-mc1000-nichecell}"
NOMC_ARM_DIR="${NOMC_ARM_DIR:-$_ARM_ROOT/decode-temp05-nonoise-nichecell}"
# paper arms (as-stored rows = current Table 2) and the stage-2 codes they decoded
ASSTORED_MC_ARM_DIR="${ASSTORED_MC_ARM_DIR:-$ART/$DATASET/stage2-ablation/decode-temp05-nonoise-mc1000-nichecell}"
ASSTORED_NOMC_ARM_DIR="${ASSTORED_NOMC_ARM_DIR:-$ART/$DATASET/stage2-ablation/decode-temp05-nonoise-nichecell}"
STAGE2_CODES_DIR="${STAGE2_CODES_DIR:-$ART/$DATASET/stage2-ablation/decode-temp05-nonoise}"
MC_DECODE_SAMPLES="${MC_DECODE_SAMPLES:-1000}"
ALLOW_CKPT_MISMATCH="${ALLOW_CKPT_MISMATCH:-0}"   # 1 = decode ckpt != predict-time ckpt is a WARN
# baseline variant folders (each must hold exactly ONE timestamp run)
BASE_SCVI="${BASE_SCVI:-$ART/$DATASET/baseline-scvi+region-holdout+coord-aligned}"
BASE_SCVI_NBR="${BASE_SCVI_NBR:-$ART/$DATASET/baseline-scvi-nbr+region-holdout+coord-aligned}"
BASE_NICHECOMPASS="${BASE_NICHECOMPASS:-$ART/$DATASET/baseline-nichecompass+region-holdout+coord-aligned}"
BASE_VANILLA_CELL="${BASE_VANILLA_CELL:-$ART/$DATASET/vanilla-vq-cell+region-holdout+coord-aligned}"
BASE_VANILLA_NBR="${BASE_VANILLA_NBR:-$ART/$DATASET/vanilla-vq-nbr+region-holdout+coord-aligned}"
BASE_GEST="${BASE_GEST:-$ART/$DATASET/gest-imputed+region-holdout+coord-aligned}"
BASE_KNN="${BASE_KNN:-$ART/$DATASET/baseline-knn-spatial+region-holdout+coord-aligned}"
# optional: folder with the existing panel CSVs (e.g. a copy of the paper's
# tables/table2_coord_aligned) for the S5 check; empty = skip S5
EXISTING_TABLES_DIR="${EXISTING_TABLES_DIR:-}"

# ---- scoring knobs ----------------------------------------------------------
NBR_K="${NBR_K:-16}"
DEVICE="${DEVICE:-cpu}"
NTHREADS="${NTHREADS:-8}"
FAST_SELFCHECKS="${FAST_SELFCHECKS:-0}"     # 1 = literal build_pearson_dataframe on seed 0 only
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ART/table2_canonical/$TS}"

# ---- LSF (CPU) --------------------------------------------------------------
LSF_QUEUE="${LSF_QUEUE:-normal}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_MEM_MB="${LSF_MEM_MB:-64000}"
LSF_WALL="${LSF_WALL:-12:00}"
LOG_DIR="${LOG_DIR:-$ART/logs/$DATASET/table2-canonical}"
JOB_NAME="${JOB_NAME:-t2-canonical}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA=""
if [[ "$FAST_SELFCHECKS" == "1" ]]; then EXTRA="$EXTRA --fast-selfchecks"; fi
if [[ "$ALLOW_CKPT_MISMATCH" == "1" ]]; then EXTRA="$EXTRA --allow-ckpt-mismatch"; fi
if [[ -n "$EXISTING_TABLES_DIR" ]]; then
    EXTRA="$EXTRA --existing-tables-dir $EXISTING_TABLES_DIR"
fi

echo "=========================================================="
echo "Table 2 canonical scoring (CPU, read-only on inputs)"
echo "  script        : $SCRIPT"
echo "  venv          : $VENV"
echo "  squint repo   : $SQUINT_REPO"
echo "  silver        : $SILVER_DIR"
echo "  artifacts     : $ART/$DATASET"
echo "  SQUINT variant: $SQUINT_VARIANT"
echo "  SQUINT seeds  : $SQUINT_SEEDS"
echo "  multiseed CSV : $ART/$DATASET/${SQUINT_VARIANT}__multiseed/$SQUINT_MULTISEED_TS/metrics/per_seed_pearson_reconstruction.csv"
echo "  MC arm        : $MC_ARM_DIR   (re-decoded, needs npz)"
echo "  No-MC arm     : $NOMC_ARM_DIR   (re-decoded, needs npz)"
echo "  paper MC arm  : $ASSTORED_MC_ARM_DIR"
echo "  paper No-MC   : $ASSTORED_NOMC_ARM_DIR"
echo "  stage-2 codes : $STAGE2_CODES_DIR   (expected MC K=$MC_DECODE_SAMPLES)"
echo "  allow ckpt mismatch: $ALLOW_CKPT_MISMATCH"
echo "  baselines     : scvi=$BASE_SCVI"
echo "                  scvi-nbr=$BASE_SCVI_NBR"
echo "                  nichecompass=$BASE_NICHECOMPASS"
echo "                  vanilla-cell=$BASE_VANILLA_CELL"
echo "                  vanilla-nbr=$BASE_VANILLA_NBR"
echo "                  gest=$BASE_GEST"
echo "                  knn=$BASE_KNN"
echo "  existing tabs : ${EXISTING_TABLES_DIR:-<none: S5 skipped>}"
echo "  nbr-k=$NBR_K device=$DEVICE threads=$NTHREADS fast-selfchecks=$FAST_SELFCHECKS"
echo "  OUT_DIR       : $OUT_DIR"
echo "  logs          : $LOG_DIR/table2_canonical_${TS}.{out,err}"
echo "  LSF           : -q $LSF_QUEUE -G $LSF_GROUP -n $NTHREADS -M $LSF_MEM_MB -W $LSF_WALL"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================================="

# Non-fatal pre-flight: the arms' npz files (SQUINT imputation rows need them).
for arm in "$MC_ARM_DIR" "$NOMC_ARM_DIR"; do
    n_npz=0
    for s in $SQUINT_SEEDS; do
        if [[ -f "$arm/$s/metrics/heldout_predictions.npz" ]]; then n_npz=$((n_npz + 1)); fi
    done
    n_seeds=$(echo $SQUINT_SEEDS | wc -w | tr -d ' ')
    if [[ "$n_npz" != "$n_seeds" ]]; then
        echo "WARNING: $n_npz/$n_seeds heldout_predictions.npz under $arm (re-decode the arms" \
             "as in this script's header and set REDECODE_ROOT, or MC_ARM_DIR / NOMC_ARM_DIR)." >&2
    fi
done
if [[ -e "$OUT_DIR" ]] && [[ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]]; then
    echo "REFUSING: OUT_DIR $OUT_DIR exists and is not empty." >&2
    exit 1
fi

read -r -d '' JOB <<EOF || true
set -euo pipefail
source "$VENV/bin/activate"
export PYTHONPATH="$SQUINT_REPO/src:$REPRO/analysis/benchmarking/gene_expr_imputation:\${PYTHONPATH:-}"
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS OPENBLAS_NUM_THREADS=$NTHREADS
export MPLBACKEND=Agg
unset SQUINT_EXCLUDE_BATCHES
echo "squint repo : \$(git -C "$SQUINT_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || true) \$(git -C "$SQUINT_REPO" log -1 --format='%h %ci %s' 2>/dev/null || true)"
echo "repro repo  : \$(git -C "$REPRO" rev-parse --abbrev-ref HEAD 2>/dev/null || true) \$(git -C "$REPRO" log -1 --format='%h %ci %s' 2>/dev/null || true)"
echo "host        : \$(hostname)"
rc=0
python "$SCRIPT" \\
  --artifacts-root "$ART" --dataset-tag "$DATASET" --silver-dir "$SILVER_DIR" \\
  --squint-variant "$SQUINT_VARIANT" --squint-seeds $SQUINT_SEEDS \\
  --squint-multiseed-ts "$SQUINT_MULTISEED_TS" \\
  --mc-arm-dir "$MC_ARM_DIR" --nomc-arm-dir "$NOMC_ARM_DIR" \\
  --asstored-mc-arm-dir "$ASSTORED_MC_ARM_DIR" --asstored-nomc-arm-dir "$ASSTORED_NOMC_ARM_DIR" \\
  --stage2-codes-dir "$STAGE2_CODES_DIR" --mc-decode-samples $MC_DECODE_SAMPLES \\
  --baseline-dirs "scvi=$BASE_SCVI" "scvi-nbr=$BASE_SCVI_NBR" \\
    "nichecompass=$BASE_NICHECOMPASS" "vanilla-cell=$BASE_VANILLA_CELL" \\
    "vanilla-nbr=$BASE_VANILLA_NBR" "gest=$BASE_GEST" "knn=$BASE_KNN" \\
  --out-dir "$OUT_DIR" --nbr-k $NBR_K --device $DEVICE --threads $NTHREADS \\
  --squint-examples-dir "$SQUINT_REPO/examples"$EXTRA || rc=\$?
echo "[t2-canonical] exit code \$rc (1 = at least one self-check FAIL; see $OUT_DIR/selfcheck_report.txt)"
exit \$rc
EOF

BSUB=( bsub -q "$LSF_QUEUE" -G "$LSF_GROUP" -n "$NTHREADS" -M "$LSF_MEM_MB"
       -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB]" -R "span[hosts=1]"
       -W "$LSF_WALL" -J "$JOB_NAME"
       -o "$LOG_DIR/table2_canonical_${TS}.out" -e "$LOG_DIR/table2_canonical_${TS}.err" )

if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${BSUB[@]}"; printf 'bash -lc %q\n' "$JOB"
    echo "[t2-canonical] DRY_RUN: nothing submitted."
    exit 0
fi
for p in "$VENV/bin/activate" "$SCRIPT" "$SQUINT_REPO/examples/stage2_decode_pearson.py"; do
    [[ -e "$p" ]] || { echo "MISSING: $p" >&2; exit 1; }
done
mkdir -p "$LOG_DIR"
"${BSUB[@]}" bash -lc "$JOB"
echo "[t2-canonical] submitted; outputs -> $OUT_DIR"
echo "  then: python $REPRO/analysis/benchmarking/compute_table2_significance.py --tables-dir $OUT_DIR"
