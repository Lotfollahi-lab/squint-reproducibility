#!/usr/bin/env bash
# submit_imputation_recon.sh
# -----------------------------------------------------------------------------
# Re-run the imputation + reconstruction methods so their per-seed CSVs gain the
# FULL metric panel (Pearson/Spearman/MSE/RMSE on log1p+counts × gene/cell/HVG/
# markers, zero/nonzero AUROC+AUPRC) AND a neighborhood branch (X_hat_nbr vs
# X_nbr — native for NicheCompass / *_nbr models, aggregated from the cell
# prediction for the cell-only methods via add_neighborhood_layers). The metric
# code lives in the shared _holdout_utils, so a plain re-run regenerates
# everything; no flag changes vs the original runs.
#
# Methods (one GPU bsub each, each in its OWN venv):
#   gest          run_gest.py             (GeST imputed)            venv: squint
#   gestarch      run_gest_arch_squint.py (SQUINT imputed)         venv: squint  [needs PREDICTED_ADATA]
#   scvi          run_scvi.py             (scVI cell recon)        venv: cellcharter
#   scvi-nbr      run_scvi_nbr.py         (scVI niche recon)       venv: cellcharter
#   nichecompass  run_nichecompass.py     (NicheCompass recon)     venv: nichecompass
#   vanilla-cell  run_vanilla_vq_cell.py  (Vanilla VQ cell recon)  venv: squint
#   vanilla-nbr   run_vanilla_vq_nbr.py   (Vanilla VQ niche recon) venv: squint
#
# Usage:
#   bash gene_expr_imputation/submit_imputation_recon.sh                  # all 7
#   METHODS="gest gestarch" bash .../submit_imputation_recon.sh
#   DRY_RUN=1 bash .../submit_imputation_recon.sh                         # print bsubs
#   DATASET=mmb0-1b_smb1-1b_1p SEEDS=0,1,2,3,4 bash .../submit_imputation_recon.sh
#
# Env overrides: DATASET, SEEDS, SILVER_DIR, PREDICTED_ADATA, NC_SPECIES,
#   NC_ORTHOLOGS, NC_MEBOCOST, VENV_ROOT, REPO, LSF_QUEUE, LSF_GROUP, LSF_GPU,
#   LSF_WALL, LSF_MEM_MB, LSF_CORES.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO="${REPO:-"$( cd -- "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"}"
VENV_ROOT="${VENV_ROOT:-/nfs/team361/sb75/.venvs}"
ART="${ART:-/nfs/team361/sb75/squint-reproducibility/artifacts}"

DATASET="${DATASET:-mmb0-1b_smb1-1b_1p}"
SEEDS="${SEEDS:-0,1,2,3,4}"
SILVER_DIR="${SILVER_DIR:-}"                 # empty -> runner's own default
# Niche-branch spatial-kNN neighbor count. Empty -> each runner's own default,
# which is now 16 (matches SQUINT's native niche graph, +knn16+, so the
# niche-level metrics are graph-consistent across ALL methods and both figures).
# Override e.g. NBR_NEIGHS=10 for the old behavior. Threaded to the RIGHT flag
# per method (native -nbr/NicheCompass use --n-spatial-neighs = their TRAINING
# target graph -> a retrain; cell-only methods use --nbr-neighs = a cheap
# eval-time re-aggregation).
NBR_NEIGHS="${NBR_NEIGHS:-}"
# gestarch frozen stage-1 predicted_adata (FiLM-scale region-holdout, seed0):
PREDICTED_ADATA="${PREDICTED_ADATA:-$ART/$DATASET/dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1+region-holdout+$DATASET/20260629_023326_seed0/predicted_adata.h5ad}"
# NicheCompass GP args (mmb = mouse). Match submit_all_benchmarks.sh defaults.
NC_SPECIES="${NC_SPECIES:-mouse}"
NC_ORTHOLOGS="${NC_ORTHOLOGS:-$REPO/analysis/benchmarking/nichecompass/human_mouse_gene_orthologs.csv}"
NC_MEBOCOST="${NC_MEBOCOST:-$REPO/analysis/benchmarking/nichecompass/metabolite_enzyme_sensor_gps}"

LSF_GROUP="${LSF_GROUP:-s10396}"
LSF_QUEUE="${LSF_QUEUE:-training-parallel}"
LSF_GPU="${LSF_GPU:-mode=exclusive_process:num=1:block=yes}"
LSF_WALL="${LSF_WALL:-24:00}"
LSF_MEM_MB="${LSF_MEM_MB:-128000}"
LSF_CORES="${LSF_CORES:-8}"
LOG_ROOT="${LOG_ROOT:-$ART/logs/$DATASET/imputation-recon}"
DRY_RUN="${DRY_RUN:-0}"

# method -> "venv|runner_rel|extra_flags". Non-zero return = unknown method.
method_spec() {
    case "$1" in
        gest)         echo "squint|gene_expr_imputation/run_gest.py|" ;;
        gestarch)     echo "squint|gene_expr_imputation/run_gest_arch_squint.py|--predicted-adata $PREDICTED_ADATA" ;;
        scvi)         echo "cellcharter|gene_expr_imputation/run_scvi.py|" ;;
        scvi-nbr)     echo "cellcharter|gene_expr_imputation/run_scvi_nbr.py|" ;;
        nichecompass) echo "nichecompass|gene_expr_imputation/run_nichecompass.py|--species $NC_SPECIES --gene-orthologs-csv $NC_ORTHOLOGS --mebocost-dir $NC_MEBOCOST" ;;
        vanilla-cell) echo "squint|gene_expr_imputation/run_vanilla_vq_cell.py|" ;;
        vanilla-nbr)  echo "squint|gene_expr_imputation/run_vanilla_vq_nbr.py|" ;;
        *) return 1 ;;
    esac
}

# method -> the neighbor-count flag it accepts (native niche models train on the
# aggregate, so their flag is --n-spatial-neighs; cell-only methods aggregate at
# eval via --nbr-neighs). Empty NBR_NEIGHS -> no flag (runner default).
neighs_flag() {
    [[ -z "$NBR_NEIGHS" ]] && { echo ""; return 0; }
    case "$1" in
        scvi-nbr|vanilla-nbr|nichecompass) echo "--n-spatial-neighs $NBR_NEIGHS" ;;
        gest|gestarch|scvi|vanilla-cell)   echo "--nbr-neighs $NBR_NEIGHS" ;;
        *) echo "" ;;
    esac
}

METHODS="${METHODS:-gest gestarch scvi scvi-nbr nichecompass vanilla-cell vanilla-nbr}"

echo "=========================================================="
echo "Re-run imputation + reconstruction (full metric panel + niche branch)"
echo "  dataset : $DATASET    seeds: $SEEDS"
echo "  methods : $METHODS"
echo "  LSF     : -G $LSF_GROUP -q $LSF_QUEUE -gpu '$LSF_GPU' -W $LSF_WALL"
echo "  DRY_RUN=$DRY_RUN"
echo "=========================================================="

[[ "$DRY_RUN" == "1" ]] || mkdir -p "$LOG_ROOT"
n_sub=0
for m in $METHODS; do
    if ! spec="$(method_spec "$m")"; then
        echo "WARN: unknown method '$m' — skip" >&2; continue
    fi
    venv_name="${spec%%|*}"; rest="${spec#*|}"
    runner_rel="${rest%%|*}"; extra="${rest#*|}"
    venv="$VENV_ROOT/$venv_name"
    runner="$REPO/analysis/benchmarking/$runner_rel"

    common="--dataset-tag $DATASET --seeds $SEEDS"
    # cell-only/recon runners take --silver-dir; gestarch reads --predicted-adata.
    if [[ -n "$SILVER_DIR" && "$m" != "gestarch" ]]; then
        common="$common --silver-dir $SILVER_DIR"
    fi
    extra="$extra $(neighs_flag "$m")"          # NBR_NEIGHS -> per-method neighbor flag

    job="imprec-$m"
    log_out="$LOG_ROOT/${m}.out"; log_err="$LOG_ROOT/${m}.err"

    read -r -d '' JOB <<EOF || true
set -euo pipefail
source "$venv/bin/activate"
cd "$REPO"
echo "[imprec] $m  (venv=$venv_name)"
python "$runner" $common $extra
echo "[imprec] DONE $m"
EOF

    BSUB=( bsub -G "$LSF_GROUP" -q "$LSF_QUEUE" -n "$LSF_CORES" -M "$LSF_MEM_MB"
           -R "select[mem>$LSF_MEM_MB] rusage[mem=$LSF_MEM_MB]" -R "span[ptile=$LSF_CORES]"
           -gpu "$LSF_GPU" -W "$LSF_WALL" -J "$job" -o "$log_out" -e "$log_err" )
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${BSUB[@]}"; printf 'bash -lc %q\n\n' "$JOB"
    else
        if [[ ! -d "$venv" ]]; then echo "MISSING venv: $venv — skip $m" >&2; continue; fi
        "${BSUB[@]}" bash -lc "$JOB"
    fi
    n_sub=$((n_sub + 1))
done
echo "[imprec] ${n_sub} job(s) $( [[ "$DRY_RUN" == "1" ]] && echo rendered || echo submitted )."
echo "After they finish, regenerate the figures:"
echo "  python analysis/benchmarking/plots/plot_imputation_benchmark.py --metric panel --branch cell"
echo "  python analysis/benchmarking/plots/plot_reconstruction_benchmark.py --branch cell   # and --branch niche"
