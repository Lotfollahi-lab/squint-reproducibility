#!/bin/bash
# =============================================================================
# submit_basw.sh — one LSF job per (dataset, method), computing ONLY batch ASW.
#
# For R2-W1b. basw needs no kNN graph and no diffusion, so these jobs are small and
# quick, unlike the full label-conditioned set whose kbet step pushed the 199k-cell
# NSCLC jobs into TERM_MEMLIMIT. All three datasets are submitted by default.
#
# Results: one csv per (dataset, method) in
#   artifacts/label_conditioned_metrics/basw_<dataset>_<method>.csv
# The basw_ prefix keeps them separate from the lcm_ files, so nothing is
# overwritten and the two sets can be summarised independently.
#
# WHICH REPRESENTATION EACH METHOD IS SCORED ON
#   SQUINT    the four the paper's own iLISI and MMD are reported on: cell_emb /
#             neighborhood_emb (quantized codebook vectors) and cell_latent /
#             neighborhood_latent. The raw *_code_indices arrays are NOT scored, being
#             two unrelated integer categoricals on different scales.
#   baselines the emb_key recorded in that run's own
#             <ts>/metrics/batch_integration_metrics.csv, i.e. the embedding its
#             published iLISI was computed on. Not guessed.
#
# PER-DATASET LABELS, from submit_all_benchmarks.sh and DEFAULT_CELL_LABEL_KEYS:
#   xhs1000-3b_1p        new_annotation (21 classes). NOT the stray 40-class
#                        `cell_type` the benchmark deliberately does not score.
#   chl59-2b_1p          cell_type (10 classes), with --drop-label-nan for its
#                        8,980 unlabelled cells.
#   mmb0-1b_smb1-1b_1p   cell_type (49 classes). EXPECT basw TO FAIL HERE, with
#                        "No objects to concatenate": those 49 are two disjoint
#                        per-section vocabularies, so no cell type spans batches and
#                        scib's per-label loop appends nothing. Submitted anyway so
#                        the claim is evidenced by a run rather than asserted.
#
# EXCLUSIONS, reported rather than hidden: baseline-novae (saved adata has no obsm at
# any timestamp) and baseline-graphst on chl59 only (empty metrics dir, no saved
# output). The mmb `+region-holdout` and `cifm` dirs are gene-imputation runs with no
# embedding metrics and are skipped for want of an emb_key.
#
# USAGE (from an LSF submission host)
#   bash submit_basw.sh                       # all three datasets, every method
#   DRY_RUN=1 bash submit_basw.sh             # print, submit nothing
#   DATASETS=xhs1000-3b_1p bash submit_basw.sh
#   ONLY=SQUINT bash submit_basw.sh
#   FORCE=1 bash submit_basw.sh               # recompute existing csvs
# =============================================================================
set -uo pipefail

REPO="${REPO:-/nfs/team361/sb75/squint-reproducibility}"
VENV="${VENV:-/nfs/team361/sb75/.venvs/squint}"
DATASETS="${DATASETS:-xhs1000-3b_1p chl59-2b_1p mmb0-1b_smb1-1b_1p}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_CORES="${LSF_CORES:-2}"
LSF_EXTRA="${LSF_EXTRA:-}"

OUT="$REPO/artifacts/label_conditioned_metrics"
LOG="$OUT/logs"
SCRIPT="analysis/benchmarking/niche_identification/compute_basw.py"
# The four representations the paper's own iLISI and MMD are reported on
# (metrics/batch_integration_metrics.csv emb_key). The raw code-index arrays are
# NOT scored: two unrelated integer categoricals on different scales are not a
# metric space, and *_emb is the faithful vector form of the same codes.
SQUINT_KEYS="cell_emb,neighborhood_emb,cell_latent,neighborhood_latent"

mkdir -p "$LOG"

submit () {   # submit <dataset> <name> <keys> <label_key> <extra> <adata...>
    local DS=$1 NAME=$2 KEYS=$3 LABEL=$4 XTRA=$5; shift 5
    local CSV="$OUT/basw_${DS}_${NAME}.csv"
    if [[ -f "$CSV" && "$FORCE" != "1" ]]; then
        echo "    SKIP $NAME — csv exists (FORCE=1 to recompute)"; return
    fi
    local ADATA=""
    for f in "$@"; do ADATA="$ADATA --adata $f"; done
    [[ "$FORCE" == "1" ]] && XTRA="$XTRA --force"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '    %-26s %-62s %d seeds\n' "$NAME" "$KEYS" "$#"; return
    fi
    # -u: without it python buffers stdout when redirected, so the LSF .out file
    # stays empty until the job ends and progress cannot be watched.
    local CMD="source $VENV/bin/activate && cd $REPO && python -u $SCRIPT \
--method $NAME --cell-type-key $LABEL --batch-key adata_batch_id \
--latent-keys $KEYS $XTRA$ADATA --out $CSV"
    # shellcheck disable=SC2086
    bsub -G "$LSF_GROUP" -q "$QUEUE" -J "basw_${DS}_${NAME}" -W "$WALL" \
         -n "$LSF_CORES" -M "$MEM" -R "select[mem>$MEM]" -R "rusage[mem=$MEM]" \
         -R "span[ptile=$LSF_CORES]" $LSF_EXTRA \
         -o "$LOG/basw_${DS}_${NAME}.out" -e "$LOG/basw_${DS}_${NAME}.err" \
         "$CMD"
}

latest_ts () { ls -1d "$1"/*/ 2>/dev/null | sort | tail -1 | sed 's:/*$::'; }

seed_files () {   # per-seed adatas, else the single top-level one (2026-05 mmb runs)
    local TS=$1 F N=0
    for F in $(ls -1d "$TS"/seeds/seed_*/ 2>/dev/null | sort -V); do
        if [[ -f "${F}predicted_adata.h5ad" ]]; then
            echo "${F}predicted_adata.h5ad"; N=1
        fi
    done
    if ((N == 0)) && [[ -f "$TS/predicted_adata.h5ad" ]]; then
        echo "$TS/predicted_adata.h5ad"
    fi
}

for DS in $DATASETS; do
    ART="$REPO/artifacts/$DS"
    [[ -d "$ART" ]] || { echo "== $DS: no such dir, skipped"; continue; }

    case "$DS" in
        xhs1000-3b_1p)
            LABEL=new_annotation; XTRA=""
            SQD="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1+$DS"
            QUEUE=normal; MEM=32000; WALL=6:00 ;;
        chl59-2b_1p)
            LABEL=cell_type; XTRA="--drop-label-nan"
            SQD="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1+$DS"
            # 199,672 cells. silhouette is O(m^2) in each cell type's size m, so
            # this is the slow dataset even without a graph: budget hours, not
            # minutes, especially for the 768-dim foundation-model embeddings.
            QUEUE=long; MEM=128000; WALL=24:00 ;;
        mmb0-1b_smb1-1b_1p)
            LABEL=cell_type; XTRA=""
            SQD="s57_v19_reference-filmscale+$DS"
            QUEUE=normal; MEM=64000; WALL=6:00 ;;
        *)  echo "== $DS: no label mapping, skipped"; continue ;;
    esac

    echo "== $DS   label=$LABEL   queue=$QUEUE mem=$MEM"

    if [[ -z "$ONLY" || "$ONLY" == "SQUINT" ]]; then
        # latest timestamp per seed index: eczema holds two generations of seed dirs
        mapfile -t SQ < <(for N in 0 1 2 3 4; do
                              ls -1d "$ART/$SQD"/*_seed"$N"/ 2>/dev/null |
                                  sort | tail -1
                          done | sed 's:$:predicted_adata.h5ad:')
        if ((${#SQ[@]})); then
            submit "$DS" SQUINT "$SQUINT_KEYS" "$LABEL" "$XTRA" "${SQ[@]}"
        else
            echo "    !! no SQUINT seed dirs under $ART/$SQD"
        fi
    fi

    for D in "$ART"/baseline-*; do
        [[ -d "$D" ]] || continue
        NAME=$(basename "$D")
        [[ -n "$ONLY" && "$ONLY" != "$NAME" ]] && continue
        TS=$(latest_ts "$D")
        [[ -n "$TS" ]] || { echo "    SKIP $NAME — no timestamp dir"; continue; }

        BINT="$TS/metrics/batch_integration_metrics.csv"
        [[ -f "$BINT" ]] || { echo "    SKIP $NAME — no batch_integration_metrics.csv,"\
                                   "so the scored emb_key is unknown"; continue; }
        KEYS=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="emb_key") c=i; next}
                        c&&$c!=""{print $c}' "$BINT" | sort -u | paste -sd, -)
        [[ -n "$KEYS" ]] || { echo "    SKIP $NAME — no emb_key column"; continue; }
        [[ "$KEYS" == *,* ]] && { echo "    SKIP $NAME — ambiguous emb_key ($KEYS)"; continue; }

        mapfile -t FILES < <(seed_files "$TS")
        ((${#FILES[@]})) || { echo "    SKIP $NAME — no predicted_adata.h5ad"; continue; }
        if ! "$VENV/bin/python" -c \
             "import h5py,sys; sys.exit(0 if 'obsm' in h5py.File(sys.argv[1]) else 1)" \
             "${FILES[0]}" 2>/dev/null; then
            echo "    SKIP $NAME — saved adata has no obsm (novae)"; continue
        fi
        submit "$DS" "$NAME" "$KEYS" "$LABEL" "$XTRA" "${FILES[@]}"
    done
    echo
done

echo "summarise when done:"
echo "  python analysis/benchmarking/niche_identification/summarize_label_conditioned_metrics.py \\"
echo "      --pattern 'basw_*.csv' --sort-by basw --force"
