#!/bin/bash
# =============================================================================
# submit_label_metrics.sh — one LSF job per (METRIC, DATASET, METHOD).
#
# For R2-W1b. Maximum parallelisation: the metrics differ in cost by orders of
# magnitude (basw needs no graph, kbet_label builds a diffusion map per cell type),
# so bundling them meant the slowest one killed the cheap ones with it on the
# 199k-cell NSCLC set. One metric per job also means a single failure costs one
# number instead of six.
#
# Results: artifacts/label_conditioned_metrics/lm_<metric>_<dataset>_<method>.csv
# Long format, one row per (metric, seed, latent_key) with the number in `value`,
# so every csv concatenates. Summarise with summarize_label_metrics.py.
#
# JOB COUNT is large by design: 8 metrics x 3 datasets x ~14 methods, minus the
# documented exclusions. Print it first with DRY_RUN=1 and cut it down with
# METRICS=, DATASETS= or ONLY= if that is more than you want to queue.
#
# RESOURCES are set per (dataset, metric class) rather than uniformly, because the
# earlier all-in-one NSCLC jobs hit TERM_MEMLIMIT at 128 GB:
#   no-graph    basw, bras, casw, mmd       reads obsm only
#   subset      cilisi, kbet_strat, cmmd    per-cell-type subset (kNN or MMD)
#   global      kbet_label, graph_conn, ilisi, clisi   full k=50/90 graph, + diffusion
#
# WHAT IS SCORED
#   SQUINT     the four representations the paper's own integration metrics use:
#              cell_emb / neighborhood_emb (the quantized codebook vectors) and
#              cell_latent / neighborhood_latent (pre-quantization). Confirmed from
#              metrics/batch_integration_metrics.csv, whose emb_key column holds
#              exactly these four. The raw *_code_indices arrays are NOT scored: two
#              unrelated integer categoricals on different scales are not a metric
#              space. An earlier claim here that Table 1's iLISI was on the code
#              indices was wrong and is retracted.
#   baselines  the emb_key recorded in that run's own
#              <ts>/metrics/batch_integration_metrics.csv, i.e. the embedding its
#              published iLISI was computed on. Read, not guessed.
#
# KNOWN EXCLUSIONS, printed rather than hidden:
#   novae        every dataset. Its predicted_adata.h5ad holds X and an EMPTY obs
#                because write_h5ad crashed (fixed now in run_pca_leiden.py's
#                _sanitize_for_h5ad, but the runs need redoing to benefit).
#   graphst      chl59 only. PASTE pairwise_align died with CUDA OOM (37 GiB on top
#                of 115 GiB), so no output was ever written. Fine per the user.
#   mmb          EVERY label-conditioned metric is undefined there: the 49 cell types
#                are two disjoint per-section vocabularies, so no cell type spans
#                batches. Submitted anyway so that rests on runs, not on assertion.
#
# USAGE (from an LSF submission host)
#   DRY_RUN=1 bash submit_label_metrics.sh                    # count the jobs first
#   bash submit_label_metrics.sh
#   METRICS="cilisi kbet_strat basw bras" bash submit_label_metrics.sh
#   DATASETS=xhs1000-3b_1p ONLY=SQUINT bash submit_label_metrics.sh
#   FORCE=1 bash submit_label_metrics.sh                      # recompute existing
# =============================================================================
set -uo pipefail

REPO="${REPO:-/nfs/team361/sb75/squint-reproducibility}"
VENV="${VENV:-/nfs/team361/sb75/.venvs/squint}"
DATASETS="${DATASETS:-xhs1000-3b_1p chl59-2b_1p mmb0-1b_smb1-1b_1p}"
# Default order puts the label-conditioned ones first, cheapest first, so the
# numbers that answer R2-W1b land before the expensive diffusion metric.
METRICS="${METRICS:-basw bras cilisi cmmd kbet_strat graph_conn kbet_label ilisi mmd clisi}"
ONLY="${ONLY:-}"
# WHICH LABEL THE METRIC IS CONDITIONED ON. The paper runs TWO comparisons and they use
# DIFFERENT labels: cell-type identification is scored against the cell-type
# annotation, niche identification against the niche annotation. A label-conditioned
# metric must follow that, so LABEL_KIND=niche conditions on the niche label instead
# and writes to *_niche.csv, keeping the two sets of numbers separate.
#   LABEL_KIND=cell  bash submit_label_metrics.sh    # default
#   LABEL_KIND=niche bash submit_label_metrics.sh
LABEL_KIND="${LABEL_KIND:-cell}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_EXTRA="${LSF_EXTRA:-}"
# Overrides that WIN over the per-(dataset, metric) defaults in resources(). Those
# defaults encode guesses about queue limits, and a bad guess is rejected at
# SUBMISSION with "RUNLIMIT: Cannot exceed queue's hard limit(s). Job not submitted."
# -- so the jobs never start and never write a .out, which is how it presents: 13
# missing logs rather than 13 failures. Observed for chl59 at -W 24:00 on `long`.
# Find a queue's real ceiling with:  bqueues -l <queue> | grep -i runlimit
#   LSF_QUEUE=long LSF_WALL=12:00 LSF_MEM=128000 bash submit_label_metrics.sh
LSF_QUEUE_OVERRIDE="${LSF_QUEUE:-}"
LSF_WALL_OVERRIDE="${LSF_WALL:-}"
LSF_MEM_OVERRIDE="${LSF_MEM:-}"

OUT="$REPO/artifacts/label_conditioned_metrics"
LOG="$OUT/logs"
SCRIPT="analysis/benchmarking/niche_identification/compute_label_metric.py"
# The four representations the paper's own iLISI and MMD are reported on
# (metrics/batch_integration_metrics.csv emb_key). The raw code-index arrays are
# NOT scored: two unrelated integer categoricals on different scales are not a
# metric space, and *_emb is the faithful vector form of the same codes.
SQUINT_KEYS="cell_emb,neighborhood_emb,cell_latent,neighborhood_latent"

mkdir -p "$LOG"
NSUB=0; NSKIP=0

resources () {   # resources <dataset> <metric>  -> sets QUEUE MEM WALL CORES
    local DS=$1 M=$2 CLASS
    case "$M" in
        basw|bras|casw|mmd)           CLASS=nograph ;;
        cilisi|kbet_strat|cmmd)       CLASS=subset ;;
        *)                            CLASS=global ;;
    esac
    CORES=2; QUEUE=normal
    case "$DS" in
        chl59-2b_1p)                          # 199,672 cells
            # -W 24:00 was REJECTED by `long` ("Cannot exceed queue's hard limit(s)"),
            # so this is 12:00, which is under any plausible ceiling. cilisi and
            # kbet_strat only build exact kNNs inside ~19k-cell cell-type subsets, so
            # 12 h is ample; raise it with LSF_WALL= if a global-graph metric needs it.
            QUEUE=long; WALL=12:00
            case "$CLASS" in
                nograph) MEM=128000 ;;
                subset)  MEM=192000 ;;
                global)  MEM=256000 ;;        # 128000 hit TERM_MEMLIMIT before
            esac ;;
        mmb0-1b_smb1-1b_1p)
            WALL=12:00
            case "$CLASS" in
                nograph) MEM=64000 ;;
                *)       MEM=96000 ;;
            esac ;;
        *)                                    # eczema, 53,655 cells
            WALL=12:00
            case "$CLASS" in
                nograph) MEM=32000 ;;
                *)       MEM=64000 ;;
            esac ;;
    esac
    [[ -n "$LSF_QUEUE_OVERRIDE" ]] && QUEUE="$LSF_QUEUE_OVERRIDE"
    [[ -n "$LSF_WALL_OVERRIDE" ]] && WALL="$LSF_WALL_OVERRIDE"
    [[ -n "$LSF_MEM_OVERRIDE" ]] && MEM="$LSF_MEM_OVERRIDE"
}

submit () {   # submit <dataset> <metric> <name> <keys> <label> <extra> <adata...>
    local DS=$1 M=$2 NAME=$3 KEYS=$4 LABEL=$5 XTRA=$6; shift 6
    local SUF=""; [[ "$LABEL_KIND" == "niche" ]] && SUF="_niche"
    local CSV="$OUT/lm_${M}_${DS}_${NAME}${SUF}.csv"
    if [[ -f "$CSV" && "$FORCE" != "1" ]]; then NSKIP=$((NSKIP+1)); return; fi
    local ADATA=""
    for f in "$@"; do ADATA="$ADATA --adata $f"; done
    [[ "$FORCE" == "1" ]] && XTRA="$XTRA --force"
    resources "$DS" "$M"
    NSUB=$((NSUB+1))
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '    %-12s %-26s %-8s %6s MB  %s\n' "$M" "$NAME" "$QUEUE" "$MEM" \
               "$# seeds"
        return
    fi
    # -u so the LSF .out streams instead of staying empty until the job ends.
    local CMD="source $VENV/bin/activate && cd $REPO && python -u $SCRIPT \
--metric $M --method $NAME --cell-type-key $LABEL --batch-key adata_batch_id \
--latent-keys $KEYS $XTRA$ADATA --out $CSV"
    # shellcheck disable=SC2086
    bsub -G "$LSF_GROUP" -q "$QUEUE" -J "lm_${M}_${DS}_${NAME}${SUF}" -W "$WALL" \
         -n "$CORES" -M "$MEM" -R "select[mem>$MEM]" -R "rusage[mem=$MEM]" \
         -R "span[ptile=$CORES]" $LSF_EXTRA \
         -o "$LOG/lm_${M}_${DS}_${NAME}${SUF}.out" \
         -e "$LOG/lm_${M}_${DS}_${NAME}${SUF}.err" "$CMD" >/dev/null
}

latest_ts () { ls -1d "$1"/*/ 2>/dev/null | sort | tail -1 | sed 's:/*$::'; }

seed_files () {
    local TS=$1 F N=0
    for F in $(ls -1d "$TS"/seeds/seed_*/ 2>/dev/null | sort -V); do
        if [[ -f "${F}predicted_adata.h5ad" ]]; then
            echo "${F}predicted_adata.h5ad"; N=1
        fi
    done
    # 2026-05 mouse-brain runs saved per-seed metrics but only one adata.
    if ((N == 0)) && [[ -f "$TS/predicted_adata.h5ad" ]]; then
        echo "$TS/predicted_adata.h5ad"
    fi
}

for DS in $DATASETS; do
    ART="$REPO/artifacts/$DS"
    [[ -d "$ART" ]] || { echo "== $DS: no such dir, skipped"; continue; }
    case "$DS" in
        xhs1000-3b_1p)
            CELL_LABEL=new_annotation; NICHE_LABEL=niche_type; XTRA=""
            SQD="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1+$DS" ;;
        chl59-2b_1p)
            CELL_LABEL=cell_type; NICHE_LABEL=niche; XTRA="--drop-label-nan"
            SQD="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1+$DS" ;;
        mmb0-1b_smb1-1b_1p)
            # Both are section-specific here, so every label-conditioned metric is
            # undefined either way. Submitted so that rests on runs, not assertion.
            CELL_LABEL=cell_type; NICHE_LABEL=Sub_molecular_tissue_region; XTRA=""
            SQD="s57_v19_reference-filmscale+$DS" ;;
        *)  echo "== $DS: no label mapping, skipped"; continue ;;
    esac
    if [[ "$LABEL_KIND" == "niche" ]]; then LABEL="$NICHE_LABEL"
    else LABEL="$CELL_LABEL"; fi
    echo "== $DS   conditioned on $LABEL_KIND label: $LABEL"

    # Resolve the file lists ONCE per dataset, then reuse across metrics.
    mapfile -t SQ < <(for N in 0 1 2 3 4; do
                          ls -1d "$ART/$SQD"/*_seed"$N"/ 2>/dev/null | sort | tail -1
                      done | sed 's:$:predicted_adata.h5ad:')
    NAMES=(); KEYSL=(); FILESL=()
    if ((${#SQ[@]})); then
        NAMES+=("SQUINT"); KEYSL+=("$SQUINT_KEYS"); FILESL+=("${SQ[*]}")
    else
        echo "    !! no SQUINT seed dirs under $ART/$SQD"
    fi
    for D in "$ART"/baseline-*; do
        [[ -d "$D" ]] || continue
        NAME=$(basename "$D")
        TS=$(latest_ts "$D"); [[ -n "$TS" ]] || continue
        BINT="$TS/metrics/batch_integration_metrics.csv"
        if [[ ! -f "$BINT" ]]; then
            echo "    SKIP $NAME — no batch_integration_metrics.csv (no emb_key)"
            continue
        fi
        KEYS=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="emb_key") c=i; next}
                        c&&$c!=""{print $c}' "$BINT" | sort -u | paste -sd, -)
        [[ -n "$KEYS" && "$KEYS" != *,* ]] || {
            echo "    SKIP $NAME — missing or ambiguous emb_key ($KEYS)"; continue; }
        mapfile -t FILES < <(seed_files "$TS")
        ((${#FILES[@]})) || { echo "    SKIP $NAME — no predicted_adata.h5ad"; continue; }
        if ! "$VENV/bin/python" -c \
             "import h5py,sys; sys.exit(0 if 'obsm' in h5py.File(sys.argv[1]) else 1)" \
             "${FILES[0]}" 2>/dev/null; then
            echo "    SKIP $NAME — saved adata has no obsm (novae)"; continue
        fi
        NAMES+=("$NAME"); KEYSL+=("$KEYS"); FILESL+=("${FILES[*]}")
    done

    for M in $METRICS; do
        for idx in "${!NAMES[@]}"; do
            [[ -n "$ONLY" && "$ONLY" != "${NAMES[$idx]}" ]] && continue
            # shellcheck disable=SC2086
            submit "$DS" "$M" "${NAMES[$idx]}" "${KEYSL[$idx]}" "$LABEL" "$XTRA" \
                   ${FILESL[$idx]}
        done
    done
    echo
done

echo "submitted $NSUB job(s); skipped $NSKIP with an existing csv (FORCE=1 to redo)"
echo
echo "summarise:"
echo "  python analysis/benchmarking/niche_identification/summarize_label_metrics.py"
