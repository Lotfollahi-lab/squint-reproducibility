#!/bin/bash
# =============================================================================
# submit_label_conditioned_metrics.sh — one LSF job per method for the
# label-conditioned scIB metrics (kbet / clisi / casw, plus ilisi as a check).
#
# Answers R2-W1b, which asked for label-conditioned integration metrics rather
# than MMD alone. Every job writes ONE csv into
#   artifacts/label_conditioned_metrics/
# and summarize_label_conditioned_metrics.py collapses them into one table.
#
# WHICH REPRESENTATION EACH BASELINE IS SCORED ON
# Not guessed: read out of each run's own
#   <ts>/metrics/batch_integration_metrics.csv -> column `emb_key`,
# which is the key the published iLISI was computed on (the runners set
# `bint_df["emb_key"] = LATENT_KEY`). So this scores exactly the embedding the
# paper scored. If a run reports more than one emb_key the script stops rather
# than pick one.
#
# SEEDS: <ts>/seeds/seed_*/predicted_adata.h5ad, 5 per method. The top-level
# predicted_adata.h5ad is a copy of the last seed, so it is NOT used.
#
# KNOWN EXCLUSION: baseline-novae. Its predicted_adata.h5ad contains only X and
# an EMPTY obs at all five of its timestamps (no obsm at all), so there is no
# saved embedding to score. Its published metrics were computed in-run and never
# persisted. Recomputing Novae needs the novae baseline itself rerun with the
# embedding written out; it is reported as excluded rather than silently dropped.
#
# CELL-TYPE LABEL: eczema uses `new_annotation` (21 classes), per
# submit_all_benchmarks.sh:421 -- the silver also carries a stray 40-class
# `cell_type` the benchmark deliberately does not score. kbet is computable here
# because that annotation is one shared vocabulary across the three patient
# sections (18/21 classes in all three, 100% of cells scorable). Contrast the
# mouse brain, where every label is confined to one section and kbet can only
# return NaN -- see label_batch_feasibility.py.
#
# USAGE (from an LSF submission host; bsub is not on the dev nodes)
#   bash submit_label_conditioned_metrics.sh              # submit everything
#   DRY_RUN=1 bash submit_label_conditioned_metrics.sh    # print, submit nothing
#   ONLY=baseline-scvi bash submit_label_conditioned_metrics.sh
#
# NSCLC needs the `long` queue: at 199,672 cells the per-cell-type diffusion step
# inside kbet does not reliably finish inside `normal`'s wall limit.
#   DATASET_TAG=chl59-2b_1p CELL_TYPE_KEY=cell_type EXTRA_ARGS=--drop-label-nan \
#       LSF_QUEUE=long LSF_MEM=128000 LSF_WALL=48:00 \
#       bash submit_label_conditioned_metrics.sh
# =============================================================================
set -uo pipefail

REPO="${REPO:-/nfs/team361/sb75/squint-reproducibility}"
VENV="${VENV:-/nfs/team361/sb75/.venvs/squint}"
DATASET_TAG="${DATASET_TAG:-xhs1000-3b_1p}"
CELL_TYPE_KEY="${CELL_TYPE_KEY:-new_annotation}"
BATCH_KEY="${BATCH_KEY:-adata_batch_id}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"
# FORCE=1 recomputes methods that already have a csv, and passes --force through so
# the python side overwrites it. Needed whenever a metric is added to METRICS,
# since an existing csv is otherwise treated as done and skipped.
FORCE="${FORCE:-0}"
[[ "$FORCE" == "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --force"
# CPU queue, and the group that goes with it. The GPU pairing is a different
# one (training-parallel + s10396), and it would buy nothing here -- see below.
LSF_GROUP="${LSF_GROUP:-team361}"
LSF_QUEUE="${LSF_QUEUE:-normal}"
LSF_WALL="${LSF_WALL:-12:00}"
LSF_MEM="${LSF_MEM:-64000}"
LSF_CORES="${LSF_CORES:-4}"
# Raw bsub flags, appended verbatim.
#
# THIS WORK IS CPU-ONLY, AND A GPU CANNOT HELP IT. scib-metrics runs the LISI
# family through jax, but the squint venv has jax 0.6.2 / jaxlib 0.6.2 with NO
# jax-cuda12-plugin installed, so jax.default_backend() is "cpu" on every node
# (the nvidia-cuda-* wheels in the venv belong to torch). kbet is scipy, the
# neighbour graph is pynndescent/numba, and silhouette is jax. So requesting a
# GPU would reserve an idle device. Measured cost: ~4 min per seed at 53k cells.
#
# If you nonetheless need a GPU queue, they REJECT jobs that request no GPU:
#   LSF_QUEUE=training-parallel LSF_GROUP=s10396 \
#   LSF_EXTRA="-gpu mode=exclusive_process:num=1:block=yes"
LSF_EXTRA="${LSF_EXTRA:-}"

ART="$REPO/artifacts/$DATASET_TAG"
OUT="$REPO/artifacts/label_conditioned_metrics"
LOG="$OUT/logs"
SCRIPT="analysis/benchmarking/niche_identification/compute_label_conditioned_metrics.py"

# The SQUINT reference run for each dataset (the headline variant).
case "$DATASET_TAG" in
    xhs1000-3b_1p|chl59-2b_1p)
        SQUINT_DIR="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1+$DATASET_TAG" ;;
    mmb0-1b_smb1-1b_1p)
        SQUINT_DIR="s57_v19_reference-filmscale+$DATASET_TAG" ;;
    *)  SQUINT_DIR="" ;;
esac

[[ -d "$ART" ]] || { echo "no such dataset dir: $ART" >&2; exit 1; }
mkdir -p "$LOG"

submit () {   # submit <name> <latent_keys> <adata...>
    local NAME=$1 KEYS=$2; shift 2
    local CSV="$OUT/lcm_${DATASET_TAG}_${NAME}.csv"
    if [[ -f "$CSV" && "$FORCE" != "1" ]]; then
        echo "  SKIP $NAME — $CSV exists. FORCE=1 recomputes it (needed after a new"
        echo "       metric is added, e.g. basw), otherwise delete the csv by hand."
        return
    fi
    local ADATA_ARGS=""
    for f in "$@"; do ADATA_ARGS="$ADATA_ARGS --adata $f"; done
    local CMD="source $VENV/bin/activate && cd $REPO && python $SCRIPT \
--method $NAME --cell-type-key $CELL_TYPE_KEY --batch-key $BATCH_KEY \
--latent-keys $KEYS $EXTRA_ARGS$ADATA_ARGS --out $CSV"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  %-24s %-28s %d seeds\n' "$NAME" "$KEYS" "$#"
        return
    fi
    # shellcheck disable=SC2086  # LSF_EXTRA must word-split into separate flags
    bsub -G "$LSF_GROUP" -q "$LSF_QUEUE" -J "lcm_${DATASET_TAG}_${NAME}" \
         -W "$LSF_WALL" -n "$LSF_CORES" -M "$LSF_MEM" \
         -R "select[mem>$LSF_MEM]" -R "rusage[mem=$LSF_MEM]" \
         -R "span[ptile=$LSF_CORES]" $LSF_EXTRA \
         -o "$LOG/${DATASET_TAG}_${NAME}.out" \
         -e "$LOG/${DATASET_TAG}_${NAME}.err" \
         "$CMD"
}

latest_ts () { ls -1d "$1"/*/ 2>/dev/null | sort | tail -1 | sed 's:/*$::'; }

seed_files () {   # echo the per-seed adatas under <ts>/seeds/, else the top-level one
    local TS=$1 F N=0
    for F in $(ls -1d "$TS"/seeds/seed_*/ 2>/dev/null | sort -V); do
        if [[ -f "${F}predicted_adata.h5ad" ]]; then
            echo "${F}predicted_adata.h5ad"; N=1
        fi
    done
    # The 2026-05 mouse-brain baseline runs wrote per-seed METRICS but only ONE
    # adata, at the top level, so seeds/seed_*/ holds no h5ad. Fall back to it so
    # those methods can be scored at all. They then contribute a single seed and
    # get no error bar, which the summary reports as n_seeds=1 rather than hiding.
    if ((N == 0)) && [[ -f "$TS/predicted_adata.h5ad" ]]; then
        echo "$TS/predicted_adata.h5ad"
    fi
}

echo "dataset      $DATASET_TAG"
echo "cell label   $CELL_TYPE_KEY   batch  $BATCH_KEY"
echo "out          $OUT"
[[ "$DRY_RUN" == "1" ]] && echo "DRY_RUN: nothing will be submitted"
echo

# --- SQUINT ----------------------------------------------------------------
# Discrete codes: what Table 1's integration metrics are computed on.
if [[ -n "$SQUINT_DIR" && -d "$ART/$SQUINT_DIR" ]]; then
    if [[ -z "$ONLY" || "$ONLY" == "SQUINT" ]]; then
        # The SQUINT variant dir holds TWO generations of seed dirs on eczema
        # (2026-06-29 19:16-19:47 and 19:48-20:15). Take the LATEST timestamp per
        # seed index, the same "latest wins" rule used for the baselines --
        # a plain glob would score both generations as ten seeds.
        mapfile -t SQ < <(for N in 0 1 2 3 4; do
                              ls -1d "$ART/$SQUINT_DIR"/*_seed"$N"/ 2>/dev/null |
                                  sort | tail -1
                          done | sed 's:$:predicted_adata.h5ad:')
        if ((${#SQ[@]})); then
            # SQUINT is scored on its DISCRETE representation, in both encodings.
            #   *_code_indices : the integer codes, what Table 1's iLISI is on.
            #   *_emb          : the quantized vectors those indices point to.
            #                    cell_emb is a pure codebook lookup (633 distinct
            #                    rows for 53,655 eczema cells, constant within each
            #                    code pair); neighborhood_emb additionally carries
            #                    the FiLM modulation, hence 4,133 distinct rows.
            #                    Worth scoring separately from the indices because
            #                    euclidean distance between codebook VECTORS is
            #                    meaningful whereas distance between code numbers
            #                    (code 5 vs code 7) is not, which matters for casw.
            # kbet does NOT survive either encoding -- 94% of neighbour distances
            # are exactly 0, so its diffusion step hits the `score = 0` fallback for
            # 19 of 21 labels, identically for indices and quantized vectors. Read
            # kbet off SQUINT-continuous below; the summary flags the zeros.
            submit SQUINT cell_code_indices,neighborhood_code_indices "${SQ[@]}"
            submit SQUINT-quantized cell_emb,neighborhood_emb "${SQ[@]}"
            # Kept, and NOT redundant: this is the only representation on which
            # kbet, the metric R2-W1b actually asked for, can be computed at all.
            submit SQUINT-continuous cell_latent,neighborhood_latent "${SQ[@]}"
        else
            echo "  !! no seed dirs under $ART/$SQUINT_DIR"
        fi
    fi
else
    echo "  !! no SQUINT reference dir for $DATASET_TAG; baselines only"
fi

# --- baselines -------------------------------------------------------------
for D in "$ART"/baseline-*; do
    [[ -d "$D" ]] || continue
    NAME=$(basename "$D")
    [[ -n "$ONLY" && "$ONLY" != "$NAME" ]] && continue
    TS=$(latest_ts "$D")
    [[ -n "$TS" ]] || { echo "  SKIP $NAME — no timestamp dir"; continue; }

    BINT="$TS/metrics/batch_integration_metrics.csv"
    if [[ ! -f "$BINT" ]]; then
        echo "  SKIP $NAME — no $BINT, so the scored emb_key is unknown"; continue
    fi
    # column `emb_key`, deduplicated. More than one -> ambiguous, so stop.
    KEYS=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="emb_key") c=i; next}
                    c&&$c!=""{print $c}' "$BINT" | sort -u | paste -sd, -)
    if [[ -z "$KEYS" ]]; then
        echo "  SKIP $NAME — no emb_key column in $BINT"; continue
    fi
    if [[ "$KEYS" == *,* ]]; then
        echo "  SKIP $NAME — ambiguous emb_key ($KEYS); pass it by hand"; continue
    fi

    mapfile -t FILES < <(seed_files "$TS")
    if ((${#FILES[@]} == 0)); then
        echo "  SKIP $NAME — no seeds/seed_*/predicted_adata.h5ad under $TS"
        continue
    fi
    # novae: obsm is absent in every saved file, so there is nothing to score.
    # Checked here rather than left to the job, so the skip is visible up front.
    if ! "$VENV/bin/python" -c \
         "import h5py,sys; sys.exit(0 if 'obsm' in h5py.File(sys.argv[1]) else 1)" \
         "${FILES[0]}" 2>/dev/null; then
        echo "  SKIP $NAME — saved adata has no obsm group (novae is the known"
        echo "       case); rerun that baseline with the embedding written out"
        continue
    fi
    if ((${#FILES[@]} == 1)) && [[ "${FILES[0]}" == "$TS/predicted_adata.h5ad" ]]; then
        echo "  note $NAME — no per-seed adatas saved; scoring the single"
        echo "       top-level predicted_adata.h5ad (n_seeds=1, no error bar)"
    fi
    submit "$NAME" "$KEYS" "${FILES[@]}"
done

echo
echo "when the jobs finish:"
echo "  python $SCRIPT --self-test                    # sanity-check the conversion"
echo "  python analysis/benchmarking/niche_identification/summarize_label_conditioned_metrics.py"
