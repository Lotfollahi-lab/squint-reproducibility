#!/usr/bin/env bash
# submit_cell_type_baselines.sh
# -----------------------------------------------------------------------------
# Submit one LSF job per cell-type-identification baseline on the given
# dataset. Thin wrapper over `submit_all_benchmarks.sh` that pre-filters
# to the cell-type method set; queue + cost-code group are accepted as
# flags so you can switch between (queue, group) pairings without
# editing the script.
#
# Defaults: --queue training-parallel --group s10396.
#
# Runtime methodology (matches the user's spec — apples-to-apples with
# SQUINT multi-seed):
#   - TIMED per seed: shared embedding compute (PCA / FM extraction /
#     Harmony / scVI fit / etc.) + per-seed clustering (Leiden binary
#     search). Shared cost is added back to each seed's reported
#     runtime so each per_seed_runtimes.csv row represents "time to
#     obtain clusters from raw data for this seed".
#   - UNTIMED: NMI/ARI/iLISI/MMD computation, UMAP-for-visualization,
#     per-seed plot writes. Benchmark scaffolding, not method cost.
#
# Available cell-type methods:
#   pca-leiden    — PCA + Leiden (deterministic embedding, Leiden per-seed)
#   harmony       — PCA + Harmony + Leiden  (per-seed Harmony, per-seed Leiden)
#   scvi          — scVI fit + Leiden       (per-seed scVI fit + Leiden)
#   geneformer    — Geneformer FM + Leiden  (shared FM extraction)
#   nicheformer   — Nicheformer FM + Leiden (shared FM extraction)
#   scgpt         — scGPT FM + Leiden       (shared FM extraction)
#   scgpt-spatial — scGPT-spatial + Leiden  (shared FM extraction)
#   uce           — UCE FM + Leiden         (shared FM extraction)
#
# Usage:
#   bash analysis/benchmarking/submit_cell_type_baselines.sh [DATASET_TAG] [OPTIONS]
#
# Positional:
#   DATASET_TAG          Dataset key — forwarded to submit_all_benchmarks.sh.
#                        Default: same as that script (mmb0-1b_smb1-1b_1p,
#                        which resolves to the `_coord_aligned` silver dir
#                        on lustre — see submit_all_benchmarks.sh for the
#                        per-dataset leaf mapping).
#                        Other common values: chl59-8b_1p.
#
# Options:
#   --methods        / -m M1,M2,...   Comma-separated subset of cell-type methods.
#                                     Default: all 8 methods.
#                                     Example: -m pca-leiden,harmony,scvi
#   --queue          / -q QUEUE       LSF queue (default: training-parallel).
#                                     Overridable via LSF_QUEUE env var.
#   --group          / -g GROUP       LSF cost-code group (default: s10396).
#                                     Overridable via LSF_GROUP env var.
#   --resource-class / -r CLASS       Force every method onto the same
#                                     LSF resource class. Default:
#                                     gpu_high_memory (fits the heaviest
#                                     baseline, the foundation models).
#                                     Valid: cpu_small | gpu_standard |
#                                     gpu_high_memory | gpu_xtreme_memory
#                                     (768 GB; for foundation models on
#                                     the biggest spatial datasets where
#                                     384 GB OOMs).
#                                     Overridable via UNIFORM_RESOURCE
#                                     env var.
#   --wall           / -w HH:MM       LSF wall-clock limit (default:
#                                     inherits 96:00 from submit_all_benchmarks.sh).
#                                     Overridable via LSF_WALL env var.
#   --rapids-leiden                   Use rapids-singlecell (GPU-
#                                     accelerated) for the per-seed
#                                     Leiden binary search. Calls
#                                     rsc.pp.neighbors + rsc.tl.leiden
#                                     INLINE on the parent adata (same
#                                     API as scanpy). Sets
#                                     SQUINT_LEIDEN_BACKEND=rapids;
#                                     rapids-singlecell must be
#                                     importable in the venv running
#                                     the baseline (either install it
#                                     there, or run the baseline from
#                                     the rapids-singlecell conda env).
#   --help           / -h             Print this usage block and exit.
#
# Precedence for queue/group/resource: CLI flag > env var > script default.
# Precedence for methods:              CLI flag > script default (all methods).
#
# Why a uniform resource class by default:
#   For the runtime CSV produced by this wrapper to be comparable
#   across methods, every method must run on the SAME hardware. The
#   per-method classes in `submit_all_benchmarks.sh` are tuned for
#   minimum-needed resources (cpu_small for PCA, gpu_high_memory for
#   FMs), but mixing them would skew the runtime comparison. The
#   default `--resource-class gpu_high_memory` pins everyone to the
#   same big-GPU node class — the runtime delta between methods is
#   then attributable to the method, not the hardware.
#
# Other env-var overrides from `submit_all_benchmarks.sh` work too:
#
#   # Dry-run first (recommended):
#   DRY_RUN=1 bash analysis/benchmarking/submit_cell_type_baselines.sh
#
#   # Bigger wallclock / different paths:
#   LSF_WALL=48:00 ARTIFACTS_ROOT=/alt/path \
#       bash analysis/benchmarking/submit_cell_type_baselines.sh
#
#   # Submit only foundation models on mmb-smb, switching queue:
#   bash analysis/benchmarking/submit_cell_type_baselines.sh \
#       mmb0-1b_smb1-1b_1p \
#       --methods scgpt,scgpt-spatial,geneformer,nicheformer,uce \
#       --queue inference --group s10396
#
# Logs land at:
#   $LOG_ROOT/<DATASET_TAG>/<method>.{out,err}
# Per-method artifacts at:
#   $ARTIFACTS_ROOT/<DATASET_TAG>/baseline-<method>/<TS>/
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ---------------------------------------------------------------
DEFAULT_QUEUE="training-parallel"
DEFAULT_GROUP="s10396"
# Force every method onto identical compute so the runtime comparison is
# meaningful. `gpu_high_memory` (8 cores, 384 GB, 1 GPU exclusive) is
# chosen because it fits the heaviest baselines (foundation models —
# Geneformer / scGPT / scGPT-spatial / Nicheformer / UCE — AND GraphST
# on the niche-id side, which peaked at ~264 GB and was the reason
# we bumped the class from 256 GB to 384 GB). The lighter methods
# (PCA, Harmony) just leave the GPU idle — that's the trade-off for
# "all methods on the same node config". Override with
# `--resource-class` if needed.
DEFAULT_RESOURCE="gpu_high_memory"
ALL_METHODS=(
    pca-leiden harmony scvi
    geneformer nicheformer scgpt scgpt-spatial uce
)

# Seed working values from env vars (env still works for legacy callers);
# CLI flags below override.
QUEUE="${LSF_QUEUE:-$DEFAULT_QUEUE}"
GROUP="${LSF_GROUP:-$DEFAULT_GROUP}"
RESOURCE_ARG="${UNIFORM_RESOURCE:-$DEFAULT_RESOURCE}"
WALL_ARG="${LSF_WALL:-}"          # empty -> let submit_all_benchmarks.sh default kick in
METHODS_CSV=""
PASSTHROUGH_ARGS=()

# `--rapids-leiden` enables GPU Leiden via rapids-singlecell. Two
# modes coexist (see submit_niche_id_baselines.sh for the full
# rationale): subprocess (rapids in a separate conda env) and inline
# (rapids importable in the baseline's own venv). The wrapper arms
# BOTH so the helper picks subprocess when its env-setup cmd is
# present — that's the cluster default.
RAPIDS_LEIDEN_ENV_SETUP_DEFAULT="source /etc/profile.d/modules.sh && module load cellgen/conda && conda activate /nfs/team361/sb75/ENVS/rapids-singlecell"
USE_RAPIDS_LEIDEN=0

# --- Parse CLI flags --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods|-m)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --methods / -m requires a value." >&2
                exit 2
            fi
            METHODS_CSV="$1"
            shift
            ;;
        --methods=*)
            METHODS_CSV="${1#--methods=}"
            shift
            ;;
        --queue|-q)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --queue / -q requires a value." >&2
                exit 2
            fi
            QUEUE="$1"
            shift
            ;;
        --queue=*)
            QUEUE="${1#--queue=}"
            shift
            ;;
        --group|-g)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --group / -g requires a value." >&2
                exit 2
            fi
            GROUP="$1"
            shift
            ;;
        --group=*)
            GROUP="${1#--group=}"
            shift
            ;;
        --resource-class|-r)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --resource-class / -r requires a value." >&2
                exit 2
            fi
            RESOURCE_ARG="$1"
            shift
            ;;
        --resource-class=*)
            RESOURCE_ARG="${1#--resource-class=}"
            shift
            ;;
        --wall|-w)
            shift
            if [[ $# -eq 0 ]]; then
                echo "ERROR: --wall / -w requires a value (e.g. 96:00 or 168:00)." >&2
                exit 2
            fi
            WALL_ARG="$1"
            shift
            ;;
        --wall=*)
            WALL_ARG="${1#--wall=}"
            shift
            ;;
        --rapids-leiden)
            USE_RAPIDS_LEIDEN=1
            shift
            ;;
        --help|-h)
            awk '/^[^#]/ {exit} {print}' "$0"
            exit 0
            ;;
        *)
            # Forward anything else (e.g. DATASET_TAG positional) to
            # submit_all_benchmarks.sh.
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- Resolve methods --------------------------------------------------------
if [[ -z "$METHODS_CSV" ]]; then
    # Default: every cell-type method.
    ONLY_VALUE=$(IFS=,; echo "${ALL_METHODS[*]}")
else
    # User-supplied subset — validate each entry against the known set
    # so a typo like `--methods scvi,banksy` fails loudly rather than
    # silently submitting nothing.
    IFS=',' read -r -a REQUESTED <<< "$METHODS_CSV"
    if [[ ${#REQUESTED[@]} -eq 0 ]]; then
        echo "ERROR: --methods parsed to empty list." >&2
        exit 2
    fi
    for m in "${REQUESTED[@]}"; do
        valid=0
        for known in "${ALL_METHODS[@]}"; do
            if [[ "$m" == "$known" ]]; then valid=1; break; fi
        done
        if [[ "$valid" -eq 0 ]]; then
            echo "ERROR: unknown cell-type baseline '$m'." >&2
            echo "       Known: ${ALL_METHODS[*]}" >&2
            echo "       (Use submit_niche_id_baselines.sh for niche-id methods.)" >&2
            exit 2
        fi
    done
    ONLY_VALUE="$METHODS_CSV"
fi

# --- Print resolved values + delegate --------------------------------------
echo "[cell-type baselines] LSF_QUEUE        = $QUEUE"
echo "[cell-type baselines] LSF_GROUP        = $GROUP"
echo "[cell-type baselines] UNIFORM_RESOURCE = $RESOURCE_ARG"
echo "[cell-type baselines] LSF_WALL         = ${WALL_ARG:-<inherit submit_all_benchmarks.sh default>}"
echo "[cell-type baselines] RAPIDS_LEIDEN    = $USE_RAPIDS_LEIDEN"
echo "[cell-type baselines] ONLY             = $ONLY_VALUE"

# Export so submit_all_benchmarks.sh's `${LSF_QUEUE:-...}` /
# `${LSF_GROUP:-...}` / `${ONLY:-...}` / `${UNIFORM_RESOURCE:-...}` /
# `${LSF_WALL:-...}` fallbacks pick them up.
export LSF_QUEUE="$QUEUE"
export LSF_GROUP="$GROUP"
export ONLY="$ONLY_VALUE"
export UNIFORM_RESOURCE="$RESOURCE_ARG"
if [[ -n "$WALL_ARG" ]]; then
    export LSF_WALL="$WALL_ARG"
fi
# When --rapids-leiden was passed, export the bash command that
# `_run_one_benchmark.sh` will set as SQUINT_LEIDEN_RAPIDS_ENV_SETUP
# inside each per-method job. The shared Leiden helpers read that
# env var and route the binary search through rapids-singlecell.
if [[ "$USE_RAPIDS_LEIDEN" == "1" ]]; then
    export SQUINT_LEIDEN_BACKEND="rapids"
    export SQUINT_LEIDEN_RAPIDS_ENV_SETUP="${SQUINT_LEIDEN_RAPIDS_ENV_SETUP:-$RAPIDS_LEIDEN_ENV_SETUP_DEFAULT}"
    echo "[cell-type baselines] SQUINT_LEIDEN_BACKEND          = $SQUINT_LEIDEN_BACKEND"
    echo "[cell-type baselines] SQUINT_LEIDEN_RAPIDS_ENV_SETUP = $SQUINT_LEIDEN_RAPIDS_ENV_SETUP"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_all_benchmarks.sh" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
