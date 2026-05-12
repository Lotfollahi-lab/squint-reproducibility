#!/usr/bin/env bash
# submit_niche_id_baselines.sh
# -----------------------------------------------------------------------------
# Submit one LSF job per niche-identification baseline on the given
# dataset. Thin wrapper over `submit_all_benchmarks.sh` that pre-filters
# to the niche-id method set; queue + cost-code group are accepted as
# flags so you can switch between (queue, group) pairings without
# editing the script.
#
# Defaults: --queue training-parallel --group s10396.
#
# Runtime methodology (matches the user's spec — apples-to-apples with
# SQUINT multi-seed):
#   - TIMED per seed: shared embedding compute (BANKSY+Harmony /
#     neigh-PCA / spatial graph + GP masks / etc.) + per-seed model
#     fit (where seed-dependent: CellCharter, GraphST, NicheCompass,
#     Novae) + per-seed clustering (Leiden binary search, or the
#     model's own assign_domains for Novae). Shared cost is added back
#     to each seed's reported runtime so each per_seed_runtimes.csv
#     row represents "time to obtain clusters from raw data for this
#     seed".
#   - UNTIMED: NMI/ARI/iLISI/MMD computation, UMAP-for-visualization,
#     per-seed plot writes. Benchmark scaffolding, not method cost.
#
# Available niche-id methods:
#   banksy          — BANKSY + Harmony + Leiden  (shared embedding)
#   cellcharter     — log1p + spatial + scVI + aggregate + Leiden
#                     (per-seed scVI fit + Leiden)
#   graphst         — PASTE + spatial + GraphST + Leiden
#                     (per-seed GraphST fit + Leiden)
#   novae           — spatial + Novae forward + assign_domains
#                     (per-seed Novae forward; domains ARE the clusters
#                     — no Leiden)
#   nichecompass    — spatial + GP masks + NicheCompass + Leiden
#                     (per-seed NicheCompass fit + Leiden)
#   neigh-expr-pca  — spatial + neighborhood-mean PCA + Leiden
#                     (shared embedding)
#
# Usage:
#   bash analysis/benchmarking/submit_niche_id_baselines.sh [DATASET_TAG] [OPTIONS]
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
#   --methods        / -m M1,M2,...   Comma-separated subset of niche-id methods.
#                                     Default: all 6 methods.
#                                     Example: -m banksy,neigh-expr-pca
#   --queue          / -q QUEUE       LSF queue (default: training-parallel).
#                                     Overridable via LSF_QUEUE env var.
#   --group          / -g GROUP       LSF cost-code group (default: s10396).
#                                     Overridable via LSF_GROUP env var.
#   --resource-class / -r CLASS       Force every method onto the same
#                                     LSF resource class. Default:
#                                     gpu_high_memory (fits the heaviest
#                                     baseline — CellCharter / NicheCompass /
#                                     GraphST / Novae all need a GPU + decent
#                                     memory). Valid: cpu_small |
#                                     gpu_standard | gpu_high_memory.
#                                     Overridable via UNIFORM_RESOURCE
#                                     env var.
#   --help           / -h             Print this usage block and exit.
#
# Precedence for queue/group/resource: CLI flag > env var > script default.
# Precedence for methods:              CLI flag > script default (all methods).
#
# Why a uniform resource class by default:
#   For the runtime CSV produced by this wrapper to be comparable
#   across methods, every method must run on the SAME hardware. The
#   per-method classes in `submit_all_benchmarks.sh` are tuned for
#   minimum-needed resources (cpu_small for BANKSY / neigh-expr-PCA,
#   gpu_standard for CellCharter / GraphST / etc.), but mixing them
#   would skew the runtime comparison. The default
#   `--resource-class gpu_high_memory` pins everyone to the same
#   big-GPU node class — the runtime delta between methods is then
#   attributable to the method, not the hardware.
#
# Other env-var overrides from `submit_all_benchmarks.sh` work too:
#
#   # Dry-run first (recommended):
#   DRY_RUN=1 bash analysis/benchmarking/submit_niche_id_baselines.sh
#
#   # Just the deep models, on mmb-smb, longer walltime:
#   LSF_WALL=48:00 \
#       bash analysis/benchmarking/submit_niche_id_baselines.sh \
#       mmb0-1b_smb1-1b_1p \
#       --methods cellcharter,graphst,nichecompass,novae
#
#   # Skip the heavier methods, switch queue:
#   bash analysis/benchmarking/submit_niche_id_baselines.sh \
#       --methods banksy,neigh-expr-pca \
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
# See cell-type-baselines wrapper for the full rationale — same logic
# here. gpu_high_memory fits every niche-id method (the deep models
# need a GPU + 100+ GB), and the lighter ones (BANKSY,
# neigh-expr-PCA) just leave the GPU idle.
DEFAULT_RESOURCE="gpu_high_memory"
ALL_METHODS=(
    banksy cellcharter graphst novae
    nichecompass neigh-expr-pca
)

# Seed working values from env vars (env still works for legacy callers);
# CLI flags below override.
QUEUE="${LSF_QUEUE:-$DEFAULT_QUEUE}"
GROUP="${LSF_GROUP:-$DEFAULT_GROUP}"
RESOURCE_ARG="${UNIFORM_RESOURCE:-$DEFAULT_RESOURCE}"
METHODS_CSV=""
PASSTHROUGH_ARGS=()

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
    # Default: every niche-id method.
    ONLY_VALUE=$(IFS=,; echo "${ALL_METHODS[*]}")
else
    # User-supplied subset — validate each entry against the known set
    # so a typo like `--methods banksy,scvi` fails loudly rather than
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
            echo "ERROR: unknown niche-id baseline '$m'." >&2
            echo "       Known: ${ALL_METHODS[*]}" >&2
            echo "       (Use submit_cell_type_baselines.sh for cell-type methods.)" >&2
            exit 2
        fi
    done
    ONLY_VALUE="$METHODS_CSV"
fi

# --- Print resolved values + delegate --------------------------------------
echo "[niche-id baselines] LSF_QUEUE        = $QUEUE"
echo "[niche-id baselines] LSF_GROUP        = $GROUP"
echo "[niche-id baselines] UNIFORM_RESOURCE = $RESOURCE_ARG"
echo "[niche-id baselines] ONLY             = $ONLY_VALUE"

# Export so submit_all_benchmarks.sh's `${LSF_QUEUE:-...}` /
# `${LSF_GROUP:-...}` / `${ONLY:-...}` / `${UNIFORM_RESOURCE:-...}`
# fallbacks pick them up.
export LSF_QUEUE="$QUEUE"
export LSF_GROUP="$GROUP"
export ONLY="$ONLY_VALUE"
export UNIFORM_RESOURCE="$RESOURCE_ARG"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "$SCRIPT_DIR/submit_all_benchmarks.sh" \
    ${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}
