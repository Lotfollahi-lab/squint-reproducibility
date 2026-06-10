#!/usr/bin/env bash
# _run_one_benchmark.sh
# -----------------------------------------------------------------------------
# Per-job runner invoked by `submit_all_benchmarks.sh` inside each LSF job.
# Activates the method's venv, sets the standard env (TMPDIR off NFS,
# scrubbed PYTHONPATH/HOME, LD_LIBRARY_PATH with the venv's bundled
# CUDA libs), then runs the specified benchmarking script.
#
# Designed to mirror `squint/examples/_run_one_variant.sh` so the same
# defensive setup (NFS silly-rename avoidance, CUDA-lib path discovery,
# env scrubbing) applies to every benchmark method.
#
# Usage (normally invoked by submit_all_benchmarks.sh, but works standalone):
#   bash _run_one_benchmark.sh <SCRIPT_PATH> <VENV_PATH> [-- ARGS...]
#
# Args:
#   SCRIPT_PATH  — required. Path to the runner .py to execute.
#   VENV_PATH    — required. Python venv to activate before running.
#   ARGS...      — passed verbatim to the runner .py (everything after `--`).
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_PATH="${1:?usage: $0 SCRIPT_PATH VENV_PATH [--env KEY=VAL ...] [-- ARGS...]}"
VENV_PATH="${2:?usage: $0 SCRIPT_PATH VENV_PATH [--env KEY=VAL ...] [-- ARGS...]}"
shift 2

# Parse optional `--env KEY=VAL` pairs BEFORE the `--` separator. Each
# pair is exported into the job's environment. This is the robust way
# to thread env vars across LSF clusters whose `bsub` doesn't propagate
# the submitter's environment by default (the alternative — relying on
# `bsub -env "all"` or default propagation — is LSF-version-dependent).
while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
        --env)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --env needs KEY=VAL" >&2; exit 2; }
            export "$1"
            shift
            ;;
        *)
            # Anything else stops the env-arg parsing (treat as a real arg).
            break
            ;;
    esac
done

# Allow an optional `--` separator so submit-script callers can quote
# the rest unambiguously.
if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
fi
SCRIPT_ARGS=("$@")

# --- Env scrubbing --------------------------------------------------
# Stale PYTHONPATH / PYTHONHOME from the user's shell config can shadow
# the venv's site-packages and break `import torch` / `import nichecompass`
# / etc. even after `source activate`. Drop them before activation.
unset PYTHONPATH PYTHONHOME

# --- TMPDIR off NFS -------------------------------------------------
# Multiprocessing tempdirs on NFS hit `.nfsXXXX` silly-rename races at
# process exit ("OSError: [Errno 16] Device or resource busy"). Force
# every subprocess's tempdir onto the node-local SSD. Same fix as in
# squint/examples/_run_one_variant.sh.
export TMPDIR="/tmp/${USER:-$(id -un)}"
mkdir -p "$TMPDIR"

# --- Activate venv --------------------------------------------------
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "ERROR: VENV_PATH=$VENV_PATH has no bin/activate" >&2
    exit 1
fi
echo "[$(date '+%F %T')] activating venv: $VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

# --- CUDA library discovery -----------------------------------------
# Pip-installed torch ships nvidia/<lib>/lib/ shared objects that the
# dynamic linker only finds if LD_LIBRARY_PATH includes them. LSF jobs
# start with a clean shell so `module load cuda/...` hasn't run.
NVIDIA_LIB_DIRS=$(python - <<'PY' 2>/dev/null || true
import glob, os, site
try:
    sp = site.getsitepackages()[0]
    dirs = sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    print(":".join(dirs))
except Exception:
    pass
PY
)
if [[ -n "$NVIDIA_LIB_DIRS" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# --- Cwd-pin (numba stale-cwd guard) --------------------------------
# Pin to TMPDIR so any numba step that lazily calls `os.getcwd()` lands
# in a directory that still exists. Same trick we use in run_novae.py.
cd "$TMPDIR"

# --- Diagnostic preamble --------------------------------------------
echo "  host             : $(hostname)"
echo "  which python     : $(which python)"
echo "  python -V        : $(python --version 2>&1)"
echo "  TMPDIR           : $TMPDIR"
echo "  CUDA devices     : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  LD_LIBRARY_PATH  : ${LD_LIBRARY_PATH:-<empty>}"
echo "  script           : $SCRIPT_PATH"
echo "  args             : ${SCRIPT_ARGS[*]:-<none>}"

# --- Run --------------------------------------------------------------
echo "[$(date '+%F %T')] starting"
python "$SCRIPT_PATH" "${SCRIPT_ARGS[@]}"
echo "[$(date '+%F %T')] finished"
