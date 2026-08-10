#!/usr/bin/env bash
# =============================================================================
# Build the venv for the CIFM imputation baseline (run_cifm.py) and fetch the
# released checkpoint (ynyou/CIFM).
#
# CIFM is a PRETRAINED 100M-param geometric GNN — there is nothing to train, so
# this venv only needs inference deps. Three things make it fiddly, and each is
# a silent-failure mode rather than a clean error:
#
#  1. torch-scatter / torch-sparse / torch-cluster are COMPILED C extensions
#     pinned to an exact (torch, CUDA) pair. From PyPI they trigger a source
#     build that either takes ~an hour or links against the wrong ABI
#     ("undefined symbol: _ZN2at4_ops..."). They MUST come from the PyG wheel
#     index for this exact torch build:
#         https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
#  2. `radius_graph` — which CIFM calls on EVERY forward pass — is implemented
#     in torch-cluster, dispatched via torch_geometric.nn. Without torch-cluster
#     the model imports fine and then dies at graph construction. It is
#     REQUIRED, not optional.
#  3. model.safetensors is 569 MB behind git-lfs, and `git clone` on the farm
#     tends to leave a ~130-byte pointer file (no git-lfs module, or the smudge
#     step dies behind a proxy) -> a confusing safetensors header error later.
#     We therefore fetch it with huggingface_hub via download_cifm.py, which
#     needs no git at all, resumes partial transfers, and verifies the result.
#
# TORCH VERSION — deliberately the FARM's version, not CIFM's card.
# CIFM's model card suggests torch==2.1.0/2.0.1, but every other method in this
# benchmark runs on the pinned stack from squint/pyproject.toml:
#     torch==2.2.0 (cu121), torch-geometric>=2.5,<2.7, pyg-lib==0.4.0,
#     torch-cluster==1.6.3, torch-scatter==2.1.2, torch-sparse==0.6.18
# We reuse those EXACT pins so CIFM shares the CUDA runtime and PyG ABI with
# GeST / SQUINT and there is one less way for the farm run to differ. CIFM only
# uses standard nn + e3nn + torch_geometric ops (see models_cifm/cifm.py), so
# 2.2.0 is safe; override with TORCH=2.1.0 if a future release needs it (the PyG
# find-links URL follows TORCH automatically).
#
# GPU note: unlike scGPT (see ../cell_type_identification/install_scgpt_h200.sh,
# which is capped at torch 2.3.0 by torchtext), CIFM has no such ceiling. The
# default training-parallel/s10396 queue serves both L40S (sm_89) and H200
# (sm_90); torch 2.2.0+cu121 covers both, so no A100 fallback is needed. The
# verifier does NOT assert a specific sm_XX (arch_list describes the torch build,
# not the device) — it runs a real GPU matmul instead.
#
# NETWORK note: run_cifm.py resolves mouse->human orthologs through mygene.info
# and the Ensembl REST API at runtime. If the COMPUTE nodes have no outbound
# internet, precompute the mapping on a login node first (see PRECOMPUTE below)
# and pass --ortholog-csv, otherwise the job dies after loading the data.
#
# Installs with **uv**, matching squint/pyproject.toml's [tool.uv] setup (the
# data.pyg.org find-links index + the explicit pytorch-cu121 index). uv will
# fetch a managed CPython if the node has no python3.10. It ALSO downloads the
# checkpoint at the end (via download_cifm.py — no git-lfs needed).
#
# RUN ON THE FARM, on a node with internet (a login node). I cannot test this
# from the dev box — run the verifier on a GPU node before the real job.
#
# Env knobs: VENV, PYTHON, CUDA (cu121|cu118), TORCH, CIFM_REPO, PYG_INDEX,
#   PYG_LIB, TORCH_CLUSTER, TORCH_SCATTER, TORCH_SPARSE, TORCH_SPLINE.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

VENV="${VENV:-/nfs/team361/sb75/.venvs/cifm}"   # name must match submit_imputation_recon.sh
PYTHON="${PYTHON:-python3.10}"        # squint/pyproject.toml: requires-python >=3.10,<3.13
CUDA="${CUDA:-cu121}"                 # covers L40S (sm_89) and H200 (sm_90)
TORCH="${TORCH:-2.2.0}"               # == squint/pyproject.toml (farm parity)
# exact PyG companion pins from squint/pyproject.toml — must match TORCH's ABI
PYG_LIB="${PYG_LIB:-0.4.0}"
TORCH_CLUSTER="${TORCH_CLUSTER:-1.6.3}"
TORCH_SCATTER="${TORCH_SCATTER:-2.1.2}"
TORCH_SPARSE="${TORCH_SPARSE:-0.6.18}"
TORCH_SPLINE="${TORCH_SPLINE:-1.2.2}"
# Checkpoint lives beside the other benchmarked models, i.e.
# analysis/benchmarking/cifm — same convention as geneformer / nicheformer /
# scGPT / uce_model, and already covered by .gitignore so the 569 MB weight file
# is never committed.
CIFM_REPO="${CIFM_REPO:-$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )/cifm}"
PYG_INDEX="${PYG_INDEX:-https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html}"

echo "=========================================================="
echo "CIFM venv   : $VENV"
echo "  python=$PYTHON  torch=$TORCH+$CUDA"
echo "  pyg wheels: $PYG_INDEX"
echo "  checkpoint: $CIFM_REPO"
echo "=========================================================="

# ---- uv ---------------------------------------------------------------------
# We install with uv, the same tool squint/pyproject.toml is configured for
# ([tool.uv] find-links + the explicit pytorch-cu121 index). The flags below are
# the CLI equivalents of those blocks, so this venv resolves from exactly the
# same two wheel sources as the squint venv.
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found. Install it with one of:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh   # then re-open the shell" >&2
    echo "  pip install --user uv" >&2
    exit 1
fi
echo "uv: $(uv --version)"

# uv fetches a managed CPython if $PYTHON is not on the system, so this works on
# nodes without a system python3.10.
uv venv --python "${PYTHON#python}" "$VENV"
PY="$VENV/bin/python"

# 1. torch FIRST, from the CUDA index, so the PyG wheels in step 3 can match its
#    ABI. == [tool.uv.sources] torch -> index "pytorch-cu121" in the squint
#    pyproject. Do NOT add --no-deps: torch's CUDA runtime ships as separate
#    nvidia-*-cu12 wheels (cudnn/cublas/...) and skipping them breaks the import.
uv pip install --python "$PY" "torch==${TORCH}" \
    --index-url "https://download.pytorch.org/whl/${CUDA}"

# 2. numpy<2 BEFORE the compiled extensions — same bound as squint/pyproject.toml
#    ("numpy>=1.24,<2"; CIFM's card asks for 1.26.4, which is inside it). numpy 2
#    changed the C ABI and the prebuilt scatter/sparse/cluster wheels were
#    compiled against 1.x ("numpy.dtype size changed" at import).
uv pip install --python "$PY" "numpy>=1.24,<2"

# 3. The compiled PyG extensions, at the SAME pins as the squint venv, resolved
#    from the version-matched wheel index (== [tool.uv] find-links). See pitfalls
#    1/2 in the header: torch-cluster is what provides radius_graph, which CIFM
#    calls on every forward pass. --no-build-isolation only matters if no wheel
#    matches and uv falls back to a source build (their setup.py imports torch,
#    which is already installed by step 1).
uv pip install --python "$PY" --find-links "$PYG_INDEX" --no-build-isolation \
    "pyg-lib==${PYG_LIB}" "torch-cluster==${TORCH_CLUSTER}" \
    "torch-scatter==${TORCH_SCATTER}" "torch-sparse==${TORCH_SPARSE}" \
    "torch-spline-conv==${TORCH_SPLINE}"
uv pip install --python "$PY" "torch-geometric>=2.5,<2.7"

# 4. Everything else CIFM + run_cifm.py import. Version ranges follow
#    squint/pyproject.toml where they overlap, so behaviour matches GeST.
#    - e3nn            : the equivariant layers in models_cifm/egnn_*
#    - pytorch-lightning: imported by the released model package (same range as
#                        the squint venv; CIFM's card says lightning==2.1.0, but
#                        cifm.py itself does not import it)
#    - transformers    : pulled in by the PyTorchModelHubMixin paths
#    - huggingface_hub : from_pretrained() + the checkpoint download
#    - mygene/requests : the mouse->human ortholog lookup (same helper as scGPT)
#    - scikit-learn    : NearestNeighbors, for the leak-free read depth
#    - scanpy/anndata  : normalize_total/log1p + the AnnData plumbing
uv pip install --python "$PY" "e3nn" "pytorch-lightning>=2.2,<2.5" \
    "transformers" "huggingface_hub" "scanpy>=1.10" "anndata>=0.10" "pandas" \
    "scikit-learn" "mygene" "requests" "h5py>=3.10"

# ---- the verifier is a REAL FILE in the repo (verify_cifm.py) --------------
# It used to be written here as a heredoc, which meant a fix required
# rebuilding the venv. It now lives beside this script so it can be edited
# and re-run independently. Nothing to do at install time.

# ---- fetch the released checkpoint (NO git-lfs; see pitfall 3) --------------
# huggingface_hub.snapshot_download resumes partial transfers and needs neither
# git nor git-lfs, both of which are unreliable here. download_cifm.py also
# verifies the result (size + that the vocabulary really is human ENSG), so a
# truncated weight file is caught now instead of inside a GPU job.
echo
echo "=== Downloading checkpoint -> $CIFM_REPO ==="
if "$PY" "$SCRIPT_DIR/download_cifm.py" --dest "$CIFM_REPO"; then
    echo "checkpoint OK"
else
    echo >&2
    echo "WARNING: checkpoint download failed — the VENV AND VERIFIER ARE STILL" >&2
    echo "         USABLE. Retry just the download (it resumes) with:" >&2
    echo "  source $VENV/bin/activate" >&2
    echo "  python $SCRIPT_DIR/download_cifm.py --dest $CIFM_REPO" >&2
    echo "If the farm has no outbound internet on this node, run that on a login" >&2
    echo "node, or fetch the files manually from https://huggingface.co/ynyou/CIFM" >&2
    exit 1
fi

echo
echo "DONE building $VENV"
echo
echo "NEXT 1 — verify on a GPU node (tiny bsub):"
echo "  bsub -G s10396 -q training-parallel -gpu 'mode=exclusive_process:num=1:block=yes' -W 0:20 \\"
echo "    bash -lc 'source $VENV/bin/activate && python $SCRIPT_DIR/verify_cifm.py'"
echo "  Expect: sm_90 present, radius_graph OK, ~18289 human ENSG entries, ALL CHECKS PASSED."
echo
echo "NEXT 2 (PRECOMPUTE, only if compute nodes lack internet) — on a LOGIN node:"
echo "  source $VENV/bin/activate"
echo "  python analysis/benchmarking/gene_expr_imputation/run_cifm.py \\"
echo "      --cifm-repo $CIFM_REPO --write-ortholog-csv orthologs_mmb.csv --ortholog-only"
echo "  then pass --ortholog-csv orthologs_mmb.csv to the real run."
echo
echo "NEXT 3 — smoke test the full pipeline (4k cells, minutes):"
echo "  python analysis/benchmarking/gene_expr_imputation/run_cifm.py \\"
echo "      --cifm-repo $CIFM_REPO --smoke --seeds 0"
echo "  CHECK the coordinate diagnostic line: CIFM assumes MICROMETRES (r=20)."
echo
echo "THEN the real run:"
echo "  METHODS=cifm CIFM_REPO=$CIFM_REPO bash \\"
echo "      analysis/benchmarking/gene_expr_imputation/submit_imputation_recon.sh"
