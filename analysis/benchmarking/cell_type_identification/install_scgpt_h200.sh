#!/usr/bin/env bash
# =============================================================================
# Build an H200 (sm_90 / Hopper) compatible venv for scGPT / scGPT-spatial.
#
# ROOT CAUSE of the H200 crash ("no kernel image is available for execution on
# the device"): the existing scgpt venv's torch was built for <= sm_86 (Ampere/
# A100) and has NO Hopper (sm_90) kernel. It is NOT a flash-attn problem:
#   * scGPT-spatial already runs the NON-flash-attn path by default
#     (--use-fast-transformer is opt-in), and
#   * stock scGPT (run_scgpt.py) now takes --no-fast-transformer.
# So flash-attn is NOT needed. The ONLY fix is a torch with sm_90 kernels
# (torch >= 2.1, cu118/cu121). This script builds exactly that.
#
# RUN ON THE FARM, on a node that can pip-install (ideally with internet).
# I cannot test this from the dev box — VERIFY on an H200 node before submitting
# a full sweep (the script writes a verify_h200.py for that).
#
# Env knobs: VENV, PYTHON, CUDA (cu121|cu118), TORCH, SCGPT_SPATIAL_REPO, SCGPT_PIP.
# =============================================================================
set -euo pipefail

VENV="${VENV:-/nfs/team361/sb75/.venvs/scgpt-h200}"
PYTHON="${PYTHON:-python3.10}"
CUDA="${CUDA:-cu121}"                 # cu121 covers H200/sm_90 AND has a torchtext match
TORCH="${TORCH:-2.3.0}"               # CEILING for scGPT: torchtext's last release (0.18.0)
TORCHVISION="${TORCHVISION:-0.18.0}"  # is built for torch 2.3.0 — there is NO torchtext for
TORCHTEXT="${TORCHTEXT:-0.18.0}"      # torch>=2.4, and scGPT imports torchtext.vocab.
# scGPT-spatial is usually a LOCAL clone (the runner auto-detects the repo near
# the model dir). Point SCGPT_SPATIAL_REPO at that clone to install it editable;
# otherwise only stock scGPT (PyPI) is installed.
SCGPT_SPATIAL_REPO="${SCGPT_SPATIAL_REPO:-}"
SCGPT_PIP="${SCGPT_PIP:-scgpt}"

echo "=========================================================="
echo "scGPT H200 venv : $VENV"
echo "  python=$PYTHON  torch=$TORCH+$CUDA  scgpt_spatial_repo=${SCGPT_SPATIAL_REPO:-<none>}"
echo "=========================================================="

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools

# 1. Install scGPT (+ scGPT-spatial) FIRST so they pull their dependency tree.
if [[ -n "$SCGPT_SPATIAL_REPO" ]]; then
    pip install -e "$SCGPT_SPATIAL_REPO" || pip install "$SCGPT_PIP"
else
    pip install "$SCGPT_PIP"
fi
# Runtime deps the runners import (no-op if already pulled). numpy<2 avoids the
# numpy-2 ABI breakage with older compiled wheels.
pip install "scanpy" "anndata" "numpy<2" "pandas" "scikit-learn"

# 2. ...then FORCE the sm_90 torch LAST, so scGPT's torch pin (historically
#    torch<2.1, which has NO Hopper kernel) cannot downgrade it. This is THE fix.
#    NOTE 1: do NOT pass --no-deps on torch — its CUDA runtime ships as separate
#      nvidia-*-cu12 wheels (cudnn/cublas/...); --no-deps skips them and torch then
#      fails to import with "libcudnn.so.8: cannot open shared object file".
#    NOTE 2: torch/torchvision come from the cuXXX index (matched build); torchtext
#      is installed --no-deps so it can't drag a CPU torch over the cu build. All
#      three MUST be the same minor (2.3.0 / 0.18.0 / 0.18.0) or the C-extension
#      ABI breaks ("undefined symbol: _ZN2at4_ops5zeros...").
pip install --force-reinstall "torch==${TORCH}" "torchvision==${TORCHVISION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA}"
pip install --force-reinstall --no-deps "torchtext==${TORCHTEXT}"

# 3. Deliberately NO flash-attn: the slow (stock-torch attention) path is used on
#    H200. (flash-attn>=2.x does support sm_90 but needs a long Hopper compile and
#    scGPT's flash-attn-1.x API may not match — not worth it. Keep the slow path.)

# ---- write a verifier to run ON AN H200 NODE -------------------------------
cat > "$VENV/verify_h200.py" <<'PYEOF'
import torch
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "no CUDA visible — run this on a GPU node"
print("device:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("capability:", cap, "(expect (9, 0) on H200)")
print("arch_list:", torch.cuda.get_arch_list(), "(must include sm_90)")
assert "sm_90" in torch.cuda.get_arch_list(), "torch has NO sm_90 kernel — wrong build"
x = torch.randn(1024, 1024, device="cuda")
y = float((x @ x).sum())            # the exact op class that crashed before
print("GPU matmul OK:", y == y)
import scgpt; print("scgpt import OK:", getattr(scgpt, "__version__", "?"))
try:
    import scgpt_spatial  # noqa: F401
    print("scgpt_spatial import OK")
except Exception as e:               # noqa: BLE001
    print("scgpt_spatial import (set SCGPT_SPATIAL_REPO if missing):", e)
print("ALL CHECKS PASSED" )
PYEOF

echo
echo "DONE building $VENV"
echo "NEXT — verify on an H200 node (tiny bsub), e.g.:"
echo "  bsub -G s10396 -q training-parallel -gpu 'mode=exclusive_process:num=1:block=yes' -W 0:10 \\"
echo "    bash -lc 'source $VENV/bin/activate && python $VENV/verify_h200.py'"
echo "Expect: capability (9, 0), 'sm_90' in arch_list, 'GPU matmul OK: True', scgpt import OK."
echo
echo "THEN submit on training-parallel (H200), pointing the runners at this venv:"
echo "  scGPT-spatial already defaults to the slow path; stock scGPT needs --no-fast-transformer."
