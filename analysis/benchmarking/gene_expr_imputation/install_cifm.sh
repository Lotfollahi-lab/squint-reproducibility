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
#  3. model.safetensors is 569 MB behind git-lfs. A plain `git clone` yields a
#     ~130-byte pointer file and `from_pretrained` fails with a confusing
#     safetensors header error. git-lfs must be installed BEFORE cloning.
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
# H200 note: unlike scGPT (see ../cell_type_identification/install_scgpt_h200.sh,
# which is capped at torch 2.3.0 by torchtext), CIFM has no such ceiling, and
# torch 2.2.0+cu121 ships sm_90 kernels — so the default training-parallel/s10396
# (H200) queue works and no A100 fallback is needed. The verifier asserts sm_90.
#
# NETWORK note: run_cifm.py resolves mouse->human orthologs through mygene.info
# and the Ensembl REST API at runtime. If the COMPUTE nodes have no outbound
# internet, precompute the mapping on a login node first (see PRECOMPUTE below)
# and pass --ortholog-csv, otherwise the job dies after loading the data.
#
# RUN ON THE FARM, on a node that can pip-install (ideally a login node with
# internet). I cannot test this from the dev box — run the verifier on a GPU
# node before submitting the real job.
#
# Env knobs: VENV, PYTHON, CUDA (cu121|cu118), TORCH, CIFM_REPO, PYG_INDEX.
# =============================================================================
set -euo pipefail

VENV="${VENV:-/nfs/team361/sb75/.venvs/cifm}"   # name must match submit_imputation_recon.sh
PYTHON="${PYTHON:-python3.10}"        # squint/pyproject.toml: requires-python >=3.10,<3.13
CUDA="${CUDA:-cu121}"                 # cu121 has sm_90 (H200) kernels
TORCH="${TORCH:-2.2.0}"               # == squint/pyproject.toml (farm parity)
# exact PyG companion pins from squint/pyproject.toml — must match TORCH's ABI
PYG_LIB="${PYG_LIB:-0.4.0}"
TORCH_CLUSTER="${TORCH_CLUSTER:-1.6.3}"
TORCH_SCATTER="${TORCH_SCATTER:-2.1.2}"
TORCH_SPARSE="${TORCH_SPARSE:-0.6.18}"
TORCH_SPLINE="${TORCH_SPLINE:-1.2.2}"
CIFM_REPO="${CIFM_REPO:-/nfs/team361/sb75/models/CIFM}"
PYG_INDEX="${PYG_INDEX:-https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html}"

echo "=========================================================="
echo "CIFM venv   : $VENV"
echo "  python=$PYTHON  torch=$TORCH+$CUDA"
echo "  pyg wheels: $PYG_INDEX"
echo "  checkpoint: $CIFM_REPO"
echo "=========================================================="

"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools

# 1. torch FIRST, from the CUDA index, so the PyG wheels in step 2 can match it.
#    Do NOT pass --no-deps: torch's CUDA runtime ships as separate nvidia-*-cu12
#    wheels (cudnn/cublas/...) and skipping them breaks the import.
pip install "torch==${TORCH}" --index-url "https://download.pytorch.org/whl/${CUDA}"

# 2. numpy<2 BEFORE the compiled extensions — same bound as squint/pyproject.toml
#    ("numpy>=1.24,<2"; CIFM's card asks for 1.26.4, which is inside it). numpy 2
#    changed the C ABI and the prebuilt scatter/sparse/cluster wheels were
#    compiled against 1.x ("numpy.dtype size changed" at import).
pip install "numpy>=1.24,<2"

# 3. The compiled PyG extensions, at the SAME pins as the squint venv, from the
#    version-matched wheel index (pitfalls 1/2 above). torch-cluster is what
#    provides radius_graph, which CIFM calls on every forward pass.
pip install "pyg-lib==${PYG_LIB}" "torch-cluster==${TORCH_CLUSTER}" \
            "torch-scatter==${TORCH_SCATTER}" "torch-sparse==${TORCH_SPARSE}" \
            "torch-spline-conv==${TORCH_SPLINE}" -f "$PYG_INDEX"
pip install "torch-geometric>=2.5,<2.7"

# 4. Everything else CIFM + run_cifm.py import. Version ranges follow
#    squint/pyproject.toml where they overlap, so behaviour matches GeST.
#    - e3nn            : the equivariant layers in models_cifm/egnn_*
#    - pytorch-lightning: imported by the released model package (same range as
#                        the squint venv; CIFM's card says lightning==2.1.0, but
#                        cifm.py itself does not import it)
#    - transformers    : pulled in by the PyTorchModelHubMixin paths
#    - huggingface_hub : from_pretrained('ynyou/CIFM')
#    - mygene/requests : the mouse->human ortholog lookup (same helper as scGPT)
#    - scikit-learn    : NearestNeighbors, for the leak-free read depth
#    - scanpy/anndata  : normalize_total/log1p + the AnnData plumbing
pip install "e3nn" "pytorch-lightning>=2.2,<2.5" "transformers" \
            "huggingface_hub" "scanpy>=1.10" "anndata>=0.10" "pandas" \
            "scikit-learn" "mygene" "requests" "h5py>=3.10"

# ---- fetch the released checkpoint (git-lfs; see pitfall 3) -----------------
if [[ -d "$CIFM_REPO/.git" ]]; then
    echo "Checkpoint already cloned at $CIFM_REPO — pulling lfs objects"
    ( cd "$CIFM_REPO" && git lfs install --local && git lfs pull )
else
    mkdir -p "$(dirname "$CIFM_REPO")"
    if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
        echo "ERROR: git-lfs is not available. Install it (module load git-lfs, or"
        echo "       conda install git-lfs) and re-run — a plain clone gives a"
        echo "       pointer file and the model will not load." >&2
        exit 1
    fi
    git lfs install
    git clone "https://huggingface.co/ynyou/CIFM" "$CIFM_REPO"
fi

# fail loudly now rather than mid-job if lfs did not materialise the weights
_W="$CIFM_REPO/model.safetensors"
if [[ ! -f "$_W" ]] || [[ "$(stat -c%s "$_W" 2>/dev/null || stat -f%z "$_W")" -lt 100000000 ]]; then
    echo "ERROR: $_W is missing or too small (git-lfs pointer?). Run:" >&2
    echo "       cd $CIFM_REPO && git lfs install --local && git lfs pull" >&2
    exit 1
fi
echo "checkpoint OK: $(du -h "$_W" | cut -f1) $_W"

# ---- write an end-to-end verifier to run ON A GPU NODE ---------------------
cat > "$VENV/verify_cifm.py" <<PYEOF
import importlib, sys, numpy as np, torch
CIFM_REPO = "$CIFM_REPO"
EXPECT_TORCH = "$TORCH"

# ---- 0. every module run_cifm.py imports must be present -------------------
missing = []
for mod in ("torch", "torch_geometric", "torch_cluster", "torch_scatter",
            "torch_sparse", "e3nn", "scanpy", "anndata", "pandas", "sklearn",
            "mygene", "huggingface_hub", "h5py"):
    try:
        importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        missing.append(f"{mod} ({type(e).__name__}: {e})")
print("dependency import check:", "ALL OK" if not missing else "MISSING")
for m in missing:
    print("   !!", m)
assert not missing, "install incomplete — see above"

print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
assert torch.__version__.startswith(EXPECT_TORCH), (
    f"torch {torch.__version__} != farm pin {EXPECT_TORCH} — the PyG companion "
    f"wheels are ABI-matched to the pin, so this WILL break at radius_graph")
assert torch.cuda.is_available(), "no CUDA visible — run this on a GPU node"
print("device:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
print("arch_list:", torch.cuda.get_arch_list())
assert "sm_90" in torch.cuda.get_arch_list(), "torch has no sm_90 kernel (H200 will fail)"

# radius_graph is the op CIFM needs on every forward pass -> proves torch-cluster
from torch_geometric.nn import radius_graph
xyz = torch.randn(500, 3, device="cuda")
ei = radius_graph(xyz, r=1.0, max_num_neighbors=10000, loop=True)
print("radius_graph OK, edges:", int(ei.shape[1]))

sys.path.insert(0, CIFM_REPO)
from models_cifm.cifm import CIFM
args_model = torch.load(CIFM_REPO + "/models_cifm/args.pt")
model = CIFM.from_pretrained("ynyou/CIFM", args=args_model).to("cuda")
src = torch.load(CIFM_REPO + "/models_cifm/channel2ensembl.pt")
model.channel2ensembl_ids_source = src
model.eval()
print("CIFM loaded | radius_spatial_graph =", model.radius_spatial_graph,
      "| source channels =", len(src))
n_h = sum(1 for e in src for x in e if str(x).startswith("ENSG"))
print("source vocabulary is human ENSG entries:", n_h, "(expect ~18289)")

# channel_matching onto a tiny 3-gene 'panel' taken from the source vocab
tgt = [e[:1] for e in src[:3]]
model.channel_matching(tgt, src)
G = len(tgt)
import anndata as ad
X = np.abs(np.random.RandomState(0).randn(200, G)).astype("float32")
a = ad.AnnData(X=X)
a.obsm["spatial"] = (np.random.RandomState(1).rand(200, 2) * 200.0)  # ~200um field
with torch.no_grad():
    out = model.predict_cells_at_locations(a, np.array([[50.0, 50.0], [90.0, 30.0]]))
print("predict_cells_at_locations OK -> shape", tuple(out.shape), "(expect (2,", G, "))")
assert out.shape == (2, G)
assert torch.isfinite(out).all(), "non-finite predictions"
print("ALL CHECKS PASSED")
PYEOF

echo
echo "DONE building $VENV"
echo
echo "NEXT 1 — verify on a GPU node (tiny bsub):"
echo "  bsub -G s10396 -q training-parallel -gpu 'mode=exclusive_process:num=1:block=yes' -W 0:20 \\"
echo "    bash -lc 'source $VENV/bin/activate && python $VENV/verify_cifm.py'"
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
