#!/usr/bin/env python
"""
verify_cifm.py — end-to-end check of the CIFM venv + checkpoint, on a GPU node.
=============================================================================
Run this BEFORE submitting the real imputation job. It checks, in order:

  1. every module run_cifm.py needs imports,
  2. torch matches the farm pin and a real GPU op works on THIS device,
  3. radius_graph works (i.e. torch-cluster is installed and ABI-correct) —
     CIFM calls it on every forward pass,
  4. the checkpoint loads and its vocabulary is the human one (which is why
     run_cifm.py needs the mouse->human ortholog mapping),
  5. channel_matching onto a small panel, and
  6. a real predict_cells_at_locations call returning finite values.

Notes
-----
* CIFM's `predict_cells_at_locations` calls `adata.X.toarray()`, so X must be
  SPARSE. (run_cifm.py never uses that method — it feeds dense arrays straight
  to the model internals — but the released API requires sparse, so we test it
  the way the library expects.)
* A `[transformers] Disabling PyTorch because PyTorch >= 2.5 is required`
  message is BENIGN: cifm.py imports huggingface_hub, not transformers, and the
  model loads regardless. We keep torch at the farm pin (2.2.0) deliberately.
* We do NOT assert a particular sm_XX: `arch_list` describes the torch build,
  not the GPU, and this queue serves both L40S (sm_89) and H200 (sm_90). The
  matmul in step 2 is the real test.

Usage
-----
  python verify_cifm.py                        # defaults to ../cifm
  python verify_cifm.py --cifm-repo /path/to/cifm --expect-torch 2.2.0
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

DEFAULT_CIFM_REPO = Path(__file__).resolve().parents[1] / "cifm"
NEEDED = ("torch", "torch_geometric", "torch_cluster", "torch_scatter",
          "torch_sparse", "e3nn", "scanpy", "anndata", "pandas", "sklearn",
          "mygene", "huggingface_hub", "h5py", "scipy")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify the CIFM venv + checkpoint.")
    p.add_argument("--cifm-repo", type=Path, default=DEFAULT_CIFM_REPO)
    p.add_argument("--expect-torch", default="2.2.0",
                   help="Farm pin; the PyG wheels are ABI-matched to it.")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Skip the CUDA checks (for a quick import-only test).")
    args = p.parse_args(argv)

    # ---- 1. imports --------------------------------------------------------
    missing = []
    for mod in NEEDED:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod} ({type(e).__name__}: {e})")
    print("1. dependency imports:", "ALL OK" if not missing else "MISSING")
    for m in missing:
        print("   !!", m)
    if missing:
        return 1

    import numpy as np
    import torch

    # ---- 2. torch / device -------------------------------------------------
    print(f"2. torch {torch.__version__} | built for CUDA {torch.version.cuda}")
    if not torch.__version__.startswith(args.expect_torch):
        print(f"   !! torch {torch.__version__} != farm pin {args.expect_torch}; "
              f"the PyG companion wheels are ABI-matched to the pin and WILL "
              f"break at radius_graph")
        return 1
    if args.allow_cpu and not torch.cuda.is_available():
        print("   (no CUDA; --allow-cpu given, skipping GPU checks)")
        device = "cpu"
    else:
        if not torch.cuda.is_available():
            print("   !! no CUDA visible — run this on a GPU node")
            return 1
        device = "cuda"
        cap = torch.cuda.get_device_capability(0)
        print(f"   device: {torch.cuda.get_device_name(0)} | capability {cap}")
        print(f"   arch_list: {torch.cuda.get_arch_list()}")
        if f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
            print(f"   note: no exact sm_{cap[0]}{cap[1]} cubin; relying on "
                  f"PTX/minor-version compatibility — the matmul is the real test")
        x = torch.randn(1024, 1024, device=device)
        y = float((x @ x).sum())
        if y != y:
            print("   !! GPU matmul produced NaN — wrong torch build for this device")
            return 1
        print("   GPU matmul OK")

    # ---- 3. radius_graph (torch-cluster) -----------------------------------
    from torch_geometric.nn import radius_graph
    ei = radius_graph(torch.randn(500, 3, device=device), r=1.0,
                      max_num_neighbors=10000, loop=True)
    print(f"3. radius_graph OK, edges: {int(ei.shape[1])}")

    # ---- 4. checkpoint ------------------------------------------------------
    repo = args.cifm_repo.resolve()
    if not (repo / "models_cifm" / "cifm.py").is_file():
        print(f"4. !! {repo} does not look like the CIFM repo "
              f"(no models_cifm/cifm.py). Run download_cifm.py.")
        return 1
    sys.path.insert(0, str(repo))
    from models_cifm.cifm import CIFM  # noqa: E402

    args_model = torch.load(repo / "models_cifm" / "args.pt")
    src_local = (repo / "model.safetensors").is_file()
    model = CIFM.from_pretrained(str(repo) if src_local else "ynyou/CIFM",
                                 args=args_model).to(device)
    src = torch.load(repo / "models_cifm" / "channel2ensembl.pt")
    model.channel2ensembl_ids_source = src
    model.eval()
    print(f"4. CIFM loaded from {'LOCAL weights' if src_local else 'the Hub'} | "
          f"radius_spatial_graph={model.radius_spatial_graph} | "
          f"source channels={len(src)}")
    blob = (repo / "models_cifm" / "channel2ensembl.pt").read_bytes().decode("latin-1")
    n_h = len(set(re.findall(r"ENSG\d{11}", blob)))
    n_m = len(set(re.findall(r"ENSMUSG\d{11}", blob)))
    print(f"   vocabulary: {n_h} human ENSG / {n_m} mouse ENSMUSG "
          f"-> mouse data NEEDS the ortholog map (as run_cifm.py does)")
    if n_h < 1000:
        print("   !! human vocabulary too small — checkpoint may be corrupt")
        return 1

    # ---- 5/6. channel_matching + a real prediction -------------------------
    import anndata as ad
    from scipy.sparse import csr_matrix

    tgt = [e[:1] for e in src[:3]]          # a 3-gene 'panel' from the vocab
    model.channel_matching(tgt, src)
    G = len(tgt)
    X = np.abs(np.random.RandomState(0).randn(200, G)).astype("float32")
    a = ad.AnnData(X=csr_matrix(X))         # MUST be sparse: X.toarray() inside
    a.obsm["spatial"] = np.random.RandomState(1).rand(200, 2) * 200.0  # ~200um
    with torch.no_grad():
        out = model.predict_cells_at_locations(
            a, np.array([[50.0, 50.0], [90.0, 30.0]]))
    print(f"5. channel_matching + predict_cells_at_locations OK -> "
          f"shape {tuple(out.shape)} (expect (2, {G}))")
    if tuple(out.shape) != (2, G) or not bool(torch.isfinite(out).all()):
        print("   !! unexpected shape or non-finite predictions")
        return 1

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
