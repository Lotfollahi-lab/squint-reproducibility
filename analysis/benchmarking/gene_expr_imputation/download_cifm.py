#!/usr/bin/env python
"""
download_cifm.py — fetch the released CIFM checkpoint without git-lfs.
=============================================================================
`git clone https://huggingface.co/ynyou/CIFM` needs git-lfs, and on the farm it
tends to fail (no git-lfs module, or the LFS smudge step dies behind a proxy)
leaving a ~130-byte pointer where `model.safetensors` should be. This script
uses `huggingface_hub.snapshot_download` instead, which:

  * needs no git and no git-lfs,
  * RESUMES partial downloads (the 569 MB weight file is the usual casualty),
  * retries transient failures,
  * and can be pointed at a proxy via the standard HF env vars.

It also VERIFIES what it downloaded, so a truncated file is caught here rather
than 20 minutes into a GPU job:
  - model.safetensors is > 100 MB (i.e. not an LFS pointer),
  - models_cifm/{cifm.py,args.pt,channel2ensembl.pt} are present,
  - channel2ensembl.pt really is the HUMAN vocabulary (~18,289 ENSG ids and no
    ENSMUSG), which is what forces the mouse->human ortholog step in run_cifm.py.

By default we SKIP the repo's demo `adata.h5ad` and `figures/` (a Visium sample
and some gifs — not needed, and adata.h5ad is large). Use --include-demo if you
want the notebook to run as-is.

Usage
-----
  # on a login node with internet
  python download_cifm.py            # -> <repo>/analysis/benchmarking/cifm

  # behind a proxy / mirror
  HF_ENDPOINT=https://hf-mirror.com python download_cifm.py --dest ...

  # just check an existing directory, download nothing
  python download_cifm.py --dest ... --verify-only

Offline compute nodes
---------------------
run_cifm.py loads the weights from --cifm-repo when `model.safetensors` is
present there, so once this script has run the job needs NO internet for the
model. (The ortholog lookup is separate — pin it with
`run_cifm.py --ortholog-only --write-ortholog-csv ...`.)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ID = "ynyou/CIFM"
# beside the other benchmarked models: analysis/benchmarking/cifm
# (parents[0] = gene_expr_imputation, parents[1] = benchmarking)
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "cifm"
REQUIRED = [
    "model.safetensors",
    "models_cifm/cifm.py",
    "models_cifm/args.pt",
    "models_cifm/channel2ensembl.pt",
    "models_cifm/egnn_void_invariant.py",
    "models_cifm/mlp_and_gnn.py",
    "models_cifm/layers/egnn_layer_void_invariant.py",
]
MIN_WEIGHT_BYTES = 100 * 1024 * 1024        # real file is ~569 MB


def verify(dest: Path, strict: bool = True) -> bool:
    """Check the download is complete and is the human-vocabulary checkpoint."""
    ok = True
    print(f"\n=== Verifying {dest} ===")
    for rel in REQUIRED:
        p = dest / rel
        if not p.is_file():
            print(f"  MISSING  {rel}")
            ok = False
            continue
        size = p.stat().st_size
        if rel == "model.safetensors" and size < MIN_WEIGHT_BYTES:
            print(f"  TRUNCATED {rel}: {size/1e6:.1f} MB "
                  f"(expected >{MIN_WEIGHT_BYTES/1e6:.0f} MB — an LFS pointer?)")
            ok = False
        else:
            print(f"  OK       {rel}  ({size/1e6:.2f} MB)")

    # species sanity: this is WHY run_cifm.py needs the ortholog mapping
    c2e = dest / "models_cifm" / "channel2ensembl.pt"
    if c2e.is_file():
        txt = c2e.read_bytes().decode("latin-1")
        n_h = len(set(re.findall(r"ENSG\d{11}", txt)))
        n_m = len(set(re.findall(r"ENSMUSG\d{11}", txt)))
        print(f"  vocabulary: {n_h} unique human ENSG, {n_m} unique mouse ENSMUSG")
        if n_h < 1000:
            print("  WARNING: human vocabulary looks too small — file may be corrupt")
            ok = False
        if n_m:
            print("  NOTE: mouse ids present — re-check whether the ortholog "
                  "mapping in run_cifm.py is still required")
    print("=== VERIFY: " + ("PASSED" if ok else "FAILED") + " ===")
    if strict and not ok:
        return False
    return ok


def download(dest: Path, include_demo: bool, retries: int, token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # noqa: BLE001
        raise SystemExit(
            "huggingface_hub is not installed in this environment. Activate the "
            "cifm venv (or `pip install huggingface_hub`) and retry. "
            f"Original error: {e}")

    allow = ["*.py", "*.pt", "*.json", "*.md", "model.safetensors"]
    if include_demo:
        allow += ["adata.h5ad", "test.ipynb"]
    else:
        allow += ["test.ipynb"]        # tiny, useful as reference
    ignore = None if include_demo else ["adata.h5ad", "figures/*"]

    dest.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(1, retries + 1):
        try:
            print(f"\n=== snapshot_download attempt {attempt}/{retries} "
                  f"-> {dest} ===")
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(dest),
                allow_patterns=allow,
                ignore_patterns=ignore,
                max_workers=4,
                token=token,
                # resume is the default in modern hub versions; harmless if not
            )
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < retries:
                wait = min(60, 5 * attempt)
                print(f"  retrying in {wait}s (partial files are resumed)...")
                time.sleep(wait)
    raise SystemExit(
        f"All {retries} download attempts failed. Last error: {last}\n"
        "Hints:\n"
        "  * login nodes usually have internet; compute nodes often do not\n"
        "  * behind a proxy, set HTTPS_PROXY / HF_ENDPOINT\n"
        "  * to move it manually: download the files from "
        f"https://huggingface.co/{REPO_ID} and place them under {dest}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Download + verify the CIFM checkpoint (no git-lfs needed).")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                   help=f"Target directory (default {DEFAULT_DEST}, beside the "
                        f"other benchmarked models and gitignored). Pass the "
                        f"same path as run_cifm.py --cifm-repo.")
    p.add_argument("--include-demo", action="store_true",
                   help="Also fetch the repo's demo adata.h5ad and figures/.")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--token", default=None, help="HF token (not needed; public).")
    p.add_argument("--verify-only", action="store_true",
                   help="Only check an existing directory; download nothing.")
    args = p.parse_args(argv)

    if not args.verify_only:
        download(args.dest, args.include_demo, args.retries, args.token)

    if not verify(args.dest):
        print("\nDownload/verification FAILED — do not submit the job yet.",
              file=sys.stderr)
        return 1

    print(f"\nReady. Use:  --cifm-repo {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
