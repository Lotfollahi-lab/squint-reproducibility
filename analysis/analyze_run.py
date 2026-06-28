#!/usr/bin/env python3
"""
Unified post-inference analysis — ONE entry point that generates ALL per-run
analysis plots for a trained SQUINT run (or every seed of a variant):

    code_index_plots/        spatial scatter coloured by cell/niche code (L0, L1, ...)
                             -> squint/examples/plot_code_indices_spatial.py
    umap_plots/              UMAP of each embedding by batch / label
                             -> squint/examples/plot_latent_umap.py
    svg_plots/               spatially-variable-gene reconstruction panels
                             -> squint/examples/plot_svg_reconstruction.py
    codebook_usage_plots/    active-code fraction + perplexity per (L, K) level
                             -> analysis/codebook_usage/report_codebook_usage.py

Each generator is run as a subprocess (so they stay independently runnable and a
failure in one never aborts the others) and writes into a per-run subdirectory
named above. Run selection mirrors report_codebook_usage.py:

    # one run (explicit adata / run dir / variant+timestamp)
    python analyze_run.py --predicted-adata /path/predicted_adata.h5ad
    python analyze_run.py --run-dir /path/<timestamp>
    python analyze_run.py --variant <KEY> --timestamp 20260601_120000

    # ALL seed runs of a variant (e.g. the s57_v19 FiLM-scale reference)
    python analyze_run.py --variant s57_v19_reference-filmscale+mmb0-1b_smb1-1b_1p --all-seeds

Select a subset with --only / --skip (code_index,umap,svg,codebook).
Requirements: the squint env (anndata, scanpy, squidpy, matplotlib, pandas).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "mmb0-1b_smb1-1b_1p"
DEFAULT_ARTIFACTS_ROOT = "/nfs/team361/sb75/squint-reproducibility/artifacts"

# Repo layout: this file is squint-reproducibility/analysis/analyze_run.py.
#   parents[0] = analysis/ ; parents[1] = squint-reproducibility/ ;
#   parents[2] = <workspace> (sibling of squint/).
_HERE = Path(__file__).resolve()
_EXAMPLES = Path(os.environ.get(
    "SQUINT_EXAMPLES_DIR", _HERE.parents[2] / "squint" / "examples"))
_CODEBOOK_TOOL = _HERE.parent / "codebook_usage" / "report_codebook_usage.py"

# name -> (per-run subdir, script path, extra default args)
GENERATORS = {
    "code_index": ("code_index_plots", _EXAMPLES / "plot_code_indices_spatial.py", []),
    "umap":       ("umap_plots",        _EXAMPLES / "plot_latent_umap.py", []),
    "svg":        ("svg_plots",         _EXAMPLES / "plot_svg_reconstruction.py", []),
    "codebook":   ("codebook_usage_plots", _CODEBOOK_TOOL, []),
}


def _resolve_runs(args):
    """Return [(label, predicted_adata_path, run_dir)] for the selection."""
    if args.predicted_adata:
        p = Path(args.predicted_adata)
        return [(p.parent.name, str(p), str(p.parent))]
    if args.run_dir:
        rd = Path(args.run_dir)
        return [(rd.name, str(rd / "predicted_adata.h5ad"), str(rd))]

    variant_slug = args.variant.replace("/", "_").replace(" ", "_")
    base = Path(args.artifacts_root) / args.dataset / variant_slug
    if not base.is_dir():
        raise SystemExit(f"Variant dir not found: {base}\n"
                         f"Check --artifacts-root / --dataset / --variant, or pass "
                         f"--predicted-adata / --run-dir.")
    stamps = sorted(d.name for d in base.iterdir()
                    if d.is_dir() and (d / "predicted_adata.h5ad").is_file())
    if not stamps:
        raise SystemExit(f"No <timestamp>/predicted_adata.h5ad under {base}")
    if args.all_seeds:
        return [(ts, str(base / ts / "predicted_adata.h5ad"), str(base / ts))
                for ts in stamps]
    # single: explicit timestamp or latest
    ts = args.timestamp if (args.timestamp and args.timestamp != "latest") else stamps[-1]
    if ts not in stamps:
        raise SystemExit(f"Timestamp {ts!r} not under {base} (have: {stamps})")
    if args.timestamp in (None, "latest"):
        print(f"[analyze] latest timestamp: {ts} (of {len(stamps)})")
    return [(ts, str(base / ts / "predicted_adata.h5ad"), str(base / ts))]


def _run_generator(name, adata_path, out_dir, extra):
    subdir, script, default_extra = GENERATORS[name]
    out = Path(out_dir) / subdir
    if not Path(script).is_file():
        print(f"  [{name}] SKIP — generator not found: {script}", file=sys.stderr)
        return False
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script),
           "--predicted-adata", adata_path, "--out-dir", str(out),
           *default_extra, *extra]
    print(f"  [{name}] -> {out}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [{name}] FAILED (exit {exc.returncode}) — continuing.",
              file=sys.stderr)
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("run selection (pick one; else auto-resolve latest)")
    src.add_argument("--predicted-adata", default=None)
    src.add_argument("--run-dir", default=None)
    src.add_argument("--variant", default=None,
                     help="Variant key (artifact dir name under <root>/<dataset>/).")
    src.add_argument("--dataset", default=DEFAULT_DATASET)
    src.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    src.add_argument("--timestamp", default="latest")
    src.add_argument("--all-seeds", action="store_true",
                     help="Run for EVERY <timestamp> run dir under the variant.")
    ap.add_argument("--only", default=None,
                    help="Comma list of plot types to run "
                         "(code_index,umap,svg,codebook). Default: all.")
    ap.add_argument("--skip", default=None,
                    help="Comma list of plot types to skip.")
    ap.add_argument("--out-root", default=None,
                    help="Optional root: write <out-root>/<run-label>/<subdir>/ "
                         "instead of into each run dir.")
    args, passthrough = ap.parse_known_args(argv)
    if not (args.predicted_adata or args.run_dir or args.variant):
        ap.error("provide one of --predicted-adata / --run-dir / --variant")

    selected = list(GENERATORS)
    if args.only:
        selected = [s.strip() for s in args.only.split(",") if s.strip()]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",")}
        selected = [s for s in selected if s not in skip]
    bad = [s for s in selected if s not in GENERATORS]
    if bad:
        ap.error(f"unknown plot type(s) {bad}; choose from {list(GENERATORS)}")

    runs = _resolve_runs(args)
    print(f"[analyze] {len(runs)} run(s); generators: {selected}")
    print(f"[analyze] examples dir : {_EXAMPLES}")
    print(f"[analyze] codebook tool: {_CODEBOOK_TOOL}\n")

    summary = []
    for label, adata_path, run_dir in runs:
        print(f"\n===== {label} =====\n  adata: {adata_path}")
        if not Path(adata_path).is_file():
            print(f"  SKIP — no predicted_adata.h5ad", file=sys.stderr)
            summary.append((label, "no-adata"))
            continue
        base_out = (Path(args.out_root) / label) if args.out_root else Path(run_dir)
        ok = {name: _run_generator(name, adata_path, base_out, passthrough)
              for name in selected}
        summary.append((label, ",".join(f"{k}:{'ok' if v else 'fail'}"
                                         for k, v in ok.items())))

    print("\n" + "=" * 72)
    print("ANALYZE SUMMARY")
    print("=" * 72)
    for label, status in summary:
        print(f"  {label}: {status}")
    print("\n[analyze] DONE")


if __name__ == "__main__":
    main()
