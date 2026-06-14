#!/usr/bin/env python3
"""
Fetch the Human Tonsil Cell Atlas SPATIAL data (10x Visium) and store it as
one AnnData (.h5ad) per Visium sample/section, ready for the SQUINT silver
layout.

    Massoni-Badosa et al., "An atlas of cells in the human tonsil",
    Immunity 2024 (PMID 38301653).

The spatial data is distributed only as an R/Bioconductor object
(`HCATonsilData("Spatial")` -> a SpatialExperiment), so this script:

  1. Shells out to Rscript to fetch the SpatialExperiment via HCATonsilData
     and export it to ONE combined .h5ad via zellkonverter::writeH5AD
     (spatialCoords are also copied into colData so the Python side can
     always rebuild obsm["spatial"]).
  2. Loads that combined .h5ad in Python, splits it by sample, and writes
     one .h5ad per sample into the output directory.
  3. (default) Adds SQUINT-convention aliases on each per-sample object so
     it drops straight into the spatch-style pipeline:
        obs["annotation"]      = cell-type label   (from --celltype-key)
        obs["spatial_cluster"] = niche/region label (from --niche-key)
        obs["batch"]           = the sample id (constant within a file)
        obsm["spatial"]        = (x, y) spot coordinates
     The original columns (e.g. annotation_20230508, area) are preserved.

Requirements
------------
R side  : R, BiocManager, HCATonsilData, SpatialExperiment, zellkonverter,
          SummarizedExperiment  (run once with --install-deps to install).
Py side : anndata, numpy, pandas, scipy.

Usage
-----
    python fetch_tonsil_spatial.py \
        --out-dir /nfs/team361/sb75/DATASETS/silver/squint_vht
    # first run on a fresh machine may need:  --install-deps

Notes
-----
- Needs network access to Bioconductor/ExperimentHub (run on a node that
  can reach the internet).
- This was authored without a local R/HCATonsilData install to test against,
  so the R step prints the object class, colData columns and assay names
  before exporting — if the HCATonsilData API or column names differ, use
  those printouts to set --sample-key / --celltype-key / --niche-key.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

# Default candidate columns to identify the per-sample / section field.
_SAMPLE_KEY_CANDIDATES = [
    "sample_id", "sample", "library_id", "library", "donor_id", "donor",
    "slide", "section", "Sample", "orig.ident", "specimen",
]
_CELLTYPE_DEFAULT = "annotation_20230508"
_NICHE_DEFAULT = "area"


# ---------------------------------------------------------------------------
# R step: HCATonsilData("Spatial") -> combined .h5ad
# ---------------------------------------------------------------------------
_R_TEMPLATE = r"""
args <- commandArgs(trailingOnly = TRUE)
out_h5ad <- args[[1]]
install_deps <- length(args) >= 2 && args[[2]] == "install"
user_lib <- if (length(args) >= 3 && nzchar(args[[3]])) args[[3]]
            else Sys.getenv("R_LIBS_USER")

## The system R library is typically READ-ONLY on a cluster, so install
## into a WRITABLE personal library: use --r-libs / R_LIBS_USER if given,
## else a HOME-based default. Create it and put it first on .libPaths()
## (so both installs and requireNamespace() see it).
if (is.na(user_lib) || !nzchar(user_lib) || identical(user_lib, "NULL")) {
    user_lib <- file.path(Sys.getenv("HOME"), "R",
                          paste0(R.version$platform, "-library-",
                                 paste(getRversion()[, 1:2], collapse = ".")))
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
cat("R library (install target): ", user_lib, "\n", sep = "")

pkgs <- c("HCATonsilData", "SpatialExperiment", "zellkonverter",
          "SummarizedExperiment")
if (install_deps) {
    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager",
                         repos = "https://cloud.r-project.org", lib = user_lib)
    BiocManager::install(pkgs, update = FALSE, ask = FALSE, lib = user_lib)
}

missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
    stop(sprintf(
        "Missing R packages: %s. Re-run with --install-deps and a writable --r-libs dir, or install manually:\n  .libPaths(c('%s', .libPaths())); BiocManager::install(c(%s), lib='%s')",
        paste(missing, collapse = ", "), user_lib,
        paste(sprintf('\"%s\"', missing), collapse = ", "), user_lib))
}

suppressMessages({
    library(HCATonsilData); library(SpatialExperiment)
    library(zellkonverter);  library(SummarizedExperiment)
})

cat("HCATonsilData version: ",
    as.character(packageVersion("HCATonsilData")), "\n", sep = "")

## Fetch the spatial (Visium) SpatialExperiment.
spe <- HCATonsilData("Spatial")

cat("=== fetched object ===\n")
cat("class      : ", paste(class(spe), collapse = ", "), "\n", sep = "")
cat("dim (genes x spots): ", paste(dim(spe), collapse = " x "), "\n", sep = "")
cat("assays     : ", paste(assayNames(spe), collapse = ", "), "\n", sep = "")
cat("colData cols: ", paste(colnames(colData(spe)), collapse = ", "),
    "\n", sep = "")

## Copy spatialCoords into colData so the Python side can rebuild
## obsm['spatial'] regardless of how zellkonverter maps coords.
if (methods::is(spe, "SpatialExperiment")) {
    sc <- SpatialExperiment::spatialCoords(spe)
    if (!is.null(sc) && ncol(sc) >= 2) {
        colData(spe)$spatial_x <- as.numeric(sc[, 1])
        colData(spe)$spatial_y <- as.numeric(sc[, 2])
        cat("stashed spatialCoords -> colData$spatial_x / spatial_y\n")
    }
}

## Prefer raw counts as X (SQUINT's NB loss expects counts).
xname <- if ("counts" %in% assayNames(spe)) "counts" else assayNames(spe)[1]
cat("writing X from assay: ", xname, "\n", sep = "")

dir.create(dirname(out_h5ad), recursive = TRUE, showWarnings = FALSE)
zellkonverter::writeH5AD(spe, file = out_h5ad, X_name = xname)
cat("WROTE_COMBINED_H5AD: ", out_h5ad, "\n", sep = "")
"""


def _run_r_export(combined_h5ad: str, rscript: str,
                  install_deps: bool, r_libs: str | None) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as fh:
        fh.write(_R_TEMPLATE)
        r_path = fh.name
    # NOT --vanilla: that implies --no-environ and would suppress
    # R_LIBS_USER. We pass the lib explicitly (arg 3) and also keep the
    # environment so a configured R_LIBS_USER is honoured.
    cmd = [rscript, "--no-save", "--no-restore", r_path, combined_h5ad,
           "install" if install_deps else "noinstall", r_libs or ""]
    print(f"[fetch] running R: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(r_path)
    if not os.path.isfile(combined_h5ad):
        raise SystemExit(
            f"R step did not produce {combined_h5ad}. See the R output above.")


# ---------------------------------------------------------------------------
# Python step: split combined .h5ad per sample
# ---------------------------------------------------------------------------
def _detect_sample_key(obs, user_key):
    import pandas as pd  # noqa: F401
    if user_key:
        if user_key not in obs.columns:
            raise SystemExit(
                f"--sample-key {user_key!r} not in obs columns: "
                f"{list(obs.columns)}")
        return user_key
    for c in _SAMPLE_KEY_CANDIDATES:
        if c in obs.columns:
            n = obs[c].nunique()
            if 1 < n <= max(2, len(obs) // 10):
                return c
    raise SystemExit(
        "Could not auto-detect a sample key. Pass --sample-key explicitly. "
        f"Available obs columns: {list(obs.columns)}")


def _ensure_spatial(adata):
    import numpy as np
    if "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2:
        return
    cols = adata.obs.columns
    pairs = [("spatial_x", "spatial_y"), ("x", "y"),
             ("pxl_col_in_fullres", "pxl_row_in_fullres"),
             ("imagecol", "imagerow"), ("array_col", "array_row")]
    for cx, cy in pairs:
        if cx in cols and cy in cols:
            adata.obsm["spatial"] = np.c_[
                adata.obs[cx].to_numpy(dtype=float),
                adata.obs[cy].to_numpy(dtype=float)]
            print(f"[fetch] built obsm['spatial'] from obs[{cx!r},{cy!r}]")
            return
    print("[fetch] WARNING: no spatial coordinates found — obsm['spatial'] "
          "will be missing (SQUINT needs it).", file=sys.stderr)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "sample"


def split_per_sample(combined_h5ad, out_dir, sample_key, celltype_key,
                     niche_key):
    import anndata as ad
    adata = ad.read_h5ad(combined_h5ad)
    print(f"[fetch] combined: {adata.n_obs} spots x {adata.n_vars} genes")
    print(f"[fetch] obs columns: {list(adata.obs.columns)}")

    skey = _detect_sample_key(adata.obs, sample_key)
    print(f"[fetch] sample key: {skey!r} "
          f"({adata.obs[skey].nunique()} samples)")

    for key, label in [(celltype_key, "cell-type"), (niche_key, "niche")]:
        if key not in adata.obs.columns:
            print(f"[fetch] WARNING: {label} key {key!r} not in obs — point "
                  f"the SQUINT blob's label_names at the real column name.",
                  file=sys.stderr)

    os.makedirs(out_dir, exist_ok=True)
    samples = list(map(str, adata.obs[skey].astype(str).unique()))
    written = []
    for bidx, s in enumerate(samples):
        sub = adata[adata.obs[skey].astype(str) == s].copy()
        _ensure_spatial(sub)
        # Keep label columns under their ORIGINAL names (registered in the
        # SQUINT blob via label_names). Only add the infra fields the blob
        # builder requires: obs['cell_id'], obs['batch'] (graph batch_key),
        # and the canonical per-section id uns['batch'] (int).
        sub.obs["batch"] = str(bidx)
        if "cell_id" not in sub.obs.columns:
            sub.obs["cell_id"] = sub.obs_names.astype(str)
        sub.uns["batch"] = int(bidx)
        sub.uns["squint_source"] = "HCATonsilData_Spatial_Visium"
        sub.uns["sample_id"] = str(s)
        fp = os.path.join(out_dir, f"{_sanitize(s)}.h5ad")
        sub.write_h5ad(fp)
        nct = (sub.obs[celltype_key].nunique()
               if celltype_key in sub.obs else "n/a")
        nni = (sub.obs[niche_key].nunique()
               if niche_key in sub.obs else "n/a")
        print(f"[fetch]   wrote {fp}  ({sub.n_obs} spots; "
              f"cell-type col={celltype_key!r} ({nct}); "
              f"niche col={niche_key!r} ({nni}))")
        written.append(fp)
    print(f"[fetch] DONE: {len(written)} per-sample files in {out_dir}")
    return written


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir",
                   default="/nfs/team361/sb75/DATASETS/silver/squint_vht")
    p.add_argument("--combined-h5ad", default=None,
                   help="Intermediate combined .h5ad path "
                        "(default: <out-dir>/_tonsil_spatial_combined.h5ad).")
    p.add_argument("--rscript", default="Rscript",
                   help="Path to the Rscript executable.")
    p.add_argument("--install-deps", action="store_true",
                   help="Install the required R/Bioconductor packages first.")
    p.add_argument("--r-libs", default=None,
                   help="Writable R library dir to install into / load from "
                        "(the system R lib is usually read-only). On the farm, "
                        "use ample NFS space, e.g. "
                        "/nfs/team361/sb75/.R/library, to avoid HOME quota "
                        "limits. Defaults to $R_LIBS_USER or ~/R/<...>-library.")
    p.add_argument("--sample-key", default=None,
                   help="obs column identifying the Visium sample/section "
                        "(default: auto-detect).")
    p.add_argument("--celltype-key", default=_CELLTYPE_DEFAULT,
                   help="obs column with the cell-type label (kept as-is; "
                        "register it in the SQUINT blob via label_names).")
    p.add_argument("--niche-key", default=_NICHE_DEFAULT,
                   help="obs column with the manual niche/region label "
                        "(kept as-is; register via label_names).")
    p.add_argument("--force", action="store_true",
                   help="Re-run the R fetch even if the combined .h5ad "
                        "already exists.")
    p.add_argument("--keep-combined", action="store_true",
                   help="Keep the intermediate combined .h5ad after split.")
    args = p.parse_args()

    combined = args.combined_h5ad or os.path.join(
        args.out_dir, "_tonsil_spatial_combined.h5ad")

    if args.force or not os.path.isfile(combined):
        _run_r_export(combined, args.rscript, args.install_deps, args.r_libs)
    else:
        print(f"[fetch] reusing existing combined .h5ad: {combined} "
              f"(use --force to refetch)")

    split_per_sample(
        combined_h5ad=combined,
        out_dir=args.out_dir,
        sample_key=args.sample_key,
        celltype_key=args.celltype_key,
        niche_key=args.niche_key,
    )

    if not args.keep_combined and os.path.isfile(combined):
        os.unlink(combined)
        print(f"[fetch] removed intermediate {combined} "
              f"(use --keep-combined to retain)")


if __name__ == "__main__":
    main()
