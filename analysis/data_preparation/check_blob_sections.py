#!/usr/bin/env python3
"""
Report how many SECTIONS a built SQUINT blob has, without rebuilding it.

The blob maps each silver `*.h5ad` to one section keyed by `uns['batch']`
(`_derive_adata_batch_id`). Two files that share the same `uns['batch']`
collapse to a single `adata_batch_id` — which silently makes the run effectively
single-batch, so the batch-integration metrics (iLISI/MMD/ASW) are skipped. This
checks both:

  1. SILVER (anndata only): every silver/<dataset>/*.h5ad, its n_obs and
     uns['batch'] — so you can see the file count and whether the batch ids are
     DISTINCT (this is what determines the number of sections).
  2. BLOB (torch_geometric; best-effort): loads the built blob with
     overwrite=False (NO rebuild) and prints len(blob) + each section's
     adata_batch_id + n_cells.

Usage:
    python check_blob_sections.py --dataset spatch_coad_1p
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def silver_check(silver_dir):
    import anndata
    files = sorted(glob.glob(os.path.join(silver_dir, "**", "*.h5ad"), recursive=True))
    print(f"\n=== SILVER: {silver_dir} ===")
    if not files:
        print("  no *.h5ad files found")
        return
    print(f"  {len(files)} file(s):")
    batch_ids = []
    for f in files:
        try:
            a = anndata.read_h5ad(f, backed="r")
        except Exception:
            a = anndata.read_h5ad(f)
        b = a.uns.get("batch", "<MISSING>")
        batch_ids.append(str(b))
        has_abid = "adata_batch_id" in a.obs.columns
        print(f"    {os.path.basename(f):40s}  n_obs={a.n_obs:>7}  "
              f"uns['batch']={b!r}  obs has adata_batch_id={has_abid}")
    distinct = sorted(set(batch_ids))
    print(f"  distinct uns['batch'] across files: {distinct}")
    if len(distinct) < len(files):
        print("  !! WARNING: fewer distinct uns['batch'] than files — sections "
              "will COLLAPSE (a shared batch id => one section => no batch "
              "integration metrics). Stamp distinct uns['batch'] per file.")
    elif "<MISSING>" in distinct:
        print("  !! WARNING: uns['batch'] missing on some files — the blob's "
              "_derive_adata_batch_id will HARD-ERROR. Stamp uns['batch'].")
    else:
        print(f"  -> {len(distinct)} distinct section(s). Batch-integration "
              f"metrics need >=2 (and MMD needs EXACTLY 2).")


def blob_check(dataset, data_dir):
    print(f"\n=== BLOB (load, no rebuild): gold/.../{dataset} ===")
    try:
        from vqniche.dataset.in_memory_dataset_blob import InMemoryDatasetBlob
    except Exception as e:
        print(f"  (skipped — could not import InMemoryDatasetBlob: {e})")
        return
    blob = InMemoryDatasetBlob(
        name=dataset,
        feature_names=["cell_gene_counts"],
        label_names=[],
        graph_kwargs=dict(coord_type="generic", spatial_key="spatial",
                          n_neighs_list=[8, 16, 24], radius_list=None,
                          include_self_loop=True, batch_key="batch",
                          k={"lm_eigvecs": 128}),
        data_directory_path=data_dir,
        pre_transform=None, pre_filter=None,
        overwrite=False,            # LOAD existing data.pt, do not reprocess
        software_paths={"deepwalk": "", "gosh": ""},
    )
    ids = []
    for d in blob:
        bid = int(d.adata_batch_id) if hasattr(d, "adata_batch_id") else None
        ids.append(bid)
        n = getattr(d, "num_nodes", None)
        print(f"    section adata_batch_id={bid}  n_cells={n}")
    print(f"  len(blob) = {len(blob)} graph(s); distinct adata_batch_id = "
          f"{sorted(set(i for i in ids if i is not None))}")
    if len(set(ids)) < 2:
        print("  -> SINGLE effective section => iLISI/MMD/ASW are undefined "
              "(skipped). This is why no integration metrics were produced.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="e.g. spatch_coad_1p")
    p.add_argument("--data-dir", default="/nfs/team361/sb75/DATASETS")
    p.add_argument("--skip-blob", action="store_true",
                   help="Only do the silver check (no torch_geometric needed).")
    args = p.parse_args(argv)

    silver_check(os.path.join(args.data_dir, "silver", args.dataset))
    if not args.skip_blob:
        blob_check(args.dataset, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
