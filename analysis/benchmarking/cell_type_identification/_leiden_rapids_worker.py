#!/usr/bin/env python3
"""
GPU Leiden binary-search worker. Invoked as a subprocess by the
shared `_run_leiden_binary_search_rapids_subprocess` helper in
`run_pca_leiden.py` when `SQUINT_LEIDEN_RAPIDS_ENV_SETUP` is set.

The parent (running in any baseline's normal venv) writes
`embedding.npy` and `config.json` to a tempdir, then invokes:

    bash -lc "<env-setup-cmd> && python _leiden_rapids_worker.py --workdir <dir>"

inside the rapids-singlecell env (typically activated via the user's
chain `source /etc/profile.d/modules.sh && module load cellgen/conda
&& conda activate /nfs/.../rapids-singlecell`). This script:

  1. Loads the embedding from `embedding.npy`.
  2. Wraps it in a minimal AnnData (the dense X is unused — rapids
     reads only obsm['X_emb']).
  3. Runs `rapids_singlecell.pp.neighbors` + a binary search over
     Leiden resolutions to land at the target cluster count.
  4. Writes `clusters.npy` (best-found per-cell cluster labels)
     and `result.json` (`n_found`, `resolution`) back to the workdir.

The parent reads those two files, assigns the clusters to
`adata.obs['leiden']`, runs scanpy's `sc.pp.neighbors` on its own
adata so `adata.uns['neighbors']` is populated for downstream UMAP,
and continues.

Why a subprocess: rapids-singlecell pulls in cupy / cuML / cuGraph
which conflict with the torch + scanpy stacks the existing baseline
venvs ship. Isolating it as a subprocess means each baseline keeps
its own venv (no rebuilds needed) and only Leiden runs in the rapids
env. When rapids-singlecell IS importable in the baseline's venv,
prefer the inline `_run_leiden_binary_search_rapids` path instead —
faster (no subprocess overhead, no tempdir IO).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=str, required=True,
                   help="Tempdir written by the parent. Must contain "
                        "embedding.npy and config.json; clusters.npy "
                        "and result.json are written into it.")
    args = p.parse_args()
    work = Path(args.workdir)
    if not work.is_dir():
        raise SystemExit(f"workdir not found: {work}")

    # Imports INSIDE main() so any failure (e.g. rapids env not active)
    # surfaces cleanly in the parent's captured stderr.
    import numpy as np
    import anndata as ad

    try:
        import rapids_singlecell as rsc
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Failed to `import rapids_singlecell`. The env activated "
            "by SQUINT_LEIDEN_RAPIDS_ENV_SETUP must have it installed.\n"
            f"Original error: {exc!r}"
        )

    cfg = json.loads((work / "config.json").read_text())
    emb = np.load(work / "embedding.npy")
    if emb.ndim != 2:
        raise SystemExit(
            f"embedding.npy must be 2-D (cells × dims); got shape {emb.shape}"
        )
    n_cells, n_dims = emb.shape
    n_neighbors = int(cfg["n_neighbors"])
    max_iters   = int(cfg["max_iters"])
    rng_seed    = int(cfg["rng_seed"])
    target_n    = int(cfg["n_clusters"])

    print(
        f"[rapids worker] n_cells={n_cells:,d}, n_dims={n_dims}, "
        f"target_n={target_n}, n_neighbors={n_neighbors}, "
        f"max_iters={max_iters}, rng_seed={rng_seed}",
        flush=True,
    )

    # Minimal AnnData: rapids' neighbors / leiden only need obsm and
    # a populated `n_obs`. X just has to exist and have the right
    # n_obs; a (n_obs, 1) float32 zero matrix is the cheapest legal
    # placeholder.
    adata = ad.AnnData(X=np.zeros((n_cells, 1), dtype=np.float32))
    adata.obsm["X_emb"] = emb.astype(np.float32)

    # Per-phase timings — the parent reads these from result.json so
    # the runtime CSV can attribute time correctly. UMAP time is
    # excluded from `leiden_seconds` (the per-seed clustering cost);
    # it's reported in its own `umap_seconds` column for transparency
    # but not added to the headline runtime metric.
    _t_neighbors_start = time.time()
    rsc.pp.neighbors(
        adata, n_neighbors=n_neighbors, use_rep="X_emb",
        random_state=rng_seed,
    )
    neighbors_seconds = time.time() - _t_neighbors_start

    # Run UMAP on GPU right after neighbors. The parent will load the
    # resulting `X_umap` back onto its main adata so its downstream
    # `sc.tl.umap` call (via `_compute_umap_if_needed`) is a no-op —
    # avoiding a wasteful CPU recomputation that would clobber this
    # rapids-built embedding. rsc.tl.umap accepts the same kwargs as
    # scanpy's tl.umap; reading from the neighbors graph it just built.
    print(
        f"[rapids worker] running rsc.tl.umap on the GPU kNN graph "
        f"(random_state={rng_seed}) ...",
        flush=True,
    )
    _t_umap_start = time.time()
    rsc.tl.umap(adata, random_state=rng_seed)
    umap_seconds = time.time() - _t_umap_start
    print(
        f"[rapids worker] UMAP done; obsm['X_umap'] shape = "
        f"{adata.obsm['X_umap'].shape}  ({umap_seconds:.1f}s)",
        flush=True,
    )

    _t_leiden_start = time.time()
    iter_seconds: list = []
    lo, hi = 0.05, 10.0
    leiden_key = "leiden"
    best: dict = {
        "diff": None, "n_found": None,
        "resolution": None, "labels": None,
    }
    print(
        f"[rapids worker] bisecting Leiden resolution to hit {target_n} "
        f"clusters (initial range {lo}-{hi}, max {max_iters} iters):",
        flush=True,
    )
    for it in range(max_iters):
        mid = 0.5 * (lo + hi)
        _t_iter = time.time()
        rsc.tl.leiden(
            adata, resolution=mid, key_added=leiden_key,
            random_state=rng_seed,
        )
        iter_seconds.append(time.time() - _t_iter)
        labels = adata.obs[leiden_key].astype(str).to_numpy()
        n_found = int(np.unique(labels).size)
        diff = abs(n_found - target_n)
        print(
            f"  iter {it+1:>2d}  resolution={mid:.4f}  -> "
            f"n_clusters={n_found}  ({iter_seconds[-1]:.2f}s)",
            flush=True,
        )
        if best["diff"] is None or diff < best["diff"]:
            best.update({
                "diff": diff, "n_found": n_found,
                "resolution": mid, "labels": labels.copy(),
            })
        if n_found == target_n:
            break
        if n_found < target_n:
            lo = mid
        else:
            hi = mid

    leiden_seconds = time.time() - _t_leiden_start
    leiden_one_iter_seconds = (
        (sum(iter_seconds) / len(iter_seconds)) if iter_seconds else 0.0
    )
    if best["diff"] is None or best["labels"] is None:
        raise SystemExit("rapids Leiden binary search produced no result.")
    if best["diff"] != 0:
        print(
            f"  ! exact match not reached; using closest "
            f"(n={best['n_found']}, resolution={best['resolution']:.4f}).",
            flush=True,
        )

    np.save(work / "clusters.npy", best["labels"])

    # ---- Serialise the UMAP coords back to the parent ------------------
    # rsc.tl.umap writes adata.obsm['X_umap']. We just numpy-dump it;
    # the parent assigns it to its main adata's obsm so its downstream
    # `_compute_umap_if_needed` call detects it and skips the CPU
    # recomputation.
    if "X_umap" in adata.obsm:
        np.save(
            work / "umap.npy",
            np.asarray(adata.obsm["X_umap"], dtype=np.float32),
        )
        print(
            f"[rapids worker] saved umap.npy "
            f"(shape={adata.obsm['X_umap'].shape})",
            flush=True,
        )

    # ---- Serialise the kNN graph back to the parent --------------------
    # rapids-singlecell's pp.neighbors writes the same obsp /
    # uns['neighbors'] keys as scanpy. Dumping them lets the parent
    # PASTE the rapids GPU-built graph onto its main adata instead of
    # recomputing on CPU. The (large) win is on big datasets where
    # CPU sc.pp.neighbors is the new bottleneck after rapids
    # eliminates the Leiden iterations cost.
    from scipy.sparse import csr_matrix, save_npz

    def _to_scipy_csr(m):
        """Coerce rapids' obsp matrix to scipy.sparse.csr_matrix.
        rapids-singlecell typically writes scipy sparse already, but
        some configurations leave a cupy-sparse on obsp. cupy-sparse
        exposes `.get()` to materialise a scipy copy."""
        if m is None:
            return None
        if hasattr(m, "get") and not hasattr(m, "indptr"):
            # cupy-sparse (no scipy-style attrs yet) -> scipy copy
            m = m.get()
        return csr_matrix(m)

    conn = _to_scipy_csr(adata.obsp.get("connectivities"))
    dist = _to_scipy_csr(adata.obsp.get("distances"))
    if conn is not None:
        save_npz(work / "connectivities.npz", conn)
    if dist is not None:
        save_npz(work / "distances.npz", dist)
    print(
        f"[rapids worker] saved kNN graph: "
        f"connectivities={'nnz=' + str(conn.nnz) if conn is not None else 'MISSING'}, "
        f"distances={'nnz=' + str(dist.nnz) if dist is not None else 'MISSING'}",
        flush=True,
    )

    # The uns['neighbors'] dict carries the kNN metadata scanpy
    # downstream tools (sc.tl.umap, sc.tl.leiden again, etc.) read
    # to find the graph. Best-effort JSON serialisation: walk the
    # dict and stringify anything non-trivial.
    def _to_jsonable(v):
        import numpy as _np
        if isinstance(v, (str, bool, type(None))):
            return v
        if isinstance(v, (int, _np.integer)):
            return int(v)
        if isinstance(v, (float, _np.floating)):
            return float(v)
        if isinstance(v, dict):
            return {k: _to_jsonable(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(x) for x in v]
        if hasattr(v, "tolist"):
            return _to_jsonable(v.tolist())
        return str(v)

    neighbors_uns = _to_jsonable(adata.uns.get("neighbors", {}) or {})
    # Guarantee the standard scanpy schema: connectivities_key /
    # distances_key point at the obsp slots the parent will populate.
    if isinstance(neighbors_uns, dict):
        neighbors_uns.setdefault("connectivities_key", "connectivities")
        neighbors_uns.setdefault("distances_key",      "distances")
    (work / "neighbors_uns.json").write_text(json.dumps(neighbors_uns))

    (work / "result.json").write_text(json.dumps({
        "n_found":    int(best["n_found"]),
        "resolution": float(best["resolution"]),
        "leiden_key": leiden_key,
        # Per-phase wall times (seconds). Parent reads these and
        # records them into the runtime CSVs.
        #   - neighbors_seconds  : just rsc.pp.neighbors
        #   - leiden_seconds     : binary search loop wall time (sum
        #                          of all iterations)
        #   - leiden_one_iter_seconds : mean cost of ONE Leiden
        #                          iteration (= leiden_seconds /
        #                          n_iters_run). Used to compute the
        #                          "model + neighbors + ONE leiden"
        #                          runtime flavour.
        #   - umap_seconds       : rsc.tl.umap; visualisation, NOT in
        #                          the headline runtime_seconds.
        "neighbors_seconds":        float(neighbors_seconds),
        "leiden_seconds":           float(leiden_seconds),
        "leiden_one_iter_seconds":  float(leiden_one_iter_seconds),
        "umap_seconds":             float(umap_seconds),
    }))
    print(
        f"[rapids worker] wrote clusters.npy + connectivities.npz + "
        f"distances.npz + neighbors_uns.json + result.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
