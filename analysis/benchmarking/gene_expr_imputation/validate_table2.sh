#!/usr/bin/env bash
# =============================================================================
# Table 2 (tab:recon_imp) validation -- READ-ONLY w.r.t. every existing artifact.
# Writes ONLY under $OUTROOT/<timestamp>/ (validation.log + CSVs).
#
# Checks:
#   A  provenance: re-score SQUINT's stored predicted_adata with the baselines'
#      harness (_holdout_utils.build_pearson_dataframe) == published row?
#   B  exact reproduction of SQUINT's predict pass from stored codes/embeddings
#      (validates node order, covariate mapping, checkpoint, predict batch size)
#   C  decoder LayerNorm batch-composition dependence (PyG LayerNorm mode='graph'):
#      re-decode in training-like batches (512 seeds + full 1-hop) / one batch /
#      per section / held-out-only (stage-2-like) and re-score
#   D  neighbourhood target graph: SQUINT symmetrised kNN16 vs baselines'
#      directed kNN16+self; re-score SQUINT's niche level on the baselines' graph
#   E  quantisation oracle (code-mean profile x own library) + code usage
#   F  baseline checks: identical 6,052 test cells / identical X, graph
#      symmetry, NicheCompass neighbourhood scale + gene coverage, harness parity
#   G  boundary/leak share, best-vs-last checkpoints, stage-2 npz coverage,
#      train-vs-test gaps, decoder-width siblings, full-data discreteness runs
#
# Run on a CPU (or GPU) node, e.g.
#   bsub -q normal -n 8 -M 32000 -R "select[mem>32000] rusage[mem=32000]" \
#        -o table2_validation.%J.log bash table2_validation.sh
# Runtime: ~30-60 min on 8 CPU cores (most of it in the 5x full-forward passes).
# =============================================================================
set -euo pipefail

VENV=/nfs/team361/sb75/.venvs/squint
REPRO=/nfs/team361/sb75/squint-reproducibility
SQUINT=/nfs/team361/sb75/squint
ART=/nfs/team361/sb75/squint-reproducibility/artifacts
DATASET=mmb0-1b_smb1-1b_1p
H="dualvq+rvq-both+decoder-cov+no-batch-int+enc-deeper+dec-w32+knn16+sampler16+cell-w1+bs512+lr7e-4+within-sec+decoupled-enc+diversity-w10+filmscale+crossmnn-wt10-k1"
VARIANT="${H}+region-holdout+${DATASET}"
SEED_RUNS="20260629_023326_seed0 20260629_024131_seed1 20260629_025158_seed2 20260629_025456_seed3 20260629_025518_seed4"
MULTISEED_TS=20260628_230209
BASELINE_TAGS="baseline-scvi+region-holdout+coord-aligned baseline-scvi-nbr+region-holdout+coord-aligned baseline-nichecompass+region-holdout+coord-aligned vanilla-vq-cell+region-holdout+coord-aligned vanilla-vq-nbr+region-holdout+coord-aligned gest-imputed+region-holdout+coord-aligned baseline-knn-spatial+region-holdout+coord-aligned"
SIBLING_VARIANTS="${H/dec-w32/dec-w256}+region-holdout+${DATASET} ${H/dec-w32/dec-w400x2}+region-holdout+${DATASET}"
FULLDATA_VARIANTS="s57_v19_reference-filmscale+${DATASET} s57_v33_continuous-ref-filmscale+${DATASET} s57_v30_vq1level-k2700+${DATASET} s57_v31_vq1level-k120+${DATASET} s57_v32_vq1level-k30+${DATASET} s57_v28_continuous-latent+${DATASET} s57_v29_discrete-vq-ref+${DATASET}"
STAGE2_NPZ_DIR="${ART}/${DATASET}/stage2-ablation/decode-temp05-nonoise"
LOGDIR="${ART}/logs/${DATASET}/imputation-recon"
NBR_K=16
TCI_BATCH=512
NTHREADS=8
OUTROOT=/nfs/team361/sb75/squint-reproducibility/artifacts/table2_validation
OUT="${OUTROOT}/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${SQUINT}/src:${REPRO}/analysis/benchmarking/gene_expr_imputation:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS OPENBLAS_NUM_THREADS=$NTHREADS
export ART DATASET VARIANT SEED_RUNS MULTISEED_TS BASELINE_TAGS SIBLING_VARIANTS FULLDATA_VARIANTS \
       STAGE2_NPZ_DIR LOGDIR NBR_K TCI_BATCH NTHREADS OUT

{
  echo "== provenance of the code used by this validation =="
  echo "squint repo : $(git -C "$SQUINT" rev-parse --abbrev-ref HEAD 2>/dev/null || true) $(git -C "$SQUINT" log -1 --format='%h %ci %s' 2>/dev/null || true)"
  echo "repro repo  : $(git -C "$REPRO" rev-parse --abbrev-ref HEAD 2>/dev/null || true) $(git -C "$REPRO" log -1 --format='%h %ci %s' 2>/dev/null || true)"
  echo "output dir  : $OUT"
} | tee "$OUT/validation.log"

python - <<'PY' 2>&1 | tee -a "$OUT/validation.log"
import gc
import glob
import os
import re
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

T0 = time.time()
E = os.environ
ART, DATASET, VARIANT = E["ART"], E["DATASET"], E["VARIANT"]
MS_TS, OUT = E["MULTISEED_TS"], E["OUT"]
SEED_RUNS = E["SEED_RUNS"].split()
BASE_TAGS = E["BASELINE_TAGS"].split()
SIBLINGS = E.get("SIBLING_VARIANTS", "").split()
FULLDATA = E.get("FULLDATA_VARIANTS", "").split()
STAGE2_NPZ_DIR = E["STAGE2_NPZ_DIR"]
LOGDIR = E["LOGDIR"]
NBR_K = int(E["NBR_K"])
TCI_BS = int(E["TCI_BATCH"])
DS = os.path.join(ART, DATASET)
for p in (DS, OUT):
    assert os.path.isdir(p), p
assert os.path.realpath(OUT).startswith(os.path.realpath(os.path.join(ART, "table2_validation"))), \
    "refusing to write outside artifacts/table2_validation"

COLS = ["pe_cw_log", "pe_gw_log", "pe_hv_log", "sp_gw", "rm_cw_raw", "zn_au_all", "zn_ap_all"]
PUB = {  # Table 2 (coord-aligned) 5-seed means, test split
    ("cell", "SQUINT"): [0.634, 0.362, 0.702, 0.275, 0.792, 0.862, 0.500],
    ("cell", "scVI"): [0.661, 0.394, 0.738, 0.293, 0.753, 0.874, 0.535],
    ("cell", "NicheCompass"): [0.709, 0.436, 0.771, 0.311, 0.781, 0.901, 0.604],
    ("cell", "Vanilla"): [0.530, 0.304, 0.597, 0.253, 0.982, 0.844, 0.424],
    ("niche", "SQUINT"): [0.896, 0.615, 0.896, 0.576, 0.303, 0.850, 0.886],
    ("niche", "scVI-nbr"): [0.907, 0.636, 0.911, 0.598, 0.266, 0.864, 0.885],
    ("niche", "NicheCompass"): [0.901, 0.575, 0.906, 0.543, 0.301, 0.841, 0.868],
    ("niche", "Vanilla-nbr"): [0.746, 0.473, 0.665, 0.490, 0.534, 0.826, 0.836],
}
RESULTS = []
KEEP = {}


def log(*a):
    print(*a, flush=True)


def section(t):
    log("\n" + "=" * 100 + f"\n{t}   [t={time.time() - T0:.0f}s]\n" + "=" * 100)


def guarded(fn):
    def _w(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:  # noqa: BLE001
            log(f"!! step {fn.__name__} FAILED:\n" + traceback.format_exc())
            return None
    return _w


def dense(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def get_mat(a, k):
    if k in a.layers:
        return dense(a.layers[k])
    if k in a.uns:
        v = dense(a.uns[k])
        if v is not None and v.shape == tuple(a.shape):
            return v
    return None


def norm_sec(v):
    out = []
    for x in np.asarray(v):
        try:
            out.append(str(int(float(x))))
        except (TypeError, ValueError):
            out.append(str(x))
    return np.asarray(out)


from _holdout_utils import build_pearson_dataframe, _neighbor_mean, _pearson_pairwise  # noqa: E402


def extract(df, branch, split="test"):
    d = df[(df["branch"] == branch) & (df["split"] == split)]

    def g(axis, tr, gs, col):
        if col not in d.columns:
            return float("nan")
        m = (d["axis"] == axis) & (d["transform"] == tr)
        m = m & (d["gene_subset"].astype(str).str.startswith("hvg") if gs == "hvg" else d["gene_subset"] == gs)
        s = d.loc[m, col].dropna()
        return float(s.iloc[0]) if len(s) else float("nan")

    r = {
        "pe_cw_log": g("cell_wise", "log1p", "all", "pearson_mean"),
        "pe_gw_log": g("gene_wise", "log1p", "all", "pearson_mean"),
        "pe_hv_log": g("gene_wise", "log1p", "hvg", "pearson_mean"),
        "sp_gw": g("gene_wise", "log1p", "all", "spearman_mean"),
        "rm_cw_raw": g("cell_wise", "raw", "all", "rmse_mean"),
        "zn_au_all": g("entrywise", "counts", "all", "auroc_zero"),
        "zn_ap_all": g("entrywise", "counts", "all", "auprc_zero"),
        "pe_cw_raw": g("cell_wise", "raw", "all", "pearson_mean"),
    }
    nc = d["n_cells"].dropna().unique() if "n_cells" in d.columns else []
    r["n_cells"] = int(nc[0]) if len(nc) else -1
    return r


def score_test(X, mask, X_hat=None, X_nbr=None, X_hat_nbr=None):
    """Same harness as the baselines, restricted to the held-out rows (all 7
    Table-2 columns depend only on the scored rows: HVGs, ranks, pooled
    AUROC/AP are computed within the split)."""
    idx = np.where(mask)[0]
    obs = pd.DataFrame({"data_split": pd.Categorical(["test"] * idx.size, categories=["train", "test"])},
                       index=[str(i) for i in idx])
    a = ad.AnnData(X=np.asarray(X[idx], np.float32), obs=obs)
    if X_hat is not None:
        a.layers["X_hat"] = np.asarray(X_hat[idx], np.float32)
    if X_hat_nbr is not None:
        a.layers["X_hat_nbr"] = np.asarray(X_hat_nbr[idx], np.float32)
        a.layers["X_nbr"] = np.asarray(X_nbr[idx], np.float32)
    df = build_pearson_dataframe(a, seed=0, log1p=True, n_hvg=50, verbose=False, cell_type_key=None)
    res = {}
    if X_hat is not None:
        res["cell"] = extract(df, "cell")
    if X_hat_nbr is not None:
        res["niche"] = extract(df, "niche")
    return res


def fmt(m):
    return " ".join(f"{c}={m[c]:.4f}" for c in COLS)


def record(seed, variant, res):
    for br, m in res.items():
        RESULTS.append(dict(seed=seed, variant=variant, branch=br, **m))
        log(f"    [{variant:<40s}] {br:<5s} {fmt(m)}  n={m.get('n_cells')}")


def directed_knn(coords, sec, k):
    """Baselines' recipe (squidpy generic kNN, set_diag=True, per section, NOT
    symmetrised), built index-safely (no contiguity assumption)."""
    try:
        import squidpy as sq
    except ImportError:
        sq = None
        log("  (squidpy missing: using sklearn kneighbors(), which is what squidpy generic kNN calls)")
    n = coords.shape[0]
    rows, cols = [], []
    for b in np.unique(sec):
        idx = np.where(sec == b)[0]
        if sq is not None:
            sub = ad.AnnData(X=np.zeros((idx.size, 1), np.float32))
            sub.obsm["spatial"] = np.asarray(coords[idx], np.float64)
            sq.gr.spatial_neighbors(sub, coord_type="generic", spatial_key="spatial", n_neighs=k, set_diag=True)
            C = sub.obsp["spatial_connectivities"].tocoo()
            rows.append(idx[C.row])
            cols.append(idx[C.col])
        else:
            from sklearn.neighbors import NearestNeighbors
            _, nb = NearestNeighbors(n_neighbors=k).fit(coords[idx]).kneighbors()
            rows.append(idx[np.repeat(np.arange(idx.size), k)])
            cols.append(idx[nb.ravel()])
            rows.append(idx)
            cols.append(idx)
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix((np.ones(r.size, np.float32), (r, c)), shape=(n, n))
    A.data[:] = 1.0
    return A


def find_ckpts(rd):
    cdir = os.path.join(rd, "checkpoints")
    if not os.path.isdir(cdir):
        cdir = os.path.join(rd, "files", "checkpoints")
    files = sorted(glob.glob(os.path.join(cdir, "*.ckpt")))
    last = [f for f in files if "last" in os.path.basename(f).lower()]
    best, bv = None, np.inf
    for f in files:   # mirrors squint.utils.parse_test_configs.find_best_checkpoint (mode=min, val_loss)
        try:
            v = float(os.path.basename(f).split("val_loss=")[1].split(".ckpt")[0])
        except (IndexError, ValueError):
            continue
        if v < bv:
            bv, best = v, f
    return best, (last[0] if last else None), files


def torch_load(p):
    import torch
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def predict_bs(rd):
    for fn in ("user_specified_config.yaml", "config.yaml"):
        p = os.path.join(rd, fn)
        if os.path.isfile(p):
            m = re.search(r"predict_batch_size:\s*(\d+)", open(p).read())
            if m:
                return int(m.group(1)), p
    return None, None


# ----------------------------------------------------------------------------- torch / squint
import torch  # noqa: E402
torch.set_num_threads(int(E.get("NTHREADS", "8")))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import torch_geometric  # noqa: E402
import squint  # noqa: E402,F401
from squint.models import Squint_Dual  # noqa: E402
from squint.utils.loss_utils import aggregate_1hop_neighbor_features  # noqa: E402
for _n, _m in list(sys.modules.items()):   # old checkpoints may pickle 'vqniche.*' refs (pre-rename)
    if _n == "squint" or _n.startswith("squint."):
        sys.modules.setdefault("vqniche" + _n[len("squint"):], _m)
log(f"torch {torch.__version__} | torch_geometric {torch_geometric.__version__} | device {DEV}")

MS = None
ms_csv = os.path.join(DS, VARIANT + "__multiseed", MS_TS, "metrics", "per_seed_pearson_reconstruction.csv")
if os.path.isfile(ms_csv):
    MS = pd.read_csv(ms_csv)
    log(f"multiseed CSV: {ms_csv}\n  columns: {list(MS.columns)}\n  seeds: {sorted(MS['seed'].unique().tolist())}")
    for br in ("cell", "niche"):
        per = [extract(MS[MS["seed"] == s], br) for s in sorted(MS["seed"].unique())]
        mean = {c: float(np.nanmean([p[c] for p in per])) for c in COLS}
        log(f"  multiseed CSV {br:<5s} 5-seed mean: {fmt(mean)} | n_cells per seed {[p['n_cells'] for p in per]}")
        log(f"  published   {br:<5s}              : " + " ".join(f"{c}={v:.3f}" for c, v in zip(COLS, PUB[(br, 'SQUINT')])))
else:
    log(f"!! multiseed CSV not found: {ms_csv}")


# ============================================================================= SQUINT per seed
@guarded
def squint_seed(si, run):
    rd = os.path.join(DS, VARIANT, run)
    section(f"[SQUINT seed {si}] {rd}")
    A = ad.read_h5ad(os.path.join(rd, "predicted_adata.h5ad"))
    N, G = A.n_obs, A.n_vars
    X = dense(A.X)
    lib = X.sum(1)
    test = A.obs["data_split"].astype(str).to_numpy() == "test"
    train = ~test
    sec_int = A.obs["adata_batch_id"].astype(int).to_numpy()
    uniq = np.array(sorted(np.unique(sec_int)))
    sec_dense = np.searchsorted(uniq, sec_int)          # = stage2_decode_pearson._cov_index
    coords = np.asarray(A.obsm["spatial"], np.float64)[:, :2]
    Xh_st, Xhn_st, Xn_st = get_mat(A, "X_hat"), get_mat(A, "X_hat_nbr"), get_mat(A, "X_nbr")
    codes_c = np.asarray(A.obsm["cell_code_indices"]).astype(np.int64).reshape(N, -1)
    codes_n = np.asarray(A.obsm["neighborhood_code_indices"]).astype(np.int64).reshape(N, -1)
    cell_emb = np.asarray(A.obsm["cell_emb"], np.float32)
    nb_emb = np.asarray(A.obsm["neighborhood_emb"], np.float32)
    log(f"  n_obs={N} n_vars={G} test={int(test.sum())}; per section n="
        f"{ {int(s): int((sec_int == s).sum()) for s in uniq} } test="
        f"{ {int(s): int((test & (sec_int == s)).sum()) for s in uniq} }")
    chg = np.where(np.diff(sec_int) != 0)[0]
    log(f"  row order: {len(chg) + 1} contiguous section block(s); first section {sec_int[0]}; change rows {chg[:5].tolist()}")
    rel = np.abs(Xh_st.sum(1) - lib) / np.maximum(lib, 1)
    log(f"  stored X_hat row sums vs X.sum: max rel diff {rel.max():.2e} (expect < 1e-4: same rows, own library)")

    # ---- A. provenance: same harness as the baselines ----------------------
    V0 = score_test(X, test, Xh_st, Xn_st, Xhn_st)
    record(si, "V0_stored_published", V0)
    if MS is not None:
        for br in V0:
            ref = extract(MS[MS["seed"] == si], br)
            dmax = max(abs(V0[br][c] - ref[c]) for c in COLS)
            log(f"  [A] provenance {br}: max |re-score - multiseed CSV| = {dmax:.2e} (expect < 1e-5); CSV n_cells={ref['n_cells']}")

    # ---- checkpoints --------------------------------------------------------
    best, last, files = find_ckpts(rd)
    log(f"  checkpoints: {[os.path.basename(f) for f in files]}\n  best(val_loss)={best and os.path.basename(best)} last={last and os.path.basename(last)}")
    if best and last:
        try:
            sa, sb = torch_load(best), torch_load(last)
            da, db = sa["state_dict"], sb["state_dict"]
            mx = max(float((da[k].float() - db[k].float()).abs().max()) for k in da
                     if k in db and da[k].shape == db[k].shape and da[k].is_floating_point())
            log(f"  best vs last: epoch {sa.get('epoch')} vs {sb.get('epoch')}; max |param diff| = {mx:.3e} "
                f"({'IDENTICAL weights' if mx == 0 else 'DIFFERENT weights'})")
            del sa, sb, da, db
        except Exception as e:  # noqa: BLE001
            log(f"  best vs last compare failed: {e}")
    model = Squint_Dual.load_from_checkpoint(best or last, map_location=DEV)
    model.eval().to(DEV)
    dmlp = model.attribute_decoder_cell.mlp_module
    log(f"  [C] cell decoder: widths {[getattr(l, 'out_channels', getattr(l, 'out_features', None)) for l in dmlp.lins]}; "
        f"norms {[(type(n).__module__ + '.' + type(n).__name__, getattr(n, 'mode', None)) for n in dmlp.norms]}; "
        f"supports_norm_batch={getattr(dmlp, 'supports_norm_batch', None)}; decoder_covariate_dim={model.decoder_covariate_dim}; "
        f"decoupled_cov={getattr(model, 'decoupled_decoder_covariate', None)}; ccn_mode={getattr(model.encoder, 'ccn_mode', None)}")
    try:
        vq = model.encoder.vq_cell
        cc = torch.as_tensor(codes_c, device=DEV)
        zq = None
        for lvl in range(cc.shape[1]):
            emb = vq.layers[lvl]._codebook.embed
            emb = emb[0] if emb.dim() == 3 else emb
            zq = emb[cc[:, lvl]] if zq is None else zq + emb[cc[:, lvl]]
        log(f"  codes->z_q vs obsm['cell_emb']: max abs diff {float((zq.cpu() - torch.as_tensor(cell_emb)).abs().max()):.2e}")
    except Exception as e:  # noqa: BLE001
        log(f"  codes->z_q check skipped: {e}")

    # ---- graphs ---------------------------------------------------------------
    if "edge_index" in A.uns:
        EI = np.asarray(A.uns["edge_index"]).astype(np.int64)
        EI = EI if EI.shape[0] == 2 else EI.T
    else:
        raise RuntimeError("uns['edge_index'] missing")
    src, dst = EI[0], EI[1]
    A_sym = sp.csr_matrix((np.ones(src.size, np.float32), (dst, src)), shape=(N, N))
    A_sym.data[:] = 1.0
    deg = np.asarray(A_sym.sum(1)).ravel()
    log(f"  [D] SQUINT graph (uns edge_index): E={src.size} self-loops={int((src == dst).sum())} asym entries={(A_sym != A_sym.T).nnz}; "
        f"closed degree test cells mean {deg[test].mean():.2f} p10/50/90 {np.percentile(deg[test], [10, 50, 90]).tolist()} max {deg[test].max():.0f}; all cells mean {deg.mean():.2f}")
    Xn_rec = _neighbor_mean(A_sym, X)
    log(f"  [B] X_nbr rebuilt from uns edge_index vs stored X_nbr: max abs diff {np.abs(Xn_rec - Xn_st).max():.2e} "
        f"(~0 => node order == predicted_adata row order)")
    A_dir = directed_knn(coords, sec_int, NBR_K)
    rsd = np.asarray(A_dir.sum(1)).ravel()
    log(f"  [D] directed kNN{NBR_K}+self (baseline recipe): row sums {np.unique(rsd)[:6].tolist()}, asym entries {(A_dir != A_dir.T).nnz}; "
        f"in-degree test cells p10/50/90 {np.percentile(np.asarray(A_dir.sum(0)).ravel()[test], [10, 50, 90]).tolist()}")
    Xn_dir = _neighbor_mean(A_dir, X)
    cwt = _pearson_pairwise(np.log1p(Xn_dir[test]), np.log1p(Xn_st[test]), axis=1)
    log(f"  [D] SQUINT X_nbr target vs directed X_nbr target (test, log1p cell-wise r): median {np.nanmedian(cwt):.4f}; "
        f"non-zero fraction sym {float((Xn_st[test] > 0).mean()):.4f} vs dir {float((Xn_dir[test] > 0).mean()):.4f}")
    # boundary / leak share
    T = A_sym[np.where(train)[0]][:, np.where(test)[0]]
    nb_tr = np.asarray(T.sum(1)).ravel()
    T2 = A_sym[np.where(test)[0]][:, np.where(train)[0]]
    nb_te = np.asarray(T2.sum(1)).ravel()
    log(f"  [G] boundary: train cells with >=1 held-out neighbour {(nb_tr > 0).sum()}/{int(train.sum())} ({100 * (nb_tr > 0).mean():.2f}%); "
        f"held-out share of all train-row neighbour slots {T.sum() / A_sym[np.where(train)[0]].sum():.4f}; "
        f"held-out cells with >=1 observed neighbour {(nb_te > 0).sum()}/{int(test.sum())}")

    # ---- subgraph builder = PyG NeighborLoader, 1 hop, fan-out -1, 'directional'
    order = np.argsort(dst, kind="stable")
    src_s, dst_s = src[order], dst[order]
    indptr = np.searchsorted(dst_s, np.arange(N + 1))

    def subgraph(seeds):
        seeds = np.asarray(seeds, np.int64)
        st, en = indptr[seeds], indptr[seeds + 1]
        lens = en - st
        csum = np.cumsum(lens)
        idx = np.repeat(st - (csum - lens), lens) + np.arange(int(csum[-1]))
        nb, tg = src_s[idx], np.repeat(seeds, lens)
        nodes = np.concatenate([seeds, np.setdiff1d(np.unique(nb), seeds)])
        lmap = np.full(N, -1, np.int64)
        lmap[nodes] = np.arange(nodes.size)
        return nodes, np.stack([lmap[nb], lmap[tg]])

    def dec(nodes, which):
        emb = cell_emb if which == "cell" else nb_emb
        with torch.no_grad():
            z = torch.as_tensor(emb[nodes], device=DEV)
            if model.decoder_covariate_dim > 0:
                ids = torch.as_tensor(sec_dense[nodes], dtype=torch.long, device=DEV)
                if getattr(model, "decoupled_decoder_covariate", False):
                    Em = model.batch_embedding_cell if which == "cell" else model.batch_embedding_niche
                else:
                    Em = model.batch_embedding
                z = torch.cat([z, Em(ids)], dim=-1)
            D = model.attribute_decoder_cell if which == "cell" else model.attribute_decoder_niche
            o = D(x=z, read_depth=torch.as_tensor(lib[nodes], device=DEV), conditions=None)
        return o.float().cpu().numpy()

    def cell_batches(batches):
        out = np.zeros((N, G), np.float32)
        comp = []
        for seeds in batches:
            nodes, _ = subgraph(seeds)
            out[seeds] = dec(nodes, "cell")[: len(seeds)]
            comp.append((len(seeds), int(nodes.size), {int(s): int((sec_int[nodes] == s).sum()) for s in uniq}))
        return out, comp

    def full_batches(batches):
        Xh, Xhn, Xn, SM = (np.zeros((N, G), np.float32) for _ in range(4))
        ic = np.zeros_like(codes_c)
        inn = np.zeros_like(codes_n)
        for seeds in batches:
            nodes, ei = subgraph(seeds)
            Bn = len(seeds)
            x = torch.as_tensor(X[nodes], device=DEV)
            eit = torch.as_tensor(ei, dtype=torch.long, device=DEV)
            ids = torch.as_tensor(sec_dense[nodes], dtype=torch.long, device=DEV)
            with torch.no_grad():
                outs = model(batch_x=x, batch_edge_index=eit, read_depth=x.sum(dim=-1), adata_batch_ids=ids)
                idc, idn, xc, xn = outs[4], outs[5], outs[6], outs[7]
                agg = aggregate_1hop_neighbor_features(X=torch.cat([x, xn], dim=-1), edge_index=eit,
                                                       return_mean=True, batch_size=Bn)
            Xh[seeds] = xc[:Bn].float().cpu().numpy()
            SM[seeds] = xn[:Bn].float().cpu().numpy()          # seed-mode (full-neighbourhood) niche output
            Xn[seeds] = agg[:, :G].float().cpu().numpy()
            Xhn[seeds] = agg[:, G:].float().cpu().numpy()
            ic[seeds] = idc[:Bn].reshape(Bn, -1).cpu().numpy()
            inn[seeds] = idn[:Bn].reshape(Bn, -1).cpu().numpy()
        return Xh, Xhn, Xn, SM, ic, inn

    def rowrel(a, b):
        return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-8)

    # ---- B. find the predict batching that reproduces the stored X_hat -------
    B_cfg, cfg_path = predict_bs(rd)
    log(f"  [B] predict_batch_size in config: {B_cfg} ({cfg_path})")
    best_B, best_err = None, np.inf
    for B in dict.fromkeys([b for b in (B_cfg, 65536, 32768, 16384, 8192, 4096) if b]):
        batches = [np.arange(i, min(N, i + B)) for i in range(0, N, B)]
        Xh_try, comp = cell_batches(batches)
        rr = rowrel(Xh_try, Xh_st)
        cw = _pearson_pairwise(Xh_try, Xh_st, axis=1)
        log(f"  [B] cell-decoder replay B={B}: {len(batches)} batch(es) (seeds, nodes, nodes/section) {comp[:4]}; "
            f"row rel-L2 max {rr.max():.2e} median {np.median(rr):.2e}; cell-wise r vs stored median {np.nanmedian(cw):.6f} min {np.nanmin(cw):.6f}")
        if rr.max() < best_err:
            best_err, best_B = float(rr.max()), B
        if rr.max() < 1e-4:
            break
    ok = best_err < 1e-3
    log(f"  [B] -> best predict batch size {best_B} (max row rel-L2 {best_err:.2e}): "
        f"{'EXACT REPLAY of the published X_hat' if ok else 'NOT REPRODUCED (see notes)'}")
    tb = np.where(test)[0] // best_B
    log(f"  [C] held-out cells per predict batch (B={best_B}) x section: "
        f"{ {int(b): {int(s): int(((tb == b) & (sec_int[test] == s)).sum()) for s in uniq} for b in np.unique(tb)} }")

    pred_batches = [np.arange(i, min(N, i + best_B)) for i in range(0, N, best_B)]
    Xh1, Xhn1, Xn1, SM1, ic1, in1 = full_batches(pred_batches)
    log(f"  [B] full-forward replay: cell codes equal {float((ic1 == codes_c).all(1).mean()):.6f}, niche codes equal "
        f"{float((in1 == codes_n).all(1).mean()):.6f}; X_hat rel-L2 max {rowrel(Xh1, Xh_st).max():.2e}; "
        f"X_hat_nbr rel-L2 max {rowrel(Xhn1, Xhn_st).max():.2e}; X_nbr max abs {np.abs(Xn1 - Xn_st).max():.2e}")
    record(si, "V1_replay_predict", score_test(X, test, Xh1, Xn1, Xhn1))

    # ---- C. training-consistent inference: 512 random seeds + full 1-hop -------
    rng = np.random.default_rng(1000 + si)
    perm = rng.permutation(N)
    tci_batches = [np.sort(perm[i:i + TCI_BS]) for i in range(0, N, TCI_BS)]
    Xh2, Xhn2, Xn2, SM2, ic2, in2 = full_batches(tci_batches)
    log(f"  [C] TCI (bs={TCI_BS}): codes equal cell {float((ic2 == codes_c).all(1).mean()):.6f} niche {float((in2 == codes_n).all(1).mean()):.6f}; "
        f"X_hat vs stored: rel-L2 median {np.median(rowrel(Xh2, Xh_st)[test]):.3e} max {rowrel(Xh2, Xh_st)[test].max():.3e}; "
        f"cell-wise r median {np.nanmedian(_pearson_pairwise(Xh2[test], Xh_st[test], axis=1)):.6f}")
    record(si, "V2_TCI_bs512_full1hop", score_test(X, test, Xh2, Xn2, Xhn2))
    all_nodes = np.arange(N)
    Xh3 = np.zeros((N, G), np.float32)
    Xh3[all_nodes] = dec(all_nodes, "cell")
    record(si, "V3_cell_one_batch_all", score_test(X, test, Xh3))
    Xh4 = np.zeros((N, G), np.float32)
    for s in uniq:
        nd = np.where(sec_int == s)[0]
        Xh4[nd] = dec(nd, "cell")
    record(si, "V4_cell_per_section_batch", score_test(X, test, Xh4))
    te = np.where(test)[0]
    Xh5 = np.zeros((N, G), np.float32)
    Xh5[te] = dec(te, "cell")
    record(si, "V5_cell_heldout_only_batch(stage2)", score_test(X, test, Xh5))
    for s in uniq:
        m = test & (sec_int == s)
        r0 = score_test(X, m, Xh_st)["cell"]
        r2 = score_test(X, m, Xh2)["cell"]
        log(f"  [C] section {s} (n={int(m.sum())}): stored pe_cw_log {r0['pe_cw_log']:.4f} -> TCI {r2['pe_cw_log']:.4f}; "
            f"rm_cw_raw {r0['rm_cw_raw']:.4f} -> {r2['rm_cw_raw']:.4f}; pe_gw_log {r0['pe_gw_log']:.4f} -> {r2['pe_gw_log']:.4f}")

    # ---- D. neighbourhood level on the baselines' directed graph --------------
    record(si, "N1_seedmode_SQUINTgraph(pred)", score_test(X, test, None, Xn_st, _neighbor_mean(A_sym, SM1)))
    record(si, "N2_seedmode_DIRgraph(pred)", score_test(X, test, None, Xn_dir, _neighbor_mean(A_dir, SM1)))
    record(si, "N3_seedmode_DIRgraph(TCI)", score_test(X, test, None, Xn_dir, _neighbor_mean(A_dir, SM2)))

    # ---- E. quantisation oracle + code usage ------------------------------------
    P = X / np.maximum(lib, 1e-8)[:, None]
    k2 = sec_dense * 1_000_000 + codes_c[:, 0] * 1000 + codes_c[:, -1]
    k1 = sec_dense * 1_000_000 + codes_c[:, 0] * 1000 + 999

    def table(keys, rows):
        df = pd.DataFrame(P[rows])
        df["k"] = keys[rows]
        return df.groupby("k").mean()

    def oracle(fit_rows, level):
        t2, t1, t0 = table(k2, fit_rows), table(k1, fit_rows), table(sec_dense, fit_rows)
        out = np.zeros((N, G), np.float32)
        r = np.where(test)[0]
        if level == 2:
            m = np.isin(k2[r], t2.index.values)
            out[r[m]] = t2.loc[k2[r[m]]].to_numpy()
            r = r[~m]
        m = np.isin(k1[r], t1.index.values)
        out[r[m]] = t1.loc[k1[r[m]]].to_numpy()
        r = r[~m]
        if r.size:
            out[r] = t0.loc[sec_dense[r]].to_numpy()
        return out * lib[:, None]

    tr_rows = np.where(train)[0]
    seen = np.isin(k2[test], np.unique(k2[train]))
    cnt = pd.Series(k2[train]).value_counts()
    log(f"  [E] cell codes: distinct (section,L1,L2) over all {np.unique(k2).size}, among test {np.unique(k2[test]).size}; "
        f"test cells whose code occurs in train {seen.mean():.4f}; median train cells per code {cnt.median():.0f}")
    record(si, "E1_oracle_codeMean_L1L2_train", score_test(X, test, oracle(tr_rows, 2)))
    record(si, "E2_oracle_codeMean_L1only_train", score_test(X, test, oracle(tr_rows, 1)))
    record(si, "E3_oracle_codeMean_L1L2_allcells", score_test(X, test, oracle(np.arange(N), 2)))

    if si == 0:
        KEEP.update(coords=coords, sec=norm_sec(sec_int), test=test, X=X, Xn_dir=Xn_dir, N=N)
    del model, A, Xh_st, Xhn_st, Xn_st, Xh1, Xhn1, Xn1, SM1, Xh2, Xhn2, Xn2, SM2, Xh3, Xh4, Xh5, P
    gc.collect()


for si, run in enumerate(SEED_RUNS):
    seed = int(run.split("seed")[-1])
    squint_seed(seed, run)


# ============================================================================= baselines
def newest_run(tag):
    c = sorted(glob.glob(os.path.join(DS, tag, "*", "predicted_adata_seed0.h5ad")))
    return os.path.dirname(c[-1]) if c else None


@guarded
def baselines():
    from scipy.spatial import cKDTree
    section("[F] baselines (coord-aligned): cell identity, X identity, graph, harness parity, NicheCompass scale")
    for tag in BASE_TAGS:
        run = newest_run(tag)
        log(f"\n--- {tag}: {run}")
        if run is None:
            log("   (no predicted_adata_seed0.h5ad found)")
            continue
        csvp = os.path.join(run, "metrics", "per_seed_pearson_reconstruction.csv")
        CS = pd.read_csv(csvp) if os.path.isfile(csvp) else None
        if CS is not None:
            tt = CS[CS["split"] == "test"]
            log(f"   per-seed CSV seeds {sorted(CS['seed'].unique().tolist())}; test n_cells values {sorted(tt['n_cells'].dropna().unique().tolist())}")
            for br in ("cell", "niche"):
                if (CS["branch"] == br).any():
                    trn = [extract(CS[CS.seed == s], br, "train")["pe_cw_log"] for s in sorted(CS.seed.unique())]
                    tst = [extract(CS[CS.seed == s], br, "test")["pe_cw_log"] for s in sorted(CS.seed.unique())]
                    log(f"   {br}: pe_cw_log train {np.nanmean(trn):.4f} | test {np.nanmean(tst):.4f}")
        B = ad.read_h5ad(os.path.join(run, "predicted_adata_seed0.h5ad"))
        Xb = dense(B.X)
        testb = B.obs["data_split"].astype(str).to_numpy() == "test"
        secb = norm_sec(B.obs["adata_batch_id"].to_numpy())
        cb = np.asarray(B.obsm["spatial"], np.float64)[:, :2]
        if KEEP:
            cs, ss, ts = KEEP["coords"], KEEP["sec"], KEEP["test"]
            idx = np.full(cs.shape[0], -1)
            dist = np.full(cs.shape[0], np.inf)
            for s in np.unique(ss):
                ia, ib = np.where(ss == s)[0], np.where(secb == s)[0]
                if ib.size == 0:
                    continue
                d, j = cKDTree(cb[ib]).query(cs[ia], k=1)
                dist[ia] = d
                idx[ia[d <= 0.05]] = ib[j[d <= 0.05]]
            mt = idx[ts]
            n_match = int((mt >= 0).sum())
            both = int(((mt >= 0) & testb[np.clip(mt, 0, None)]).sum())
            unmatched_b = int(testb.sum() - np.isin(np.where(testb)[0], mt[mt >= 0]).sum())
            xd = float(np.abs(Xb[mt[mt >= 0]] - KEEP["X"][ts][mt >= 0]).max()) if n_match else float("nan")
            log(f"   test cells: SQUINT {int(ts.sum())}, baseline {int(testb.sum())}; SQUINT-test matched by coords {n_match}, "
                f"of which baseline-test {both}; baseline-test not matched {unmatched_b}; max |X diff| on matched {xd:.3g}; "
                f"max coord dist {np.nanmax(dist[ts]):.3g}  (expect 6052/6052/6052/0/0)")
            if "X_nbr" in B.layers and n_match:
                xn = dense(B.layers["X_nbr"])
                log(f"   baseline X_nbr vs directed-kNN X_nbr rebuilt on SQUINT coords (test): max abs diff "
                    f"{float(np.abs(xn[mt[mt >= 0]] - KEEP['Xn_dir'][ts][mt >= 0]).max()):.3g} (expect ~0)")
        if "spatial_connectivities" in B.obsp:
            Ab = B.obsp["spatial_connectivities"].tocsr()
            log(f"   obsp graph: row sums {np.unique(np.asarray(Ab.sum(1)).ravel())[:6].tolist()}, asym entries {(Ab != Ab.T).nnz}")
        Xh = get_mat(B, "X_hat")
        Xhn = get_mat(B, "X_hat_nbr")
        Xn = get_mat(B, "X_nbr")
        res = score_test(Xb, testb, Xh, Xn if Xhn is not None else None, Xhn)
        if CS is not None:
            for br, m in res.items():
                ref = extract(CS[CS["seed"] == 0], br)
                log(f"   harness parity seed0 {br}: max |my score - runner CSV| = {max(abs(m[c] - ref[c]) for c in COLS):.2e}")
        for nm, M in (("X_hat", Xh), ("X_hat_nbr", Xhn)):
            if M is not None:
                v = M[testb].var(0)
                log(f"   {nm}: zero-variance genes on test {int((v == 0).sum())}/{M.shape[1]}")
        if Xhn is not None and Xn is not None:
            r = Xhn[testb].sum(1) / np.maximum(Xn[testb].sum(1), 1e-8)
            log(f"   X_hat_nbr / X_nbr row-sum ratio (test): mean {r.mean():.4f} CV {r.std() / r.mean():.4f} p5/p95 {np.percentile(r, [5, 95]).tolist()}")
        del B
        gc.collect()

    # NicheCompass, all seeds: in- vs out-neighbourhood scale artefact
    run = newest_run("baseline-nichecompass+region-holdout+coord-aligned")
    if run:
        section("[F] NicheCompass neighbourhood scale: rescale X_hat_nbr rows to the X_nbr library and re-score")
        for f in sorted(glob.glob(os.path.join(run, "predicted_adata_seed*.h5ad"))):
            sd = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
            B = ad.read_h5ad(f)
            Xb = dense(B.X)
            testb = B.obs["data_split"].astype(str).to_numpy() == "test"
            Xhn, Xn = get_mat(B, "X_hat_nbr"), get_mat(B, "X_nbr")
            r = Xhn.sum(1) / np.maximum(Xn.sum(1), 1e-8)
            if "spatial_connectivities" in B.obsp:
                indeg = np.asarray(B.obsp["spatial_connectivities"].sum(0)).ravel()
                rho = pd.Series(r[testb]).corr(pd.Series(indeg[testb]), method="spearman")
                log(f"  seed {sd}: Spearman(row-sum ratio, in-degree) on test = {rho:.3f} (>~0.3 => in-neighbour/GCN aggregation artefact)")
            Xhn_rs = Xhn * (Xn.sum(1) / np.maximum(Xhn.sum(1), 1e-8))[:, None]
            record(sd, "NC_nbr_original", score_test(Xb, testb, None, Xn, Xhn))
            record(sd, "NC_nbr_rowsum_rescaled", score_test(Xb, testb, None, Xn, Xhn_rs))
            del B
            gc.collect()


baselines()


# ============================================================================= G. misc
@guarded
def misc():
    section("[G] stage-2 npz coverage, train-vs-test gaps, siblings, full-data discreteness, scVI epochs")
    if KEEP:
        want = np.where(KEEP["test"])[0]
        for run in SEED_RUNS:
            p = os.path.join(STAGE2_NPZ_DIR, run, "stage2_predicted_codes.npz")
            if os.path.isfile(p):
                g = np.load(p)["global_idx"].astype(np.int64)
                log(f"  npz {run}: n={g.size} unique={np.unique(g).size} equals seed0 test rows: {np.array_equal(np.unique(g), want)}")
    if MS is not None:
        for br in ("cell", "niche"):
            trn = [extract(MS[MS.seed == s], br, "train")["pe_cw_log"] for s in sorted(MS.seed.unique())]
            tst = [extract(MS[MS.seed == s], br, "test")["pe_cw_log"] for s in sorted(MS.seed.unique())]
            log(f"  SQUINT {br}: pe_cw_log train {np.nanmean(trn):.4f} | test {np.nanmean(tst):.4f}")
    for sib in SIBLINGS:
        runs = sorted(glob.glob(os.path.join(DS, sib, "*_seed*", "predicted_adata.h5ad")))
        log(f"  sibling {sib}: {len(runs)} seed run(s)")
        for f in runs:
            A = ad.read_h5ad(f)
            X = dense(A.X)
            t = A.obs["data_split"].astype(str).to_numpy() == "test"
            sd = int(re.search(r"seed(\d+)", f).group(1))
            record(sd, "SIB_" + ("dec-w256" if "dec-w256" in sib else "dec-w400x2"),
                   score_test(X, t, get_mat(A, "X_hat"), get_mat(A, "X_nbr"), get_mat(A, "X_hat_nbr")))
            del A
            gc.collect()
    for v in FULLDATA:
        c = sorted(glob.glob(os.path.join(DS, v + "__multiseed", "*", "metrics", "per_seed_pearson_reconstruction.csv")))
        if not c:
            log(f"  full-data {v}: no multiseed CSV")
            continue
        D = pd.read_csv(c[-1])
        d = D[(D.branch == "cell") & (D.split == "all") & (D.transform == "log1p") & (D.gene_subset == "all")]
        log(f"  full-data {v} ({c[-1]}): in-sample cell pe_cw_log {d[d.axis == 'cell_wise'].pearson_mean.mean():.4f} "
            f"pe_gw_log {d[d.axis == 'gene_wise'].pearson_mean.mean():.4f} (n_seeds={D.seed.nunique()})")
    for f in sorted(glob.glob(os.path.join(LOGDIR, "*scvi*"))):
        try:
            with open(f, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 400000))
                txt = fh.read().decode("utf-8", "replace")
            ep = re.findall(r"Epoch (\d+)/(\d+)", txt)
            log(f"  scVI log {os.path.basename(f)}: last 'Epoch i/n' = {ep[-1] if ep else None}")
        except OSError:
            pass


misc()

# ============================================================================= summary
@guarded
def summary():
    section("SUMMARY (5-seed means; deltas vs V0 = the published SQUINT row)")
    R = pd.DataFrame(RESULTS)
    R.to_csv(os.path.join(OUT, "variants_per_seed.csv"), index=False)
    M = R.groupby(["variant", "branch"], sort=False)[COLS + ["pe_cw_raw"]].mean().reset_index()
    M["n_seeds"] = R.groupby(["variant", "branch"], sort=False)["seed"].nunique().values
    M.to_csv(os.path.join(OUT, "variants_mean.csv"), index=False)
    for br in ("cell", "niche"):
        v0 = M[(M.variant == "V0_stored_published") & (M.branch == br)]
        log(f"\n{br.upper()} level      " + " ".join(f"{c:>10s}" for c in COLS))
        log(f"{'published SQUINT':<40s}" + " ".join(f"{v:10.3f}" for v in PUB[(br, 'SQUINT')]))
        for _, r in M[M.branch == br].iterrows():
            line = f"{r.variant:<40s}" + " ".join(f"{r[c]:10.4f}" for c in COLS)
            if len(v0) and r.variant != "V0_stored_published" and not r.variant.startswith("NC_"):
                line += "   d=" + ",".join(f"{r[c] - v0.iloc[0][c]:+.4f}" for c in COLS)
            log(line)
        for k, v in PUB.items():
            if k[0] == br and k[1] != "SQUINT":
                log(f"{'published ' + k[1]:<40s}" + " ".join(f"{x:10.3f}" for x in v))
    log(f"\nwrote {OUT}/variants_per_seed.csv and variants_mean.csv  [total {time.time() - T0:.0f}s]")


summary()
PY
echo "done -> $OUT"
