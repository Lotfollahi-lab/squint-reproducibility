"""
GeST training loop + imputation inference (torch). Per-dataset training, like
the paper's GP/MLP baselines: fit on the TRAIN (non-held-out) cells, predict the
held-out region from observed (train) neighbors.

Coordinates are centred per section and scaled by the section's median
nearest-neighbor distance, so the SPE sees relative geometry in "cell-diameter"
units -- scale-invariant across platforms (mmb mixes MERFISH + STARmap with very
different coordinate frames).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .tokenizer import MetaCellVocab, _to_dense
from .serialization import (
    crop_square, diagonal_serialize, neighbor_context, spatial_attention_mask,
)


def _norm_coords_per_section(coords: np.ndarray, section: np.ndarray) -> np.ndarray:
    """Centre coords per section and scale by the section's median NN distance."""
    out = np.zeros_like(coords, dtype=np.float32)
    for s in np.unique(section):
        m = section == s
        c = coords[m].astype(np.float64)
        c = c - c.mean(axis=0, keepdims=True)
        # median nearest-neighbor distance as the length unit
        try:
            from sklearn.neighbors import NearestNeighbors
            kk = min(2, c.shape[0])
            if kk >= 2:
                nn = NearestNeighbors(n_neighbors=2).fit(c)
                d, _ = nn.kneighbors(c)
                unit = float(np.median(d[:, 1]))
            else:
                unit = 1.0
        except Exception:
            unit = 1.0
        out[m] = (c / (unit if unit > 0 else 1.0)).astype(np.float32)
    return out


def _additive(bool_mask: np.ndarray, device, dtype):
    import torch
    m = torch.as_tensor(bool_mask, device=device)
    bias = torch.zeros(m.shape, dtype=dtype, device=device)
    return bias.masked_fill(~m, float("-inf"))


def train_gest(
    X: np.ndarray,                 # (n, T) raw counts (full adata)
    coords: np.ndarray,            # (n, 2)
    section: np.ndarray,           # (n,) section id
    train_mask: np.ndarray,        # (n,) bool, True == train cell
    model,                         # GeST (vocab already set)
    vocab: MetaCellVocab,
    *,
    window: float,
    seq_n: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
    log_every: int = 50,
) -> None:
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    Xd = _to_dense(X)
    tok_expr = vocab.token_expr(Xd)                        # (n, T) g(x), tokenized
    labels = vocab.labels_all_levels(Xd)                   # 4 x (n,)
    ncoords = _norm_coords_per_section(coords, section)

    # train-cell pools per section
    sections = np.unique(section)
    train_idx_by_sec = {
        int(s): np.where((section == s) & train_mask)[0] for s in sections
    }
    sec_pool = [s for s in sections if train_idx_by_sec[int(s)].size >= seq_n]
    if not sec_pool:
        # shrink seq_n to the largest available train section
        biggest = max(int(train_idx_by_sec[int(s)].size) for s in sections)
        seq_n = max(8, min(seq_n, biggest))
        sec_pool = [s for s in sections if train_idx_by_sec[int(s)].size >= seq_n]
    if not sec_pool:
        raise RuntimeError("no section has enough train cells for GeST training")
    L = seq_n - 1
    bias = _additive(spatial_attention_mask(L), device, torch.float32)  # (2L, 2L)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    cap = 2 * seq_n                                        # serialize-input cap
    def _one_crop():
        for _ in range(30):
            s = int(sec_pool[rng.integers(len(sec_pool))])
            pool = train_idx_by_sec[s]
            local = crop_square(ncoords[pool], window, rng, min_cells=seq_n)
            if local is None or local.size < seq_n:
                continue
            g = pool[local]                                # global idx of crop
            # A dense square can hold hundreds-thousands of cells; the diagonal
            # serializer is an O(n^2) python loop, so cap its input to the `cap`
            # cells nearest a random seed in the crop before ordering. We only
            # keep the first seq_n of the order anyway -> bounded cost, same
            # connected-patch semantics.
            if g.size > cap:
                seed = int(rng.integers(g.size))
                d2 = ((ncoords[g] - ncoords[g][seed]) ** 2).sum(1)
                g = g[np.argpartition(d2, cap - 1)[:cap]]
            order = diagonal_serialize(ncoords[g], rng)[:seq_n]
            return g[order]                                # (seq_n,) global, ordered
        return None

    for step in range(1, steps + 1):
        crops = []
        while len(crops) < batch_size:
            c = _one_crop()
            if c is not None:
                crops.append(c)
        crops = np.stack(crops, 0)                         # (B, seq_n) global idx

        cg = torch.as_tensor(tok_expr[crops[:, :L]], dtype=torch.float32, device=device)
        cp = torch.as_tensor(ncoords[crops[:, :L]], dtype=torch.float32, device=device)
        tp = torch.as_tensor(ncoords[crops[:, 1:seq_n]], dtype=torch.float32, device=device)
        gt = [torch.as_tensor(labels[i][crops[:, 1:seq_n]], dtype=torch.long, device=device)
              for i in range(len(vocab.level_sizes))]

        yhat = model(cg, cp, tp, bias)                     # (B, L, T)
        loss = model.hierarchical_loss(yhat, gt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"  [gest] step {step}/{steps}  loss={loss.item():.4f}  "
                  f"(seq_n={seq_n}, B={batch_size})")


def predict_all(
    X: np.ndarray,
    coords: np.ndarray,
    section: np.ndarray,
    train_mask: np.ndarray,
    model,
    vocab: MetaCellVocab,
    *,
    neighbors_k: int,
    device: str,
    batch_size: int = 2048,
    mode: str = "weighted",
) -> np.ndarray:
    """Predict X_hat for EVERY cell from its nearest OBSERVED (train) neighbors.

    Held-out (test) cells are predicted purely from train neighbors (their own
    expression is never used). Train cells are predicted from their nearest
    OTHER train cells (self excluded). Returns (n, T)."""
    import torch

    Xd = _to_dense(X)
    tok_expr = vocab.token_expr(Xd)
    ncoords = _norm_coords_per_section(coords, section)
    n, T = Xd.shape
    X_hat = np.zeros((n, T), dtype=np.float32)
    model.eval()

    for s in np.unique(section):
        sec = section == s
        obs = np.where(sec & train_mask)[0]                # observed pool
        tgt = np.where(sec)[0]                             # predict all cells here
        if obs.size == 0:
            X_hat[tgt] = tok_expr[tgt]                     # no context -> self token
            continue
        k = int(min(neighbors_k, obs.size))
        # nearest observed neighbors (+1 so a train cell can drop itself)
        nbr = neighbor_context(ncoords[tgt], ncoords[obs], min(k + 1, obs.size))
        # full-attend bias for (k content + 1 target)
        bias = torch.zeros(k + 1, k + 1, dtype=torch.float32, device=device)
        for start in range(0, tgt.size, batch_size):
            bt = tgt[start:start + batch_size]
            bn = nbr[start:start + batch_size]             # (b, >=k)
            # drop self where present, keep first k
            ctx = np.empty((bt.size, k), dtype=np.int64)
            for j, (ti, row) in enumerate(zip(bt, bn)):
                cand = obs[row]
                cand = cand[cand != ti][:k]
                if cand.size < k:                          # pad by repeating last
                    cand = np.concatenate([cand, np.full(k - cand.size, cand[-1])])
                ctx[j] = cand
            cg = torch.as_tensor(tok_expr[ctx], dtype=torch.float32, device=device)
            cp = torch.as_tensor(ncoords[ctx], dtype=torch.float32, device=device)
            tpos = torch.as_tensor(ncoords[bt][:, None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                yhat = model(cg, cp, tpos, bias)           # (b, 1, T)
                xh = model.decode(yhat, mode=mode)[:, 0, :]
            X_hat[bt] = xh.detach().cpu().numpy().astype(np.float32)
    return X_hat
