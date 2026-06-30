"""
Training loop + per-cell inference for the GeST-arch-on-SQUINT-codes arm (torch).

Mirrors ``gest/train.py`` (``train_gest`` / ``predict_all``) one-to-one, but the
tokens are SQUINT code stacks (not meta-cell expression profiles) and the
supervision is the four code targets (not the hierarchical meta-cell CE):

  * training: crop a square of TRAIN cells, diagonal-serialize it (Eq. 6), build
    the Spatial-Attention suffix mask, embed the L content cells' KNOWN code
    stacks, and predict the L target cells' four codes -> summed CE.
  * inference: each target cell is predicted from its k nearest OBSERVED (train)
    neighbors with a full-attend mask -> argmax codes per head.

Coordinate normalisation (centre per section + scale by median NN distance) is
reused verbatim from ``gest/train.py``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .serialization import (
    crop_square, diagonal_serialize, neighbor_context, spatial_attention_mask,
)
from .train import _norm_coords_per_section, _additive


def train_gest_codes(
    codes_stack: np.ndarray,       # (n, T) int64 true SQUINT codes (cell..niche)
    coords: np.ndarray,            # (n, 2)
    section: np.ndarray,           # (n,) section id
    train_mask: np.ndarray,        # (n,) bool, True == train cell
    model,                         # GeSTSquintCodes
    *,
    window: float,
    seq_n: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    seed: int,
    label_smoothing: float = 0.0,
    log_every: int = 50,
) -> None:
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    codes_stack = np.asarray(codes_stack, dtype=np.int64)
    T = codes_stack.shape[1]
    ncoords = _norm_coords_per_section(coords, section)

    sections = np.unique(section)
    train_idx_by_sec = {
        int(s): np.where((section == s) & train_mask)[0] for s in sections
    }
    sec_pool = [s for s in sections if train_idx_by_sec[int(s)].size >= seq_n]
    if not sec_pool:
        biggest = max(int(train_idx_by_sec[int(s)].size) for s in sections)
        seq_n = max(8, min(seq_n, biggest))
        sec_pool = [s for s in sections if train_idx_by_sec[int(s)].size >= seq_n]
    if not sec_pool:
        raise RuntimeError("no section has enough train cells for GeST training")
    L = seq_n - 1
    bias = _additive(spatial_attention_mask(L), device, torch.float32)  # (2L, 2L)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    cap = 2 * seq_n
    def _one_crop():
        for _ in range(30):
            s = int(sec_pool[rng.integers(len(sec_pool))])
            pool = train_idx_by_sec[s]
            local = crop_square(ncoords[pool], window, rng, min_cells=seq_n)
            if local is None or local.size < seq_n:
                continue
            g = pool[local]
            if g.size > cap:
                seed_i = int(rng.integers(g.size))
                d2 = ((ncoords[g] - ncoords[g][seed_i]) ** 2).sum(1)
                g = g[np.argpartition(d2, cap - 1)[:cap]]
            order = diagonal_serialize(ncoords[g], rng)[:seq_n]
            return g[order]
        return None

    for step in range(1, steps + 1):
        crops = []
        while len(crops) < batch_size:
            c = _one_crop()
            if c is not None:
                crops.append(c)
        crops = np.stack(crops, 0)                         # (B, seq_n) global idx

        # content = first L cells' code stacks; targets = cells 1..seq_n-1.
        cc = torch.as_tensor(codes_stack[crops[:, :L]], dtype=torch.long, device=device)
        cp = torch.as_tensor(ncoords[crops[:, :L]], dtype=torch.float32, device=device)
        tp = torch.as_tensor(ncoords[crops[:, 1:seq_n]], dtype=torch.float32, device=device)
        gt = [torch.as_tensor(codes_stack[crops[:, 1:seq_n], t], dtype=torch.long,
                              device=device) for t in range(T)]

        logits = model(cc, cp, tp, bias)                   # T x (B, L, K_t)
        loss = model.code_loss(logits, gt, label_smoothing=label_smoothing)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"  [gest-codes] step {step}/{steps}  loss={loss.item():.4f}  "
                  f"(seq_n={seq_n}, B={batch_size}, T={T})")


def predict_codes_all(
    codes_stack: np.ndarray,
    coords: np.ndarray,
    section: np.ndarray,
    train_mask: np.ndarray,
    model,
    *,
    neighbors_k: int,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predict the code stack for EVERY cell from its nearest OBSERVED (train)
    neighbors. Held-out cells use train neighbors only (their codes never seen).
    Returns (n, T) int64 predicted codes.

    Mirrors ``gest/train.py:predict_all`` but emits argmax codes (4 heads)
    instead of decoded expression.
    """
    import torch

    codes_stack = np.asarray(codes_stack, dtype=np.int64)
    n, T = codes_stack.shape
    ncoords = _norm_coords_per_section(coords, section)
    pred = codes_stack.copy()                              # default: self codes
    model.eval()

    for s in np.unique(section):
        sec = section == s
        obs = np.where(sec & train_mask)[0]
        tgt = np.where(sec)[0]
        if obs.size == 0:
            continue                                       # no context -> keep self
        k = int(min(neighbors_k, obs.size))
        nbr = neighbor_context(ncoords[tgt], ncoords[obs], min(k + 1, obs.size))
        bias = torch.zeros(k + 1, k + 1, dtype=torch.float32, device=device)
        for start in range(0, tgt.size, batch_size):
            bt = tgt[start:start + batch_size]
            bn = nbr[start:start + batch_size]
            ctx = np.empty((bt.size, k), dtype=np.int64)
            for j, (ti, row) in enumerate(zip(bt, bn)):
                cand = obs[row]
                cand = cand[cand != ti][:k]                # drop self if present
                if cand.size < k:                          # pad by repeating last
                    cand = np.concatenate([cand, np.full(k - cand.size, cand[-1])])
                ctx[j] = cand
            cc = torch.as_tensor(codes_stack[ctx], dtype=torch.long, device=device)
            cp = torch.as_tensor(ncoords[ctx], dtype=torch.float32, device=device)
            tpos = torch.as_tensor(ncoords[bt][:, None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(cc, cp, tpos, bias)         # T x (b, 1, K_t)
                codes = model.predict_codes(logits)[:, 0, :]   # (b, T)
            pred[bt] = codes.detach().cpu().numpy().astype(np.int64)
    return pred
