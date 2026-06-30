"""
GeST serialization (sec 3.2, Eq. 6) + inference neighbor gathering. Pure numpy.

Training serialization: crop a square window from a section, pick a cell at one
of the square's four corners as x1, then sample the remaining cells WITHOUT
replacement with probability proportional to their Euclidean distance to x1
(Eq. 6). This orders cells along diagonal bands so spatially adjacent cells get
similar indices while keeping order randomness.

Inference: each held-out (target) cell is predicted from its k nearest OBSERVED
(train) cells -- the model's trained conditional P(g(x) | s(x), g(N(x)), s(N(x)))
applied with observed neighbors (no autoregressive error accumulation, since the
context cells are real observations, not predictions).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def crop_square(
    coords: np.ndarray,
    window: float,
    rng: np.random.Generator,
    min_cells: int = 16,
    max_tries: int = 20,
) -> Optional[np.ndarray]:
    """Indices of cells inside a randomly-placed square of side `window`.

    The square is centred on a random cell; returns None if no placement
    yields >= `min_cells` after `max_tries`.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n == 0:
        return None
    half = float(window) / 2.0
    for _ in range(max_tries):
        c = coords[rng.integers(n)]
        inside = (
            (np.abs(coords[:, 0] - c[0]) <= half)
            & (np.abs(coords[:, 1] - c[1]) <= half)
        )
        idx = np.where(inside)[0]
        if idx.size >= min_cells:
            return idx
    # fall back to the densest attempt (last idx) if it has any cells
    return idx if idx.size > 0 else None


def diagonal_serialize(coords_sub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Eq. 6 ordering of the cropped cells. Returns a permutation (n,) of indices
    into `coords_sub`: x1 (a corner cell) first, then distance-to-x1-weighted
    sampling without replacement."""
    coords_sub = np.asarray(coords_sub, dtype=np.float64)
    n = coords_sub.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    # x1 = cell nearest a randomly chosen corner of the crop's bounding box.
    lo = coords_sub.min(axis=0)
    hi = coords_sub.max(axis=0)
    corner = np.array([
        lo[0] if rng.integers(2) == 0 else hi[0],
        lo[1] if rng.integers(2) == 0 else hi[1],
    ])
    x1 = int(np.argmin(((coords_sub - corner) ** 2).sum(axis=1)))

    dist = np.sqrt(((coords_sub - coords_sub[x1]) ** 2).sum(axis=1))  # weights
    order = [x1]
    remaining = np.ones(n, dtype=bool)
    remaining[x1] = False
    w = dist.copy()
    w[x1] = 0.0
    # iterative weighted sampling without replacement (Eq. 6)
    for _ in range(n - 1):
        wr = np.where(remaining, w, 0.0)
        tot = wr.sum()
        if tot <= 0:                       # all-zero weights (degenerate) -> uniform
            choices = np.where(remaining)[0]
            pick = int(choices[rng.integers(choices.size)])
        else:
            pick = int(rng.choice(n, p=wr / tot))
        order.append(pick)
        remaining[pick] = False
    return np.asarray(order, dtype=np.int64)


def neighbor_context(
    target_coords: np.ndarray,
    observed_coords: np.ndarray,
    k: int,
) -> np.ndarray:
    """For each target cell, indices (into `observed_coords`) of its k nearest
    observed neighbors. Returns (n_target, k) int (k capped at #observed).

    Uses sklearn when available, else a vectorised brute force (fine for the
    held-out-region sizes here)."""
    target_coords = np.asarray(target_coords, dtype=np.float64)
    observed_coords = np.asarray(observed_coords, dtype=np.float64)
    n_obs = observed_coords.shape[0]
    k = int(min(k, n_obs))
    if k <= 0:
        return np.zeros((target_coords.shape[0], 0), dtype=np.int64)
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k).fit(observed_coords)
        return nn.kneighbors(target_coords, return_distance=False).astype(np.int64)
    except Exception:
        out = np.zeros((target_coords.shape[0], k), dtype=np.int64)
        for i in range(target_coords.shape[0]):
            d2 = ((observed_coords - target_coords[i]) ** 2).sum(axis=1)
            out[i] = np.argsort(d2, kind="stable")[:k]
        return out


def spatial_attention_mask(seq_len_L: int) -> np.ndarray:
    """GeST Spatial-Attention mask (Eq. 5) as a (2L, 2L) bool array.

    Token layout: positions 0..L-1 are the neighbor content tokens
    gs(x_1..x_L); positions L..2L-1 are the target position tokens
    s(x_2..x_{N}). The target token at position i+L (predicting cell x_{i+1},
    i in [1, L]) attends to neighbor tokens 1..i (indices 0..i-1) and the
    target tokens L+1..L+i (indices L..L+i-1). True == attend.

    Content rows (0..L-1) use causal self-attention (content row j attends
    0..j). The paper specifies only the target rows (Eq. 5); content rows need
    *some* valid attention or their softmax row is all -inf (NaN). Causal is the
    natural decoder-only choice and leaves the target readout unchanged.
    """
    L = int(seq_len_L)
    M = np.zeros((2 * L, 2 * L), dtype=bool)
    for j in range(L):                          # content causal self-attention
        M[j, 0:j + 1] = True
    for i in range(1, L + 1):
        row = i - 1 + L                         # 0-based position i+L
        M[row, 0:i] = True                      # neighbor content tokens 1..i
        M[row, L:L + i] = True                  # target tokens L+1..L+i (incl self)
    return M
