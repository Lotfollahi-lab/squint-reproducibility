"""
2D sinusoidal spatial position encoding (SPE), GeST sec 3.3 (ViT-style).

Maps a 2D coordinate to R^d: half the dimensions encode x, half encode y, each
via sin/cos pairs at geometric frequencies. Coordinates are RELATIVE to the
section centre (the caller centres them), so only relative geometry matters.

Pure numpy so it's unit-testable; the torch model calls `spe` and wraps the
result in a tensor.
"""

from __future__ import annotations

import numpy as np


def spe_dim(dim: int) -> int:
    """Round a requested embedding dim down to a multiple of 4 (2 axes x sin/cos)."""
    return max(4, (int(dim) // 4) * 4)


def spe(coords: np.ndarray, dim: int, max_period: float = 10000.0) -> np.ndarray:
    """2D sinusoidal encoding of `coords` (n, 2) -> (n, dim).

    `dim` must be a multiple of 4 (use `spe_dim`). For each axis we use
    dim/4 frequencies and emit [sin, cos], so each axis contributes dim/2
    features; x then y are concatenated.
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got {coords.shape}")
    if dim % 4 != 0:
        raise ValueError(f"dim must be a multiple of 4, got {dim}")
    n_freq = dim // 4
    # geometric frequencies, ViT/Transformer convention
    inv_freq = 1.0 / (max_period ** (np.arange(n_freq, dtype=np.float32) / n_freq))
    out = np.zeros((coords.shape[0], dim), dtype=np.float32)
    for axis in range(2):
        ang = coords[:, axis:axis + 1] * inv_freq[None, :]        # (n, n_freq)
        base = axis * (dim // 2)
        out[:, base:base + n_freq] = np.sin(ang)
        out[:, base + n_freq:base + 2 * n_freq] = np.cos(ang)
    return out
