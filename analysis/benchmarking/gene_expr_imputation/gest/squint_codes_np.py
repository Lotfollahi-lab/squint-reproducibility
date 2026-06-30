"""
Numpy-only wiring for the GeST-arch-on-SQUINT-codes arm (no torch).

Factors out the torch-free logic so it is unit-testable without the ML stack:

  * ``read_squint_codes``  -- pull the per-cell code stack (cell L0/L1 +
    niche L0/L1) and codebook sizes out of a frozen ``predicted_adata`` using
    the SAME conventions as ``vqniche.stage2.data.AnnDataCodeSource``
    (obsm['cell_code_indices'] / obsm['neighborhood_code_indices'], with the
    obsm/uns key aliases). We mirror rather than import it so this analysis
    module stays decoupled from the squint package import path.

  * ``mask_target_codes``  -- replace held-out target slots with the per-target
    "unknown" row (K_t), the index the model's code embeddings reserve as the
    mask token. This is the exact indexing the torch ``_embed_codes`` performs;
    keeping it here lets a numpy test pin the index arithmetic.

  * ``build_target_codes_array`` -- assemble the (n, T) ground-truth code-stack
    matrix from the per-branch arrays, in the target order
    [cell L0, cell L1, niche L0, niche L1].

  * ``per_target_sizes`` -- the codebook sizes in target order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Code-target spec (pure python; lives here so it imports without torch)
# ---------------------------------------------------------------------------
@dataclass
class CodeStackSpec:
    """The ordered (branch, level) code targets and their codebook sizes.

    Mirrors SQUINT stage-2's ``prediction_targets`` ordering: cell stack first
    (coarse->fine), then niche stack. ``names`` and ``sizes`` are parallel.
    """

    names: List[str]                # e.g. ["cell.L0", "cell.L1", "niche.L0", "niche.L1"]
    sizes: List[int]                # codebook size K_t per target

    def __post_init__(self) -> None:
        if len(self.names) != len(self.sizes):
            raise ValueError("names and sizes must be parallel")
        if not self.names:
            raise ValueError("at least one code target required")
        self.sizes = [int(k) for k in self.sizes]
        if any(k <= 0 for k in self.sizes):
            raise ValueError("codebook sizes must be positive")

    @property
    def n_targets(self) -> int:
        return len(self.names)

    @property
    def cell_levels(self) -> List[int]:
        """Indices of the cell-branch targets (for decoding the cell stack)."""
        return [i for i, n in enumerate(self.names) if n.startswith("cell")]

    @classmethod
    def from_branch_sizes(cls, sizes_cell: Sequence[int],
                          sizes_niche: Sequence[int]) -> "CodeStackSpec":
        names, sizes = [], []
        for lvl, k in enumerate(sizes_cell):
            names.append(f"cell.L{lvl}"); sizes.append(int(k))
        for lvl, k in enumerate(sizes_niche):
            names.append(f"niche.L{lvl}"); sizes.append(int(k))
        return cls(names=names, sizes=sizes)


# obsm/uns key aliases per branch (matches AnnDataCodeSource._PREFIX_ALIASES /
# the obsm names SQUINT's predict pipeline writes).
_CELL_OBSM = ("cell_code_indices",)
_NICHE_OBSM = ("neighborhood_code_indices", "niche_code_indices")
_CELL_SIZES_UNS = ("codebook_sizes_cell",)
_NICHE_SIZES_UNS = ("codebook_sizes_niche", "codebook_sizes")


def _to_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "detach"):              # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _read_obsm_indices(adata, keys: Sequence[str], n_cells: int,
                       branch: str) -> np.ndarray:
    for k in keys:
        if k in adata.obsm:
            idx = _to_numpy(adata.obsm[k])
            idx = np.asarray(idx)
            if idx.ndim == 1:
                idx = idx[:, None]
            if idx.shape[0] != n_cells:
                raise ValueError(
                    f"branch '{branch}': {idx.shape[0]} code rows != {n_cells} cells")
            return idx.astype(np.int64)
    raise KeyError(
        f"could not find code indices for branch '{branch}' "
        f"(looked for obsm{list(keys)})")


def _resolve_sizes(adata, uns_keys: Sequence[str], idx: np.ndarray) -> List[int]:
    sizes = None
    for k in uns_keys:
        if k in adata.uns:
            sizes = adata.uns[k]
            break
    if sizes is not None:
        sizes = [int(s) for s in np.asarray(sizes).ravel().tolist()]
    if not sizes or len(sizes) != idx.shape[1]:
        # fall back to observed max per level (+1)
        sizes = [int(idx[:, q].max()) + 1 for q in range(idx.shape[1])]
    return sizes


def read_squint_codes(adata) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Read the true SQUINT code stacks + codebook sizes from a predicted_adata.

    Returns
    -------
    codes_cell  : (n, Lc) int64   cell-branch residual codes (level 0..Lc-1)
    codes_niche : (n, Ln) int64   niche-branch residual codes (level 0..Ln-1)
    sizes_cell  : list[int] of length Lc
    sizes_niche : list[int] of length Ln
    """
    n = int(adata.n_obs)
    codes_cell = _read_obsm_indices(adata, _CELL_OBSM, n, "cell")
    codes_niche = _read_obsm_indices(adata, _NICHE_OBSM, n, "niche")
    sizes_cell = _resolve_sizes(adata, _CELL_SIZES_UNS, codes_cell)
    sizes_niche = _resolve_sizes(adata, _NICHE_SIZES_UNS, codes_niche)
    return codes_cell, codes_niche, sizes_cell, sizes_niche


def build_target_codes_array(codes_cell: np.ndarray,
                             codes_niche: np.ndarray) -> np.ndarray:
    """Stack per-branch codes into one (n, T) matrix in target order
    [cell L0..Lc-1, niche L0..Ln-1]."""
    codes_cell = np.asarray(codes_cell, dtype=np.int64)
    codes_niche = np.asarray(codes_niche, dtype=np.int64)
    if codes_cell.shape[0] != codes_niche.shape[0]:
        raise ValueError("cell/niche code rows differ")
    return np.concatenate([codes_cell, codes_niche], axis=1)


def per_target_sizes(sizes_cell: Sequence[int],
                     sizes_niche: Sequence[int]) -> List[int]:
    """Codebook sizes in target order [cell..., niche...]."""
    return [int(k) for k in sizes_cell] + [int(k) for k in sizes_niche]


def target_names(sizes_cell: Sequence[int],
                 sizes_niche: Sequence[int]) -> List[str]:
    """Target names in order, e.g. ['cell.L0','cell.L1','niche.L0','niche.L1']."""
    names = [f"cell.L{l}" for l in range(len(sizes_cell))]
    names += [f"niche.L{l}" for l in range(len(sizes_niche))]
    return names


def mask_target_codes(codes_stack: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Replace every code with its per-target 'unknown' row (K_t).

    This is the indexing the torch model applies to held-out target slots
    (``_embed_codes`` sees row K_t == the mask token). Returns (n, T) int64
    where column t is filled with sizes[t]. Kept as a pure-numpy mirror so the
    index arithmetic is testable without torch.
    """
    codes_stack = np.asarray(codes_stack, dtype=np.int64)
    sizes = list(sizes)
    if codes_stack.shape[1] != len(sizes):
        raise ValueError(
            f"codes have {codes_stack.shape[1]} targets but {len(sizes)} sizes")
    out = np.empty_like(codes_stack)
    for t, k in enumerate(sizes):
        out[:, t] = int(k)            # the unknown/mask row for target t
    return out


def clamp_observed_codes(codes_stack: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Clamp observed (known) codes into [0, K_t-1] -- defensive against stray
    out-of-range entries (mirrors the torch ``.clamp(0, K-1)`` for content)."""
    codes_stack = np.asarray(codes_stack, dtype=np.int64)
    sizes = list(sizes)
    out = codes_stack.copy()
    for t, k in enumerate(sizes):
        np.clip(out[:, t], 0, int(k) - 1, out=out[:, t])
    return out
