"""
Faithful reimplementation of GeST (Hao et al., MLCB 2025, PMLR v311) for the
gene-expression imputation benchmark.

GeST has no public code/weights, so this reimplements the paper's "unseen cell
generation" mechanism (spatial imputation: predict a held-out region's per-cell
expression from the surrounding cells). Components map 1:1 to the paper:

  tokenizer.MetaCellVocab   meta-cell vocabulary (Eq. 7) + 4-level hierarchy
                            (Eq. 8) + projection of model output to meta-cell
                            logits and weighted-aggregation / picking decode.
  positional.spe            2D sinusoidal spatial position encoding (SPE, ViT).
  serialization             square crop + diagonal-path distance-weighted
                            ordering (Eq. 6); per-target neighbor gathering.
  model.GeST                decoder-only transformer + Spatial-Attention mask
                            (Eq. 5) + hierarchical cross-entropy loss (Eq. 9).

The numpy core (tokenizer / positional / serialization) imports without torch
and is unit-tested; `model` needs torch.
"""

from .tokenizer import MetaCellVocab          # noqa: F401
from .positional import spe, spe_dim          # noqa: F401
from .serialization import (                   # noqa: F401
    crop_square,
    diagonal_serialize,
    neighbor_context,
)

__all__ = [
    "MetaCellVocab",
    "spe",
    "spe_dim",
    "crop_square",
    "diagonal_serialize",
    "neighbor_context",
]
