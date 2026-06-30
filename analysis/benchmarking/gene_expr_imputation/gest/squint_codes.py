"""
GeST ARCHITECTURE on SQUINT's FROZEN discrete codes (torch).

This is the "architecture vs codes/decoder" ablation arm. It reuses the exact
GeST transformer body + Spatial-Attention suffix mask + diagonal serialization +
2D sinusoidal SPE from ``model.py`` / ``serialization.py`` / ``positional.py``,
but replaces:

  * INPUT  -- instead of a meta-cell expression profile g(x), each cell's content
              token is an embedding of its SQUINT code stack
              (cell L0, cell L1, niche L0, niche L1): one learned ``nn.Embedding``
              per (branch, level), summed. Held-out (target) cells use the per-code
              "unknown/mask" row (index K_t) -- the same convention SQUINT stage-2
              uses (``model.py``: ``nn.Embedding(K+1, d)``, last row == unknown).
  * OUTPUT -- instead of one meta-cell readout + hierarchical CE over C_expr, four
              independent classification heads (cell L0, cell L1, niche L0,
              niche L1) over the SQUINT codebook sizes. Loss = summed cross-entropy
              over the four code targets on the supervised (held-out) cells.

Predicted codes are then decoded to expression by SQUINT's FROZEN stage-1 NB
decoder (see ``examples/stage2_decode_pearson.py`` -- reused by the runner), so
this arm shares the EXACT codes + decoder with the "SQUINT (imputed)" bar and
differs only in the prediction transformer.

v1 simplification (noted): the four heads are predicted INDEPENDENTLY (no
hierarchical teacher-forced cell->niche conditioning, unlike SQUINT stage-2's
``head_logits``). Independent CE per head is acceptable for an architecture
ablation; a hierarchical head can be layered on later behind a flag.

Nothing here edits the shared stage-2 files (we only mirror their conventions).
``torch`` is required for this module; the numpy-only wiring lives in
``squint_codes_np.py`` and is unit-tested without torch.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the GeST transformer body + SPE verbatim (no fork of the arch).
from .model import Block, GeSTConfig, RMSNorm
from .positional import spe, spe_dim
# CodeStackSpec lives in the torch-free module so it is importable + testable
# without the ML stack; re-exported here for convenience.
from .squint_codes_np import CodeStackSpec   # noqa: F401


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GeSTSquintCodes(nn.Module):
    """GeST transformer that predicts a cell's SQUINT code stack from context.

    Token layout follows GeST exactly (``model.GeST``):
      * content tokens (neighbor cells) carry the embedding of their KNOWN code
        stack + SPE(pos) + type_embed[content];
      * target tokens (held-out cells) carry the per-code "unknown" embedding
        (mask row) + SPE(pos) + type_embed[target].
    The transformer body + the caller-supplied Spatial-Attention bias are reused
    unchanged; only the embedding (code-stack, not g(x)) and the head (4 code
    classifiers, not a meta-cell readout) differ.
    """

    def __init__(self, spec: CodeStackSpec, cfg: GeSTConfig):
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.d_model = cfg.d_model
        self.spe_d = spe_dim(cfg.d_model)

        # one input embedding per code target; row index K_t == "unknown/mask"
        # (same +1 convention as SQUINT stage-2 model.py). Content (observed)
        # cells index rows 0..K_t-1; target (held-out) cells index row K_t.
        self.code_embed = nn.ModuleList(
            [nn.Embedding(k + 1, cfg.d_model) for k in spec.sizes]
        )
        self.spe_proj = nn.Linear(self.spe_d, cfg.d_model)
        self.type_embed = nn.Embedding(2, cfg.d_model)   # 0=content, 1=target

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

        # one classification head per code target (over its codebook size)
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, k) for k in spec.sizes]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    # ---- embedding -----------------------------------------------------------
    @property
    def unknown_rows(self) -> List[int]:
        """Per-target 'unknown/mask' embedding row index (== K_t)."""
        return list(self.spec.sizes)

    def _spe(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B, L, 2) -> (B, L, d_model). Identical to GeST._spe.
        B, L, _ = pos.shape
        flat = pos.reshape(-1, 2).detach().cpu().numpy()
        e = spe(flat, self.spe_d)
        e = torch.as_tensor(e, dtype=torch.float32, device=pos.device).reshape(B, L, self.spe_d)
        return self.spe_proj(e)

    def _embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum the per-target code embeddings. codes: (B, L, T) long -> (B, L, D).

        Caller passes the "unknown" row (K_t) for any target whose code is
        masked (held-out target slot).
        """
        B, L, T = codes.shape
        out = codes.new_zeros((B, L, self.d_model), dtype=torch.float32)
        for t in range(T):
            idx = codes[:, :, t].clamp(0, self.spec.sizes[t])   # K_t == unknown row
            out = out + self.code_embed[t](idx)
        return out

    def forward(
        self,
        content_codes: torch.Tensor,   # (B, Lc, T) long -- known code stacks
        content_pos: torch.Tensor,     # (B, Lc, 2)
        target_pos: torch.Tensor,      # (B, Lt, 2)
        attn_bias: torch.Tensor,       # (S, S) or (B, S, S) additive, S = Lc + Lt
        key_padding: Optional[torch.Tensor] = None,  # (B, S) True == PAD
    ) -> List[torch.Tensor]:
        """Returns a list of T logit tensors, each (B, Lt, K_t), at target slots."""
        B, Lc, T = content_codes.shape
        Lt = target_pos.shape[1]
        content = (self._embed_codes(content_codes)
                   + self._spe(content_pos)
                   + self.type_embed.weight[0][None, None, :])
        # target slots: every code is "unknown" (the mask row), like SQUINT
        # stage-2's masked held-out cells.
        unk = torch.as_tensor(self.unknown_rows, dtype=torch.long,
                              device=content_codes.device)
        tgt_codes = unk[None, None, :].expand(B, Lt, T)
        target = (self._embed_codes(tgt_codes)
                  + self._spe(target_pos)
                  + self.type_embed.weight[1][None, None, :])
        x = torch.cat([content, target], dim=1)             # (B, S, D)

        bias = attn_bias
        if key_padding is not None:
            pad_bias = torch.zeros_like(key_padding, dtype=x.dtype)
            pad_bias = pad_bias.masked_fill(key_padding, float("-inf"))   # (B, S)
            bias = attn_bias[None] + pad_bias[:, None, :]    # (B, S, S)
        for blk in self.blocks:
            x = blk(x, bias)
        x = self.norm_f(x)
        h = x[:, Lc:, :]                                     # (B, Lt, D) target slots
        return [self.heads[t](h) for t in range(T)]          # T x (B, Lt, K_t)

    # ---- loss + predict ------------------------------------------------------
    def code_loss(
        self,
        logits: List[torch.Tensor],            # T x (B, Lt, K_t)
        gt_codes: List[torch.Tensor],          # T x (B, Lt) long
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Summed independent cross-entropy over the T code targets (v1).

        Every target slot is supervised here (the held-out cell whose codes we
        predict), so no per-cell supervised mask is needed -- the runner builds
        crops where every target slot is a genuine prediction target.
        """
        total = logits[0].new_zeros(())
        for t, lg in enumerate(logits):
            K = self.spec.sizes[t]
            total = total + F.cross_entropy(
                lg.reshape(-1, K), gt_codes[t].reshape(-1),
                label_smoothing=label_smoothing,
            )
        return total

    @torch.no_grad()
    def predict_codes(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """Argmax each head -> predicted code stack. Returns (B, Lt, T) long."""
        cols = [lg.argmax(dim=-1) for lg in logits]          # T x (B, Lt)
        return torch.stack(cols, dim=-1)                     # (B, Lt, T)
