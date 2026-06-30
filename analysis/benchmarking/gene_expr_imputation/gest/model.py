"""
GeST decoder-only transformer + hierarchical loss + decode (torch).

Faithful to Hao et al. (MLCB 2025):
  - content token gs(x) = Linear_T->D(g(x)) + SPE(pos) + type_embed  (g(x) is the
    meta-cell expression profile, Eq. 7); target token = SPE(pos) + type_embed.
  - RMSNorm pre-norm decoder blocks + FFN (Fig. 2); attention restricted by a
    caller-supplied boolean mask (the Spatial-Attention mask, Eq. 5, at train
    time; a target->all-context mask at inference).
  - MLP head -> yhat in R^T; loss projects yhat to meta-cell logits
    z = yhat @ C_expr^T and applies hierarchical CE over 4 levels (Eq. 8-9,
    alpha=0.25). Decode = weighted aggregation p(c)@C_expr (or picking).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import spe, spe_dim


@dataclass
class GeSTConfig:
    d_model: int = 256
    n_layers: int = 8                 # paper default L8H8
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    alpha: float = 0.25               # hierarchical-loss weight per level


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class Block(nn.Module):
    def __init__(self, cfg: GeSTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.norm1 = RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_bias):
        # x: (B, S, D); attn_bias: (S, S) or (B, S, S) additive (0 / -inf)
        B, S, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)                          # (B, S, H, dh)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)   # (B, H, S, S)
        att = att + (attn_bias[None, None] if attn_bias.dim() == 2
                     else attn_bias[:, None])
        att = att.softmax(dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GeST(nn.Module):
    """GeST spatial generative transformer.

    Set the meta-cell vocabulary tensors via `set_vocab` before training/decode.
    """

    def __init__(self, n_genes: int, cfg: GeSTConfig):
        super().__init__()
        self.cfg = cfg
        self.n_genes = int(n_genes)
        self.d_model = cfg.d_model
        self.spe_d = spe_dim(cfg.d_model)
        self.gene_embed = nn.Linear(self.n_genes, cfg.d_model)
        self.spe_proj = nn.Linear(self.spe_d, cfg.d_model)
        self.type_embed = nn.Embedding(2, cfg.d_model)   # 0=content, 1=target
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, self.n_genes),
        )
        # vocab buffers (filled by set_vocab)
        self.register_buffer("C_expr", torch.zeros(1, self.n_genes), persistent=True)
        self._level_maps: List[torch.Tensor] = []
        self.level_sizes: List[int] = []

    # ---- vocab ---------------------------------------------------------------
    def set_vocab(self, C_expr: np.ndarray, level_labels: List[np.ndarray],
                  level_sizes: List[int]) -> None:
        dev = next(self.parameters()).device
        self.C_expr = torch.as_tensor(np.asarray(C_expr), dtype=torch.float32, device=dev)
        self._level_maps = [torch.as_tensor(np.asarray(m), dtype=torch.long, device=dev)
                            for m in level_labels]
        self.level_sizes = [int(s) for s in level_sizes]

    # ---- embedding -----------------------------------------------------------
    def _spe(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B, L, 2) -> (B, L, d_model) via numpy SPE + learned projection
        B, L, _ = pos.shape
        flat = pos.reshape(-1, 2).detach().cpu().numpy()
        e = spe(flat, self.spe_d)                          # (B*L, spe_d)
        e = torch.as_tensor(e, dtype=torch.float32, device=pos.device).reshape(B, L, self.spe_d)
        return self.spe_proj(e)

    def forward(
        self,
        content_gene: torch.Tensor,    # (B, Lc, T) neighbor meta-cell profiles g(x)
        content_pos: torch.Tensor,     # (B, Lc, 2)
        target_pos: torch.Tensor,      # (B, Lt, 2)
        attn_bias: torch.Tensor,       # (S, S) additive, S = Lc + Lt
        key_padding: Optional[torch.Tensor] = None,  # (B, S) True == PAD
    ) -> torch.Tensor:
        B, Lc, _ = content_gene.shape
        Lt = target_pos.shape[1]
        # log1p the raw-count neighbor profiles before the embedding Linear:
        # raw counts have a huge dynamic range and destabilise the embedding.
        # The token identity is still the meta-cell profile C_expr[i]; only its
        # numeric scale into the network changes (decode is unaffected).
        content = (self.gene_embed(torch.log1p(content_gene.clamp_min(0)))
                   + self._spe(content_pos)
                   + self.type_embed.weight[0][None, None, :])
        target = self._spe(target_pos) + self.type_embed.weight[1][None, None, :]
        x = torch.cat([content, target], dim=1)            # (B, S, D)
        bias = attn_bias
        if key_padding is not None:
            pad_bias = torch.zeros_like(key_padding, dtype=x.dtype)
            pad_bias = pad_bias.masked_fill(key_padding, float("-inf"))  # (B, S)
            bias = attn_bias[None] + pad_bias[:, None, :]   # (B, S, S)
        for blk in self.blocks:
            x = blk(x, bias)
        x = self.norm_f(x)
        yhat = self.head(x[:, Lc:, :])                     # (B, Lt, T) target slots
        return yhat

    # ---- loss + decode -------------------------------------------------------
    def meta_probs(self, yhat: torch.Tensor) -> torch.Tensor:
        """p(c) = softmax(yhat @ C_expr^T) over the K meta cells (Eq. 8). (B, Lt, K).

        z is divided by sqrt(T) as a fixed temperature: C_expr holds RAW mean
        counts, so the un-scaled dot product over T genes has std ~tens at init
        -> near-one-hot softmax + vanishing gradients. The 1/sqrt(T) scale
        (attention-style) keeps logits O(1) without changing the ranking. Decode
        uses the RAW C_expr unchanged, so predicted-expression units are intact.
        """
        z = (yhat @ self.C_expr.t()) / (self.n_genes ** 0.5)
        return z.softmax(dim=-1)

    def hierarchical_loss(self, yhat: torch.Tensor,
                          gt_levels: List[torch.Tensor]) -> torch.Tensor:
        """Eq. 9: sum_i alpha * CE(p^(i), gt_i). gt_levels[i]: (B, Lt) long."""
        p = self.meta_probs(yhat)                          # (B, Lt, K)
        B, Lt, K = p.shape
        pf = p.reshape(-1, K)
        loss = yhat.new_zeros(())
        for i, lvl_map in enumerate(self._level_maps):
            n_lab = self.level_sizes[i]
            # aggregate meta-cell probs into level-i label probs via scatter-add
            agg = yhat.new_zeros(pf.shape[0], n_lab)
            agg.index_add_(1, lvl_map, pf)                 # sum_c [l_i(c)==k] p(c)
            gt = gt_levels[i].reshape(-1)
            agg = agg.clamp_min(1e-9)
            loss = loss + self.cfg.alpha * F.nll_loss(agg.log(), gt)
        return loss

    @torch.no_grad()
    def decode(self, yhat: torch.Tensor, mode: str = "weighted") -> torch.Tensor:
        """yhat (B, Lt, T) -> predicted expression (B, Lt, T)."""
        if mode == "picking":
            idx = self.meta_probs(yhat).argmax(dim=-1)     # (B, Lt)
            return self.C_expr[idx]
        p = self.meta_probs(yhat)                          # (B, Lt, K)
        return p @ self.C_expr                             # weighted aggregation
