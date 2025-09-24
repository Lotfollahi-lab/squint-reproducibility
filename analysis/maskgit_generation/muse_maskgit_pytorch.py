import math
from random import random
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import pathlib
from pathlib import Path
import torchvision.transforms as T

from typing import Callable, Optional, List, Dict

from einops import rearrange, repeat

# from beartype import beartype

from attend import Attend

from tqdm.auto import tqdm

from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningModule

from vqniche.utils.loss_utils import aggregate_1hop_neighbor_features

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.ff = FeedForward(out_dim)

    def forward(self, x, mask = None):
        if mask is not None:
            m = mask.to(dtype=x.dtype).unsqueeze(-1)   # (B, N, 1)
            x = x * m
        z = self.in_proj(x)
        z = z + self.ff(z)
        
        # re-applying the mask on the output guarantees zeros even if later change internals (e.g., add biases)
        if mask is not None:
            z = z * m
        return z

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        scale = 8,
        flash = True,
        dropout = 0.
    ):
        super().__init__()
        self.scale = scale
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.attend = Attend(
            flash = flash,
            dropout = dropout,
            scale = scale
        )

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        x_mask = None,
        context = None,
        context_mask = None
    ):
        assert not (exists(context) ^ self.cross_attend)

        n = x.shape[-2]
        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        if exists(x_mask):
            x_mask = repeat(x_mask, 'b j -> b h i j', h = h, i = n)
            x_mask = F.pad(x_mask, (1, 0), value = True)
        elif exists(context_mask):
            context_mask = repeat(context_mask, 'b j -> b h i j', h = h, i = n)
            context_mask = F.pad(context_mask, (1, 0), value = True)

        out = self.attend(q, k, v, mask = x_mask if exists(x_mask) else context_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        flash = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash),
                Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend = True, flash = flash),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x, x_mask = None, context = None, context_mask = None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x, x_mask = x_mask) + x

            x = cross_attn(x, context = context, context_mask = context_mask) + x

            x = ff(x) + x

        return self.norm(x)

# transformer - it's all we need

class MaskGitTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        num_timepoints,
        dim,
        add_mask_id = True,
        add_pad_id = True,
        dim_out = None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        
        if add_mask_id:
            self.mask_id = num_tokens
        else:
            self.mask_id = None
        num_tokens_with_mask_id = num_tokens + int(add_mask_id)
        if add_pad_id:
            self.pad_id = num_tokens_with_mask_id
            self.tp_pad_id = num_timepoints
            # zero inputs produce zero outputs and no gradients flow from padded positions as long as bias=False in MLPBlock
            self.pad_xy_value = 0
        else:
            self.pad_id = None
            self.tp_pad_id = None
            self.pad_xy_value = None
        num_tokens_with_mask_id_and_pad_id = num_tokens_with_mask_id + int(add_pad_id)

        self.num_tokens = num_tokens
        self.token_embed = nn.Embedding(num_tokens_with_mask_id_and_pad_id, dim)
        if num_timepoints > 1:
            self.timepoint_embed = nn.Embedding(num_timepoints + int(add_pad_id), dim)
        else:
            self.timepoint_embed = None

        self.xy_embed = MLPBlock(2, dim)
        
        self.transformer_blocks = TransformerBlocks(dim = dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias = False)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        if cond_scale == 1:
            return self.forward(*args, return_embed = return_embed, cond_drop_prob = 0., **kwargs)

        logits, embed = self.forward(*args, return_embed = True, cond_drop_prob = 0., **kwargs)

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    # TODO: Adjust it to be used for mask-free editing
    # def forward_with_neg_prompt(
    #     self,
    #     *args,
    #     text_embed: torch.Tensor,
    #     neg_text_embed: torch.Tensor,
    #     cond_scale = 3.,
    #     return_embed = False,
    #     **kwargs
    # ):
    #     neg_logits = self.forward(*args, neg_text_embed = neg_text_embed, cond_drop_prob = 0., **kwargs)
    #     pos_logits, embed = self.forward(*args, return_embed = True, text_embed = text_embed, cond_drop_prob = 0., **kwargs)

    #     scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

    #     if return_embed:
    #         return scaled_logits, embed

    #     return scaled_logits

    def forward(
        self,
        anchor,
        context,
        return_embed = False,
        return_logits = False,
        labels = None,
        ignore_index = 0,
        cond_drop_prob = 0.,
    ):

        # prepare context

        context_emb = self.token_embed(context['ids'])
        if self.timepoint_embed is not None:
            context_emb = context_emb + self.timepoint_embed(context['tids'])

        # embed tokens

        anchor_emb = self.token_embed(anchor['ids'])
        
        # classifier free guidance on xy embedding

        if cond_drop_prob > 0.:
            mask = prob_mask_like((anchor['xy'].shape[0], 1), 1. - cond_drop_prob, anchor['xy'].device)
            pos_emb = self.xy_embed(anchor['xy'], mask = anchor['mask'] & mask)
        else:
            pos_emb = self.xy_embed(anchor['xy'], mask = anchor['mask'])
        anchor_emb = anchor_emb + pos_emb

        # transformer blocks

        embed = self.transformer_blocks(anchor_emb, x_mask = anchor['mask'], context = context_emb, context_mask = context['mask'])

        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels)
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = ignore_index)

        if not return_logits:
            return loss

        return loss, logits

# classifier free guidance functions

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device = None):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return uniform(shape, device = device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# noise schedules

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

# main maskgit classes

# @beartype
class MaskGit(nn.Module):
    def __init__(
        self,
        vae,
        transformer: MaskGitTransformer,
        noise_schedule: Callable = cosine_schedule,
        cond_drop_prob = 0.5,
        no_mask_token_prob = 0.,
        anchor_time_point: Optional[str] = 't3',
        context_time_points: Optional[List[str]] = ['t1', 't2', 't4'],
    ):
        super().__init__()
        self.vae = vae.eval()

        self.transformer = transformer
        self.mask_id = transformer.mask_id
        self.pad_id = transformer.pad_id
        self.tp_pad_id = transformer.tp_pad_id
        self.pad_xy_value = transformer.pad_xy_value

        self.noise_schedule = noise_schedule
        self.cond_drop_prob = cond_drop_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob
        
        self.anchor_time_point = anchor_time_point
        self.context_time_points = context_time_points
        self.time_point_keys = [self.anchor_time_point] + self.context_time_points

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def tokenize_batch(self, batch):
        # tokenized batch
        tokenized_batch = {}
        # xy conditioning
        xy_batch = {}
        # batch sizes for each timepoint
        batch_sizes = []
        # tokenize batch
        for k in self.time_point_keys:
            tp = batch[k]
            encoder_conditions = getattr(tp, 'encoder_conditions', None)
            spatial_prior_features = getattr(tp, 'spatial_prior_features', None)
            _, _, ids, _ = self.vae.encoder(tp.x, tp.edge_index, encoder_conditions, spatial_prior_features)
                
            # splits along dim=0 by graph boundaries (ptr) to get batch of chunks
            sizes = (tp.ptr[1:] - tp.ptr[:-1]).tolist()
            tokenized_batch[k] = list(torch.split(ids, sizes, dim=0))
            xy_batch[k] = list(torch.split(tp.xy_coordinates, sizes, dim=0))
            batch_sizes.append(len(sizes))
        return tokenized_batch, xy_batch, batch_sizes

    def get_padded_anchor_and_context(self, tokenized_batch, xy_batch, batch_size):
        # batch anchor
        batch_anchor = {'ids': tokenized_batch[self.anchor_time_point], 'xy': xy_batch[self.anchor_time_point]}
        # pad anchor ids, xy
        batch_anchor['ids'] = pad_sequence(batch_anchor['ids'], batch_first=True, padding_value=self.pad_id)
        batch_anchor['xy'] = pad_sequence(batch_anchor['xy'], batch_first=True, padding_value=self.pad_xy_value)
        batch_anchor['mask'] = batch_anchor['ids'] != self.pad_id

        # concatenate context ids, tids, and xy across timepoints
        batch_context = {'ids': [], 'tids': [], 'xy': []}
        for i in range(batch_size):
            context_ids = []
            context_tids = []
            context_xy = []
            for t_idx, k in enumerate(self.context_time_points):
                context_ids.append(tokenized_batch[k][i])
                context_tids.append(torch.full((tokenized_batch[k][i].shape[0],), t_idx, dtype=torch.long, device=tokenized_batch[k][i].device))
                context_xy.append(xy_batch[k][i])
            
            batch_context['ids'].append(torch.cat(context_ids, dim=0))
            batch_context['tids'].append(torch.cat(context_tids, dim=0))
            batch_context['xy'].append(torch.cat(context_xy, dim=0))
        
        # pad context ids, tids, and xy across timepoints
        batch_context['ids'] = pad_sequence(batch_context['ids'], batch_first=True, padding_value=self.pad_id)
        batch_context['tids'] = pad_sequence(batch_context['tids'], batch_first=True, padding_value=self.tp_pad_id)
        batch_context['xy'] = pad_sequence(batch_context['xy'], batch_first=True, padding_value=self.pad_xy_value)
        batch_context['mask'] = batch_context['ids'] != self.pad_id

        return batch_anchor, batch_context

    def detokenize_batch(self, ids, batch):
        h_quantized = self.vae.encoder.vq.get_codes_from_indices(ids)
        batch_attr_decoder_conditions = getattr(batch[self.anchor_time_point], 'attr_decoder_conditions', None)
        xhat = self.vae.attribute_decoder(x=h_quantized, read_depth=batch[self.anchor_time_point].x.sum(dim=-1), conditions=batch_attr_decoder_conditions)
        return xhat

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        batch: Dict[str, Batch],
        temperature = 1.,
        topk_filter_thres = 0.9,
        timesteps = 18,  # ideal number of steps is 18 in maskgit paper
        cond_scale = 3,
    ):
        
        # tokenize batch
        tokenized_batch, xy_batch, batch_sizes = self.tokenize_batch(batch)
        
        assert len(set(batch_sizes)) == 1, f"Mismatched batch sizes across timepoints: {batch_sizes}"
        batch_size = batch_sizes[0]
        
        # pad batch and split into anchor and context
        batch_anchor, batch_context = self.get_padded_anchor_and_context(tokenized_batch, xy_batch, batch_size)

        # get some basic variables

        ids = batch_anchor['ids']
        seq_lens = batch[self.anchor_time_point].ptr[1:] - batch[self.anchor_time_point].ptr[:-1]
        padded_seq_len, device = ids.shape[1], ids.device

        # mask all token ids in anchor
        ids = torch.where(batch_anchor['mask'], self.mask_id, ids)
        scores = torch.zeros((batch_size, padded_seq_len), dtype = torch.float32, device = device)
        max_neg_value = -torch.finfo(torch.float32).max
        scores[~batch_anchor['mask']] = max_neg_value

        starting_temperature = temperature

        demask_fn = self.transformer.forward_with_cond_scale

        # TODO: Adjust it to be used for mask-free editing
        # negative prompting, as in paper

        # neg_text_embeds = None
        # if exists(negative_texts):
        #     assert len(texts) == len(negative_texts)

        #     neg_text_embeds = self.transformer.encode_text(negative_texts)
        #     demask_fn = partial(self.transformer.forward_with_neg_prompt, neg_text_embeds = neg_text_embeds)

        for timestep, steps_until_x0 in tqdm(zip(torch.linspace(0, 1, timesteps, device = device), reversed(range(timesteps))), total = timesteps):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = (rand_mask_prob * seq_lens).round().to(torch.int32).clamp(min = 1)
            max_k = num_token_masked.max()

            masked_indices = scores.topk(max_k, dim = -1).indices
            
            # boolean mask: which of the topk positions to actually use per row
            take = torch.arange(max_k, device=device).unsqueeze(0) < num_token_masked.unsqueeze(1)  # [B, max_k]
            
            # row indices aligned with masked_indices
            rows = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(masked_indices)  # [B, max_k]

            ids[rows[take], masked_indices[take]] = self.mask_id
            batch_anchor['ids'] = ids

            logits = demask_fn(
                batch_anchor,
                batch_context,
                cond_scale = cond_scale,
            )

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            is_mask = ids == self.mask_id

            ids = torch.where(
                is_mask,
                pred_ids,
                ids
            )

            probs_without_temperature = logits.softmax(dim = -1)

            scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')

            scores = scores.masked_fill(~is_mask, max_neg_value)

        # get ids

        if not exists(self.vae):
            return ids

        xhat_batch = self.detokenize_batch(ids[batch_anchor['mask']], batch)
        return xhat_batch

    def forward(
        self,
        batch: Dict[str, Batch],
        ignore_index = -1,
        cond_drop_prob = None,
    ):

        # tokenize batch
        tokenized_batch, xy_batch, batch_sizes = self.tokenize_batch(batch)
        
        assert len(set(batch_sizes)) == 1, f"Mismatched batch sizes across timepoints: {batch_sizes}"
        batch_size = batch_sizes[0]
        
        # pad batch and split into anchor and context
        batch_anchor, batch_context = self.get_padded_anchor_and_context(tokenized_batch, xy_batch, batch_size)

        # get some basic variables

        ids = batch_anchor['ids']
        seq_lens = batch[self.anchor_time_point].ptr[1:] - batch[self.anchor_time_point].ptr[:-1]
        batch, padded_seq_len, device, cond_drop_prob = *ids.shape, ids.device, default(cond_drop_prob, self.cond_drop_prob)

        # prepare mask

        rand_time = uniform((batch,), device = device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_lens * rand_mask_probs).round().clamp(min = 1)

        rand_int = torch.rand((batch, padded_seq_len), device = device)
        # Set padded positions to 1 to exclude from masking
        rand_int[~batch_anchor['mask']] = 1
        batch_randperm = rand_int.argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        # Since padded positons are excluded previously, they will always be ignored and will not contribute to loss calculation
        labels = torch.where(mask, ids, ignore_index)

        if self.no_mask_token_prob > 0.:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        batch_anchor['ids'] = torch.where(mask, self.mask_id, ids)

        # get loss

        ce_loss = self.transformer(
            batch_anchor,
            batch_context,
            labels = labels,
            cond_drop_prob = cond_drop_prob,
            ignore_index = ignore_index,
        )

        return ce_loss

# LightningModule class

class MaskGitRunner(LightningModule):
    def __init__(
        self,
        base: MaskGit,
        batch_size,
        learning_rate = 0.0001,
        weight_decay = 0.001,
        cond_scale = 3.,
    ):
        super().__init__()
        self.base_maskgit = base
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cond_scale = cond_scale

    def forward(self, batch):
        return self.base_maskgit(batch)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log(
            'train/loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            rank_zero_only=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log(
            'val/loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            rank_zero_only=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self(batch)
        self.log(
            'test/loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
            rank_zero_only=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx):
        anchor_tp = batch[self.base_maskgit.anchor_time_point]
        
        # Get xhat from VAE
        encoder_conditions = getattr(anchor_tp, 'encoder_conditions', None)
        attr_decoder_conditions = getattr(anchor_tp, 'attr_decoder_conditions', None)
        adj_decoder_conditions = getattr(anchor_tp, 'adj_decoder_conditions', None)
        h_latent, \
        h_quantized, \
        indices, \
        vae_xhat_batch, \
        h_adj_batch, \
        unnormalized_logits_batch, \
        _ \
            = self.base_maskgit.vae(
                anchor_tp.x, 
                anchor_tp.edge_index, 
                batch_encoder_conditions=encoder_conditions,
                batch_attr_decoder_conditions=attr_decoder_conditions,
                batch_adj_decoder_conditions=adj_decoder_conditions,
            )

        # Get xhat from MaskGit
        maskgit_xhat_batch = self.base_maskgit.generate(batch, cond_scale=self.cond_scale)
        
        original_X = {
            'X': anchor_tp.x[anchor_tp.ptr[:-1]],
            'X_nbr': aggregate_1hop_neighbor_features(
                X=anchor_tp.x,
                edge_index=anchor_tp.edge_index,
                return_mean=True,
                batch_size=None,
            )[anchor_tp.ptr[:-1]],
        }

        step_results = {
            'vae': {
                'X': original_X['X'],
                'X_nbr': original_X['X_nbr'],
                'X_hat': vae_xhat_batch[anchor_tp.ptr[:-1]],
                'X_hat_nbr': aggregate_1hop_neighbor_features(
                    X=vae_xhat_batch,
                    edge_index=anchor_tp.edge_index,
                    return_mean=True,
                    batch_size=None,
                )[anchor_tp.ptr[:-1]],
            },
            'maskgit': {
                'X': original_X['X'],
                'X_nbr': original_X['X_nbr'],
                'X_hat': maskgit_xhat_batch[anchor_tp.ptr[:-1]],
                'X_hat_nbr': aggregate_1hop_neighbor_features(
                    X=maskgit_xhat_batch,
                    edge_index=anchor_tp.edge_index,
                    return_mean=True,
                    batch_size=None,
                )[anchor_tp.ptr[:-1]],
            },
        }

        return step_results
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.base_maskgit.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {
            'optimizer': optimizer,
            'monitor': 'train/masking_loss',
        }