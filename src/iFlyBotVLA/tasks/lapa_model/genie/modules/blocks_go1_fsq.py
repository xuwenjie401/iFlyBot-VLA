import math
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, int32


def patchify(videos: Tensor, size: int) -> Tensor:
    B, T, C, H, W  = videos.shape
    videos = videos[:, :, :, :H - (H % size), :W - (W % size)]
    x = rearrange(videos, "b t c (hn hp) (wn wp)  -> b t (hn wn) (hp wp c)", hp=size, wp=size)
    return x


def unpatchify(patches: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = rearrange(patches, "b t (hn wn) (hp wp c) -> b t c (hn hp) (wn wp) ", hp=size, wp=size, hn=hn)
    return x[:, :, :, :h_out, :w_out]


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]].cuda()


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super(SelfAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim // 2)

    def scaled_dot_product_attention(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            is_causal: bool = False,
            attn_mask: Tensor = None,
    ) -> Tensor:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            attn_bias = attn_bias.unsqueeze(0).repeat(query.shape[0], 1, 1)
            attn_bias.masked_fill_((attn_mask>0).logical_not().unsqueeze(1), float("-inf"))
            attn_bias = attn_bias.unsqueeze(1)
            
        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor, is_causal: bool = False, attn_mask: Tensor = None) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q)
            k = self.rotary_embedding.rotate_queries_or_keys(k)

        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioTemporalBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, causal_temporal: bool = False, attn_mask: Tensor = None) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_, is_causal=False, attn_mask=attn_mask)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x_ = self.norm2(x)
        if causal_temporal:
            x_ = self.temporal_attn(x_, is_causal=True)
        else:
            x_ = self.temporal_attn(x_)
        x = x + x_
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        x_ = self.norm3(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTemporalTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            causal_temporal: bool = False,
            to_out: bool = True,
    ) -> None:
        super(SpatioTemporalTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        if to_out:
            self.out = nn.Linear(model_dim, out_dim)
        else:
            self.out = nn.Identity()

        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=2)

        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal, attn_mask)

        x = self.out(x)
        return x  # (B, T, E)


class MVSpatioTemporalTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
            causal_temporal: bool = False,
            to_out: bool = True,
    ) -> None:
        super(MVSpatioTemporalTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.view_embed = nn.Parameter(torch.zeros(2, model_dim), requires_grad=True)
        nn.init.normal_(self.view_embed, std=0.02)

        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        if to_out:
            self.out = nn.Linear(model_dim, out_dim)
        else:
            self.out = nn.Identity()

        self.causal_temporal = causal_temporal

    def forward(self, latent_action: Tensor, view1: Tensor, view2: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        view1 = self.ffn(view1) + repeat(self.view_embed[0], 'd -> b m n d', b = view1.shape[0], m = view1.shape[1], n=1)
        view2 = self.ffn(view2) + repeat(self.view_embed[1], 'd -> b m n d', b = view1.shape[0], m = view1.shape[1], n=1)
        
        x = torch.cat([latent_action, view1, view2], dim=2)
        x = self.pos_enc(x)

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=2)

        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal, attn_mask)

        x = self.out(x)
        return x  # (B, T, E)

class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)


    def forward(self, x: Tensor, attn_mask: Tensor = None) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_, attn_mask=attn_mask)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
    ) -> None:
        super(SpatioTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=2)
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.out(x)
        return x  # (B, T, E)


class MVSpatioTransformer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            out_dim: int,
            num_blocks: int,
            num_heads: int,
            dropout: float = 0.0,
    ) -> None:
        super(MVSpatioTransformer, self).__init__()
        self.ffn = nn.Linear(in_dim, model_dim)

        self.pos_enc = PositionalEncoding(model_dim)
        # self.view_embed = nn.Parameter(torch.zeros(2, model_dim), requires_grad=True)
        # nn.init.normal_(self.view_embed, std=0.02)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, latent_action: Tensor, view1: Tensor, lang_embed: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        view1 = self.ffn(view1) #+ repeat(self.view_embed[0], 'd -> b m n d', b = view1.shape[0], m = view1.shape[1], n=1)
        # view2 = self.ffn(view2) + repeat(self.view_embed[1], 'd -> b m n d', b = view1.shape[0], m = view1.shape[1], n=1)
        
        x = torch.cat([latent_action, view1], dim=2)
        x = self.pos_enc(x)

        if lang_embed is not None:
            x = torch.cat([x, lang_embed], dim=2)

        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.out(x)
        return x  # (B, T, E)


class FSQ(nn.Module):
    def __init__(self, levels, eps=1e-3, dim=4):
        super(FSQ, self).__init__()
        self._levels = levels
        self._eps = eps
        self._dim = dim
        self._levels_tensor = torch.tensor(levels, dtype=int32).to('cuda')
        self._basis = torch.cumprod(torch.tensor([1] + self._levels[:-1]), dim=0, dtype=int32).to('cuda')
        self._implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size, device='cuda'))

    @property
    def num_dimensions(self) -> int:
        """Number of dimensions expected from inputs."""
        return len(self._levels)

    @property
    def codebook_size(self) -> torch.Tensor:
        """Size of the codebook."""
        return self._levels_tensor.prod().item()
    @property
    def codebook(self) -> torch.Tensor:
        """Returns the implicit codebook. Shape (prod(levels), num_dimensions)."""
        return self._implicit_codebook

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels_tensor - 1) * (1 - self._eps) / 2
        offset = torch.where(self._levels_tensor % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z))
        # Renormalize to [-1, 1]
        half_width = self._levels_tensor // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        # Scale and shift to range [0, ..., L-1]
        half_width = self._levels_tensor // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels_tensor // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.num_dimensions
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `indexes_to_codes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels_tensor
        return self._scale_and_shift_inverse(codes_non_centered)

    def round_ste(self, z: torch.Tensor) -> torch.Tensor:
        """Round with straight through gradients."""
        zhat = torch.round(z)
        return z + (zhat - z).detach()

    def forward(self, x: Tensor):
        # Compute distances
        codes = self.quantize(x)
        indices = self.codes_to_indices(codes)
        
        return codes, indices
