"""TimeXer backbone: all OHLCV channels are endogenous patches + per-variate G_en.

Exogenous tokens are *not* price channels. Text is attached by Model C1 only
(G_en as query). Adapted from thuml/Time-Series-Library models/TimeXer.py
(multivariate endogenous / M-style patching).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(-1)])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class EndogenousPatchEmbed(nn.Module):
    """Patch every endogenous channel and append a learnable G_en per channel."""

    def __init__(
        self,
        n_vars: int,
        d_model: int,
        patch_len: int,
        patch_stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model) * 0.02)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # y: [B, T, C] all endogenous
        batch, _seq, n_vars = y.shape
        series = y.permute(0, 2, 1)
        patches = series.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        _b, _c, n_patches, _p = patches.shape
        flat = patches.reshape(batch * n_vars, n_patches, self.patch_len)
        tokens = self.value_embedding(flat) + self.position_embedding(flat)
        tokens = self.dropout(tokens).view(batch, n_vars, n_patches, -1)
        g_en = self.glb_token.expand(batch, -1, -1, -1)
        return tokens, g_en


class _MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, key, value, need_weights=False)
        return out


class TimeXerEncoderLayer(nn.Module):
    """Per-variate self-attn on [P; G_en]; G_en cross-attn to exo (text only in C1)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = _MHA(d_model, n_heads, dropout)
        self.cross_attn = _MHA(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tokens: torch.Tensor, exo: torch.Tensor | None, n_vars: int
    ) -> torch.Tensor:
        # tokens: [B * n_vars, S+1, D]; exo: [B, E, D] or None
        tokens = self.norm1(tokens + self.dropout(self.self_attn(tokens, tokens, tokens)))
        patches, g_en = tokens[:, :-1, :], tokens[:, -1:, :]
        if exo is not None and exo.numel() > 0 and exo.size(1) > 0:
            batch = exo.size(0)
            d_model = g_en.size(-1)
            g = g_en.reshape(batch, n_vars, d_model)
            g = g + self.dropout(self.cross_attn(g, exo, exo))
            g_en = self.norm2(g.reshape(batch * n_vars, 1, d_model))
        else:
            g_en = self.norm2(g_en)
        y = torch.cat([patches, g_en], dim=1)
        ff = self.dropout(F.gelu(self.conv1(y.transpose(1, 2))))
        ff = self.dropout(self.conv2(ff).transpose(1, 2))
        return self.norm3(y + ff)


class TimeXerBackbone(nn.Module):
    """Patch all OHLCV channels → optional proto injection (caller) → encoder."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        target_idx: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        patch_len: int,
        patch_stride: int,
        dropout: float,
        d_ff: int | None = None,
        use_norm: bool = False,
    ) -> None:
        super().__init__()
        if not 0 <= target_idx < n_features:
            raise ValueError(f"target_idx={target_idx} not in [0, {n_features})")
        self.seq_len = seq_len
        self.n_features = n_features
        self.target_idx = target_idx
        self.use_norm = use_norm
        self.patch_embed = EndogenousPatchEmbed(
            n_features, d_model, patch_len, patch_stride, dropout
        )
        d_ff = d_ff or 4 * d_model
        self.layers = nn.ModuleList(
            [TimeXerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )

    def embed(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # x: [B, T, C] — all channels endogenous
        stats: dict[str, torch.Tensor] = {}
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev
            stats["means"] = means
            stats["stdev"] = stdev
        patches, g_en = self.patch_embed(x)
        return patches, g_en, stats

    def encode(
        self, patches: torch.Tensor, g_en: torch.Tensor, exo: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # patches: [B, C, S, D], g_en: [B, C, 1, D], exo: [B, E, D] or None
        batch, n_vars, n_patches, d_model = patches.shape
        tokens = torch.cat([patches, g_en], dim=2).reshape(
            batch * n_vars, n_patches + 1, d_model
        )
        for layer in self.layers:
            tokens = layer(tokens, exo, n_vars=n_vars)
        tokens = tokens.view(batch, n_vars, n_patches + 1, d_model)
        return tokens[:, :, :-1, :], tokens[:, :, -1:, :]
