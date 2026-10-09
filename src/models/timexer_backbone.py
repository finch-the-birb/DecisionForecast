"""TimeXer-style backbone: multivariate PatchEmbed + one G_en + optional exo cross-attn.

OHLCV channels are endogenous PatchEmbed features (flattened per patch), not exogenous
variate tokens. The exo slot is reserved for text in C1. Adapted from
thuml/Time-Series-Library models/TimeXer.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def n_patches(seq_len: int, patch_len: int, patch_stride: int) -> int:
    if seq_len < patch_len:
        raise ValueError(f"seq_len={seq_len} < patch_len={patch_len}")
    return (seq_len - patch_len) // patch_stride + 1


def unfold_time(x: torch.Tensor, patch_len: int, patch_stride: int) -> torch.Tensor:
    """Unfold the time axis. x: [B, T, ...] -> [B, N, ..., P]."""
    return x.unfold(dimension=1, size=patch_len, step=patch_stride)


def days_to_patches(
    text_seq: torch.Tensor, patch_len: int, patch_stride: int
) -> torch.Tensor:
    """Mean-pool days inside each time patch. text_seq [B, T, D] -> [B, N, D]."""
    patches = unfold_time(text_seq, patch_len, patch_stride)
    return patches.mean(dim=-1)


class _PositionalEmbedding(nn.Module):
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

    def forward(self, n: int) -> torch.Tensor:
        return self.pe[:, :n]


class PatchEmbed(nn.Module):
    """Multivariate patches: each window of length P over C channels → one token."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        patch_len: int,
        patch_stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.value = nn.Linear(n_features * patch_len, d_model)
        self.position = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> P: [B, N, D]
        patches = unfold_time(x, self.patch_len, self.patch_stride)
        batch, n_p, n_feat, plen = patches.shape
        flat = patches.reshape(batch, n_p, n_feat * plen)
        tokens = self.value(flat) + self.position(n_p)
        return self.dropout(tokens)


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
    """Self-attn on [P; G_en]; optional G_en←exo cross-attn (query is the single G_en)."""

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

    def forward(self, tokens: torch.Tensor, exo: torch.Tensor | None) -> torch.Tensor:
        # tokens: [B, N+1, D]; last token is G_en
        tokens = self.norm1(tokens + self.dropout(self.self_attn(tokens, tokens, tokens)))
        patches, g_en = tokens[:, :-1, :], tokens[:, -1:, :]
        if exo is not None and exo.numel() > 0 and exo.size(1) > 0:
            if g_en.size(1) != 1:
                raise AssertionError(f"cross-attn query must be [B,1,D], got {tuple(g_en.shape)}")
            g_en = self.norm2(g_en + self.dropout(self.cross_attn(g_en, exo, exo)))
        else:
            g_en = self.norm2(g_en)
        y = torch.cat([patches, g_en], dim=1)
        ff = self.dropout(F.gelu(self.conv1(y.transpose(1, 2))))
        ff = self.dropout(self.conv2(ff).transpose(1, 2))
        return self.norm3(y + ff)


class TimeXerBackbone(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        patch_len: int,
        patch_stride: int,
        dropout: float,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            n_features, d_model, patch_len, patch_stride, dropout
        )
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        d_ff = d_ff or 4 * d_model
        self.layers = nn.ModuleList(
            [TimeXerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )

    def embed(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self.patch_embed(x)
        g_en = self.glb_token.expand(patches.size(0), -1, -1)
        return patches, g_en

    def encode(
        self, patches: torch.Tensor, g_en: torch.Tensor, exo: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([patches, g_en], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, exo)
        return tokens[:, :-1, :], tokens[:, -1:, :]

    def forward(self, x: torch.Tensor, exo: torch.Tensor | None = None) -> torch.Tensor:
        patches, g_en = self.embed(x)
        enc_p, _g = self.encode(patches, g_en, exo)
        return enc_p
