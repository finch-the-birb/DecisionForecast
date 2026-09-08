"""TimeXer backbone (endogenous patches + G_en + optional exo cross-attn).

Adapted from thuml/Time-Series-Library models/TimeXer.py for univariate target
(MS): the target channel is patched; other channels are inverted variate tokens.
Text is not fused here — Model B does late fusion in timexl_integration.py.
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
    """Patch the target variate and append a learnable global token G_en."""

    def __init__(self, d_model: int, patch_len: int, patch_stride: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def n_patches(self, seq_len: int) -> int:
        if seq_len < self.patch_len:
            raise ValueError(f"seq_len={seq_len} < patch_len={self.patch_len}")
        return (seq_len - self.patch_len) // self.patch_stride + 1

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # y: [B, T] endogenous target
        patches = y.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        tokens = self.value_embedding(patches) + self.position_embedding(patches)
        tokens = self.dropout(tokens)
        g_en = self.glb_token.expand(y.size(0), -1, -1)
        return tokens, g_en


class InvertedVariateEmbed(nn.Module):
    """Map each exogenous channel (length T) to one variate token [B, C, D]."""

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        return self.dropout(self.proj(x.transpose(1, 2)))


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
    """Self-attn on [P; G_en], then G_en cross-attn to exogenous variate tokens."""

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
        tokens = self.norm1(tokens + self.dropout(self.self_attn(tokens, tokens, tokens)))
        patches, g_en = tokens[:, :-1, :], tokens[:, -1:, :]
        if exo is not None and exo.numel() > 0 and exo.size(1) > 0:
            g_en = self.norm2(g_en + self.dropout(self.cross_attn(g_en, exo, exo)))
        else:
            g_en = self.norm2(g_en)
        y = torch.cat([patches, g_en], dim=1)
        ff = self.dropout(F.gelu(self.conv1(y.transpose(1, 2))))
        ff = self.dropout(self.conv2(ff).transpose(1, 2))
        return self.norm3(y + ff)


class TimeXerBackbone(nn.Module):
    """PatchEmbed → (caller may inject prototypes) → TimeXer encoder."""

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
        self.patch_embed = EndogenousPatchEmbed(d_model, patch_len, patch_stride, dropout)
        n_exo = n_features - 1
        self.exo_embed: InvertedVariateEmbed | None
        if n_exo > 0:
            self.exo_embed = InvertedVariateEmbed(seq_len, d_model, dropout)
        else:
            self.exo_embed = None
        d_ff = d_ff or 4 * d_model
        self.layers = nn.ModuleList(
            [TimeXerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )

    def embed(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        # x: [B, T, C]
        stats: dict[str, torch.Tensor] = {}
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev
            stats["means"] = means
            stats["stdev"] = stdev
        target = x[:, :, self.target_idx]
        patches, g_en = self.patch_embed(target)
        exo = None
        if self.exo_embed is not None:
            exo_idx = [i for i in range(self.n_features) if i != self.target_idx]
            exo = self.exo_embed(x[:, :, exo_idx])
        return patches, g_en, exo, stats

    def encode(
        self, patches: torch.Tensor, g_en: torch.Tensor, exo: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.cat([patches, g_en], dim=1)
        for layer in self.layers:
            tokens = layer(tokens, exo)
        return tokens[:, :-1, :], tokens[:, -1:, :]
