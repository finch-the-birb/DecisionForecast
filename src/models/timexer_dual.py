"""Dual-token TimeXer: OHLCV patches plus patch rFFT, text and indicators isolated.

Patches are ``Linear(5 * patch_len + 6, d_model)``. ``G_ts`` and ``G_text`` sit
after the patches. Self-attention lets patches see both tokens and blocks the
tokens from attending to each other. Each token then reads only its own
exogenous sequence. A zero-initialized gate writes both tokens back into the
patches.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.fft_patch import PatchFFT
from src.models.fusion import assert_fusion
from src.models.head import ForecastHead
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.prototypes import PrototypeLosses
from src.models.timexer_backbone import n_patches, unfold_time
from src.models.timexl_integration import PrototypeResidual


def dual_attention_mask(n_tokens_patches: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask ``[K+2, K+2]``. ``G_ts`` is index ``K``, ``G_text`` is ``K+1``."""
    width = int(n_tokens_patches) + 2
    mask = torch.zeros(width, width, device=device, dtype=dtype)
    ts_idx = int(n_tokens_patches)
    text_idx = ts_idx + 1
    mask[ts_idx, text_idx] = float("-inf")
    mask[text_idx, ts_idx] = float("-inf")
    return mask


class _GatedGlobalToPatch(nn.Module):
    """Patches query ``G_ts`` and ``G_text`` through separate attention modules.

    ``alpha_*`` start at 0, so ``tanh(alpha)`` is 0 and the residual is closed.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_ts = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha_ts = nn.Parameter(torch.zeros(()))
        self.alpha_text = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        patches: torch.Tensor,
        g_ts: torch.Tensor,
        g_text: torch.Tensor,
    ) -> torch.Tensor:
        delta_ts, _weights = self.attn_ts(patches, g_ts, g_ts, need_weights=False)
        delta_text, _weights = self.attn_text(patches, g_text, g_text, need_weights=False)
        mixed = torch.tanh(self.alpha_ts) * self.dropout(delta_ts)
        mixed = mixed + torch.tanh(self.alpha_text) * self.dropout(delta_text)
        return self.norm(patches + mixed)


class _DualEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d_model)
        self.cross_ts = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ts = nn.LayerNorm(d_model)
        self.norm_text = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.bridge = _GatedGlobalToPatch(d_model, n_heads, dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        text_exo: torch.Tensor | None,
        ts_exo: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mixed, _weights = self.self_attn(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
        tokens = self.norm_attn(tokens + self.dropout(mixed))
        patches, g_ts, g_text = _split_tokens(tokens)
        if ts_exo is not None and ts_exo.size(1) > 0:
            delta, _weights = self.cross_ts(g_ts, ts_exo, ts_exo, need_weights=False)
            g_ts = self.norm_ts(g_ts + self.dropout(delta))
        if text_exo is not None and text_exo.size(1) > 0:
            delta, _weights = self.cross_text(g_text, text_exo, text_exo, need_weights=False)
            g_text = self.norm_text(g_text + self.dropout(delta))
        tokens = torch.cat([patches, g_ts, g_text], dim=1)
        tokens = self.norm_ff(tokens + self.ff(tokens))
        patches, g_ts, g_text = _split_tokens(tokens)
        patches = self.bridge(patches, g_ts, g_text)
        return torch.cat([patches, g_ts, g_text], dim=1)


class TimeXerDual(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        patch_len: int,
        patch_stride: int,
        dropout: float,
        text_dim: int,
        fusion: Any,
        n_ts_features: int = 0,
        close_idx: int = 0,
        use_prototypes: bool = False,
        n_prototypes: int = 10,
        d_min: float = 0.5,
        d_ff: int | None = None,
        head_type: str = "linear",
        head_hidden: int = 128,
        head_dropout: float = 0.0,
        head_pool: str = "last",
    ) -> None:
        super().__init__()
        if e_layers < 1:
            raise ValueError("e_layers must be >= 1")
        if n_features != 5:
            raise ValueError(f"TimeXerDual expects 5 OHLCV channels, got {n_features}")
        if text_dim != 15:
            raise ValueError(f"TimeXerDual expects text_dim 15, got {text_dim}")
        if not 0 <= int(close_idx) < 5:
            raise ValueError("close_idx must point at one of the 5 OHLCV channels")
        self.fusion = assert_fusion(
            fusion,
            kind="c1_dual",
            text_at_head=False,
            text_to_patches=False,
            text_as_exogenous=True,
            global_to_patch=True,
        )
        self.n_features = 5
        self.text_dim = 15
        self.n_ts_features = int(n_ts_features)
        self.close_idx = int(close_idx)
        self.use_prototypes = bool(use_prototypes)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.n_patches = n_patches(seq_len, self.patch_len, self.patch_stride)
        d_ff = int(d_ff) if d_ff is not None else 4 * int(d_model)
        patch_in = 5 * self.patch_len + 6
        self.patch_proj = nn.Linear(patch_in, d_model)
        self.fft = PatchFFT()
        self.g_text = nn.Linear(self.text_dim, d_model)
        self.g_ts = nn.Linear(self.n_ts_features, d_model) if self.n_ts_features > 0 else None
        self.ts_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList(
            [_DualEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        if self.use_prototypes:
            if n_prototypes < 1:
                raise ValueError("n_prototypes must be >= 1 when use_prototypes is true")
            self.prototype_block = PrototypeResidual(int(n_prototypes), d_model, float(d_min))
        self.head = ForecastHead(
            d_model=d_model,
            horizon=horizon,
            pool=head_pool,
            n_patches=self.n_patches,
            head_type=head_type,
            head_hidden=head_hidden,
            dropout=head_dropout,
            extra_dim=0,
        )

    @property
    def proto(self):
        if not self.use_prototypes:
            raise AttributeError("TimeXerDual has no prototype bank when use_prototypes is false")
        return self.prototype_block.proto

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor | None = None,
        text_seq: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
        **_kwargs,
    ) -> ModelOutput:
        del _kwargs
        if x.size(-1) != self.n_features:
            raise ValueError(f"expected {self.n_features} OHLCV channels, got {x.size(-1)}")
        patches = self._embed_patches(x)
        text_exo = self.g_text(_exo_sequence(text, text_seq, self.text_dim))
        g_text = text_exo.mean(dim=1, keepdim=True)
        ts_exo = self._project_ts(ts, x.size(0), x.device, x.dtype)
        g_ts = ts_exo.mean(dim=1, keepdim=True) if ts_exo is not None else self.ts_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([patches, g_ts, g_text], dim=1)
        mask = dual_attention_mask(self.n_patches, tokens.device, tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens, text_exo, ts_exo, mask)
        patches, g_ts, g_text = _split_tokens(tokens)
        proto_losses: PrototypeLosses | None = None
        if self.use_prototypes:
            patches, proto_losses = self.prototype_block(patches)
        pred = self.head(patches, g_en=g_text)
        return ModelOutput(pred=pred, proto_losses=proto_losses, segments=patches)

    def compute_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        lambda_c: float,
        lambda_e: float,
        lambda_d: float,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_pred_loss(output, target, lambda_c, lambda_e, lambda_d, **kwargs)

    def _embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        unfolded = unfold_time(x, self.patch_len, self.patch_stride)
        flat = unfolded.reshape(unfolded.size(0), unfolded.size(1), -1)
        patch_close = unfolded[:, :, self.close_idx, :]
        freq = self.fft(patch_close)
        return self.patch_proj(torch.cat([flat, freq], dim=-1))

    def _project_ts(
        self,
        ts: torch.Tensor | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        del batch, device, dtype
        if ts is None or self.g_ts is None:
            return None
        if ts.dim() == 2:
            ts = ts.unsqueeze(1)
        if ts.dim() != 3 or ts.size(-1) != self.n_ts_features:
            raise ValueError(
                f"ts must be [B, F] or [B, T, F] with F={self.n_ts_features}, got {tuple(ts.shape)}"
            )
        return self.g_ts(ts)


def _split_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tokens[:, :-2, :], tokens[:, -2:-1, :], tokens[:, -1:, :]


def _exo_sequence(
    text: torch.Tensor | None,
    text_seq: torch.Tensor | None,
    text_dim: int,
) -> torch.Tensor:
    if text_seq is not None:
        exo = text_seq
    elif text is not None:
        exo = text.unsqueeze(1) if text.dim() == 2 else text
    else:
        raise ValueError("TimeXerDual requires text [B, 15] or text_seq [B, L, 15]")
    if exo.dim() != 3 or exo.size(-1) != text_dim:
        raise ValueError(f"text exo must have shape [B, L, {text_dim}], got {tuple(exo.shape)}")
    return exo
