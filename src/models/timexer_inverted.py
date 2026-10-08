"""Inverted variate tokens: indicators stay whole-window series, not patches.

Endogenous patches are 5 OHLCV channels plus 6 close-price rFFT bins.
Each of the 25 exogenous series over ``T`` is one token, ``Linear(T, d_model)``.
Patches cross-attend to those tokens. ``G_text`` cross-attends to 15D text and
shares self-attention with the patches.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.fft_patch import PatchFFT
from src.models.fusion import assert_fusion
from src.models.head import ForecastHead
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.prototypes import PrototypeLosses
from src.models.timexer_backbone import n_patches, unfold_time
from src.models.timexl_integration import PrototypeResidual


class _InvertedLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.cross_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_text = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d_model)
        self.cross_var = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_var = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        patches: torch.Tensor,
        g_text: torch.Tensor,
        text_exo: torch.Tensor,
        variates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta, _weights = self.cross_text(g_text, text_exo, text_exo, need_weights=False)
        g_text = self.norm_text(g_text + self.dropout(delta))
        tokens = torch.cat([patches, g_text], dim=1)
        mixed, _weights = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm_attn(tokens + self.dropout(mixed))
        patches, g_text = tokens[:, :-1, :], tokens[:, -1:, :]
        delta, _weights = self.cross_var(patches, variates, variates, need_weights=False)
        patches = self.norm_var(patches + self.dropout(delta))
        tokens = torch.cat([patches, g_text], dim=1)
        tokens = self.norm_ff(tokens + self.ff(tokens))
        return tokens[:, :-1, :], tokens[:, -1:, :]


class TimeXerInverted(nn.Module):
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
        n_ts_features: int = 25,
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
        if int(n_features) != 5:
            raise ValueError(f"c1_inverted expects 5 OHLCV channels, got {n_features}")
        if int(n_ts_features) != 25:
            raise ValueError(f"c1_inverted expects 25 exogenous indicators, got {n_ts_features}")
        if text_dim != 15:
            raise ValueError(f"c1_inverted expects text_dim 15, got {text_dim}")
        if int(seq_len) < int(patch_len):
            raise ValueError(f"seq_len={seq_len} < patch_len={patch_len}")
        if not 0 <= int(close_idx) < int(n_features):
            raise ValueError("close_idx must point at one endogenous channel")
        self.fusion = assert_fusion(
            fusion,
            kind="c1_inverted",
            text_at_head=False,
            text_to_patches=False,
            text_as_exogenous=True,
            global_to_patch=True,
        )
        self.n_features = int(n_features)
        self.text_dim = 15
        self.n_ts_features = 25
        self.seq_len = int(seq_len)
        self.close_idx = int(close_idx)
        self.use_prototypes = bool(use_prototypes)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.n_patches = n_patches(self.seq_len, self.patch_len, self.patch_stride)
        d_model = int(d_model)
        d_ff = 4 * d_model if d_ff is None else int(d_ff)
        self.patch_proj = nn.Linear(self.n_features * self.patch_len + 6, d_model)
        self.fft = PatchFFT()
        self.variate_proj = nn.Linear(self.seq_len, d_model)
        self.g_text = nn.Linear(self.text_dim, d_model)
        self.layers = nn.ModuleList(
            [_InvertedLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
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
            raise AttributeError("c1_inverted has no prototype bank when use_prototypes is false")
        return self.prototype_block.proto

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor | None = None,
        text_seq: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
        proto_mode: str = "none",
        text_mode: str = "none",
        ablation_generator: torch.Generator | None = None,
        **_kwargs,
    ) -> ModelOutput:
        del _kwargs
        if x.size(-1) != self.n_features or x.size(1) != self.seq_len:
            raise ValueError(
                f"expected x [B, {self.seq_len}, {self.n_features}], got {tuple(x.shape)}"
            )
        indicators = _as_ts(ts, self.n_ts_features, self.seq_len)
        text_in = _as_text(text, text_seq, self.text_dim)
        if text_mode != "none":
            text_in = apply_feature_ablation(text_in, text_mode, ablation_generator)
        patches = self._embed_patches(x)
        variates = self.variate_proj(indicators.transpose(1, 2))
        text_exo = self.g_text(text_in)
        g_text = text_exo.mean(dim=1, keepdim=True)
        for layer in self.layers:
            patches, g_text = layer(patches, g_text, text_exo, variates)
        bank = patches
        proto_losses: PrototypeLosses | None = None
        if self.use_prototypes:
            patches, proto_losses = self.prototype_block(
                patches, proto_mode=proto_mode, ablation_generator=ablation_generator
            )
        pred = self.head(patches, g_en=g_text)
        return ModelOutput(pred=pred, proto_losses=proto_losses, segments=bank)

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
        freq = self.fft(unfolded[:, :, self.close_idx, :])
        return self.patch_proj(torch.cat([flat, freq], dim=-1))


def _as_text(
    text: torch.Tensor | None,
    text_seq: torch.Tensor | None,
    text_dim: int,
) -> torch.Tensor:
    if text_seq is not None:
        exo = text_seq
    elif text is not None:
        exo = text.unsqueeze(1) if text.dim() == 2 else text
    else:
        raise ValueError("c1_inverted requires text [B, text_dim] or text_seq [B, L, text_dim]")
    if exo.dim() != 3 or exo.size(-1) != text_dim:
        raise ValueError(f"text exo must have shape [B, L, {text_dim}], got {tuple(exo.shape)}")
    return exo


def _as_ts(ts: torch.Tensor | None, n_ts: int, time: int) -> torch.Tensor:
    if ts is None or ts.dim() != 3 or ts.size(1) != time or ts.size(-1) != n_ts:
        got = None if ts is None else tuple(ts.shape)
        raise ValueError(f"c1_inverted requires ts [B, {time}, {n_ts}], got {got}")
    return ts
