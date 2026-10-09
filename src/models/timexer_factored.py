"""Factored PatchEmbed: price, indicators, and rFFT keep separate coordinates.

OHLCV maps ``60 → 36``, the first eight exogenous indicators map ``96 → 16``,
and six close-price rFFT bins map ``6 → 12``. Those widths sum to ``d_model``
64. Compact text still enters through one ``G_en`` and ``_GlobalToPatch``.
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
from src.models.timexer_backbone import TimeXerEncoderLayer, n_patches, unfold_time
from src.models.timexer_selected import _GlobalToPatch
from src.models.timexl_integration import PrototypeResidual

_OHLCV_DIM = 36
_IND_DIM = 16
_FFT_DIM = 12
_N_PATCH_INDICATORS = 8


class TimeXerFactored(nn.Module):
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
        n_patch_indicators: int = _N_PATCH_INDICATORS,
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
        if int(d_model) != _OHLCV_DIM + _IND_DIM + _FFT_DIM:
            raise ValueError(
                "c1_factored requires d_model 64 so the 36/16/12 projections sum to the token, "
                f"got {d_model}"
            )
        if int(n_features) != 5:
            raise ValueError(f"c1_factored expects 5 OHLCV channels, got {n_features}")
        if int(patch_len) != 12:
            raise ValueError(f"c1_factored expects patch_len 12, got {patch_len}")
        if int(n_patch_indicators) != _N_PATCH_INDICATORS:
            raise ValueError(
                f"c1_factored expects {_N_PATCH_INDICATORS} patch indicators, got {n_patch_indicators}"
            )
        if int(n_ts_features) < _N_PATCH_INDICATORS:
            raise ValueError(
                f"c1_factored needs at least {_N_PATCH_INDICATORS} exogenous indicators, "
                f"got {n_ts_features}"
            )
        if text_dim != 15:
            raise ValueError(f"c1_factored expects text_dim 15, got {text_dim}")
        if not 0 <= int(close_idx) < int(n_features):
            raise ValueError("close_idx must point at one endogenous channel")
        self.fusion = assert_fusion(
            fusion,
            kind="c1_factored",
            text_at_head=False,
            text_to_patches=False,
            text_as_exogenous=True,
            global_to_patch=True,
        )
        self.n_features = int(n_features)
        self.text_dim = 15
        self.n_ts_features = int(n_ts_features)
        self.n_patch_indicators = _N_PATCH_INDICATORS
        self.close_idx = int(close_idx)
        self.use_prototypes = bool(use_prototypes)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.n_patches = n_patches(seq_len, self.patch_len, self.patch_stride)
        d_ff = 4 * int(d_model) if d_ff is None else int(d_ff)
        self.ohlcv_proj = nn.Linear(self.n_features * self.patch_len, _OHLCV_DIM)
        self.ind_proj = nn.Linear(self.n_patch_indicators * self.patch_len, _IND_DIM)
        self.fft_proj = nn.Linear(6, _FFT_DIM)
        self.fft = PatchFFT()
        self.dropout = nn.Dropout(dropout)
        self.glb_token = nn.Parameter(torch.randn(1, 1, int(d_model)) * 0.02)
        self.exo_embed = nn.Linear(self.text_dim, int(d_model))
        self.layers = nn.ModuleList(
            [TimeXerEncoderLayer(int(d_model), n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.global_to_patch = nn.ModuleList(
            [_GlobalToPatch(int(d_model), n_heads, dropout) for _ in range(e_layers)]
        )
        if self.use_prototypes:
            if n_prototypes < 1:
                raise ValueError("n_prototypes must be >= 1 when use_prototypes is true")
            self.prototype_block = PrototypeResidual(int(n_prototypes), int(d_model), float(d_min))
        self.head = ForecastHead(
            d_model=int(d_model),
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
            raise AttributeError("c1_factored has no prototype bank when use_prototypes is false")
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
        if x.size(-1) != self.n_features:
            raise ValueError(f"expected {self.n_features} OHLCV channels, got {x.size(-1)}")
        indicators = _as_ts(ts, self.n_ts_features, x.size(1))
        text_in = _as_text(text, text_seq, self.text_dim)
        if text_mode != "none":
            text_in = apply_feature_ablation(text_in, text_mode, ablation_generator)
        patches = self._embed_patches(x, indicators)
        if patches.size(1) != self.n_patches:
            raise ValueError(
                f"expected {self.n_patches} patches, got {patches.size(1)} from x {tuple(x.shape)}"
            )
        g_en = self.glb_token.expand(x.size(0), -1, -1)
        exo = self.exo_embed(text_in)
        tokens = torch.cat([patches, g_en], dim=1)
        for layer, bridge in zip(self.layers, self.global_to_patch, strict=True):
            tokens = layer(tokens, exo)
            patches = bridge(tokens[:, :-1, :], tokens[:, -1:, :])
            tokens = torch.cat([patches, tokens[:, -1:, :]], dim=1)
        g_en = tokens[:, -1:, :]
        bank = patches
        proto_losses: PrototypeLosses | None = None
        if self.use_prototypes:
            patches, proto_losses = self.prototype_block(
                patches, proto_mode=proto_mode, ablation_generator=ablation_generator
            )
        pred = self.head(patches, g_en=g_en)
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

    def _embed_patches(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        unfolded = unfold_time(x, self.patch_len, self.patch_stride)
        ohlcv = unfolded.reshape(unfolded.size(0), unfolded.size(1), -1)
        freq = self.fft(unfolded[:, :, self.close_idx, :])
        ind = unfold_time(ts[:, :, : self.n_patch_indicators], self.patch_len, self.patch_stride)
        ind_flat = ind.reshape(ind.size(0), ind.size(1), -1)
        tokens = torch.cat(
            [self.ohlcv_proj(ohlcv), self.ind_proj(ind_flat), self.fft_proj(freq)],
            dim=-1,
        )
        return self.dropout(tokens)


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
        raise ValueError("c1_factored requires text [B, text_dim] or text_seq [B, L, text_dim]")
    if exo.dim() != 3 or exo.size(-1) != text_dim:
        raise ValueError(f"text exo must have shape [B, L, {text_dim}], got {tuple(exo.shape)}")
    return exo


def _as_ts(ts: torch.Tensor | None, n_ts: int, time: int) -> torch.Tensor:
    if ts is None or ts.dim() != 3 or ts.size(1) != time or ts.size(-1) != n_ts:
        got = None if ts is None else tuple(ts.shape)
        raise ValueError(f"c1_factored requires ts [B, {time}, {n_ts}], got {got}")
    return ts
