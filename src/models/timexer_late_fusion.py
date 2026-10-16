"""Late residual fusion: the encoder sees prices, rFFT, and text only.

The attention block matches compact TimeXer, with patch input ``5 * P + 6``.
After the encoder pools the patch tokens, a two-layer MLP reads the 25
indicators on the last bar. ``LayerNorm(Z_patch + Z_tech)`` is the head input
and, when prototypes are on, the tensor passed to ``PrototypeResidual``.
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
from src.models.timexer_backbone import (
    TimeXerEncoderLayer,
    _PositionalEmbedding,
    n_patches,
    unfold_time,
)
from src.models.timexer_selected import _GlobalToPatch
from src.models.timexl_integration import PrototypeResidual

_MLP_HIDDEN = 64


class TimeXerLateFusion(nn.Module):
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
            raise ValueError(f"c1_late_fusion expects 5 OHLCV channels, got {n_features}")
        if int(n_ts_features) != 25:
            raise ValueError(f"c1_late_fusion expects 25 exogenous indicators, got {n_ts_features}")
        if text_dim != 15:
            raise ValueError(f"c1_late_fusion expects text_dim 15, got {text_dim}")
        if head_pool not in {"last", "mean", "global"}:
            raise ValueError(f"c1_late_fusion pool must be last, mean, or global, got {head_pool!r}")
        if not 0 <= int(close_idx) < int(n_features):
            raise ValueError("close_idx must point at one endogenous channel")
        self.fusion = assert_fusion(
            fusion,
            kind="c1_late_fusion",
            text_at_head=False,
            text_to_patches=False,
            text_as_exogenous=True,
            global_to_patch=True,
        )
        self.n_features = int(n_features)
        self.text_dim = 15
        self.n_ts_features = 25
        self.close_idx = int(close_idx)
        self.use_prototypes = bool(use_prototypes)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.head_pool = str(head_pool)
        self.n_patches = n_patches(seq_len, self.patch_len, self.patch_stride)
        d_model = int(d_model)
        d_ff = 4 * d_model if d_ff is None else int(d_ff)
        self.patch_proj = nn.Linear(self.n_features * self.patch_len + 6, d_model)
        self.fft = PatchFFT()
        self.position = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.glb_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.exo_embed = nn.Linear(self.text_dim, d_model)
        self.layers = nn.ModuleList(
            [TimeXerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )
        self.global_to_patch = nn.ModuleList(
            [_GlobalToPatch(d_model, n_heads, dropout) for _ in range(e_layers)]
        )
        self.tech_mlp = nn.Sequential(
            nn.Linear(self.n_ts_features, _MLP_HIDDEN),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(_MLP_HIDDEN, d_model),
        )
        self.fuse_norm = nn.LayerNorm(d_model)
        if self.use_prototypes:
            if n_prototypes < 1:
                raise ValueError("n_prototypes must be >= 1 when use_prototypes is true")
            self.prototype_block = PrototypeResidual(int(n_prototypes), d_model, float(d_min))
        self.head = ForecastHead(
            d_model=d_model,
            horizon=horizon,
            pool=self.head_pool,
            n_patches=self.n_patches,
            head_type=head_type,
            head_hidden=head_hidden,
            dropout=head_dropout,
            extra_dim=0,
        )

    @property
    def proto(self):
        if not self.use_prototypes:
            raise AttributeError("c1_late_fusion has no prototype bank when use_prototypes is false")
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
        patches = self._embed_patches(x)
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
        z_patch = _pool_patches(patches, g_en, self.head_pool)
        z_tech = self.tech_mlp(indicators[:, -1, :])
        z_final = self.fuse_norm(z_patch + z_tech).unsqueeze(1)
        bank = z_final
        proto_losses: PrototypeLosses | None = None
        if self.use_prototypes:
            z_final, proto_losses = self.prototype_block(
                z_final, proto_mode=proto_mode, ablation_generator=ablation_generator
            )
        pred = self.head(z_final, g_en=z_final)
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
        tokens = self.patch_proj(torch.cat([flat, freq], dim=-1))
        tokens = tokens + self.position(tokens.size(1))
        return self.dropout(tokens)


def _pool_patches(patches: torch.Tensor, g_en: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "last":
        return patches[:, -1, :]
    if pool == "mean":
        return patches.mean(dim=1)
    if pool == "global":
        return g_en[:, 0, :]
    raise ValueError(f"c1_late_fusion pool must be last, mean, or global, got {pool!r}")


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
        raise ValueError("c1_late_fusion requires text [B, text_dim] or text_seq [B, L, text_dim]")
    if exo.dim() != 3 or exo.size(-1) != text_dim:
        raise ValueError(f"text exo must have shape [B, L, {text_dim}], got {tuple(exo.shape)}")
    return exo


def _as_ts(ts: torch.Tensor | None, n_ts: int, time: int) -> torch.Tensor:
    if ts is None or ts.dim() != 3 or ts.size(1) != time or ts.size(-1) != n_ts:
        got = None if ts is None else tuple(ts.shape)
        raise ValueError(f"c1_late_fusion requires ts [B, {time}, {n_ts}], got {got}")
    return ts
