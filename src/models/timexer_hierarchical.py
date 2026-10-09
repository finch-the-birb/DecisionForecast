"""Two-layer TimeXer: indicators first, then text.

Layer 1 reads only ``G_ts``. Layer 2 reads only ``G_text`` and the patches that
left layer 1. The tokens never share a sequence, so no cross-token mask is
used. Inside a layer the exogenous cross-attention updates the global token
before self-attention, so the patches can read that modality while the bridge
gate is still closed.
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


class _GatedTokenBridge(nn.Module):
    """Patches query one global token. ``alpha`` starts at 0, so the residual is closed."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, patches: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        delta, _weights = self.attn(patches, token, token, need_weights=False)
        return self.norm(patches + torch.tanh(self.alpha) * self.dropout(delta))


class _ModalityEncoderLayer(nn.Module):
    """One modality: cross-attention updates ``g``, then self-attention mixes it into the patches.

    The bridge gate starts closed, so the exogenous sequence has to reach the
    patches through this self-attention. Cross-attention therefore runs before
    the tokens are concatenated.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.bridge = _GatedTokenBridge(d_model, n_heads, dropout)

    def forward(
        self,
        patches: torch.Tensor,
        token: torch.Tensor,
        exo: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta, _weights = self.cross(token, exo, exo, need_weights=False)
        token = self.norm_cross(token + self.dropout(delta))
        tokens = torch.cat([patches, token], dim=1)
        mixed, _weights = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm_attn(tokens + self.dropout(mixed))
        tokens = self.norm_ff(tokens + self.ff(tokens))
        patches, token = tokens[:, :-1, :], tokens[:, -1:, :]
        patches = self.bridge(patches, token)
        return patches, token


class TimeXerHierarchical(nn.Module):
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
        if int(e_layers) != 2:
            raise ValueError(
                "TimeXerHierarchical is exactly two layers (indicators, then text), "
                f"not a stack of identical dual layers; got e_layers={e_layers}"
            )
        if int(n_features) not in (1, 2, 5):
            raise ValueError(
                f"TimeXerHierarchical expects 1, 2, or 5 endogenous channels, got {n_features}"
            )
        if int(text_dim) != 15:
            raise ValueError(f"TimeXerHierarchical expects text_dim 15, got {text_dim}")
        if not 0 <= int(close_idx) < int(n_features):
            raise ValueError("close_idx must point at one endogenous channel")
        if int(n_ts_features) < 1:
            raise ValueError("TimeXerHierarchical requires n_ts_features > 0")
        self.fusion = assert_fusion(
            fusion,
            kind="c1_hierarchical",
            text_at_head=False,
            text_to_patches=False,
            text_as_exogenous=True,
            global_to_patch=True,
        )
        self.n_features = int(n_features)
        self.text_dim = 15
        self.n_ts_features = int(n_ts_features)
        self.close_idx = int(close_idx)
        self.use_prototypes = bool(use_prototypes)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.n_patches = n_patches(seq_len, self.patch_len, self.patch_stride)
        d_ff = int(d_ff) if d_ff is not None else 4 * int(d_model)
        self.patch_proj = nn.Linear(self.n_features * self.patch_len + 6, d_model)
        self.fft = PatchFFT()
        self.g_text = nn.Linear(self.text_dim, d_model)
        self.g_ts = nn.Linear(self.n_ts_features, d_model)
        self.text_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ts_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layer_ts = _ModalityEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.layer_text = _ModalityEncoderLayer(d_model, n_heads, d_ff, dropout)
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
            raise AttributeError("TimeXerHierarchical has no prototype bank when use_prototypes is false")
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
            raise ValueError(f"expected {self.n_features} endogenous channels, got {x.size(-1)}")
        if ts is None:
            raise ValueError(
                f"TimeXerHierarchical with n_ts_features={self.n_ts_features} requires ts "
                f"[B, T, {self.n_ts_features}] or [B, {self.n_ts_features}]"
            )
        patches = self._embed_patches(x)
        text_in = _exo_sequence(text, text_seq, self.text_dim)
        if text_mode != "none":
            text_in = apply_feature_ablation(text_in, text_mode, ablation_generator)
        text_exo = self.g_text(text_in)
        ts_exo = self._project_ts(ts)
        g_ts = self.ts_token.expand(x.size(0), -1, -1)
        g_text = self.text_token.expand(x.size(0), -1, -1)
        patches, g_ts = self.layer_ts(patches, g_ts, ts_exo)
        patches, g_text = self.layer_text(patches, g_text, text_exo)
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

    def _project_ts(self, ts: torch.Tensor) -> torch.Tensor:
        if ts.dim() == 2:
            ts = ts.unsqueeze(1)
        if ts.dim() != 3 or ts.size(-1) != self.n_ts_features:
            raise ValueError(
                f"ts must be [B, F] or [B, T, F] with F={self.n_ts_features}, got {tuple(ts.shape)}"
            )
        return self.g_ts(ts)


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
        raise ValueError("TimeXerHierarchical requires text [B, 15] or text_seq [B, L, 15]")
    if exo.dim() != 3 or exo.size(-1) != text_dim:
        raise ValueError(f"text exo must have shape [B, L, {text_dim}], got {tuple(exo.shape)}")
    return exo
