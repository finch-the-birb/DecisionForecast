"""TimeXer on the frozen 40D fold vector: 25D patches, 15D text through G_en.

Phase 4, Sprint 12. Technical channels are ordinary endogenous patches.
Compact text is the only exogenous input: G_en reads it by cross-attention,
then the patches read that updated G_en (global-to-patch) in the same layer.
The forecast head pools the last patch. Raw text is not concatenated onto
patches or onto the head. There is no TimeXL prototype bank; event prototypes
already live inside the 15D text block.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.fusion import assert_fusion
from src.models.head import ForecastHead
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.timexer_backbone import TimeXerBackbone, n_patches


class _GlobalToPatch(nn.Module):
    """Patches query G_en after G_en has read the text tokens."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor, g_en: torch.Tensor) -> torch.Tensor:
        mixed, _weights = self.attn(patches, g_en, g_en, need_weights=False)
        return self.norm(patches + self.dropout(mixed))


class TimeXerSelected(nn.Module):
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
        d_ff: int | None = None,
        head_type: str = "linear",
        head_hidden: int = 128,
        head_dropout: float = 0.0,
        head_pool: str = "last",
    ) -> None:
        super().__init__()
        if e_layers < 1:
            raise ValueError("e_layers must be >= 1")
        self.fusion = assert_fusion(
            fusion,
            kind="selected_40d",
            text_at_head=False,
            text_to_patches=False,
            text_as_exogenous=True,
            global_to_patch=True,
        )
        self.n_features = int(n_features)
        self.text_dim = int(text_dim)
        self.backbone = TimeXerBackbone(
            n_features=self.n_features,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            patch_len=patch_len,
            patch_stride=patch_stride,
            dropout=dropout,
            d_ff=d_ff,
        )
        self.exo_embed = nn.Linear(self.text_dim, d_model)
        self.global_to_patch = nn.ModuleList(
            [_GlobalToPatch(d_model, n_heads, dropout) for _ in range(e_layers)]
        )
        self.n_patches = n_patches(seq_len, patch_len, patch_stride)
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

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor | None = None,
        text_seq: torch.Tensor | None = None,
        **_kwargs,
    ) -> ModelOutput:
        del _kwargs
        exo_in = _exo_sequence(text, text_seq, self.text_dim)
        if x.size(-1) != self.n_features:
            raise ValueError(f"expected {self.n_features} technical channels, got {x.size(-1)}")
        patches, g_en = self.backbone.embed(x)
        exo = self.exo_embed(exo_in)
        tokens = torch.cat([patches, g_en], dim=1)
        for layer, bridge in zip(self.backbone.layers, self.global_to_patch, strict=True):
            tokens = layer(tokens, exo)
            patches = bridge(tokens[:, :-1, :], tokens[:, -1:, :])
            tokens = torch.cat([patches, tokens[:, -1:, :]], dim=1)
        g_en = tokens[:, -1:, :]
        pred = self.head(patches, g_en=g_en)
        return ModelOutput(pred=pred, proto_losses=None, segments=patches)

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
        raise ValueError("TimeXerSelected requires text [B, text_dim] or text_seq [B, L, text_dim]")
    if exo.dim() != 3 or exo.size(-1) != text_dim:
        raise ValueError(f"text exo must have shape [B, L, {text_dim}], got {tuple(exo.shape)}")
    return exo
