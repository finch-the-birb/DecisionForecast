"""Model C1: TimeXer + G1+G2 + G_en cross-attn to text exo tokens.

Patches are not modified by text. Cross-attn query is the single G_en [B,1,d].
Head has no text concat. Default exo_tokens=per_day.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.fusion import assert_fusion
from src.models.head import ForecastHead
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.timexer_backbone import TimeXerBackbone, days_to_patches, n_patches
from src.models.timexl_integration import PrototypeResidual


class TimeXerC1(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        d_model: int,
        n_prototypes: int,
        d_min: float,
        n_heads: int,
        e_layers: int,
        patch_len: int,
        patch_stride: int,
        dropout: float,
        text_dim: int,
        head_hidden: int,
        fusion: Any,
        d_ff: int | None = None,
        head_type: str = "linear",
        head_dropout: float = 0.0,
        head_pool: str = "mean",
    ) -> None:
        super().__init__()
        self.fusion = assert_fusion(
            fusion,
            kind="mid_cross_attn",
            text_at_head=False,
            text_to_patches=False,
            text_as_exogenous=True,
            exo_tokens={"per_day", "per_patch"},
        )
        self.exo_tokens = str(self.fusion["exo_tokens"])
        self.backbone = TimeXerBackbone(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            patch_len=patch_len,
            patch_stride=patch_stride,
            dropout=dropout,
            d_ff=d_ff,
        )
        self.g12 = PrototypeResidual(n_prototypes, d_model, d_min)
        self.exo_embed = nn.Linear(text_dim, d_model)
        n_p = n_patches(seq_len, patch_len, patch_stride)
        self.head = ForecastHead(
            d_model=d_model,
            horizon=horizon,
            pool=head_pool,
            n_patches=n_p,
            head_type=head_type,
            head_hidden=head_hidden,
            dropout=head_dropout,
            extra_dim=0,
        )

    @property
    def proto(self):
        return self.g12.proto

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_seq: torch.Tensor | None = None,
        proto_mode: str = "none",
        text_mode: str = "none",
        ablation_generator: torch.Generator | None = None,
    ) -> ModelOutput:
        del text
        if text_seq is None:
            raise ValueError("C1 requires text_seq [B,T,text_dim]")
        patches, g_en = self.backbone.embed(x)
        bank = patches
        patches, proto_losses = self.g12(patches, proto_mode, ablation_generator)
        embed = self.backbone.patch_embed
        days = (
            text_seq
            if self.exo_tokens == "per_day"
            else days_to_patches(text_seq, embed.patch_len, embed.patch_stride)
        )
        exo = apply_feature_ablation(
            self.exo_embed(days), text_mode, ablation_generator
        )
        enc_p, _g = self.backbone.encode(patches, g_en, exo=exo)
        pred = self.head(enc_p, g_en=_g)
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
