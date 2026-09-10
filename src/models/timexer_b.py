"""Model B: TimeXer + G1+G2 prototypes + late text fusion at the head.

Encoder is purely endogenous (``exo=None``). Pooled ``text`` is concatenated
at the head. ``text_seq`` is ignored. OHLCV channels are PatchEmbed features.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.fusion import assert_fusion
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.timexer_backbone import TimeXerBackbone
from src.models.timexl_integration import PrototypeResidual


class TimeXerB(nn.Module):
    def __init__(
        self,
        n_features: int,
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
        text_hidden: int,
        head_hidden: int,
        fusion: Any,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        self.fusion = assert_fusion(
            fusion,
            kind="late",
            text_at_head=True,
            text_to_patches=False,
            text_as_exogenous=False,
        )
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
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, text_hidden),
            nn.ReLU(),
            nn.Linear(text_hidden, d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, horizon),
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
        del text_seq  # late fusion uses pooled ``text`` only
        patches, g_en = self.backbone.embed(x)
        bank = patches
        patches, proto_losses = self.g12(patches, proto_mode, ablation_generator)
        enc_p, _g = self.backbone.encode(patches, g_en, exo=None)
        ts_repr = enc_p.mean(dim=1)
        text_repr = apply_feature_ablation(
            self.text_mlp(text), text_mode, ablation_generator
        )
        pred = self.head(torch.cat([ts_repr, text_repr], dim=-1))
        return ModelOutput(pred=pred, proto_losses=proto_losses, segments=bank)

    def compute_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        lambda_c: float,
        lambda_e: float,
        lambda_d: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_pred_loss(output, target, lambda_c, lambda_e, lambda_d)
