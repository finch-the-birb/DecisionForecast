"""TimeXer without prototypes or text (pipeline sanity, not a hypothesis)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.timexer_backbone import TimeXerBackbone


class TimeXerPlain(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        patch_len: int,
        patch_stride: int,
        dropout: float,
        d_ff: int | None = None,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
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
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, horizon),
        )

    def forward(self, x: torch.Tensor, text: torch.Tensor | None = None) -> ModelOutput:
        enc_p = self.backbone(x, exo=None)
        pred = self.head(enc_p.mean(dim=1))
        return ModelOutput(pred=pred, proto_losses=None, segments=enc_p)

    def compute_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        lambda_c: float,
        lambda_e: float,
        lambda_d: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_pred_loss(output, target, lambda_c, lambda_e, lambda_d)
