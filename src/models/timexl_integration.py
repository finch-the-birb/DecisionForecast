"""G1+G2 prototype injection into TimeXer patch tokens + Model B (late fusion).

G1+G2: similarity / residual only on endogenous patch tokens; G_en is not
projected. Injection is P <- P + W S after PatchEmbed, then TimeXer layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.prototypes import PrototypeLosses, PrototypeModule
from src.models.timexer import TimeXerBackbone


@dataclass
class ModelBOutput:
    pred: torch.Tensor
    proto_losses: PrototypeLosses
    segments: torch.Tensor


class TimeXerModelB(nn.Module):
    """Model B: TimeXer + prototype-before-attn + late text fusion."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        target_idx: int,
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
        d_ff: int | None = None,
        use_norm: bool = False,
        use_prototypes: bool = True,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.use_prototypes = use_prototypes
        self.backbone = TimeXerBackbone(
            seq_len=seq_len,
            n_features=n_features,
            target_idx=target_idx,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            patch_len=patch_len,
            patch_stride=patch_stride,
            dropout=dropout,
            d_ff=d_ff,
            use_norm=use_norm,
        )
        self.proto = PrototypeModule(n_prototypes, d_model, d_min)
        self.inject = nn.Linear(d_model, d_model)
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

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> ModelBOutput:
        patches, g_en, exo, _stats = self.backbone.embed(x)
        if self.use_prototypes:
            proto_mix, proto_losses = self.proto(patches)
            patches = patches + self.inject(proto_mix)
        else:
            proto_losses = PrototypeLosses(
                l_c=patches.new_zeros(()),
                l_e=patches.new_zeros(()),
                l_d=patches.new_zeros(()),
            )
        enc_patches, _enc_g = self.backbone.encode(patches, g_en, exo)
        ts_repr = enc_patches.mean(dim=1)
        text_repr = self.text_mlp(text)
        fused = torch.cat([ts_repr, text_repr], dim=-1)
        pred = self.head(fused)
        return ModelBOutput(pred=pred, proto_losses=proto_losses, segments=patches)

    def compute_loss(
        self,
        output: ModelBOutput,
        target: torch.Tensor,
        lambda_c: float,
        lambda_e: float,
        lambda_d: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_pred = nn.functional.mse_loss(output.pred, target)
        pl = output.proto_losses
        total = l_pred + lambda_c * pl.l_c + lambda_e * pl.l_e + lambda_d * pl.l_d
        metrics = {
            "loss": float(total.detach()),
            "l_pred": float(l_pred.detach()),
            "l_c": float(pl.l_c.detach()),
            "l_e": float(pl.l_e.detach()),
            "l_d": float(pl.l_d.detach()),
        }
        return total, metrics
