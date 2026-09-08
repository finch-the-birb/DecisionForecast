"""G1+G2 prototype injection + TimeXer ablations B / C0 / C1.

G1+G2: similarity / residual only on endogenous patch tokens; G_en is not
projected. Injection is P <- P + W S after PatchEmbed, then TimeXer layers.

Fusion (one factor at a time):
- late (B): text concat at the prediction head
- mid_no_attn (C0): add projected text to patch tokens; G_en is not a text query
- mid_cross_attn (C1): text as an extra exogenous variate token; G_en cross-attends
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.prototypes import PrototypeLosses, PrototypeModule
from src.models.timexer import TimeXerBackbone

_LATE = "late"
_MID_ADD = "mid_no_attn"
_MID_CROSS = "mid_cross_attn"
_FUSIONS = (_LATE, _MID_ADD, _MID_CROSS)


@dataclass
class ModelBOutput:
    pred: torch.Tensor
    proto_losses: PrototypeLosses
    segments: torch.Tensor


class TimeXerFusionModel(nn.Module):
    """TimeXer + prototype-before-attn with late / mid-add / G_en-cross text fusion."""

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
        fusion: str = _LATE,
        d_ff: int | None = None,
        use_norm: bool = False,
        use_prototypes: bool = True,
    ) -> None:
        super().__init__()
        if fusion not in _FUSIONS:
            raise ValueError(f"fusion={fusion!r}; expected one of {_FUSIONS}")
        self.horizon = horizon
        self.fusion = fusion
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
        head_in = d_model * 2 if fusion == _LATE else d_model
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
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
        text_tok = self.text_mlp(text)
        if self.fusion == _MID_ADD:
            patches = patches + text_tok.unsqueeze(1)
        elif self.fusion == _MID_CROSS:
            extra = text_tok.unsqueeze(1)
            exo = extra if exo is None else torch.cat([exo, extra], dim=1)
        enc_patches, _enc_g = self.backbone.encode(patches, g_en, exo)
        ts_repr = enc_patches.mean(dim=1)
        if self.fusion == _LATE:
            fused = torch.cat([ts_repr, text_tok], dim=-1)
            pred = self.head(fused)
        else:
            pred = self.head(ts_repr)
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


class TimeXerModelB(TimeXerFusionModel):
    """Model B: late fusion (H1)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("fusion", _LATE)
        super().__init__(**kwargs)
