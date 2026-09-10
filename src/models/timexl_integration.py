"""G1+G2 prototype residual on TimeXer patches (shared by B / C0 / C1).

Similarity and ``project()`` run only on endogenous patch tokens ``P``;
``G_en`` is never prototyped. Residual: ``P <- P + W_s(S)`` after PatchEmbed,
before any text entry point.

The legacy ``TimeXerFusionModel`` (per-variate TimeXer) still backs C0/C1
until those models are rewritten onto ``timexer_backbone``. Model B lives in
``src/models/timexer_b.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.prototypes import PrototypeLosses, PrototypeModule
from src.models.timexer import TimeXerBackbone as LegacyTimeXerBackbone

_LATE = "late"
_MID_ADD = "mid_no_attn"
_MID_CROSS = "mid_cross_attn"
_FUSIONS = (_LATE, _MID_ADD, _MID_CROSS)


class PrototypeResidual(nn.Module):
    """G1+G2: ``S, losses = proto(P)`` then ``P = P + W_s(S)``. ``G_en`` never enters."""

    def __init__(self, n_prototypes: int, d_model: int, d_min: float) -> None:
        super().__init__()
        self.proto = PrototypeModule(n_prototypes, d_model, d_min)
        self.w_s = nn.Linear(d_model, d_model)

    def forward(
        self,
        patches: torch.Tensor,
        proto_mode: str = "none",
        ablation_generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, PrototypeLosses]:
        proto_mix, proto_losses = self.proto(patches)
        if proto_mode != "zero":
            proto_mix = apply_feature_ablation(proto_mix, proto_mode, ablation_generator)
            patches = patches + self.w_s(proto_mix)
        return patches, proto_losses


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
        self.backbone = LegacyTimeXerBackbone(
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

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_seq: torch.Tensor | None = None,
        proto_mode: str = "none",
        text_mode: str = "none",
        ablation_generator: torch.Generator | None = None,
    ) -> ModelBOutput:
        del text_seq
        patches, g_en, _stats = self.backbone.embed(x)
        batch, n_vars, n_patches, d_model = patches.shape
        if self.use_prototypes:
            flat = patches.reshape(batch, n_vars * n_patches, d_model)
            proto_mix, proto_losses = self.proto(flat)
            if proto_mode != "zero":
                proto_mix = apply_feature_ablation(
                    proto_mix, proto_mode, ablation_generator
                )
                patches = patches + self.inject(proto_mix).view(
                    batch, n_vars, n_patches, d_model
                )
        else:
            proto_losses = PrototypeLosses(
                l_c=patches.new_zeros(()),
                l_e=patches.new_zeros(()),
                l_d=patches.new_zeros(()),
            )
        text_tok = apply_feature_ablation(
            self.text_mlp(text), text_mode, ablation_generator
        )
        exo = None
        if text_mode != "zero":
            if self.fusion == _MID_ADD:
                patches = patches + text_tok.view(batch, 1, 1, d_model)
            elif self.fusion == _MID_CROSS:
                exo = text_tok.unsqueeze(1)
        enc_patches, _enc_g = self.backbone.encode(patches, g_en, exo)
        target_patches = enc_patches[:, self.backbone.target_idx, :, :]
        ts_repr = target_patches.mean(dim=1)
        if self.fusion == _LATE:
            fused = torch.cat([ts_repr, text_tok], dim=-1)
            pred = self.head(fused)
        else:
            pred = self.head(ts_repr)
        segments = patches.reshape(batch, n_vars * n_patches, d_model)
        return ModelBOutput(pred=pred, proto_losses=proto_losses, segments=segments)

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
