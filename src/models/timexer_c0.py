"""Model C0: TimeXer + G1+G2 + per-patch text add (no cross-attn, no head text).

Text enters once, after ``proto(P)``, via ``days_to_patches(text_seq)``. Encoder
``exo=None``. ``G_en`` is not written by text. Head is ``Linear(d → H)``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.fusion import assert_fusion, fusion_map
from src.models.head import ForecastHead
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.timexer_backbone import TimeXerBackbone, days_to_patches, n_patches
from src.models.timexl_integration import PrototypeResidual


class TimeXerC0(nn.Module):
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
        flags = fusion_map(fusion)
        if flags.get("text_align") == "pooled":
            raise ValueError("pooled text-add is B, not C0")
        layers = flags.get("inject_layers")
        if list(layers) != [0]:
            raise NotImplementedError(
                f"C0 inject_layers={layers!r}; only [0] is implemented"
            )
        self.fusion = assert_fusion(
            fusion,
            kind="mid_no_attn",
            text_at_head=False,
            text_to_patches=True,
            text_align="per_patch",
            text_inject={"add", "concat_proj"},
            text_as_exogenous=False,
        )
        self.text_inject = str(self.fusion["text_inject"])
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
        self.w_t = nn.Linear(text_dim, d_model)
        self.merge = (
            nn.Linear(d_model * 2, d_model) if self.text_inject == "concat_proj" else None
        )
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

    def _inject_text(
        self,
        patches: torch.Tensor,
        text_seq: torch.Tensor,
        text_mode: str,
        ablation_generator: torch.Generator | None,
    ) -> torch.Tensor:
        embed = self.backbone.patch_embed
        aligned = days_to_patches(text_seq, embed.patch_len, embed.patch_stride)
        delta = apply_feature_ablation(
            self.w_t(aligned), text_mode, ablation_generator
        )
        if self.text_inject == "add":
            return patches + delta
        assert self.merge is not None
        return self.merge(torch.cat([patches, delta], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_seq: torch.Tensor | None = None,
        proto_mode: str = "none",
        text_mode: str = "none",
        ablation_generator: torch.Generator | None = None,
    ) -> ModelOutput:
        del text  # C0 does not use pooled text / head concat
        if text_seq is None:
            raise ValueError("C0 requires text_seq [B,T,text_dim]")
        patches, g_en = self.backbone.embed(x)
        bank = patches
        patches, proto_losses = self.g12(patches, proto_mode, ablation_generator)
        patches = self._inject_text(patches, text_seq, text_mode, ablation_generator)
        enc_p, _g = self.backbone.encode(patches, g_en, exo=None)
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
