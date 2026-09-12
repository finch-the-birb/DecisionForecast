"""G1+G2 prototype residual on TimeXer patches (shared by B / C0 / C1).

Similarity and ``project()`` run only on endogenous patch tokens ``P``
(pre-injection); ``G_en`` is never prototyped. Residual: ``P <- P + W_s(S)``
after PatchEmbed, before any text entry point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.prototypes import PrototypeLosses, PrototypeModule


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
