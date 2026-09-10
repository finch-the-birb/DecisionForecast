"""Eval-time feature knockouts for H3 faithfulness (not used in training)."""

from __future__ import annotations

import torch

PROTO_MODES = ("none", "zero", "shuffle")
TEXT_MODES = ("none", "zero", "shuffle")


def apply_feature_ablation(
    tensor: torch.Tensor,
    mode: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if mode == "none":
        return tensor
    if mode == "zero":
        return torch.zeros_like(tensor)
    if mode == "shuffle":
        perm = torch.randperm(tensor.size(0), generator=generator)
        return tensor[perm.to(tensor.device)]
    raise ValueError(f"ablation mode={mode!r}; expected none|zero|shuffle")
