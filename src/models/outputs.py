"""Shared forecast output + optional prototype loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.prototypes import PrototypeLosses


@dataclass
class ModelOutput:
    pred: torch.Tensor
    proto_losses: PrototypeLosses | None
    segments: torch.Tensor  # pre-injection embeddings (proto similarity space)


def compute_pred_loss(
    output: ModelOutput,
    target: torch.Tensor,
    lambda_c: float,
    lambda_e: float,
    lambda_d: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_pred = F.mse_loss(output.pred, target)
    if output.proto_losses is None:
        zeros = l_pred.new_zeros(())
        total = l_pred
        pl = PrototypeLosses(l_c=zeros, l_e=zeros, l_d=zeros)
    else:
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
