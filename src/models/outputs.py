"""Shared forecast output + optional prototype loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.losses import compute_task_loss
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
    loss_kind: str = "mse",
    huber_delta: float = 1.0,
    gamma_dir: float = 0.1,
    alpha_corr: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_task, task_metrics = compute_task_loss(
        output.pred,
        target,
        loss_kind=loss_kind,
        huber_delta=huber_delta,
        gamma_dir=gamma_dir,
        alpha_corr=alpha_corr,
    )
    if output.proto_losses is None:
        zeros = l_task.new_zeros(())
        pl = PrototypeLosses(l_c=zeros, l_e=zeros, l_d=zeros)
        total = l_task
    else:
        pl = output.proto_losses
        total = l_task + lambda_c * pl.l_c + lambda_e * pl.l_e + lambda_d * pl.l_d

    metrics = {
        "loss": float(total.detach()),
        "l_pred": float(l_task.detach()),
        "l_c": float(pl.l_c.detach()),
        "l_e": float(pl.l_e.detach()),
        "l_d": float(pl.l_d.detach()),
        **task_metrics,
    }
    return total, metrics
