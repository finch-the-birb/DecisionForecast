from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def _horizon_scale(scale: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    while scale.dim() < y.dim():
        scale = scale.unsqueeze(-1)
    return scale.to(dtype=y.dtype, device=y.device)


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mean: torch.Tensor | None = None,
    y_std: torch.Tensor | None = None,
) -> dict[str, float]:
    out = {"mse": mse(pred, target), "mae": mae(pred, target)}
    if y_mean is not None and y_std is not None:
        mean = _horizon_scale(y_mean, pred)
        std = _horizon_scale(y_std, pred)
        out["mae_denorm"] = mae(pred * std + mean, target * std + mean)
    return out
