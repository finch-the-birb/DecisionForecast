from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {"mse": mse(pred, target), "mae": mae(pred, target)}
