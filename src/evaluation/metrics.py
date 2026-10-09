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


def directional_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    last_close: torch.Tensor,
) -> float:
    """Directional Accuracy (DA) of predictions at horizon H vs last close T.

    Calculates:
        Delta y = y[:, -1] - last_close
        Delta y_hat = pred[:, -1] - last_close
        DA = mean(sign(Delta y_hat) == sign(Delta y))
    """
    y_h = target[:, -1] if target.dim() >= 2 else target
    y_hat_h = pred[:, -1] if pred.dim() >= 2 else pred
    lc = last_close.reshape(-1).to(dtype=pred.dtype, device=pred.device)
    delta_y = y_h - lc
    delta_pred = y_hat_h - lc
    correct = torch.sign(delta_pred) == torch.sign(delta_y)
    return float(correct.float().mean().item())


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mean: torch.Tensor | None = None,
    y_std: torch.Tensor | None = None,
    last_close: torch.Tensor | None = None,
) -> dict[str, float]:
    out = {"mse": mse(pred, target), "mae": mae(pred, target)}
    if y_mean is not None and y_std is not None:
        mean = _horizon_scale(y_mean, pred)
        std = _horizon_scale(y_std, pred)
        out["mae_denorm"] = mae(pred * std + mean, target * std + mean)
    if last_close is not None:
        out["da"] = directional_accuracy(pred, target, last_close)
    return out
