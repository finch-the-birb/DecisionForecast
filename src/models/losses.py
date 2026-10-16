"""Task losses for financial LTSF: MSE, Huber, directional, correlation, and multi-quantile pinball."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_REGRESSION_KINDS = frozenset({"mse", "directional", "correlation"})
_HUBER_KINDS = frozenset({"huber", "huber_directional"})
_DIR_KINDS = frozenset({"directional", "huber_directional"})
_CORR_KINDS = frozenset({"correlation"})


def _regression_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_kind: str, huber_delta: float
) -> torch.Tensor:
    if loss_kind in _REGRESSION_KINDS:
        return F.mse_loss(pred, target)
    if loss_kind in _HUBER_KINDS:
        return F.smooth_l1_loss(pred, target, beta=huber_delta)
    raise ValueError(
        f"Unknown loss_kind: {loss_kind!r}; expected "
        "'mse' | 'huber' | 'directional' | 'correlation' | 'huber_directional' | 'pinball'"
    )


def directional_penalty(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean ReLU of opposite-sign step increments along the horizon."""
    if pred.size(-1) < 2:
        return pred.new_zeros(())
    delta_y = target[:, 1:] - target[:, :-1]
    delta_hat = pred[:, 1:] - pred[:, :-1]
    return F.relu(-delta_y * delta_hat).mean()


def correlation_penalty(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - mean Pearson r along the horizon (per batch item). Range ~[0, 2]."""
    y_c = target - target.mean(dim=-1, keepdim=True)
    hat_c = pred - pred.mean(dim=-1, keepdim=True)
    num = (y_c * hat_c).sum(dim=-1)
    den = (y_c.pow(2).sum(dim=-1) + eps).sqrt() * (hat_c.pow(2).sum(dim=-1) + eps).sqrt()
    r = num / den
    return 1.0 - r.mean()


def scale_penalty(pred: torch.Tensor) -> torch.Tensor:
    """L2 shrinkage penalty on forecast amplitude (mean square of predicted deltas)."""
    return pred.pow(2).mean()


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """Multi-quantile pinball loss across batch, horizon, and quantiles.

    L_tau(y, q_tau) = max(tau * (y - q_tau), (tau - 1) * (y - q_tau))

    Args:
        pred: Predicted quantiles [B, H, K] or [B, H].
        target: Ground truth targets [B, H].
        quantiles: Quantile levels tau in (0, 1). Default (0.1, 0.5, 0.9).
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(-1)
    losses = []
    for i, tau in enumerate(quantiles):
        err = target - pred[..., i]
        losses.append(torch.max(tau * err, (tau - 1.0) * err))
    return torch.stack(losses, dim=-1).mean()


def non_crossing_penalty(pred: torch.Tensor) -> torch.Tensor:
    """Penalize quantile crossing: q_0.10 > q_0.50 and q_0.50 > q_0.90."""
    if pred.dim() < 3 or pred.size(-1) < 2:
        return pred.new_zeros(())
    diff1 = F.relu(pred[..., 0] - pred[..., 1])
    diff2 = F.relu(pred[..., 1] - pred[..., 2])
    return (diff1 + diff2).mean()


def compute_task_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_kind: str = "huber",
    huber_delta: float = 0.5,
    gamma_dir: float = 0.0,
    alpha_corr: float = 0.0,
    beta_scale: float = 0.0,
    lambda_cross: float = 1.0,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> tuple[torch.Tensor, dict[str, float]]:
    if loss_kind == "pinball":
        base = pinball_loss(pred, target, quantiles=quantiles)
        l_cross = non_crossing_penalty(pred) if float(lambda_cross) > 0.0 else pred.new_zeros(())
        l_task = base + float(lambda_cross) * l_cross
        metrics = {
            "l_pred": float(base.detach()),
            "l_pinball": float(base.detach()),
            "l_cross": float(l_cross.detach()),
            "l_dir": 0.0,
            "l_corr": 0.0,
            "l_scale": 0.0,
        }
        return l_task, metrics

    base = _regression_loss(pred, target, loss_kind, float(huber_delta))
    l_task = base

    # Direction penalty
    l_dir = directional_penalty(pred, target) if float(gamma_dir) > 0.0 else pred.new_zeros(())
    if float(gamma_dir) > 0.0:
        l_task = l_task + float(gamma_dir) * l_dir

    # Correlation penalty (shape)
    l_corr = correlation_penalty(pred, target) if float(alpha_corr) > 0.0 else pred.new_zeros(())
    if float(alpha_corr) > 0.0:
        l_task = l_task + float(alpha_corr) * l_corr

    # Scale shrinkage penalty
    l_scale = scale_penalty(pred) if float(beta_scale) > 0.0 else pred.new_zeros(())
    if float(beta_scale) > 0.0:
        l_task = l_task + float(beta_scale) * l_scale

    metrics = {
        "l_pred": float(base.detach()),
        "l_dir": float(l_dir.detach()),
        "l_corr": float(l_corr.detach()),
        "l_scale": float(l_scale.detach()),
    }
    return l_task, metrics
