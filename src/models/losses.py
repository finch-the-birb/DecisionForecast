"""Task losses for financial LTSF: MSE, Huber, directional, correlation."""

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
        "'mse' | 'huber' | 'directional' | 'correlation' | 'huber_directional'"
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


def compute_task_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_kind: str = "mse",
    huber_delta: float = 1.0,
    gamma_dir: float = 0.1,
    alpha_corr: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    kind = str(loss_kind)
    base = _regression_loss(pred, target, kind, float(huber_delta))
    l_dir = directional_penalty(pred, target) if kind in _DIR_KINDS else pred.new_zeros(())
    l_corr = correlation_penalty(pred, target) if kind in _CORR_KINDS else pred.new_zeros(())
    l_task = base
    if kind in _DIR_KINDS:
        l_task = l_task + float(gamma_dir) * l_dir
    if kind in _CORR_KINDS:
        l_task = l_task + float(alpha_corr) * l_corr
    metrics = {
        "l_pred": float(base.detach()),
        "l_dir": float(l_dir.detach()),
        "l_corr": float(l_corr.detach()),
    }
    return l_task, metrics
