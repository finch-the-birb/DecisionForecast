"""Unit tests for financial task losses (CPU, no data)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.models.dlinear import DLinear
from src.models.losses import compute_task_loss, correlation_penalty, directional_penalty
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.prototypes import PrototypeLosses


def test_mse_matches_functional() -> None:
    pred = torch.randn(4, 7)
    target = torch.randn(4, 7)
    l_task, metrics = compute_task_loss(pred, target, loss_kind="mse")
    torch.testing.assert_close(l_task, F.mse_loss(pred, target))
    assert metrics["l_dir"] == 0.0
    assert metrics["l_corr"] == 0.0
    torch.testing.assert_close(torch.tensor(metrics["l_pred"]), l_task.detach())


def test_huber_matches_smooth_l1() -> None:
    pred = torch.randn(4, 7)
    target = torch.randn(4, 7)
    for delta in (1.0, 0.5):
        l_task, _m = compute_task_loss(pred, target, loss_kind="huber", huber_delta=delta)
        torch.testing.assert_close(l_task, F.smooth_l1_loss(pred, target, beta=delta))


def test_directional_penalizes_antiphase() -> None:
    # Rising target vs falling prediction → opposite increments.
    target = torch.linspace(0.0, 1.0, 7).unsqueeze(0).repeat(3, 1)
    anti = torch.linspace(1.0, 0.0, 7).unsqueeze(0).repeat(3, 1)
    aligned = target.clone()
    l_anti, m_anti = compute_task_loss(anti, target, loss_kind="directional", gamma_dir=1.0)
    l_ok, m_ok = compute_task_loss(aligned, target, loss_kind="directional", gamma_dir=1.0)
    assert m_anti["l_dir"] > 0.0
    assert m_ok["l_dir"] == pytest.approx(0.0, abs=1e-6)
    assert l_anti > l_ok
    raw = directional_penalty(anti, target)
    torch.testing.assert_close(l_anti, F.mse_loss(anti, target) + raw)


def test_correlation_perfect_and_negative() -> None:
    target = torch.linspace(-1.0, 1.0, 7).unsqueeze(0).repeat(4, 1)
    perfect = target.clone()
    opposite = -target
    l_perf, m_perf = compute_task_loss(perfect, target, loss_kind="correlation", alpha_corr=1.0)
    l_neg, m_neg = compute_task_loss(opposite, target, loss_kind="correlation", alpha_corr=1.0)
    assert m_perf["l_corr"] == pytest.approx(0.0, abs=1e-5)
    assert l_perf == pytest.approx(0.0, abs=1e-5)
    assert m_neg["l_corr"] == pytest.approx(2.0, abs=1e-4)
    assert l_neg > l_perf

    const = torch.zeros_like(target)
    _l_zero, m_zero = compute_task_loss(const, target, loss_kind="correlation", alpha_corr=1.0)
    assert m_zero["l_corr"] == pytest.approx(1.0, abs=1e-4)
    raw = correlation_penalty(opposite, target)
    torch.testing.assert_close(l_neg, F.mse_loss(opposite, target) + raw)


def test_huber_directional_combines() -> None:
    pred = torch.randn(2, 7)
    target = torch.randn(2, 7)
    l_task, metrics = compute_task_loss(
        pred, target, loss_kind="huber_directional", huber_delta=0.5, gamma_dir=0.25
    )
    base = F.smooth_l1_loss(pred, target, beta=0.5)
    torch.testing.assert_close(l_task, base + 0.25 * directional_penalty(pred, target))
    assert metrics["l_dir"] > 0.0 or torch.allclose(pred, target)


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown loss_kind"):
        compute_task_loss(torch.randn(2, 7), torch.randn(2, 7), loss_kind="quantile")


def test_task_and_pred_loss_backward_finite() -> None:
    pred = torch.randn(5, 7, requires_grad=True)
    target = torch.randn(5, 7)
    for kind in ("mse", "huber", "directional", "correlation", "huber_directional"):
        x = pred.clone().detach().requires_grad_(True)
        l_task, _m = compute_task_loss(x, target, loss_kind=kind)
        l_task.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    out = ModelOutput(
        pred=torch.randn(3, 7, requires_grad=True),
        proto_losses=PrototypeLosses(
            l_c=torch.tensor(0.2, requires_grad=True),
            l_e=torch.tensor(0.1, requires_grad=True),
            l_d=torch.tensor(0.0, requires_grad=True),
        ),
        segments=torch.zeros(3, 1, 1),
    )
    total, metrics = compute_pred_loss(
        out,
        torch.randn(3, 7),
        lambda_c=0.3,
        lambda_e=0.3,
        lambda_d=0.01,
        loss_kind="huber",
        huber_delta=0.5,
    )
    total.backward()
    assert torch.isfinite(out.pred.grad).all()
    assert "l_dir" in metrics and "l_corr" in metrics
    assert "l_c" in metrics


def test_dlinear_compute_loss_forwards_kwargs() -> None:
    model = DLinear(seq_len=16, horizon=7, n_features=5, target_idx=0)
    x = torch.randn(2, 16, 5)
    y = torch.randn(2, 7)
    out = model(x)
    loss, metrics = model.compute_loss(
        out, y, 0.0, 0.0, 0.0, loss_kind="huber", huber_delta=1.0
    )
    assert loss.ndim == 0
    torch.testing.assert_close(loss, F.smooth_l1_loss(out.pred, y))
    assert "l_pred" in metrics
