"""Unit tests for Distributional LTSF (QuantileHead, Pinball loss, and evaluation metrics)."""

from __future__ import annotations

import pytest
import torch

from src.evaluation.metrics import (
    compute_metrics,
    compute_quantile_metrics,
    empirical_coverage,
    tunnel_sharpness,
    winkler_score,
)
from src.models.heads import LinearHead, QuantileHead, RegressionHead, build_head
from src.models.losses import compute_task_loss, non_crossing_penalty, pinball_loss
from src.models.timexer_hierarchical_mlp import TimeXerHierarchicalMLP


def test_quantile_head_shape_and_gradients():
    head_nf = 64
    horizon = 7
    batch_size = 4
    tokens = 10
    d_model = 16

    head = QuantileHead(head_nf=head_nf, horizon=horizon, hidden=32, quantiles=(0.1, 0.5, 0.9))
    x = torch.randn(batch_size, 4, d_model, requires_grad=True)
    assert 4 * d_model == head_nf

    out = head(x)
    assert out.shape == (batch_size, horizon, 3)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_build_head_factory():
    qh = build_head("quantile", head_nf=64, horizon=7, hidden=32, quantiles=(0.1, 0.5, 0.9))
    assert isinstance(qh, QuantileHead)
    assert qh.n_quantiles == 3

    rh = build_head("mlp", head_nf=64, horizon=7, hidden=32)
    assert isinstance(rh, RegressionHead)

    lh = build_head("linear", head_nf=64, horizon=7)
    assert isinstance(lh, LinearHead)

    with pytest.raises(ValueError):
        build_head("unsupported_head", head_nf=64, horizon=7)


def test_pinball_loss_exact_values():
    # target = 1.0, pred = [0.5, 1.0, 1.5] for quantiles (0.1, 0.5, 0.9)
    # tau=0.1: err = 0.5 -> max(0.1*0.5, -0.9*0.5) = 0.05
    # tau=0.5: err = 0.0 -> max(0, 0) = 0.0
    # tau=0.9: err = -0.5 -> max(0.9*-0.5, -0.1*-0.5) = 0.05
    # mean = (0.05 + 0.0 + 0.05) / 3 = 0.10 / 3
    pred = torch.tensor([[[0.5, 1.0, 1.5]]], dtype=torch.float32)  # [1, 1, 3]
    target = torch.tensor([[1.0]], dtype=torch.float32)             # [1, 1]

    loss = pinball_loss(pred, target, quantiles=(0.1, 0.5, 0.9))
    expected = (0.05 + 0.0 + 0.05) / 3.0
    assert torch.isclose(loss, torch.tensor(expected, dtype=torch.float32), atol=1e-5)


def test_non_crossing_penalty():
    # Ordered: q_0.10 <= q_0.50 <= q_0.90 -> penalty must be 0
    ordered_pred = torch.tensor([[[0.1, 0.5, 0.9]]], dtype=torch.float32)
    penalty_zero = non_crossing_penalty(ordered_pred)
    assert float(penalty_zero) == 0.0

    # Inverted: q_0.10 = 0.8 > q_0.50 = 0.5, and q_0.50 = 0.5 > q_0.90 = 0.2
    # diff1 = 0.8 - 0.5 = 0.3
    # diff2 = 0.5 - 0.2 = 0.3
    # penalty = 0.3 + 0.3 = 0.6
    inverted_pred = torch.tensor([[[0.8, 0.5, 0.2]]], dtype=torch.float32, requires_grad=True)
    penalty = non_crossing_penalty(inverted_pred)
    assert torch.isclose(penalty, torch.tensor(0.6), atol=1e-5)

    penalty.backward()
    assert inverted_pred.grad is not None


def test_compute_task_loss_pinball():
    pred = torch.randn(2, 7, 3, requires_grad=True)
    target = torch.randn(2, 7)

    loss, metrics = compute_task_loss(
        pred,
        target,
        loss_kind="pinball",
        lambda_cross=1.0,
        quantiles=(0.1, 0.5, 0.9),
    )
    assert loss.ndim == 0
    assert "l_pinball" in metrics
    assert "l_cross" in metrics
    loss.backward()
    assert pred.grad is not None


def test_empirical_coverage_and_sharpness():
    # Tunnel [0.0, 10.0]
    # targets: [5.0, 15.0] -> 1 covered out of 2 -> coverage = 0.5
    pred = torch.tensor([[[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]]], dtype=torch.float32)  # [1, 2, 3]
    target = torch.tensor([[5.0, 15.0]], dtype=torch.float32)                           # [1, 2]

    cov = empirical_coverage(pred, target)
    assert cov == 0.5

    shp = tunnel_sharpness(pred)
    assert shp == 10.0  # 10.0 - 0.0


def test_winkler_score():
    # Target inside tunnel [2.0, 8.0]: y = 5.0 -> width = 6.0, penalty = 0
    pred_in = torch.tensor([[[2.0, 5.0, 8.0]]], dtype=torch.float32)
    tgt_in = torch.tensor([[5.0]], dtype=torch.float32)
    score_in = winkler_score(pred_in, tgt_in, alpha=0.2)
    assert score_in == 6.0

    # Target below tunnel [2.0, 8.0]: y = 1.0 (l - y = 1.0)
    # width = 6.0, penalty = (2 / 0.2) * 1.0 = 10.0 -> score = 16.0
    tgt_low = torch.tensor([[1.0]], dtype=torch.float32)
    score_low = winkler_score(pred_in, tgt_low, alpha=0.2)
    assert score_low == 16.0


def test_compute_metrics_3d_quantile_integration():
    pred = torch.tensor([
        [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]
    ], dtype=torch.float32)  # [1, 2, 3]
    target = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # [1, 2]

    # Median forecast matches target exactly [1.0, 2.0]
    metrics = compute_metrics(pred=pred, target=target, target_mode="level")
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert "quantile_coverage" in metrics
    assert "quantile_sharpness" in metrics
    assert "quantile_winkler" in metrics
    assert metrics["quantile_coverage"] == 1.0
    assert metrics["quantile_sharpness"] == 2.0


def test_timexer_hierarchical_mlp_quantile_forward():
    model = TimeXerHierarchicalMLP(
        n_features=5,
        seq_len=32,
        horizon=7,
        d_model=32,
        n_heads=2,
        e_layers=3,
        patch_len=16,
        patch_stride=8,
        n_ts_features=30,
        fft_mode="bridge",
        head_type="quantile",
        head_hidden=32,
        quantiles=(0.1, 0.5, 0.9),
    )

    B = 2
    x = torch.randn(B, 32, 5)
    text = torch.randn(B, 15)
    text_seq = torch.randn(B, 32, 15)
    ts = torch.randn(B, 32, 30)

    out = model(x, text, text_seq=text_seq, ts=ts)
    assert out.pred.shape == (B, 7, 3)

    target = torch.randn(B, 7)
    loss, train_metrics = model.compute_loss(
        out,
        target,
        lambda_c=0.0,
        lambda_e=0.0,
        lambda_d=0.0,
        loss_kind="pinball",
        lambda_cross=1.0,
    )
    assert loss.ndim == 0
    assert "l_pinball" in train_metrics
    assert "l_cross" in train_metrics

    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0
