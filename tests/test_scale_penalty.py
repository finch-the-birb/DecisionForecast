"""Unit tests for Amplitude Scale Shrinkage Penalty (beta_scale) and loss integrations."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.losses import compute_task_loss, scale_penalty
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.timexer_hierarchical_mlp import TimeXerHierarchicalMLP
from src.training.train import build_model


def test_scale_penalty_properties() -> None:
    # 1. Zero vector -> zero penalty
    pred_zero = torch.zeros(4, 7)
    assert scale_penalty(pred_zero).item() == 0.0

    # 2. Known vector
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # sum of squares: 1 + 4 + 9 + 16 = 30, mean = 7.5
    assert pytest.approx(scale_penalty(pred).item(), abs=1e-5) == 7.5

    # 3. Scaling property: scaling pred by factor c scales penalty by c^2
    pred_c = pred * 2.0
    assert pytest.approx(scale_penalty(pred_c).item(), abs=1e-5) == 7.5 * 4.0


def test_compute_task_loss_scale_penalty_integration() -> None:
    torch.manual_seed(42)
    pred = torch.randn(8, 7)
    target = torch.randn(8, 7)

    # Base loss (no scale penalty)
    loss_0, m_0 = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.0, alpha_corr=0.0, beta_scale=0.0
    )
    # With scale penalty
    loss_b005, m_b005 = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.0, alpha_corr=0.0, beta_scale=0.05
    )
    loss_b010, m_b010 = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.0, alpha_corr=0.0, beta_scale=0.10
    )

    # Strict monotonicity with beta_scale
    assert loss_b005.item() > loss_0.item()
    assert loss_b010.item() > loss_b005.item()

    # Exact additivity check
    assert "l_scale" in m_b005
    assert pytest.approx(m_b005["l_scale"], rel=1e-5) == scale_penalty(pred).item()
    expected_b005 = loss_0.item() + 0.05 * m_b005["l_scale"]
    assert pytest.approx(loss_b005.item(), rel=1e-5) == expected_b005

    # Combine with directional and correlation penalties
    loss_all, m_all = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.6, alpha_corr=0.3, beta_scale=0.05
    )
    assert loss_all.item() > loss_b005.item()
    expected_all = (
        loss_0.item()
        + 0.6 * m_all["l_dir"]
        + 0.3 * m_all["l_corr"]
        + 0.05 * m_all["l_scale"]
    )
    assert pytest.approx(loss_all.item(), rel=1e-5) == expected_all


def test_scale_penalty_gradient_flow() -> None:
    torch.manual_seed(42)
    pred = torch.randn(4, 7, requires_grad=True)
    target = torch.randn(4, 7)

    loss, _ = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.0, alpha_corr=0.0, beta_scale=0.05
    )
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_compute_pred_loss_forwards_beta_scale() -> None:
    torch.manual_seed(42)
    pred = torch.randn(4, 7)
    target = torch.randn(4, 7)
    output = ModelOutput(pred=pred)

    l_0, m_0 = compute_pred_loss(
        output, target, lambda_c=0.0, lambda_e=0.0, lambda_d=0.0,
        loss_kind="huber", gamma_dir=0.6, alpha_corr=0.3, beta_scale=0.0
    )
    l_scale, m_scale = compute_pred_loss(
        output, target, lambda_c=0.0, lambda_e=0.0, lambda_d=0.0,
        loss_kind="huber", gamma_dir=0.6, alpha_corr=0.3, beta_scale=0.05
    )

    assert l_scale.item() > l_0.item()
    assert "l_scale" in m_scale
    assert pytest.approx(m_scale["l_scale"]) == scale_penalty(pred).item()


def test_timexer_hierarchical_mlp_backward_with_all_penalties() -> None:
    torch.manual_seed(42)
    model = TimeXerHierarchicalMLP(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=32,
        n_heads=2,
        e_layers=3,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=15,
        n_ts_features=30,
        close_idx=0,
        head_type="mlp",
        head_hidden=64,
        head_dropout=0.1,
        fft_mode="bridge",
    )

    x = torch.randn(4, 60, 5, requires_grad=True)
    text = torch.randn(4, 15)
    ts = torch.randn(4, 60, 30)
    target = torch.randn(4, 7)

    out = model(x, text=text, ts=ts)
    loss, metrics = model.compute_loss(
        out,
        target,
        lambda_c=0.0,
        lambda_e=0.0,
        lambda_d=0.0,
        loss_kind="huber",
        huber_delta=0.5,
        gamma_dir=0.6,
        alpha_corr=0.3,
        beta_scale=0.05,
    )

    assert "l_pred" in metrics
    assert "l_dir" in metrics
    assert "l_corr" in metrics
    assert "l_scale" in metrics
    assert metrics["l_scale"] > 0.0

    loss.backward()

    # Verify head parameters receive gradients
    for name, param in model.head.named_parameters():
        assert param.grad is not None, f"Gradient missing for head param {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in head param {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient in head param {name}"


def test_hydra_build_model_scale_penalty_configs() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    configs = [
        ("c1_hierarchical_mlp_delta_g06_a03_b001", 0.6, 0.3, 0.01),
        ("c1_hierarchical_mlp_delta_g06_a03_b005", 0.6, 0.3, 0.05),
        ("c1_hierarchical_mlp_delta_g06_a03_b010", 0.6, 0.3, 0.10),
    ]

    for model_name, expected_g, expected_a, expected_b in configs:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[f"model={model_name}", "data=fnspid_dual_ts"],
            )
            assert cfg.model.target_mode == "delta"
            assert cfg.model.loss.gamma_dir == expected_g
            assert cfg.model.loss.alpha_corr == expected_a
            assert cfg.model.loss.beta_scale == expected_b
            assert cfg.model.n_ts_features == 30
            assert cfg.model.fft_mode == "bridge"
            assert cfg.model.head.type == "mlp"

            model = build_model(cfg)
            assert isinstance(model, TimeXerHierarchicalMLP)
            assert model.n_ts_features == 30
