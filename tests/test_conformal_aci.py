"""Unit tests for Adaptive Conformal Inference (ACI) calibration."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import norm

from src.evaluation.conformal import apply_adaptive_conformal_inference


def test_aci_adaptation_step_direction():
    # Setup single step with known violation and known hit
    alpha_target = 0.20
    gamma = 0.01

    # Pred tunnel [-1.0, 0.0, 1.0]
    preds = torch.tensor([[[-1.0, 0.0, 1.0]], [[-1.0, 0.0, 1.0]]], dtype=torch.float32)  # [2, 1, 3]

    # Target 0: y = 2.0 (out of bounds -> err_0 = 1)
    # Target 1: y = 0.0 (inside bounds -> err_1 = 0)
    targets = torch.tensor([[2.0], [0.0]], dtype=torch.float32)  # [2, 1]

    out = apply_adaptive_conformal_inference(preds, targets, alpha=alpha_target, gamma=gamma)
    alphas = out["alphas"]  # [2, 1]

    # At t=0, alpha_0 = alpha_target = 0.20
    assert torch.isclose(alphas[0, 0], torch.tensor(0.20), atol=1e-5)

    # After err_0 = 1: alpha_1 = alpha_0 + gamma * (0.20 - 1.0) = 0.20 - 0.008 = 0.192
    assert torch.isclose(alphas[1, 0], torch.tensor(0.192), atol=1e-5)

    # When alpha decreases, the corridor widens:
    cal_preds = out["calibrated_preds"]
    width_0 = cal_preds[0, 0, 2] - cal_preds[0, 0, 0]
    width_1 = cal_preds[1, 0, 2] - cal_preds[1, 0, 0]
    assert width_1 > width_0


def test_aci_improves_undercoverage():
    torch.manual_seed(42)
    N, H = 400, 7
    alpha_target = 0.20
    z_nom = norm.ppf(1.0 - alpha_target / 2.0)

    # True process has std = 2.0, but model under-estimated with std = 1.0
    true_std = 2.0
    model_std = 1.0

    targets = torch.randn(N, H) * true_std
    preds = torch.zeros(N, H, 3)
    preds[..., 0] = -z_nom * model_std
    preds[..., 1] = 0.0
    preds[..., 2] = z_nom * model_std

    out = apply_adaptive_conformal_inference(preds, targets, alpha=alpha_target, gamma=0.005)

    # Raw coverage must be substantially below nominal 80%
    assert out["raw_coverage"] < 0.65

    # Calibrated coverage must adapt upwards closer to nominal 80%
    assert out["coverage"] > out["raw_coverage"]
    assert out["coverage"] >= 0.70

    # Final alpha should be lower than initial alpha_target
    assert out["alphas"][-1].mean() < alpha_target


def test_aci_2d_input_support():
    N = 50
    preds_2d = torch.zeros(N, 3)
    preds_2d[:, 0] = -1.28
    preds_2d[:, 1] = 0.0
    preds_2d[:, 2] = 1.28

    targets_1d = torch.randn(N) * 0.8

    out = apply_adaptive_conformal_inference(preds_2d, targets_1d, alpha=0.20, gamma=0.005)

    assert out["calibrated_preds"].shape == (N, 3)
    assert out["alphas"].shape == (N,)
    assert 0.0 <= out["coverage"] <= 1.0
    assert out["sharpness"] > 0.0
    assert out["winkler"] > 0.0
