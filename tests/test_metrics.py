"""Tests for evaluation metrics: MSE, MAE, and Directional Accuracy (DA)."""

from __future__ import annotations

import torch
import pytest

from src.evaluation.metrics import compute_metrics, directional_accuracy, mae, mse


def test_directional_accuracy_100_percent_agreement() -> None:
    # All predictions move in the exact same direction as targets relative to last_close
    last_close = torch.tensor([100.0, 50.0, 20.0, 80.0])
    # Horizon H=7 targets: last step is at index -1
    target = torch.tensor([
        [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 110.0],  # up (+10)
        [49.0, 48.0, 47.0, 46.0, 45.0, 44.0, 40.0],         # down (-10)
        [20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 25.0],         # up (+5)
        [79.0, 78.0, 77.0, 76.0, 75.0, 74.0, 70.0],         # down (-10)
    ])
    pred = torch.tensor([
        [100.5, 101.0, 101.5, 102.0, 103.0, 104.0, 105.0],  # up (+5) -> matches
        [49.5, 49.0, 48.5, 48.0, 47.0, 46.0, 45.0],         # down (-5) -> matches
        [20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 21.0],         # up (+1) -> matches
        [79.5, 79.0, 78.5, 78.0, 77.0, 76.0, 75.0],         # down (-5) -> matches
    ])

    da = directional_accuracy(pred, target, last_close)
    assert da == 1.0


def test_directional_accuracy_zero_percent_agreement() -> None:
    # All predictions move in the opposite direction relative to last_close
    last_close = torch.tensor([100.0, 50.0])
    target = torch.tensor([
        [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 110.0],  # target up
        [49.0, 48.0, 47.0, 46.0, 45.0, 44.0, 40.0],         # target down
    ])
    pred = torch.tensor([
        [99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 90.0],         # pred down -> mismatch
        [51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 60.0],         # pred up -> mismatch
    ])

    da = directional_accuracy(pred, target, last_close)
    assert da == 0.0


def test_directional_accuracy_flat_market() -> None:
    # Case 1: Market is flat and prediction correctly predicts flat
    last_close = torch.tensor([100.0, 50.0])
    target_flat = torch.tensor([
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],  # flat (diff=0)
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],         # flat (diff=0)
    ])
    pred_flat = torch.tensor([
        [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],  # flat (diff=0)
        [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],         # flat (diff=0)
    ])
    da_both_flat = directional_accuracy(pred_flat, target_flat, last_close)
    assert da_both_flat == 1.0

    # Case 2: Market is flat but prediction predicts movement -> mismatch
    pred_moving = torch.tensor([
        [100.5, 101.0, 101.5, 102.0, 103.0, 104.0, 105.0],  # pred up (sign=1 != 0)
        [49.5, 49.0, 48.5, 48.0, 47.0, 46.0, 45.0],         # pred down (sign=-1 != 0)
    ])
    da_mismatch = directional_accuracy(pred_moving, target_flat, last_close)
    assert da_mismatch == 0.0


def test_directional_accuracy_mixed() -> None:
    # 4 samples: 3 correct, 1 wrong -> DA = 0.75
    last_close = torch.tensor([10.0, 10.0, 10.0, 10.0])
    target = torch.tensor([
        [10.0, 12.0],  # up
        [10.0, 8.0],   # down
        [10.0, 15.0],  # up
        [10.0, 5.0],   # down
    ])
    pred = torch.tensor([
        [10.0, 11.0],  # up -> match
        [10.0, 9.0],   # down -> match
        [10.0, 13.0],  # up -> match
        [10.0, 12.0],  # up -> mismatch!
    ])
    da = directional_accuracy(pred, target, last_close)
    assert pytest.approx(da, rel=1e-5) == 0.75


def test_compute_metrics_includes_da_and_denorm() -> None:
    pred = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    target = torch.tensor([[0.5, 1.5], [0.2, -0.8]])
    last_close = torch.tensor([0.0, 0.0])
    y_mean = torch.tensor([100.0, 100.0])
    y_std = torch.tensor([10.0, 10.0])

    metrics = compute_metrics(
        pred, target, y_mean=y_mean, y_std=y_std, last_close=last_close
    )
    assert "mse" in metrics
    assert "mae" in metrics
    assert "mae_denorm" in metrics
    assert "da" in metrics
    assert metrics["da"] == 1.0  # both moved in the right direction
    assert metrics["mae_denorm"] > metrics["mae"]
