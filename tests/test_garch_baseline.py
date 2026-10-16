"""Unit tests for econometric GARCH(1,1) uncertainty baseline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.garch_baseline import (
    fit_garch_corridor,
    compute_garch_corridor_metrics,
    _fallback_corridor,
    Z_80,
)


def test_fit_garch_corridor_shape_and_monotonicity():
    np.random.seed(42)
    # Generate 300 daily returns with typical financial volatility (annualized ~20%)
    rets = np.random.normal(0.0004, 0.013, 300)
    horizon = 7

    tunnel, mu = fit_garch_corridor(rets, horizon=horizon, z_score=Z_80)
    assert tunnel.shape == (horizon, 3)
    assert np.isfinite(tunnel).all()

    # Monotonicity: q_0.10 <= q_0.50 <= q_0.90 at each horizon step
    for h in range(horizon):
        q_low, q_med, q_high = tunnel[h]
        assert q_low <= q_med <= q_high

    # Dispersion expands with horizon (multi-step cumulative variance)
    width_h1 = tunnel[0, 2] - tunnel[0, 0]
    width_h7 = tunnel[6, 2] - tunnel[6, 0]
    assert width_h7 > width_h1


def test_fit_garch_corridor_fallback_on_constant_returns():
    # Constant returns: GARCH cannot fit, must cleanly fallback
    rets = np.full(100, 0.001)
    tunnel, mu = fit_garch_corridor(rets, horizon=7)
    assert tunnel.shape == (7, 3)
    assert np.isfinite(tunnel).all()
    assert np.all(tunnel[:, 0] <= tunnel[:, 1])
    assert np.all(tunnel[:, 1] <= tunnel[:, 2])


def test_compute_garch_corridor_metrics_synthetic_dataset():
    horizon = 7
    N = 20

    # Create synthetic dataset items
    class MockDataset:
        def __init__(self):
            np.random.seed(123)
            self.items = []
            for _ in range(N):
                # 60 historical close prices
                close = 100.0 + np.cumsum(np.random.normal(0.0, 1.0, 60))
                x = torch.tensor(close[:, None], dtype=torch.float32)
                # 7 forward returns
                y = torch.tensor(np.random.normal(0.001, 0.02, horizon), dtype=torch.float32)
                self.items.append({"x": x, "y": y})

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    mock_ds = MockDataset()
    metrics = compute_garch_corridor_metrics(mock_ds, horizon=horizon, min_history=40)

    assert "garch_coverage" in metrics
    assert "garch_sharpness" in metrics
    assert "garch_winkler" in metrics
    assert "garch_da" in metrics

    assert 0.0 <= metrics["garch_coverage"] <= 1.0
    assert metrics["garch_sharpness"] > 0.0
    assert metrics["garch_winkler"] > 0.0
    assert 0.0 <= metrics["garch_da"] <= 1.0
