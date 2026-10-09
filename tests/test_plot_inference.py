"""Unit tests for inference trajectory visualization."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.evaluation.plot_inference import plot_single_ticker


def test_plot_single_ticker_renders_without_error(tmp_path: Path) -> None:
    dates = pd.date_range("2023-01-01", periods=30)
    prices = np.linspace(100.0, 120.0, 30)
    trajectories = [
        {"dates": dates[:8], "prices": prices[:8] + 0.5, "start_date": dates[0]},
        {"dates": dates[7:15], "prices": prices[7:15] - 0.5, "start_date": dates[7]},
    ]
    result = {
        "ticker": "AAPL",
        "gt_dates": dates,
        "gt_prices": prices,
        "trajectories": trajectories,
        "mae_dollars": 1.25,
        "da_pct": 85.0,
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_single_ticker(ax, result, horizon=7)
    out_file = tmp_path / "AAPL_test.png"
    fig.savefig(out_file, dpi=100, bbox_inches="tight")
    plt.close(fig)

    assert out_file.is_file()
    assert out_file.stat().st_size > 1000
