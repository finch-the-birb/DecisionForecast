"""Unit tests for scripts/paper_economic_eval.py."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.paper_economic_eval import (
    BacktestMetrics,
    aggregate_seed_metrics,
    compute_portfolio_metrics,
    compute_quantile_snr_signal,
    format_tables,
    simulate_portfolio_strategy,
)


def test_compute_quantile_snr_signal():
    # Return space: q50=0.04, q10=-0.02, q90=0.08
    q10 = np.array([-0.02, -0.05])
    q50 = np.array([0.04, -0.01])
    q90 = np.array([0.08, 0.03])

    snr = compute_quantile_snr_signal(q10, q50, q90)
    # Asset 0: 0.04 / (0.08 - (-0.02)) = 0.04 / 0.10 = 0.40
    # Asset 1: -0.01 / (0.03 - (-0.05)) = -0.01 / 0.08 = -0.125
    assert np.isclose(snr[0], 0.40, atol=1e-3)
    assert np.isclose(snr[1], -0.125, atol=1e-3)

    # Price space: P=100.0, q50=104.0, q10=98.0, q90=108.0
    p = np.array([100.0])
    q10_p = np.array([98.0])
    q50_p = np.array([104.0])
    q90_p = np.array([108.0])
    snr_p = compute_quantile_snr_signal(q10_p, q50_p, q90_p, current_price=p)
    # (104 - 100) / (108 - 98) = 4 / 10 = 0.40
    assert np.isclose(snr_p[0], 0.40, atol=1e-3)


def test_compute_portfolio_metrics():
    # 36 periods (approx 1 year with stride=7)
    # Constant 1% per period, zero turnover
    period_rets = [0.01] * 36
    period_turnovers = [0.0] * 36

    metrics = compute_portfolio_metrics(period_rets, period_turnovers, stride=7, cost_bps=10.0)
    # (1.01)^36 - 1 ~= 0.43076 (43.08%)
    assert metrics.cagr > 40.0
    # Zero sample std -> zero volatility
    assert metrics.volatility == 0.0
    assert metrics.max_drawdown == 0.0

    # Drawdown test: alternating +10%, -10%
    alt_rets = [0.10, -0.10] * 10
    metrics_alt = compute_portfolio_metrics(alt_rets, [1.0] * len(alt_rets), stride=7)
    assert metrics_alt.max_drawdown >= 10.0
    assert metrics_alt.volatility > 0.0


def test_simulate_portfolio_strategy():
    rebalance_dates = ["2021-07-01", "2021-07-10", "2021-07-20"]
    universe = {
        "2021-07-01": {
            "AAPL": {"realized_return": 0.05, "timexer_q50": 0.04, "timexer_raw_snr": 0.8, "timexer_aci_snr": 0.7, "garch_q50": 0.01, "garch_snr": 0.2},
            "NVDA": {"realized_return": 0.10, "timexer_q50": 0.06, "timexer_raw_snr": 1.2, "timexer_aci_snr": 1.1, "garch_q50": 0.02, "garch_snr": 0.3},
            "MSFT": {"realized_return": -0.02, "timexer_q50": -0.01, "timexer_raw_snr": -0.2, "timexer_aci_snr": -0.2, "garch_q50": 0.00, "garch_snr": 0.0},
            "INTC": {"realized_return": -0.05, "timexer_q50": -0.03, "timexer_raw_snr": -0.6, "timexer_aci_snr": -0.5, "garch_q50": -0.01, "garch_snr": -0.2},
        },
        "2021-07-10": {
            "AAPL": {"realized_return": 0.02, "timexer_q50": 0.02, "timexer_raw_snr": 0.4, "timexer_aci_snr": 0.4, "garch_q50": 0.01, "garch_snr": 0.1},
            "NVDA": {"realized_return": -0.03, "timexer_q50": -0.02, "timexer_raw_snr": -0.3, "timexer_aci_snr": -0.3, "garch_q50": 0.00, "garch_snr": 0.0},
            "MSFT": {"realized_return": 0.01, "timexer_q50": 0.01, "timexer_raw_snr": 0.2, "timexer_aci_snr": 0.2, "garch_q50": 0.01, "garch_snr": 0.1},
            "INTC": {"realized_return": -0.01, "timexer_q50": -0.01, "timexer_raw_snr": -0.2, "timexer_aci_snr": -0.2, "garch_q50": -0.01, "garch_snr": -0.1},
        },
    }

    # Test Buy & Hold
    bh_metrics = simulate_portfolio_strategy(rebalance_dates, universe, "buy_and_hold", top_k=2, cost_bps=10.0)
    assert isinstance(bh_metrics, BacktestMetrics)

    # Test Long-Short
    ls_metrics = simulate_portfolio_strategy(rebalance_dates, universe, "timexer_raw_long_short", top_k=1, cost_bps=10.0)
    assert isinstance(ls_metrics, BacktestMetrics)

    # Test Long-Only
    lo_metrics = simulate_portfolio_strategy(rebalance_dates, universe, "timexer_raw_long_only", top_k=2, cost_bps=10.0)
    assert isinstance(lo_metrics, BacktestMetrics)


def test_aggregate_and_format_tables():
    seed_res = [
        {
            "buy_and_hold": BacktestMetrics(10.0, 15.0, 0.67, 0.90, 8.0, 0.1),
            "directional_long_short": BacktestMetrics(12.0, 14.0, 0.85, 1.10, 6.0, 1.2),
            "directional_long_only": BacktestMetrics(11.0, 16.0, 0.69, 0.95, 7.5, 0.8),
            "garch_long_short": BacktestMetrics(9.0, 13.0, 0.69, 0.92, 6.5, 1.1),
            "garch_long_only": BacktestMetrics(10.5, 15.5, 0.68, 0.91, 7.8, 0.7),
            "timexer_raw_long_short": BacktestMetrics(15.0, 12.0, 1.25, 1.60, 5.0, 1.3),
            "timexer_raw_long_only": BacktestMetrics(13.5, 14.0, 0.96, 1.30, 6.0, 0.9),
            "timexer_aci_long_short": BacktestMetrics(16.0, 11.5, 1.39, 1.80, 4.5, 1.3),
            "timexer_aci_long_only": BacktestMetrics(14.0, 13.5, 1.04, 1.40, 5.5, 0.9),
        },
        {
            "buy_and_hold": BacktestMetrics(10.2, 15.1, 0.68, 0.91, 8.1, 0.1),
            "directional_long_short": BacktestMetrics(12.5, 14.2, 0.88, 1.12, 6.1, 1.2),
            "directional_long_only": BacktestMetrics(11.2, 16.1, 0.70, 0.96, 7.6, 0.8),
            "garch_long_short": BacktestMetrics(9.2, 13.1, 0.70, 0.93, 6.6, 1.1),
            "garch_long_only": BacktestMetrics(10.7, 15.6, 0.69, 0.92, 7.9, 0.7),
            "timexer_raw_long_short": BacktestMetrics(15.4, 12.2, 1.26, 1.62, 5.1, 1.3),
            "timexer_raw_long_only": BacktestMetrics(13.8, 14.1, 0.98, 1.32, 6.1, 0.9),
            "timexer_aci_long_short": BacktestMetrics(16.3, 11.6, 1.41, 1.82, 4.6, 1.3),
            "timexer_aci_long_only": BacktestMetrics(14.2, 13.6, 1.04, 1.42, 5.6, 0.9),
        },
    ]

    summary = aggregate_seed_metrics(seed_res)
    assert "timexer_aci_long_short" in summary
    assert "cagr" in summary["timexer_aci_long_short"]

    console_txt, md_txt, tex_txt = format_tables(summary, fold=1)
    assert "TimeXer Hierarchical (ACI Calibrated SNR)" in console_txt
    assert "| **TimeXer Hierarchical (ACI Calibrated SNR)** |" in md_txt
    assert "\\begin{table}" in tex_txt
    assert "\\end{table}" in tex_txt
