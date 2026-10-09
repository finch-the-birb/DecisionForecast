"""Selective scaling keeps flat windows finite and oscillators inside [-1, 1]."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import FNSPIDForecastDataset, TickerSeries, WindowIndex
from src.data.selective_norm import apply_static_and_robust, feature_roles, selective_window


def test_flat_price_and_volatility_do_not_divide_by_zero() -> None:
    names = ["close", "open", "log_ret_k1", "parkinson_n5", "gk_n5", "volume"]
    rows = 12
    values = np.column_stack(
        [
            np.full(rows, 40.0),
            np.full(rows, 39.0),
            np.full(rows, 0.0),
            np.full(rows, 0.02),
            np.full(rows, 0.01),
            np.full(rows, 1_000.0),
        ]
    )
    mask = np.ones(rows, dtype=bool)
    scaled = apply_static_and_robust(values, names, mask)
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled[:, 2:], 0.0, atol=1e-6)
    window, target_mean, target_std = selective_window(scaled[:8], feature_roles(names), close_idx=0)
    assert np.isfinite(window).all()
    np.testing.assert_allclose(window[:, 0], 0.0, atol=1e-6)
    assert target_mean == 40.0
    assert target_std == 1.0


def test_oscillators_and_calendar_stay_in_unit_range() -> None:
    names = ["rsi_n14", "bb_pctb_n20", "day_sin", "month_cos", "close"]
    values = np.array(
        [
            [0.0, 0.0, -1.0, 0.25, 10.0],
            [50.0, 0.5, 0.0, -0.5, 11.0],
            [100.0, 1.0, 1.0, 1.0, 12.0],
            [25.0, 0.25, 0.5, -1.0, 13.0],
        ],
        dtype=np.float64,
    )
    scaled = apply_static_and_robust(values, names, np.ones(len(values), dtype=bool))
    np.testing.assert_allclose(scaled[:, 0], [-1.0, 0.0, 1.0, -0.5])
    np.testing.assert_allclose(scaled[:, 1], [-1.0, 0.0, 1.0, -0.5])
    np.testing.assert_allclose(scaled[:, 2], values[:, 2])
    np.testing.assert_allclose(scaled[:, 3], values[:, 3])
    assert np.max(np.abs(scaled[:, :4])) <= 1.0
    roles = feature_roles(names)
    assert roles[:4] == ("rsi", "pctb", "calendar", "calendar")
    assert roles[4] == "level"


def test_dataset_window_keeps_close_stats_for_the_target() -> None:
    names = ["close", "rsi_n14", "log_ret_k1"]
    rows = np.array(
        [
            [10.0, 50.0, 0.0],
            [10.0, 80.0, 0.0],
            [10.0, 20.0, 0.0],
            [12.0, 50.0, 0.1],
            [14.0, 50.0, 0.1],
        ],
        dtype=np.float64,
    )
    stored = apply_static_and_robust(rows, names, np.ones(len(rows), dtype=bool))
    series = {
        "AAA": TickerSeries(
            features=stored.astype(np.float32),
            target=rows[:, 0].astype(np.float32),
            dates=pd.Series(pd.bdate_range("2021-01-04", periods=len(rows))),
        )
    }
    index = WindowIndex(ticker="AAA", end_idx=3, start_idx=0, end_date="2021-01-06")
    dataset = FNSPIDForecastDataset(
        [index],
        store=None,
        series=series,
        horizon=2,
        text_dim=15,
        text_enabled=False,
        normalize="selective",
        target_idx=0,
        feature_roles=feature_roles(names),
        close_idx=0,
    )
    item = dataset[0]
    assert np.isfinite(item["x"].numpy()).all()
    np.testing.assert_allclose(item["x"][:, 0].numpy(), 0.0, atol=1e-6)
    assert float(item["x"][:, 1].abs().max()) <= 1.0
    assert float(item["y_mean"]) == 10.0
    assert float(item["y_std"]) == 1.0
    assert item["y"].shape == (2,)


def test_bounded_columns_including_cfi_stay_in_unit_range() -> None:
    from src.data.selective_norm import _normalize_column

    # Test direct _normalize_column behavior
    arr = np.array([-50.0, -1.0, 0.0, 0.5, 1.0, 50.0])
    clipped = _normalize_column(arr, "bounded")
    np.testing.assert_allclose(clipped, [-1.0, -1.0, 0.0, 0.5, 1.0, 1.0])

    # Test through apply_static_and_robust with cfi, mrd, cgo
    names = ["cfi_n3", "mrd_n20", "cgo_n40", "close"]
    vals = np.array(
        [
            [100.0, 5.0, -10.0, 50.0],
            [0.0, 0.5, 0.2, 51.0],
            [25.0, -2.0, 1.5, 52.0],
        ]
    )
    scaled = apply_static_and_robust(vals, names, np.ones(len(vals), dtype=bool))
    assert np.max(np.abs(scaled[:, :3])) <= 1.0
    roles = feature_roles(names)
    assert roles == ("bounded", "bounded", "bounded", "level")
