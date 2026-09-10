"""Synthetic FNSPID window protocol: train_start + per_window normalize (no FNSPID)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.collate import forecast_collate
from src.data.dataset import (
    FNSPIDForecastDataset,
    TickerDataStore,
    TickerSeries,
    WindowIndex,
    _as_naive_day,
    _load_ticker_series,
)


def _write_csv(tmp_path, ticker: str, dates: pd.DatetimeIndex, close: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "close": close,
            "volume": np.ones(len(dates)),
            "open": close,
            "high": close,
            "low": close,
        }
    )
    df.to_csv(tmp_path / f"{ticker}.csv", index=False)


def test_train_start_drops_early_rows(tmp_path) -> None:
    dates = pd.bdate_range("2014-12-01", "2015-06-01")
    close = np.linspace(10.0, 50.0, len(dates))
    _write_csv(tmp_path, "AAA", dates, close)
    train_start = _as_naive_day("2015-01-01")
    ts = _load_ticker_series(
        tmp_path,
        "AAA",
        ["close", "volume", "open", "high", "low"],
        "close",
        train_end=_as_naive_day("2021-12-31"),
        train_start=train_start,
        normalize="per_window",
    )
    assert ts.dates.min() >= train_start
    lookback, horizon = 10, 3
    indices: list[WindowIndex] = []
    max_end = len(ts.features) - horizon
    for end_idx in range(lookback, max_end):
        indices.append(
            WindowIndex(
                ticker="AAA",
                end_idx=end_idx,
                start_idx=end_idx - lookback,
                end_date=str(_as_naive_day(ts.dates.iloc[end_idx]).date()),
            )
        )
    ds = FNSPIDForecastDataset(
        indices,
        store=TickerDataStore(tmp_path, "x", ["close"], "close"),
        series={"AAA": ts},
        horizon=horizon,
        text_dim=4,
        text_enabled=False,
        normalize="per_window",
        target_idx=0,
    )
    first = min(wi.end_date for wi in ds.indices)
    # First window uses dates[0:lookback]; end_date is dates[lookback] = T trading days after first kept bar.
    expected = str(_as_naive_day(ts.dates.iloc[lookback]).date())
    assert first == expected
    assert pd.Timestamp(first) >= train_start


def test_per_window_zero_mean_unit_std_and_denorm() -> None:
    t, c, h = 20, 3, 4
    rng = np.random.default_rng(0)
    raw = rng.normal(100.0, 5.0, size=(80, c)).astype(np.float32)
    dates = pd.bdate_range("2016-01-04", periods=80)
    ts = TickerSeries(features=raw, target=raw[:, 0], dates=pd.Series(dates))
    end_idx = 40
    wi = WindowIndex("ZZZ", end_idx=end_idx, start_idx=end_idx - t, end_date="2016-03-01")
    ds = FNSPIDForecastDataset(
        [wi],
        store=object(),  # unused when text is disabled
        series={"ZZZ": ts},
        horizon=h,
        text_dim=2,
        text_enabled=False,
        normalize="per_window",
        target_idx=0,
    )
    item = ds[0]
    x = item["x"].numpy()
    assert abs(float(x[:, 0].mean())) < 1e-5
    assert abs(float(x[:, 0].std()) - 1.0) < 1e-5
    y_hat = item["y"] * item["y_std"] + item["y_mean"]
    raw_y = raw[end_idx : end_idx + h, 0]
    np.testing.assert_allclose(y_hat.numpy(), raw_y, rtol=1e-5, atol=1e-5)
    batch = forecast_collate([item, item])
    assert batch["y_mean"].shape == (2,)
    assert batch["y_std"].shape == (2,)


def test_per_ticker_zscore_unchanged(tmp_path) -> None:
    dates = pd.bdate_range("2015-01-01", periods=40)
    close = np.arange(40, dtype=np.float64) + 10.0
    _write_csv(tmp_path, "BBB", dates, close)
    train_end = _as_naive_day("2015-02-02")
    ts = _load_ticker_series(
        tmp_path,
        "BBB",
        ["close", "volume", "open", "high", "low"],
        "close",
        train_end=train_end,
        train_start=_as_naive_day("2015-01-01"),
        normalize="per_ticker_zscore",
    )
    train_mask = (ts.dates <= train_end).to_numpy()
    ref = close[train_mask]
    mean, std = ref.mean(), ref.std()
    np.testing.assert_allclose(ts.target[train_mask], (ref - mean) / std, rtol=1e-5)


def test_split_by_last_target_date() -> None:
    """Train/val/test bucket by last target bar date (unchanged)."""
    lookback, horizon = 5, 2
    dates = pd.bdate_range("2021-12-01", periods=80)
    train_end = _as_naive_day("2021-12-31")
    val_end = _as_naive_day("2022-01-14")
    raw = np.ones((80, 1), dtype=np.float32)
    ts = TickerSeries(features=raw, target=raw[:, 0], dates=pd.Series(dates))
    buckets = {"train": 0, "val": 0, "test": 0}
    max_end = len(ts.features) - horizon
    for end_idx in range(lookback, max_end):
        last_target = _as_naive_day(ts.dates.iloc[end_idx + horizon - 1])
        if last_target <= train_end:
            buckets["train"] += 1
        elif last_target <= val_end:
            buckets["val"] += 1
        else:
            buckets["test"] += 1
    assert buckets["train"] > 0 and buckets["val"] > 0 and buckets["test"] > 0
