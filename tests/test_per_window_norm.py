"""Per-window RevIN-style normalize + train_start filter (synthetic, no FNSPID)."""

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


def test_per_window_normalization_properties() -> None:
    t, c, h = 20, 5, 4
    rng = np.random.default_rng(1)
    raw = rng.normal(80.0, 12.0, size=(90, c)).astype(np.float32)
    dates = pd.bdate_range("2016-01-04", periods=90)
    ts = TickerSeries(features=raw, target=raw[:, 0], dates=pd.Series(dates))
    end_idx = 50
    wi = WindowIndex("ZZZ", end_idx=end_idx, start_idx=end_idx - t, end_date="2016-03-15")
    ds = FNSPIDForecastDataset(
        [wi],
        store=object(),
        series={"ZZZ": ts},
        horizon=h,
        text_dim=2,
        text_enabled=False,
        normalize="per_window",
        target_idx=0,
    )
    x = ds[0]["x"].numpy()
    for ch in range(c):
        assert abs(float(x[:, ch].mean())) < 1e-5
        assert abs(float(x[:, ch].std()) - 1.0) < 1e-5


def test_target_denormalization_exact() -> None:
    t, h = 16, 5
    rng = np.random.default_rng(2)
    raw = rng.normal(150.0, 8.0, size=(70, 3)).astype(np.float32)
    dates = pd.bdate_range("2017-01-03", periods=70)
    ts = TickerSeries(features=raw, target=raw[:, 0], dates=pd.Series(dates))
    end_idx = 30
    wi = WindowIndex("QQQ", end_idx=end_idx, start_idx=end_idx - t, end_date="2017-02-14")
    ds = FNSPIDForecastDataset(
        [wi],
        store=object(),
        series={"QQQ": ts},
        horizon=h,
        text_dim=2,
        text_enabled=False,
        normalize="per_window",
        target_idx=0,
    )
    item = ds[0]
    y_raw = item["y"] * item["y_std"] + item["y_mean"]
    np.testing.assert_allclose(
        y_raw.numpy(), raw[end_idx : end_idx + h, 0], rtol=1e-5, atol=1e-5
    )
    batch = forecast_collate([item, item])
    assert batch["y_mean"].shape == (2,)
    assert batch["y_std"].dtype == torch.float32


def test_train_start_filtering(tmp_path) -> None:
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
    for wi in ds.indices:
        start_date = _as_naive_day(ts.dates.iloc[wi.start_idx])
        assert start_date >= train_start
        assert _as_naive_day(wi.end_date) >= train_start


def test_legacy_zscore_fallback(tmp_path) -> None:
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
    lookback, horizon = 5, 2
    end_idx = lookback
    wi = WindowIndex(
        "BBB",
        end_idx=end_idx,
        start_idx=end_idx - lookback,
        end_date=str(_as_naive_day(ts.dates.iloc[end_idx]).date()),
    )
    ds = FNSPIDForecastDataset(
        [wi],
        store=TickerDataStore(tmp_path, "x", ["close"], "close"),
        series={"BBB": ts},
        horizon=horizon,
        text_dim=2,
        text_enabled=False,
        normalize="per_ticker_zscore",
        target_idx=0,
    )
    item = ds[0]
    np.testing.assert_allclose(item["x"].numpy(), ts.features[0:lookback], rtol=1e-5)
    np.testing.assert_allclose(item["y"].numpy(), ts.target[lookback : lookback + horizon], rtol=1e-5)
