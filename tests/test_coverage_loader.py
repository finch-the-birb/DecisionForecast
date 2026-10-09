"""Coverage without __getitem__, and DataLoader worker kwargs (synthetic, no FNSPID)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.collate import make_forecast_loader
from src.data.dataset import (
    FNSPIDForecastDataset,
    TickerSeries,
    WindowIndex,
    log_text_coverage,
    split_text_coverage,
)


def _series(
    ticker: str,
    n_days: int,
    dim: int,
    seed: int,
) -> TickerSeries:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_days, 5)).astype(np.float32)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    text = rng.normal(size=(n_days, dim)).astype(np.float32)
    has_news = rng.random(n_days) > 0.55
    # Force a few empty-news stretches so zero_windows is not 0 or 1.
    has_news[:8] = False
    has_news[20:28] = True
    return TickerSeries(
        features=features,
        target=features[:, 0],
        dates=pd.Series(dates),
        text_seq=text,
        has_news=has_news,
    )


def _windows(ticker: str, dates: pd.Series, lookback: int, n_windows: int) -> list[WindowIndex]:
    out: list[WindowIndex] = []
    for j in range(n_windows):
        end_idx = lookback + j
        out.append(
            WindowIndex(
                ticker=ticker,
                end_idx=end_idx,
                start_idx=end_idx - lookback,
                end_date=str(pd.Timestamp(dates.iloc[end_idx]).date()),
            )
        )
    return out


def _dataset(
    tickers: dict[str, TickerSeries],
    lookback: int = 12,
    horizon: int = 4,
    n_windows: int = 24,
    window_agg: str = "recency_weighted",
    renormalize: bool = False,
    text_enabled: bool = True,
) -> FNSPIDForecastDataset:
    indices: list[WindowIndex] = []
    dim = 8
    for ticker, ts in tickers.items():
        indices.extend(_windows(ticker, ts.dates, lookback, n_windows))
        if ts.text_seq is not None:
            dim = int(ts.text_seq.shape[1])
    return FNSPIDForecastDataset(
        indices,
        store=object(),
        series=tickers,
        horizon=horizon,
        text_dim=dim,
        text_enabled=text_enabled,
        window_agg=window_agg,
        decay_lambda=0.03,
        renormalize=renormalize,
        normalize="per_window",
        target_idx=0,
    )


def _getitem_coverage(ds: FNSPIDForecastDataset) -> tuple[float, float, float]:
    fracs: list[float] = []
    norms: list[float] = []
    for i in range(len(ds)):
        item = ds[i]
        fracs.append(float(item["has_news_frac"]))
        norms.append(float(torch.linalg.vector_norm(item["text"]).item()))
    arr = np.asarray(fracs, dtype=np.float64)
    return float(arr.mean()), float((arr == 0.0).mean()), float(np.mean(norms))


def test_split_text_coverage_matches_getitem_recency() -> None:
    tickers = {
        "AAA": _series("AAA", 80, 8, seed=0),
        "BBB": _series("BBB", 80, 8, seed=1),
    }
    ds = _dataset(tickers, window_agg="recency_weighted")
    fast = split_text_coverage(ds)
    slow = _getitem_coverage(ds)
    np.testing.assert_allclose(fast, slow, rtol=1e-5, atol=1e-5)


def test_split_text_coverage_matches_getitem_mean_and_last() -> None:
    ts = _series("AAA", 70, 6, seed=2)
    for agg in ("mean", "last"):
        ds = _dataset({"AAA": ts}, window_agg=agg, n_windows=16, lookback=10)
        fast = split_text_coverage(ds)
        slow = _getitem_coverage(ds)
        np.testing.assert_allclose(fast, slow, rtol=1e-5, atol=1e-5)


def test_split_text_coverage_does_not_call_getitem() -> None:
    ts = _series("AAA", 60, 4, seed=3)

    class Boom(FNSPIDForecastDataset):
        def __getitem__(self, idx: int) -> dict:
            raise AssertionError("coverage must not call __getitem__")

    base = _dataset({"AAA": ts}, n_windows=10)
    ds = Boom(
        base.indices,
        store=object(),
        series=base.series,
        horizon=base.horizon,
        text_dim=base.text_dim,
        text_enabled=True,
        window_agg="recency_weighted",
        decay_lambda=0.03,
        renormalize=False,
        normalize="per_window",
        target_idx=0,
    )
    mean_frac, zero_frac, mean_l2 = split_text_coverage(ds)
    assert mean_frac >= 0.0
    assert 0.0 <= zero_frac <= 1.0
    assert mean_l2 >= 0.0
    logged = log_text_coverage({"train": ds}, ds.series)
    assert "text_coverage_train_zero_windows" in logged
    assert "text_coverage_ticker_AAA" in logged


def test_text_disabled_coverage_is_zero() -> None:
    ts = _series("AAA", 50, 4, seed=4)
    ds = _dataset({"AAA": ts}, text_enabled=False, n_windows=8)
    mean_frac, zero_frac, mean_l2 = split_text_coverage(ds)
    assert mean_frac == 0.0
    assert zero_frac == 1.0
    assert mean_l2 == 0.0


def test_make_forecast_loader_worker_kwargs() -> None:
    ds = _dataset({"AAA": _series("AAA", 50, 4, seed=5)}, n_windows=12, lookback=8)
    workers = make_forecast_loader(
        ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=False
    )
    assert workers.num_workers == 2
    assert workers.persistent_workers is True
    iterator = iter(workers)
    try:
        batch = next(iterator)
    finally:
        del iterator
        del workers
    assert batch["x"].shape[0] == 4
    assert batch["x"].device.type == "cpu"

    pinned = make_forecast_loader(
        ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True
    )
    assert pinned.pin_memory is True

    serial = make_forecast_loader(
        ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=False
    )
    assert serial.num_workers == 0
    assert serial.pin_memory is False
    assert serial.persistent_workers is False
    batch0 = next(iter(serial))
    assert batch0["has_news_frac"].shape[0] == 4
