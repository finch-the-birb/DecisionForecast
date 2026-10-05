"""Causal OHLCV features: formulas and no look-ahead (Sprint 9)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.technical import (
    EPS,
    RETURN_LAGS,
    REVIN_STD_FLOOR,
    WINDOWS,
    base_feature_names,
    compute_technical_features,
    revin_causal_rows,
    revin_window,
    technical_feature_names,
)


def _prices(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=n)))
    open_ = close * np.exp(rng.normal(0.0, 0.003, size=n))
    high = np.maximum(open_, close) * np.exp(rng.uniform(0.0, 0.01, size=n))
    low = np.minimum(open_, close) * np.exp(-rng.uniform(0.0, 0.01, size=n))
    volume = rng.uniform(1_000.0, 5_000.0, size=n)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2018-01-02", periods=n),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13"]
            ),
            "open": [10.0, 11.0, 12.0, 11.5, 13.0, 14.0],
            "high": [12.0, 13.0, 14.0, 13.0, 15.0, 16.0],
            "low": [9.0, 10.0, 11.0, 10.5, 12.0, 13.0],
            "close": [11.0, 12.0, 13.0, 12.0, 14.0, 15.0],
            "volume": [100.0, 200.0, 300.0, 100.0, 400.0, 500.0],
        }
    )


def _recursive_ema(values: np.ndarray, span: int) -> np.ndarray:
    """Span EMA, alpha = 2 / (span + 1), seeded at the first finite observation."""
    alpha = 2.0 / (span + 1.0)
    out = np.full(len(values), np.nan)
    started = False
    prev = 0.0
    valid = 0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            if not started:
                continue
            break
        prev = float(value) if not started else alpha * float(value) + (1.0 - alpha) * prev
        started = True
        valid += 1
        if valid >= span:
            out[i] = prev
    return out


def test_default_inventory_is_68_base_by_5_columns() -> None:
    n_windows = len(WINDOWS)
    n_base = len(RETURN_LAGS) + 2 + 9 * n_windows + 1 + 4
    assert n_base == 68
    assert len(base_feature_names()) == n_base
    names = technical_feature_names()
    assert len(names) == n_base * 5
    assert len(names) == len(set(names))
    feats = compute_technical_features(_prices(8))
    assert list(feats.columns) == names
    assert len(feats) == 8


def test_prefix_matches_full_series() -> None:
    """Features at t computed on bars 0..t equal the same rows of the full frame."""
    df = _prices(90, seed=1)
    full = compute_technical_features(df)
    for end in (30, 61, 70, 89):
        part = compute_technical_features(df.iloc[: end + 1])
        pd.testing.assert_frame_equal(full.iloc[: end + 1], part, rtol=1e-12, atol=1e-12)


def test_future_bars_do_not_change_past() -> None:
    df = _prices(90, seed=2)
    cut = 70
    edited = df.copy()
    idx = edited.index[cut:]
    edited.loc[idx, "open"] *= 1.3
    edited.loc[idx, "high"] *= 1.7
    edited.loc[idx, "low"] *= 0.8
    edited.loc[idx, "close"] *= 1.5
    edited.loc[idx, "volume"] += 2_500.0
    edited.loc[edited.index[-1], "date"] = edited.loc[edited.index[-1], "date"] + pd.Timedelta(days=1)
    original = compute_technical_features(df)
    changed = compute_technical_features(edited)
    pd.testing.assert_frame_equal(original.iloc[:cut], changed.iloc[:cut], rtol=1e-12, atol=1e-12)
    assert not np.allclose(
        original["log_ret_k1"].iloc[cut:].to_numpy(),
        changed["log_ret_k1"].iloc[cut:].to_numpy(),
        equal_nan=True,
    )


def test_future_spike_does_not_enter_bollinger() -> None:
    df = _prices(30)
    df[["open", "high", "low", "close"]] = 10.0
    calm = compute_technical_features(df, windows=(5,), return_lags=(1,), feature_lags=())
    spiked = df.copy()
    spiked.loc[spiked.index[-1], ["close", "high"]] = 50.0
    moved = compute_technical_features(spiked, windows=(5,), return_lags=(1,), feature_lags=())
    pd.testing.assert_series_equal(
        calm["bb_pctb_n5"].iloc[:-1],
        moved["bb_pctb_n5"].iloc[:-1],
        check_names=False,
    )
    assert not np.isclose(calm["bb_pctb_n5"].iloc[-1], moved["bb_pctb_n5"].iloc[-1])


def test_warmup_stays_missing_and_log_return_uses_the_past() -> None:
    df = _prices(70, seed=5)
    feats = compute_technical_features(df)
    assert feats["log_ret_k60"].iloc[:60].isna().all()
    assert feats["parkinson_n60"].iloc[:59].isna().all()
    close = df["close"].to_numpy()
    np.testing.assert_allclose(
        feats["log_ret_k60"].iloc[60],
        np.log(close[60]) - np.log(close[0]),
    )
    assert np.isfinite(feats["parkinson_n60"].iloc[59])


def test_lags_and_delta_are_past_shifts() -> None:
    feats = compute_technical_features(
        _prices(40, seed=6),
        windows=(3, 5),
        return_lags=(1,),
        feature_lags=(1, 2, 3),
    )
    for base in ("log_ret_k1", "parkinson_n3", "day_sin"):
        pd.testing.assert_series_equal(
            feats[f"{base}_lag2"], feats[base].shift(2), check_names=False
        )
        pd.testing.assert_series_equal(
            feats[f"{base}_delta"], feats[base] - feats[base].shift(1), check_names=False
        )


def test_rolling_indicators_match_their_own_window() -> None:
    df = _prices(40, seed=3)
    kwargs = dict(windows=(5,), return_lags=(1, 2), feature_lags=(1,))
    full = compute_technical_features(df, **kwargs)
    t = 25
    local = compute_technical_features(df.iloc[t - 4 : t + 1], **kwargs)
    for column in (
        "parkinson_n5",
        "gk_n5",
        "bb_pctb_n5",
        "bb_width_n5",
        "vwap_spread_n5",
        "volume_z_n5",
    ):
        np.testing.assert_allclose(full[column].iloc[t], local[column].iloc[-1], rtol=1e-12, atol=1e-12)
    wider = compute_technical_features(df.iloc[t - 5 : t + 1], **kwargs)
    for column in ("natr_n5", "obv_z_n5"):
        np.testing.assert_allclose(full[column].iloc[t], wider[column].iloc[-1], rtol=1e-12, atol=1e-12)
    returns = compute_technical_features(df.iloc[t - 2 : t + 1], **kwargs)
    np.testing.assert_allclose(full["log_ret_k2"].iloc[t], returns["log_ret_k2"].iloc[-1])


def test_block_formulas_match_independent_oracle() -> None:
    df = _fixture()
    feats = compute_technical_features(df, windows=(3,), return_lags=(1, 2), feature_lags=(1,))
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    volume = df["volume"].to_numpy()
    t = 5
    window = 3
    sl = slice(t - window + 1, t + 1)

    assert np.isnan(feats["log_ret_k1"].iloc[0])
    np.testing.assert_allclose(feats["log_ret_k1"].iloc[t], np.log(close[t]) - np.log(close[t - 1]))
    np.testing.assert_allclose(feats["log_ret_k2"].iloc[t], np.log(close[t]) - np.log(close[t - 2]))
    np.testing.assert_allclose(feats["intraday_ret"].iloc[t], (close[t] - open_[t]) / open_[t])
    np.testing.assert_allclose(
        feats["body_range"].iloc[t],
        abs(close[t] - open_[t]) / (high[t] - low[t] + EPS),
    )

    hl = np.log(high[sl] / low[sl])
    park = np.sqrt(np.mean(hl**2 / (4.0 * np.log(2.0))))
    co = np.log(close[sl] / open_[sl])
    gk_bar = 0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2
    assert gk_bar.mean() > 0.0
    np.testing.assert_allclose(feats["parkinson_n3"].iloc[t], park)
    np.testing.assert_allclose(feats["gk_n3"].iloc[t], np.sqrt(gk_bar.mean()))

    true_range = []
    for i in range(t - window + 1, t + 1):
        hl_i = high[i] - low[i]
        true_range.append(
            max(hl_i, abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        )
    np.testing.assert_allclose(feats["natr_n3"].iloc[t], np.mean(true_range) / close[t] * 100.0)

    delta = np.diff(close, prepend=np.nan)
    up = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan)
    down = np.where(np.isfinite(delta), np.maximum(-delta, 0.0), np.nan)
    avg_up = _recursive_ema(up, window)
    avg_down = _recursive_ema(down, window)
    rsi = 100.0 - 100.0 / (1.0 + avg_up / (avg_down + EPS))
    np.testing.assert_allclose(feats["rsi_n3"].to_numpy(), rsi, equal_nan=True)

    mean = close[sl].mean()
    sigma = close[sl].std(ddof=0)
    upper = mean + 2.0 * sigma
    lower = mean - 2.0 * sigma
    np.testing.assert_allclose(
        feats["bb_pctb_n3"].iloc[t],
        (close[t] - lower) / (upper - lower + EPS),
    )
    np.testing.assert_allclose(feats["bb_width_n3"].iloc[t], (upper - lower) / mean)

    typical = (high + low + close) / 3.0
    vwap = np.sum(typical[sl] * volume[sl]) / np.sum(volume[sl])
    np.testing.assert_allclose(feats["vwap_spread_n3"].iloc[t], (close[t] - vwap) / vwap)
    vol_std = volume[sl].std(ddof=0)
    np.testing.assert_allclose(
        feats["volume_z_n3"].iloc[t],
        (volume[t] - volume[sl].mean()) / (vol_std + EPS),
    )

    direction = np.sign(np.diff(close, prepend=np.nan))
    direction[0] = 0.0
    obv = np.cumsum(volume * direction)
    obv_window = obv[sl]
    np.testing.assert_allclose(
        feats["obv_z_n3"].iloc[t],
        (obv[t] - obv_window.mean()) / (obv_window.std(ddof=0) + EPS),
    )

    dow = df["date"].dt.dayofweek.to_numpy()
    tau = (df["date"].dt.day.to_numpy() - 1) / df["date"].dt.days_in_month.to_numpy()
    np.testing.assert_allclose(feats["day_sin"].to_numpy(), np.sin(2.0 * np.pi * dow / 5.0))
    np.testing.assert_allclose(feats["day_cos"].to_numpy(), np.cos(2.0 * np.pi * dow / 5.0))
    np.testing.assert_allclose(feats["month_sin"].to_numpy(), np.sin(2.0 * np.pi * tau))
    np.testing.assert_allclose(feats["month_cos"].to_numpy(), np.cos(2.0 * np.pi * tau))
    assert dow[0] == 0
    np.testing.assert_allclose(tau[0], 5.0 / 31.0)


def test_macd_histogram_matches_recursive_ema() -> None:
    df = _prices(80, seed=7)
    feats = compute_technical_features(df, windows=(3,), return_lags=(1,), feature_lags=())
    close = df["close"].to_numpy()
    macd = _recursive_ema(close, 12) - _recursive_ema(close, 26)
    signal = _recursive_ema(macd, 9)
    expected = (macd - signal) / close
    got = feats["macd_hist_norm"].to_numpy()
    assert np.isfinite(expected).sum() > 0
    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(expected))
    np.testing.assert_allclose(got[np.isfinite(expected)], expected[np.isfinite(expected)])


def test_flat_prices_leave_non_calendar_features_at_zero() -> None:
    df = _prices(90)
    df[["open", "high", "low", "close"]] = 25.0
    df["volume"] = 100.0
    feats = compute_technical_features(df)
    last = feats.iloc[-1]
    for name, value in last.items():
        if str(name).startswith(("day_", "month_")):
            assert np.isfinite(value)
            continue
        assert np.isfinite(value)
        assert abs(float(value)) < 1e-8


def test_zero_volume_window_stays_missing() -> None:
    df = _fixture()
    df["volume"] = 0.0
    feats = compute_technical_features(df, windows=(3,), return_lags=(1,), feature_lags=())
    assert feats["vwap_spread_n3"].isna().all()


def test_input_frame_is_not_mutated() -> None:
    df = _prices(12)
    before = df.copy()
    compute_technical_features(df, windows=(3,), return_lags=(1,), feature_lags=(1,))
    pd.testing.assert_frame_equal(df, before)


def test_datetime_index_and_column_case() -> None:
    df = _prices(12).rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    df = df.set_index("date")
    feats = compute_technical_features(df, windows=(3,), return_lags=(1,), feature_lags=())
    assert feats.index.equals(df.index)
    assert "intraday_ret_delta" in feats.columns
    assert "intraday_ret_lag1" not in feats.columns
    assert np.isfinite(feats["day_cos"].iloc[0])


def test_timezone_aware_date_keeps_the_calendar_day() -> None:
    df = _fixture()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    feats = compute_technical_features(df, windows=(3,), return_lags=(1,), feature_lags=())
    np.testing.assert_allclose(feats["day_sin"].iloc[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(feats["day_cos"].iloc[0], 1.0, atol=1e-12)


def test_rejects_leaky_or_invalid_inputs() -> None:
    df = _prices(6)
    with pytest.raises(ValueError, match="looks ahead"):
        compute_technical_features(df, feature_lags=(-1,))
    with pytest.raises(ValueError, match="unique"):
        compute_technical_features(df, windows=(3, 3))
    with pytest.raises(KeyError, match="volume"):
        compute_technical_features(df.drop(columns=["volume"]))

    missing = df.copy()
    missing.loc[0, "close"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_technical_features(missing)

    negative = df.copy()
    negative.loc[0, "low"] = -1.0
    with pytest.raises(ValueError, match="positive"):
        compute_technical_features(negative)

    unsorted = df.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_technical_features(unsorted)

    mixed = df.copy()
    mixed["ticker"] = ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"]
    with pytest.raises(ValueError, match="single ticker"):
        compute_technical_features(mixed)


def test_revin_window_matches_per_window_dataset_rule() -> None:
    rng = np.random.default_rng(8)
    window = rng.normal(loc=4.0, scale=2.0, size=(16, 5))
    norm, mean, std = revin_window(window)
    expected_mean = window.mean(axis=0)
    expected_std = window.std(axis=0)
    expected_std = np.where(expected_std < REVIN_STD_FLOOR, 1.0, expected_std)
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(std, expected_std)
    np.testing.assert_allclose(norm, (window - expected_mean) / expected_std)
    np.testing.assert_allclose(norm.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(norm.std(axis=0), 1.0, atol=1e-10)

    constant = np.full((8, 2), 3.0)
    norm_c, mean_c, std_c = revin_window(constant)
    np.testing.assert_allclose(norm_c, 0.0)
    np.testing.assert_allclose(mean_c, 3.0)
    np.testing.assert_allclose(std_c, 1.0)

    with pytest.raises(ValueError, match="finite"):
        broken = window.copy()
        broken[1, 1] = np.nan
        revin_window(broken)


def test_revin_causal_rows_do_not_see_the_future_or_backfill() -> None:
    rng = np.random.default_rng(9)
    values = rng.normal(size=(24, 3))
    lookback = 5
    got = revin_causal_rows(values, lookback)
    naive = np.full_like(values, np.nan)
    for end in range(lookback, len(values) + 1):
        window = values[end - lookback : end]
        naive[end - 1] = revin_window(window)[0][-1]
    np.testing.assert_allclose(got, naive, equal_nan=True)
    assert np.isnan(got[: lookback - 1]).all()
    assert np.isfinite(got[lookback - 1 :]).all()

    edited = values.copy()
    edited[18:] += 50.0
    np.testing.assert_allclose(
        revin_causal_rows(values, lookback)[:18],
        revin_causal_rows(edited, lookback)[:18],
    )

    holed = values.copy()
    holed[10, 0] = np.nan
    holed_out = revin_causal_rows(holed, lookback)
    assert np.isfinite(holed_out[9]).all()
    assert np.isnan(holed_out[10:15]).all()
    assert np.isfinite(holed_out[15]).all()

    with pytest.raises(ValueError, match="lookback"):
        revin_causal_rows(values, 0)


def test_revin_on_a_feature_window_uses_only_that_slice() -> None:
    feats = compute_technical_features(_prices(80, seed=10))
    values = feats.to_numpy(dtype=np.float64)
    lookback = 10
    # rsi_n60 / log_ret_k60 become finite at index 60; their lag-3 columns at 63.
    end = 75
    window = values[end - lookback + 1 : end + 1]
    assert np.isfinite(window).all()
    norm, _, _ = revin_window(window)
    causal = revin_causal_rows(values, lookback)
    np.testing.assert_allclose(causal[end], norm[-1])
    edited = values.copy()
    edited[end + 1 :] += 25.0
    np.testing.assert_allclose(
        revin_causal_rows(edited, lookback)[: end + 1],
        causal[: end + 1],
        equal_nan=True,
    )


def test_implementation_does_not_name_lookahead_operations() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "features" / "technical.py"
    ).read_text(encoding="utf-8")
    assert "center=True" not in source
    assert "bfill(" not in source
    assert "backfill(" not in source
    assert "fillna(" not in source
    assert ".shift(-" not in source
    assert "center=False" in source
    assert "adjust=False" in source
