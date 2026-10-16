"""Causal technical features from daily OHLCV bars.

Phase 4, Sprint 9. Row ``t`` is a function of bars ``0..t`` and of that row's
own calendar date. Warm-up stays missing: rolling windows are right-aligned
(``center=False``), exponential averages are recursive (``adjust=False``),
and later rows are never copied backwards into earlier ones.

Defaults follow the architecture note, section 3:

* log-returns at 7 lags, intraday return, body-to-range
* Parkinson, Garman-Klass, and NATR on ``N in {3, 5, 10, 20, 40, 60}``
* RSI, MACD histogram, Bollinger ``%B`` and width
* VWAP spread, volume z-score, OBV z-score
* weekday and month-phase cycles
* each base series plus lags 1..3 and a one-bar delta

That inventory is 68 base series x 5 columns = 340 features (the ~350-D pool).
Lags are shifts along the bar index (trading sessions), not calendar days.

Per-window RevIN is separate from feature generation. Fitting a scaler on the
whole series would leak future bars; call :func:`revin_window` on one lookback
slice, or :func:`revin_causal_rows` to read only the normalized last bar.
The std floor matches ``FNSPIDForecastDataset`` (``normalize="per_window"``).

Pass one ticker at a time. OBV and the EMAs are path-dependent, so a frame
that concatenates several tickers mixes their history.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

WINDOWS: tuple[int, ...] = (3, 5, 10, 20, 40, 60)
EII_WINDOWS: tuple[int, ...] = (5, 10, 20, 40)
RETURN_LAGS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 60)
FEATURE_LAGS: tuple[int, ...] = (1, 2, 3)
EPS: float = 1e-8
REVIN_STD_FLOOR: float = 1e-8
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

_PRICE_COLUMNS = ("open", "high", "low", "close")
_OHLCV_COLUMNS = (*_PRICE_COLUMNS, "volume")
_CALENDAR_COLUMNS = ("day_sin", "day_cos", "month_sin", "month_cos")


def base_feature_names(
    windows: Sequence[int] = WINDOWS,
    return_lags: Sequence[int] = RETURN_LAGS,
    eii_windows: Sequence[int] | None = None,
) -> list[str]:
    """Names of the base indicators, before lag and delta expansion."""
    windows, return_lags = _windows_and_returns(windows, return_lags)
    if eii_windows is None:
        eii_win = tuple(w for w in windows if w in EII_WINDOWS) or tuple(windows)
    else:
        eii_win = _as_unique_ints(eii_windows, "eii_windows", minimum=2)
    names: list[str] = [f"log_ret_k{k}" for k in return_lags]
    names += ["intraday_ret", "body_range"]
    names += [f"parkinson_n{n}" for n in windows]
    names += [f"gk_n{n}" for n in windows]
    names += [f"natr_n{n}" for n in windows]
    names += [f"rsi_n{n}" for n in windows]
    names.append("macd_hist_norm")
    names += [f"bb_pctb_n{n}" for n in windows]
    names += [f"bb_width_n{n}" for n in windows]
    names += [f"vwap_spread_n{n}" for n in windows]
    names += [f"volume_z_n{n}" for n in windows]
    names += [f"obv_z_n{n}" for n in windows]
    # Gary Becker (1991) externalities and microstructure features
    names += [f"eii_n{n}" for n in eii_win]
    names += [f"queue_acc_n{n}" for n in windows]
    names += [f"cfi_n{n}" for n in windows]
    names += [f"mrd_n{n}" for n in windows]
    names += [f"cgo_n{n}" for n in windows]
    names += [f"fft_harm_{k}" for k in range(1, 11)]
    names += list(_CALENDAR_COLUMNS)
    return names


def technical_feature_names(
    windows: Sequence[int] = WINDOWS,
    return_lags: Sequence[int] = RETURN_LAGS,
    feature_lags: Sequence[int] = FEATURE_LAGS,
    eii_windows: Sequence[int] | None = None,
) -> list[str]:
    """Column order produced by :func:`compute_technical_features`."""
    windows, return_lags = _windows_and_returns(windows, return_lags)
    feature_lags = _as_unique_ints(feature_lags, "feature_lags", minimum=1, allow_empty=True)
    if eii_windows is None:
        eii_win = tuple(w for w in windows if w in EII_WINDOWS) or tuple(windows)
    else:
        eii_win = _as_unique_ints(eii_windows, "eii_windows", minimum=2)
    names: list[str] = []
    for base in base_feature_names(windows, return_lags, eii_win):
        names.append(base)
        names.extend(f"{base}_lag{lag}" for lag in feature_lags)
        names.append(f"{base}_delta")
    return names


def compute_technical_features(
    frame: pd.DataFrame,
    *,
    windows: Sequence[int] = WINDOWS,
    return_lags: Sequence[int] = RETURN_LAGS,
    feature_lags: Sequence[int] = FEATURE_LAGS,
    eii_windows: Sequence[int] | None = None,
    eps: float = EPS,
) -> pd.DataFrame:
    """Causal OHLCV features aligned to ``frame``.

    Required columns (case-insensitive): open, high, low, close, volume.
    Calendar features read a ``date`` column, or a ``DatetimeIndex`` when that
    column is absent. Dates must be strictly increasing. Prices must be
    positive and volume must be non-negative.

    ``EMA_N`` is the recursive span average, ``alpha = 2 / (N + 1)``. That is
    the usual MACD definition. It is not Wilder smoothing (``alpha = 1 / N``).
    RSI uses this same ``EMA_N``, as written in the architecture note.
    Garman-Klass bar variances can be slightly negative; the window mean is
    clipped at 0 before the square root.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    windows, return_lags = _windows_and_returns(windows, return_lags)
    feature_lags = _as_unique_ints(feature_lags, "feature_lags", minimum=1, allow_empty=True)
    if eii_windows is None:
        eii_win = tuple(w for w in windows if w in EII_WINDOWS) or tuple(windows)
    else:
        eii_win = _as_unique_ints(eii_windows, "eii_windows", minimum=2)
    ohlcv, dates = _prepare_inputs(frame)
    base = _base_features(ohlcv, dates, windows, return_lags, eps, eii_win)
    if list(base.columns) != base_feature_names(windows, return_lags, eii_win):
        raise RuntimeError("base feature inventory does not match base_feature_names")
    expanded = _expand_lags(base, feature_lags)
    expected = technical_feature_names(windows, return_lags, feature_lags, eii_win)
    if list(expanded.columns) != expected:
        raise RuntimeError("expanded feature inventory does not match technical_feature_names")
    return expanded


def revin_window(
    values: np.ndarray,
    *,
    std_floor: float = REVIN_STD_FLOOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RevIN of one lookback window ``[T, F]``.

    Mean and population std are taken along time, independently per column.
    A column whose std is below ``std_floor`` is treated as scale 1, matching
    the dataset's per-window path. The window must already be finite: missing
    warm-up is not imputed here.
    """
    if std_floor <= 0.0:
        raise ValueError("std_floor must be positive")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"revin_window expected shape [T, F], got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("revin_window expected at least one row")
    if not np.isfinite(arr).all():
        raise ValueError("revin_window requires a finite window")
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < std_floor, 1.0, std)
    return (arr - mean) / std, mean, std


def revin_causal_rows(
    values: np.ndarray,
    lookback: int,
    *,
    std_floor: float = REVIN_STD_FLOOR,
) -> np.ndarray:
    """Point-in-time RevIN: row ``t`` uses only ``values[t-lookback+1:t+1]``.

    The stored value is the last row of that window after :func:`revin_window`.
    Rows before the first full window, and any window that still contains a
    non-finite value, stay missing. Later finite windows are not copied back.
    """
    if std_floor <= 0.0:
        raise ValueError("std_floor must be positive")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"revin_causal_rows expected shape [T, F], got {arr.shape}")
    rows, _cols = arr.shape
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if rows < lookback:
        return out
    view = sliding_window_view(arr, lookback, axis=0)
    finite = np.isfinite(view).all(axis=(1, 2))
    mean = view.mean(axis=2)
    std = view.std(axis=2)
    std = np.where(std < std_floor, 1.0, std)
    norm_last = (view[:, :, -1] - mean) / std
    out[lookback - 1 :][finite] = norm_last[finite]
    return out


def _base_features(
    ohlcv: pd.DataFrame,
    dates: pd.Series,
    windows: tuple[int, ...],
    return_lags: tuple[int, ...],
    eps: float,
    eii_windows: tuple[int, ...] = EII_WINDOWS,
) -> pd.DataFrame:
    index = ohlcv.index
    open_ = ohlcv["open"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    volume = ohlcv["volume"]
    columns: dict[str, pd.Series] = {}
    columns.update(_block_returns(open_, high, low, close, return_lags, eps))
    columns.update(_block_volatility(open_, high, low, close, windows))
    columns.update(_block_momentum(close, windows, eps))
    columns.update(_block_volume(high, low, close, volume, windows, eps))
    columns.update(_block_becker(open_, high, low, close, volume, windows, eii_windows, eps))
    columns.update(_block_fft(close, lookback=60))
    columns.update(_block_calendar(dates, index))
    names = base_feature_names(windows, return_lags, eii_windows)
    missing = [name for name in names if name not in columns]
    extra = [name for name in columns if name not in names]
    if missing or extra:
        raise RuntimeError(f"feature inventory mismatch missing={missing} extra={extra}")
    return pd.DataFrame({name: columns[name] for name in names}, index=index, dtype=np.float64)


def _block_returns(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    return_lags: tuple[int, ...],
    eps: float,
) -> dict[str, pd.Series]:
    log_close = pd.Series(np.log(close.to_numpy(dtype=np.float64)), index=close.index)
    columns = {f"log_ret_k{k}": log_close - log_close.shift(k) for k in return_lags}
    columns["intraday_ret"] = (close - open_) / open_
    columns["body_range"] = (close - open_).abs() / (high - low + eps)
    return columns


def _block_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    windows: tuple[int, ...],
) -> dict[str, pd.Series]:
    high_np = high.to_numpy(dtype=np.float64)
    low_np = low.to_numpy(dtype=np.float64)
    hl = np.log(high_np / low_np)
    co = np.log(close.to_numpy(dtype=np.float64) / open_.to_numpy(dtype=np.float64))
    parkinson_bar = pd.Series(hl**2 / (4.0 * np.log(2.0)), index=close.index)
    gk_bar = pd.Series(0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2, index=close.index)
    true_range = _true_range(high, low, close)
    columns: dict[str, pd.Series] = {}
    for window in windows:
        roll = dict(window=window, min_periods=window, center=False)
        columns[f"parkinson_n{window}"] = parkinson_bar.rolling(**roll).mean().clip(lower=0.0).pow(0.5)
        columns[f"gk_n{window}"] = gk_bar.rolling(**roll).mean().clip(lower=0.0).pow(0.5)
        columns[f"natr_n{window}"] = true_range.rolling(**roll).mean() / close * 100.0
    return columns


def _block_momentum(
    close: pd.Series,
    windows: tuple[int, ...],
    eps: float,
) -> dict[str, pd.Series]:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    columns: dict[str, pd.Series] = {}
    for window in windows:
        avg_up = _ema(up, window)
        avg_down = _ema(down, window)
        columns[f"rsi_n{window}"] = 100.0 - 100.0 / (1.0 + avg_up / (avg_down + eps))
    columns["macd_hist_norm"] = _macd_hist_norm(close)
    for window in windows:
        roll = dict(window=window, min_periods=window, center=False)
        mean = close.rolling(**roll).mean()
        sigma = close.rolling(**roll).std(ddof=0)
        upper = mean + 2.0 * sigma
        lower = mean - 2.0 * sigma
        columns[f"bb_pctb_n{window}"] = (close - lower) / (upper - lower + eps)
        columns[f"bb_width_n{window}"] = (upper - lower) / mean
    return columns


def _block_volume(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    windows: tuple[int, ...],
    eps: float,
) -> dict[str, pd.Series]:
    typical = (high + low + close) / 3.0
    typical_volume = typical * volume
    obv = _on_balance_volume(close, volume)
    columns: dict[str, pd.Series] = {}
    for window in windows:
        roll = dict(window=window, min_periods=window, center=False)
        numerator = typical_volume.rolling(**roll).sum()
        denominator = volume.rolling(**roll).sum()
        vwap = numerator / denominator.where(denominator > 0.0)
        columns[f"vwap_spread_n{window}"] = (close - vwap) / vwap
        volume_mean = volume.rolling(**roll).mean()
        volume_std = volume.rolling(**roll).std(ddof=0)
        columns[f"volume_z_n{window}"] = (volume - volume_mean) / (volume_std + eps)
        obv_mean = obv.rolling(**roll).mean()
        obv_std = obv.rolling(**roll).std(ddof=0)
        columns[f"obv_z_n{window}"] = (obv - obv_mean) / (obv_std + eps)
    return columns


def _block_calendar(dates: pd.Series, index: pd.Index) -> dict[str, pd.Series]:
    dow = dates.dt.dayofweek.to_numpy(dtype=np.float64)
    day = dates.dt.day.to_numpy(dtype=np.float64)
    days_in_month = dates.dt.days_in_month.to_numpy(dtype=np.float64)
    tau = (day - 1.0) / days_in_month
    day_angle = 2.0 * np.pi * dow / 5.0
    month_angle = 2.0 * np.pi * tau
    return {
        "day_sin": pd.Series(np.sin(day_angle), index=index, dtype=np.float64),
        "day_cos": pd.Series(np.cos(day_angle), index=index, dtype=np.float64),
        "month_sin": pd.Series(np.sin(month_angle), index=index, dtype=np.float64),
        "month_cos": pd.Series(np.cos(month_angle), index=index, dtype=np.float64),
    }


def _block_becker(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    windows: tuple[int, ...],
    eii_windows: tuple[int, ...],
    eps: float,
) -> dict[str, pd.Series]:
    high_np = high.to_numpy(dtype=np.float64)
    low_np = low.to_numpy(dtype=np.float64)
    close_np = close.to_numpy(dtype=np.float64)
    open_np = open_.to_numpy(dtype=np.float64)
    vol_np = volume.to_numpy(dtype=np.float64)

    # Garman-Klass volatility bar series (sigma_GK_N)
    hl = np.log(high_np / low_np)
    co = np.log(close_np / open_np)
    gk_bar = pd.Series(0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2, index=close.index)

    # Intraday candle body, range, and direction
    body = (close - open_).abs()
    candle_range = high - low
    body_to_range = body / (candle_range + eps)
    direction = pd.Series(np.sign(close_np - open_np), index=close.index, dtype=np.float64)

    # Log-returns for semivariance (CFI)
    log_close = pd.Series(np.log(close_np), index=close.index)
    log_ret = log_close.diff()
    neg_ret_sq = (log_ret.clip(upper=0.0)) ** 2
    pos_ret_sq = (log_ret.clip(lower=0.0)) ** 2

    # True range for ATR (Q)
    true_range = _true_range(high, low, close)
    overnight_raw = open_ - close.shift(1)

    # Typical price and volume for decayed VWAP (CGO)
    typical = (high + low + close) / 3.0
    typical_vol_np = (typical * volume).to_numpy(dtype=np.float64)

    columns: dict[str, pd.Series] = {}

    # 1. Endogenous Inelasticity Index (EII)
    for window in eii_windows:
        roll = dict(window=window, min_periods=window, center=False)
        sma_v = volume.rolling(**roll).mean()
        gk_vol = gk_bar.rolling(**roll).mean().clip(lower=0.0).pow(0.5)
        rel_vol_log = np.log1p(volume / (sma_v + eps))
        columns[f"eii_n{window}"] = direction * body_to_range * rel_vol_log / (gk_vol + eps)

    # 2. Unmet Demand / Queue Accumulator (Q)
    for window in windows:
        roll = dict(window=window, min_periods=window, center=False)
        roll_med_v = volume.rolling(**roll).median()
        pinning = ((2.0 * close - (high + low)) / (candle_range + eps)) * (volume / (roll_med_v + eps))
        pinning_ema = _ema(pinning, window, min_periods=1)
        atr = true_range.rolling(**roll).mean()
        overnight_norm = overnight_raw / (atr + eps)
        columns[f"queue_acc_n{window}"] = pinning_ema + overnight_norm

    # 3. Crash Fragility Index (CFI)
    for window in windows:
        roll = dict(window=window, min_periods=window, center=False)
        rsv_minus = neg_ret_sq.rolling(**roll).sum()
        rsv_plus = pos_ret_sq.rolling(**roll).sum()
        rsv_ratio = np.clip(rsv_minus / (rsv_plus + 1e-4), 0.0, 50.0)

        roll_min = low.rolling(**roll).min()
        roll_max = high.rolling(**roll).max()
        norm_pos = (close - roll_min) / (roll_max - roll_min + eps)
        exp_pos = np.exp(norm_pos)

        ema_c = _ema(close, window)
        above_ema = (close > ema_c).astype(np.float64)
        frac_above = above_ema.rolling(**roll).mean()
        cfi_val = rsv_ratio * exp_pos * frac_above
        columns[f"cfi_n{window}"] = np.clip(cfi_val, 0.0, 50.0)

    # 4. Market Resiliency Deficit (MRD)
    for window in windows:
        roll = dict(window=window, min_periods=window, center=False)
        sma_v = volume.rolling(**roll).mean()
        vol_ratio = volume / (sma_v + eps)
        instant_mrd = body_to_range * vol_ratio
        columns[f"mrd_n{window}"] = _ema(instant_mrd, window, min_periods=1)

    # 5. Capital Gains Overhang Proxy (CGO)
    for window in windows:
        decay_weights = np.array(
            [(1.0 - 1.0 / float(window)) ** (window - 1 - j) for j in range(window)],
            dtype=np.float64,
        )
        num = _rolling_weighted_sum(typical_vol_np, decay_weights)
        den = _rolling_weighted_sum(vol_np, decay_weights)
        vwap_decay = np.where(den > eps, num / (den + eps), close_np)
        cgo = (close_np - vwap_decay) / (close_np + eps)
        columns[f"cgo_n{window}"] = pd.Series(cgo, index=close.index, dtype=np.float64)

    return columns


def _block_fft(close: pd.Series, lookback: int = 60) -> dict[str, pd.Series]:
    """Causal rolling rFFT harmonic log-amplitudes on Close price.

    For each bar t, uses only bars t-lookback+1..t+1 (window of length lookback).
    Bars before the first full window (t < lookback - 1) stay NaN.
    Window is centered by subtracting its mean, rFFT is computed, and log-amplitudes
    of the first 10 non-zero harmonics (periods 60 down to 6 bars) are extracted:
    fft_harm_k = ln(1 + |spectrum[k]|).
    """
    if lookback < 12:
        raise ValueError(f"lookback must be at least 12 for 10 harmonics, got {lookback}")
    close_np = close.to_numpy(dtype=np.float64)
    n_rows = len(close_np)
    out = np.full((n_rows, 10), np.nan, dtype=np.float64)
    if n_rows >= lookback:
        view = sliding_window_view(close_np, lookback)
        finite = np.isfinite(view).all(axis=1)
        if finite.any():
            valid_windows = view[finite]
            centered = valid_windows - np.mean(valid_windows, axis=1, keepdims=True)
            spectrum = np.abs(np.fft.rfft(centered, axis=1))
            harmonics = np.log1p(spectrum[:, 1:11])
            out[lookback - 1 :][finite] = harmonics
    return {
        f"fft_harm_{k}": pd.Series(out[:, k - 1], index=close.index, dtype=np.float64)
        for k in range(1, 11)
    }


def _rolling_weighted_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Right-aligned causal weighted sum over a sliding window of length len(weights)."""
    n = len(weights)
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < n:
        return out
    conv = np.convolve(values, weights[::-1], mode="valid")
    out[n - 1 :] = conv
    return out


def _expand_lags(base: pd.DataFrame, feature_lags: tuple[int, ...]) -> pd.DataFrame:
    expanded: dict[str, pd.Series] = {}
    for name in base.columns:
        series = base[name]
        expanded[name] = series
        for lag in feature_lags:
            expanded[f"{name}_lag{lag}"] = series.shift(lag)
        expanded[f"{name}_delta"] = series - series.shift(1)
    return pd.DataFrame(expanded, index=base.index, dtype=np.float64)


def _ema(series: pd.Series, span: int, min_periods: int | None = None) -> pd.Series:
    """Recursive span EMA. Leading gaps do not pull later observations back."""
    periods = span if min_periods is None else int(min_periods)
    return series.ewm(span=span, adjust=False, min_periods=periods, ignore_na=False).mean()


def _macd_hist_norm(close: pd.Series) -> pd.Series:
    macd = _ema(close, MACD_FAST) - _ema(close, MACD_SLOW)
    signal = _ema(macd, MACD_SIGNAL)
    return (macd - signal) / close


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous = close.shift(1)
    high_low = high - low
    high_close = (high - previous).abs()
    low_close = (low - previous).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=True)


def _on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().to_numpy(dtype=np.float64))
    direction[0] = 0.0
    signed = volume.to_numpy(dtype=np.float64) * direction
    return pd.Series(np.cumsum(signed), index=close.index, dtype=np.float64)


def _prepare_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns = _column_map(frame)
    if "ticker" in columns:
        tickers = frame[columns["ticker"]].dropna()
        if tickers.nunique() > 1:
            raise ValueError("compute_technical_features expects a single ticker series")
    missing = [name for name in _OHLCV_COLUMNS if name not in columns]
    if missing:
        raise KeyError(f"missing columns {missing}")
    ohlcv = frame.loc[:, [columns[name] for name in _OHLCV_COLUMNS]].copy()
    ohlcv.columns = list(_OHLCV_COLUMNS)
    ohlcv = ohlcv.astype(np.float64)
    if not np.isfinite(ohlcv.to_numpy()).all():
        raise ValueError("OHLCV contains non-finite values")
    if (ohlcv.loc[:, list(_PRICE_COLUMNS)] <= 0.0).any().any():
        raise ValueError("prices must be positive")
    if (ohlcv["volume"] < 0.0).any():
        raise ValueError("volume must be non-negative")
    dates = _extract_dates(frame, columns)
    if len(dates) != len(ohlcv):
        raise ValueError("date length does not match OHLCV")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("dates must be parsed, unique, and strictly increasing")
    return ohlcv, dates


def _extract_dates(frame: pd.DataFrame, columns: dict[str, str]) -> pd.Series:
    if "date" in columns:
        raw = frame[columns["date"]]
    elif isinstance(frame.index, pd.DatetimeIndex):
        raw = pd.Series(frame.index, index=frame.index)
    else:
        raise ValueError("calendar features need a date column or a DatetimeIndex")
    dates = pd.to_datetime(raw, errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    return pd.Series(dates.to_numpy(), index=frame.index)


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    lowered = [str(column).strip().lower() for column in frame.columns]
    if len(lowered) != len(set(lowered)):
        raise ValueError("duplicate columns after case normalization")
    return dict(zip(lowered, frame.columns, strict=True))


def _windows_and_returns(
    windows: Sequence[int],
    return_lags: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        _as_unique_ints(windows, "windows", minimum=2),
        _as_unique_ints(return_lags, "return_lags", minimum=1),
    )


def _as_unique_ints(
    values: Sequence[int],
    name: str,
    *,
    minimum: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} values must be integers")
        parsed.append(int(value))
    out = tuple(parsed)
    if not out and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if any(value < minimum for value in out):
        hint = "; a negative shift looks ahead" if any(value < 0 for value in out) else ""
        raise ValueError(f"{name} values must be >= {minimum}{hint}")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} values must be unique")
    return out
