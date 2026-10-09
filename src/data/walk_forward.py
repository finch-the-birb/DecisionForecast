"""Expanding walk-forward folds with purge and embargo.

Phase 4, Sprint 12. Three folds:

* train 2015–2019, val 2020, test 2021
* train 2015–2020, val 2021, test 2022
* train 2015–2021, val 2022, test 2023

A window is indexed by ``end_idx``, the first label bar, matching
``FNSPIDForecastDataset`` (features are ``[end_idx - lookback, end_idx)``,
labels are ``[end_idx, end_idx + horizon)``).

Before each later split, the previous split drops ``(horizon - 1) + embargo``
windows. ``horizon - 1`` is the purge: those windows' labels reach into the
next split. ``embargo`` (14 trading bars) is the extra buffer after that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

EMBARGO_BARS = 14


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class FoldWindows:
    fold: WalkForwardFold
    train_end_idx: np.ndarray
    val_end_idx: np.ndarray
    test_end_idx: np.ndarray


def _bounds(*stamps: str) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(stamp).normalize() for stamp in stamps)


DEFAULT_FOLDS: tuple[WalkForwardFold, ...] = (
    WalkForwardFold(
        1,
        *_bounds("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ),
    WalkForwardFold(
        2,
        *_bounds("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ),
    WalkForwardFold(
        3,
        *_bounds("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ),
)


def purge_bars(horizon: int) -> int:
    """How many boundary windows have labels that spill into the next split."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return horizon - 1


def walk_forward_splits(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    *,
    horizon: int,
    embargo: int = EMBARGO_BARS,
    lookback: int = 0,
    folds: Sequence[WalkForwardFold] = DEFAULT_FOLDS,
) -> list[FoldWindows]:
    """Return train/val/test ``end_idx`` arrays for each expanding fold."""
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    calendar = _calendar(dates)
    lookback = int(lookback)
    if lookback < 0:
        raise ValueError("lookback must be >= 0")
    return [
        _assign_fold(calendar, fold, horizon=horizon, embargo=embargo, lookback=lookback)
        for fold in folds
    ]


def _assign_fold(
    dates: pd.DatetimeIndex,
    fold: WalkForwardFold,
    *,
    horizon: int,
    embargo: int,
    lookback: int,
) -> FoldWindows:
    train_origin = _first_on_or_after(dates, fold.train_start)
    val_origin = _first_on_or_after(dates, fold.val_start)
    test_origin = _first_on_or_after(dates, fold.test_start)
    val_last = _last_on_or_before(dates, fold.val_end)
    test_last = _last_on_or_before(dates, fold.test_end)
    if not (train_origin < val_origin < test_origin <= test_last):
        raise ValueError(f"fold {fold.fold_id} boundaries are not increasing on this calendar")
    train_idx = _window_indices(
        first=max(lookback, train_origin),
        last=_last_end_before(val_origin, horizon, embargo),
        horizon=horizon,
        n_dates=len(dates),
    )
    val_idx = _window_indices(
        first=max(lookback, val_origin),
        last=min(
            _last_end_before(test_origin, horizon, embargo),
            val_last - horizon + 1,
        ),
        horizon=horizon,
        n_dates=len(dates),
    )
    test_idx = _window_indices(
        first=max(lookback, test_origin),
        last=test_last - horizon + 1,
        horizon=horizon,
        n_dates=len(dates),
    )
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"fold {fold.fold_id} has an empty split "
            f"(train={train_idx.size}, val={val_idx.size}, test={test_idx.size})"
        )
    return FoldWindows(fold=fold, train_end_idx=train_idx, val_end_idx=val_idx, test_end_idx=test_idx)


def _last_end_before(boundary: int, horizon: int, embargo: int) -> int:
    """Inclusive last end_idx whose labels stop ``embargo`` bars before ``boundary``."""
    return boundary - horizon - embargo


def _window_indices(*, first: int, last: int, horizon: int, n_dates: int) -> np.ndarray:
    last = min(last, n_dates - horizon)
    if last < first:
        return np.zeros(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


def _calendar(dates: pd.Series | pd.DatetimeIndex | np.ndarray) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if index.tz is not None:
        index = index.tz_convert(None)
    index = index.normalize()
    if index.hasnans or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("dates must be unique and strictly increasing")
    return index


def _first_on_or_after(dates: pd.DatetimeIndex, stamp: pd.Timestamp) -> int:
    position = int(dates.searchsorted(pd.Timestamp(stamp).normalize(), side="left"))
    if position >= len(dates):
        raise ValueError(f"{pd.Timestamp(stamp).date()} is after the calendar")
    return position


def _last_on_or_before(dates: pd.DatetimeIndex, stamp: pd.Timestamp) -> int:
    position = int(dates.searchsorted(pd.Timestamp(stamp).normalize(), side="right")) - 1
    if position < 0:
        raise ValueError(f"{pd.Timestamp(stamp).date()} is before the calendar")
    return position
