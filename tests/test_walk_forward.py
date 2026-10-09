"""Expanding walk-forward: purge (H-1) plus a 14-bar embargo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.walk_forward import (
    EMBARGO_BARS,
    DEFAULT_FOLDS,
    WalkForwardFold,
    purge_bars,
    walk_forward_splits,
)


def _labels(end_indices: np.ndarray, horizon: int) -> set[int]:
    covered: set[int] = set()
    for end_idx in end_indices:
        covered.update(range(int(end_idx), int(end_idx) + horizon))
    return covered


def test_gap_before_validation_is_purge_plus_embargo() -> None:
    horizon = 7
    embargo = 14
    dates = pd.bdate_range("2014-01-02", "2024-03-29")
    folds = walk_forward_splits(dates, horizon=horizon, embargo=embargo, lookback=60)
    assert [item.fold.fold_id for item in folds] == [1, 2, 3]
    assert purge_bars(horizon) + embargo == 20
    assert EMBARGO_BARS == 14
    calendar = pd.DatetimeIndex(dates).normalize()
    for assigned in folds:
        val_origin = int(calendar.searchsorted(assigned.fold.val_start))
        test_origin = int(calendar.searchsorted(assigned.fold.test_start))
        last_train = int(assigned.train_end_idx[-1])
        assert last_train == val_origin - horizon - embargo
        assert last_train + 1 not in set(assigned.train_end_idx)
        dropped = (val_origin - 1) - last_train
        assert dropped == purge_bars(horizon) + embargo
        assert int(assigned.val_end_idx[0]) == val_origin
        assert int(assigned.val_end_idx[-1]) == test_origin - horizon - embargo
        assert int(assigned.test_end_idx[0]) == test_origin
        train_labels = _labels(assigned.train_end_idx, horizon)
        val_labels = _labels(assigned.val_end_idx, horizon)
        test_labels = _labels(assigned.test_end_idx, horizon)
        assert train_labels.isdisjoint(val_labels)
        assert train_labels.isdisjoint(test_labels)
        assert val_labels.isdisjoint(test_labels)
        assert max(train_labels) <= val_origin - embargo - 1


def test_folds_expand_and_keep_their_test_years() -> None:
    dates = pd.bdate_range("2014-01-02", "2024-03-29")
    folds = walk_forward_splits(dates, horizon=7, embargo=14, lookback=60)
    calendar = pd.DatetimeIndex(dates).normalize()
    june_2020 = int(calendar.searchsorted(pd.Timestamp("2020-06-15")))
    assert june_2020 in set(folds[0].val_end_idx)
    assert june_2020 in set(folds[1].train_end_idx)
    assert june_2020 not in set(folds[0].train_end_idx)
    assert calendar[folds[0].test_end_idx].min() >= pd.Timestamp("2021-01-01")
    assert calendar[folds[0].test_end_idx].max().year == 2021
    assert calendar[folds[1].test_end_idx].min() >= pd.Timestamp("2022-01-01")
    assert calendar[folds[1].test_end_idx].max().year == 2022
    assert calendar[folds[2].test_end_idx].min() >= pd.Timestamp("2023-01-01")
    assert calendar[folds[2].test_end_idx].max().year == 2023
    assert len(DEFAULT_FOLDS) == 3


def test_custom_fold_matches_the_boundary_formula() -> None:
    dates = pd.bdate_range("2020-01-01", periods=120)
    horizon, embargo, lookback = 3, 2, 5
    fold = WalkForwardFold(
        9,
        dates[0],
        dates[39],
        dates[40],
        dates[79],
        dates[80],
        dates[119],
    )
    assigned = walk_forward_splits(
        dates,
        horizon=horizon,
        embargo=embargo,
        lookback=lookback,
        folds=(fold,),
    )[0]
    assert assigned.train_end_idx[0] == lookback
    assert assigned.train_end_idx[-1] == 40 - horizon - embargo
    assert assigned.val_end_idx[0] == 40
    assert assigned.val_end_idx[-1] == 80 - horizon - embargo
    assert assigned.test_end_idx[0] == 80
    assert assigned.test_end_idx[-1] == 119 - horizon + 1
    assert _labels(assigned.train_end_idx, horizon).isdisjoint(_labels(assigned.val_end_idx, horizon))
