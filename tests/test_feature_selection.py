"""Train-only TreeSHAP selection and the Spearman collinearity filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.text_compact import COMPACT_COLUMNS, COMPACT_DIM
from src.features.selection import (
    CORR_THRESHOLD,
    HUBER_DELTA,
    TOP_K,
    _make_regressor,
    apply_fold_signature,
    decorrelated_top_k,
    select_fold_features,
    spearman_corr,
)

_FIT = dict(n_estimators=40, num_leaves=8, min_child_samples=5, learning_rate=0.1)


def test_huber_delta_matches_the_locked_forecasting_loss() -> None:
    model = _make_regressor(
        huber_delta=HUBER_DELTA,
        n_estimators=2,
        learning_rate=0.1,
        num_leaves=4,
        min_child_samples=2,
        seed=0,
    )
    assert model.get_params()["objective"] == "huber"
    assert model.get_params()["alpha"] == 0.5
    assert HUBER_DELTA == 0.5
    assert TOP_K == 25
    assert CORR_THRESHOLD == 0.85


def test_spearman_is_rank_correlation_and_constants_are_zero() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "cube": [1.0, 8.0, 27.0, 64.0],
            "neg": [-1.0, -2.0, -3.0, -4.0],
            "flat": [5.0, 5.0, 5.0, 5.0],
            "ties": [1.0, 1.0, 2.0, 3.0],
        }
    )
    corr = spearman_corr(frame)
    np.testing.assert_allclose(corr.loc["x", "cube"], 1.0)
    np.testing.assert_allclose(corr.loc["x", "neg"], -1.0)
    np.testing.assert_allclose(corr.loc["flat", "x"], 0.0)
    np.testing.assert_allclose(corr.loc["flat", "flat"], 1.0)
    tied = spearman_corr(frame.loc[:, ["ties"]])
    assert tied.shape == (1, 1)


def test_greedy_filter_skips_the_weaker_correlated_feature() -> None:
    importances = pd.Series({"a": 5.0, "b": 4.0, "c": 1.0})
    corr = pd.DataFrame(
        [
            [1.0, 0.91, 0.1],
            [0.91, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    assert decorrelated_top_k(importances, corr, top_k=2, threshold=0.85) == ["a", "c"]
    at_threshold = corr.copy()
    at_threshold.loc["a", "b"] = 0.85
    at_threshold.loc["b", "a"] = 0.85
    assert decorrelated_top_k(importances, at_threshold, top_k=2, threshold=0.85) == ["a", "b"]


def test_equal_importance_keeps_the_earlier_column() -> None:
    importances = pd.Series({"a": 1.0, "b": 1.0})
    corr = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"])
    assert decorrelated_top_k(importances, corr, top_k=1, threshold=0.85) == ["a"]


def test_selector_ignores_rows_outside_the_train_mask() -> None:
    rng = np.random.default_rng(0)
    n_train, n_test = 120, 80
    signal = rng.normal(size=n_train + n_test)
    decoy = rng.normal(size=n_train + n_test)
    noise = rng.normal(size=n_train + n_test)
    target = np.concatenate(
        [
            signal[:n_train] + 0.01 * noise[:n_train],
            decoy[n_train:] * 5.0,
        ]
    )
    frame = pd.DataFrame({"signal": signal, "decoy": decoy, "noise": noise})
    mask = np.arange(len(frame)) < n_train
    original = select_fold_features(
        frame, target, mask, text_columns=(), top_k=1, seed=0, **_FIT
    )
    scrambled = frame.copy()
    scrambled.loc[~mask, "decoy"] = np.linspace(0.0, 100.0, n_test)
    scrambled_target = target.copy()
    scrambled_target[~mask] = scrambled.loc[~mask, "decoy"].to_numpy()
    again = select_fold_features(
        scrambled, scrambled_target, mask, text_columns=(), top_k=1, seed=0, **_FIT
    )
    assert original.ts_columns == ("close", "signal")
    assert again.ts_columns == original.ts_columns
    assert "close" not in original.importances.index
    pd.testing.assert_series_equal(original.importances, again.importances)
    assert original.n_train_rows == n_train


def test_correlation_filter_uses_the_train_split_only() -> None:
    rng = np.random.default_rng(1)
    n_train, n_test = 100, 500
    n = n_train + n_test
    train_a = rng.normal(size=n_train)
    train_b = rng.normal(size=n_train)
    test_a = rng.normal(size=n_test)
    frame = pd.DataFrame(
        {
            "a": np.concatenate([train_a, test_a]),
            "b": np.concatenate([train_b, test_a]),
            "noise": rng.normal(size=n),
        }
    )
    target = np.concatenate([train_a + train_b, np.zeros(n_test)])
    mask = np.arange(n) < n_train
    full = spearman_corr(frame.loc[:, ["a", "b"]])
    train = spearman_corr(frame.loc[mask, ["a", "b"]])
    assert abs(float(full.loc["a", "b"])) > 0.85
    assert abs(float(train.loc["a", "b"])) < 0.5
    signature = select_fold_features(
        frame, target, mask, text_columns=(), top_k=2, seed=0, **_FIT
    )
    assert signature.ts_columns[0] == "close"
    assert set(signature.ts_columns) == {"close", "a", "b"}
    assert "close" not in signature.importances.index


def test_text_columns_are_excluded_from_the_booster() -> None:
    rng = np.random.default_rng(2)
    n = 150
    signal = rng.normal(size=n)
    target = signal + 0.01 * rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=n),
            "sent_pos": target,
        }
    )
    mask = np.ones(n, dtype=bool)
    signature = select_fold_features(
        frame,
        target,
        mask,
        text_columns=("sent_pos",),
        top_k=1,
        seed=0,
        **_FIT,
    )
    assert signature.ts_columns == ("close", "signal")
    assert "close" not in signature.importances.index
    assert "sent_pos" not in signature.importances.index
    assert signature.text_columns == ("sent_pos",)


def test_non_finite_train_rows_are_dropped_and_test_nans_are_ignored() -> None:
    rng = np.random.default_rng(3)
    n_train, n_test = 80, 40
    n = n_train + n_test
    signal = rng.normal(size=n)
    frame = pd.DataFrame({"signal": signal, "other": rng.normal(size=n)})
    target = signal + 0.01 * rng.normal(size=n)
    mask = np.arange(n) < n_train
    dropped = mask.copy()
    dropped[5] = False
    by_mask = select_fold_features(
        frame, target, dropped, text_columns=(), top_k=1, seed=0, **_FIT
    )
    with_nan = frame.copy()
    with_nan.loc[5, "signal"] = np.nan
    by_nan = select_fold_features(
        with_nan, target, mask, text_columns=(), top_k=1, seed=0, **_FIT
    )
    assert by_nan.n_train_rows == n_train - 1
    pd.testing.assert_series_equal(by_mask.importances, by_nan.importances)

    test_nan = frame.copy()
    test_nan.loc[n_train + 3, "other"] = np.nan
    untouched = select_fold_features(
        frame, target, mask, text_columns=(), top_k=1, seed=0, **_FIT
    )
    ignored = select_fold_features(
        test_nan, target, mask, text_columns=(), top_k=1, seed=0, **_FIT
    )
    pd.testing.assert_series_equal(untouched.importances, ignored.importances)
    assert ignored.n_train_rows == n_train


def test_default_signature_pins_close_then_25_indicators() -> None:
    rng = np.random.default_rng(4)
    n = 500
    values = rng.normal(size=(n, TOP_K))
    target = values.sum(axis=1)
    columns = [f"f{i:02d}" for i in range(TOP_K)]
    frame = pd.DataFrame(values, columns=columns)
    off_diagonal = spearman_corr(frame).to_numpy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert np.abs(off_diagonal).max() < CORR_THRESHOLD
    frame["close"] = 100.0 + rng.normal(size=n)
    for name in COMPACT_COLUMNS:
        frame[name] = rng.normal(size=n)
    mask = np.ones(n, dtype=bool)
    signature = select_fold_features(frame, target, mask, seed=0, n_estimators=20, num_leaves=8)
    assert signature.ts_columns[0] == "close"
    assert set(signature.ts_columns[1:]) == set(columns)
    assert len(signature.ts_columns) == TOP_K + 1
    assert "close" not in signature.importances.index
    assert signature.text_columns == COMPACT_COLUMNS
    assert signature.columns[0] == "close"
    assert len(signature.columns) == TOP_K + 1 + COMPACT_DIM
    ordered = apply_fold_signature(frame, signature)
    assert list(ordered.columns) == list(signature.columns)
    assert ordered.index.equals(frame.index)


def test_apply_requires_the_frozen_columns() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    target = np.array([1.0, 2.0])
    mask = np.array([True, True])
    signature = select_fold_features(
        frame, target, mask, text_columns=("sent_pos",), top_k=1, seed=0, **_FIT
    )
    with pytest.raises(KeyError, match="sent_pos"):
        apply_fold_signature(frame, signature)
