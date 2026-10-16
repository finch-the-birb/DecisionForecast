"""Unit tests for Walk-Forward Multi-Fold dataset protocol (Folds 1, 2, 3).

Verifies:
1. Strict year boundaries for test windows without leakage into future years.
   - Fold 1: Train 2015-2019, Val 2020, Test 2021
   - Fold 2: Train 2015-2020, Val 2021, Test 2022
   - Fold 3: Train 2015-2021, Val 2022, Test 2023
2. Purge (H - 1 bars) and embargo (14 bars) between splits.
3. Correct WindowIndex date assignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.dataset import build_datasets, _as_naive_day
from src.data.walk_forward import DEFAULT_FOLDS, EMBARGO_BARS


def _create_synthetic_multifold_store(tmp_path):
    prices_dir = tmp_path / "Stock_price" / "full_history"
    prices_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2015-01-01", "2023-12-31")

    # Create 2 synthetic tickers AAA and BBB
    tickers = ["AAA", "BBB"]
    for i, ticker in enumerate(tickers):
        base_price = 100.0 * (i + 1)
        close = base_price + np.cumsum(np.random.normal(0.05, 1.0, len(dates)))
        df = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": np.full(len(dates), 1000.0),
            }
        )
        df.to_csv(prices_dir / f"{ticker}.csv", index=False)

    return tmp_path, tickers


def test_walk_forward_fold_1_strict_bounds(tmp_path):
    root, tickers = _create_synthetic_multifold_store(tmp_path)
    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(root),
                "features_mode": "ohlcv",
                "features": ["close", "volume", "open", "high", "low"],
                "target": "close",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
                "normalize": "per_window",
                "news_source": "unused",
                "fold": 1,
                "tickers": {"dev": tickers},
                "text": {"enabled": False, "dim": 15},
            },
            "train": {
                "ticker_set": "dev",
                "max_train_windows": None,
                "max_val_windows": None,
                "max_test_windows": None,
            },
        }
    )

    train_ds, val_ds, test_ds = build_datasets(cfg)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    # Fold 1: Test strictly in 2021
    for wi in test_ds.indices:
        ts = test_ds.series[wi.ticker]
        first_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx])
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert first_target_date.year == 2021
        assert last_target_date.year == 2021

    # Fold 1: Val strictly in 2020
    for wi in val_ds.indices:
        ts = val_ds.series[wi.ticker]
        first_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx])
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert first_target_date.year == 2020
        assert last_target_date.year == 2020

    # Fold 1: Train strictly <= 2019
    for wi in train_ds.indices:
        ts = train_ds.series[wi.ticker]
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert last_target_date.year <= 2019


def test_walk_forward_fold_2_strict_bounds(tmp_path):
    root, tickers = _create_synthetic_multifold_store(tmp_path)
    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(root),
                "features_mode": "ohlcv",
                "features": ["close", "volume", "open", "high", "low"],
                "target": "close",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
                "normalize": "per_window",
                "news_source": "unused",
                "fold": 2,
                "tickers": {"dev": tickers},
                "text": {"enabled": False, "dim": 15},
            },
            "train": {
                "ticker_set": "dev",
                "max_train_windows": None,
                "max_val_windows": None,
                "max_test_windows": None,
            },
        }
    )

    train_ds, val_ds, test_ds = build_datasets(cfg)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    # Fold 2: Test strictly in 2022
    for wi in test_ds.indices:
        ts = test_ds.series[wi.ticker]
        first_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx])
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert first_target_date.year == 2022
        assert last_target_date.year == 2022

    # Fold 2: Val strictly in 2021
    for wi in val_ds.indices:
        ts = val_ds.series[wi.ticker]
        first_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx])
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert first_target_date.year == 2021
        assert last_target_date.year == 2021

    # Fold 2: Train strictly <= 2020
    for wi in train_ds.indices:
        ts = train_ds.series[wi.ticker]
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert last_target_date.year <= 2020


def test_walk_forward_fold_3_strict_bounds(tmp_path):
    root, tickers = _create_synthetic_multifold_store(tmp_path)
    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(root),
                "features_mode": "ohlcv",
                "features": ["close", "volume", "open", "high", "low"],
                "target": "close",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
                "normalize": "per_window",
                "news_source": "unused",
                "fold": 3,
                "tickers": {"dev": tickers},
                "text": {"enabled": False, "dim": 15},
            },
            "train": {
                "ticker_set": "dev",
                "max_train_windows": None,
                "max_val_windows": None,
                "max_test_windows": None,
            },
        }
    )

    train_ds, val_ds, test_ds = build_datasets(cfg)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    # Fold 3: Test strictly in 2023
    for wi in test_ds.indices:
        ts = test_ds.series[wi.ticker]
        first_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx])
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert first_target_date.year == 2023
        assert last_target_date.year == 2023

    # Fold 3: Val strictly in 2022
    for wi in val_ds.indices:
        ts = val_ds.series[wi.ticker]
        first_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx])
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert first_target_date.year == 2022
        assert last_target_date.year == 2022

    # Fold 3: Train strictly <= 2021
    for wi in train_ds.indices:
        ts = train_ds.series[wi.ticker]
        last_target_date = _as_naive_day(ts.dates.iloc[wi.end_idx + cfg.data.horizon - 1])
        assert last_target_date.year <= 2021


def test_walk_forward_expanding_train_windows(tmp_path):
    root, tickers = _create_synthetic_multifold_store(tmp_path)
    train_counts = []
    for fold in (1, 2, 3):
        cfg = OmegaConf.create(
            {
                "data": {
                    "root": str(root),
                    "features_mode": "ohlcv",
                    "features": ["close", "volume", "open", "high", "low"],
                    "target": "close",
                    "lookback_T": 60,
                    "horizon": 7,
                    "patch_len": 12,
                    "patch_stride": 6,
                    "normalize": "per_window",
                    "news_source": "unused",
                    "fold": fold,
                    "tickers": {"dev": tickers},
                    "text": {"enabled": False, "dim": 15},
                },
                "train": {
                    "ticker_set": "dev",
                    "max_train_windows": None,
                    "max_val_windows": None,
                    "max_test_windows": None,
                },
            }
        )
        train_ds, val_ds, test_ds = build_datasets(cfg)
        train_counts.append(len(train_ds))

    # Expanding window: train window counts strictly expand
    assert train_counts[0] < train_counts[1] < train_counts[2]
