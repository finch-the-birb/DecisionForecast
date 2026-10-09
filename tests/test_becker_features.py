"""Unit tests for Becker (1991) and microstructure features (Sprint 12).

Tests:
1. Causal invariance: future bar mutations do not alter past feature rows.
2. Quiet candle robustness: flat bars (high == low) produce finite values (no NaN/Inf).
3. Selective norm registration: mrd/cgo are bounded in [-1, 1], eii/queue_acc/cfi are stationary.
4. Feature selection integration: select_fold_features ranks expanded pool and pins close at channel 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.selective_norm import apply_static_and_robust, feature_role, feature_roles
from src.features.selection import (
    CORR_THRESHOLD,
    HUBER_DELTA,
    LEVEL_COLUMN,
    TOP_K,
    decorrelated_top_k,
    select_fold_features,
    spearman_corr,
)
from src.features.technical import (
    EII_WINDOWS,
    WINDOWS,
    base_feature_names,
    compute_technical_features,
    technical_feature_names,
)


def _synthetic_prices(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    log_ret = rng.normal(0.0003, 0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = close * np.exp(rng.normal(0.0, 0.005, size=n))
    spread = np.abs(rng.normal(0.005, 0.003, size=n)) + 1e-4
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=12.0, sigma=0.8, size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_becker_inventory_and_naming() -> None:
    base = base_feature_names()
    for w in EII_WINDOWS:
        assert f"eii_n{w}" in base
    for w in WINDOWS:
        assert f"queue_acc_n{w}" in base
        assert f"cfi_n{w}" in base
        assert f"mrd_n{w}" in base
        assert f"cgo_n{w}" in base

    expanded = technical_feature_names()
    for w in EII_WINDOWS:
        assert f"eii_n{w}" in expanded
        assert f"eii_n{w}_lag1" in expanded
        assert f"eii_n{w}_lag2" in expanded
        assert f"eii_n{w}_lag3" in expanded
        assert f"eii_n{w}_delta" in expanded

    for w in WINDOWS:
        for prefix in ("queue_acc", "cfi", "mrd", "cgo"):
            assert f"{prefix}_n{w}" in expanded
            assert f"{prefix}_n{w}_lag1" in expanded
            assert f"{prefix}_n{w}_lag2" in expanded
            assert f"{prefix}_n{w}_lag3" in expanded
            assert f"{prefix}_n{w}_delta" in expanded

    assert len(expanded) == len(set(expanded))


def test_causality_future_bars_do_not_alter_past() -> None:
    """Modifying future bars (t >= 60) must have zero effect on past rows (t < 60)."""
    df = _synthetic_prices(120, seed=10)
    original = compute_technical_features(df)

    mutated = df.copy()
    mutated.loc[60:, "close"] = mutated.loc[60:, "close"] * 2.5
    mutated.loc[60:, "open"] = mutated.loc[60:, "open"] * 2.2
    mutated.loc[60:, "high"] = mutated.loc[60:, "high"] * 3.0
    mutated.loc[60:, "low"] = mutated.loc[60:, "low"] * 0.5
    mutated.loc[60:, "volume"] = mutated.loc[60:, "volume"] * 10.0

    after_mutation = compute_technical_features(mutated)

    becker_prefixes = ("eii_", "queue_acc_", "cfi_", "mrd_", "cgo_")
    becker_cols = [c for c in original.columns if c.startswith(becker_prefixes)]
    assert len(becker_cols) > 0

    past_orig = original.loc[:59, becker_cols]
    past_mut = after_mutation.loc[:59, becker_cols]
    pd.testing.assert_frame_equal(past_orig, past_mut, rtol=1e-12, atol=1e-12)

    # Full frame comparison on past bars
    pd.testing.assert_frame_equal(original.iloc[:60], after_mutation.iloc[:60], rtol=1e-12, atol=1e-12)


def test_prefix_matches_full_series() -> None:
    """Features at row t computed on df.iloc[:t+1] equal the same row in full calculation."""
    df = _synthetic_prices(100, seed=7)
    full = compute_technical_features(df)

    becker_prefixes = ("eii_", "queue_acc_", "cfi_", "mrd_", "cgo_")
    becker_cols = [c for c in full.columns if c.startswith(becker_prefixes)]

    for end in (45, 65, 85):
        sliced = compute_technical_features(df.iloc[: end + 1])
        pd.testing.assert_frame_equal(
            full.loc[:end, becker_cols],
            sliced.loc[:end, becker_cols],
            rtol=1e-12,
            atol=1e-12,
        )


def test_quiet_candles_no_nan_or_inf() -> None:
    """Stretches of flat candles (high == low == open == close) must remain finite."""
    n = 100
    dates = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(99)

    open_ = np.full(n, 50.0)
    high = np.full(n, 50.0)
    low = np.full(n, 50.0)
    close = np.full(n, 50.0)
    volume = np.full(n, 1000.0)

    # Halfway through, let volume drop to zero on flat candles
    volume[50:] = 0.0

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    feats = compute_technical_features(df)

    becker_prefixes = ("eii_", "queue_acc_", "cfi_", "mrd_", "cgo_")
    becker_cols = [c for c in feats.columns if c.startswith(becker_prefixes)]

    # Warm-up is max(WINDOWS) + max(FEATURE_LAGS) = 63 bars. Beyond warm-up, all values must be finite.
    warmup = max(WINDOWS) + 3
    steady_state = feats.iloc[warmup:][becker_cols]

    assert not np.isinf(steady_state.to_numpy()).any(), "Inf detected in steady state"
    assert not np.isnan(steady_state.to_numpy()).any(), "NaN detected in steady state"


def test_selective_norm_typing() -> None:
    """Check role typing and static/robust normalization for Becker features."""
    for w in EII_WINDOWS:
        assert feature_role(f"eii_n{w}") == "stationary"
        assert feature_role(f"eii_n{w}_lag1") == "stationary"
        assert feature_role(f"eii_n{w}_delta") == "stationary"

    for w in WINDOWS:
        assert feature_role(f"queue_acc_n{w}") == "stationary"
        assert feature_role(f"cfi_n{w}") == "stationary"
        assert feature_role(f"mrd_n{w}") == "bounded"
        assert feature_role(f"cgo_n{w}") == "bounded"
        assert feature_role(f"mrd_n{w}_lag2") == "bounded"
        assert feature_role(f"cgo_n{w}_delta") == "bounded"

    # Test static bounded clipping in [-1, 1]
    cols = ["mrd_n5", "cgo_n10", "eii_n5", "close"]
    vals = np.array(
        [
            [2.5, -3.0, 10.0, 50.0],
            [0.5, 0.2, 5.0, 51.0],
            [-0.5, 0.8, -2.0, 52.0],
            [1.0, 1.0, 0.0, 53.0],
        ]
    )
    mask = np.ones(len(vals), dtype=bool)
    scaled = apply_static_and_robust(vals, cols, mask)

    # mrd and cgo are bounded
    assert (scaled[:, 0] <= 1.0).all() and (scaled[:, 0] >= -1.0).all()
    assert (scaled[:, 1] <= 1.0).all() and (scaled[:, 1] >= -1.0).all()
    # close is level (unmodified by static/robust)
    np.testing.assert_allclose(scaled[:, 3], vals[:, 3])


def test_selection_integration_with_becker_features() -> None:
    """Ensure select_fold_features ranks the expanded pool and pins close to channel 0."""
    df = _synthetic_prices(150, seed=123)
    feats = compute_technical_features(df)
    feats.insert(0, "close", df["close"])

    target = np.log(df["close"].shift(-7) / df["close"]).to_numpy()
    train_mask = np.arange(len(df)) < 120

    sig = select_fold_features(
        feats,
        target,
        train_mask,
        text_columns=(),
        top_k=TOP_K,
        corr_threshold=CORR_THRESHOLD,
        huber_delta=HUBER_DELTA,
        n_estimators=30,
        num_leaves=8,
        min_child_samples=5,
        seed=0,
    )

    assert sig.ts_columns[0] == LEVEL_COLUMN
    assert len(sig.ts_columns) == TOP_K + 1
    # Check that Becker features were part of the ranked pool
    becker_prefixes = ("eii_", "queue_acc_", "cfi_", "mrd_", "cgo_")
    ranked_becker = [c for c in sig.importances.index if c.startswith(becker_prefixes)]
    assert len(ranked_becker) > 0, "Becker features must be ranked in TreeSHAP pool"
