"""Selected 40D path: fold-1 freeze, dataset batch, and a finite backward pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.collate import make_forecast_loader
from src.data.dataset import build_datasets
from src.data.selected_pipeline import (
    article_bound_index,
    compact_daily_frame,
    fit_signature,
    require_fold,
    selection_sample,
    signature_cache_path,
    technical_cache_path,
    technical_frame,
    text_compact_cache_path,
    ticker_train_rows,
    train_article_mask,
    train_end_indices,
    write_parquet,
    write_signature,
)
from src.data.text_compact import fit_compact_state, probabilities_from_logits
from src.data.walk_forward import walk_forward_splits
from src.features.selection import HUBER_DELTA, select_fold_features
from src.training.train import build_model

_FUSION = {
    "kind": "selected_40d",
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "global_to_patch": True,
}


def _prices(seed: int, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2013-01-02", periods=n)
    close = 40.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n)))
    open_ = close * np.exp(rng.normal(0.0, 0.004, n))
    spread = np.abs(rng.normal(0.004, 0.002, n)) + 1e-4
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    low = np.maximum(low, 1e-3)
    volume = rng.lognormal(15.0, 0.7, n)
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


def _articles(dates: pd.DatetimeIndex, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_days = dates[(dates >= "2016-01-01") & (dates <= "2018-12-31")]
    test_days = dates[(dates >= "2022-03-01") & (dates <= "2022-06-30")]
    picked = np.concatenate(
        [
            rng.choice(train_days.to_numpy(), size=25, replace=False),
            rng.choice(test_days.to_numpy(), size=8, replace=False),
        ]
    )
    article_dates = (pd.DatetimeIndex(picked) + pd.Timedelta(hours=12)).to_numpy()
    embeddings = rng.normal(size=(len(picked), 32))
    probs = probabilities_from_logits(rng.normal(size=(len(picked), 3)))
    return article_dates, embeddings, probs


def test_next_session_binding_matches_tuesday_noon() -> None:
    days = pd.bdate_range("2021-01-04", "2021-01-08")
    tuesday = article_bound_index(["2021-01-05T12:00:00Z"], days)
    sunday = article_bound_index(["2021-01-03T12:00:00Z"], days)
    assert days[int(tuesday[0])] == pd.Timestamp("2021-01-06")
    assert days[int(sunday[0])] == pd.Timestamp("2021-01-04")


def test_train_indices_match_walk_forward_and_stop_before_2020() -> None:
    dates = pd.bdate_range("2014-01-02", "2024-03-29")
    fold = require_fold(1)
    got = train_end_indices(dates, fold, horizon=7, embargo=14, lookback=60)
    expected = walk_forward_splits(
        dates, horizon=7, embargo=14, lookback=60, folds=(fold,)
    )[0].train_end_idx
    np.testing.assert_array_equal(got, expected)
    label_dates = dates[got + 7 - 1]
    assert label_dates.max() <= pd.Timestamp("2019-12-31")
    val_origin = int(pd.DatetimeIndex(dates).searchsorted(pd.Timestamp("2020-01-01")))
    assert int(got[-1]) == val_origin - 7 - 14


def test_selected_batch_reaches_the_model_without_nan_grads(tmp_path) -> None:
    fold = require_fold(1)
    frames = {}
    bundles = []
    for ticker, seed in (("AAA", 0), ("BBB", 1)):
        prices = _prices(seed)
        frame = technical_frame(prices)
        article_dates, embeddings, probs = _articles(pd.DatetimeIndex(frame["date"]), seed + 10)
        frames[ticker] = frame
        bundles.append((ticker, frame["date"], article_dates, embeddings, probs))
        late = prices["date"] >= "2022-01-01"
        edited = prices.copy()
        edited.loc[late, "close"] = edited.loc[late, "close"] * 1.3
        edited.loc[late, "high"] = np.maximum(edited.loc[late, "high"], edited.loc[late, "close"])
        edited_rows, _, _ = ticker_train_rows(
            technical_frame(edited), fold, horizon=7, embargo=14, lookback=60
        )
        original_rows, _, label_dates = ticker_train_rows(frame, fold, horizon=7, embargo=14, lookback=60)
        pd.testing.assert_frame_equal(original_rows, edited_rows)
        assert label_dates.max() <= pd.Timestamp("2019-12-31")

    embeddings = np.concatenate([item[3] for item in bundles], axis=0)
    probs = np.concatenate([item[4] for item in bundles], axis=0)
    mask = np.concatenate(
        [train_article_mask(item[2], item[1], fold) for item in bundles]
    )
    state = fit_compact_state(embeddings, probs, train_mask=mask, n_init=2, max_iter=15, seed=0)
    leaked = embeddings.copy()
    leaked[~mask] += 5.0
    state_leaked = fit_compact_state(leaked, probs, train_mask=mask, n_init=2, max_iter=15, seed=0)
    np.testing.assert_allclose(state.centroids, state_leaked.centroids)
    np.testing.assert_allclose(state.mu, state_leaked.mu)

    features, target, label_dates = selection_sample(frames, fold, horizon=7, embargo=14, lookback=60)
    assert label_dates.max() <= pd.Timestamp("2019-12-31")
    signature = fit_signature(features, target, n_estimators=40, min_child_samples=20, seed=0)
    extra = features.iloc[[0]].copy()
    leaked_features = pd.concat([features, extra], ignore_index=True)
    leaked_target = np.concatenate([target, np.array([50.0])])
    leaked_mask = np.concatenate([np.ones(len(features), dtype=bool), np.array([False])])
    leaked_signature = select_fold_features(
        leaked_features,
        leaked_target,
        leaked_mask,
        n_estimators=40,
        min_child_samples=20,
        seed=0,
    )
    assert signature.ts_columns == leaked_signature.ts_columns
    assert len(signature.ts_columns) == 25
    assert len(signature.text_columns) == 15

    for ticker, trading_dates, article_dates, vectors, ticker_probs in bundles:
        write_parquet(technical_cache_path(tmp_path, ticker), frames[ticker])
        daily = compact_daily_frame(trading_dates, article_dates, vectors, ticker_probs, state)
        write_parquet(text_compact_cache_path(tmp_path, ticker), daily)
    signature_path = signature_cache_path(tmp_path, "dev", 1)
    write_signature(
        signature_path,
        signature,
        {
            "ticker_set": "dev",
            "fold_id": 1,
            "train_start": "2015-01-01",
            "train_end": "2019-12-31",
            "horizon": 7,
            "embargo": 14,
            "lookback": 60,
            "huber_delta": HUBER_DELTA,
        },
    )
    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(tmp_path),
                "features_mode": "selected_40d",
                "signature_path": str(signature_path),
                "technical_cache_dir": str(tmp_path / "cache" / "technical"),
                "text_compact_cache_dir": str(tmp_path / "cache" / "text_compact"),
                "features": ["close", "volume", "open", "high", "low"],
                "target": "close",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
                "normalize": "per_window",
                "news_source": "unused",
                "split": {
                    "train_start": "2015-01-01",
                    "train_end": "2021-12-31",
                    "val_end": "2022-12-31",
                },
                "tickers": {"dev": ["AAA", "BBB"]},
                "text": {
                    "enabled": True,
                    "dim": 15,
                    "compact": True,
                    "window_agg": "recency_weighted",
                    "decay_lambda": 0.03,
                    "renormalize": False,
                },
            },
            "train": {
                "ticker_set": "dev",
                "max_train_windows": 8,
                "max_val_windows": 4,
                "max_test_windows": 4,
            },
            "model": {
                "name": "timexer_selected",
                "d_model": 32,
                "n_heads": 4,
                "e_layers": 1,
                "d_ff": 64,
                "dropout": 0.0,
                "text_dim": 15,
                "fusion": _FUSION,
                "head": {"type": "linear", "hidden": 32, "dropout": 0.0, "pool": "last"},
                "loss": {"kind": "huber", "delta": 0.5},
            },
        }
    )
    train_ds, val_ds, test_ds = build_datasets(cfg)
    assert len(train_ds) == 8 and len(val_ds) == 4 and len(test_ds) == 4
    assert list(cfg.data.features) == list(signature.ts_columns)
    item = train_ds[0]
    assert item["x"].shape == (60, 25)
    assert item["text_seq"].shape == (60, 15)
    assert item["y"].shape == (7,)
    x = item["x"].numpy()
    assert np.allclose(x.mean(axis=0), 0.0, atol=1e-5)
    loader = make_forecast_loader(train_ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert tuple(batch["x"].shape) == (4, 60, 25)
    assert tuple(batch["text_seq"].shape) == (4, 60, 15)
    model = build_model(cfg)
    model.eval()
    text_seq = batch["text_seq"].detach().clone().requires_grad_(True)
    output = model(batch["x"], batch["text"], text_seq=text_seq)
    loss, _metrics = model.compute_loss(
        output,
        batch["y"],
        0.0,
        0.0,
        0.0,
        loss_kind="huber",
        huber_delta=0.5,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert text_seq.grad is not None
    assert torch.isfinite(text_seq.grad).all()
    assert float(text_seq.grad.abs().sum()) > 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.n_features == 25
    assert model.text_dim == 15
    assert model.n_patches == 9
