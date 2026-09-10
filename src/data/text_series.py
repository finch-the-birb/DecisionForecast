"""Daily news embedding series with next-session binding and missing-day policy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.embeddings import (
    TextEmbeddingCache,
    encoder_slug,
    extract_article_payloads,
)

log = logging.getLogger(__name__)

_POLICIES = ("zero", "carry", "decay")
_WINDOW_AGGS = ("last", "mean", "recency_weighted")


def build_daily_series(
    trading_dates: pd.Series | np.ndarray,
    article_dates: np.ndarray,
    article_vecs: np.ndarray,
    mu: np.ndarray,
    lam: float,
    missing_policy: str = "decay",
) -> tuple[np.ndarray, np.ndarray]:
    """Map articles in [prev_trading_day, d) onto trading day d.

    Days before the first news are zeros. A day with news is mean(raw) - mu.
    A day without news follows missing_policy: zero | carry | decay (× exp(-lam)).
    """
    if missing_policy not in _POLICIES:
        raise ValueError(f"missing_policy={missing_policy!r}; expected {_POLICIES}")
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates, utc=False)).tz_localize(None).normalize()
    n_days = len(dates)
    dim = int(np.asarray(mu).reshape(-1).shape[0])
    mu = np.asarray(mu, dtype=np.float32).reshape(dim)
    e = np.zeros((n_days, dim), dtype=np.float32)
    has_news = np.zeros(n_days, dtype=bool)
    if n_days == 0:
        return e, has_news

    if article_vecs.size == 0 or len(article_dates) == 0:
        return e, has_news

    art_dates = pd.DatetimeIndex(pd.to_datetime(article_dates, utc=True)).tz_convert(None).normalize()
    vecs = np.asarray(article_vecs, dtype=np.float32)
    order = np.argsort(art_dates.values)
    art_dates = art_dates[order]
    vecs = vecs[order]
    art_ns = art_dates.asi8
    day_ns = dates.asi8

    seen_news = False
    left = 0
    n_art = len(art_ns)
    for i in range(n_days):
        lo = day_ns[i - 1] if i > 0 else np.iinfo(np.int64).min
        hi = day_ns[i]
        while left < n_art and art_ns[left] < lo:
            left += 1
        right = left
        while right < n_art and art_ns[right] < hi:
            right += 1
        if right > left:
            raw_mean = vecs[left:right].mean(axis=0)
            e[i] = raw_mean - mu
            has_news[i] = True
            seen_news = True
        elif not seen_news:
            e[i] = 0.0
        elif missing_policy == "zero":
            e[i] = 0.0
        elif missing_policy == "carry":
            e[i] = e[i - 1]
        else:
            e[i] = e[i - 1] * np.float32(np.exp(-float(lam)))
        left = right
    return e, has_news


def pool_window(
    series: np.ndarray,
    start_idx: int,
    end_idx: int,
    window_agg: str,
    lam: float,
    renormalize: bool = False,
) -> np.ndarray:
    if window_agg not in _WINDOW_AGGS:
        raise ValueError(f"window_agg={window_agg!r}; expected {_WINDOW_AGGS}")
    sl = series[start_idx:end_idx]
    if sl.shape[0] == 0:
        raise ValueError("empty window")
    if window_agg == "last":
        pooled = sl[-1].astype(np.float32, copy=True)
    elif window_agg == "mean":
        pooled = sl.mean(axis=0).astype(np.float32)
    else:
        n = sl.shape[0]
        ages = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-float(lam) * ages)
        pooled = (weights[:, None] * sl.astype(np.float64)).sum(axis=0) / weights.sum()
        pooled = pooled.astype(np.float32)
    if renormalize:
        norm = float(np.linalg.norm(pooled))
        if norm > 1e-8:
            pooled = pooled / np.float32(norm)
    return pooled


def series_cache_dir(cache_root: Path, model_name: str) -> Path:
    path = Path(cache_root) / encoder_slug(model_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def mu_path(cache_root: Path, model_name: str, ticker_set: str) -> Path:
    return series_cache_dir(cache_root, model_name) / f"mu_{ticker_set}.npy"


def ticker_series_path(cache_root: Path, model_name: str, ticker: str) -> Path:
    return series_cache_dir(cache_root, model_name) / f"{ticker}.npz"


def compute_train_mu(
    tickers: list[str],
    news_by_ticker: dict[str, pd.DataFrame],
    cache: TextEmbeddingCache,
    train_end: pd.Timestamp,
    article_field: str,
    article_fallback: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for ticker in tickers:
        news = news_by_ticker.get(ticker)
        if news is None or news.empty:
            continue
        keys, texts, mask = extract_article_payloads(news, article_field, article_fallback)
        if not keys:
            continue
        cache.ensure_encoded(ticker, keys, texts)
        vecs = cache.lookup(ticker, keys)
        dates = pd.to_datetime(news.iloc[np.flatnonzero(mask)]["Date"], utc=True, errors="coerce")
        dates = dates.dt.tz_convert(None).dt.normalize()
        keep = (dates <= train_end).to_numpy()
        if keep.any():
            chunks.append(vecs[keep])
    if not chunks:
        log.warning("No train articles for mu; using zeros dim=%d", cache.dim)
        return np.zeros(cache.dim, dtype=np.float32)
    mu = np.concatenate(chunks, axis=0).mean(axis=0).astype(np.float32)
    return mu


def load_or_build_daily_series(
    ticker: str,
    trading_dates: pd.Series,
    news: pd.DataFrame,
    cache: TextEmbeddingCache,
    mu: np.ndarray,
    cache_root: Path,
    model_name: str,
    article_field: str,
    article_fallback: str,
    lam: float,
    missing_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = ticker_series_path(cache_root, model_name, ticker)
    date_iso = pd.DatetimeIndex(pd.to_datetime(trading_dates)).tz_localize(None).strftime("%Y-%m-%d").to_numpy()
    if path.exists():
        with np.load(path, allow_pickle=True) as data:
            cached_dates = np.asarray(data["dates"]).astype(str)
            if (
                cached_dates.shape == date_iso.shape
                and np.array_equal(cached_dates, date_iso)
                and data["E"].shape[-1] == mu.shape[-1]
            ):
                return np.asarray(data["E"], dtype=np.float32), np.asarray(data["has_news"], dtype=bool)
    keys, texts, mask = extract_article_payloads(news, article_field, article_fallback)
    if keys:
        cache.ensure_encoded(ticker, keys, texts)
        vecs = cache.lookup(ticker, keys)
        art_dates = news.iloc[np.flatnonzero(mask)]["Date"].to_numpy()
    else:
        vecs = np.zeros((0, cache.dim), dtype=np.float32)
        art_dates = np.array([], dtype="datetime64[ns]")
    e, has_news = build_daily_series(
        trading_dates, art_dates, vecs, mu, lam, missing_policy=missing_policy
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, E=e, has_news=has_news, dates=date_iso)
    return e, has_news
