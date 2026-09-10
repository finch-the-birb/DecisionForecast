"""Precompute per-article embeddings, train mu, and daily text series."""

from __future__ import annotations

import argparse
import logging

import numpy as np
from omegaconf import OmegaConf

from src.data.dataset import TickerDataStore, _as_naive_day, precache_news_for_tickers
from src.data.embeddings import TextEmbeddingCache
from src.data.paths import resolve_data_root
from src.data.text_series import (
    compute_train_mu,
    load_or_build_daily_series,
    mu_path,
)
from src.utils.device import resolve_device

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Cache article embeddings and daily text series.")
    parser.add_argument("--config", default="configs/data/fnspid.yaml")
    parser.add_argument("--ticker-set", default="dev", choices=["dev", "paper"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    root = resolve_data_root(str(cfg.root))
    tickers = list(cfg.tickers[args.ticker_set])
    train_end = _as_naive_day(cfg.split.train_end)
    text = cfg.text
    cache_root = root / "cache" / "text_series"

    counts = precache_news_for_tickers(root, str(cfg.news_source), tickers)
    for ticker in tickers:
        log.info("News parquet %s: %d rows", ticker, counts.get(ticker, 0))

    device = resolve_device(str(args.device))
    store = TickerDataStore(root, str(cfg.news_source), list(cfg.features), str(cfg.target))
    cache = TextEmbeddingCache(
        cache_dir=cache_root,
        model_name=str(text.encoder),
        dim=int(text.dim),
        encode_batch_size=int(text.get("encode_batch_size", 64)),
        max_seq_tokens=int(text.get("max_seq_tokens", 512)),
        device=device,
    )
    news_by_ticker: dict[str, object] = {}
    loaded: list[str] = []
    for ticker in tickers:
        try:
            series = store.get_series(ticker, train_end=train_end)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.warning("Skipping ticker %s: %s", ticker, exc)
            continue
        news_by_ticker[ticker] = (store.get_news(ticker), series)
        loaded.append(ticker)

    news_only = {t: news_by_ticker[t][0] for t in loaded}
    mu_file = mu_path(cache_root, str(text.encoder), args.ticker_set)
    if mu_file.exists():
        mu = np.load(mu_file).astype(np.float32)
        log.info("mu cache hit %s", mu_file)
    else:
        mu = compute_train_mu(
            loaded,
            news_only,
            cache,
            train_end,
            str(text.article_field),
            str(text.article_fallback),
        )
        mu_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(mu_file, mu)
        log.info("Wrote %s L2=%.4f", mu_file, float(np.linalg.norm(mu)))

    for ticker in loaded:
        news, series = news_by_ticker[ticker]
        e, has_news = load_or_build_daily_series(
            ticker=ticker,
            trading_dates=series.dates,
            news=news,
            cache=cache,
            mu=mu,
            cache_root=cache_root,
            model_name=str(text.encoder),
            article_field=str(text.article_field),
            article_fallback=str(text.article_fallback),
            lam=float(text.decay_lambda),
            missing_policy=str(text.missing_policy),
        )
        frac = float(has_news.mean()) if len(has_news) else 0.0
        log.info("%s daily series E%s has_news=%.3f", ticker, tuple(e.shape), frac)


if __name__ == "__main__":
    main()
