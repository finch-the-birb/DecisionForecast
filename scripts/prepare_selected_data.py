"""Build the 40D caches for one ticker set.

Writes, under the FNSPID root:

* ``cache/technical/{ticker}.parquet`` — causal technical columns plus close
* ``cache/text_compact/{ticker}.parquet`` — 15D daily text
* ``cache/text_compact/sentiment/{ticker}.parquet`` — FinBERT probabilities
* ``cache/selected_signatures/{ticker_set}_fold{id}_signature.json``
* ``cache/selected_signatures/{ticker_set}_fold{id}_text_state.npz``

Prototypes, μ, and the TreeSHAP signature use walk-forward fold 1 train rows
only (2015–2019, purge H−1, embargo 14). The paper split in ``dataset.py``
is unchanged. Run from the repo root.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import TickerDataStore, _load_price_ticker, precache_news_for_tickers
from src.data.embeddings import TextEmbeddingCache, extract_article_payloads
from src.data.paths import local_prices_dir, resolve_data_root
from src.data.selected_pipeline import (
    compact_daily_frame,
    fit_signature,
    require_fold,
    selection_sample,
    sentiment_cache_path,
    signature_cache_path,
    technical_cache_path,
    technical_frame,
    text_compact_cache_path,
    text_state_cache_path,
    train_article_mask,
    write_parquet,
    write_signature,
)
from src.data.text_compact import fit_compact_state
from src.data.walk_forward import EMBARGO_BARS
from src.features.selection import CORR_THRESHOLD, HUBER_DELTA, TOP_K
from src.utils.device import resolve_device

log = logging.getLogger(__name__)


def _load_config(ticker_set: str):
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[f"train.ticker_set={ticker_set}", "data=fnspid"],
        )


def _clean_prices(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "open", "high", "low", "close", "volume"]
    out = frame.dropna(subset=columns).copy()
    positive = (out.loc[:, ["open", "high", "low", "close"]] > 0).all(axis=1)
    return out.loc[positive & (out["volume"] >= 0)].reset_index(drop=True)


def _cached_sentiment(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    probs = frame.loc[:, ["prob_pos", "prob_neg", "prob_neu"]].to_numpy(dtype=np.float64)
    return {str(key): row for key, row in zip(frame["key"].astype(str), probs, strict=True)}


def _sentiment_matrix(
    root: Path,
    ticker: str,
    keys: list[str],
    texts: list[str],
    device: str,
) -> np.ndarray:
    if not keys:
        return np.zeros((0, 3), dtype=np.float64)
    path = sentiment_cache_path(root, ticker)
    cached = _cached_sentiment(path)
    missing = [i for i, key in enumerate(keys) if key not in cached]
    if missing:
        from src.data.text_compact import finbert_probabilities

        log.info("FinBERT %s: %d articles", ticker, len(missing))
        fresh = finbert_probabilities([texts[i] for i in missing], device=device)
        for index, row in zip(missing, fresh, strict=True):
            cached[keys[index]] = np.asarray(row, dtype=np.float64)
        saved = pd.DataFrame(
            {
                "key": list(cached),
                "prob_pos": [cached[key][0] for key in cached],
                "prob_neg": [cached[key][1] for key in cached],
                "prob_neu": [cached[key][2] for key in cached],
            }
        )
        write_parquet(path, saved)
    return np.stack([cached[key] for key in keys], axis=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Cache 340D technical features, 15D text, and a fold signature.")
    parser.add_argument("--ticker-set", required=True, choices=["dev", "paper"])
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(args.ticker_set)
    root = resolve_data_root(str(cfg.data.root))
    fold = require_fold(args.fold)
    horizon = int(cfg.data.horizon)
    lookback = int(cfg.data.lookback_T)
    tickers = [str(ticker) for ticker in cfg.data.tickers[args.ticker_set]]
    text = cfg.data.text
    device = resolve_device(args.device)
    log.info("FNSPID root %s fold %s device %s", root, fold.fold_id, device)

    prices_dir = local_prices_dir(root)
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            prices = _clean_prices(_load_price_ticker(prices_dir, ticker))
            frame = technical_frame(prices)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.warning("Skipping ticker %s: %s", ticker, exc)
            continue
        write_parquet(technical_cache_path(root, ticker), frame)
        frames[ticker] = frame
        log.info("Technical cache %s rows=%d", ticker, len(frame))
    if not frames:
        raise RuntimeError(f"No price series loaded from {prices_dir}")

    source = str(cfg.data.news_source)
    precache_news_for_tickers(root, source, list(frames))
    store = TickerDataStore(root, source, ["close"], "close")
    cache = TextEmbeddingCache(
        cache_dir=root / "cache" / "text_series",
        model_name=str(text.encoder),
        dim=int(text.dim),
        encode_batch_size=int(text.get("encode_batch_size", 64)),
        max_seq_tokens=int(text.get("max_seq_tokens", 512)),
        device=device,
        prefix=str(text.get("prefix", "") or ""),
    )
    article_field = str(text.article_field)
    article_fallback = str(text.article_fallback)
    bundles: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    embeddings: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for ticker, frame in frames.items():
        news = store.get_news(ticker)
        keys, texts, _mask = extract_article_payloads(news, article_field, article_fallback)
        trading_dates = frame["date"]
        if keys:
            cache.ensure_encoded(ticker, keys, texts)
            vectors = np.asarray(cache.lookup(ticker, keys), dtype=np.float64)
            probs = _sentiment_matrix(root, ticker, keys, texts, str(device))
            kept = news.loc[np.asarray(_mask, dtype=bool), "Date"]
            article_dates = pd.to_datetime(kept, utc=True).to_numpy()
        else:
            vectors = np.zeros((0, int(text.dim)), dtype=np.float64)
            probs = np.zeros((0, 3), dtype=np.float64)
            article_dates = np.zeros((0,), dtype="datetime64[ns]")
        mask = train_article_mask(article_dates, trading_dates, fold)
        bundles.append((ticker, trading_dates.to_numpy(), article_dates, vectors, probs))
        embeddings.append(vectors)
        probabilities.append(probs)
        masks.append(mask)
        log.info("Articles %s total=%d train=%d", ticker, len(mask), int(mask.sum()))

    state = fit_compact_state(
        np.concatenate(embeddings, axis=0),
        np.concatenate(probabilities, axis=0),
        train_mask=np.concatenate(masks),
        seed=int(args.seed),
    )
    state_path = text_state_cache_path(root, args.ticker_set, fold.fold_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        state_path,
        centroids=state.centroids,
        mu=state.mu,
        n_train_articles=int(np.concatenate(masks).sum()),
    )
    log.info("Text state %s train_articles=%d", state_path, int(np.concatenate(masks).sum()))
    for ticker, trading_dates, article_dates, vectors, probs in bundles:
        daily = compact_daily_frame(trading_dates, article_dates, vectors, probs, state)
        write_parquet(text_compact_cache_path(root, ticker), daily)

    features, target, label_dates = selection_sample(
        frames,
        fold,
        horizon=horizon,
        embargo=EMBARGO_BARS,
        lookback=lookback,
    )
    signature = fit_signature(
        features,
        target,
        n_estimators=int(args.n_estimators),
        seed=int(args.seed),
    )
    signature_path = signature_cache_path(root, args.ticker_set, fold.fold_id)
    write_signature(
        signature_path,
        signature,
        {
            "ticker_set": args.ticker_set,
            "fold_id": fold.fold_id,
            "train_start": str(pd.Timestamp(fold.train_start).date()),
            "train_end": str(pd.Timestamp(fold.train_end).date()),
            "val_start": str(pd.Timestamp(fold.val_start).date()),
            "horizon": horizon,
            "embargo": EMBARGO_BARS,
            "lookback": lookback,
            "huber_delta": HUBER_DELTA,
            "corr_threshold": CORR_THRESHOLD,
            "top_k": TOP_K,
            "n_tickers": len(frames),
            "tickers": list(frames),
            "label_date_min": str(pd.Timestamp(label_dates.min()).date()),
            "label_date_max": str(pd.Timestamp(label_dates.max()).date()),
            "target": "log_return",
            "target_definition": "log(close[end_idx + horizon - 1] / close[end_idx - 1])",
            "text_fit": "pooled_train_articles",
        },
    )
    log.info(
        "Signature %s rows=%d label_dates=%s..%s columns=%s",
        signature_path,
        signature.n_train_rows,
        pd.Timestamp(label_dates.min()).date(),
        pd.Timestamp(label_dates.max()).date(),
        ", ".join(signature.ts_columns),
    )


if __name__ == "__main__":
    main()
