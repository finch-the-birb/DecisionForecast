from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.data.embeddings import TextEmbeddingCache
from src.data.paths import local_news_path, local_prices_dir, resolve_data_root

log = logging.getLogger(__name__)

_NEWS_CHUNKSIZE = 200_000


@dataclass(frozen=True)
class WindowIndex:
    ticker: str
    end_idx: int
    start_idx: int
    end_date: str


@dataclass
class TickerSeries:
    features: np.ndarray
    target: np.ndarray
    dates: pd.Series


class TickerDataStore:
    """Lazy price/news loaders with per-ticker parquet cache."""

    def __init__(self, root: Path, source: str, features: list[str], target: str) -> None:
        self.root = root
        self.source = source
        self.features = features
        self.target = target
        self.prices_dir = local_prices_dir(root)
        self._price_index = _price_csv_index(self.prices_dir)
        self._news: dict[str, pd.DataFrame] = {}

    def get_series(self, ticker: str, train_end: pd.Timestamp | None = None) -> TickerSeries:
        return _load_ticker_series(
            self.prices_dir,
            ticker,
            self.features,
            self.target,
            train_end,
            price_index=self._price_index,
        )

    def get_news(self, ticker: str) -> pd.DataFrame:
        if ticker not in self._news:
            self._news[ticker] = _load_news_for_ticker(self.root, ticker, self.source)
        return self._news[ticker]


def _normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def _as_naive_day(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _price_csv_index(prices_dir: Path) -> dict[str, Path]:
    """Map case-folded ticker stem -> csv path (FNSPID extracts mixed-case names)."""
    index: dict[str, Path] = {}
    if not prices_dir.exists():
        return index
    for path in prices_dir.glob("*.csv"):
        index.setdefault(path.stem.casefold(), path)
    return index


def _resolve_price_csv(
    prices_dir: Path, ticker: str, price_index: dict[str, Path] | None = None
) -> Path:
    exact = prices_dir / f"{ticker}.csv"
    if exact.exists():
        return exact
    mapping = price_index if price_index is not None else _price_csv_index(prices_dir)
    path = mapping.get(ticker.casefold())
    if path is None:
        raise FileNotFoundError(f"No price file for {ticker}: {exact}")
    if path.name != exact.name:
        log.info("Resolved price file %s -> %s", exact.name, path.name)
    return path


def _load_price_ticker(
    prices_dir: Path, ticker: str, price_index: dict[str, Path] | None = None
) -> pd.DataFrame:
    path = _resolve_price_csv(prices_dir, ticker, price_index)
    df = pd.read_csv(path)
    df.columns = [_normalize_col(c) for c in df.columns]
    date_col = "date" if "date" in df.columns else None
    if date_col is None:
        raise KeyError(f"{path} has no date column; columns={list(df.columns)}")
    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_convert(None).dt.normalize()
    return df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def _load_ticker_series(
    prices_dir: Path,
    ticker: str,
    features: list[str],
    target: str,
    train_end: pd.Timestamp | None = None,
    price_index: dict[str, Path] | None = None,
) -> TickerSeries:
    df = _load_price_ticker(prices_dir, ticker, price_index)
    feat_cols = [_normalize_col(c) for c in features]
    target_col = _normalize_col(target)
    missing = [c for c in feat_cols + [target_col] if c not in df.columns]
    if missing:
        raise KeyError(f"{ticker}: missing columns {missing}; have {list(df.columns)}")
    df = df.dropna(subset=feat_cols + [target_col]).reset_index(drop=True)
    feat_values = df[feat_cols].astype(np.float64).values
    target_raw = df[target_col].astype(np.float64).values
    dates = df["date"]
    if train_end is not None:
        train_mask = dates <= train_end
        if int(train_mask.sum()) < 2:
            raise ValueError(f"{ticker}: fewer than 2 train rows for z-score")
        feat_ref = feat_values[train_mask.to_numpy()]
        target_ref = target_raw[train_mask.to_numpy()]
    else:
        feat_ref = feat_values
        target_ref = target_raw
    mean = feat_ref.mean(axis=0)
    std = feat_ref.std(axis=0)
    std[std < 1e-8] = 1.0
    features_norm = ((feat_values - mean) / std).astype(np.float32)
    target_mean = target_ref.mean()
    target_std = target_ref.std() if target_ref.std() > 1e-8 else 1.0
    target_norm = ((target_raw - target_mean) / target_std).astype(np.float32)
    return TickerSeries(features=features_norm, target=target_norm, dates=dates)


def _news_cache_path(root: Path, source: str, ticker: str) -> Path:
    return root / "cache" / source / f"{ticker}.parquet"


def _parse_news_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date")


def precache_news_for_tickers(
    root: Path,
    source: str,
    tickers: list[str],
    chunksize: int = _NEWS_CHUNKSIZE,
) -> dict[str, int]:
    """Scan the news CSV once and write per-ticker parquet caches."""
    cache_dir = root / "cache" / source
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    remaining = [t for t in tickers if not _news_cache_path(root, source, t).exists()]
    for ticker in tickers:
        if ticker not in remaining:
            cached = pd.read_parquet(_news_cache_path(root, source, ticker))
            counts[ticker] = len(cached)
    if not remaining:
        return counts

    news_path = local_news_path(root, source)
    if not news_path.exists():
        log.warning("News file missing at %s; writing empty caches", news_path)
        for ticker in remaining:
            counts[ticker] = 0
        return counts

    parts: dict[str, list[pd.DataFrame]] = {t: [] for t in remaining}
    wanted = set(remaining)
    for chunk in pd.read_csv(news_path, chunksize=chunksize, low_memory=False):
        if "Stock_symbol" not in chunk.columns:
            raise KeyError(f"{news_path} has no Stock_symbol column")
        hits = chunk.loc[chunk["Stock_symbol"].isin(wanted)]
        if hits.empty:
            continue
        for ticker, group in hits.groupby("Stock_symbol", sort=False):
            parts[str(ticker)].append(group)

    for ticker in remaining:
        df = pd.concat(parts[ticker], ignore_index=True) if parts[ticker] else pd.DataFrame()
        if not df.empty:
            _news_cache_path(root, source, ticker).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(_news_cache_path(root, source, ticker), index=False)
        counts[ticker] = len(df)
        log.info("Cached %s: %d news rows", ticker, counts[ticker])
    return counts


def _load_news_for_ticker(root: Path, ticker: str, source: str) -> pd.DataFrame:
    cache_path = _news_cache_path(root, source, ticker)
    news_path = local_news_path(root, source)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return _parse_news_dates(df)
    if not news_path.exists():
        return pd.DataFrame()
    log.warning("News parquet missing for %s; scanning %s (prefer precache)", ticker, news_path)
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(news_path, chunksize=_NEWS_CHUNKSIZE, low_memory=False):
        hits = chunk.loc[chunk["Stock_symbol"] == ticker]
        if not hits.empty:
            parts.append(hits)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return _parse_news_dates(df)


def _aggregate_window_text(
    news: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    title_field: str,
    summary_field: str,
    max_chars: int,
) -> str:
    if news.empty:
        return ""
    start_utc = pd.Timestamp(start)
    end_utc = pd.Timestamp(end)
    if start_utc.tzinfo is None:
        start_utc = start_utc.tz_localize("UTC")
    else:
        start_utc = start_utc.tz_convert("UTC")
    if end_utc.tzinfo is None:
        end_utc = end_utc.tz_localize("UTC")
    else:
        end_utc = end_utc.tz_convert("UTC")
    mask = (news["Date"] >= start_utc) & (news["Date"] < end_utc)
    subset = news.loc[mask]
    if subset.empty:
        return ""
    chunks: list[str] = []
    for _, row in subset.iterrows():
        title = str(row.get(title_field, "") or "").strip()
        summary = str(row.get(summary_field, "") or "").strip()
        piece = summary or title
        if piece:
            chunks.append(piece)
    return " ".join(chunks)[:max_chars]


class FNSPIDForecastDataset(Dataset):
    """Sliding-window LTSF samples; text aggregated lazily per __getitem__."""

    def __init__(
        self,
        indices: list[WindowIndex],
        store: TickerDataStore,
        series: dict[str, TickerSeries],
        text_cache: TextEmbeddingCache | None,
        horizon: int,
        title_field: str,
        summary_field: str,
        max_chars: int,
        text_enabled: bool = True,
    ) -> None:
        self.indices = indices
        self.store = store
        self.series = series
        self.text_cache = text_cache
        self.horizon = horizon
        self.title_field = title_field
        self.summary_field = summary_field
        self.max_chars = max_chars
        self.text_enabled = text_enabled

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        wi = self.indices[idx]
        ts = self.series[wi.ticker]
        x = ts.features[wi.start_idx : wi.end_idx]
        y = ts.target[wi.end_idx : wi.end_idx + self.horizon]
        text_str = ""
        if self.text_enabled and self.text_cache is not None:
            news = self.store.get_news(wi.ticker)
            start_date = ts.dates.iloc[wi.start_idx]
            end_date = ts.dates.iloc[wi.end_idx]
            text_str = _aggregate_window_text(
                news,
                start_date,
                end_date,
                self.title_field,
                self.summary_field,
                self.max_chars,
            )
            text = self.text_cache.encode(wi.ticker, wi.end_date, text_str)
            text_t = torch.from_numpy(text)
        else:
            dim = self.text_cache.dim if self.text_cache else 384
            text_t = torch.zeros(dim, dtype=torch.float32)
        return {
            "x": torch.from_numpy(x.copy()),
            "y": torch.from_numpy(y.copy()),
            "text": text_t,
            "ticker": wi.ticker,
            "end_idx": wi.end_idx,
            "end_date": wi.end_date,
            "raw_text": text_str,
        }


def _cap_recent(items: list[WindowIndex], limit) -> list[WindowIndex]:
    """Keep the most recent windows, round-robin across tickers.

    Prefix slices would take the earliest AAPL bars (1980s) and skip news.
    """
    if limit is None:
        return items
    n = int(limit)
    if n <= 0 or len(items) <= n:
        return items
    by_ticker: dict[str, list[WindowIndex]] = {}
    for wi in items:
        by_ticker.setdefault(wi.ticker, []).append(wi)
    tickers = list(by_ticker.keys())
    pointers = {ticker: len(windows) - 1 for ticker, windows in by_ticker.items()}
    picked: list[WindowIndex] = []
    while len(picked) < n:
        progressed = False
        for ticker in tickers:
            if len(picked) >= n:
                break
            idx = pointers[ticker]
            if idx < 0:
                continue
            picked.append(by_ticker[ticker][idx])
            pointers[ticker] = idx - 1
            progressed = True
        if not progressed:
            break
    picked.reverse()
    return picked


def _log_split(name: str, ds: FNSPIDForecastDataset) -> None:
    if len(ds) == 0:
        log.info("Split %s: 0 windows", name)
        return
    dates = [wi.end_date for wi in ds.indices]
    tickers = sorted({wi.ticker for wi in ds.indices})
    log.info(
        "Split %s: %d windows, end_date %s .. %s, tickers=%s",
        name,
        len(ds),
        dates[0],
        dates[-1],
        ",".join(tickers),
    )


def build_datasets(cfg: DictConfig) -> tuple[FNSPIDForecastDataset, ...]:
    root = resolve_data_root(cfg.data.root)
    tickers = list(cfg.data.tickers[cfg.train.ticker_set])
    lookback = int(cfg.data.lookback_T)
    horizon = int(cfg.data.horizon)
    features = list(cfg.data.features)
    target_col = str(cfg.data.target)
    train_end = _as_naive_day(cfg.data.split.train_end)
    val_end = _as_naive_day(cfg.data.split.val_end)
    text_enabled = bool(cfg.data.text.get("enabled", True))

    store = TickerDataStore(root, str(cfg.data.news_source), features, target_col)
    series: dict[str, TickerSeries] = {}
    for ticker in tickers:
        try:
            series[ticker] = store.get_series(ticker, train_end=train_end)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.warning("Skipping ticker %s: %s", ticker, exc)
    if not series:
        raise RuntimeError(f"No price series loaded from {store.prices_dir} for {tickers}")

    text_cache = (
        TextEmbeddingCache(
            cache_dir=Path(cfg.data.text.cache_dir),
            model_name=str(cfg.data.text.encoder),
            dim=int(cfg.data.text.dim),
            max_chars=int(cfg.data.text.max_chars),
        )
        if text_enabled
        else None
    )

    train_idx: list[WindowIndex] = []
    val_idx: list[WindowIndex] = []
    test_idx: list[WindowIndex] = []

    for ticker, ts in series.items():
        max_end = len(ts.features) - horizon
        for end_idx in range(lookback, max_end):
            last_target_date = _as_naive_day(ts.dates.iloc[end_idx + horizon - 1])
            if last_target_date <= train_end:
                bucket = train_idx
            elif last_target_date <= val_end:
                bucket = val_idx
            else:
                bucket = test_idx
            end_date = _as_naive_day(ts.dates.iloc[end_idx])
            bucket.append(
                WindowIndex(
                    ticker=ticker,
                    end_idx=end_idx,
                    start_idx=end_idx - lookback,
                    end_date=str(end_date.date()),
                )
            )

    def _cap(items: list[WindowIndex], limit) -> list[WindowIndex]:
        return _cap_recent(items, limit)

    common = dict(
        store=store,
        series=series,
        text_cache=text_cache,
        horizon=horizon,
        title_field=str(cfg.data.text.title_field),
        summary_field=str(cfg.data.text.summary_field),
        max_chars=int(cfg.data.text.max_chars),
        text_enabled=text_enabled,
    )
    train_ds = FNSPIDForecastDataset(_cap(train_idx, cfg.train.max_train_windows), **common)
    val_ds = FNSPIDForecastDataset(_cap(val_idx, cfg.train.max_val_windows), **common)
    test_ds = FNSPIDForecastDataset(_cap(test_idx, cfg.train.max_test_windows), **common)
    _log_split("train", train_ds)
    _log_split("val", val_ds)
    _log_split("test", test_ds)
    return train_ds, val_ds, test_ds
