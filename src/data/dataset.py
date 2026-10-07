from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import Dataset

from src.data.embeddings import TextEmbeddingCache
from src.data.selective_norm import apply_static_and_robust, feature_roles, selective_window
from src.data.paths import local_news_path, local_prices_dir, resolve_data_root
from src.data.text_series import (
    compute_train_mu,
    load_or_build_daily_series,
    mu_path,
    pool_window,
)

log = logging.getLogger(__name__)

_NEWS_CHUNKSIZE = 200_000
_ENDOGENOUS_CHANNELS = (
    ("close",),
    ("close", "volume"),
    ("close", "volume", "open", "high", "low"),
)
_N_EXOGENOUS_TS = 25


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
    text_seq: np.ndarray | None = None
    has_news: np.ndarray | None = None
    ts_exo: np.ndarray | None = None
    target_mean: float = 0.0
    target_std: float = 1.0


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

    def get_series(
        self,
        ticker: str,
        train_end: pd.Timestamp | None = None,
        train_start: pd.Timestamp | None = None,
        normalize: str = "per_ticker_zscore",
    ) -> TickerSeries:
        return _load_ticker_series(
            self.prices_dir,
            ticker,
            self.features,
            self.target,
            train_end,
            price_index=self._price_index,
            train_start=train_start,
            normalize=normalize,
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
    train_start: pd.Timestamp | None = None,
    normalize: str = "per_ticker_zscore",
) -> TickerSeries:
    df = _load_price_ticker(prices_dir, ticker, price_index)
    feat_cols = [_normalize_col(c) for c in features]
    target_col = _normalize_col(target)
    missing = [c for c in feat_cols + [target_col] if c not in df.columns]
    if missing:
        raise KeyError(f"{ticker}: missing columns {missing}; have {list(df.columns)}")
    df = df.dropna(subset=feat_cols + [target_col]).reset_index(drop=True)
    if train_start is not None:
        df = df.loc[df["date"] >= train_start].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError(f"{ticker}: fewer than 2 rows after train_start filter")
    feat_values = df[feat_cols].astype(np.float64).values
    target_raw = df[target_col].astype(np.float64).values
    dates = df["date"]
    if normalize == "per_window":
        return TickerSeries(
            features=feat_values.astype(np.float32),
            target=target_raw.astype(np.float32),
            dates=dates,
        )
    if normalize == "selective":
        train_mask = np.ones(len(df), dtype=bool)
        if train_end is not None:
            train_mask &= (dates <= train_end).to_numpy()
        if int(train_mask.sum()) < 1:
            raise ValueError(f"{ticker}: no train rows for selective scaling")
        scaled = apply_static_and_robust(feat_values, feat_cols, train_mask)
        return TickerSeries(
            features=scaled.astype(np.float32),
            target=target_raw.astype(np.float32),
            dates=dates,
        )
    if normalize != "per_ticker_zscore":
        raise ValueError(f"normalize={normalize!r}; expected per_window|per_ticker_zscore|selective")
    train_mask = np.ones(len(df), dtype=bool)
    if train_end is not None:
        train_mask &= (dates <= train_end).to_numpy()
    if int(train_mask.sum()) < 2:
        raise ValueError(f"{ticker}: fewer than 2 train rows for z-score")
    feat_ref = feat_values[train_mask]
    target_ref = target_raw[train_mask]
    mean = feat_ref.mean(axis=0)
    std = feat_ref.std(axis=0)
    std[std < 1e-8] = 1.0
    features_norm = ((feat_values - mean) / std).astype(np.float32)
    target_mean = float(target_ref.mean())
    target_std = float(target_ref.std()) if target_ref.std() > 1e-8 else 1.0
    target_norm = ((target_raw - target_mean) / target_std).astype(np.float32)
    return TickerSeries(
        features=features_norm,
        target=target_norm,
        dates=dates,
        target_mean=target_mean,
        target_std=target_std,
    )


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


class FNSPIDForecastDataset(Dataset):
    """Sliding-window LTSF samples with pooled and sequential text from daily series."""

    def __init__(
        self,
        indices: list[WindowIndex],
        store: TickerDataStore,
        series: dict[str, TickerSeries],
        horizon: int,
        text_dim: int,
        text_enabled: bool = True,
        window_agg: str = "recency_weighted",
        decay_lambda: float = 0.03,
        renormalize: bool = False,
        normalize: str = "per_ticker_zscore",
        target_idx: int = 0,
        scale_y_from_target: bool = False,
        feature_roles: tuple[str, ...] | None = None,
        close_idx: int | None = None,
        ts_roles: tuple[str, ...] | None = None,
    ) -> None:
        self.indices = indices
        self.store = store
        self.series = series
        self.horizon = horizon
        self.text_dim = text_dim
        self.text_enabled = text_enabled
        self.window_agg = window_agg
        self.decay_lambda = decay_lambda
        self.renormalize = renormalize
        self.normalize = normalize
        self.target_idx = target_idx
        self.scale_y_from_target = bool(scale_y_from_target)
        self.feature_roles = None if feature_roles is None else tuple(feature_roles)
        self.close_idx = self.target_idx if close_idx is None else int(close_idx)
        self.ts_roles = None if ts_roles is None else tuple(ts_roles)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        wi = self.indices[idx]
        ts = self.series[wi.ticker]
        raw_x = ts.features[wi.start_idx : wi.end_idx]
        raw_y = ts.target[wi.end_idx : wi.end_idx + self.horizon]
        if self.normalize == "per_window":
            mean = raw_x.mean(axis=0)
            std = raw_x.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std).astype(np.float64)
            x = ((raw_x - mean) / std).astype(np.float32)
            if self.scale_y_from_target:
                level = np.asarray(ts.target[wi.start_idx : wi.end_idx], dtype=np.float64)
                t_mean = float(level.mean())
                t_std = float(np.std(level))
                if t_std < 1e-8:
                    t_std = 1.0
            else:
                t_mean = float(mean[self.target_idx])
                t_std = float(std[self.target_idx])
            y = ((raw_y.astype(np.float64) - t_mean) / t_std).astype(np.float32)
            y_mean, y_std = t_mean, t_std
        elif self.normalize == "selective":
            if not self.feature_roles:
                raise RuntimeError("normalize=selective requires feature_roles")
            x, y_mean, y_std = selective_window(raw_x, self.feature_roles, self.close_idx)
            y = ((raw_y.astype(np.float64) - y_mean) / y_std).astype(np.float32)
        else:
            x = np.asarray(raw_x, dtype=np.float32)
            y = np.asarray(raw_y, dtype=np.float32)
            y_mean, y_std = float(ts.target_mean), float(ts.target_std)
        t_len = wi.end_idx - wi.start_idx
        if self.text_enabled and ts.text_seq is not None:
            text_seq = ts.text_seq[wi.start_idx : wi.end_idx]
            text = pool_window(
                ts.text_seq,
                wi.start_idx,
                wi.end_idx,
                self.window_agg,
                self.decay_lambda,
                renormalize=self.renormalize,
            )
            if ts.has_news is not None:
                has_news_frac = float(ts.has_news[wi.start_idx : wi.end_idx].mean())
            else:
                has_news_frac = 0.0
        else:
            text_seq = np.zeros((t_len, self.text_dim), dtype=np.float32)
            text = np.zeros(self.text_dim, dtype=np.float32)
            has_news_frac = 0.0
        item = {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32).copy()),
            "y": torch.from_numpy(np.asarray(y, dtype=np.float32).copy()),
            "y_mean": torch.tensor(y_mean, dtype=torch.float32),
            "y_std": torch.tensor(y_std, dtype=torch.float32),
            "text": torch.from_numpy(np.asarray(text, dtype=np.float32)),
            "text_seq": torch.from_numpy(np.asarray(text_seq, dtype=np.float32).copy()),
            "has_news_frac": torch.tensor(has_news_frac, dtype=torch.float32),
            "ticker": wi.ticker,
            "end_idx": wi.end_idx,
            "end_date": wi.end_date,
        }
        if ts.ts_exo is not None:
            if not self.ts_roles:
                raise RuntimeError("exogenous indicators require ts_roles")
            raw_exo = ts.ts_exo[wi.start_idx : wi.end_idx]
            exo, _, _ = selective_window(raw_exo, self.ts_roles, 0)
            item["ts"] = torch.from_numpy(np.asarray(exo, dtype=np.float32).copy())
        return item


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


def _validate_lookback(lookback: int, patch_len: int, patch_stride: int) -> None:
    if lookback < patch_len:
        raise ValueError(f"lookback_T={lookback} < patch_len={patch_len}")
    rem = (lookback - patch_len) % patch_stride
    if rem != 0:
        log.warning(
            "lookback_T=%s patch_len=%s stride=%s leaves remainder %s",
            lookback,
            patch_len,
            patch_stride,
            rem,
        )


def _window_news_fracs(
    has_news: np.ndarray | None, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    n = int(starts.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if has_news is None or len(has_news) == 0:
        return np.zeros(n, dtype=np.float64)
    hn = np.asarray(has_news, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(hn)])
    length = np.maximum((ends - starts).astype(np.float64), 1.0)
    return (c[ends] - c[starts]) / length


def _pooled_text_l2(
    text_seq: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    window_agg: str,
    lam: float,
    renormalize: bool,
) -> np.ndarray:
    n_win = int(starts.shape[0])
    if n_win == 0:
        return np.zeros(0, dtype=np.float64)
    lengths = ends - starts
    unique_len = np.unique(lengths)
    if unique_len.size == 1:
        t_len = int(unique_len[0])
        if t_len <= 0:
            return np.zeros(n_win, dtype=np.float64)
        dim = int(text_seq.shape[1])
        n_days = int(text_seq.shape[0])
        if window_agg == "last":
            pooled = np.asarray(text_seq[ends - 1], dtype=np.float32)
        elif window_agg == "mean":
            c = np.cumsum(text_seq.astype(np.float64), axis=0)
            cpad = np.vstack([np.zeros((1, dim), dtype=np.float64), c])
            pooled = ((cpad[ends] - cpad[starts]) / float(t_len)).astype(np.float32)
        elif window_agg == "recency_weighted":
            ages = np.arange(t_len - 1, -1, -1, dtype=np.float64)
            weights = np.exp(-float(lam) * ages)
            wsum = float(weights.sum())
            n_pool = n_days - t_len + 1
            if n_pool <= 0:
                pooled = np.stack(
                    [
                        pool_window(text_seq, int(s), int(e), window_agg, lam, renormalize=False)
                        for s, e in zip(starts, ends, strict=True)
                    ]
                )
            else:
                seq64 = np.asarray(text_seq, dtype=np.float64)
                acc = np.zeros((n_pool, dim), dtype=np.float64)
                for k, weight in enumerate(weights):
                    acc += weight * seq64[k : k + n_pool]
                pooled = (acc / wsum).astype(np.float32)[starts]
        else:
            raise ValueError(f"window_agg={window_agg!r}; expected last|mean|recency_weighted")
        if renormalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            pooled = pooled / norms.astype(np.float32)
        return np.linalg.norm(pooled.astype(np.float64), axis=1)
    out = np.empty(n_win, dtype=np.float64)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        pooled = pool_window(
            text_seq, int(start), int(end), window_agg, lam, renormalize=renormalize
        )
        out[i] = float(np.linalg.norm(pooled))
    return out


def split_text_coverage(ds: FNSPIDForecastDataset) -> tuple[float, float, float]:
    """Window news-frac / pooled-text L2 from WindowIndex + series (no __getitem__)."""
    n = len(ds)
    if n == 0:
        return 0.0, 0.0, 0.0
    fracs = np.zeros(n, dtype=np.float64)
    norms = np.zeros(n, dtype=np.float64)
    by_ticker: dict[str, list[int]] = {}
    for i, wi in enumerate(ds.indices):
        by_ticker.setdefault(wi.ticker, []).append(i)
    for ticker, idxs in by_ticker.items():
        ts = ds.series[ticker]
        pos = np.asarray(idxs, dtype=np.int64)
        starts = np.asarray([ds.indices[i].start_idx for i in idxs], dtype=np.int64)
        ends = np.asarray([ds.indices[i].end_idx for i in idxs], dtype=np.int64)
        if not ds.text_enabled or ts.text_seq is None:
            continue
        fracs[pos] = _window_news_fracs(ts.has_news, starts, ends)
        norms[pos] = _pooled_text_l2(
            ts.text_seq,
            starts,
            ends,
            ds.window_agg,
            ds.decay_lambda,
            ds.renormalize,
        )
    return float(fracs.mean()), float((fracs == 0.0).mean()), float(norms.mean())


def log_text_coverage(
    splits: dict[str, FNSPIDForecastDataset],
    series: dict[str, TickerSeries],
) -> dict[str, float]:
    """Log per-split / per-ticker news coverage. Returns flat metrics for MLflow."""
    metrics: dict[str, float] = {}
    for ticker, ts in series.items():
        if ts.has_news is None or len(ts.has_news) == 0:
            frac = 0.0
        else:
            frac = float(np.asarray(ts.has_news).mean())
        metrics[f"text_coverage_ticker_{ticker}"] = frac
        log.info("TEXT_COVERAGE ticker=%s days_with_news=%.3f", ticker, frac)
    for split_name, ds in splits.items():
        if len(ds) == 0:
            log.info("TEXT_COVERAGE split=%s windows=0", split_name)
            continue
        mean_frac, zero_frac, mean_norm = split_text_coverage(ds)
        metrics[f"text_coverage_{split_name}_news_frac"] = mean_frac
        metrics[f"text_coverage_{split_name}_zero_windows"] = zero_frac
        metrics[f"text_coverage_{split_name}_mean_text_l2"] = mean_norm
        log.info(
            "TEXT_COVERAGE split=%s windows=%d mean_has_news_frac=%.3f "
            "zero_windows=%.3f mean_text_l2=%.4f",
            split_name,
            len(ds),
            mean_frac,
            zero_frac,
            mean_norm,
        )
    return metrics


def _attach_text_series(
    cfg: DictConfig,
    store: TickerDataStore,
    series: dict[str, TickerSeries],
    tickers: list[str],
    train_end: pd.Timestamp,
    device: str | torch.device | None,
) -> None:
    text_cfg = cfg.data.text
    dim = int(text_cfg.dim)
    cache_root = Path(text_cfg.cache_dir)
    model_name = str(text_cfg.encoder)
    article_field = str(text_cfg.article_field)
    article_fallback = str(text_cfg.article_fallback)
    lam = float(text_cfg.decay_lambda)
    missing_policy = str(text_cfg.missing_policy)
    ticker_set = str(cfg.train.ticker_set)
    train_start = _as_naive_day(cfg.data.split.train_start) if cfg.data.split.get("train_start") else None
    daily_agg = str(text_cfg.get("daily_agg", "mean"))
    if daily_agg != "mean":
        raise ValueError(f"daily_agg={daily_agg!r}; only mean is implemented")
    cache = TextEmbeddingCache(
        cache_dir=cache_root,
        model_name=model_name,
        dim=dim,
        encode_batch_size=int(text_cfg.get("encode_batch_size", 64)),
        max_seq_tokens=int(text_cfg.get("max_seq_tokens", 512)),
        device=device,
        prefix=str(text_cfg.get("prefix", "") or ""),
    )
    news_by_ticker = {ticker: store.get_news(ticker) for ticker in series}
    mu_file = mu_path(cache_root, model_name, ticker_set, train_start, train_end)
    if str(text_cfg.get("neutral", "train_mean")) == "zero":
        mu = np.zeros(dim, dtype=np.float32)
    elif mu_file.exists():
        mu = np.load(mu_file).astype(np.float32).reshape(-1)
        if mu.shape[0] != dim:
            log.warning("mu dim %s != cfg dim %s; recomputing", mu.shape[0], dim)
            mu = compute_train_mu(
                list(series.keys()),
                news_by_ticker,
                cache,
                train_end,
                article_field,
                article_fallback,
                train_start=train_start,
            )
            np.save(mu_file, mu)
    else:
        mu = compute_train_mu(
            list(series.keys()),
            news_by_ticker,
            cache,
            train_end,
            article_field,
            article_fallback,
            train_start=train_start,
        )
        mu_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(mu_file, mu)
    log.info(
        "Text mu ticker_set=%s train_start=%s train_end=%s L2=%.4f dim=%d file=%s",
        ticker_set,
        None if train_start is None else str(train_start.date()),
        str(train_end.date()),
        float(np.linalg.norm(mu)),
        mu.size,
        mu_file.name,
    )
    for ticker, ts in series.items():
        e, has_news = load_or_build_daily_series(
            ticker=ticker,
            trading_dates=ts.dates,
            news=news_by_ticker[ticker],
            cache=cache,
            mu=mu,
            cache_root=cache_root,
            model_name=model_name,
            article_field=article_field,
            article_fallback=article_fallback,
            lam=lam,
            missing_policy=missing_policy,
        )
        if e.shape[-1] != dim:
            raise ValueError(f"{ticker}: text dim {e.shape[-1]} != cfg {dim}")
        ts.text_seq = e
        ts.has_news = has_news


def _attach_compact_text(
    cfg: DictConfig,
    series: dict[str, TickerSeries],
) -> dict[str, TickerSeries]:
    """Replace 768D article text with the frozen 15D compact daily cache."""
    from src.data.selected_pipeline import align_compact_text

    text_dir = Path(str(cfg.data.text_compact_cache_dir))
    kept: dict[str, TickerSeries] = {}
    for ticker, ts in series.items():
        path = text_dir / f"{ticker}.parquet"
        if not path.is_file():
            log.warning("Skipping ticker %s: missing compact text %s", ticker, path)
            continue
        try:
            text_seq, has_news, n_missing = align_compact_text(path, ts.dates)
        except (KeyError, ValueError, OSError) as exc:
            log.warning("Skipping ticker %s: %s", ticker, exc)
            continue
        if text_seq.shape != (len(ts.features), 15):
            log.warning(
                "Skipping ticker %s: compact text shape %s does not match %d price rows",
                ticker,
                tuple(text_seq.shape),
                len(ts.features),
            )
            continue
        ts.text_seq = text_seq
        ts.has_news = has_news
        kept[ticker] = ts
        if n_missing:
            log.info(
                "Compact text %s zero-filled %d/%d price days",
                ticker,
                n_missing,
                len(ts.dates),
            )
    return kept


def _attach_exogenous_ts(
    cfg: DictConfig,
    series: dict[str, TickerSeries],
    feature_names: list[str],
    train_end: pd.Timestamp | None,
    horizon: int,
    lookback: int,
) -> tuple[dict[str, TickerSeries], tuple[str, ...]]:
    """Align the frozen 25 TreeSHAP indicators and scale them with the price rows.

    Rows without a finite indicator vector are dropped from the price series,
    the compact text, and the indicators together. Robust scaling then uses
    the remaining train dates, and ``__getitem__`` still runs ``selective_window``
    so any level column inside the indicator block gets per-window RevIN.
    """
    from src.data.selected_pipeline import read_signature

    signature = read_signature(Path(str(cfg.data.signature_path)), horizon=horizon, lookback=lookback)
    columns = [str(name) for name in signature["ts_columns"][1:]]
    if len(columns) != _N_EXOGENOUS_TS or len(set(columns)) != _N_EXOGENOUS_TS:
        raise ValueError(f"expected {_N_EXOGENOUS_TS} unique indicator columns, got {columns}")
    roles = feature_roles(columns)
    technical_dir = Path(str(cfg.data.technical_cache_dir))
    needed = lookback + horizon
    kept: dict[str, TickerSeries] = {}
    for ticker, ts in series.items():
        path = technical_dir / f"{ticker}.parquet"
        if not path.is_file():
            log.warning("Skipping ticker %s: missing technical cache %s", ticker, path)
            continue
        try:
            aligned = _align_indicator_rows(path, ts.dates, columns)
        except (KeyError, ValueError, OSError) as exc:
            log.warning("Skipping ticker %s: %s", ticker, exc)
            continue
        keep = np.isfinite(aligned).all(axis=1)
        if int(keep.sum()) < needed:
            log.warning(
                "Skipping ticker %s: %d finite indicator rows, need %d",
                ticker,
                int(keep.sum()),
                needed,
            )
            continue
        _slice_ticker_series(ts, keep)
        aligned = aligned[keep]
        train_mask = np.ones(len(ts.features), dtype=bool)
        if train_end is not None:
            train_mask &= (ts.dates <= train_end).to_numpy()
        if int(train_mask.sum()) < 1:
            log.warning("Skipping ticker %s: no train rows after indicator alignment", ticker)
            continue
        ts.features = apply_static_and_robust(ts.features, feature_names, train_mask).astype(np.float32)
        ts.ts_exo = apply_static_and_robust(aligned, columns, train_mask).astype(np.float32)
        kept[ticker] = ts
    return kept, roles


def _align_indicator_rows(path: Path, dates: pd.Series, columns: list[str]) -> np.ndarray:
    """Place technical columns on the price calendar. Missing days stay NaN."""
    from src.data.selected_pipeline import _naive_days

    frame = pd.read_parquet(path)
    missing = [column for column in ("date", *columns) if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing indicator columns {missing[:8]}")
    frame = frame.copy()
    frame["date"] = _naive_days(frame["date"])
    frame = frame.drop_duplicates("date", keep="last").set_index("date")
    wanted = _naive_days(dates)
    if wanted.has_duplicates:
        raise ValueError("price dates must be unique before indicator alignment")
    present = wanted.isin(frame.index)
    if not bool(present.any()):
        raise ValueError(f"{path.name}: technical cache does not overlap the price calendar")
    out = np.full((len(wanted), len(columns)), np.nan, dtype=np.float64)
    block = frame.loc[wanted[present], list(columns)].to_numpy(dtype=np.float64)
    out[np.asarray(present)] = block
    return out


def _slice_ticker_series(ts: TickerSeries, keep: np.ndarray) -> None:
    mask = np.asarray(keep, dtype=bool)
    if mask.shape != (len(ts.features),):
        raise ValueError("keep mask does not match the price series")
    ts.features = np.asarray(ts.features)[mask]
    ts.target = np.asarray(ts.target)[mask]
    ts.dates = pd.Series(pd.DatetimeIndex(np.asarray(ts.dates)[mask])).reset_index(drop=True)
    if ts.text_seq is not None:
        if len(ts.text_seq) != len(mask):
            raise ValueError("text length does not match the price series")
        ts.text_seq = np.asarray(ts.text_seq)[mask]
    if ts.has_news is not None:
        if len(ts.has_news) != len(mask):
            raise ValueError("has_news length does not match the price series")
        ts.has_news = np.asarray(ts.has_news)[mask]


def build_datasets(
    cfg: DictConfig, device: str | torch.device | None = None
) -> tuple[FNSPIDForecastDataset, ...]:
    root = resolve_data_root(cfg.data.root)
    tickers = list(cfg.data.tickers[cfg.train.ticker_set])
    lookback = int(cfg.data.lookback_T)
    horizon = int(cfg.data.horizon)
    features = list(cfg.data.features)
    target_col = str(cfg.data.target)
    train_end = _as_naive_day(cfg.data.split.train_end)
    val_end = _as_naive_day(cfg.data.split.val_end)
    train_start = (
        _as_naive_day(cfg.data.split.train_start) if cfg.data.split.get("train_start") else None
    )
    normalize = str(cfg.data.get("normalize", "per_ticker_zscore"))
    text_enabled = bool(cfg.data.text.get("enabled", True))
    text_dim = int(cfg.data.text.dim)
    _validate_lookback(lookback, int(cfg.data.patch_len), int(cfg.data.patch_stride))
    mode = str(cfg.data.get("features_mode", "ohlcv"))
    selected = mode == "selected_40d"
    scale_y_from_target = False
    ts_roles: tuple[str, ...] | None = None
    series: dict[str, TickerSeries] = {}
    if selected:
        if normalize != "per_window":
            raise ValueError("selected_40d requires data.normalize=per_window")
        from src.data.selected_pipeline import load_selected_arrays, read_signature

        signature = read_signature(Path(str(cfg.data.signature_path)), horizon=horizon, lookback=lookback)
        features = [str(column) for column in signature["ts_columns"]]
        text_columns = [str(column) for column in signature["text_columns"]]
        with open_dict(cfg):
            cfg.data.features = list(features)
            cfg.data.text.dim = len(text_columns)
        text_dim = len(text_columns)
        if not features or features[0] != "close":
            raise ValueError(f"selected signature channel 0 must be close, got {features[:1]}")
        target_idx = 0
        technical_dir = Path(str(cfg.data.technical_cache_dir))
        text_dir = Path(str(cfg.data.text_compact_cache_dir))
        store = None
        for ticker in tickers:
            technical_path = technical_dir / f"{ticker}.parquet"
            text_path = text_dir / f"{ticker}.parquet"
            if not technical_path.is_file() or not text_path.is_file():
                log.warning("Skipping ticker %s: missing selected cache", ticker)
                continue
            try:
                loaded = load_selected_arrays(
                    technical_path, text_path, features, text_columns, train_start
                )
            except (KeyError, ValueError, OSError) as exc:
                log.warning("Skipping ticker %s: %s", ticker, exc)
                continue
            feat_values, close, dates, text_seq, has_news = loaded
            series[ticker] = TickerSeries(
                features=feat_values,
                target=close,
                dates=dates,
                text_seq=text_seq,
                has_news=has_news,
            )
        if not series:
            raise RuntimeError(f"No selected_40d series loaded for {tickers}")
        log.info(
            "Selected signature %s fold=%s train_end=%s technical=%d text=%d",
            cfg.data.signature_path,
            signature.get("fold_id"),
            signature.get("train_end"),
            len(features),
            text_dim,
        )
    else:
        endo = tuple(str(column) for column in features)
        if mode == "dual_ts":
            if normalize != "selective":
                raise ValueError("dual_ts requires data.normalize=selective")
            if endo not in _ENDOGENOUS_CHANNELS:
                raise ValueError(
                    "dual_ts features must be [close], [close, volume], or "
                    f"[close, volume, open, high, low], got {list(endo)}"
                )
            if text_dim != 15:
                raise ValueError(f"dual_ts requires text.dim 15, got {text_dim}")
        try:
            target_idx = [str(f) for f in features].index(target_col)
        except ValueError as exc:
            raise ValueError(f"target {target_col!r} not in features {features}") from exc
        store = TickerDataStore(root, str(cfg.data.news_source), features, target_col)
        price_normalize = "per_window" if mode == "dual_ts" else normalize
        for ticker in tickers:
            try:
                series[ticker] = store.get_series(
                    ticker,
                    train_end=train_end,
                    train_start=train_start,
                    normalize=price_normalize,
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                log.warning("Skipping ticker %s: %s", ticker, exc)
        if not series:
            raise RuntimeError(f"No price series loaded from {store.prices_dir} for {tickers}")
        if mode == "ohlcv_compact_text":
            if normalize not in {"per_window", "selective"}:
                raise ValueError("ohlcv_compact_text requires data.normalize=per_window or selective")
            if [str(column) for column in features] != ["close", "volume", "open", "high", "low"]:
                raise ValueError(
                    "ohlcv_compact_text requires features [close, volume, open, high, low], "
                    f"got {features}"
                )
            if text_dim != 15:
                raise ValueError(f"ohlcv_compact_text requires text.dim 15, got {text_dim}")
            series = _attach_compact_text(cfg, series)
            if not series:
                raise RuntimeError(f"No compact-text series loaded for {tickers}")
            log.info(
                "Compact text cache %s tickers=%d dim=%d",
                cfg.data.text_compact_cache_dir,
                len(series),
                text_dim,
            )
        elif mode == "dual_ts":
            series = _attach_compact_text(cfg, series)
            if not series:
                raise RuntimeError(f"No compact-text series loaded for {tickers}")
            series, ts_roles = _attach_exogenous_ts(
                cfg,
                series,
                [str(column) for column in features],
                train_end,
                horizon,
                lookback,
            )
            if not series:
                raise RuntimeError(f"No dual_ts series loaded for {tickers}")
            log.info(
                "Dual TS cache %s indicators=%d tickers=%d endogenous=%s",
                cfg.data.signature_path,
                len(ts_roles),
                len(series),
                list(endo),
            )
        elif text_enabled:
            _attach_text_series(cfg, store, series, tickers, train_end, device)

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
        horizon=horizon,
        text_dim=text_dim,
        text_enabled=text_enabled,
        window_agg=str(cfg.data.text.get("window_agg", "recency_weighted")),
        decay_lambda=float(cfg.data.text.get("decay_lambda", 0.03)),
        renormalize=bool(cfg.data.text.get("renormalize", False)),
        normalize=normalize,
        target_idx=target_idx,
        scale_y_from_target=scale_y_from_target,
        feature_roles=feature_roles([str(column) for column in features]) if normalize == "selective" else None,
        close_idx=features.index("close") if normalize == "selective" and "close" in [str(column) for column in features] else target_idx,
        ts_roles=ts_roles,
    )
    train_ds = FNSPIDForecastDataset(_cap(train_idx, cfg.train.max_train_windows), **common)
    val_ds = FNSPIDForecastDataset(_cap(val_idx, cfg.train.max_val_windows), **common)
    test_ds = FNSPIDForecastDataset(_cap(test_idx, cfg.train.max_test_windows), **common)
    _log_split("train", train_ds)
    _log_split("val", val_ds)
    _log_split("test", test_ds)
    return train_ds, val_ds, test_ds
