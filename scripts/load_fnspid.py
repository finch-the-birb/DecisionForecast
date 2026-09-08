"""Helpers for loading FNSPID data with pandas / Hugging Face datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO_ID = "Zihan1004/FNSPID"
DATA_ROOT = Path(__file__).resolve().parents[1] / "Data" / "FNSPID"


def local_news_path(name: str = "nasdaq_exteral_data") -> Path:
    """Return path to a local news CSV (after download)."""
    return DATA_ROOT / "Stock_news" / f"{name}.csv"


def local_prices_dir() -> Path:
    """Return directory with per-ticker price CSVs (after download + extract)."""
    nested = DATA_ROOT / "Stock_price" / "full_history" / "full_history"
    if nested.exists():
        return nested
    return DATA_ROOT / "Stock_price" / "full_history"


def load_news_sample(
    nrows: int = 10_000,
    source: str = "nasdaq_exteral_data",
) -> pd.DataFrame:
    """Load a slice of news data for exploratory analysis."""
    path = local_news_path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_fnspid.py"
        )
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def _news_cache_path(ticker: str, source: str) -> Path:
    return DATA_ROOT / "cache" / source / f"{ticker}.parquet"


def load_news_for_ticker(
    ticker: str,
    *,
    max_rows: int | None = None,
    source: str = "nasdaq_exteral_data",
    cache: bool = True,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Load news rows for one ticker from the local FNSPID CSV.

    Uses vectorized pandas chunk filtering (no per-row Python loop).
    When ``max_rows`` is None and ``cache=True``, saves/loads a per-ticker parquet
    file so repeat queries are near-instant.
    """
    path = local_news_path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_fnspid.py"
        )

    cache_path = _news_cache_path(ticker, source)
    if max_rows is None and cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    parts: list[pd.DataFrame] = []
    collected = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        hits = chunk.loc[chunk["Stock_symbol"] == ticker]
        if hits.empty:
            continue
        if max_rows is not None:
            hits = hits.head(max_rows - collected)
        parts.append(hits)
        collected += len(hits)
        if max_rows is not None and collected >= max_rows:
            break

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if max_rows is None and cache and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def load_price_ticker(ticker: str) -> pd.DataFrame:
    """Load OHLCV history for a single ticker symbol."""
    path = local_prices_dir() / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No price file for {ticker}: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def load_news_hf(split: str = "train", streaming: bool = True):
    """Stream news via Hugging Face datasets (no full local copy required)."""
    return load_dataset(REPO_ID, split=split, streaming=streaming)
