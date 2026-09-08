"""FNSPID data loading, splits, and PyTorch datasets."""

from src.data.dataset import (
    FNSPIDForecastDataset,
    TickerDataStore,
    WindowIndex,
    build_datasets,
    precache_news_for_tickers,
)
from src.data.embeddings import TextEmbeddingCache
from src.data.paths import resolve_data_root

__all__ = [
    "FNSPIDForecastDataset",
    "TextEmbeddingCache",
    "TickerDataStore",
    "WindowIndex",
    "build_datasets",
    "precache_news_for_tickers",
    "resolve_data_root",
]
