"""FNSPID data loading, splits, and PyTorch datasets."""

from src.data.collate import forecast_collate, make_forecast_loader
from src.data.dataset import (
    FNSPIDForecastDataset,
    TickerDataStore,
    WindowIndex,
    build_datasets,
    log_text_coverage,
    precache_news_for_tickers,
    split_text_coverage,
)
from src.data.embeddings import TextEmbeddingCache
from src.data.paths import resolve_data_root
from src.data.text_compact import (
    CompactTextState,
    build_compact_daily_series,
    fit_compact_state,
)
from src.data.text_series import build_daily_series, pool_window
from src.data.walk_forward import (
    DEFAULT_FOLDS,
    EMBARGO_BARS,
    walk_forward_splits,
)

__all__ = [
    "CompactTextState",
    "DEFAULT_FOLDS",
    "EMBARGO_BARS",
    "FNSPIDForecastDataset",
    "TextEmbeddingCache",
    "TickerDataStore",
    "WindowIndex",
    "build_compact_daily_series",
    "build_daily_series",
    "build_datasets",
    "fit_compact_state",
    "forecast_collate",
    "log_text_coverage",
    "make_forecast_loader",
    "pool_window",
    "precache_news_for_tickers",
    "resolve_data_root",
    "split_text_coverage",
    "walk_forward_splits",
]
