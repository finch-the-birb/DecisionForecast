"""Offline 40D caches: 340 technical columns, 15D text, one frozen signature.

The signature is fit on walk-forward fold 1 only (train through 2019, with
the same purge and 14-bar embargo as ``walk_forward``). ``build_datasets``
keeps the paper split and only reads these caches. Compact prototypes and μ
are fit on articles whose next-session day falls inside that train window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.text_compact import (
    COMPACT_COLUMNS,
    DECAY_LAMBDA,
    CompactTextState,
    build_compact_daily_series,
)
from src.data.walk_forward import DEFAULT_FOLDS, EMBARGO_BARS, WalkForwardFold
from src.features.selection import LEVEL_COLUMN, TOP_K, TS_WIDTH, FoldFeatureSignature, select_fold_features
from src.features.technical import compute_technical_features, technical_feature_names

log = logging.getLogger(__name__)

SIGNATURE_FORMAT = "selected_40d_v1"
_ONE_DAY_NS = np.int64(24 * 60 * 60 * 10**9)


def technical_cache_path(root: Path, ticker: str) -> Path:
    return Path(root) / "cache" / "technical" / f"{ticker}.parquet"


def text_compact_cache_path(root: Path, ticker: str) -> Path:
    return Path(root) / "cache" / "text_compact" / f"{ticker}.parquet"


def sentiment_cache_path(root: Path, ticker: str) -> Path:
    return Path(root) / "cache" / "text_compact" / "sentiment" / f"{ticker}.parquet"


def signature_cache_path(root: Path, ticker_set: str, fold_id: int = 1) -> Path:
    return Path(root) / "cache" / "selected_signatures" / f"{ticker_set}_fold{fold_id}_signature.json"


def text_state_cache_path(root: Path, ticker_set: str, fold_id: int = 1) -> Path:
    return Path(root) / "cache" / "selected_signatures" / f"{ticker_set}_fold{fold_id}_text_state.npz"


def require_fold(fold_id: int) -> WalkForwardFold:
    for fold in DEFAULT_FOLDS:
        if fold.fold_id == int(fold_id):
            return fold
    known = [fold.fold_id for fold in DEFAULT_FOLDS]
    raise ValueError(f"unknown fold_id {fold_id}; expected one of {known}")


def technical_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Causal technical columns plus the close level used as the forecast target."""
    feats = compute_technical_features(prices).reset_index(drop=True)
    dates = pd.to_datetime(prices["date"], utc=True).dt.tz_convert(None).dt.normalize()
    out = feats.copy()
    out.insert(0, "date", dates.to_numpy())
    out.insert(1, "close", prices["close"].to_numpy(dtype=np.float64))
    return out


def article_bound_index(
    article_dates: pd.Series | np.ndarray | list,
    trading_dates: pd.Series | pd.DatetimeIndex | np.ndarray,
) -> np.ndarray:
    """Index of the trading day that owns each article, or -1 when none does.

    Matches ``build_daily_series``: an article normalized to midnight belongs
    to the next session, and an article more than one day before the first bar
    is dropped.
    """
    days = _naive_days(trading_dates)
    articles = _naive_days(article_dates)
    if len(articles) == 0:
        return np.zeros(0, dtype=np.int64)
    day_ns = days.asi8
    art_ns = articles.asi8
    pos = np.searchsorted(day_ns, art_ns, side="right")
    bound = np.full(len(art_ns), -1, dtype=np.int64)
    candidates = np.flatnonzero(pos < len(day_ns))
    if len(candidates) == 0:
        return bound
    picked = pos[candidates]
    lower = np.empty(len(candidates), dtype=np.int64)
    first = picked == 0
    lower[first] = day_ns[0] - _ONE_DAY_NS
    lower[~first] = day_ns[picked[~first] - 1]
    keep = art_ns[candidates] >= lower
    bound[candidates[keep]] = picked[keep]
    return bound


def train_article_mask(
    article_dates: pd.Series | np.ndarray | list,
    trading_dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    fold: WalkForwardFold,
) -> np.ndarray:
    """True when the article's next session is inside the fold train window."""
    days = _naive_days(trading_dates)
    bound = article_bound_index(article_dates, days)
    mask = np.zeros(len(bound), dtype=bool)
    known = np.flatnonzero(bound >= 0)
    if len(known) == 0:
        return mask
    bound_days = days[bound[known]]
    start = pd.Timestamp(fold.train_start).normalize()
    end = pd.Timestamp(fold.train_end).normalize()
    mask[known] = (bound_days >= start) & (bound_days <= end)
    return mask


def train_end_indices(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    fold: WalkForwardFold,
    *,
    horizon: int,
    embargo: int = EMBARGO_BARS,
    lookback: int,
) -> np.ndarray:
    """Train ``end_idx`` values: first label bar, purge (H-1) plus embargo before val."""
    days = _naive_days(dates)
    if days.hasnans or days.has_duplicates or not days.is_monotonic_increasing:
        raise ValueError("dates must be unique and strictly increasing")
    train_origin = int(days.searchsorted(pd.Timestamp(fold.train_start).normalize(), side="left"))
    val_origin = int(days.searchsorted(pd.Timestamp(fold.val_start).normalize(), side="left"))
    if train_origin >= len(days) or val_origin >= len(days) or train_origin >= val_origin:
        raise ValueError("fold train/val boundaries are not inside this calendar")
    last = min(val_origin - int(horizon) - int(embargo), len(days) - int(horizon))
    first = max(int(lookback), train_origin)
    if last < first:
        return np.zeros(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


def ticker_train_rows(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    horizon: int,
    embargo: int = EMBARGO_BARS,
    lookback: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """One row per train window: features at the last observed bar, log-return label."""
    names = list(technical_feature_names())
    missing = [name for name in names if name not in frame.columns]
    if missing or "close" not in frame.columns or "date" not in frame.columns:
        raise KeyError(f"technical frame is missing {missing or ['date', 'close']}")
    days = _naive_days(frame["date"])
    end_idx = train_end_indices(days, fold, horizon=horizon, embargo=embargo, lookback=lookback)
    if len(end_idx) == 0:
        empty = pd.DataFrame(columns=names)
        return empty, np.zeros(0, dtype=np.float64), pd.DatetimeIndex([])
    pos = end_idx - 1
    label_pos = end_idx + int(horizon) - 1
    close = frame["close"].to_numpy(dtype=np.float64)
    ok = (
        np.isfinite(close[pos])
        & np.isfinite(close[label_pos])
        & (close[pos] > 0.0)
        & (close[label_pos] > 0.0)
    )
    pos = pos[ok]
    label_pos = label_pos[ok]
    block = frame.iloc[pos].loc[:, names].reset_index(drop=True)
    target = np.log(close[label_pos] / close[pos])
    return block, target, days[label_pos]


def selection_sample(
    frames: dict[str, pd.DataFrame],
    fold: WalkForwardFold,
    *,
    horizon: int,
    embargo: int = EMBARGO_BARS,
    lookback: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """Stack train rows across tickers. A calendar that misses the fold is skipped."""
    blocks: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    labels: list[pd.DatetimeIndex] = []
    for ticker, frame in frames.items():
        try:
            block, target, label_dates = ticker_train_rows(
                frame, fold, horizon=horizon, embargo=embargo, lookback=lookback
            )
        except ValueError as exc:
            log.warning("Skipping %s for selection: %s", ticker, exc)
            continue
        if len(block) == 0:
            log.warning("Skipping %s for selection: no finite train rows", ticker)
            continue
        blocks.append(block)
        targets.append(target)
        labels.append(label_dates)
    if not blocks:
        raise ValueError("no train rows for feature selection")
    features = pd.concat(blocks, ignore_index=True)
    label_dates = pd.DatetimeIndex(np.concatenate([np.asarray(index, dtype="datetime64[ns]") for index in labels]))
    return features, np.concatenate(targets), label_dates


def fit_signature(
    features: pd.DataFrame,
    target: np.ndarray,
    *,
    top_k: int = TOP_K,
    n_estimators: int = 200,
    min_child_samples: int = 20,
    seed: int = 0,
) -> FoldFeatureSignature:
    mask = np.ones(len(features), dtype=bool)
    return select_fold_features(
        features,
        target,
        mask,
        top_k=top_k,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        seed=seed,
    )


def write_signature(path: Path, signature: FoldFeatureSignature, meta: dict) -> None:
    payload = {
        "format": SIGNATURE_FORMAT,
        **meta,
        "ts_columns": list(signature.ts_columns),
        "text_columns": list(signature.text_columns),
        "n_train_rows": int(signature.n_train_rows),
        "importances": [
            [str(name), float(signature.importances[name])] for name in signature.importances.index
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_signature(path: Path, *, horizon: int, lookback: int) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing selected signature {path}; run scripts/prepare_selected_data.py"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != SIGNATURE_FORMAT:
        raise ValueError(f"{path} format is {payload.get('format')!r}, expected {SIGNATURE_FORMAT}")
    ts_columns = [str(name) for name in payload["ts_columns"]]
    text_columns = [str(name) for name in payload["text_columns"]]
    if len(ts_columns) not in (26, 31, TS_WIDTH) or ts_columns[0] != LEVEL_COLUMN or text_columns != list(COMPACT_COLUMNS):
        raise ValueError(
            f"{path} has technical columns {ts_columns[:1]}… ({len(ts_columns)}) and text {text_columns}; "
            f"expected {LEVEL_COLUMN} plus 25 or 30 indicators and {list(COMPACT_COLUMNS)}"
        )
    if int(payload["horizon"]) != int(horizon) or int(payload["lookback"]) != int(lookback):
        raise ValueError(
            f"{path} was built for horizon={payload['horizon']} lookback={payload['lookback']}, "
            f"config asks for horizon={horizon} lookback={lookback}"
        )
    payload["ts_columns"] = ts_columns
    payload["text_columns"] = text_columns
    return payload


def compact_daily_frame(
    trading_dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    article_dates: np.ndarray | list,
    embeddings: np.ndarray,
    sentiment_probs: np.ndarray,
    state: CompactTextState,
    lam: float = DECAY_LAMBDA,
) -> pd.DataFrame:
    series, has_news = build_compact_daily_series(
        trading_dates,
        np.asarray(article_dates),
        embeddings,
        sentiment_probs,
        state,
        lam=lam,
    )
    days = _naive_days(trading_dates)
    frame = pd.DataFrame(series, columns=list(COMPACT_COLUMNS))
    frame.insert(0, "date", days.to_numpy())
    frame["has_news"] = np.asarray(has_news, dtype=bool)
    return frame


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def align_compact_text(
    path: Path,
    dates: pd.Series | pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Place the frozen 15D daily text on a price calendar.

    Days present in the cache keep their compact vector and ``has_news``.
    Days the cache does not cover are zeros with ``has_news`` false, so the
    OHLCV row count stays the price calendar. A cache that overlaps no price
    day is an error.
    """
    frame = pd.read_parquet(path)
    missing = [column for column in ("date", "has_news", *COMPACT_COLUMNS) if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing compact columns {missing}")
    frame = frame.copy()
    frame["date"] = _naive_days(frame["date"])
    frame = frame.drop_duplicates("date", keep="last").set_index("date")
    wanted = _naive_days(dates)
    if wanted.has_duplicates:
        raise ValueError("price dates must be unique before compact-text alignment")
    present = wanted.isin(frame.index)
    n_missing = int((~present).sum())
    if n_missing == len(wanted):
        raise ValueError(f"{path.name}: compact text does not overlap the price calendar")
    text = np.zeros((len(wanted), len(COMPACT_COLUMNS)), dtype=np.float32)
    has_news = np.zeros(len(wanted), dtype=bool)
    if bool(present.any()):
        block = frame.loc[wanted[present], list(COMPACT_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError(f"{path.name}: compact text has non-finite values")
        text[np.asarray(present)] = block.astype(np.float32)
        has_news[np.asarray(present)] = (
            frame.loc[wanted[present], "has_news"].to_numpy(dtype=bool)
        )
    return text, has_news, n_missing


def load_selected_arrays(
    technical_path: Path,
    text_path: Path,
    ts_columns: list[str],
    text_columns: list[str],
    train_start: pd.Timestamp | None,
) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray]:
    """Align caches and drop warm-up. Features stay raw; RevIN happens per window."""
    technical = pd.read_parquet(technical_path)
    text = pd.read_parquet(text_path)
    technical["date"] = _naive_days(technical["date"])
    text["date"] = _naive_days(text["date"])
    merged = technical.merge(text, on="date", how="inner", validate="one_to_one")
    if train_start is not None:
        merged = merged.loc[merged["date"] >= pd.Timestamp(train_start).normalize()]
    merged = merged.reset_index(drop=True)
    level_columns = list(dict.fromkeys([*ts_columns, LEVEL_COLUMN]))
    values = merged.loc[:, level_columns].to_numpy(dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    merged = merged.loc[finite].reset_index(drop=True)
    if len(merged) < 2:
        raise ValueError(f"{technical_path.name}: fewer than 2 finite rows after the train_start cut")
    text_seq = merged.loc[:, list(text_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(text_seq).all():
        raise ValueError(f"{text_path.name}: compact text has non-finite values")
    dates = pd.Series(pd.DatetimeIndex(merged["date"]))
    features = merged.loc[:, list(ts_columns)].to_numpy(dtype=np.float32)
    close = merged["close"].to_numpy(dtype=np.float32)
    has_news = merged["has_news"].to_numpy(dtype=bool)
    return features, close, dates, text_seq.astype(np.float32), has_news


def _naive_days(values: pd.Series | pd.DatetimeIndex | np.ndarray | list) -> pd.DatetimeIndex:
    stamps = pd.to_datetime(values, utc=True)
    if isinstance(stamps, pd.Series):
        stamps = pd.DatetimeIndex(stamps)
    elif not isinstance(stamps, pd.DatetimeIndex):
        stamps = pd.DatetimeIndex(stamps)
    if stamps.tz is not None:
        stamps = stamps.tz_convert(None)
    return stamps.normalize()
