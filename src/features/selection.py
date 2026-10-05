"""Train-only GBDT + TreeSHAP selection for one walk-forward fold.

Phase 4, Sprint 11. LightGBM with Huber loss (delta 0.5, the locked
forecasting setting) is fit on finite train rows only. Importance is the
mean absolute TreeSHAP contribution on those same rows. Features are then
taken in that order until 25 remain, skipping a candidate whose absolute
Spearman correlation with an already kept feature is above 0.85.

The frozen signature is those 25 technical names plus the 15 compact text
columns. Text is not ranked by the booster and is not dropped for
collinearity with the technical block. Rows outside ``train_mask``, and
train rows that are still in warm-up, do not enter the fit, the SHAP
average, or the correlation matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.text_compact import COMPACT_COLUMNS

TOP_K = 25
CORR_THRESHOLD = 0.85
HUBER_DELTA = 0.5


@dataclass(frozen=True)
class FoldFeatureSignature:
    """Column order of the 40D fold vector and the train-only importances."""

    ts_columns: tuple[str, ...]
    text_columns: tuple[str, ...]
    importances: pd.Series
    n_train_rows: int

    def __post_init__(self) -> None:
        importances = pd.Series(self.importances).astype(np.float64).copy()
        object.__setattr__(self, "ts_columns", tuple(self.ts_columns))
        object.__setattr__(self, "text_columns", tuple(self.text_columns))
        object.__setattr__(self, "importances", importances)
        object.__setattr__(self, "n_train_rows", int(self.n_train_rows))

    @property
    def columns(self) -> tuple[str, ...]:
        return self.ts_columns + self.text_columns


def select_fold_features(
    frame: pd.DataFrame,
    target: np.ndarray,
    train_mask: np.ndarray,
    *,
    text_columns: Sequence[str] = COMPACT_COLUMNS,
    top_k: int = TOP_K,
    corr_threshold: float = CORR_THRESHOLD,
    huber_delta: float = HUBER_DELTA,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    seed: int = 0,
) -> FoldFeatureSignature:
    """Fit on the train split and freeze the fold's technical-plus-text names."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not 0.0 < corr_threshold <= 1.0:
        raise ValueError("corr_threshold must be in (0, 1]")
    if huber_delta <= 0.0:
        raise ValueError("huber_delta must be positive")
    if frame.columns.duplicated().any():
        raise ValueError("feature columns must be unique")
    text_names = tuple(text_columns)
    text_set = set(text_names)
    if len(text_set) != len(text_names):
        raise ValueError("text columns must be unique")
    ts_names = [str(column) for column in frame.columns if column not in text_set]
    if not ts_names:
        raise ValueError("frame has no technical columns to select from")
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.asarray(train_mask, dtype=bool)
    if y.shape != (len(frame),) or mask.shape != (len(frame),):
        raise ValueError("target and train_mask must have one entry per row")
    values = frame.loc[:, ts_names].to_numpy(dtype=np.float64)
    usable = mask & np.isfinite(y) & np.isfinite(values).all(axis=1)
    if int(usable.sum()) < 2:
        raise ValueError("need at least 2 finite train rows")
    train_x = pd.DataFrame(values[usable], columns=ts_names)
    train_y = y[usable]
    model = _make_regressor(
        huber_delta=huber_delta,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        seed=seed,
    )
    model.fit(train_x, train_y)
    contributions = np.asarray(model.predict(train_x, pred_contrib=True), dtype=np.float64)
    if contributions.shape != (len(train_x), len(ts_names) + 1):
        raise RuntimeError(
            f"TreeSHAP shape {contributions.shape} does not match "
            f"[{len(train_x)}, {len(ts_names) + 1}]"
        )
    importances = pd.Series(
        np.abs(contributions[:, :-1]).mean(axis=0),
        index=ts_names,
        dtype=np.float64,
    )
    corr = spearman_corr(train_x)
    selected = decorrelated_top_k(importances, corr, top_k, corr_threshold)
    return FoldFeatureSignature(
        ts_columns=tuple(selected),
        text_columns=text_names,
        importances=importances.sort_values(ascending=False, kind="mergesort"),
        n_train_rows=int(usable.sum()),
    )


def apply_fold_signature(frame: pd.DataFrame, signature: FoldFeatureSignature) -> pd.DataFrame:
    """Slice and order columns as frozen for the fold. Does not refit."""
    missing = [column for column in signature.columns if column not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing signature columns {missing}")
    return frame.loc[:, list(signature.columns)].copy()


def decorrelated_top_k(
    importances: pd.Series,
    corr: pd.DataFrame,
    top_k: int,
    threshold: float,
) -> list[str]:
    """Keep up to ``top_k`` names, highest importance first.

    A later feature whose absolute Spearman correlation with any kept
    feature is strictly above ``threshold`` is skipped. Ties in importance
    keep the earlier index order.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    ranked = importances.sort_values(ascending=False, kind="mergesort")
    selected: list[str] = []
    for name in ranked.index:
        if len(selected) >= top_k:
            break
        if _collides(str(name), selected, corr, threshold):
            continue
        selected.append(str(name))
    if len(selected) < top_k:
        raise ValueError(
            f"only {len(selected)} features survived the correlation filter; need {top_k}"
        )
    return selected


def spearman_corr(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman correlation. A constant column correlates as 0 with others."""
    values = frame.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("spearman_corr needs at least 2 rows")
    if not np.isfinite(values).all():
        raise ValueError("spearman_corr requires finite values")
    ranks = np.column_stack([_average_rank(values[:, i]) for i in range(values.shape[1])])
    corr = _pearson_corr(ranks)
    labels = [str(column) for column in frame.columns]
    return pd.DataFrame(corr, index=labels, columns=labels)


def _make_regressor(
    *,
    huber_delta: float,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    seed: int,
) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="huber",
        alpha=float(huber_delta),
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=int(seed),
        deterministic=True,
        n_jobs=1,
        verbosity=-1,
        force_col_wise=True,
    )


def _collides(name: str, selected: list[str], corr: pd.DataFrame, threshold: float) -> bool:
    for kept in selected:
        rho = corr.loc[name, kept]
        if np.isfinite(rho) and abs(float(rho)) > threshold:
            return True
    return False


def _average_rank(column: np.ndarray) -> np.ndarray:
    order = np.argsort(column, kind="mergesort")
    ranks = np.empty(len(column), dtype=np.float64)
    sorted_values = column[order]
    start = 0
    n_rows = len(column)
    while start < n_rows:
        stop = start + 1
        while stop < n_rows and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _pearson_corr(columns: np.ndarray) -> np.ndarray:
    centered = columns - columns.mean(axis=0, keepdims=True)
    scale = np.sqrt((centered**2).sum(axis=0))
    constant = scale <= 0.0
    centered[:, constant] = 0.0
    scale = scale.copy()
    scale[constant] = 1.0
    corr = (centered.T @ centered) / np.outer(scale, scale)
    if constant.any():
        corr[constant, :] = 0.0
        corr[:, constant] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr
