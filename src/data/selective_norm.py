"""Selective scaling: window RevIN for prices, fixed bounds, train-set robust scale.

Level columns keep a per-window mean and std. Oscillators and calendar cycles
are not divided by a local std. Stationary columns use the train-period median
and interquartile range, so a quiet stretch cannot explode.
"""

from __future__ import annotations

import numpy as np

STD_FLOOR = 1e-8
IQR_FLOOR = 1e-8

LEVEL_COLUMNS = frozenset({"close", "open", "high", "low", "vwap"})
CALENDAR_PREFIXES = ("day_sin", "day_cos", "month_sin", "month_cos")
BOUNDED_PREFIXES = ("mrd_", "cgo_", "cfi_")
BOUNDED_COLUMNS = BOUNDED_PREFIXES
STATIONARY_PREFIXES = (
    "log_ret_",
    "parkinson_",
    "gk_",
    "natr_",
    "bb_width_",
    "volume_z_",
    "obv_z_",
    "eii_",
    "queue_acc_",
)


def _normalize_column(val: np.ndarray, role: str) -> np.ndarray:
    """Normalize a column based on its role."""
    if role == "rsi":
        return (val - 50.0) / 50.0
    if role == "pctb":
        return (val - 0.5) * 2.0
    if role == "bounded":
        return np.clip(val, -1.0, 1.0)
    return val


def feature_role(name: str) -> str:
    """Return ``level``, ``rsi``, ``pctb``, ``calendar``, ``bounded``, or ``stationary``."""
    column = str(name)
    if column in LEVEL_COLUMNS:
        return "level"
    if column.startswith("rsi_"):
        return "rsi"
    if column.startswith("bb_pctb_"):
        return "pctb"
    if column.startswith(CALENDAR_PREFIXES):
        return "calendar"
    if column.startswith(BOUNDED_PREFIXES) or column in BOUNDED_COLUMNS:
        return "bounded"
    if column.startswith(STATIONARY_PREFIXES):
        return "stationary"
    return "stationary"


def feature_roles(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(feature_role(name) for name in names)


def apply_static_and_robust(
    values: np.ndarray,
    names: list[str] | tuple[str, ...],
    train_mask: np.ndarray,
) -> np.ndarray:
    """Scale non-level columns. Price levels stay raw for window RevIN.

    RSI maps ``[0, 100]`` to ``[-1, 1]``. Bollinger %B maps ``[0, 1]`` to
    ``[-1, 1]``. Bounded columns (mrd, cgo, cfi) are clamped to ``[-1, 1]``.
    Calendar sines and cosines are copied. Every other column uses the
    train-row median and IQR. A zero IQR becomes 1, so a flat train
    stretch stays finite.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    roles = feature_roles(names)
    if out.ndim != 2 or out.shape[1] != len(roles):
        raise ValueError(f"values shape {out.shape} does not match {len(roles)} columns")
    mask = np.asarray(train_mask, dtype=bool)
    if mask.shape != (out.shape[0],):
        raise ValueError("train_mask must have one flag per row")
    if int(mask.sum()) < 1:
        raise ValueError("selective robust scale needs at least one train row")
    stat_idx = [i for i, role in enumerate(roles) if role == "stationary"]
    if stat_idx:
        reference = out[mask][:, stat_idx]
        median = np.median(reference, axis=0)
        q75 = np.percentile(reference, 75, axis=0)
        q25 = np.percentile(reference, 25, axis=0)
        iqr = np.where((q75 - q25) < IQR_FLOOR, 1.0, q75 - q25)
        out[:, stat_idx] = (out[:, stat_idx] - median) / iqr
    for index, role in enumerate(roles):
        if role in ("rsi", "pctb", "bounded"):
            out[:, index] = _normalize_column(out[:, index], role)
    return out


def selective_window(
    raw_x: np.ndarray,
    roles: tuple[str, ...] | list[str],
    close_idx: int,
) -> tuple[np.ndarray, float, float]:
    """RevIN price columns inside one lookback. Return ``x``, close mean, close std.

    A flat price window uses std 1, so the scaled column is 0 rather than NaN.
    Non-level columns are already scaled and are copied through.
    """
    window = np.asarray(raw_x, dtype=np.float64)
    if window.ndim != 2 or window.shape[1] != len(roles):
        raise ValueError(f"window shape {window.shape} does not match {len(roles)} roles")
    if not 0 <= int(close_idx) < window.shape[1]:
        raise ValueError(f"close_idx {close_idx} is outside {window.shape[1]} columns")
    scaled = window.copy()
    for index, role in enumerate(roles):
        if role != "level":
            continue
        column = window[:, index]
        mean = float(column.mean())
        std = float(column.std())
        if std < STD_FLOOR:
            std = 1.0
        scaled[:, index] = (column - mean) / std
    close = window[:, int(close_idx)]
    target_mean = float(close.mean())
    target_std = float(close.std())
    if target_std < STD_FLOOR:
        target_std = 1.0
    return scaled.astype(np.float32), target_mean, target_std
