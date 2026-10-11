from __future__ import annotations

from typing import Any
import numpy as np
import torch
from src.models.losses import pinball_loss


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())


def _horizon_scale(scale: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    while scale.dim() < y.dim():
        scale = scale.unsqueeze(-1)
    return scale.to(dtype=y.dtype, device=y.device)


def directional_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    last_close: torch.Tensor,
) -> float:
    """Directional Accuracy (DA) of predictions at horizon H vs last close T.

    Calculates:
        Delta y = y[:, -1] - last_close
        Delta y_hat = pred[:, -1] - last_close
        DA = mean(sign(Delta y_hat) == sign(Delta y))
    """
    y_h = target[:, -1] if target.dim() >= 2 else target
    y_hat_h = pred[:, -1] if pred.dim() >= 2 else pred
    lc = last_close.reshape(-1).to(dtype=pred.dtype, device=pred.device)
    delta_y = y_h - lc
    delta_pred = y_hat_h - lc
    correct = torch.sign(delta_pred) == torch.sign(delta_y)
    return float(correct.float().mean().item())


def compute_delta_metrics(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    last_raw_close: torch.Tensor,
    raw_target_price: torch.Tensor,
    y_mean: torch.Tensor | None = None,
    y_std: torch.Tensor | None = None,
) -> dict[str, float]:
    """Dual metric computation for delta / return forecasting:

    Set A (delta space):
      - delta_mse: MSE between pred_delta and target_delta
      - delta_mae: MAE in relative returns
      - delta_da: mean(sign(pred_delta[:, -1]) == sign(target_delta[:, -1]))

    Set B (unrolled price space):
      - P_hat_{t+h} = P_t * (1 + y_hat_{delta, h})
      - test_mae_denorm: dollar MAE error vs actual dollar price
      - test_da: directional accuracy on price space vs last_raw_close
      - test_mse_price: normalized price MSE (via RevIN / selective window stats)
    """
    d_mse = mse(pred_delta, target_delta)
    d_mae = mae(pred_delta, target_delta)

    # Set A: delta_da
    p_last = pred_delta[:, -1] if pred_delta.dim() >= 2 else pred_delta
    t_last = target_delta[:, -1] if target_delta.dim() >= 2 else target_delta
    delta_da = float((torch.sign(p_last) == torch.sign(t_last)).float().mean().item())

    # Set B: Unroll price
    lc = _horizon_scale(last_raw_close, pred_delta)
    pred_price = lc * (1.0 + pred_delta)
    mae_denorm = mae(pred_price, raw_target_price)
    da = directional_accuracy(pred_price, raw_target_price, last_raw_close)

    if y_mean is not None and y_std is not None:
        mean = _horizon_scale(y_mean, pred_price)
        std = _horizon_scale(y_std, pred_price)
        test_mse_price = mse((pred_price - mean) / std, (raw_target_price - mean) / std)
    else:
        test_mse_price = mse(pred_price, raw_target_price)

    return {
        "mse": d_mse,
        "mae": d_mae,
        "delta_mse": d_mse,
        "delta_mae": d_mae,
        "delta_da": delta_da,
        "test_mae_denorm": mae_denorm,
        "mae_denorm": mae_denorm,
        "test_da": da,
        "da": da,
        "test_mse_price": test_mse_price,
        "mse_price": test_mse_price,
    }


def empirical_coverage(
    pred: torch.Tensor,
    target: torch.Tensor,
    lower_idx: int = 0,
    upper_idx: int = -1,
) -> float:
    """Fraction of targets falling within the [q_lower, q_upper] prediction interval.

    Nominal coverage for quantiles (0.10, 0.90) is 80% (0.80).
    """
    if pred.dim() < 3 or pred.size(-1) < 2:
        raise ValueError(f"pred must be [B, H, num_quantiles], got {tuple(pred.shape)}")
    q_low = pred[..., lower_idx]
    q_high = pred[..., upper_idx]
    covered = (target >= q_low) & (target <= q_high)
    return float(covered.float().mean().item())


def tunnel_sharpness(
    pred: torch.Tensor,
    lower_idx: int = 0,
    upper_idx: int = -1,
) -> float:
    """Mean width of the prediction tunnel across batch and horizon: mean(q_upper - q_lower)."""
    if pred.dim() < 3 or pred.size(-1) < 2:
        raise ValueError(f"pred must be [B, H, num_quantiles], got {tuple(pred.shape)}")
    q_low = pred[..., lower_idx]
    q_high = pred[..., upper_idx]
    return float((q_high - q_low).mean().item())


def winkler_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.2,
    lower_idx: int = 0,
    upper_idx: int = -1,
) -> float:
    """Winkler Interval Score for nominal (1 - alpha) prediction interval.

    S_alpha(l, u, y) = (u - l) + (2/alpha)*(l - y)*I(y < l) + (2/alpha)*(y - u)*I(y > u).
    For 80% central interval (q_0.10 to q_0.90), alpha = 0.20.
    """
    if pred.dim() < 3 or pred.size(-1) < 2:
        raise ValueError(f"pred must be [B, H, num_quantiles], got {tuple(pred.shape)}")
    l = pred[..., lower_idx]
    u = pred[..., upper_idx]
    width = u - l
    penalty_low = (2.0 / alpha) * (l - target) * (target < l).float()
    penalty_high = (2.0 / alpha) * (target - u) * (target > u).float()
    score = width + penalty_low + penalty_high
    return float(score.mean().item())


def compute_quantile_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.2,
) -> dict[str, float]:
    """Calculate distributional LTSF metrics: empirical coverage, sharpness, and Winkler score."""
    if pred.dim() != 3 or pred.size(-1) < 2:
        raise ValueError(f"pred must be [B, H, num_quantiles], got {tuple(pred.shape)}")
    cov = empirical_coverage(pred, target, lower_idx=0, upper_idx=-1)
    shp = tunnel_sharpness(pred, lower_idx=0, upper_idx=-1)
    wkl = winkler_score(pred, target, alpha=alpha, lower_idx=0, upper_idx=-1)
    pin = float(pinball_loss(pred, target).item())
    return {
        "quantile_coverage": cov,
        "quantile_sharpness": shp,
        "quantile_winkler": wkl,
        "coverage": cov,
        "sharpness": shp,
        "winkler": wkl,
        "pinball": pin,
        "quantile_pinball": pin,
    }


def compute_naive_historical_corridor_metrics(
    test_ds: Any,
    horizon: int = 7,
    quantiles: tuple[float, float] = (0.1, 0.9),
) -> dict[str, float]:
    """Вычисляет Coverage, Sharpness и Winkler для наивного коридора на основе 60-дневной волатильности окна: q = +/- 1.28 * std(window)."""
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    # 80% confidence interval standard normal critical value (z ≈ 1.28155)
    z_score = 1.28155
    steps = np.sqrt(np.arange(1, horizon + 1))

    has_series = hasattr(test_ds, "series") and hasattr(test_ds, "indices")
    close_idx = getattr(test_ds, "close_idx", 0)

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        y = item["y"]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        targets.append(y.cpu() if y.dim() == 1 else y.cpu().squeeze(-1))

        if has_series:
            wi = test_ds.indices[idx]
            ts = test_ds.series[wi.ticker]
            raw_close = ts.features[wi.start_idx : wi.end_idx, close_idx]
            if len(raw_close) > 1:
                daily_rets = (raw_close[1:] - raw_close[:-1]) / np.maximum(raw_close[:-1], 1e-6)
                sigma = float(np.std(daily_rets))
            else:
                sigma = 0.01
        else:
            x = item.get("x")
            if isinstance(x, torch.Tensor) and x.dim() >= 2 and x.size(0) > 1:
                close_vals = x[:, 0].cpu().numpy()
                diffs = close_vals[1:] - close_vals[:-1]
                sigma = float(np.std(diffs))
            else:
                sigma = 0.01

        if sigma < 1e-8 or np.isnan(sigma):
            sigma = 0.01

        q_low = -z_score * sigma * steps
        q_med = np.zeros(horizon)
        q_high = z_score * sigma * steps

        tunnel = np.stack([q_low, q_med, q_high], axis=-1)  # [horizon, 3]
        preds.append(torch.tensor(tunnel, dtype=torch.float32))

    if not targets:
        return {"naive_coverage": 0.0, "naive_sharpness": 0.0, "naive_winkler": 0.0}

    pred_cat = torch.stack(preds, dim=0)   # [N, horizon, 3]
    tgt_cat = torch.stack(targets, dim=0)  # [N, horizon]

    cov = empirical_coverage(pred_cat, tgt_cat)
    shp = tunnel_sharpness(pred_cat)
    wkl = winkler_score(pred_cat, tgt_cat, alpha=0.20)

    return {
        "naive_coverage": cov,
        "naive_sharpness": shp,
        "naive_winkler": wkl,
        "coverage": cov,
        "sharpness": shp,
        "winkler": wkl,
    }


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_mean: torch.Tensor | None = None,
    y_std: torch.Tensor | None = None,
    last_close: torch.Tensor | None = None,
    target_mode: str = "level",
    last_raw_close: torch.Tensor | None = None,
    raw_target_price: torch.Tensor | None = None,
) -> dict[str, float]:
    quantile_metrics: dict[str, float] = {}
    if pred.dim() == 3:
        quantile_metrics = compute_quantile_metrics(pred, target)
        median_idx = 1 if pred.size(-1) == 3 else pred.size(-1) // 2
        pred = pred[..., median_idx]

    if target_mode == "delta" and last_raw_close is not None and raw_target_price is not None:
        out = compute_delta_metrics(
            pred_delta=pred,
            target_delta=target,
            last_raw_close=last_raw_close,
            raw_target_price=raw_target_price,
            y_mean=y_mean,
            y_std=y_std,
        )
        out.update(quantile_metrics)
        return out
    out = {"mse": mse(pred, target), "mae": mae(pred, target)}
    if y_mean is not None and y_std is not None:
        mean = _horizon_scale(y_mean, pred)
        std = _horizon_scale(y_std, pred)
        out["mae_denorm"] = mae(pred * std + mean, target * std + mean)
    if last_close is not None:
        out["da"] = directional_accuracy(pred, target, last_close)
    out.update(quantile_metrics)
    return out

