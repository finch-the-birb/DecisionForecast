from __future__ import annotations

import torch


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
    return {
        "quantile_coverage": cov,
        "quantile_sharpness": shp,
        "quantile_winkler": wkl,
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

