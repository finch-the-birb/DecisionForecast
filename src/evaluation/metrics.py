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
    if target_mode == "delta" and last_raw_close is not None and raw_target_price is not None:
        return compute_delta_metrics(
            pred_delta=pred,
            target_delta=target,
            last_raw_close=last_raw_close,
            raw_target_price=raw_target_price,
            y_mean=y_mean,
            y_std=y_std,
        )
    out = {"mse": mse(pred, target), "mae": mae(pred, target)}
    if y_mean is not None and y_std is not None:
        mean = _horizon_scale(y_mean, pred)
        std = _horizon_scale(y_std, pred)
        out["mae_denorm"] = mae(pred * std + mean, target * std + mean)
    if last_close is not None:
        out["da"] = directional_accuracy(pred, target, last_close)
    return out
