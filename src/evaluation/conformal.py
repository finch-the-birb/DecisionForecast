"""Adaptive Conformal Inference (ACI) for time series uncertainty calibration.

Implements online quantile corridor adaptation following Gibbs & Candès (2021):
    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
    err_t = I(y_t not in [q_{alpha_t/2}, q_{1 - alpha_t/2}])

Adapts confidence intervals dynamically over non-stationary financial regimes,
widening intervals during volatile drawdowns and tightening them during stable periods.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
import torch

from src.evaluation.metrics import empirical_coverage, tunnel_sharpness, winkler_score


def apply_adaptive_conformal_inference(
    preds: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    alpha: float = 0.20,
    gamma: float = 0.005,
    clip_alpha: tuple[float, float] = (0.001, 0.999),
) -> dict[str, Any]:
    """Apply Adaptive Conformal Inference (ACI) to sequential quantile forecasts.

    Args:
        preds: predicted quantiles of shape [N, 3] or [N, H, 3] with columns [q_low, q_med, q_high].
        targets: ground truth targets of shape [N] or [N, H].
        alpha: target miscoverage rate alpha (default 0.20 for nominal 80% coverage).
        gamma: adaptation rate step size (default 0.005).
        clip_alpha: bounds for clipping alpha_t (default (0.001, 0.999)).

    Returns:
        dict containing calibrated predictions, metrics, and alpha trajectory:
            - calibrated_preds: torch.Tensor of shape matching preds
            - coverage: empirical coverage of calibrated corridor
            - sharpness: width of calibrated corridor
            - winkler: Winkler score of calibrated corridor
            - alphas: trajectory of alpha_t over time
            - raw_coverage, raw_sharpness, raw_winkler: base metrics before calibration
    """
    if isinstance(preds, np.ndarray):
        preds_tensor = torch.from_numpy(preds).float()
    else:
        preds_tensor = preds.clone().float()

    if isinstance(targets, np.ndarray):
        targets_tensor = torch.from_numpy(targets).float()
    else:
        targets_tensor = targets.clone().float()

    device = preds_tensor.device
    is_3d = preds_tensor.dim() == 3
    if not is_3d:
        # Standardize 2D [N, 3] to [N, 1, 3]
        preds_tensor = preds_tensor.unsqueeze(1)
        targets_tensor = targets_tensor.unsqueeze(1) if targets_tensor.dim() == 1 else targets_tensor

    N, H, n_q = preds_tensor.shape
    if n_q != 3:
        raise ValueError(f"preds last dimension must be 3 (low, med, high), got {n_q}")

    # Baseline metrics before calibration
    raw_cov = empirical_coverage(preds_tensor, targets_tensor)
    raw_shp = tunnel_sharpness(preds_tensor)
    raw_wkl = winkler_score(preds_tensor, targets_tensor, alpha=alpha)

    # Standard normal critical value for nominal alpha
    z_nom = float(norm.ppf(1.0 - alpha / 2.0))
    min_alpha, max_alpha = clip_alpha

    calibrated_preds = preds_tensor.clone()
    alpha_t = np.full(H, alpha, dtype=np.float64)
    alphas_history: list[np.ndarray] = []

    for t in range(N):
        alphas_history.append(alpha_t.copy())

        # Dynamic quantile critical value z_t
        z_t = norm.ppf(1.0 - alpha_t / 2.0)
        scale_t = torch.from_numpy(z_t / z_nom).float().to(device)  # [H]

        q_low_base = preds_tensor[t, :, 0]
        q_med_base = preds_tensor[t, :, 1]
        q_high_base = preds_tensor[t, :, 2]

        # Scale interval around median
        q_low_cal = q_med_base - scale_t * (q_med_base - q_low_base)
        q_high_cal = q_med_base + scale_t * (q_high_base - q_med_base)

        calibrated_preds[t, :, 0] = q_low_cal
        calibrated_preds[t, :, 1] = q_med_base
        calibrated_preds[t, :, 2] = q_high_cal

        # Evaluate coverage indicator on actual observation
        y_t = targets_tensor[t, :]
        err_t = ((y_t < q_low_cal) | (y_t > q_high_cal)).float().cpu().numpy()

        # Update alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)
        alpha_t = np.clip(alpha_t + gamma * (alpha - err_t), min_alpha, max_alpha)

    alphas_tensor = torch.from_numpy(np.stack(alphas_history, axis=0)).float().to(device)

    # Evaluate metrics on calibrated predictions
    cal_cov = empirical_coverage(calibrated_preds, targets_tensor)
    cal_shp = tunnel_sharpness(calibrated_preds)
    cal_wkl = winkler_score(calibrated_preds, targets_tensor, alpha=alpha)

    if not is_3d:
        calibrated_preds = calibrated_preds.squeeze(1)
        alphas_tensor = alphas_tensor.squeeze(1)

    return {
        "calibrated_preds": calibrated_preds,
        "coverage": cal_cov,
        "sharpness": cal_shp,
        "winkler": cal_wkl,
        "alphas": alphas_tensor,
        "raw_coverage": raw_cov,
        "raw_sharpness": raw_shp,
        "raw_winkler": raw_wkl,
    }
