"""Econometric GARCH(1,1) corridor baseline for uncertainty forecasting.

Computes conditional multi-step volatility projections on financial returns
using T >= 252 trading days of history per test window:
    E[sigma_{t+h}^2 | F_t] = sigma_bar^2 + (alpha + beta)^{h-1} (sigma_{t+1}^2 - sigma_bar^2)
    sigma_{cum}^2(h) = sum_{j=1}^h E[sigma_{t+j}^2 | F_t]
    q_{0.10}(h) = mu * h - 1.28155 * sigma_{cum}(h)
    q_{0.50}(h) = mu * h
    q_{0.90}(h) = mu * h + 1.28155 * sigma_{cum}(h)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.metrics import empirical_coverage, tunnel_sharpness, winkler_score

log = logging.getLogger(__name__)

Z_80 = 1.28155  # 80% two-sided normal quantile critical value (Phi(0.90) approx 1.28155)


def fit_garch_corridor(
    returns: np.ndarray,
    horizon: int = 7,
    z_score: float = Z_80,
) -> tuple[np.ndarray, float]:
    """Fit GARCH(1,1) on return history and project multi-step quantiles.

    Args:
        returns: 1D array of historical returns (e.g. daily relative returns).
        horizon: forecast horizon H (default 7).
        z_score: standard normal critical value (default 1.28155 for 80% coverage).

    Returns:
        tunnel: array of shape [horizon, 3] with [q_0.10, q_0.50, q_0.90] columns.
        mu: estimated drift per step.
    """
    clean_rets = np.asarray(returns, dtype=np.float64)
    clean_rets = clean_rets[np.isfinite(clean_rets)]

    if len(clean_rets) < 30 or np.allclose(clean_rets, clean_rets[0]):
        # Fallback to simple empirical sample statistics
        return _fallback_corridor(clean_rets, horizon=horizon, z_score=z_score)

    try:
        from arch import arch_model

        # Scale by 100 for numerical stability in optimization
        scaled_rets = clean_rets * 100.0
        am = arch_model(scaled_rets, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
        res = am.fit(disp="off", show_warning=False)

        params = res.params
        mu = float(params.get("mu", 0.0)) / 100.0
        omega = float(params.get("omega", 1e-4)) / 10000.0
        alpha = float(params.get("alpha[1]", 0.05))
        beta = float(params.get("beta[1]", 0.90))

        persistence = alpha + beta
        if 0.0 <= persistence < 0.9999:
            sigma_bar2 = max(omega / (1.0 - persistence), 1e-8)
        else:
            sigma_bar2 = max(float(np.var(clean_rets)), 1e-8)

        # One-step ahead conditional variance
        fc = res.forecast(horizon=horizon)
        if hasattr(fc, "variance") and not fc.variance.empty:
            var_proj = fc.variance.iloc[-1].to_numpy(dtype=np.float64) / 10000.0
        else:
            sigma1_2 = float(res.conditional_volatility[-1] ** 2) / 10000.0
            var_proj = np.zeros(horizon, dtype=np.float64)
            for h in range(1, horizon + 1):
                var_proj[h - 1] = sigma_bar2 + (persistence ** (h - 1)) * (sigma1_2 - sigma_bar2)

        var_proj = np.maximum(var_proj, 1e-8)
        cum_var = np.cumsum(var_proj)
        cum_sigma = np.sqrt(cum_var)

        steps = np.arange(1, horizon + 1, dtype=np.float64)
        q_low = mu * steps - z_score * cum_sigma
        q_med = mu * steps
        q_high = mu * steps + z_score * cum_sigma

        tunnel = np.stack([q_low, q_med, q_high], axis=-1)
        return tunnel, mu

    except Exception as exc:
        log.debug("GARCH fit error: %s, falling back to sample variance", exc)
        return _fallback_corridor(clean_rets, horizon=horizon, z_score=z_score)


def _fallback_corridor(
    returns: np.ndarray,
    horizon: int = 7,
    z_score: float = Z_80,
) -> tuple[np.ndarray, float]:
    """Fallback sample variance corridor if GARCH fitting fails."""
    if len(returns) > 1:
        sigma = float(np.std(returns))
        mu = float(np.mean(returns))
    else:
        sigma = 0.01
        mu = 0.0

    if sigma < 1e-8 or not np.isfinite(sigma):
        sigma = 0.01
    if not np.isfinite(mu):
        mu = 0.0

    steps = np.arange(1, horizon + 1, dtype=np.float64)
    cum_sigma = sigma * np.sqrt(steps)

    q_low = mu * steps - z_score * cum_sigma
    q_med = mu * steps
    q_high = mu * steps + z_score * cum_sigma

    tunnel = np.stack([q_low, q_med, q_high], axis=-1)
    return tunnel, mu


def compute_garch_corridor_metrics(
    test_ds: Any,
    horizon: int = 7,
    min_history: int = 252,
    max_windows: int | None = None,
) -> dict[str, float]:
    """Evaluate econometric GARCH(1,1) uncertainty corridor on a dataset.

    Args:
        test_ds: forecast dataset (e.g. FNSPIDForecastDataset).
        horizon: forecast horizon H (default 7).
        min_history: minimum trading bars of return history (default 252).
        max_windows: optional cap on number of windows to evaluate.

    Returns:
        dict containing coverage, sharpness, winkler, and da metrics.
    """
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    da_correct: list[bool] = []

    has_series = hasattr(test_ds, "series") and hasattr(test_ds, "indices")
    close_idx = getattr(test_ds, "close_idx", 0)

    n_items = len(test_ds)
    if max_windows is not None and max_windows > 0:
        n_items = min(n_items, int(max_windows))

    log.info("Computing GARCH(1,1) corridor metrics on %d windows...", n_items)

    for idx in range(n_items):
        item = test_ds[idx]
        y = item["y"]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        y_1d = y.cpu() if y.dim() == 1 else y.cpu().squeeze(-1)

        # Extract target return trajectory and long return history
        if has_series:
            wi = test_ds.indices[idx]
            ts = test_ds.series[wi.ticker]
            target_mode = getattr(test_ds, "target_mode", "level")
            if target_mode == "delta":
                y_target = y_1d
            else:
                p_t = float(ts.features[wi.end_idx - 1, close_idx])
                p_future = ts.features[wi.end_idx : wi.end_idx + horizon, close_idx]
                if len(p_future) == horizon and p_t > 0:
                    y_ret = (p_future - p_t) / p_t
                    y_target = torch.tensor(y_ret, dtype=torch.float32)
                else:
                    y_target = y_1d

            raw_close = ts.features[: wi.end_idx, close_idx]
            if len(raw_close) > min_history:
                raw_close_slice = raw_close[-min_history - 1 :]
            else:
                raw_close_slice = raw_close
            if len(raw_close_slice) > 1:
                rets = np.diff(raw_close_slice) / np.maximum(raw_close_slice[:-1], 1e-6)
            else:
                rets = np.zeros(0, dtype=np.float64)
        else:
            y_target = y_1d
            x = item.get("x")
            if isinstance(x, torch.Tensor) and x.dim() >= 2 and x.size(0) > 1:
                close_vals = x[:, 0].cpu().numpy()
                rets = np.diff(close_vals) / np.maximum(close_vals[:-1], 1e-6)
            else:
                rets = np.zeros(0, dtype=np.float64)

        targets.append(y_target)

        tunnel, mu = fit_garch_corridor(rets, horizon=horizon, z_score=Z_80)
        tunnel_tensor = torch.tensor(tunnel, dtype=torch.float32)
        preds.append(tunnel_tensor)

        # Directional accuracy at horizon H vs zero / median
        y_last = float(y_1d[-1])
        q_med_last = float(tunnel_tensor[-1, 1])
        da_correct.append((np.sign(q_med_last) == np.sign(y_last)))

    if not targets:
        return {
            "garch_coverage": 0.0,
            "garch_sharpness": 0.0,
            "garch_winkler": 0.0,
            "garch_da": 0.0,
            "coverage": 0.0,
            "sharpness": 0.0,
            "winkler": 0.0,
            "da": 0.0,
        }

    pred_cat = torch.stack(preds, dim=0)   # [N, horizon, 3]
    tgt_cat = torch.stack(targets, dim=0)  # [N, horizon]

    cov = empirical_coverage(pred_cat, tgt_cat)
    shp = tunnel_sharpness(pred_cat)
    wkl = winkler_score(pred_cat, tgt_cat, alpha=0.20)
    da = float(np.mean(da_correct)) if da_correct else 0.50

    metrics = {
        "garch_coverage": cov,
        "garch_sharpness": shp,
        "garch_winkler": wkl,
        "garch_da": da,
        "coverage": cov,
        "sharpness": shp,
        "winkler": wkl,
        "da": da,
    }

    log.info(
        "GARCH(1,1) Corridor Baseline: Coverage(80%%)=%.4f Sharpness=%.4f Winkler=%.4f DA=%.4f (N=%d)",
        cov,
        shp,
        wkl,
        da,
        len(targets),
    )
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate econometric GARCH(1,1) uncertainty corridor baseline.")
    parser.add_argument("--ticker-set", default="paper", choices=["dev", "paper"])
    parser.add_argument("--fold", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--min-history", type=int, default=252)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from src.data.dataset import build_datasets

    config_dir = str(Path(__file__).resolve().parents[2] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"train.ticker_set={args.ticker_set}",
                f"data.fold={args.fold}",
                "data=fnspid",
                "data.text.enabled=false",
            ],
        )

    _, _, test_ds = build_datasets(cfg, device=args.device)
    metrics = compute_garch_corridor_metrics(
        test_ds,
        horizon=args.horizon,
        min_history=args.min_history,
        max_windows=args.max_windows,
    )

    print("\n" + "=" * 60)
    print(f"GARCH(1,1) Baseline Results — Fold {args.fold} ({args.ticker_set})")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
