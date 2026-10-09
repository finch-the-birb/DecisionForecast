"""Inference trajectory visualization for 2023 test set.

Generates publication-quality charts comparing ground truth stock prices
with 7-day multi-horizon forecast trajectories branch-by-branch.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.data.dataset import build_datasets
from src.evaluation.metrics import directional_accuracy
from src.training.train import _forecast_forward, build_model
from src.utils.device import resolve_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("plot_inference")


def load_inference_config(
    checkpoint_path: Path, config_path: str | None = None
) -> DictConfig:
    """Resolve and load Hydra configuration matching the trained checkpoint."""
    if config_path and Path(config_path).is_file():
        log.info("Loading explicit config from %s", config_path)
        return OmegaConf.load(config_path)

    # Check potential output directory locations
    candidates = [
        checkpoint_path.parent / "config_resolved.yaml",
        checkpoint_path.parent.parent / "config_resolved.yaml",
        checkpoint_path.parent.parent.parent / "config_resolved.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            log.info("Discovered resolved configuration at %s", candidate)
            return OmegaConf.load(candidate)

    # Fallback to composing default Hydra config with paper ticker set
    log.warning("No config_resolved.yaml found near checkpoint. Composing default config.")
    cfg_dir = str(Path(__file__).resolve().parents[2] / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name="config", overrides=["train.ticker_set=paper"])


def collect_ticker_predictions(
    model: torch.nn.Module,
    test_ds: Any,
    ticker: str,
    device: torch.device,
    horizon: int = 7,
    stride: int = 7,
) -> dict[str, Any] | None:
    """Extract ground truth and forecast trajectory branches for a specific ticker."""
    if ticker not in test_ds.series:
        log.warning("Ticker %s not present in test dataset series", ticker)
        return None

    ts = test_ds.series[ticker]
    dates = pd.to_datetime(ts.dates).dt.tz_localize(None)
    prices = np.asarray(ts.target, dtype=np.float64)

    # Identify all window indices in test_ds belonging to this ticker
    ticker_windows = [
        (idx, wi) for idx, wi in enumerate(test_ds.indices) if wi.ticker == ticker
    ]
    if not ticker_windows:
        log.warning("No test windows found for ticker %s", ticker)
        return None

    ticker_windows.sort(key=lambda item: item[1].end_idx)
    min_end_idx = ticker_windows[0][1].end_idx
    max_end_idx = ticker_windows[-1][1].end_idx + horizon

    gt_dates = dates.iloc[min_end_idx - 1 : max_end_idx].values
    gt_prices = prices[min_end_idx - 1 : max_end_idx]

    all_pred_errors: list[float] = []
    all_da_matches: list[bool] = []
    trajectories: list[dict[str, Any]] = []

    for step_num, (dataset_idx, wi) in enumerate(ticker_windows):
        item = test_ds[dataset_idx]
        x = item["x"].unsqueeze(0).to(device)
        text = item["text"].unsqueeze(0).to(device)
        text_seq = item["text_seq"].unsqueeze(0).to(device)
        ts_exo = item["ts"].unsqueeze(0).to(device) if "ts" in item else None
        y_mean = float(item["y_mean"].item())
        y_std = float(item["y_std"].item())

        with torch.no_grad():
            out = _forecast_forward(model, x, text, text_seq, ts_exo)
        pred_norm = out.pred.squeeze(0).cpu().numpy()  # [H]
        pred_dollars = pred_norm * y_std + y_mean

        last_close_date = dates.iloc[wi.end_idx - 1]
        last_close_price = prices[wi.end_idx - 1]
        future_dates = dates.iloc[wi.end_idx : wi.end_idx + len(pred_dollars)].values
        future_prices = prices[wi.end_idx : wi.end_idx + len(pred_dollars)]

        # Error and directional statistics across all test windows
        abs_err = np.abs(pred_dollars - future_prices).mean()
        all_pred_errors.append(float(abs_err))

        delta_actual = future_prices[-1] - last_close_price
        delta_pred = pred_dollars[-1] - last_close_price
        sign_match = np.sign(delta_actual) == np.sign(delta_pred)
        all_da_matches.append(bool(sign_match))

        # Sample trajectories for plotting based on stride
        if step_num % stride == 0 or step_num == len(ticker_windows) - 1:
            traj_dates = [last_close_date, *future_dates]
            traj_prices = [last_close_price, *pred_dollars]
            trajectories.append({
                "dates": traj_dates,
                "prices": traj_prices,
                "start_date": last_close_date,
            })

    mae_dollars = float(np.mean(all_pred_errors))
    da_pct = float(np.mean(all_da_matches)) * 100.0

    return {
        "ticker": ticker,
        "gt_dates": gt_dates,
        "gt_prices": gt_prices,
        "trajectories": trajectories,
        "mae_dollars": mae_dollars,
        "da_pct": da_pct,
    }


def plot_single_ticker(
    ax: plt.Axes, result: dict[str, Any], horizon: int = 7
) -> None:
    """Render a single stock chart with ground truth and forecast trajectory branches."""
    ticker = result["ticker"]
    gt_dates = result["gt_dates"]
    gt_prices = result["gt_prices"]
    trajectories = result["trajectories"]
    mae_val = result["mae_dollars"]
    da_val = result["da_pct"]

    # Plot ground truth actual close
    ax.plot(
        gt_dates,
        gt_prices,
        color="#1a1a1a",
        label=f"Actual Close ({ticker})",
        linewidth=1.75,
        zorder=3,
    )

    # Plot trajectory branches
    color_forecast = "#1f77b4"
    for i, traj in enumerate(trajectories):
        is_first = (i == 0)
        ax.plot(
            traj["dates"],
            traj["prices"],
            color=color_forecast,
            linestyle="--",
            linewidth=1.35,
            marker="o",
            markersize=3.0,
            alpha=0.85,
            label=f"Forecast (H={horizon})" if is_first else None,
            zorder=4,
        )

    ax.set_title(
        f"{ticker} — 7-Day LTSF Inference Trajectories (2023 Test Interval)\n"
        f"Test MAE: ${mae_val:.2f}  |  Directional Accuracy (DA): {da_val:.1f}%",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax.set_ylabel("Price ($)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)


def generate_inference_plots(
    checkpoint_path: str,
    tickers: list[str],
    out_dir: str,
    config_path: str | None = None,
    stride: int = 7,
    device_str: str = "cuda",
    dpi: int = 300,
) -> list[Path]:
    """Generate and save publication-ready inference plots for selected tickers."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = load_inference_config(ckpt_file, config_path)
    # Ensure paper ticker set and full test evaluation
    cfg.train.ticker_set = "paper"
    cfg.train.max_test_windows = None

    device = resolve_device(device_str)
    log.info("Using device: %s", device)

    # Build and load model
    model = build_model(cfg).to(device)
    log.info("Loading weights from %s", ckpt_file)
    state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Load dataset
    log.info("Building 2023 test dataset...")
    _train_ds, _val_ds, test_ds = build_datasets(cfg, device=device)
    log.info("Loaded test dataset with %d windows across %d tickers", len(test_ds), len(test_ds.series))

    saved_files: list[Path] = []
    collected_results: list[dict[str, Any]] = []

    for ticker in tickers:
        log.info("Processing inference trajectories for %s...", ticker)
        result = collect_ticker_predictions(
            model=model,
            test_ds=test_ds,
            ticker=ticker,
            device=device,
            horizon=int(cfg.data.horizon),
            stride=stride,
        )
        if result is None:
            continue
        collected_results.append(result)

        # Plot individual ticker chart
        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=dpi)
        plot_single_ticker(ax, result, horizon=int(cfg.data.horizon))
        plt.tight_layout()
        single_path = out_path / f"{ticker}_inference_2023.png"
        fig.savefig(single_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", single_path)
        saved_files.append(single_path)

    # Generate combined grid if multiple tickers available
    if len(collected_results) >= 2:
        n_plots = min(len(collected_results), 4)
        nrows = 2 if n_plots > 2 else 1
        ncols = 2 if n_plots >= 2 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10.5), dpi=dpi)
        axes_flat = np.atleast_1d(axes).flatten()

        for ax, res in zip(axes_flat, collected_results[:n_plots], strict=False):
            plot_single_ticker(ax, res, horizon=int(cfg.data.horizon))

        # Hide any unused subplots
        for ax in axes_flat[n_plots:]:
            ax.set_visible(False)

        plt.tight_layout()
        combined_path = out_path / "combined_inference_2023.png"
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved combined grid to %s", combined_path)
        saved_files.append(combined_path)

    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2023 test set inference trajectory plots for publication."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model weights checkpoint (e.g. outputs/.../best.pt)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,NVDA,MSFT,JPM",
        help="Comma-separated ticker symbols to plot (e.g. AAPL,NVDA,MSFT,JPM)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/inference_plots",
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config_resolved.yaml",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7,
        help="Stride (in days) between sampled trajectory start dates (default: 7)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda|cpu)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image resolution DPI (default: 300)",
    )

    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    generate_inference_plots(
        checkpoint_path=args.checkpoint,
        tickers=tickers,
        out_dir=args.out_dir,
        config_path=args.config,
        stride=args.stride,
        device_str=args.device,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
