"""Generate publication-quality corridor comparison plots: GARCH(1,1) vs TimeXer Raw vs TimeXer ACI.

Target tickers: AAPL (Big Tech), NVDA (High Volatility), JPM (Financial Sector).
Saves figures to agent-handoff/2026-10-14_1200_walkforward-folds-and-garch-baseline/figures/
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
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.data.dataset import build_datasets
from src.evaluation.garch_baseline import fit_garch_corridor, Z_80

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("corridor_plots")


def build_test_dataset(fold: int = 3, ticker_set: str = "paper"):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"train.ticker_set={ticker_set}",
                f"data.fold={fold}",
                "data=fnspid_dual_ts",
                f"data.signature_path=data/signatures/paper_fold{fold}_signature_volatility_30ts.json",
            ],
        )
    _, _, test_ds = build_datasets(cfg, device="cpu")
    return test_ds, cfg


def generate_plots_for_fold(
    fold: int = 3,
    run_dir: Path = Path("outputs/2026-10-13/19-47-23"),
    tickers: tuple[str, ...] = ("AAPL", "NVDA", "JPM"),
    output_dir: Path = Path("agent-handoff/2026-10-14_1200_walkforward-folds-and-garch-baseline/figures"),
    stride: int = 7,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    test_ds, _ = build_test_dataset(fold=fold)

    # Load saved model predictions
    preds_data = torch.load(run_dir / "test_predictions.pt", weights_only=True)
    cal_preds = torch.load(run_dir / "calibrated_predictions.pt", weights_only=True)
    raw_preds = preds_data["preds"].numpy()  # [N, H, 3]
    cal_preds = cal_preds.numpy()            # [N, H, 3]

    # Map windows by ticker
    ticker_window_indices = {t: [] for t in tickers}
    for dataset_idx, wi in enumerate(test_ds.indices):
        if wi.ticker in ticker_window_indices:
            ticker_window_indices[wi.ticker].append((dataset_idx, wi))

    for ticker in tickers:
        windows = ticker_window_indices[ticker]
        if not windows:
            log.warning("No windows found for %s", ticker)
            continue

        windows.sort(key=lambda item: item[1].end_idx)
        ts = test_ds.series[ticker]
        dates = pd.to_datetime(ts.dates).dt.tz_localize(None)
        prices = np.asarray(ts.target, dtype=np.float64)

        min_end = windows[0][1].end_idx
        max_end = windows[-1][1].end_idx + 7
        gt_dates = dates.iloc[min_end - 1 : max_end].values
        gt_prices = prices[min_end - 1 : max_end]

        # Sample trajectories according to stride
        garch_trajs = []
        raw_trajs = []
        aci_trajs = []

        for step_num, (d_idx, wi) in enumerate(windows):
            if step_num % stride != 0 and step_num != len(windows) - 1:
                continue

            last_close = float(prices[wi.end_idx - 1])
            anchor_date = dates.iloc[wi.end_idx - 1]
            future_dates = dates.iloc[wi.end_idx : wi.end_idx + 7].values

            # GARCH prediction
            raw_close = ts.features[: wi.end_idx, 0]
            raw_close_slice = raw_close[-253:] if len(raw_close) > 252 else raw_close
            rets = np.diff(raw_close_slice) / np.maximum(raw_close_slice[:-1], 1e-6)
            garch_tunnel, _ = fit_garch_corridor(rets, horizon=7, z_score=Z_80)
            garch_low = last_close * (1.0 + garch_tunnel[:, 0])
            garch_med = last_close * (1.0 + garch_tunnel[:, 1])
            garch_high = last_close * (1.0 + garch_tunnel[:, 2])

            # TimeXer Raw
            q_raw = raw_preds[d_idx]
            raw_low = last_close * (1.0 + q_raw[:, 0])
            raw_med = last_close * (1.0 + q_raw[:, 1])
            raw_high = last_close * (1.0 + q_raw[:, 2])

            # TimeXer ACI
            q_aci = cal_preds[d_idx]
            aci_low = last_close * (1.0 + q_aci[:, 0])
            aci_med = last_close * (1.0 + q_aci[:, 1])
            aci_high = last_close * (1.0 + q_aci[:, 2])

            traj_d = [anchor_date, *future_dates]
            garch_trajs.append({
                "dates": traj_d,
                "low": [last_close, *garch_low],
                "med": [last_close, *garch_med],
                "high": [last_close, *garch_high],
            })
            raw_trajs.append({
                "dates": traj_d,
                "low": [last_close, *raw_low],
                "med": [last_close, *raw_med],
                "high": [last_close, *raw_high],
            })
            aci_trajs.append({
                "dates": traj_d,
                "low": [last_close, *aci_low],
                "med": [last_close, *aci_med],
                "high": [last_close, *aci_high],
            })

        # Create 3-panel comparison figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, sharey=True)

        panels = [
            (axes[0], garch_trajs, "GARCH(1,1) Econometric Baseline", "#e66101", "#fdb863"),
            (axes[1], raw_trajs, "TimeXer Hierarchical MLP (Raw Quantiles)", "#2b83ba", "#abdda4"),
            (axes[2], aci_trajs, "TimeXer Hierarchical MLP (ACI Calibrated)", "#7b3294", "#c2a5cf"),
        ]

        for ax, trajs, title, line_color, fill_color in panels:
            # Ground truth line
            ax.plot(gt_dates, gt_prices, color="#111111", linewidth=1.8, label="Actual Close Price", zorder=4)

            for i, tr in enumerate(trajs):
                is_first = (i == 0)
                ax.fill_between(
                    tr["dates"],
                    tr["low"],
                    tr["high"],
                    color=fill_color,
                    alpha=0.35,
                    label="80% Uncertainty Corridor [q0.10, q0.90]" if is_first else None,
                    zorder=2,
                )
                ax.plot(
                    tr["dates"],
                    tr["med"],
                    color=line_color,
                    linestyle="--",
                    linewidth=1.3,
                    marker="o",
                    markersize=2.5,
                    alpha=0.85,
                    label="Forecast Median (H=7)" if is_first else None,
                    zorder=3,
                )

            ax.set_title(f"{ticker} — {title} (Fold {fold} Test: 2023)", fontsize=12, fontweight="bold", pad=8)
            ax.set_ylabel("Price ($)", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.55)
            ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)

        axes[-1].set_xlabel("Date", fontsize=11, fontweight="bold")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.tight_layout()

        fig_path = output_dir / f"corridor_comparison_{ticker.lower()}_fold{fold}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
        log.info("Saved %s", fig_path)

    # Also create a composite 1x3 overview plot showing the 3 tickers side-by-side
    log.info("Generating multi-ticker composite comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax_idx, ticker in enumerate(tickers):
        ax = axes[ax_idx]
        windows = ticker_window_indices[ticker]
        windows.sort(key=lambda item: item[1].end_idx)
        ts = test_ds.series[ticker]
        dates = pd.to_datetime(ts.dates).dt.tz_localize(None)
        prices = np.asarray(ts.target, dtype=np.float64)

        min_end = windows[0][1].end_idx
        max_end = windows[-1][1].end_idx + 7
        gt_dates = dates.iloc[min_end - 1 : max_end].values
        gt_prices = prices[min_end - 1 : max_end]

        ax.plot(gt_dates, gt_prices, color="#111111", linewidth=1.6, label="Actual Close", zorder=5)

        for step_num, (d_idx, wi) in enumerate(windows):
            if step_num % stride != 0 and step_num != len(windows) - 1:
                continue
            last_close = float(prices[wi.end_idx - 1])
            anchor_date = dates.iloc[wi.end_idx - 1]
            future_dates = dates.iloc[wi.end_idx : wi.end_idx + 7].values
            traj_d = [anchor_date, *future_dates]

            # GARCH corridor
            raw_close = ts.features[: wi.end_idx, 0]
            raw_close_slice = raw_close[-253:] if len(raw_close) > 252 else raw_close
            rets = np.diff(raw_close_slice) / np.maximum(raw_close_slice[:-1], 1e-6)
            garch_tunnel, _ = fit_garch_corridor(rets, horizon=7, z_score=Z_80)
            g_low = [last_close, *(last_close * (1.0 + garch_tunnel[:, 0]))]
            g_high = [last_close, *(last_close * (1.0 + garch_tunnel[:, 2]))]

            # ACI corridor
            q_aci = cal_preds[d_idx]
            a_low = [last_close, *(last_close * (1.0 + q_aci[:, 0]))]
            a_high = [last_close, *(last_close * (1.0 + q_aci[:, 2]))]
            a_med = [last_close, *(last_close * (1.0 + q_aci[:, 1]))]

            is_first = (step_num == 0)
            ax.fill_between(
                traj_d, g_low, g_high,
                color="#e66101", alpha=0.18,
                label="GARCH(1,1) Corridor" if is_first else None,
                zorder=2,
            )
            ax.fill_between(
                traj_d, a_low, a_high,
                color="#7b3294", alpha=0.30,
                label="TimeXer ACI Corridor" if is_first else None,
                zorder=3,
            )
            ax.plot(
                traj_d, a_med,
                color="#7b3294", linestyle="--", linewidth=1.1,
                label="TimeXer Median" if is_first else None,
                zorder=4,
            )

        ax.set_title(f"{ticker} (Fold {fold} — 2023)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10, fontweight="bold")
        ax.set_ylabel("Price ($)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    composite_path = output_dir / f"corridor_comparison_overview_fold{fold}.png"
    plt.savefig(composite_path, dpi=300)
    plt.close(fig)
    log.info("Saved composite plot %s", composite_path)


def main():
    parser = argparse.ArgumentParser(description="Generate corridor comparison figures.")
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--run-dir", type=str, default="outputs/2026-10-13/19-47-23")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--stride", type=int, default=7)
    args = parser.parse_args()

    fold_subdirs = {
        1: "fold1_bull_market_2021",
        2: "fold2_bear_market_2022",
        3: "fold3_recovery_2023",
    }
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("plots/walkforward_garch_aci") / fold_subdirs.get(args.fold, f"fold{args.fold}")

    generate_plots_for_fold(
        fold=args.fold,
        run_dir=Path(args.run_dir),
        output_dir=out_dir,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()

