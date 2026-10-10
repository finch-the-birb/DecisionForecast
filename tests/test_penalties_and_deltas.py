"""Unit tests for loss penalties grid and delta return forecasting."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader, Dataset

from src.data.collate import forecast_collate
from src.evaluation.metrics import (
    compute_delta_metrics,
    compute_metrics,
    directional_accuracy,
    mae,
    mse,
)
from src.models.losses import compute_task_loss, correlation_penalty, directional_penalty
from src.models.outputs import ModelOutput, compute_pred_loss
from src.training.train import evaluate


def test_directional_penalty_properties() -> None:
    # Target goes up: [1.0, 2.0, 3.0]
    target = torch.tensor([[1.0, 2.0, 3.0]])
    # Pred goes opposite (down): [3.0, 2.0, 1.0] -> penalty should be positive
    pred_bad = torch.tensor([[3.0, 2.0, 1.0]])
    # Pred goes up: [1.0, 2.5, 3.5] -> penalty should be zero
    pred_good = torch.tensor([[1.0, 2.5, 3.5]])

    pen_bad = directional_penalty(pred_bad, target)
    pen_good = directional_penalty(pred_good, target)

    assert pen_bad.item() > 0.0
    assert pen_good.item() == 0.0


def test_correlation_penalty_properties() -> None:
    # Target: [1.0, 2.0, 3.0, 4.0]
    target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # Perfectly positively correlated: [10.0, 20.0, 30.0, 40.0] -> r=1.0 -> penalty=0.0
    pred_pos = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    # Perfectly negatively correlated: [40.0, 30.0, 20.0, 10.0] -> r=-1.0 -> penalty=2.0
    pred_neg = torch.tensor([[40.0, 30.0, 20.0, 10.0]])

    pen_pos = correlation_penalty(pred_pos, target)
    pen_neg = correlation_penalty(pred_neg, target)

    assert pytest.approx(pen_pos.item(), abs=1e-4) == 0.0
    assert pytest.approx(pen_neg.item(), abs=1e-4) == 2.0


def test_compute_task_loss_additive_penalties() -> None:
    torch.manual_seed(42)
    pred = torch.randn(8, 7)
    target = torch.randn(8, 7)

    loss_base, m_base = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.0, alpha_corr=0.0
    )
    loss_dir, m_dir = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.1, alpha_corr=0.0
    )
    loss_corr, m_corr = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.0, alpha_corr=0.3
    )
    loss_both, m_both = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.1, alpha_corr=0.3
    )

    # Penalties must strictly increase the loss
    assert loss_dir.item() > loss_base.item()
    assert loss_corr.item() > loss_base.item()
    assert loss_both.item() > loss_dir.item()
    assert loss_both.item() > loss_corr.item()

    # Penalties must be additive
    expected_both = loss_base + 0.1 * m_both["l_dir"] + 0.3 * m_both["l_corr"]
    assert pytest.approx(loss_both.item(), rel=1e-5) == expected_both.item()


def test_compute_pred_loss_forwards_penalties() -> None:
    torch.manual_seed(42)
    pred = torch.randn(4, 7)
    target = torch.randn(4, 7)
    output = ModelOutput(pred=pred)

    l_task_0, _ = compute_pred_loss(
        output, target, lambda_c=0.0, lambda_e=0.0, lambda_d=0.0,
        loss_kind="huber", gamma_dir=0.0, alpha_corr=0.0
    )
    l_task_p, _ = compute_pred_loss(
        output, target, lambda_c=0.0, lambda_e=0.0, lambda_d=0.0,
        loss_kind="huber", gamma_dir=0.2, alpha_corr=0.3
    )
    assert l_task_p.item() > l_task_0.item()


def test_delta_forward_and_inverse_unrolling() -> None:
    # Given Pt = 100.0, future prices [101, 102, 99, 105, 100, 98, 103]
    pt = 100.0
    raw_future = np.array([101.0, 102.0, 99.0, 105.0, 100.0, 98.0, 103.0], dtype=np.float32)

    # Forward delta (returns)
    delta_target = (raw_future - pt) / pt
    expected_delta = np.array([0.01, 0.02, -0.01, 0.05, 0.0, -0.02, 0.03], dtype=np.float32)
    np.testing.assert_allclose(delta_target, expected_delta, rtol=1e-5)

    # Inverse unrolling: P_hat = Pt * (1 + delta)
    pred_delta_t = torch.from_numpy(delta_target).unsqueeze(0)  # [1, 7]
    pt_t = torch.tensor([pt], dtype=torch.float32)  # [1]
    raw_future_t = torch.from_numpy(raw_future).unsqueeze(0)  # [1, 7]

    unrolled = pt_t.unsqueeze(-1) * (1.0 + pred_delta_t)
    np.testing.assert_allclose(unrolled.numpy(), raw_future_t.numpy(), rtol=1e-5)


def test_compute_delta_metrics_dual_sets() -> None:
    # 2 batch samples, horizon 7
    pt = torch.tensor([100.0, 200.0])
    raw_prices = torch.tensor([
        [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        [198.0, 196.0, 194.0, 192.0, 190.0, 188.0, 186.0],
    ])
    # Target delta
    tgt_delta = (raw_prices - pt.unsqueeze(1)) / pt.unsqueeze(1)

    # Perfect prediction in delta space
    pred_delta = tgt_delta.clone()
    metrics_perfect = compute_delta_metrics(
        pred_delta=pred_delta,
        target_delta=tgt_delta,
        last_raw_close=pt,
        raw_target_price=raw_prices,
    )

    assert pytest.approx(metrics_perfect["delta_mse"], abs=1e-6) == 0.0
    assert pytest.approx(metrics_perfect["delta_mae"], abs=1e-6) == 0.0
    assert metrics_perfect["delta_da"] == 1.0
    assert pytest.approx(metrics_perfect["test_mae_denorm"], abs=1e-5) == 0.0
    assert metrics_perfect["test_da"] == 1.0
    assert pytest.approx(metrics_perfect["test_mse_price"], abs=1e-5) == 0.0

    # Wrong direction prediction
    pred_delta_wrong = -tgt_delta.clone()
    metrics_wrong = compute_delta_metrics(
        pred_delta=pred_delta_wrong,
        target_delta=tgt_delta,
        last_raw_close=pt,
        raw_target_price=raw_prices,
    )
    assert metrics_wrong["delta_da"] == 0.0
    assert metrics_wrong["test_da"] == 0.0
    assert metrics_wrong["delta_mse"] > 0.0
    assert metrics_wrong["test_mae_denorm"] > 0.0


def test_compute_metrics_delegates_to_delta() -> None:
    pt = torch.tensor([50.0])
    raw_prices = torch.tensor([[52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0]])
    tgt_delta = (raw_prices - pt.unsqueeze(1)) / pt.unsqueeze(1)

    m = compute_metrics(
        pred=tgt_delta,
        target=tgt_delta,
        target_mode="delta",
        last_raw_close=pt,
        raw_target_price=raw_prices,
    )
    assert "delta_mse" in m
    assert "delta_da" in m
    assert "test_mae_denorm" in m
    assert "test_mse_price" in m
    assert m["delta_da"] == 1.0


def test_forecast_collate_with_delta_keys() -> None:
    sample = {
        "x": torch.zeros(60, 5),
        "y": torch.zeros(7),
        "target": torch.zeros(7),
        "text": torch.zeros(15),
        "text_seq": torch.zeros(60, 15),
        "has_news_frac": torch.tensor(0.5),
        "y_mean": torch.tensor(100.0),
        "y_std": torch.tensor(2.0),
        "ticker": "AAPL",
        "end_idx": 60,
        "end_date": "2023-01-01",
        "last_raw_close": torch.tensor(150.0),
        "raw_target_price": torch.ones(7) * 155.0,
    }
    batch = forecast_collate([sample, sample])
    assert "last_raw_close" in batch
    assert batch["last_raw_close"].shape == (2,)
    assert "raw_target_price" in batch
    assert batch["raw_target_price"].shape == (2, 7)
    assert "target" in batch
    assert batch["target"].shape == (2, 7)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(5, 7)

    def forward(self, x, text, text_seq=None, ts=None):
        # x is [B, T=60, 5], pool last step -> [B, 7]
        return ModelOutput(pred=self.fc(x[:, -1]))


class DummyDeltaDataset(Dataset):
    def __init__(self) -> None:
        self.target_mode = "delta"

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": torch.randn(60, 5),
            "y": torch.randn(7) * 0.02,
            "target": torch.randn(7) * 0.02,
            "text": torch.zeros(15),
            "text_seq": torch.zeros(60, 15),
            "has_news_frac": torch.tensor(0.0),
            "y_mean": torch.tensor(150.0),
            "y_std": torch.tensor(2.5),
            "ticker": "AAPL",
            "end_idx": 60 + idx,
            "end_date": "2023-01-01",
            "last_raw_close": torch.tensor(150.0),
            "raw_target_price": torch.tensor([151.0, 152.0, 153.0, 152.0, 154.0, 155.0, 156.0]),
        }


def test_evaluate_records_dual_metrics() -> None:
    dataset = DummyDeltaDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=forecast_collate)
    model = DummyModel()
    device = torch.device("cpu")

    results = evaluate(model, loader, device, target_mode="delta")

    required_keys = [
        "mse",
        "mae",
        "delta_mse",
        "delta_mae",
        "delta_da",
        "test_mae_denorm",
        "test_da",
        "test_mse_price",
    ]
    for key in required_keys:
        assert key in results, f"Missing required metric {key}"
        assert isinstance(results[key], float)


def test_new_configs_composition() -> None:
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    configs_to_test = [
        "c1_hierarchical_penalties_g01_a01",
        "c1_hierarchical_penalties_g01_a03",
        "c1_hierarchical_penalties_g02_a01",
        "c1_hierarchical_penalties_g02_a03",
        "c1_hierarchical_delta",
    ]

    for cfg_name in configs_to_test:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            cfg = compose(config_name="config", overrides=[f"model={cfg_name}", "data=fnspid_dual_ts"])
            assert cfg.model.name == cfg_name
            assert cfg.model.variant == "timexer_hierarchical"
            assert cfg.model.fft_mode == "bridge"
            assert cfg.model.e_layers == 3
            if "delta" in cfg_name:
                assert cfg.model.target_mode == "delta"
            else:
                assert cfg.model.loss.gamma_dir > 0.0
                assert cfg.model.loss.alpha_corr > 0.0


def _setup_mock_dual_ts_environment(tmp_path: Path):
    import json
    import pandas as pd
    from omegaconf import OmegaConf
    from src.data.text_compact import COMPACT_COLUMNS

    dates = pd.bdate_range("2021-01-04", periods=180)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, len(dates)))
    prices_dir = tmp_path / "Stock_price" / "full_history" / "full_history"
    prices_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": rng.integers(1000, 2000, len(dates)),
        }
    ).to_csv(prices_dir / "AAA.csv", index=False)

    text_dir = tmp_path / "cache" / "text_compact"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_df = pd.DataFrame(rng.normal(size=(len(dates), 15)), columns=list(COMPACT_COLUMNS))
    text_df.insert(0, "date", dates)
    text_df["has_news"] = True
    text_df.to_parquet(text_dir / "AAA.parquet", index=False)

    indicator_names = ["rsi_n14", "bb_pctb_n20", "parkinson_n5", *[f"log_ret_{i}" for i in range(1, 23)]]
    tech_dir = tmp_path / "cache" / "technical"
    tech_dir.mkdir(parents=True, exist_ok=True)
    tech_df = pd.DataFrame(rng.normal(size=(len(dates), 25)), columns=indicator_names)
    tech_df.insert(0, "date", dates)
    tech_df.to_parquet(tech_dir / "AAA.parquet", index=False)

    sig_dir = tmp_path / "cache" / "selected_signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)
    sig_path = sig_dir / "paper_fold1_signature.json"
    signature = {
        "format": "selected_40d_v1",
        "horizon": 7,
        "lookback": 60,
        "ts_columns": ["close", *indicator_names],
        "text_columns": list(COMPACT_COLUMNS),
    }
    sig_path.write_text(json.dumps(signature), encoding="utf-8")
    return prices_dir, text_dir, tech_dir, sig_path


def test_end_to_end_smoke_training_penalties(tmp_path: Path) -> None:
    from omegaconf import OmegaConf
    from src.training.train import run_training

    prices_dir, text_dir, tech_dir, sig_path = _setup_mock_dual_ts_environment(tmp_path)
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "smoke_penalties"},
            "paths": {
                "output_dir": str(tmp_path / "outputs"),
                "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
            },
            "data": {
                "root": str(tmp_path),
                "features_mode": "dual_ts",
                "features": ["close", "volume", "open", "high", "low"],
                "target": "close",
                "target_mode": "level",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
                "normalize": "selective",
                "news_source": "unused",
                "signature_path": str(sig_path),
                "technical_cache_dir": str(tech_dir),
                "text_compact_cache_dir": str(text_dir),
                "split": {
                    "train_start": "2021-01-01",
                    "train_end": "2021-05-31",
                    "val_end": "2021-07-31",
                },
                "tickers": {"dev": ["AAA"]},
                "text": {
                    "enabled": True,
                    "dim": 15,
                    "compact": True,
                    "window_agg": "mean",
                    "decay_lambda": 0.03,
                    "renormalize": False,
                },
            },
            "model": {
                "name": "c1_hierarchical_penalties_g01_a01",
                "variant": "timexer_hierarchical",
                "fusion": {
                    "kind": "c1_hierarchical",
                    "text_at_head": False,
                    "text_to_patches": False,
                    "text_as_exogenous": True,
                    "global_to_patch": True,
                },
                "fft_mode": "bridge",
                "d_model": 32,
                "n_heads": 2,
                "e_layers": 3,
                "d_ff": 64,
                "dropout": 0.0,
                "text_dim": 15,
                "n_features": 5,
                "n_ts_features": 25,
                "use_prototypes": False,
                "head": {"pool": "last", "type": "linear", "hidden": 64, "dropout": 0.0},
                "loss": {
                    "kind": "huber",
                    "delta": 0.5,
                    "lambda_c": 0.0,
                    "lambda_e": 0.0,
                    "lambda_d": 0.0,
                    "gamma_dir": 0.1,
                    "alpha_corr": 0.1,
                },
            },
            "train": {
                "ticker_set": "dev",
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 4,
                "num_workers": 0,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "patience": 2,
                "eval_every": 1,
                "max_train_windows": 16,
                "max_val_windows": 8,
                "max_test_windows": 8,
                "mlflow": {"enabled": False},
            },
            "explain": {},
        }
    )
    test_metrics = run_training(cfg)
    assert "mse" in test_metrics
    assert "mae" in test_metrics
    assert "da" in test_metrics
    assert test_metrics["mse"] > 0.0


def test_end_to_end_smoke_training_delta(tmp_path: Path) -> None:
    import json
    from omegaconf import OmegaConf
    from src.training.train import run_training

    prices_dir, text_dir, tech_dir, sig_path = _setup_mock_dual_ts_environment(tmp_path)
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "smoke_delta"},
            "paths": {
                "output_dir": str(tmp_path / "outputs"),
                "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
            },
            "data": {
                "root": str(tmp_path),
                "features_mode": "dual_ts",
                "features": ["close", "volume", "open", "high", "low"],
                "target": "close",
                "target_mode": "delta",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
                "normalize": "selective",
                "news_source": "unused",
                "signature_path": str(sig_path),
                "technical_cache_dir": str(tech_dir),
                "text_compact_cache_dir": str(text_dir),
                "split": {
                    "train_start": "2021-01-01",
                    "train_end": "2021-05-31",
                    "val_end": "2021-07-31",
                },
                "tickers": {"dev": ["AAA"]},
                "text": {
                    "enabled": True,
                    "dim": 15,
                    "compact": True,
                    "window_agg": "mean",
                    "decay_lambda": 0.03,
                    "renormalize": False,
                },
            },
            "model": {
                "name": "c1_hierarchical_delta",
                "variant": "timexer_hierarchical",
                "target_mode": "delta",
                "fusion": {
                    "kind": "c1_hierarchical",
                    "text_at_head": False,
                    "text_to_patches": False,
                    "text_as_exogenous": True,
                    "global_to_patch": True,
                },
                "fft_mode": "bridge",
                "d_model": 32,
                "n_heads": 2,
                "e_layers": 3,
                "d_ff": 64,
                "dropout": 0.0,
                "text_dim": 15,
                "n_features": 5,
                "n_ts_features": 25,
                "use_prototypes": False,
                "head": {"pool": "last", "type": "linear", "hidden": 64, "dropout": 0.0},
                "loss": {
                    "kind": "huber",
                    "delta": 0.5,
                    "lambda_c": 0.0,
                    "lambda_e": 0.0,
                    "lambda_d": 0.0,
                    "gamma_dir": 0.0,
                    "alpha_corr": 0.0,
                },
            },
            "train": {
                "ticker_set": "dev",
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 4,
                "num_workers": 0,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "patience": 2,
                "eval_every": 1,
                "max_train_windows": 16,
                "max_val_windows": 8,
                "max_test_windows": 8,
                "mlflow": {"enabled": False},
            },
            "explain": {},
        }
    )
    test_metrics = run_training(cfg)
    assert "delta_mse" in test_metrics
    assert "delta_mae" in test_metrics
    assert "delta_da" in test_metrics
    assert "test_mae_denorm" in test_metrics
    assert "test_da" in test_metrics
    assert "test_mse_price" in test_metrics

    # Check metrics.json
    metrics_file = tmp_path / "outputs" / "metrics.json"
    assert metrics_file.exists()
    saved = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved["target_mode"] == "delta"
    assert "delta_mse" in saved
    assert "test_mae_denorm" in saved

