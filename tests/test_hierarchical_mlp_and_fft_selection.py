"""Unit tests for TimeXerHierarchicalMLP, causal FFT features, and delta return penalties."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.training.train import build_model, run_training
from src.data.selective_norm import feature_role
from src.data.text_compact import COMPACT_COLUMNS
from src.features.selection import TOP_K, FoldFeatureSignature, select_fold_features
from src.features.technical import (
    _block_fft,
    base_feature_names,
    compute_technical_features,
    technical_feature_names,
)
from src.models.losses import compute_task_loss
from src.models.outputs import ModelOutput
from src.models.timexer_hierarchical_mlp import TimeXerHierarchicalMLP


def test_timexer_hierarchical_mlp_initialization_and_forward() -> None:
    torch.manual_seed(42)
    model = TimeXerHierarchicalMLP(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=64,
        n_heads=4,
        e_layers=3,
        patch_len=12,
        patch_stride=6,
        dropout=0.1,
        text_dim=15,
        n_ts_features=30,
        close_idx=0,
        head_type="mlp",
        head_hidden=128,
        head_dropout=0.1,
        fft_mode="bridge",
    )

    # Verify head architecture: Flatten -> Linear -> GELU -> Dropout -> Linear
    assert isinstance(model.head, nn.Sequential)
    assert len(model.head) == 5
    assert isinstance(model.head[0], nn.Flatten)
    assert isinstance(model.head[1], nn.Linear)
    assert model.head[1].in_features == model.head_nf
    assert model.head[1].out_features == 128
    assert isinstance(model.head[2], nn.GELU)
    assert isinstance(model.head[3], nn.Dropout)
    assert model.head[3].p == 0.1
    assert isinstance(model.head[4], nn.Linear)
    assert model.head[4].in_features == 128
    assert model.head[4].out_features == 7

    assert model.n_ts_features == 30
    assert model.fft_mode == "bridge"

    # Forward pass
    x = torch.randn(3, 60, 5)
    text = torch.randn(3, 15)
    ts = torch.randn(3, 60, 30)

    out = model(x, text=text, ts=ts)
    assert isinstance(out, ModelOutput)
    assert out.pred.shape == (3, 7)
    assert torch.isfinite(out.pred).all()


def test_causal_fft_block_properties() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    close = pd.Series(100.0 + np.cumsum(rng.normal(size=120)), index=dates)

    # 1. Warm-up and output structure
    fft_cols = _block_fft(close, lookback=60)
    assert len(fft_cols) == 10
    assert list(fft_cols.keys()) == [f"fft_harm_{k}" for k in range(1, 11)]

    # Warm-up rows < 59 must be NaN
    for k in range(1, 11):
        col = fft_cols[f"fft_harm_{k}"]
        assert col.iloc[:59].isna().all(), f"Row < 59 in fft_harm_{k} must be NaN"
        assert not col.iloc[59:].isna().any(), f"Row >= 59 in fft_harm_{k} must be finite"
        # Log-amplitudes ln(1 + amp) must be non-negative
        assert (col.iloc[59:] >= 0.0).all()

    # 2. Strict causality: perturbing future rows must NOT alter past rows
    close_modified = close.copy()
    close_modified.iloc[80:] += 50.0  # Big future jump at bar 80
    fft_modified = _block_fft(close_modified, lookback=60)

    for k in range(1, 11):
        orig_past = fft_cols[f"fft_harm_{k}"].iloc[:80].to_numpy()
        mod_past = fft_modified[f"fft_harm_{k}"].iloc[:80].to_numpy()
        np.testing.assert_allclose(
            orig_past[59:],
            mod_past[59:],
            rtol=1e-12,
            err_msg=f"Causality leak in fft_harm_{k}: past rows changed after future edit!",
        )

    # 3. Inventory integration: base and expanded feature names
    base_names = base_feature_names()
    for k in range(1, 11):
        assert f"fft_harm_{k}" in base_names

    all_names = technical_feature_names()
    for k in range(1, 11):
        assert f"fft_harm_{k}" in all_names
        assert f"fft_harm_{k}_lag1" in all_names
        assert f"fft_harm_{k}_delta" in all_names

    # 4. Selective normalization role
    for k in range(1, 11):
        assert feature_role(f"fft_harm_{k}") == "stationary"
        assert feature_role(f"fft_harm_{k}_lag2") == "stationary"
        assert feature_role(f"fft_harm_{k}_delta") == "stationary"


def test_gradient_flow_under_high_penalties() -> None:
    torch.manual_seed(42)
    model = TimeXerHierarchicalMLP(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=32,
        n_heads=2,
        e_layers=3,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=15,
        n_ts_features=30,
        close_idx=0,
        head_type="mlp",
        head_hidden=64,
        head_dropout=0.1,
        fft_mode="bridge",
    )

    x = torch.randn(4, 60, 5, requires_grad=True)
    text = torch.randn(4, 15)
    ts = torch.randn(4, 60, 30)
    target = torch.randn(4, 7)

    out = model(x, text=text, ts=ts)
    pred = out.pred

    # High penalties: gamma_dir=0.6, alpha_corr=0.5
    loss, metrics = compute_task_loss(
        pred, target, loss_kind="huber", huber_delta=0.5, gamma_dir=0.6, alpha_corr=0.5
    )
    assert "l_dir" in metrics
    assert "l_corr" in metrics
    assert metrics["l_dir"] >= 0.0
    assert metrics["l_corr"] >= 0.0

    loss.backward()

    # Check MLP head parameters receive finite non-zero gradients
    for name, param in model.head.named_parameters():
        assert param.grad is not None, f"Gradient missing for head param {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in head param {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient in head param {name}"

    # Check backbone and global tokens receive gradients
    assert model.en_embedding.glb_tokens.grad is not None
    assert model.en_embedding.glb_tokens.grad.abs().sum() > 0
    assert model.ts_proj.weight.grad is not None
    assert model.ts_proj.weight.grad.abs().sum() > 0


def test_hydra_build_model_all_5_delta_configs() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    config_names = [
        ("c1_hierarchical_mlp_delta_g02_a03", 0.2, 0.3),
        ("c1_hierarchical_mlp_delta_g04_a03", 0.4, 0.3),
        ("c1_hierarchical_mlp_delta_g04_a05", 0.4, 0.5),
        ("c1_hierarchical_mlp_delta_g06_a03", 0.6, 0.3),
        ("c1_hierarchical_mlp_delta_g06_a05", 0.6, 0.5),
    ]

    for model_name, expected_g, expected_a in config_names:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[f"model={model_name}", "data=fnspid_dual_ts"],
            )
            assert cfg.model.target_mode == "delta"
            assert cfg.model.loss.gamma_dir == expected_g
            assert cfg.model.loss.alpha_corr == expected_a
            assert cfg.model.n_ts_features == 30
            assert cfg.model.fft_mode == "bridge"
            assert cfg.model.head.type == "mlp"

            model = build_model(cfg)
            assert isinstance(model, TimeXerHierarchicalMLP)
            assert model.n_ts_features == 30
            assert isinstance(model.head, nn.Sequential)
            assert len(model.head) == 5


def test_mini_smoke_run_hierarchical_mlp_delta(tmp_path: Path) -> None:
    # Build synthetic prices and technical cache with 30 features
    tech_dir = tmp_path / "cache" / "technical"
    tech_dir.mkdir(parents=True)
    text_dir = tmp_path / "cache" / "text_compact"
    text_dir.mkdir(parents=True)
    sig_dir = tmp_path / "cache" / "selected_signatures"
    sig_dir.mkdir(parents=True)

    dates = pd.date_range("2021-01-01", periods=180, freq="B")
    n_days = len(dates)
    rng = np.random.default_rng(123)

    price_df = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0 + np.cumsum(rng.normal(size=n_days)),
            "high": 105.0 + np.cumsum(rng.normal(size=n_days)),
            "low": 95.0 + np.cumsum(rng.normal(size=n_days)),
            "close": 100.0 + np.cumsum(rng.normal(size=n_days)),
            "volume": 1000.0 + np.abs(rng.normal(size=n_days)) * 100.0,
        }
    )
    prices_dir = tmp_path / "Stock_price" / "full_history" / "full_history"
    prices_dir.mkdir(parents=True)
    price_df.to_csv(prices_dir / "AAA.csv", index=False)

    tech_df = price_df.copy()
    ts_cols = [f"feat_{i:02d}" for i in range(30)]
    for col in ts_cols:
        tech_df[col] = rng.normal(size=n_days)
    tech_df.to_parquet(tech_dir / "AAA.parquet", index=False)

    text_cols = list(COMPACT_COLUMNS)
    text_df = pd.DataFrame({"date": dates, "has_news": True})
    for col in text_cols:
        text_df[col] = rng.normal(size=n_days)
    text_df.to_parquet(text_dir / "AAA.parquet", index=False)

    sig_payload = {
        "format": "selected_40d_v1",
        "horizon": 7,
        "lookback": 60,
        "ts_columns": ["close", *ts_cols],
        "text_columns": text_cols,
    }
    sig_path = sig_dir / "paper_fold1_signature_30ts.json"
    sig_path.write_text(json.dumps(sig_payload), encoding="utf-8")

    from omegaconf import OmegaConf
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "smoke_mlp_delta"},
            "paths": {
                "output_dir": str(tmp_path / "outputs"),
                "checkpoint_dir": str(tmp_path / "outputs" / "checkpoints"),
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
                "name": "c1_hierarchical_mlp_delta",
                "variant": "timexer_hierarchical_mlp",
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
                "n_ts_features": 30,
                "use_prototypes": False,
                "head": {"type": "mlp", "hidden": 64, "dropout": 0.1},
                "loss": {
                    "kind": "huber",
                    "delta": 0.5,
                    "lambda_c": 0.0,
                    "lambda_e": 0.0,
                    "lambda_d": 0.0,
                    "gamma_dir": 0.4,
                    "alpha_corr": 0.3,
                },
            },
            "explain": {},
        }
    )
    test_metrics = run_training(cfg)
    assert "delta_mse" in test_metrics
    assert "delta_mae" in test_metrics
    assert "delta_da" in test_metrics
    assert "test_mse_price" in test_metrics
    assert "test_mae_denorm" in test_metrics
    assert "test_da" in test_metrics

    metrics_file = tmp_path / "outputs" / "metrics.json"
    assert metrics_file.exists()
    saved = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved["target_mode"] == "delta"
    assert "delta_mse" in saved
    assert "test_mse_price" in saved
