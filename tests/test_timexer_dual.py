"""Dual-token TimeXer: depth, width, stride, finite grads, and masked tokens."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.data.collate import forecast_collate
from src.data.dataset import build_datasets
from src.explain.bank import collect_segment_bank
from src.data.text_compact import COMPACT_COLUMNS
from src.models.timexer_backbone import n_patches
from src.models.timexer_dual import TimeXerDual, dual_attention_mask
from src.training.train import build_model

_FUSION = {
    "kind": "c1_dual",
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "global_to_patch": True,
}


def _model(**overrides) -> TimeXerDual:
    kwargs = dict(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=64,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=15,
        fusion=_FUSION,
        n_ts_features=8,
        close_idx=0,
        use_prototypes=False,
        head_pool="last",
    )
    kwargs.update(overrides)
    return TimeXerDual(**kwargs)


@pytest.mark.parametrize("e_layers", [1, 2])
@pytest.mark.parametrize("d_model", [64, 128])
@pytest.mark.parametrize("patch_stride", [6, 12])
def test_forward_and_backward_stay_finite(e_layers: int, d_model: int, patch_stride: int) -> None:
    torch.manual_seed(e_layers + d_model + patch_stride)
    model = _model(e_layers=e_layers, d_model=d_model, patch_stride=patch_stride)
    model.train()
    expected_k = n_patches(60, 12, patch_stride)
    assert model.n_patches == expected_k
    assert model.patch_proj.in_features == 66
    assert model.patch_proj.out_features == d_model
    assert model.layers[0].ff[0].out_features == 4 * d_model
    assert float(model.layers[0].bridge.alpha_ts.detach()) == 0.0
    assert float(model.layers[0].bridge.alpha_text.detach()) == 0.0
    x = torch.randn(2, 60, 5)
    text = torch.randn(2, 60, 15)
    ts = torch.randn(2, 60, 8)
    y = torch.randn(2, 7)
    output = model(x, text_seq=text, ts=ts)
    assert output.pred.shape == (2, 7)
    assert output.proto_losses is None
    assert torch.isfinite(output.pred).all()
    loss, _metrics = model.compute_loss(
        output,
        y,
        lambda_c=0.0,
        lambda_e=0.0,
        lambda_d=0.0,
        loss_kind="huber",
        huber_delta=0.5,
    )
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_attention_mask_isolates_the_two_global_tokens() -> None:
    for n_patches_ in (9, 5):
        mask = dual_attention_mask(n_patches_, torch.device("cpu"), torch.float32)
        assert mask.shape == (n_patches_ + 2, n_patches_ + 2)
        ts_idx = n_patches_
        text_idx = n_patches_ + 1
        assert math.isinf(float(mask[ts_idx, text_idx])) and float(mask[ts_idx, text_idx]) < 0
        assert math.isinf(float(mask[text_idx, ts_idx])) and float(mask[text_idx, ts_idx]) < 0
        assert float(mask[ts_idx, ts_idx]) == 0.0
        assert float(mask[text_idx, text_idx]) == 0.0
        assert torch.isfinite(mask[:n_patches_]).all()
        probs = torch.softmax(torch.zeros_like(mask) + mask, dim=-1)
        assert float(probs[ts_idx, text_idx]) == 0.0
        assert float(probs[text_idx, ts_idx]) == 0.0
        assert torch.isfinite(probs).all()


def test_prototypes_switch_and_hydra_build() -> None:
    with_proto = _model(use_prototypes=True, d_model=32, n_heads=4)
    x = torch.randn(2, 60, 5)
    text = torch.randn(2, 60, 15)
    output = with_proto(x, text_seq=text, ts=torch.randn(2, 8))
    assert output.proto_losses is not None
    assert torch.isfinite(output.proto_losses.l_c)
    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=c1_dual",
                "data=fnspid_selective",
                "model.e_layers=2",
                "model.d_model=128",
                "data.patch_stride=12",
                "model.n_features=5",
                "model.n_ts_features=0",
            ],
        )
    assert cfg.data.normalize == "selective"
    assert int(cfg.data.patch_stride) == 12
    assert int(cfg.data.text.dim) == 15
    assert cfg.model.use_prototypes is False
    assert cfg.model.loss.kind == "huber"
    model = build_model(cfg)
    assert isinstance(model, TimeXerDual)
    assert model.n_patches == 5
    assert len(model.layers) == 2
    assert model.patch_proj.out_features == 128
    assert model.layers[0].ff[0].out_features == 512
    assert model.patch_proj.in_features == 66


@pytest.mark.parametrize("n_features", [1, 2, 5])
def test_endogenous_width_keeps_six_rfft_bins(n_features: int) -> None:
    torch.manual_seed(n_features)
    model = _model(n_features=n_features, n_ts_features=25, e_layers=2, patch_stride=12)
    assert model.patch_proj.in_features == n_features * 12 + 6
    assert model.g_ts is not None
    assert model.g_ts.in_features == 25
    assert model.g_ts.out_features == 64
    x = torch.randn(2, 60, n_features)
    text = torch.randn(2, 60, 15)
    ts = torch.randn(2, 60, 25, requires_grad=True)
    y = torch.randn(2, 7)
    output = model(x, text_seq=text, ts=ts)
    assert output.pred.shape == (2, 7)
    assert torch.isfinite(output.pred).all()
    loss, _metrics = model.compute_loss(
        output,
        y,
        lambda_c=0.0,
        lambda_e=0.0,
        lambda_d=0.0,
        loss_kind="huber",
        huber_delta=0.5,
    )
    loss.backward()
    assert ts.grad is not None
    assert torch.isfinite(ts.grad).all()
    assert float(ts.grad.abs().sum()) > 0.0
    assert model.g_ts.weight.grad is not None
    assert torch.isfinite(model.g_ts.weight.grad).all()
    assert float(model.g_ts.weight.grad.abs().sum()) > 0.0
    cross_grad = model.layers[0].cross_ts.in_proj_weight.grad
    assert cross_grad is not None
    assert torch.isfinite(cross_grad).all()
    assert float(cross_grad.abs().sum()) > 0.0
    freq_grad = model.patch_proj.weight.grad
    assert freq_grad is not None
    assert torch.isfinite(freq_grad[:, -6:]).all()
    assert float(freq_grad[:, -6:].abs().sum()) > 0.0


def test_rejected_channel_count_and_missing_ts() -> None:
    with pytest.raises(ValueError, match="1, 2, or 5"):
        _model(n_features=3)
    model = _model(n_features=1, n_ts_features=25)
    with pytest.raises(ValueError, match="requires ts"):
        model(torch.randn(2, 60, 1), text_seq=torch.randn(2, 60, 15))
    flat = model(
        torch.randn(2, 60, 1),
        text_seq=torch.randn(2, 60, 15),
        ts=torch.randn(2, 25),
    )
    assert flat.pred.shape == (2, 7)
    assert torch.isfinite(flat.pred).all()


def test_dual_ts_config_builds_f1_f2_and_f5() -> None:
    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        blocked = compose(
            config_name="config",
            overrides=[
                "model=c1_dual",
                "data=fnspid_dual_ts",
                "model.n_features=5",
                "model.n_ts_features=0",
            ],
        )
        mismatch = compose(
            config_name="config",
            overrides=[
                "model=c1_dual",
                "data=fnspid_dual_ts",
                "model.n_ts_features=25",
                "model.n_features=1",
            ],
        )
        f1 = compose(
            config_name="config",
            overrides=[
                "model=c1_dual",
                "data=fnspid_dual_ts",
                "model.n_ts_features=25",
                "model.n_features=1",
                "data.features=[close]",
            ],
        )
        f2 = compose(
            config_name="config",
            overrides=[
                "model=c1_dual",
                "data=fnspid_dual_ts",
                "model.n_ts_features=25",
                "model.n_features=2",
                "data.features=[close,volume]",
            ],
        )
        f5 = compose(
            config_name="config",
            overrides=[
                "model=c1_dual",
                "data=fnspid_dual_ts",
                "model.n_features=5",
            ],
        )
    assert blocked.data.features_mode == "dual_ts"
    assert blocked.data.normalize == "selective"
    assert str(blocked.data.signature_path).endswith("paper_fold1_signature.json")
    with pytest.raises(ValueError, match="n_ts_features"):
        build_model(blocked)
    with pytest.raises(ValueError, match="model.n_features"):
        build_model(mismatch)
    built = {
        1: build_model(f1),
        2: build_model(f2),
        5: build_model(f5),
    }
    assert built[1].patch_proj.in_features == 18
    assert built[2].patch_proj.in_features == 30
    assert built[5].patch_proj.in_features == 66
    for model in built.values():
        assert isinstance(model, TimeXerDual)
        assert model.g_ts is not None
        assert model.g_ts.in_features == 25
        assert model.close_idx == 0
    assert list(f1.data.features) == ["close"]
    assert list(f2.data.features) == ["close", "volume"]
    assert list(f5.data.features) == ["close", "volume", "open", "high", "low"]


def test_dual_ts_dataset_returns_normalized_indicator_windows(tmp_path: Path) -> None:
    dates = pd.bdate_range("2021-01-04", periods=90)
    rng = np.random.default_rng(0)
    close = 50.0 + np.cumsum(rng.normal(0.0, 0.4, len(dates)))
    prices = tmp_path / "Stock_price" / "full_history" / "full_history"
    prices.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": rng.integers(1000, 2000, len(dates)),
        }
    ).to_csv(prices / "AAA.csv", index=False)
    text = pd.DataFrame(rng.normal(size=(len(dates), 15)), columns=list(COMPACT_COLUMNS))
    text.insert(0, "date", dates)
    text["has_news"] = True
    text_dir = tmp_path / "cache" / "text_compact"
    text_dir.mkdir(parents=True)
    text.to_parquet(text_dir / "AAA.parquet", index=False)
    indicator_names = ["rsi_n14", "bb_pctb_n20", "parkinson_n5", *[f"log_ret_{i}" for i in range(1, 23)]]
    raw = rng.normal(size=(len(dates), 25))
    raw[:, 0] = 80.0
    raw[:, 1] = 0.25
    raw[:, 2] = 3.0
    technical = pd.DataFrame(raw, columns=indicator_names)
    technical.insert(0, "date", dates)
    technical = technical.iloc[5:].reset_index(drop=True)
    technical.loc[0, "parkinson_n5"] = np.nan
    technical_dir = tmp_path / "cache" / "technical"
    technical_dir.mkdir(parents=True)
    technical.to_parquet(technical_dir / "AAA.parquet", index=False)
    signature = {
        "format": "selected_40d_v1",
        "horizon": 7,
        "lookback": 60,
        "ts_columns": ["close", *indicator_names],
        "text_columns": list(COMPACT_COLUMNS),
    }
    signature_path = tmp_path / "cache" / "selected_signatures" / "paper_fold1_signature.json"
    signature_path.parent.mkdir(parents=True)
    signature_path.write_text(json.dumps(signature), encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(tmp_path),
                "features_mode": "dual_ts",
                "features": ["close", "volume"],
                "target": "close",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 12,
                "normalize": "selective",
                "news_source": "unused",
                "signature_path": str(signature_path),
                "technical_cache_dir": str(technical_dir),
                "text_compact_cache_dir": str(text_dir),
                "split": {
                    "train_start": "2021-01-01",
                    "train_end": "2021-12-31",
                    "val_end": "2022-12-31",
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
            "train": {
                "ticker_set": "dev",
                "max_train_windows": None,
                "max_val_windows": None,
                "max_test_windows": None,
            },
        }
    )
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    bad.data.features = ["close", "open"]
    with pytest.raises(ValueError, match="dual_ts features"):
        build_datasets(bad)
    train_ds, val_ds, test_ds = build_datasets(cfg)
    assert len(val_ds) == 0 and len(test_ds) == 0
    assert len(train_ds) > 0
    item = train_ds[0]
    assert tuple(item["x"].shape) == (60, 2)
    assert tuple(item["ts"].shape) == (60, 25)
    assert torch.isfinite(item["x"]).all()
    assert torch.isfinite(item["ts"]).all()
    np.testing.assert_allclose(item["ts"][:, 0].numpy(), 0.6, atol=1e-5)
    np.testing.assert_allclose(item["ts"][:, 1].numpy(), -0.5, atol=1e-5)
    np.testing.assert_allclose(item["ts"][:, 2].numpy(), 0.0, atol=1e-5)
    batch = forecast_collate([train_ds[0], train_ds[1]])
    assert tuple(batch["ts"].shape) == (2, 60, 25)
    model = _model(n_features=2, n_ts_features=25, e_layers=2, patch_stride=12, dropout=0.0)
    output = model(batch["x"], batch["text"], text_seq=batch["text_seq"], ts=batch["ts"])
    assert output.pred.shape == (2, 7)
    assert torch.isfinite(output.pred).all()


def test_dual_ts_collect_segment_bank_and_kmeans_init() -> None:
    torch.manual_seed(0)
    d_model = 64
    model = _model(
        n_features=2,
        n_ts_features=25,
        use_prototypes=True,
        d_model=d_model,
        dropout=0.0,
    )
    items = []
    for end_idx in range(4):
        items.append(
            {
                "x": torch.randn(60, 2),
                "y": torch.randn(7),
                "text": torch.randn(15),
                "text_seq": torch.randn(60, 15),
                "ts": torch.randn(60, 25),
                "has_news_frac": torch.tensor(1.0),
                "y_mean": torch.tensor(0.0),
                "y_std": torch.tensor(1.0),
                "ticker": "AAA",
                "end_idx": end_idx,
                "end_date": "2021-01-04",
            }
        )
    loader = DataLoader(items, batch_size=2, collate_fn=forecast_collate)
    bank, _meta = collect_segment_bank(model, loader)
    assert bank.ndim == 2
    assert bank.size(0) >= model.proto.prototypes.size(0)
    assert bank.size(1) == d_model
    assert torch.isfinite(bank).all()
    model.proto.init_from_bank(bank)
    assert model.proto.prototypes.shape == (10, d_model)
    assert torch.isfinite(model.proto.prototypes).all()
