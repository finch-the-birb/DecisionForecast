"""OHLCV TimeXer with compact text: shapes, text gradient, prototype switch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.data.dataset import build_datasets
from src.data.text_compact import COMPACT_COLUMNS
from src.models.prototypes import PrototypeLosses
from src.models.timexer_backbone import n_patches
from src.models.timexer_c1_compact import TimeXerC1Compact
from src.training.train import build_model

_FUSION = {
    "kind": "c1_compact",
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "global_to_patch": True,
}


def _model(**overrides) -> TimeXerC1Compact:
    kwargs = dict(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=32,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=15,
        fusion=_FUSION,
        use_prototypes=False,
        n_prototypes=10,
        d_min=0.5,
        head_pool="last",
        head_dropout=0.0,
    )
    kwargs.update(overrides)
    model = TimeXerC1Compact(**kwargs)
    model.eval()
    return model


def test_patch_embed_is_five_ohlcv_channels() -> None:
    model = _model()
    assert model.backbone.patch_embed.value.in_features == 5 * 12
    assert model.exo_embed.in_features == 15
    assert model.exo_embed.out_features == 32
    assert model.n_patches == n_patches(60, 12, 6) == 9
    assert len(model.global_to_patch) == 1
    assert model.head.pool == "last"
    assert not hasattr(model, "g12")


def test_text_gradient_reaches_the_loss_at_one_layer() -> None:
    torch.manual_seed(0)
    model = _model(e_layers=1)
    model.train()
    x = torch.randn(4, 60, 5)
    text = torch.randn(4, 60, 15, requires_grad=True)
    y = torch.randn(4, 7)
    output = model(x, text_seq=text)
    assert output.pred.shape == (4, 7)
    assert output.proto_losses is None
    loss, metrics = model.compute_loss(
        output,
        y,
        lambda_c=0.1,
        lambda_e=0.1,
        lambda_d=0.01,
        loss_kind="huber",
        huber_delta=0.5,
    )
    loss.backward()
    assert text.grad is not None
    assert float(text.grad.abs().sum()) > 0.0
    assert metrics["l_c"] == 0.0
    assert metrics["l_e"] == 0.0
    assert metrics["l_d"] == 0.0


def test_two_layers_each_bridge_text_into_patches() -> None:
    torch.manual_seed(1)
    model = _model(e_layers=2)
    model.train()
    assert len(model.backbone.layers) == 2
    assert len(model.global_to_patch) == 2
    x = torch.randn(2, 60, 5)
    text = torch.randn(2, 60, 15, requires_grad=True)
    model(x, text_seq=text).pred.sum().backward()
    assert text.grad is not None
    assert float(text.grad.abs().sum()) > 0.0
    for bridge in model.global_to_patch:
        grad_sum = sum(
            float(param.grad.abs().sum())
            for param in bridge.parameters()
            if param.grad is not None
        )
        assert grad_sum > 0.0


def test_prototypes_add_the_three_penalties() -> None:
    torch.manual_seed(2)
    model = _model(use_prototypes=True, n_prototypes=10)
    model.train()
    x = torch.randn(4, 60, 5)
    text = torch.randn(4, 60, 15)
    y = torch.randn(4, 7)
    output = model(x, text_seq=text)
    assert isinstance(output.proto_losses, PrototypeLosses)
    assert torch.isfinite(output.proto_losses.l_c)
    assert torch.isfinite(output.proto_losses.l_e)
    assert torch.isfinite(output.proto_losses.l_d)
    loss, metrics = model.compute_loss(
        output,
        y,
        lambda_c=0.1,
        lambda_e=0.1,
        lambda_d=0.01,
        loss_kind="huber",
        huber_delta=0.5,
    )
    assert metrics["l_c"] > 0.0
    assert loss.item() > metrics["l_pred"]
    assert hasattr(model, "g12")
    assert model.g12.proto.prototypes.shape == (10, 32)


def test_build_model_reads_depth_and_prototype_flags() -> None:
    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        plain = compose(
            config_name="config",
            overrides=["model=c1_compact", "data=fnspid_compact", "model.e_layers=1", "model.use_prototypes=false"],
        )
        deep = compose(
            config_name="config",
            overrides=["model=c1_compact", "data=fnspid_compact", "model.e_layers=2", "model.use_prototypes=true"],
        )
    assert list(plain.data.features) == ["close", "volume", "open", "high", "low"]
    assert plain.data.features_mode == "ohlcv_compact_text"
    assert int(plain.data.text.dim) == 15
    assert plain.model.loss.kind == "huber"
    assert float(plain.model.loss.delta) == 0.5
    without = build_model(plain)
    with_proto = build_model(deep)
    assert isinstance(without, TimeXerC1Compact)
    assert without.n_features == 5
    assert without.text_dim == 15
    assert without.use_prototypes is False
    assert len(without.backbone.layers) == 1
    assert without.backbone.patch_embed.value.in_features == 60
    assert with_proto.use_prototypes is True
    assert len(with_proto.global_to_patch) == 2
    assert with_proto.g12.proto.prototypes.shape == (10, 64)
    assert with_proto.head.pool == "last"


def test_compact_cache_aligns_to_the_ohlcv_calendar(tmp_path: Path) -> None:
    dates = pd.bdate_range("2021-01-04", periods=90)
    rng = np.random.default_rng(0)
    close = 50.0 + np.cumsum(rng.normal(0.0, 0.4, len(dates)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": rng.integers(1000, 2000, len(dates)),
        }
    )
    prices = tmp_path / "Stock_price" / "full_history" / "full_history"
    prices.mkdir(parents=True)
    frame.to_csv(prices / "AAA.csv", index=False)
    text = pd.DataFrame(rng.normal(size=(len(dates), 15)), columns=list(COMPACT_COLUMNS))
    text.insert(0, "date", dates)
    text["has_news"] = True
    dropped = dates[10]
    text = text.loc[text["date"] != dropped].reset_index(drop=True)
    cache = tmp_path / "cache" / "text_compact"
    cache.mkdir(parents=True)
    text.to_parquet(cache / "AAA.parquet", index=False)
    cfg = OmegaConf.create(
        {
            "data": {
                "root": str(tmp_path),
                "features_mode": "ohlcv_compact_text",
                "text_compact_cache_dir": str(cache),
                "features": ["close", "volume", "open", "high", "low"],
                "target": "close",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
                "normalize": "per_window",
                "news_source": "unused",
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
    train_ds, val_ds, test_ds = build_datasets(cfg)
    assert len(val_ds) == 0 and len(test_ds) == 0
    assert len(train_ds) > 0
    series = train_ds.series["AAA"]
    assert series.features.shape[1] == 5
    assert series.text_seq is not None
    assert series.text_seq.shape == (len(series.features), 15)
    missing_at = int(series.dates.searchsorted(pd.Timestamp(dropped)))
    assert np.all(series.text_seq[missing_at] == 0.0)
    assert bool(series.has_news[missing_at]) is False
    item = train_ds[0]
    assert tuple(item["x"].shape) == (60, 5)
    assert tuple(item["text_seq"].shape) == (60, 15)
    assert tuple(item["y"].shape) == (7,)
