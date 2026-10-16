"""Tests for learnable tokens in TimeXer variants and 26-channel DLinear."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.models.dlinear import DLinear
from src.models.timexer_dual import TimeXerDual
from src.models.timexer_hierarchical import TimeXerHierarchical
from src.models.timexer_inverted import TimeXerInverted
from src.training.train import build_model


_CFG = str(Path(__file__).resolve().parents[1] / "configs")


def _compose(*overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_CFG):
        return compose(config_name="config", overrides=list(overrides))


def test_c1_dual_tokens_are_parameters_and_receive_nonzero_gradients() -> None:
    torch.manual_seed(42)
    model = TimeXerDual(
        n_features=2,
        seq_len=60,
        horizon=7,
        d_model=64,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=12,
        dropout=0.0,
        text_dim=15,
        fusion={
            "kind": "c1_dual",
            "text_at_head": False,
            "text_to_patches": False,
            "text_as_exogenous": True,
            "global_to_patch": True,
        },
        n_ts_features=25,
        close_idx=0,
    )
    assert isinstance(model.text_token, nn.Parameter)
    assert isinstance(model.ts_token, nn.Parameter)
    assert model.text_token.shape == (1, 1, 64)
    assert model.ts_token.shape == (1, 1, 64)

    model.train()
    x = torch.randn(2, 60, 2)
    text = torch.randn(2, 60, 15)
    ts = torch.randn(2, 60, 25)
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

    assert model.text_token.grad is not None
    assert torch.isfinite(model.text_token.grad).all()
    assert float(model.text_token.grad.abs().sum()) > 0.0

    assert model.ts_token.grad is not None
    assert torch.isfinite(model.ts_token.grad).all()
    assert float(model.ts_token.grad.abs().sum()) > 0.0


def test_c1_hierarchical_tokens_are_parameters_and_receive_nonzero_gradients() -> None:
    torch.manual_seed(43)
    model = TimeXerHierarchical(
        n_features=2,
        seq_len=60,
        horizon=7,
        d_model=64,
        n_heads=4,
        e_layers=2,
        patch_len=12,
        patch_stride=12,
        dropout=0.0,
        text_dim=15,
        fusion={
            "kind": "c1_hierarchical",
            "text_at_head": False,
            "text_to_patches": False,
            "text_as_exogenous": True,
            "global_to_patch": True,
        },
        n_ts_features=25,
        close_idx=0,
        d_ff=256,
    )
    assert isinstance(model.text_token, nn.Parameter)
    assert isinstance(model.ts_token, nn.Parameter)
    assert model.text_token.shape == (1, 1, 64)
    assert model.ts_token.shape == (1, 1, 64)

    model.train()
    x = torch.randn(2, 60, 2)
    text = torch.randn(2, 60, 15)
    ts = torch.randn(2, 60, 25)
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

    assert model.text_token.grad is not None
    assert torch.isfinite(model.text_token.grad).all()
    assert float(model.text_token.grad.abs().sum()) > 0.0

    assert model.ts_token.grad is not None
    assert torch.isfinite(model.ts_token.grad).all()
    assert float(model.ts_token.grad.abs().sum()) > 0.0


def test_c1_inverted_text_token_is_parameter_and_receives_nonzero_gradient() -> None:
    torch.manual_seed(44)
    model = TimeXerInverted(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=64,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=12,
        dropout=0.0,
        text_dim=15,
        fusion={
            "kind": "c1_inverted",
            "text_at_head": False,
            "text_to_patches": False,
            "text_as_exogenous": True,
            "global_to_patch": True,
        },
        n_ts_features=25,
        close_idx=0,
    )
    assert isinstance(model.text_token, nn.Parameter)
    assert model.text_token.shape == (1, 1, 64)

    model.train()
    x = torch.randn(2, 60, 5)
    text = torch.randn(2, 60, 15)
    ts = torch.randn(2, 60, 25)
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

    assert model.text_token.grad is not None
    assert torch.isfinite(model.text_token.grad).all()
    assert float(model.text_token.grad.abs().sum()) > 0.0


def test_dlinear_on_dual_ts_26_channels() -> None:
    cfg = _compose(
        "model=dlinear",
        "data=fnspid_dual_ts",
        "data.features=[close]",
    )
    model = build_model(cfg)
    assert isinstance(model, DLinear)
    assert model.n_features == 26
    assert model.target_idx == 0

    x = torch.randn(2, 60, 1)
    ts = torch.randn(2, 60, 25)
    y = torch.randn(2, 7)

    out = model(x, ts=ts)
    assert out.pred.shape == (2, 7)
    assert torch.isfinite(out.pred).all()

    loss, _ = model.compute_loss(
        out, y, lambda_c=0.0, lambda_e=0.0, lambda_d=0.0, loss_kind="huber", huber_delta=0.5
    )
    loss.backward()
    assert model.linear_seasonal.weight.grad is not None
    assert float(model.linear_seasonal.weight.grad.abs().sum()) > 0.0
    assert model.linear_trend.weight.grad is not None
    assert float(model.linear_trend.weight.grad.abs().sum()) > 0.0


def test_dlinear_on_fnspid_selected_26_channels() -> None:
    cfg = _compose(
        "model=dlinear",
        "data=fnspid_selected",
    )
    model = build_model(cfg)
    assert isinstance(model, DLinear)
    assert model.n_features == 26
    assert model.target_idx == 0

    x = torch.randn(2, 60, 26)
    y = torch.randn(2, 7)

    out = model(x)
    assert out.pred.shape == (2, 7)
    assert torch.isfinite(out.pred).all()

    loss, _ = model.compute_loss(
        out, y, lambda_c=0.0, lambda_e=0.0, lambda_d=0.0, loss_kind="huber", huber_delta=0.5
    )
    loss.backward()
    assert model.linear_seasonal.weight.grad is not None
    assert float(model.linear_seasonal.weight.grad.abs().sum()) > 0.0
    assert model.linear_trend.weight.grad is not None
    assert float(model.linear_trend.weight.grad.abs().sum()) > 0.0
