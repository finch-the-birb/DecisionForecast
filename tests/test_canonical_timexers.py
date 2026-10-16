"""Unit tests for canonical TimeXerDual and TimeXerHierarchical (NeurIPS 2024 / TSLib)
including patch and bridge FFT modes.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.timexer_dual import TimeXerDual
from src.models.timexer_hierarchical import TimeXerHierarchical
from src.training.train import build_model


def _make_dual(
    n_features: int = 5,
    e_layers: int = 2,
    n_ts_features: int = 25,
    fft_mode: str = "patch",
) -> TimeXerDual:
    return TimeXerDual(
        n_features=n_features,
        seq_len=60,
        horizon=7,
        d_model=64,
        n_heads=4,
        e_layers=e_layers,
        patch_len=16,
        patch_stride=12,
        dropout=0.0,
        text_dim=15,
        n_ts_features=n_ts_features,
        close_idx=0,
        fft_mode=fft_mode,
    )


def _make_hierarchical(
    n_features: int = 5,
    n_ts_features: int = 25,
    fft_mode: str = "patch",
) -> TimeXerHierarchical:
    return TimeXerHierarchical(
        n_features=n_features,
        seq_len=60,
        horizon=7,
        d_model=64,
        n_heads=4,
        e_layers=3,
        patch_len=16,
        patch_stride=12,
        dropout=0.0,
        text_dim=15,
        n_ts_features=n_ts_features,
        close_idx=0,
        fft_mode=fft_mode,
    )


@pytest.mark.parametrize("fft_mode", ["patch", "bridge", "none"])
@pytest.mark.parametrize("n_features", [1, 2, 5])
def test_timexer_dual_shape(n_features: int, fft_mode: str) -> None:
    B, T, H = 4, 60, 7
    model = _make_dual(n_features=n_features, fft_mode=fft_mode)
    model.eval()

    x = torch.randn(B, T, n_features)
    text = torch.randn(B, 15)
    ts = torch.randn(B, T, 25)

    with torch.no_grad():
        out = model(x, text=text, ts=ts)

    assert out.pred.shape == (B, H)
    assert not torch.isnan(out.pred).any()


@pytest.mark.parametrize("fft_mode", ["patch", "bridge", "none"])
@pytest.mark.parametrize("n_features", [1, 2, 5])
def test_timexer_hierarchical_shape(n_features: int, fft_mode: str) -> None:
    B, T, H = 4, 60, 7
    model = _make_hierarchical(n_features=n_features, fft_mode=fft_mode)
    model.eval()

    x = torch.randn(B, T, n_features)
    text = torch.randn(B, 15)
    ts = torch.randn(B, T, 25)

    with torch.no_grad():
        out = model(x, text=text, ts=ts)

    assert out.pred.shape == (B, H)
    assert not torch.isnan(out.pred).any()


@pytest.mark.parametrize("fft_mode", ["patch", "bridge", "none"])
def test_timexer_dual_all_parameters_receive_gradients(fft_mode: str) -> None:
    torch.manual_seed(42)
    B, T, C_endo, C_ts = 4, 60, 5, 25
    model = _make_dual(n_features=C_endo, e_layers=2, n_ts_features=C_ts, fft_mode=fft_mode)
    model.train()

    x = torch.randn(B, T, C_endo, requires_grad=True)
    text = torch.randn(B, 15)
    ts = torch.randn(B, T, C_ts)

    out = model(x, text=text, ts=ts)
    target = torch.randn(B, 7)
    loss = nn.functional.mse_loss(out.pred, target)
    loss.backward()

    named_params = dict(model.named_parameters())
    assert len(named_params) > 0

    for name, param in named_params.items():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN in gradient"
            assert (param.grad.abs() > 0).any(), f"Parameter {name} has all-zero gradient"

    # Verify both global tokens receive gradients
    glb_grad = model.en_embedding.glb_tokens.grad
    assert glb_grad is not None
    assert (glb_grad[:, :, 0, :].abs() > 0).any(), "G_ts global token has zero gradient"
    assert (glb_grad[:, :, 1, :].abs() > 0).any(), "G_text global token has zero gradient"

    # Specific FFT checks
    if fft_mode == "patch":
        assert model.en_embedding.freq_proj is not None
        assert model.en_embedding.freq_proj.weight.grad is not None
        assert (model.en_embedding.freq_proj.weight.grad.abs() > 0).any()
    elif fft_mode == "bridge":
        assert model.fft_variate_proj is not None
        assert model.fft_variate_proj.weight.grad is not None
        assert (model.fft_variate_proj.weight.grad.abs() > 0).any()


@pytest.mark.parametrize("fft_mode", ["patch", "bridge", "none"])
def test_timexer_hierarchical_all_parameters_receive_gradients(fft_mode: str) -> None:
    torch.manual_seed(42)
    B, T, C_endo, C_ts = 4, 60, 5, 25
    model = _make_hierarchical(n_features=C_endo, n_ts_features=C_ts, fft_mode=fft_mode)
    model.train()

    x = torch.randn(B, T, C_endo, requires_grad=True)
    text = torch.randn(B, 15)
    ts = torch.randn(B, T, C_ts)

    out = model(x, text=text, ts=ts)
    target = torch.randn(B, 7)
    loss = nn.functional.mse_loss(out.pred, target)
    loss.backward()

    named_params = dict(model.named_parameters())
    assert len(named_params) > 0

    for name, param in named_params.items():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN in gradient"
            assert (param.grad.abs() > 0).any(), f"Parameter {name} has all-zero gradient"

    # Verify both global tokens receive gradients
    glb_grad = model.en_embedding.glb_tokens.grad
    assert glb_grad is not None
    assert (glb_grad[:, :, 0, :].abs() > 0).any(), "G_ts global token has zero gradient"
    assert (glb_grad[:, :, 1, :].abs() > 0).any(), "G_text global token has zero gradient"

    # Specific FFT checks
    if fft_mode == "patch":
        assert model.en_embedding.freq_proj is not None
        assert model.en_embedding.freq_proj.weight.grad is not None
        assert (model.en_embedding.freq_proj.weight.grad.abs() > 0).any()
    elif fft_mode == "bridge":
        assert model.fft_variate_proj is not None
        assert model.fft_variate_proj.weight.grad is not None
        assert (model.fft_variate_proj.weight.grad.abs() > 0).any()


def test_text_sequence_input_shape() -> None:
    B, T, C_endo, C_ts = 2, 60, 2, 25
    model = _make_dual(n_features=C_endo, n_ts_features=C_ts)
    model.eval()

    x = torch.randn(B, T, C_endo)
    text_seq = torch.randn(B, T, 15)
    ts = torch.randn(B, T, C_ts)

    with torch.no_grad():
        out = model(x, text_seq=text_seq, ts=ts)
    assert out.pred.shape == (B, 7)


def test_missing_ts_fallback() -> None:
    B, T, C_endo = 2, 60, 2
    model = _make_hierarchical(n_features=C_endo, n_ts_features=25)
    model.eval()

    x = torch.randn(B, T, C_endo)
    text = torch.randn(B, 15)

    with torch.no_grad():
        out = model(x, text=text, ts=None)
    assert out.pred.shape == (B, 7)


@pytest.mark.parametrize(
    "config_name, expected_cls, expected_fft_mode",
    [
        ("c1_dual_patch_fft", TimeXerDual, "patch"),
        ("c1_dual_bridge_fft", TimeXerDual, "bridge"),
        ("c1_hierarchical_patch_fft", TimeXerHierarchical, "patch"),
        ("c1_hierarchical_bridge_fft", TimeXerHierarchical, "bridge"),
    ],
)
def test_hydra_build_all_ablation_configs(
    config_name: str, expected_cls: type, expected_fft_mode: str
) -> None:
    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=[f"model={config_name}", "data=fnspid_dual_ts"])
    model = build_model(cfg)
    assert isinstance(model, expected_cls)
    assert model.fft_mode == expected_fft_mode
