"""Four TS-integration architectures: shipped dual plus factored, inverted, late."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.timexer_dual import TimeXerDual, dual_attention_mask
from src.models.timexer_factored import TimeXerFactored
from src.models.timexer_inverted import TimeXerInverted
from src.models.timexer_late_fusion import TimeXerLateFusion
from src.training.train import build_model

_FLAGS = {
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "global_to_patch": True,
}


def _fusion(kind: str) -> dict:
    return {"kind": kind, **_FLAGS}


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(2, 60, 5), torch.randn(2, 60, 15), torch.randn(2, 60, 25)


def _dual(**overrides) -> TimeXerDual:
    kwargs = dict(
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
        fusion=_fusion("c1_dual"),
        n_ts_features=25,
        close_idx=0,
        use_prototypes=False,
        head_pool="last",
    )
    kwargs.update(overrides)
    return TimeXerDual(**kwargs)


def _factored(**overrides) -> TimeXerFactored:
    kwargs = dict(
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
        fusion=_fusion("c1_factored"),
        n_ts_features=25,
        n_patch_indicators=8,
        close_idx=0,
        use_prototypes=False,
        head_pool="last",
    )
    kwargs.update(overrides)
    return TimeXerFactored(**kwargs)


def _inverted(**overrides) -> TimeXerInverted:
    kwargs = dict(
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
        fusion=_fusion("c1_inverted"),
        n_ts_features=25,
        close_idx=0,
        use_prototypes=False,
        head_pool="last",
    )
    kwargs.update(overrides)
    return TimeXerInverted(**kwargs)


def _late(**overrides) -> TimeXerLateFusion:
    kwargs = dict(
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
        fusion=_fusion("c1_late_fusion"),
        n_ts_features=25,
        close_idx=0,
        use_prototypes=False,
        head_pool="last",
    )
    kwargs.update(overrides)
    return TimeXerLateFusion(**kwargs)


def _compose(overrides: list[str]):
    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize("e_layers", [1, 2])
@pytest.mark.parametrize(
    ("builder", "segments"),
    [
        (_dual, (2, 5, 64)),
        (_factored, (2, 5, 64)),
        (_inverted, (2, 5, 64)),
        (_late, (2, 1, 64)),
    ],
)
def test_each_architecture_forwards_at_both_depths(builder, segments, e_layers: int) -> None:
    model = builder(e_layers=e_layers).eval()
    x, text, ts = _inputs()
    output = model(x, text_seq=text, ts=ts)
    assert output.pred.shape == (2, 7)
    assert torch.isfinite(output.pred).all()
    assert output.proto_losses is None
    assert output.segments.shape == segments
    assert not hasattr(model, "proto")
    loss, _metrics = model.compute_loss(
        output, torch.randn(2, 7), 0.0, 0.0, 0.0, loss_kind="huber", huber_delta=0.5
    )
    assert torch.isfinite(loss)


def test_shipped_dual_keeps_the_f5_mask_and_zero_gates() -> None:
    model = _dual()
    assert model.patch_proj.in_features == 66
    assert model.g_ts is not None and model.g_ts.in_features == 25
    assert float(model.layers[0].bridge.alpha_ts.detach()) == 0.0
    assert float(model.layers[0].bridge.alpha_text.detach()) == 0.0
    mask = dual_attention_mask(model.n_patches, torch.device("cpu"), torch.float32)
    assert math.isinf(float(mask[-1, -2])) and float(mask[-1, -2]) < 0
    assert math.isinf(float(mask[-2, -1])) and float(mask[-2, -1]) < 0
    assert float(mask[0, -1]) == 0.0


def test_shipped_dual_config_leaves_indicators_off() -> None:
    cfg = _compose(["model=c1_dual", "data=fnspid_selective"])
    model = build_model(cfg)
    assert isinstance(model, TimeXerDual)
    assert model.n_features == 5
    assert model.n_ts_features == 0
    assert model.g_ts is None
    assert model.patch_proj.in_features == 66


def test_factored_projections_and_top_eight_indicators() -> None:
    model = _factored().eval()
    assert model.ohlcv_proj.in_features == 60 and model.ohlcv_proj.out_features == 36
    assert model.ind_proj.in_features == 96 and model.ind_proj.out_features == 16
    assert model.fft_proj.in_features == 6 and model.fft_proj.out_features == 12
    x, text, ts = _inputs()
    base = model(x, text_seq=text, ts=ts).pred
    ignored = ts.clone()
    ignored[:, :, 8:] += 3
    torch.testing.assert_close(base, model(x, text_seq=text, ts=ignored).pred)
    used = ts.clone()
    used[:, :, :8] += 3
    assert not torch.allclose(base, model(x, text_seq=text, ts=used).pred)


def test_factored_rejects_a_wider_token() -> None:
    with pytest.raises(ValueError, match="d_model 64"):
        _factored(d_model=128)


def test_inverted_variate_tokens_read_the_full_window() -> None:
    model = _inverted().eval()
    assert model.patch_proj.in_features == 66
    assert model.variate_proj.in_features == 60
    assert model.variate_proj.out_features == 64
    x, text, ts = _inputs()
    text = text.detach().requires_grad_(True)
    ts = ts.detach().requires_grad_(True)
    model(x, text_seq=text, ts=ts).pred.sum().backward()
    assert text.grad is not None and float(text.grad.abs().sum()) > 0
    assert ts.grad is not None and float(ts.grad.abs().sum()) > 0


def test_late_fusion_adds_only_the_last_indicator_bar() -> None:
    model = _late().eval()
    assert model.patch_proj.in_features == 66
    assert model.tech_mlp[0].in_features == 25
    assert model.tech_mlp[0].out_features == 64
    assert model.tech_mlp[3].out_features == 64
    x, text, ts = _inputs()
    base = model(x, text_seq=text, ts=ts).pred
    earlier = ts.clone()
    earlier[:, :-1, :] += 3
    torch.testing.assert_close(base, model(x, text_seq=text, ts=earlier).pred)
    last = ts.clone()
    last[:, -1, :] += 3
    assert not torch.allclose(base, model(x, text_seq=text, ts=last).pred)


@pytest.mark.parametrize("builder", [_factored, _inverted, _late, _dual])
def test_prototypes_attach_without_changing_the_preinjection_bank(builder) -> None:
    model = builder(use_prototypes=True, n_prototypes=10).eval()
    x, text, ts = _inputs()
    output = model(x, text_seq=text, ts=ts)
    assert output.proto_losses is not None
    assert torch.isfinite(output.proto_losses.l_c)
    assert torch.isfinite(output.pred).all()
    assert output.segments.shape[0] == 2
    assert output.segments.shape[-1] == 64
    assert hasattr(model, "proto")


def test_new_models_require_the_indicator_tensor() -> None:
    x, text, _ts = _inputs()
    for model in (_factored(), _inverted(), _late()):
        with pytest.raises(ValueError, match="requires ts"):
            model(x, text_seq=text, ts=None)


def test_hydra_builds_the_three_new_models() -> None:
    factored = build_model(_compose(["model=c1_factored", "data=fnspid_selective", "model.e_layers=2"]))
    assert isinstance(factored, TimeXerFactored)
    assert factored.n_patches == 9
    assert factored.layers[0].conv1.out_channels == 256
    assert factored.ohlcv_proj.out_features == 36

    inverted = build_model(_compose(["model=c1_inverted", "data=fnspid_selective", "model.d_model=128"]))
    assert isinstance(inverted, TimeXerInverted)
    assert inverted.variate_proj.in_features == 60
    assert inverted.variate_proj.out_features == 128
    assert inverted.layers[0].ff[0].out_features == 512

    late = build_model(_compose(["model=c1_late_fusion", "data=fnspid_selective", "model.d_ff=200"]))
    assert isinstance(late, TimeXerLateFusion)
    assert late.layers[0].conv1.out_channels == 200
    assert late.tech_mlp[0].out_features == 64
    assert late.head.pool == "last"
