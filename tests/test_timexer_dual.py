"""Dual-token TimeXer: depth, width, stride, finite grads, and masked tokens."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

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
