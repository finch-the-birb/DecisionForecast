"""Canonical Hierarchical TimeXer tests (3-layer cross-modal hierarchy)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.timexer_hierarchical import TimeXerHierarchical
from src.training.train import build_model

_FUSION = {
    "kind": "c1_hierarchical",
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "global_to_patch": True,
}


def _model(**overrides) -> TimeXerHierarchical:
    kwargs = dict(
        n_features=2,
        seq_len=60,
        horizon=7,
        d_model=64,
        n_heads=4,
        e_layers=3,
        patch_len=12,
        patch_stride=12,
        dropout=0.0,
        text_dim=15,
        fusion=_FUSION,
        n_ts_features=25,
        close_idx=0,
        use_prototypes=False,
        d_ff=256,
        head_pool="last",
        fft_mode="patch",
    )
    kwargs.update(overrides)
    return TimeXerHierarchical(**kwargs)


def _batch(n_features: int = 2):
    x = torch.randn(2, 60, n_features)
    text = torch.randn(2, 60, 15)
    ts = torch.randn(2, 60, 25)
    y = torch.randn(2, 7)
    return x, text, ts, y


def test_depth_must_be_exactly_three_layers() -> None:
    for depth in (1, 2, 4):
        with pytest.raises(ValueError, match="exactly 3 layers"):
            _model(e_layers=depth)


def test_forward_shapes_and_finite_outputs() -> None:
    torch.manual_seed(0)
    model = _model()
    x, text, ts, _y = _batch()
    output = model(x, text_seq=text, ts=ts)
    assert output.pred.shape == (2, 7)
    assert torch.isfinite(output.pred).all()
    assert output.proto_losses is None


def test_exogenous_cross_attention_receives_gradients() -> None:
    torch.manual_seed(1)
    model = _model()
    model.train()
    x, text, ts, y = _batch()
    text.requires_grad_()
    ts.requires_grad_()
    output = model(x, text_seq=text, ts=ts)
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
    assert ts.grad is not None and float(ts.grad.abs().sum()) > 0.0
    assert text.grad is not None and float(text.grad.abs().sum()) > 0.0

    # Layer 1 has cross_ts
    assert model.layer1.cross_ts.in_proj_weight.grad is not None
    assert torch.isfinite(model.layer1.cross_ts.in_proj_weight.grad).all()
    assert float(model.layer1.cross_ts.in_proj_weight.grad.abs().sum()) > 0.0

    # Layer 2 has cross_text
    assert model.layer2.cross_text.in_proj_weight.grad is not None
    assert torch.isfinite(model.layer2.cross_text.in_proj_weight.grad).all()
    assert float(model.layer2.cross_text.in_proj_weight.grad.abs().sum()) > 0.0


def test_prototypes_switch_and_hydra_build() -> None:
    with_proto = _model(use_prototypes=True)
    x, text, ts, _y = _batch()
    output = with_proto(x, text_seq=text, ts=ts)
    assert output.proto_losses is not None
    assert torch.isfinite(output.proto_losses.l_c)

    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=c1_hierarchical",
                "data=fnspid_dual_ts",
                "data.features=[close,volume]",
                "model.n_features=2",
                "model.e_layers=3",
            ],
        )
    assert cfg.model.name == "c1_hierarchical"
    assert int(cfg.model.e_layers) == 3
    assert int(cfg.model.n_ts_features) == 25
    model = build_model(cfg)
    assert isinstance(model, TimeXerHierarchical)
