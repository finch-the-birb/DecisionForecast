"""40D TimeXer: text reaches the last patch through G_en in one layer."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.timexer_backbone import n_patches
from src.models.timexer_selected import TimeXerSelected
from src.training.train import build_model

_FUSION = {
    "kind": "selected_40d",
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "global_to_patch": True,
}


def _model(**overrides) -> TimeXerSelected:
    kwargs = dict(
        n_features=26,
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
        head_pool="last",
        head_dropout=0.0,
    )
    kwargs.update(overrides)
    model = TimeXerSelected(**kwargs)
    model.eval()
    return model


def test_patch_count_and_head_read_the_last_patch() -> None:
    model = _model()
    assert model.n_patches == n_patches(60, 12, 6) == 9
    assert model.backbone.patch_embed.value.in_features == 26 * 12
    assert model.exo_embed.in_features == 15
    assert model.head.pool == "last"
    assert model.head.net[-1].in_features == 32
    assert model.head.net[-1].out_features == 7


def test_text_changes_the_forecast_with_one_layer() -> None:
    torch.manual_seed(0)
    model = _model()
    x = torch.randn(4, 60, 26)
    text_a = torch.randn(4, 60, 15)
    text_b = text_a + 3.0
    pred_a = model(x, text_seq=text_a).pred
    pred_b = model(x, text_seq=text_b).pred
    assert pred_a.shape == (4, 7)
    assert not torch.allclose(pred_a, pred_b)
    text_a.requires_grad_(True)
    model(x, text_seq=text_a).pred.sum().backward()
    assert text_a.grad is not None
    assert float(text_a.grad.abs().sum()) > 0.0


def test_pooled_text_vector_is_one_exogenous_token() -> None:
    torch.manual_seed(1)
    model = _model()
    x = torch.randn(2, 60, 26)
    text = torch.randn(2, 15)
    pred = model(x, text=text).pred
    assert pred.shape == (2, 7)
    other = model(x, text=text + 2.0).pred
    assert not torch.allclose(pred, other)


def test_build_model_reads_the_selected_config() -> None:
    features = ["close", *[f"f{i}" for i in range(25)]]
    cfg = OmegaConf.create(
        {
            "data": {
                "features": features,
                "target": "close",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
            },
            "model": {
                "name": "timexer_selected",
                "d_model": 64,
                "n_heads": 4,
                "e_layers": 1,
                "d_ff": 256,
                "dropout": 0.0,
                "text_dim": 15,
                "fusion": _FUSION,
                "head": {"type": "linear", "hidden": 128, "dropout": 0.0, "pool": "last"},
            },
        }
    )
    model = build_model(cfg)
    assert isinstance(model, TimeXerSelected)
    assert model.n_features == 26
    assert model.text_dim == 15
    assert len(model.backbone.layers) == 1
    assert len(model.global_to_patch) == 1


def test_two_layers_each_pass_updated_global_token_to_patches() -> None:
    torch.manual_seed(2)
    model = _model(e_layers=2)
    model.train()
    assert len(model.backbone.layers) == 2
    assert len(model.global_to_patch) == 2
    x = torch.randn(4, 60, 26)
    text = torch.randn(4, 60, 15, requires_grad=True)
    pred_a = model(x, text_seq=text).pred
    pred_b = model(x, text_seq=text.detach() + 3.0).pred
    assert pred_a.shape == (4, 7)
    assert not torch.allclose(pred_a, pred_b)
    pred_a.sum().backward()
    assert text.grad is not None
    assert float(text.grad.abs().sum()) > 0.0
    for bridge in model.global_to_patch:
        grad_sum = sum(
            float(param.grad.abs().sum())
            for param in bridge.parameters()
            if param.grad is not None
        )
        assert grad_sum > 0.0


def test_build_model_honors_e_layers_override() -> None:
    features = ["close", *[f"f{i}" for i in range(25)]]
    cfg = OmegaConf.create(
        {
            "data": {
                "features": features,
                "target": "close",
                "lookback_T": 60,
                "horizon": 7,
                "patch_len": 12,
                "patch_stride": 6,
            },
            "model": {
                "name": "timexer_selected",
                "d_model": 64,
                "n_heads": 4,
                "e_layers": 2,
                "d_ff": 256,
                "dropout": 0.0,
                "text_dim": 15,
                "fusion": _FUSION,
                "head": {"type": "linear", "hidden": 128, "dropout": 0.0, "pool": "last"},
            },
        }
    )
    model = build_model(cfg)
    assert len(model.backbone.layers) == 2
    assert len(model.global_to_patch) == 2
