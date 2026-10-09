"""Hierarchical TimeXer: indicator layer, then text layer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader

from src.data.collate import forecast_collate
from src.explain.bank import collect_segment_bank
from src.models.timexer_dual import TimeXerDual
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
        e_layers=2,
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
    )
    kwargs.update(overrides)
    return TimeXerHierarchical(**kwargs)


def _batch(n_features: int = 2):
    x = torch.randn(2, 60, n_features)
    text = torch.randn(2, 60, 15)
    ts = torch.randn(2, 60, 25)
    y = torch.randn(2, 7)
    return x, text, ts, y


def test_depth_must_be_exactly_two_layers() -> None:
    for depth in (1, 3):
        with pytest.raises(ValueError, match="exactly two layers"):
            _model(e_layers=depth)


def test_forward_shapes_and_closed_gates() -> None:
    torch.manual_seed(0)
    model = _model()
    x, text, ts, _y = _batch()
    output = model(x, text_seq=text, ts=ts)
    assert output.pred.shape == (2, 7)
    assert output.segments.shape == (2, 5, 64)
    assert torch.isfinite(output.pred).all()
    assert torch.isfinite(output.segments).all()
    assert float(model.layer_ts.bridge.alpha.detach()) == 0.0
    assert float(model.layer_text.bridge.alpha.detach()) == 0.0
    assert output.proto_losses is None


def test_each_layer_attends_over_patches_plus_one_token() -> None:
    model = _model()
    assert sum(isinstance(module, nn.MultiheadAttention) for module in model.modules()) == 6
    assert not hasattr(model, "dual_attention_mask")
    assert not hasattr(model.layer_ts, "cross_ts")
    assert not hasattr(model.layer_text, "cross_text")
    lengths: dict[str, int] = {}

    def _record(name: str):
        def hook(module, args):
            del module
            lengths[name] = int(args[0].shape[1])

        return hook

    model.layer_ts.self_attn.register_forward_pre_hook(_record("ts"))
    model.layer_text.self_attn.register_forward_pre_hook(_record("text"))
    x, text, ts, _y = _batch()
    model(x, text_seq=text, ts=ts)
    assert lengths["ts"] == model.n_patches + 1
    assert lengths["text"] == model.n_patches + 1
    assert lengths["ts"] != model.n_patches + 2


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
    for layer in (model.layer_ts, model.layer_text):
        grad = layer.cross.in_proj_weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0


def test_text_layer_sees_indicator_patches_with_a_closed_gate() -> None:
    torch.manual_seed(2)
    model = _model()
    model.eval()
    with torch.no_grad():
        model.layer_ts.bridge.alpha.zero_()
    x, text, ts, _y = _batch()
    base = model(x, text_seq=text, ts=ts).pred
    changed_ts = model(x, text_seq=text, ts=torch.randn_like(ts)).pred
    changed_text = model(x, text_seq=torch.zeros_like(text), ts=ts).pred
    assert torch.isfinite(base).all()
    assert not torch.allclose(base, changed_ts)
    assert not torch.allclose(base, changed_text)


def test_parameter_budget_is_below_the_dual_stack() -> None:
    hierarchical = _model(use_prototypes=False, d_ff=256)
    dual = TimeXerDual(
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
            "kind": "c1_dual",
            "text_at_head": False,
            "text_to_patches": False,
            "text_as_exogenous": True,
            "global_to_patch": True,
        },
        n_ts_features=25,
        use_prototypes=False,
        d_ff=256,
        head_pool="last",
    )
    n_hierarchical = sum(p.numel() for p in hierarchical.parameters())
    n_dual = sum(p.numel() for p in dual.parameters())
    assert n_hierarchical < 185_000
    assert n_hierarchical < n_dual


def test_prototypes_keep_pre_injection_segments_and_the_bank_accepts_ts() -> None:
    torch.manual_seed(3)
    model = _model(use_prototypes=True)
    x, text, ts, _y = _batch()
    seen: dict[str, torch.Tensor] = {}

    def _capture(module, args):
        del module
        seen["patches"] = args[0].detach()

    model.head.register_forward_pre_hook(_capture)
    output = model(x, text_seq=text, ts=ts, proto_mode="zero")
    assert output.proto_losses is not None
    assert torch.isfinite(output.proto_losses.l_c)
    assert torch.equal(seen["patches"], output.segments)
    items = []
    for end_idx in range(2):
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
    bank, _meta = collect_segment_bank(model, loader, max_batches=1)
    assert bank.ndim == 2 and bank.size(1) == 64
    assert torch.isfinite(bank).all()


def test_hydra_builds_the_hierarchical_model() -> None:
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
                "model.e_layers=2",
            ],
        )
    assert cfg.model.name == "c1_hierarchical"
    assert cfg.model.fusion.kind == "c1_hierarchical"
    assert int(cfg.model.e_layers) == 2
    assert int(cfg.model.n_ts_features) == 25
    assert int(cfg.model.d_ff) == 256
    assert cfg.data.features_mode == "dual_ts"
    model = build_model(cfg)
    assert isinstance(model, TimeXerHierarchical)
    assert model.n_features == 2
    assert model.g_ts.in_features == 25
    assert model.patch_proj.in_features == 30
    assert model.layer_ts.ff[0].out_features == 256
    assert model.head.pool == "last"
