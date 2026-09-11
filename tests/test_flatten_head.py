"""Tests for FlattenHead and all models with temporal flattening head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.head import FlattenHead
from src.models.timexer_b import TimeXerB
from src.models.timexer_c0 import TimeXerC0
from src.models.timexer_c1 import TimeXerC1
from src.models.timexer_plain import TimeXerPlain
from src.models.timexl_a import TimeXLModelA
from src.training.train import build_model


def test_flatten_head_linear_and_mlp() -> None:
    b, n, d, h = 4, 9, 64, 7
    x = torch.randn(b, n, d)

    # Linear without extra
    h_lin = FlattenHead(n_patches=n, d_model=d, horizon=h, head_type="linear", dropout=0.1)
    out_lin = h_lin(x)
    assert out_lin.shape == (b, h)
    assert h_lin.net[1].in_features == n * d
    assert h_lin.net[1].out_features == h

    # MLP with extra
    extra_d = 32
    extra = torch.randn(b, extra_d)
    h_mlp = FlattenHead(
        n_patches=n,
        d_model=d,
        horizon=h,
        head_type="mlp",
        head_hidden=128,
        dropout=0.1,
        extra_dim=extra_d,
    )
    out_mlp = h_mlp(x, extra=extra)
    assert out_mlp.shape == (b, h)
    assert h_mlp.net[1].in_features == n * d + extra_d
    assert h_mlp.net[3].out_features == h

    # Invalid type
    with pytest.raises(ValueError, match="Unknown head_type"):
        FlattenHead(n_patches=n, d_model=d, horizon=h, head_type="invalid")


@pytest.mark.parametrize("lookback_t,expected_n", [(36, 5), (60, 9), (96, 15)])
def test_all_models_flatten_head_dimensions(lookback_t: int, expected_n: int) -> None:
    b = 2
    horizon = 7
    d_model = 32
    text_dim = 64
    x = torch.randn(b, lookback_t, 5)
    text = torch.randn(b, text_dim)
    text_seq = torch.randn(b, lookback_t, text_dim)

    # 1. Plain
    m_plain = TimeXerPlain(
        n_features=5,
        seq_len=lookback_t,
        horizon=horizon,
        d_model=d_model,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        head_type="linear",
    )
    out_plain = m_plain(x)
    assert out_plain.pred.shape == (b, horizon)
    assert m_plain.head.net[1].in_features == expected_n * d_model

    # 2. Model A
    m_a = TimeXLModelA(
        n_features=5,
        seq_len=lookback_t,
        horizon=horizon,
        d_model=d_model,
        n_prototypes=4,
        d_min=0.5,
        patch_len=12,
        patch_stride=6,
        cnn_channels=[32],
        cnn_kernel=3,
        text_dim=text_dim,
        text_hidden=32,
        head_hidden=64,
        fusion={"kind": "late", "text_at_head": True, "text_to_patches": False, "text_as_exogenous": False},
        head_type="linear",
    )
    out_a = m_a(x, text)
    assert out_a.pred.shape == (b, horizon)
    # in_features = expected_n * d_model + extra_dim (d_model)
    assert m_a.head.net[1].in_features == expected_n * d_model + d_model

    # 3. Model B
    m_b = TimeXerB(
        n_features=5,
        seq_len=lookback_t,
        horizon=horizon,
        d_model=d_model,
        n_prototypes=4,
        d_min=0.5,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=text_dim,
        text_hidden=32,
        head_hidden=64,
        fusion={"kind": "late", "text_at_head": True, "text_to_patches": False, "text_as_exogenous": False},
        head_type="linear",
    )
    out_b = m_b(x, text)
    assert out_b.pred.shape == (b, horizon)
    assert m_b.head.net[1].in_features == expected_n * d_model + d_model

    # 4. Model C0
    m_c0 = TimeXerC0(
        n_features=5,
        seq_len=lookback_t,
        horizon=horizon,
        d_model=d_model,
        n_prototypes=4,
        d_min=0.5,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=text_dim,
        head_hidden=64,
        fusion={
            "kind": "mid_no_attn",
            "text_at_head": False,
            "text_to_patches": True,
            "text_align": "per_patch",
            "text_inject": "add",
            "inject_layers": [0],
            "text_as_exogenous": False,
        },
        head_type="linear",
    )
    out_c0 = m_c0(x, text, text_seq=text_seq)
    assert out_c0.pred.shape == (b, horizon)
    assert m_c0.head.net[1].in_features == expected_n * d_model

    # 5. Model C1
    m_c1 = TimeXerC1(
        n_features=5,
        seq_len=lookback_t,
        horizon=horizon,
        d_model=d_model,
        n_prototypes=4,
        d_min=0.5,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=text_dim,
        head_hidden=64,
        fusion={
            "kind": "mid_cross_attn",
            "text_at_head": False,
            "text_to_patches": False,
            "text_as_exogenous": True,
            "exo_tokens": "per_day",
        },
        head_type="linear",
    )
    out_c1 = m_c1(x, text, text_seq=text_seq)
    assert out_c1.pred.shape == (b, horizon)
    assert m_c1.head.net[1].in_features == expected_n * d_model


def test_hydra_build_all_models_with_flatten_head() -> None:
    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        for m in ("a", "b", "c0", "c1", "timexer_plain"):
            cfg = compose(
                config_name="config",
                overrides=[f"model={m}", "model.head.type=mlp", "model.head.dropout=0.2"],
            )
            model = build_model(cfg)
            assert model is not None
            assert hasattr(model, "head")
            assert isinstance(model.head, FlattenHead)
            assert model.head.head_type == "mlp"
        
        # dlinear doesn't use head config
        cfg_dl = compose(config_name="config", overrides=["model=dlinear"])
        m_dl = build_model(cfg_dl)
        assert m_dl is not None
