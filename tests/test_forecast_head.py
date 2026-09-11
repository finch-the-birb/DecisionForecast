"""Tests for ForecastHead pooling (mean / last / global / flatten)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.head import ForecastHead
from src.models.timexer_b import TimeXerB
from src.models.timexer_c0 import TimeXerC0
from src.models.timexer_c1 import TimeXerC1
from src.models.timexer_plain import TimeXerPlain
from src.models.timexl_a import TimeXLModelA
from src.training.train import build_model

_FUSION_LATE = {
    "kind": "late",
    "text_at_head": True,
    "text_to_patches": False,
    "text_as_exogenous": False,
}
_FUSION_C0 = {
    "kind": "mid_no_attn",
    "text_at_head": False,
    "text_to_patches": True,
    "text_align": "per_patch",
    "text_inject": "add",
    "inject_layers": [0],
    "text_as_exogenous": False,
}
_FUSION_C1 = {
    "kind": "mid_cross_attn",
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "exo_tokens": "per_day",
}


def test_forecast_head_pools_and_mlp() -> None:
    b, n, d, h = 4, 9, 64, 7
    patches = torch.randn(b, n, d)
    g_en = torch.randn(b, 1, d)

    h_mean = ForecastHead(d_model=d, horizon=h, pool="mean", dropout=0.0)
    out_mean = h_mean(patches)
    assert out_mean.shape == (b, h)
    assert h_mean.net[1].in_features == d
    torch.testing.assert_close(out_mean, h_mean.net(patches.mean(dim=1)))

    h_last = ForecastHead(d_model=d, horizon=h, pool="last")
    out_last = h_last(patches)
    assert out_last.shape == (b, h)
    torch.testing.assert_close(out_last, h_last.net(patches[:, -1, :]))
    assert not torch.allclose(out_mean, out_last)

    h_glob = ForecastHead(d_model=d, horizon=h, pool="global")
    out_glob = h_glob(patches, g_en=g_en)
    assert out_glob.shape == (b, h)
    torch.testing.assert_close(out_glob, h_glob.net(g_en[:, 0, :]))
    with pytest.raises(ValueError, match="global token required"):
        h_glob(patches, g_en=None)

    extra = torch.randn(b, 32)
    h_flat = ForecastHead(
        d_model=d,
        horizon=h,
        pool="flatten",
        n_patches=n,
        head_type="mlp",
        head_hidden=128,
        extra_dim=32,
    )
    out_flat = h_flat(patches, extra=extra)
    assert out_flat.shape == (b, h)
    assert h_flat.net[1].in_features == n * d + 32

    with pytest.raises(ValueError, match="n_patches is required"):
        ForecastHead(d_model=d, horizon=h, pool="flatten")
    with pytest.raises(ValueError, match="Unknown pool"):
        ForecastHead(d_model=d, horizon=h, pool="median")
    with pytest.raises(ValueError, match="Unknown head_type"):
        ForecastHead(d_model=d, horizon=h, head_type="invalid")


def _plain(lookback_t: int, d_model: int, pool: str = "mean") -> TimeXerPlain:
    return TimeXerPlain(
        n_features=5,
        seq_len=lookback_t,
        horizon=7,
        d_model=d_model,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        head_type="linear",
        head_pool=pool,
    )


def _model_a(lookback_t: int, d_model: int, text_dim: int, pool: str = "mean") -> TimeXLModelA:
    return TimeXLModelA(
        n_features=5,
        seq_len=lookback_t,
        horizon=7,
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
        fusion=_FUSION_LATE,
        head_type="linear",
        head_pool=pool,
    )


def _model_b(lookback_t: int, d_model: int, text_dim: int, pool: str = "mean") -> TimeXerB:
    return TimeXerB(
        n_features=5,
        seq_len=lookback_t,
        horizon=7,
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
        fusion=_FUSION_LATE,
        head_type="linear",
        head_pool=pool,
    )


def _model_c0(lookback_t: int, d_model: int, text_dim: int, pool: str = "mean") -> TimeXerC0:
    return TimeXerC0(
        n_features=5,
        seq_len=lookback_t,
        horizon=7,
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
        fusion=_FUSION_C0,
        head_type="linear",
        head_pool=pool,
    )


def _model_c1(lookback_t: int, d_model: int, text_dim: int, pool: str = "mean") -> TimeXerC1:
    return TimeXerC1(
        n_features=5,
        seq_len=lookback_t,
        horizon=7,
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
        fusion=_FUSION_C1,
        head_type="linear",
        head_pool=pool,
    )


@pytest.mark.parametrize("pool", ["mean", "last", "flatten"])
@pytest.mark.parametrize("lookback_t,expected_n", [(36, 5), (60, 9), (96, 15)])
def test_all_models_mean_last_flatten(pool: str, lookback_t: int, expected_n: int) -> None:
    b = 2
    d_model = 32
    text_dim = 64
    x = torch.randn(b, lookback_t, 5)
    text = torch.randn(b, text_dim)
    text_seq = torch.randn(b, lookback_t, text_dim)
    compact_in = d_model
    flatten_in = expected_n * d_model

    m_plain = _plain(lookback_t, d_model, pool)
    assert m_plain(x).pred.shape == (b, 7)
    assert m_plain.head.pool == pool
    assert m_plain.head.net[1].in_features == (flatten_in if pool == "flatten" else compact_in)

    m_a = _model_a(lookback_t, d_model, text_dim, pool)
    assert m_a(x, text).pred.shape == (b, 7)
    extra = d_model
    assert m_a.head.net[1].in_features == (
        flatten_in + extra if pool == "flatten" else compact_in + extra
    )

    m_b = _model_b(lookback_t, d_model, text_dim, pool)
    assert m_b(x, text).pred.shape == (b, 7)
    assert m_b.head.net[1].in_features == (
        flatten_in + extra if pool == "flatten" else compact_in + extra
    )

    m_c0 = _model_c0(lookback_t, d_model, text_dim, pool)
    assert m_c0(x, text, text_seq=text_seq).pred.shape == (b, 7)
    assert m_c0.head.net[1].in_features == (flatten_in if pool == "flatten" else compact_in)

    m_c1 = _model_c1(lookback_t, d_model, text_dim, pool)
    assert m_c1(x, text, text_seq=text_seq).pred.shape == (b, 7)
    assert m_c1.head.net[1].in_features == (flatten_in if pool == "flatten" else compact_in)


def test_c1_global_pool_uses_g_en() -> None:
    b, t, d_model, text_dim = 2, 60, 32, 64
    x = torch.randn(b, t, 5)
    text = torch.randn(b, text_dim)
    text_seq = torch.randn(b, t, text_dim)
    model = _model_c1(t, d_model, text_dim, pool="global")
    assert model.head.pool == "global"
    assert model.head.net[1].in_features == d_model
    out = model(x, text, text_seq=text_seq)
    assert out.pred.shape == (b, 7)

    with pytest.raises(ValueError, match="global token required"):
        model.head(torch.randn(b, 9, d_model), g_en=None)


def test_hydra_build_default_mean_and_overrides() -> None:
    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        for m in ("a", "b", "c0", "c1", "timexer_plain"):
            cfg = compose(config_name="config", overrides=[f"model={m}"])
            model = build_model(cfg)
            assert isinstance(model.head, ForecastHead)
            assert model.head.pool == "mean"
            assert model.head.head_type == "linear"

        cfg_last = compose(
            config_name="config",
            overrides=["model=c1", "model.head.pool=last", "model.head.type=mlp"],
        )
        m_last = build_model(cfg_last)
        assert isinstance(m_last, TimeXerC1)
        assert m_last.head.pool == "last"
        assert m_last.head.head_type == "mlp"

        cfg_glob = compose(
            config_name="config",
            overrides=["model=c1", "model.head.pool=global"],
        )
        m_glob = build_model(cfg_glob)
        assert m_glob.head.pool == "global"

        cfg_flat = compose(
            config_name="config",
            overrides=["model=timexer_plain", "model.head.pool=flatten"],
        )
        m_flat = build_model(cfg_flat)
        assert m_flat.head.pool == "flatten"
        assert m_flat.head.net[1].in_features == 9 * int(cfg_flat.model.d_model)

        cfg_dl = compose(config_name="config", overrides=["model=dlinear"])
        assert build_model(cfg_dl) is not None
