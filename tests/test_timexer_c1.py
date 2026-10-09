"""Model C1 cross-attn fusion (CPU, synthetic)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.timexer_b import TimeXerB
from src.models.timexer_c0 import TimeXerC0
from src.models.timexer_c1 import TimeXerC1
from src.training.train import build_model


def _c1_fusion(**overrides):
    flags = {
        "kind": "mid_cross_attn",
        "text_at_head": False,
        "text_to_patches": False,
        "text_as_exogenous": True,
        "exo_tokens": "per_day",
    }
    flags.update(overrides)
    return flags


def _make_c1(**fusion_overrides) -> TimeXerC1:
    return TimeXerC1(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=32,
        n_prototypes=4,
        d_min=0.5,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=768,
        head_hidden=16,
        fusion=_c1_fusion(**fusion_overrides),
        d_ff=64,
    )


def _shared_kwargs() -> dict:
    return dict(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=32,
        n_prototypes=4,
        d_min=0.5,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        d_ff=64,
        head_hidden=16,
    )


def test_c1_fusion_rejects_wrong_flags() -> None:
    _make_c1()
    with pytest.raises(AssertionError, match="text_at_head"):
        _make_c1(text_at_head=True)
    with pytest.raises(AssertionError, match="text_to_patches"):
        _make_c1(text_to_patches=True)
    with pytest.raises(AssertionError, match="text_as_exogenous"):
        _make_c1(text_as_exogenous=False)


def test_c1_proto_and_patches_independent_of_text() -> None:
    model = TimeXerC1(
        n_features=5,
        seq_len=60,
        horizon=7,
        d_model=32,
        n_prototypes=4,
        d_min=0.5,
        n_heads=4,
        e_layers=2,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        text_dim=768,
        head_hidden=16,
        fusion=_c1_fusion(),
        d_ff=64,
    )
    model.eval()
    x = torch.randn(2, 60, 5)
    text = torch.randn(2, 768)
    seq_a = torch.randn(2, 60, 768)
    seq_b = torch.zeros(2, 60, 768)
    proto_in: list[torch.Tensor] = []
    encode_p: list[torch.Tensor] = []
    encode_exo: list[torch.Tensor] = []
    orig_proto = model.g12.proto.forward
    orig_encode = model.backbone.encode

    def proto_wrap(segments):
        proto_in.append(segments.detach().clone())
        return orig_proto(segments)

    def encode_wrap(patches, g_en, exo=None):
        encode_p.append(patches.detach().clone())
        assert exo is not None
        encode_exo.append(exo.detach().clone())
        return orig_encode(patches, g_en, exo)

    model.g12.proto.forward = proto_wrap  # type: ignore[method-assign]
    model.backbone.encode = encode_wrap  # type: ignore[method-assign]
    with torch.no_grad():
        out_a = model(x, text, text_seq=seq_a)
        out_b = model(x, text, text_seq=seq_b)
    torch.testing.assert_close(proto_in[0], proto_in[1])
    torch.testing.assert_close(encode_p[0], encode_p[1])
    assert encode_exo[0].shape == (2, 60, 32)
    assert not torch.allclose(out_a.pred, out_b.pred)
    assert model.head.pool == "mean"
    assert model.head.net[1].in_features == 32


def test_c1_per_patch_exo_shape() -> None:
    model = _make_c1(exo_tokens="per_patch")
    captured = []
    orig = model.backbone.encode

    def wrap(patches, g_en, exo=None):
        captured.append(exo.shape if exo is not None else None)
        return orig(patches, g_en, exo)

    model.backbone.encode = wrap  # type: ignore[method-assign]
    n = (60 - 12) // 6 + 1
    model(torch.randn(2, 60, 5), torch.randn(2, 768), text_seq=torch.randn(2, 60, 768))
    assert captured[0] == (2, n, 32)


def test_backbone_g12_param_parity() -> None:
    kw = _shared_kwargs()
    b = TimeXerB(
        **kw,
        text_dim=768,
        text_hidden=16,
        fusion={
            "kind": "late",
            "text_at_head": True,
            "text_to_patches": False,
            "text_as_exogenous": False,
        },
    )
    c0 = TimeXerC0(
        **kw,
        text_dim=768,
        fusion={
            "kind": "mid_no_attn",
            "text_at_head": False,
            "text_to_patches": True,
            "text_align": "per_patch",
            "text_inject": "add",
            "inject_layers": [0],
            "text_as_exogenous": False,
        },
    )
    c1 = TimeXerC1(
        **kw,
        text_dim=768,
        fusion=_c1_fusion(),
    )

    def n_core(m):
        return sum(p.numel() for p in m.backbone.parameters()) + sum(
            p.numel() for p in m.g12.parameters()
        )

    assert n_core(b) == n_core(c0) == n_core(c1)


def test_hydra_model_c1() -> None:
    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=["model=c1"])
    assert cfg.model.fusion.kind == "mid_cross_attn"
    assert cfg.model.fusion.exo_tokens == "per_day"
    model = build_model(cfg)
    assert isinstance(model, TimeXerC1)
    assert int(cfg.model.e_layers) == 1
    assert model.head.pool == "last"
    assert model.head.net[1].in_features == int(cfg.model.d_model)
