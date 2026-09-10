"""C0/C1 fusion invariants (CPU, synthetic, no FNSPID)."""

from __future__ import annotations

import pytest
import torch

from src.models.timexer_b import TimeXerB
from src.models.timexer_backbone import days_to_patches, n_patches, unfold_time
from src.models.timexer_c0 import TimeXerC0
from src.training.train import build_model


def _c0_fusion(**overrides):
    flags = {
        "kind": "mid_no_attn",
        "text_at_head": False,
        "text_to_patches": True,
        "text_align": "per_patch",
        "text_inject": "add",
        "inject_layers": [0],
        "text_as_exogenous": False,
    }
    flags.update(overrides)
    return flags


def _make_c0(**fusion_overrides) -> TimeXerC0:
    return TimeXerC0(
        n_features=5,
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
        fusion=_c0_fusion(**fusion_overrides),
        d_ff=64,
    )


def _make_b() -> TimeXerB:
    return TimeXerB(
        n_features=5,
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
        text_hidden=16,
        head_hidden=16,
        fusion={
            "kind": "late",
            "text_at_head": True,
            "text_to_patches": False,
            "text_as_exogenous": False,
        },
        d_ff=64,
    )


def test_days_to_patches_shares_patchembed_indices() -> None:
    b, t, c, d = 2, 60, 5, 768
    patch_len, stride = 12, 6
    x = torch.randn(b, t, c)
    text = torch.randn(b, t, d)
    n = n_patches(t, patch_len, stride)
    assert unfold_time(x, patch_len, stride).shape[1] == n
    assert unfold_time(text, patch_len, stride).shape[1] == n
    model = _make_c0()
    p = model.backbone.patch_embed(x)
    e = days_to_patches(text, patch_len, stride)
    assert p.shape[1] == e.shape[1] == n


def test_c0_rejects_pooled_and_multi_inject() -> None:
    with pytest.raises(ValueError, match="pooled text-add is B, not C0"):
        _make_c0(text_align="pooled")
    with pytest.raises(NotImplementedError, match="inject_layers"):
        _make_c0(inject_layers=[0, 1])
    with pytest.raises(AssertionError, match="text_at_head"):
        _make_c0(text_at_head=True)


def test_c0_proto_before_text_and_g_en_unmodified() -> None:
    model = _make_c0()
    model.eval()
    x = torch.randn(2, 60, 5)
    text = torch.randn(2, 768)
    seq_a = torch.randn(2, 60, 768)
    seq_b = torch.zeros(2, 60, 768)
    proto_inputs: list[torch.Tensor] = []
    g_ens: list[torch.Tensor] = []
    orig_proto = model.g12.proto.forward
    orig_encode = model.backbone.encode

    def proto_wrap(segments: torch.Tensor):
        proto_inputs.append(segments.detach().clone())
        return orig_proto(segments)

    def encode_wrap(patches, g_en, exo=None):
        g_ens.append(g_en.detach().clone())
        assert exo is None
        return orig_encode(patches, g_en, exo)

    model.g12.proto.forward = proto_wrap  # type: ignore[method-assign]
    model.backbone.encode = encode_wrap  # type: ignore[method-assign]
    with torch.no_grad():
        out_a = model(x, text, text_seq=seq_a)
        out_b = model(x, text, text_seq=seq_b)
    assert len(proto_inputs) == 2
    torch.testing.assert_close(proto_inputs[0], proto_inputs[1])
    torch.testing.assert_close(out_a.proto_losses.l_c, out_b.proto_losses.l_c)
    torch.testing.assert_close(g_ens[0], g_ens[1])
    assert not torch.allclose(out_a.pred, out_b.pred)


def test_c0_head_has_no_text_concat() -> None:
    model = _make_c0()
    assert model.head[0].in_features == 32
    out = model(torch.randn(3, 60, 5), torch.randn(3, 768), text_seq=torch.randn(3, 60, 768))
    assert out.pred.shape == (3, 7)


def test_zero_text_seq_b_c0_c1_pred_shapes() -> None:
    b_m, c0_m = _make_b(), _make_c0()
    from src.models.timexer_c1 import TimeXerC1

    c1_m = TimeXerC1(
        n_features=5,
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
        fusion={
            "kind": "mid_cross_attn",
            "text_at_head": False,
            "text_to_patches": False,
            "text_as_exogenous": True,
            "exo_tokens": "per_day",
        },
        d_ff=64,
    )
    x = torch.randn(2, 60, 5)
    text = torch.zeros(2, 768)
    text_seq = torch.zeros(2, 60, 768)
    with torch.no_grad():
        pb = b_m(x, text, text_seq=text_seq).pred
        pc = c0_m(x, text, text_seq=text_seq).pred
        p1 = c1_m(x, text, text_seq=text_seq).pred
    assert pb.shape == pc.shape == p1.shape == (2, 7)


def test_hydra_model_c0_nested_fusion() -> None:
    from hydra.core.global_hydra import GlobalHydra
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=["model=c0"])
    assert cfg.model.fusion.kind == "mid_no_attn"
    assert cfg.model.fusion.text_align == "per_patch"
    model = build_model(cfg)
    assert isinstance(model, TimeXerC0)
    assert model.head[0].in_features == int(cfg.model.d_model)
