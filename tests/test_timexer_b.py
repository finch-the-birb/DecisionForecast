"""Model B late-fusion flags and forward (CPU, synthetic)."""

from __future__ import annotations

import pytest
import torch

from src.models.fusion import assert_fusion
from src.models.timexer_b import TimeXerB
from src.training.train import build_model


def _fusion(**overrides):
    flags = {
        "kind": "late",
        "text_at_head": True,
        "text_to_patches": False,
        "text_as_exogenous": False,
    }
    flags.update(overrides)
    return flags


def _make_b(**fusion_overrides) -> TimeXerB:
    return TimeXerB(
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
        text_hidden=16,
        head_hidden=16,
        fusion=_fusion(**fusion_overrides),
        d_ff=64,
    )


def test_b_fusion_asserts_nested_flags() -> None:
    _make_b()
    with pytest.raises(AssertionError, match="text_at_head"):
        _make_b(text_at_head=False)
    with pytest.raises(AssertionError, match="must be a mapping"):
        TimeXerB(
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
            text_hidden=16,
            head_hidden=16,
            fusion="late",
        )


def test_b_head_is_late_concat_and_ignores_text_seq() -> None:
    model = _make_b()
    model.eval()
    # Mean-pool + late text: in_features = d_model + extra_dim = 32 + 32
    assert model.head.pool == "mean"
    assert model.head.net[1].in_features == 32 + 32
    b, t, c = 2, 60, 5
    x = torch.randn(b, t, c)
    text = torch.randn(b, 768)
    seq_a = torch.randn(b, t, 768)
    seq_b = torch.zeros(b, t, 768)
    with torch.no_grad():
        out_a = model(x, text, text_seq=seq_a)
        out_b = model(x, text, text_seq=seq_b)
    assert out_a.pred.shape == (b, 7)
    assert out_a.proto_losses is not None
    torch.testing.assert_close(out_a.pred, out_b.pred)
    loss, metrics = model.compute_loss(out_a, torch.randn(b, 7), 0.1, 0.1, 0.01)
    assert loss.ndim == 0 and "l_c" in metrics


def test_b_g_en_not_in_segments_bank() -> None:
    model = _make_b()
    x = torch.randn(2, 60, 5)
    text = torch.randn(2, 768)
    out = model(x, text)
    # segments are patch tokens P (no G_en concatenated)
    n = (60 - 12) // 6 + 1
    assert out.segments.shape == (2, n, 32)


def test_hydra_model_b_nested_fusion() -> None:
    from hydra.core.global_hydra import GlobalHydra
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    GlobalHydra.instance().clear()
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=["model=b"])
    assert cfg.model.fusion.kind == "late"
    assert cfg.model.fusion.text_at_head is True
    model = build_model(cfg)
    assert isinstance(model, TimeXerB)
    assert_fusion(
        cfg.model.fusion,
        kind="late",
        text_at_head=True,
        text_to_patches=False,
        text_as_exogenous=False,
    )
