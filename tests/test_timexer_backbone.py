"""TimeXer backbone patching and encoder (CPU, no data)."""

from __future__ import annotations

import torch

from src.models.dlinear import DLinear
from src.models.timexer_backbone import (
    TimeXerBackbone,
    days_to_patches,
    n_patches,
    unfold_time,
)
from src.models.timexer_plain import TimeXerPlain


def test_patch_count_matches_days_to_patches() -> None:
    b, t, c, d = 2, 60, 5, 768
    patch_len, stride = 12, 6
    x = torch.randn(b, t, c)
    text = torch.randn(b, t, d)
    n = n_patches(t, patch_len, stride)
    assert unfold_time(x, patch_len, stride).shape[1] == n
    tp = days_to_patches(text, patch_len, stride)
    assert tp.shape == (b, n, d)


def test_backbone_exo_none_and_cross_attn_query_shape() -> None:
    b, t, c = 2, 60, 5
    bb = TimeXerBackbone(
        n_features=c, d_model=32, n_heads=4, e_layers=2, patch_len=12, patch_stride=6, dropout=0.0
    )
    x = torch.randn(b, t, c)
    p, g = bb.embed(x)
    assert p.shape[0] == b and p.dim() == 3 and g.shape == (b, 1, 32)
    enc, g2 = bb.encode(p, g, exo=None)
    assert enc.shape == p.shape and g2.shape == (b, 1, 32)
    exo = torch.randn(b, 7, 32)
    enc2, g3 = bb.encode(p, g, exo=exo)
    assert enc2.shape == p.shape and g3.shape == (b, 1, 32)


def test_plain_and_dlinear_forward_shapes() -> None:
    b, t, c, h = 3, 60, 5, 7
    x = torch.randn(b, t, c)
    text = torch.randn(b, 768)
    plain = TimeXerPlain(
        n_features=c,
        horizon=h,
        d_model=32,
        n_heads=4,
        e_layers=1,
        patch_len=12,
        patch_stride=6,
        dropout=0.0,
        head_hidden=16,
    )
    dl = DLinear(seq_len=t, horizon=h, n_features=c, target_idx=0)
    for model in (plain, dl):
        model.eval()
        out = model(x, text)
        assert out.pred.shape == (b, h)
        assert out.proto_losses is None
        loss, metrics = model.compute_loss(out, torch.randn(b, h), 0.0, 0.0, 0.0)
        assert loss.ndim == 0 and "l_pred" in metrics
