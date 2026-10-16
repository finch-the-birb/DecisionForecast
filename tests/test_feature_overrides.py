"""Hydra feature-list overrides resize PatchEmbed / DLinear without training."""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.models.dlinear import DLinear
from src.models.timexer_c1 import TimeXerC1
from src.models.timexer_plain import TimeXerPlain
from src.training.train import build_model

_CFG = str(Path(__file__).resolve().parents[1] / "configs")
_F1 = "data.features=[close]"
_F2 = "data.features=[close,volume]"
_F5 = "data.features=[close,volume,open,high,low]"


def _compose(*overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=_CFG):
        return compose(config_name="config", overrides=list(overrides))


def test_hydra_feature_lists_resolve() -> None:
    for override, expected in (
        (_F1, ["close"]),
        (_F2, ["close", "volume"]),
        (_F5, ["close", "volume", "open", "high", "low"]),
    ):
        cfg = _compose("model=dlinear", override)
        assert [str(f) for f in cfg.data.features] == expected
        assert str(cfg.data.target) == "close"
        assert list(cfg.data.features).index("close") == 0


def test_n_features_drives_patchembed_and_dlinear() -> None:
    cases = (
        (_F1, 1),
        (_F2, 2),
        (_F5, 5),
    )
    for override, n_feat in cases:
        cfg_c1 = _compose("model=c1", "model.n_prototypes=5", override)
        c1 = build_model(cfg_c1)
        assert isinstance(c1, TimeXerC1)
        patch_in = n_feat * int(cfg_c1.data.patch_len)
        assert c1.backbone.patch_embed.value.in_features == patch_in
        x = torch.randn(2, int(cfg_c1.data.lookback_T), n_feat)
        text = torch.randn(2, int(cfg_c1.data.text.dim))
        text_seq = torch.randn(2, int(cfg_c1.data.lookback_T), int(cfg_c1.data.text.dim))
        assert c1(x, text, text_seq=text_seq).pred.shape == (2, int(cfg_c1.data.horizon))

        cfg_dl = _compose("model=dlinear", override)
        dl = build_model(cfg_dl)
        assert isinstance(dl, DLinear)
        assert dl.target_idx == 0
        assert dl.linear_seasonal.in_features == int(cfg_dl.data.lookback_T)
        assert dl(x).pred.shape == (2, int(cfg_dl.data.horizon))


def test_plain_e_layers_override() -> None:
    cfg = _compose("model=timexer_plain", "model.e_layers=1", "model.head.pool=last", _F1)
    model = build_model(cfg)
    assert isinstance(model, TimeXerPlain)
    assert len(model.backbone.layers) == 1
    assert model.head.pool == "last"
    assert model.backbone.patch_embed.value.in_features == 1 * int(cfg.data.patch_len)
    x = torch.randn(2, int(cfg.data.lookback_T), 1)
    assert model(x).pred.shape == (2, int(cfg.data.horizon))
