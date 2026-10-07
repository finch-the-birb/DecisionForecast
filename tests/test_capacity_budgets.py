"""Locked parameter counts for the 75% and 50% width budgets.

The attention graph stays the dual stack or the two specialized layers.
Width is only ``d_model``, ``n_heads``, and ``d_ff``.
"""

from __future__ import annotations

import pytest
import torch

from src.models.timexer_dual import TimeXerDual
from src.models.timexer_hierarchical import TimeXerHierarchical

_DUAL_FUSION = {
    "kind": "c1_dual",
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "global_to_patch": True,
}
_HIERARCHICAL_FUSION = {
    "kind": "c1_hierarchical",
    "text_at_head": False,
    "text_to_patches": False,
    "text_as_exogenous": True,
    "global_to_patch": True,
}

# Section 12.1 bases, measured on the full-width F2 runs.
_BASES = (
    ("dual", 1, 64, 4, 256, 122_121),
    ("dual", 2, 64, 4, 256, 239_051),
    ("hierarchical", 2, 64, 4, 256, 172_169),
)

# Section 12.1 width cuts. Counts are exact, not a rounded fraction of the base.
_BUDGETS = (
    ("dual", 1, 56, 2, 200, 91_601),
    ("dual", 2, 56, 2, 200, 178_659),
    ("hierarchical", 2, 56, 2, 208, 129_113),
    ("dual", 1, 48, 3, 96, 60_777),
    ("dual", 2, 48, 3, 104, 119_211),
    ("hierarchical", 2, 48, 3, 128, 85_993),
)


def _build(kind: str, e_layers: int, d_model: int, n_heads: int, d_ff: int):
    common = dict(
        n_features=2,
        seq_len=60,
        horizon=7,
        patch_len=12,
        patch_stride=12,
        dropout=0.0,
        text_dim=15,
        n_ts_features=25,
        close_idx=0,
        use_prototypes=False,
        head_pool="last",
        e_layers=e_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
    )
    if kind == "dual":
        return TimeXerDual(fusion=_DUAL_FUSION, **common)
    if kind == "hierarchical":
        return TimeXerHierarchical(fusion=_HIERARCHICAL_FUSION, **common)
    raise ValueError(kind)


def _assert_budget(kind: str, e_layers: int, d_model: int, n_heads: int, d_ff: int, n_params: int) -> None:
    assert d_model % n_heads == 0
    assert d_model // n_heads >= 16
    model = _build(kind, e_layers, d_model, n_heads, d_ff)
    assert sum(parameter.numel() for parameter in model.parameters()) == n_params
    pred = model(
        torch.randn(2, 60, 2),
        text_seq=torch.randn(2, 60, 15),
        ts=torch.randn(2, 60, 25),
    ).pred
    assert pred.shape == (2, 7)
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize(("kind", "e_layers", "d_model", "n_heads", "d_ff", "n_params"), _BASES)
def test_full_width_bases_match_measured_counts(
    kind: str,
    e_layers: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_params: int,
) -> None:
    _assert_budget(kind, e_layers, d_model, n_heads, d_ff, n_params)


@pytest.mark.parametrize(("kind", "e_layers", "d_model", "n_heads", "d_ff", "n_params"), _BUDGETS)
def test_width_budgets_match_the_table(
    kind: str,
    e_layers: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_params: int,
) -> None:
    _assert_budget(kind, e_layers, d_model, n_heads, d_ff, n_params)


@pytest.mark.parametrize("e_layers", [1, 3])
def test_hierarchical_still_rejects_other_depths(e_layers: int) -> None:
    with pytest.raises(ValueError, match="exactly two layers"):
        _build("hierarchical", e_layers, 64, 4, 256)
