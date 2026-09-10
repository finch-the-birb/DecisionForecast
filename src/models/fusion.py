"""Read and assert nested ``cfg.model.fusion`` flags (one text entry point)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def fusion_map(fusion: Any) -> dict[str, Any]:
    """Return a plain dict of fusion flags. Nested mapping required (not a string)."""
    if fusion is None:
        raise AssertionError("cfg.model.fusion is required")
    if isinstance(fusion, (DictConfig, ListConfig)) or OmegaConf.is_config(fusion):
        raw = OmegaConf.to_container(fusion, resolve=True)
    elif isinstance(fusion, Mapping):
        raw = dict(fusion)
    else:
        raise AssertionError(
            "cfg.model.fusion must be a mapping with kind/text_at_head/...; "
            f"got {type(fusion).__name__}: {fusion!r}"
        )
    if not isinstance(raw, dict):
        raise AssertionError(
            f"cfg.model.fusion must be a mapping, got {type(raw).__name__}: {raw!r}"
        )
    return raw


def assert_fusion(fusion: Any, **expected: Any) -> dict[str, Any]:
    """Assert selected flags.

    ``set`` expected values mean membership. ``list`` expected values mean exact
    sequence (e.g. ``inject_layers=[0]``).
    """
    flags = fusion_map(fusion)
    for key, want in expected.items():
        got: Any = flags.get(key)
        if isinstance(got, ListConfig):
            got = list(got)
        if isinstance(want, (set, frozenset)):
            if got not in want:
                raise AssertionError(f"fusion.{key}={got!r}, expected one of {set(want)}")
        elif isinstance(want, list):
            if list(got) != list(want):
                raise AssertionError(f"fusion.{key}={got!r}, expected {want!r}")
        elif got != want:
            raise AssertionError(f"fusion.{key}={got!r}, expected {want!r}")
    return flags


def loggable_fusion(fusion: Any) -> dict[str, Any]:
    """Flatten fusion flags for MLflow. String ``fusion: late`` logs as kind."""
    try:
        flags = fusion_map(fusion)
    except AssertionError:
        return {"kind": fusion}
    out: dict[str, Any] = {}
    for key, value in flags.items():
        if isinstance(value, (list, tuple)):
            out[key] = ",".join(str(v) for v in value)
        else:
            out[key] = value
    return out
