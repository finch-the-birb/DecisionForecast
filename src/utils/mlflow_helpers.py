from __future__ import annotations

from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf


def flatten_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Resolve and flatten Hydra config for MLflow params."""
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    return _flatten("", container)


def _flatten(prefix: str, obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(path, value))
    elif isinstance(obj, list):
        out[prefix] = ",".join(str(v) for v in obj)
    else:
        out[prefix] = obj
    return out


def log_cfg_params(cfg: DictConfig) -> None:
    flat = flatten_cfg(cfg)
    for key, value in flat.items():
        mlflow.log_param(key, value)
