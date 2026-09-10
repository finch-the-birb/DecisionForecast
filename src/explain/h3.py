"""Write H3 projection + faithfulness artifacts for one trained model."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import FNSPIDForecastDataset
from src.explain.faithfulness import save_faithfulness_ablation
from src.explain.projection import save_projection_examples

log = logging.getLogger(__name__)


def h3_models(cfg: DictConfig) -> set[str]:
    return {str(m) for m in cfg.explain.get("models", ["a", "b", "c1"])}


def write_h3_artifacts(
    model: torch.nn.Module,
    train_ds: FNSPIDForecastDataset,
    test_loader: DataLoader,
    cfg: DictConfig,
    out_dir: Path,
) -> dict[str, Path]:
    """Projection for any proto model; faithfulness for A/B/C1 (cfg.explain.models)."""
    explain_dir = out_dir / "explain"
    paths: dict[str, Path] = {}
    if hasattr(model, "proto"):
        proj = save_projection_examples(
            model,
            train_ds,
            cfg,
            explain_dir,
            n_examples=int(cfg.explain.get("n_projection", 3)),
            max_bank_segments=int(cfg.explain.get("max_bank_segments", 5000)),
        )
        paths["projection"] = proj
        log.info("Saved projection examples to %s", proj)
    if bool(cfg.explain.get("enabled", True)) and str(cfg.model.name) in h3_models(cfg):
        faith = save_faithfulness_ablation(
            model,
            test_loader,
            cfg,
            explain_dir,
            seed=int(cfg.train.seed),
        )
        paths["faithfulness"] = faith
        log.info("Saved faithfulness ablation to %s", faith)
    return paths
