"""Load a checkpoint and write H3 projection + faithfulness (no training)."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.collate import forecast_collate
from src.data.dataset import build_datasets
from src.explain.h3 import write_h3_artifacts
from src.training.train import build_model
from src.utils.device import log_torch_device, resolve_device
from src.utils.seed import set_seed

log = logging.getLogger(__name__)


def run_explain(cfg: DictConfig) -> dict[str, Path]:
    set_seed(int(cfg.train.seed))
    device = resolve_device(str(cfg.train.device))
    log_torch_device(device, role="explain")

    ckpt = cfg.explain.get("checkpoint")
    if not ckpt:
        raise ValueError(
            "explain.checkpoint is required for src.explain.run "
            "(path to best.pt). Or run src.training.train which writes H3 after test."
        )
    from hydra.utils import to_absolute_path

    ckpt_path = Path(to_absolute_path(str(ckpt)))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    train_ds, _val_ds, test_ds = build_datasets(cfg, device=device)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=forecast_collate,
    )
    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    log.info("Loaded checkpoint %s", ckpt_path)

    out_dir = Path(cfg.paths.output_dir)
    paths = write_h3_artifacts(model, train_ds, test_loader, cfg, out_dir)
    if "faithfulness" not in paths:
        raise RuntimeError(
            f"H3 faithfulness skipped for model={cfg.model.name}; "
            f"expected one of {sorted(cfg.explain.models)}"
        )
    return paths


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    run_explain(cfg)


if __name__ == "__main__":
    main()
