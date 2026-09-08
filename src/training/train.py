from __future__ import annotations

import logging
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.collate import forecast_collate
from src.data.dataset import build_datasets
from src.evaluation.metrics import compute_metrics
from src.explain.projection import save_projection_examples
from src.models.timexl_a import TimeXLModelA
from src.utils.mlflow_helpers import log_cfg_params
from src.utils.seed import set_seed

log = logging.getLogger(__name__)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_model(cfg: DictConfig) -> torch.nn.Module:
    if cfg.model.name != "a":
        raise NotImplementedError(
            f"Model '{cfg.model.name}' not implemented yet (Phase 2+). Use model=a."
        )
    n_features = len(cfg.data.features)
    return TimeXLModelA(
        n_features=n_features,
        horizon=int(cfg.data.horizon),
        d_model=int(cfg.model.d_model),
        n_prototypes=int(cfg.model.n_prototypes),
        d_min=float(cfg.model.d_min),
        patch_len=int(cfg.data.patch_len),
        patch_stride=int(cfg.data.patch_stride),
        cnn_channels=list(cfg.model.cnn.channels),
        cnn_kernel=int(cfg.model.cnn.kernel_size),
        text_dim=int(cfg.data.text.dim),
        text_hidden=int(cfg.model.text_mlp.hidden),
        head_hidden=int(cfg.model.head.hidden),
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        text = batch["text"].to(device)
        if isinstance(model, TimeXLModelA):
            out = model(x, text)
            pred = out.pred
        else:
            pred = model(x, text)
        preds.append(pred.cpu())
        targets.append(y.cpu())
    return compute_metrics(torch.cat(preds), torch.cat(targets))


def run_training(cfg: DictConfig) -> dict[str, float]:
    set_seed(int(cfg.train.seed))
    device = _resolve_device(str(cfg.train.device))
    log.info("Device: %s", device)

    train_ds, val_ds, test_ds = build_datasets(cfg)
    log.info(
        "Dataset sizes — train: %d, val: %d, test: %d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    if min(len(train_ds), len(val_ds), len(test_ds)) == 0:
        raise RuntimeError(
            "Empty train/val/test split after capping; check tickers, dates, and max_*_windows"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=forecast_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=forecast_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=forecast_collate,
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience_left = int(cfg.train.patience)
    lambda_c = float(cfg.model.loss.lambda_c)
    lambda_e = float(cfg.model.loss.lambda_e)
    lambda_d = float(cfg.model.loss.lambda_d)

    mlflow_enabled = bool(cfg.train.mlflow.enabled)
    if mlflow_enabled:
        mlflow.set_tracking_uri(str(cfg.train.mlflow.tracking_uri))
        mlflow.set_experiment(str(cfg.train.mlflow.experiment_name))

    with mlflow.start_run(run_name=str(cfg.experiment.name), nested=False) if mlflow_enabled else _nullcontext():
        if mlflow_enabled:
            log_cfg_params(cfg)
            mlflow.log_param("device", str(device))

        for epoch in range(1, int(cfg.train.epochs) + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                text = batch["text"].to(device)
                if n_batches == 0:
                    log.info(
                        "First batch text L2 mean=%.4f (0 means empty/zero embeddings)",
                        float(text.norm(dim=-1).mean()),
                    )
                optimizer.zero_grad(set_to_none=True)
                if isinstance(model, TimeXLModelA):
                    out = model(x, text)
                    loss, train_metrics = model.compute_loss(
                        out, y, lambda_c, lambda_e, lambda_d
                    )
                else:
                    raise NotImplementedError
                loss.backward()
                if cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                optimizer.step()
                epoch_loss += train_metrics["loss"]
                n_batches += 1

            if epoch % int(cfg.train.eval_every) == 0:
                val_metrics = evaluate(model, val_loader, device)
                avg_loss = epoch_loss / max(n_batches, 1)
                log.info(
                    "Epoch %d — train_loss=%.4f val_mse=%.4f val_mae=%.4f",
                    epoch,
                    avg_loss,
                    val_metrics["mse"],
                    val_metrics["mae"],
                )
                if mlflow_enabled:
                    mlflow.log_metrics(
                        {
                            "train_loss": avg_loss,
                            "val_mse": val_metrics["mse"],
                            "val_mae": val_metrics["mae"],
                        },
                        step=epoch,
                    )
                if val_metrics["mse"] < best_val:
                    best_val = val_metrics["mse"]
                    patience_left = int(cfg.train.patience)
                    torch.save(model.state_dict(), ckpt_dir / "best.pt")
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        log.info("Early stopping at epoch %d", epoch)
                        break

        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

        test_metrics = evaluate(model, test_loader, device)
        log.info("Test — mse=%.4f mae=%.4f", test_metrics["mse"], test_metrics["mae"])
        if mlflow_enabled:
            mlflow.log_metrics(
                {"test_mse": test_metrics["mse"], "test_mae": test_metrics["mae"]}
            )

        if isinstance(model, TimeXLModelA):
            proj_path = save_projection_examples(
                model,
                train_ds,
                cfg,
                Path(cfg.paths.output_dir) / "explain",
                n_examples=3,
            )
            log.info("Saved projection examples to %s", proj_path)
            if mlflow_enabled:
                mlflow.log_artifact(str(proj_path))

        OmegaConf.save(cfg, Path(cfg.paths.output_dir) / "config_resolved.yaml")
        return test_metrics


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log.info("Resolved experiment: %s", cfg.experiment.name)
    run_training(cfg)


if __name__ == "__main__":
    main()
