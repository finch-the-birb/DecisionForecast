from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.collate import forecast_collate
from src.data.dataset import build_datasets, log_text_coverage
from src.evaluation.metrics import compute_metrics
from src.explain.h3 import write_h3_artifacts
from src.models.dlinear import DLinear
from src.models.fusion import loggable_fusion
from src.models.timexer_b import TimeXerB
from src.models.timexer_plain import TimeXerPlain
from src.models.timexl_a import TimeXLModelA
from src.models.timexl_integration import TimeXerFusionModel
from src.utils.device import log_cuda_memory, log_torch_device, resolve_device
from src.utils.mlflow_helpers import log_cfg_params
from src.utils.seed import set_seed

log = logging.getLogger(__name__)


def _resolve_device(device_cfg: str) -> torch.device:
    return resolve_device(device_cfg)


def build_model(cfg: DictConfig) -> torch.nn.Module:
    n_features = len(cfg.data.features)
    features = [str(f) for f in cfg.data.features]
    target = str(cfg.data.target)
    try:
        target_idx = features.index(target)
    except ValueError as exc:
        raise ValueError(f"target {target!r} not in features {features}") from exc
    name = str(cfg.model.name)
    if name == "a":
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
    if name == "dlinear":
        return DLinear(
            seq_len=int(cfg.data.lookback_T),
            horizon=int(cfg.data.horizon),
            n_features=n_features,
            target_idx=target_idx,
            kernel_size=int(cfg.model.get("kernel_size", 25)),
        )
    if name == "timexer_plain":
        return TimeXerPlain(
            n_features=n_features,
            horizon=int(cfg.data.horizon),
            d_model=int(cfg.model.d_model),
            n_heads=int(cfg.model.n_heads),
            e_layers=int(cfg.model.e_layers),
            patch_len=int(cfg.data.patch_len),
            patch_stride=int(cfg.data.patch_stride),
            dropout=float(cfg.model.dropout),
            d_ff=int(cfg.model.get("d_ff", 4 * int(cfg.model.d_model))),
            head_hidden=int(cfg.model.head.hidden),
        )
    if name == "b":
        return TimeXerB(
            n_features=n_features,
            horizon=int(cfg.data.horizon),
            d_model=int(cfg.model.d_model),
            n_prototypes=int(cfg.model.n_prototypes),
            d_min=float(cfg.model.d_min),
            n_heads=int(cfg.model.n_heads),
            e_layers=int(cfg.model.e_layers),
            patch_len=int(cfg.data.patch_len),
            patch_stride=int(cfg.data.patch_stride),
            dropout=float(cfg.model.dropout),
            text_dim=int(cfg.data.text.dim),
            text_hidden=int(cfg.model.text_mlp.hidden),
            head_hidden=int(cfg.model.head.hidden),
            fusion=cfg.model.fusion,
            d_ff=int(cfg.model.get("d_ff", 4 * int(cfg.model.d_model))),
        )
    if name in {"c0", "c1"}:
        return TimeXerFusionModel(
            n_features=n_features,
            seq_len=int(cfg.data.lookback_T),
            horizon=int(cfg.data.horizon),
            target_idx=target_idx,
            d_model=int(cfg.model.d_model),
            n_prototypes=int(cfg.model.n_prototypes),
            d_min=float(cfg.model.d_min),
            n_heads=int(cfg.model.n_heads),
            e_layers=int(cfg.model.e_layers),
            patch_len=int(cfg.data.patch_len),
            patch_stride=int(cfg.data.patch_stride),
            dropout=float(cfg.model.dropout),
            text_dim=int(cfg.data.text.dim),
            text_hidden=int(cfg.model.text_mlp.hidden),
            head_hidden=int(cfg.model.head.hidden),
            fusion=str(cfg.model.fusion),
            d_ff=int(cfg.model.get("d_ff", 4 * int(cfg.model.d_model))),
            use_norm=bool(cfg.model.get("use_norm", False)),
            use_prototypes=bool(cfg.model.get("use_prototypes", True)),
        )
    raise NotImplementedError(
        f"Model '{name}' not implemented. Use model=a, b, c0, c1, timexer_plain, or dlinear."
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
        text_seq = batch["text_seq"].to(device)
        out = model(x, text, text_seq=text_seq)
        pred = out.pred
        preds.append(pred.cpu())
        targets.append(y.cpu())
    return compute_metrics(torch.cat(preds), torch.cat(targets))


def run_training(cfg: DictConfig) -> dict[str, float]:
    set_seed(int(cfg.train.seed))
    device = _resolve_device(str(cfg.train.device))
    log_torch_device(device, role="train")

    train_ds, val_ds, test_ds = build_datasets(cfg, device=device)
    log.info(
        "Dataset sizes — train: %d, val: %d, test: %d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    coverage = log_text_coverage(
        {"train": train_ds, "val": val_ds, "test": test_ds},
        train_ds.series,
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
    param_device = next(model.parameters()).device
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters on %s n_params=%d", param_device, n_params)
    if device.type == "cuda" and param_device.type != "cuda":
        raise RuntimeError(f"Model parameters on {param_device}, expected {device}")
    log_cuda_memory("after model.to")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience_left = int(cfg.train.patience)
    loss_cfg = cfg.model.get("loss", {})
    lambda_c = float(loss_cfg.get("lambda_c", 0.0))
    lambda_e = float(loss_cfg.get("lambda_e", 0.0))
    lambda_d = float(loss_cfg.get("lambda_d", 0.0))

    mlflow_enabled = bool(cfg.train.mlflow.enabled)
    if mlflow_enabled:
        mlflow.set_tracking_uri(str(cfg.train.mlflow.tracking_uri))
        mlflow.set_experiment(str(cfg.train.mlflow.experiment_name))

    with mlflow.start_run(run_name=str(cfg.experiment.name), nested=False) if mlflow_enabled else _nullcontext():
        if mlflow_enabled:
            log_cfg_params(cfg)
            mlflow.log_param("device", str(device))
            mlflow.log_param("n_params", n_params)
            fusion_cfg = cfg.model.get("fusion")
            if fusion_cfg is not None:
                for key, value in loggable_fusion(fusion_cfg).items():
                    mlflow.log_param(f"fusion.{key}", value)
            if coverage:
                mlflow.log_metrics({k: float(v) for k, v in coverage.items()})

        for epoch in range(1, int(cfg.train.epochs) + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                text = batch["text"].to(device)
                text_seq = batch["text_seq"].to(device)
                if n_batches == 0:
                    log.info(
                        "First batch text L2 mean=%.4f (0 means empty/zero embeddings)",
                        float(text.norm(dim=-1).mean()),
                    )
                    log.info(
                        "First batch shapes x=%s text=%s text_seq=%s",
                        tuple(x.shape),
                        tuple(text.shape),
                        tuple(text_seq.shape),
                    )
                    log.info("First batch tensors on x=%s text=%s", x.device, text.device)
                    log_cuda_memory("first train batch")
                optimizer.zero_grad(set_to_none=True)
                out = model(x, text, text_seq=text_seq)
                if not hasattr(model, "compute_loss"):
                    raise NotImplementedError(f"{type(model).__name__} has no compute_loss")
                loss, train_metrics = model.compute_loss(
                    out, y, lambda_c, lambda_e, lambda_d
                )
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
        log.info(
            "METRICS_ROW model=%s horizon=%s mse=%.4f mae=%.4f",
            cfg.model.name,
            cfg.data.horizon,
            test_metrics["mse"],
            test_metrics["mae"],
        )
        log.info(
            "H1_ROW model=%s horizon=%s mse=%.4f mae=%.4f",
            cfg.model.name,
            cfg.data.horizon,
            test_metrics["mse"],
            test_metrics["mae"],
        )
        metrics_path = Path(cfg.paths.output_dir) / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "model": str(cfg.model.name),
                    "horizon": int(cfg.data.horizon),
                    "mse": test_metrics["mse"],
                    "mae": test_metrics["mae"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if mlflow_enabled:
            mlflow.log_metrics(
                {"test_mse": test_metrics["mse"], "test_mae": test_metrics["mae"]}
            )
            mlflow.log_artifact(str(metrics_path))

        h3_paths = write_h3_artifacts(
            model,
            train_ds,
            test_loader,
            cfg,
            Path(cfg.paths.output_dir),
        )
        if mlflow_enabled:
            for path in h3_paths.values():
                mlflow.log_artifact(str(path))

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
