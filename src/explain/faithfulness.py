"""H3 faithfulness: knockout / shuffle of prototypes and text on the test split."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics

log = logging.getLogger(__name__)

# (json_key, proto_mode, text_mode)
_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("full", "none", "none"),
    ("proto_zero", "zero", "none"),
    ("proto_shuffle", "shuffle", "none"),
    ("text_zero", "none", "zero"),
    ("text_shuffle", "none", "shuffle"),
)


@torch.no_grad()
def _eval_variant(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    proto_mode: str,
    text_mode: str,
    generator: torch.Generator,
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        text = batch["text"].to(device)
        text_seq = batch["text_seq"].to(device)
        out = model(
            x,
            text,
            text_seq=text_seq,
            proto_mode=proto_mode,
            text_mode=text_mode,
            ablation_generator=generator,
        )
        pred = out.pred
        preds.append(pred.cpu())
        targets.append(y.cpu())
    return compute_metrics(torch.cat(preds), torch.cat(targets))


def save_faithfulness_ablation(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: DictConfig,
    out_dir: Path,
    seed: int,
) -> Path:
    """Write faithfulness_{model}.json with mse/mae and Δ vs full (ablated − full)."""
    device = next(model.parameters()).device
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    variants: dict[str, dict[str, float]] = {}
    full: dict[str, float] | None = None
    for key, proto_mode, text_mode in _VARIANTS:
        metrics = _eval_variant(
            model, loader, device, proto_mode, text_mode, generator
        )
        row = {"mse": metrics["mse"], "mae": metrics["mae"]}
        if key == "full":
            full = row
        else:
            assert full is not None
            row["delta_mse"] = metrics["mse"] - full["mse"]
            row["delta_mae"] = metrics["mae"] - full["mae"]
        variants[key] = row
        log.info(
            "H3_ROW model=%s variant=%s mse=%.4f mae=%.4f delta_mse=%s delta_mae=%s",
            cfg.model.name,
            key,
            row["mse"],
            row["mae"],
            f"{row['delta_mse']:+.4f}" if "delta_mse" in row else "n/a",
            f"{row['delta_mae']:+.4f}" if "delta_mae" in row else "n/a",
        )
        if mlflow.active_run() is not None:
            metrics_to_log = {f"h3_{key}_mse": row["mse"], f"h3_{key}_mae": row["mae"]}
            if "delta_mse" in row:
                metrics_to_log[f"h3_{key}_delta_mse"] = row["delta_mse"]
                metrics_to_log[f"h3_{key}_delta_mae"] = row["delta_mae"]
            mlflow.log_metrics(metrics_to_log)

    payload = {
        "model": str(cfg.model.name),
        "horizon": int(cfg.data.horizon),
        "n_test": int(len(loader.dataset)),  # type: ignore[arg-type]
        "variants": variants,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"faithfulness_{cfg.model.name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _mean_std(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, var**0.5, n


def _fmt_mean_std(xs: list[float], signed: bool = False) -> str:
    mean, std, n = _mean_std(xs)
    core = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    if n <= 1:
        return core
    return f"{core} ± {std:.4f} (n={n})"


def _as_report_list(value: dict | list) -> list[dict]:
    if isinstance(value, list):
        return value
    return [value]


def format_h3_table(reports: dict[str, dict | list[dict]]) -> str:
    """Markdown H3 table. Values are one JSON object or a list (one per seed)."""
    order = [m for m in ("a", "b", "c0", "c1") if m in reports]
    labels = {"a": "A", "b": "B", "c0": "C0", "c1": "C1"}
    lines = [
        "| model | mse | proto_zero Δmse | proto_shuffle Δmse | text_zero Δmse | text_shuffle Δmse |",
        "|-------|-----|-----------------|--------------------|----------------|-------------------|",
    ]
    for key in order:
        reps = _as_report_list(reports[key])
        full, pz, ps, tz, ts = [], [], [], [], []
        for rep in reps:
            variants = rep["variants"]
            full.append(variants["full"]["mse"])
            pz.append(variants["proto_zero"]["delta_mse"])
            ps.append(variants["proto_shuffle"]["delta_mse"])
            tz.append(variants["text_zero"]["delta_mse"])
            ts.append(variants["text_shuffle"]["delta_mse"])
        lines.append(
            f"| {labels.get(key, key)} | {_fmt_mean_std(full)} | {_fmt_mean_std(pz, True)} | "
            f"{_fmt_mean_std(ps, True)} | {_fmt_mean_std(tz, True)} | {_fmt_mean_std(ts, True)} |"
        )
    return "\n".join(lines) + "\n"
