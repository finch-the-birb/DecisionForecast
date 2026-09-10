"""H3 faithfulness: knockout / shuffle of prototypes and text on the test split."""

from __future__ import annotations

import json
import logging
from pathlib import Path

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


def format_h3_table(reports: dict[str, dict]) -> str:
    """Markdown H3 table. Pass decoded faithfulness JSON objects keyed by model id."""
    order = [m for m in ("a", "b", "c1") if m in reports]
    labels = {"a": "A", "b": "B", "c1": "C1"}
    lines = [
        "| model | mse | proto_zero Δmse | proto_shuffle Δmse | text_zero Δmse | text_shuffle Δmse |",
        "|-------|-----|-----------------|--------------------|----------------|-------------------|",
    ]
    for key in order:
        variants = reports[key]["variants"]
        full_mse = variants["full"]["mse"]
        lines.append(
            "| {lab} | {mse:.4f} | {pz:+.4f} | {ps:+.4f} | {tz:+.4f} | {ts:+.4f} |".format(
                lab=labels.get(key, key),
                mse=full_mse,
                pz=variants["proto_zero"]["delta_mse"],
                ps=variants["proto_shuffle"]["delta_mse"],
                tz=variants["text_zero"]["delta_mse"],
                ts=variants["text_shuffle"]["delta_mse"],
            )
        )
    return "\n".join(lines) + "\n"
