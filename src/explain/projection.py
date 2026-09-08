from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.collate import forecast_collate
from src.data.dataset import FNSPIDForecastDataset
from src.models.timexl_a import TimeXLModelA


@torch.no_grad()
def save_projection_examples(
    model: TimeXLModelA,
    dataset: FNSPIDForecastDataset,
    cfg: DictConfig,
    out_dir: Path,
    n_examples: int = 3,
) -> Path:
    """Save nearest train segments for prototypes (Model A, H3 prep)."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=forecast_collate)
    segment_bank: list[torch.Tensor] = []
    meta: list[dict] = []
    for batch in loader:
        x = batch["x"].to(device)
        text = batch["text"].to(device)
        out = model(x, text)
        segs = out.segments
        segment_bank.append(segs.reshape(-1, segs.size(-1)).cpu())
        for i in range(len(batch["ticker"])):
            for seg_i in range(int(segs.size(1))):
                meta.append(
                    {
                        "ticker": batch["ticker"][i],
                        "end_date": batch["end_date"][i],
                        "end_idx": int(batch["end_idx"][i]),
                        "segment": seg_i,
                    }
                )
        if sum(t.size(0) for t in segment_bank) > 5000:
            break

    bank = torch.cat(segment_bank, dim=0)
    idx, dist = model.proto.project(bank.to(device))
    examples = []
    for proto_i in range(min(n_examples, model.proto.prototypes.size(0))):
        j = int(idx[proto_i].item())
        examples.append(
            {
                "prototype_id": proto_i,
                "nearest_segment_index": j,
                "distance": float(dist[proto_i].item()),
                "meta": meta[j] if j < len(meta) else {},
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "projection_examples_a.json"
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    return path
