from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.collate import forecast_collate
from src.data.dataset import FNSPIDForecastDataset
from src.explain.bank import collect_segment_bank

log = logging.getLogger(__name__)


def _seg_dates(
    dataset: FNSPIDForecastDataset,
    ticker: str,
    end_idx: int,
    seg_i: int,
    lookback: int,
    patch_len: int,
    patch_stride: int,
) -> tuple[str | None, str | None]:
    ts = dataset.series.get(ticker)
    if ts is None:
        return None, None
    start_idx = int(end_idx) - int(lookback)
    i0 = start_idx + int(seg_i) * int(patch_stride)
    i1 = i0 + int(patch_len) - 1
    dates = ts.dates
    if i0 < 0 or i1 >= len(dates):
        return None, None
    return str(pd.Timestamp(dates.iloc[i0]).date()), str(pd.Timestamp(dates.iloc[i1]).date())


@torch.no_grad()
def save_projection_examples(
    model: torch.nn.Module,
    dataset: FNSPIDForecastDataset,
    cfg: DictConfig,
    out_dir: Path,
    n_examples: int = 3,
    max_bank_segments: int = 5000,
) -> Path:
    """Save top-k nearest train segments per prototype (pre-injection space)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=forecast_collate)
    bank, meta = collect_segment_bank(model, loader, max_segments=max_bank_segments)
    device = next(model.parameters()).device
    z = bank.to(device)
    p = model.proto.prototypes.detach()
    dist = torch.cdist(p, z, p=2)
    k = int(cfg.explain.get("k_nearest", n_examples))
    k = max(1, min(k, z.size(0)))
    lookback = int(cfg.data.lookback_T)
    patch_len = int(cfg.data.patch_len)
    patch_stride = int(cfg.data.patch_stride)

    n_proto = int(p.size(0))
    nn_idx = dist.argmin(dim=1)
    nearest_keys = []
    for proto_i in range(n_proto):
        j = int(nn_idx[proto_i].item())
        m = meta[j] if j < len(meta) else {}
        nearest_keys.append((m.get("ticker"), m.get("end_date"), m.get("segment")))
    n_distinct = len(set(nearest_keys))
    if n_distinct <= 1:
        log.warning(
            "All prototypes project to one segment (n_distinct_nearest=%s)", n_distinct
        )
    log.info("n_distinct_nearest=%d n_prototypes=%d", n_distinct, n_proto)

    examples = []
    topk = dist.topk(k, dim=1, largest=False)
    for proto_i in range(n_proto):
        neighbors = []
        for rank in range(k):
            j = int(topk.indices[proto_i, rank].item())
            m = dict(meta[j]) if j < len(meta) else {}
            seg_start, seg_end = _seg_dates(
                dataset,
                str(m.get("ticker", "")),
                int(m.get("end_idx", 0)),
                int(m.get("segment", 0)),
                lookback,
                patch_len,
                patch_stride,
            )
            neighbors.append(
                {
                    "rank": rank + 1,
                    "nearest_segment_index": j,
                    "distance": float(topk.values[proto_i, rank].item()),
                    "ticker": m.get("ticker"),
                    "end_date": m.get("end_date"),
                    "end_idx": m.get("end_idx"),
                    "segment": m.get("segment"),
                    "seg_start_date": seg_start,
                    "seg_end_date": seg_end,
                }
            )
        examples.append({"prototype_id": proto_i, "neighbors": neighbors})

    out_dir.mkdir(parents=True, exist_ok=True)
    name = str(cfg.model.name)
    path = out_dir / f"projection_examples_{name}.json"
    payload = {
        "n_distinct_nearest": n_distinct,
        "n_prototypes": n_proto,
        "k_nearest": k,
        "prototypes": examples,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
