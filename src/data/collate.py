from __future__ import annotations

import torch


def forecast_collate(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "text": torch.stack([item["text"] for item in batch], dim=0),
        "ticker": [item["ticker"] for item in batch],
        "end_idx": torch.tensor([item["end_idx"] for item in batch], dtype=torch.long),
        "end_date": [item["end_date"] for item in batch],
        "raw_text": [item["raw_text"] for item in batch],
    }
