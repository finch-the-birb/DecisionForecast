"""Collect pre-injection segment banks for prototype init and H3 projection."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_segment_bank(
    model: torch.nn.Module,
    loader: DataLoader,
    max_segments: int = 5000,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Return (bank [N,D], per-row meta) from ``output.segments`` (pre-injection)."""
    model.eval()
    device = next(model.parameters()).device
    chunks: list[torch.Tensor] = []
    meta: list[dict] = []
    for b_i, batch in enumerate(loader):
        if max_batches is not None and b_i >= int(max_batches):
            break
        x = batch["x"].to(device)
        text = batch["text"].to(device)
        text_seq = batch["text_seq"].to(device)
        out = model(x, text, text_seq=text_seq)
        segs = out.segments
        chunks.append(segs.reshape(-1, segs.size(-1)).detach().cpu())
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
        if sum(t.size(0) for t in chunks) >= max_segments:
            break
    if not chunks:
        raise RuntimeError("empty segment bank")
    bank = torch.cat(chunks, dim=0)
    if bank.size(0) > max_segments:
        bank = bank[:max_segments]
        meta = meta[:max_segments]
    return bank, meta
