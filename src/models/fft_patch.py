"""Log-amplitude rFFT on each price patch. The DC bin is removed by centering."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchFFT(nn.Module):
    """Map ``patch_close`` ``[B, K, P]`` to six log amplitudes ``[B, K, 6]``.

    ``P`` is 12 in the forecast protocol. Centering removes the zero-frequency
    bin, ``rfft`` keeps bins ``0..6``, and the output is bins ``1..6``.
    """

    def forward(self, patch_close: torch.Tensor) -> torch.Tensor:
        if patch_close.dim() != 3:
            raise ValueError(f"patch_close must be [B, K, P], got {tuple(patch_close.shape)}")
        if patch_close.size(-1) < 12:
            raise ValueError(f"patch length must be at least 12, got {patch_close.size(-1)}")
        centered = patch_close - patch_close.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1)
        return torch.log(spectrum[:, :, 1:7].abs() + 1e-6)
