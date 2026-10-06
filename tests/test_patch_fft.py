"""Patch rFFT returns six log amplitudes and ignores a constant price shift."""

from __future__ import annotations

import torch

from src.models.fft_patch import PatchFFT


def test_patch_fft_shape_and_shift_invariance() -> None:
    torch.manual_seed(0)
    module = PatchFFT()
    patch = torch.randn(3, 9, 12)
    spectrum = module(patch)
    assert spectrum.shape == (3, 9, 6)
    assert torch.isfinite(spectrum).all()
    shifted = module(patch + 4.5)
    torch.testing.assert_close(spectrum, shifted)
