"""Canonical 3-Layer Hierarchical TimeXer with Modular Forecasting Head.

Supports both point forecasting (MLP / Linear regression) and
distributional uncertainty forecasting (Multi-Quantile Pinball head)
via `build_head`.
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from src.models.heads import build_head
from src.models.timexer_hierarchical import TimeXerHierarchical


class TimeXerHierarchicalMLP(TimeXerHierarchical):
    """3-layer hierarchical TimeXer with modular forecasting head."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        e_layers: int = 3,
        patch_len: int = 16,
        patch_stride: int = 12,
        dropout: float = 0.1,
        text_dim: int = 15,
        fusion: Any = None,
        n_ts_features: int = 30,
        close_idx: int = 0,
        use_prototypes: bool = False,
        n_prototypes: int = 10,
        d_min: float = 0.5,
        d_ff: int | None = None,
        head_type: str = "mlp",
        head_hidden: int = 128,
        head_dropout: float = 0.1,
        head_pool: str = "last",
        fft_mode: str = "bridge",
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        **kwargs,
    ) -> None:
        super().__init__(
            n_features=n_features,
            seq_len=seq_len,
            horizon=horizon,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            patch_len=patch_len,
            patch_stride=patch_stride,
            dropout=dropout,
            text_dim=text_dim,
            fusion=fusion,
            n_ts_features=n_ts_features,
            close_idx=close_idx,
            use_prototypes=use_prototypes,
            n_prototypes=n_prototypes,
            d_min=d_min,
            d_ff=d_ff,
            head_type=head_type,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
            head_pool=head_pool,
            fft_mode=fft_mode,
        )
        del kwargs
        self.head_type = str(head_type)
        self.head_hidden = int(head_hidden)
        self.head_dropout = float(head_dropout)
        self.quantiles = tuple(float(q) for q in quantiles) if quantiles is not None else (0.1, 0.5, 0.9)

        # Build modular forecasting head (MLP regression or multi-quantile tunnel)
        self.head = build_head(
            head_type=self.head_type,
            head_nf=self.head_nf,
            horizon=self.horizon,
            hidden=self.head_hidden,
            dropout=self.head_dropout,
            quantiles=self.quantiles,
        )
