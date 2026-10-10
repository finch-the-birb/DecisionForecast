"""Canonical 3-Layer Hierarchical TimeXer with 2-Layer MLP Projection Head.

Replaces the affine Linear head with a 2-layer MLP projection:
Flatten -> Linear(head_nf, head_hidden) -> GELU -> Dropout(head_dropout) -> Linear(head_hidden, horizon)
providing mathematical degrees of freedom to generate non-linear trajectory shapes (curvature,
turns, and market regime transitions) over the forecast horizon.
"""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn

from src.models.timexer_hierarchical import TimeXerHierarchical


class TimeXerHierarchicalMLP(TimeXerHierarchical):
    """3-layer hierarchical TimeXer with non-linear MLP projection head."""

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
            **kwargs,
        )
        self.head_type = str(head_type)
        self.head_hidden = int(head_hidden)
        self.head_dropout = float(head_dropout)

        # 2-layer MLP head for non-linear return forecasting:
        # Flatten -> Linear(head_nf, head_hidden) -> GELU -> Dropout -> Linear(head_hidden, horizon)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, self.head_hidden),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_hidden, self.horizon),
        )
