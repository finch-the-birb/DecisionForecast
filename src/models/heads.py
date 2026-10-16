"""Forecasting head architectures and factory for LTSF models.

Supports:
- RegressionHead: 2-layer MLP projection for point return/price forecasting.
- QuantileHead: Multi-quantile projection outputting shape [B, horizon, num_quantiles].
- LinearHead: Canonical 1-layer affine projection head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """2-layer MLP projection head for LTSF point forecasting.

    Structure:
        Flatten -> Linear(head_nf, hidden) -> GELU -> Dropout(dropout) -> Linear(hidden, horizon)
    """

    def __init__(
        self,
        head_nf: int,
        horizon: int,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head_nf = int(head_nf)
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, self.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects token sequence [B, N_tokens, d_model] -> [B, horizon]."""
        return self.net(x)


class QuantileHead(nn.Module):
    """Multi-quantile projection head for Distributional LTSF uncertainty forecasting.

    Structure:
        Flatten -> Linear(head_nf, hidden) -> GELU -> Dropout(dropout) -> Linear(hidden, horizon * K)
        followed by reshape to [B, horizon, K].
    """

    def __init__(
        self,
        head_nf: int,
        horizon: int,
        hidden: int = 128,
        dropout: float = 0.1,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> None:
        super().__init__()
        self.head_nf = int(head_nf)
        self.horizon = int(horizon)
        self.hidden = int(hidden)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.n_quantiles = len(self.quantiles)
        if self.n_quantiles < 1:
            raise ValueError("quantiles tuple must contain at least one quantile level")

        self.net = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, self.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.horizon * self.n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects token sequence [B, N_tokens, d_model] -> [B, horizon, K]."""
        out = self.net(x).view(x.size(0), self.horizon, self.n_quantiles)
        # Гарантия монотонности q_0.10 <= q_0.50 <= q_0.90 вдоль последнего измерения
        out, _ = torch.sort(out, dim=-1)
        return out


class LinearHead(nn.Module):
    """Canonical 1-layer affine projection head."""

    def __init__(
        self,
        head_nf: int,
        horizon: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head_nf = int(head_nf)
        self.horizon = int(horizon)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.head_nf, self.horizon)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects token sequence [B, N_tokens, d_model] -> [B, horizon]."""
        return self.dropout(self.linear(self.flatten(x)))


def build_head(
    head_type: str,
    head_nf: int,
    horizon: int,
    hidden: int = 128,
    dropout: float = 0.1,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    **kwargs,
) -> nn.Module:
    """Factory creating forecasting head based on head_type."""
    del kwargs
    kind = str(head_type).lower()
    if kind in ("quantile", "multi_quantile", "distributional"):
        return QuantileHead(
            head_nf=head_nf,
            horizon=horizon,
            hidden=hidden,
            dropout=dropout,
            quantiles=quantiles,
        )
    if kind in ("mlp", "regression"):
        return RegressionHead(
            head_nf=head_nf,
            horizon=horizon,
            hidden=hidden,
            dropout=dropout,
        )
    if kind == "linear":
        return LinearHead(
            head_nf=head_nf,
            horizon=horizon,
            dropout=dropout,
        )
    raise ValueError(
        f"Unknown head_type: {head_type!r}; expected 'quantile' | 'mlp' | 'linear'"
    )
