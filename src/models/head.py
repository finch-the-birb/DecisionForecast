"""Modular prediction heads for patch-based and sequence models."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """Pooling + Linear / MLP head.

    ``pool`` chooses how patch tokens ``[B, N, D]`` become a vector before the
    linear map to horizon ``H``:

    - ``mean``: ``enc_p.mean(dim=1)`` — spatial ensemble / noise smoother (default)
    - ``last``: ``enc_p[:, -1, :]`` — most recent patch (already mixed by self-attn)
    - ``global``: ``g_en[:, 0, :]`` — TimeXer ``G_en`` (sees patches + optional text exo)
    - ``flatten``: ``enc_p.reshape(B, N*D)`` — PatchTST-style; wide, easy to overfit

    Compact pools map ``d_model (+ extra) → H``. Flatten needs ``n_patches``.
    """

    def __init__(
        self,
        d_model: int,
        horizon: int,
        pool: str = "mean",  # "mean" | "last" | "global" | "flatten"
        n_patches: int | None = None,
        head_type: str = "linear",  # "linear" | "mlp"
        head_hidden: int = 128,
        dropout: float = 0.0,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.pool = str(pool)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.head_type = str(head_type)
        self.extra_dim = int(extra_dim)
        self.n_patches = int(n_patches) if n_patches is not None else None

        if self.pool == "flatten":
            if self.n_patches is None:
                raise ValueError("n_patches is required for flatten pool")
            in_dim = self.n_patches * self.d_model + self.extra_dim
        elif self.pool in {"mean", "last", "global"}:
            in_dim = self.d_model + self.extra_dim
        else:
            raise ValueError(
                f"Unknown pool: {self.pool!r}; expected 'mean' | 'last' | 'global' | 'flatten'"
            )

        if self.head_type == "linear":
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, self.horizon),
            )
        elif self.head_type == "mlp":
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, self.horizon),
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type!r}; expected 'linear' | 'mlp'")

    def forward(
        self,
        patches: torch.Tensor,
        g_en: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # patches: [B, N, D], g_en: [B, 1, D]
        if self.pool == "mean":
            pooled = patches.mean(dim=1)
        elif self.pool == "last":
            pooled = patches[:, -1, :]
        elif self.pool == "global":
            if g_en is None:
                raise ValueError("global token required for pool='global'")
            pooled = g_en[:, 0, :]
        elif self.pool == "flatten":
            pooled = patches.reshape(patches.size(0), -1)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        if extra is not None:
            pooled = torch.cat([pooled, extra], dim=-1)
        return self.net(pooled)
