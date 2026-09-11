"""Modular prediction heads for patch-based and sequence models."""

from __future__ import annotations

import torch
import torch.nn as nn


class FlattenHead(nn.Module):
    """Canonical PatchTST / Time-Series-Library style head: flatten patches -> Linear / MLP.

    Preserves temporal ordering of patches rather than averaging them into a single token.
    Supports optional extra features (e.g. late text embeddings concatenated at the head).
    """

    def __init__(
        self,
        n_patches: int,
        d_model: int,
        horizon: int,
        head_type: str = "linear",  # "linear" | "mlp"
        head_hidden: int = 128,
        dropout: float = 0.0,
        extra_dim: int = 0,  # for late text in A and B
    ) -> None:
        super().__init__()
        self.n_patches = int(n_patches)
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.head_type = str(head_type)
        self.extra_dim = int(extra_dim)

        in_features = self.n_patches * self.d_model + self.extra_dim
        if self.head_type == "linear":
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, self.horizon),
            )
        elif self.head_type == "mlp":
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, self.horizon),
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type!r}; expected 'linear' | 'mlp'")

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, D] -> [B, N * D]
        b = x.size(0)
        flat = x.reshape(b, -1)
        if extra is not None:
            flat = torch.cat([flat, extra], dim=-1)
        return self.net(flat)
