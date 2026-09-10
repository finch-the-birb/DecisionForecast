from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.prototypes import PrototypeLosses, PrototypeModule


@dataclass
class ModelAOutput:
    pred: torch.Tensor
    proto_losses: PrototypeLosses
    segments: torch.Tensor


class ConvSegmentEncoder(nn.Module):
    """1D-CNN segment encoder (TimeXL-style) producing overlapping patch tokens."""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        channels: list[int],
        kernel_size: int,
        patch_len: int,
        patch_stride: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(c_out),
                ]
            )
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=patch_len, stride=patch_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> segments [B, S, D]
        x = x.transpose(1, 2)
        h = self.conv(x)
        h = self.proj(h)
        return h.transpose(1, 2)


class TimeXLModelA(nn.Module):
    """Model A: 1D-CNN + prototype-before-attn + late text fusion."""

    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int,
        n_prototypes: int,
        d_min: float,
        patch_len: int,
        patch_stride: int,
        cnn_channels: list[int],
        cnn_kernel: int,
        text_dim: int,
        text_hidden: int,
        head_hidden: int,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoder = ConvSegmentEncoder(
            in_channels=n_features,
            d_model=d_model,
            channels=cnn_channels,
            kernel_size=cnn_kernel,
            patch_len=patch_len,
            patch_stride=patch_stride,
        )
        self.proto = PrototypeModule(n_prototypes, d_model, d_min)
        self.inject = nn.Linear(d_model, d_model)
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, text_hidden),
            nn.ReLU(),
            nn.Linear(text_hidden, d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, horizon),
        )

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        proto_mode: str = "none",
        text_mode: str = "none",
        ablation_generator: torch.Generator | None = None,
    ) -> ModelAOutput:
        segments = self.encoder(x)
        proto_mix, proto_losses = self.proto(segments)
        if proto_mode != "zero":
            proto_mix = apply_feature_ablation(proto_mix, proto_mode, ablation_generator)
            segments = segments + self.inject(proto_mix)
        ts_repr = segments.mean(dim=1)
        text_repr = apply_feature_ablation(
            self.text_mlp(text), text_mode, ablation_generator
        )
        fused = torch.cat([ts_repr, text_repr], dim=-1)
        pred = self.head(fused)
        return ModelAOutput(pred=pred, proto_losses=proto_losses, segments=segments)

    def compute_loss(
        self,
        output: ModelAOutput,
        target: torch.Tensor,
        lambda_c: float,
        lambda_e: float,
        lambda_d: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_pred = nn.functional.mse_loss(output.pred, target)
        pl = output.proto_losses
        total = (
            l_pred
            + lambda_c * pl.l_c
            + lambda_e * pl.l_e
            + lambda_d * pl.l_d
        )
        metrics = {
            "loss": float(total.detach()),
            "l_pred": float(l_pred.detach()),
            "l_c": float(pl.l_c.detach()),
            "l_e": float(pl.l_e.detach()),
            "l_d": float(pl.l_d.detach()),
        }
        return total, metrics
