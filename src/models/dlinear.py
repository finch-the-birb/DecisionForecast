"""DLinear (Zeng et al.) — compact reimplementation of thuml Time-Series-Library DLinear.

Predicts the target channel over horizon H. Pipeline sanity control, not a hypothesis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.outputs import ModelOutput, compute_pred_loss


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        extra = self.kernel_size - 1 - pad
        end = x[:, -1:, :].repeat(1, extra, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    def __init__(
        self,
        seq_len: int,
        horizon: int,
        n_features: int,
        target_idx: int,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        if not 0 <= target_idx < n_features:
            raise ValueError(f"target_idx={target_idx} not in [0, {n_features})")
        self.target_idx = target_idx
        self.moving_avg = _MovingAvg(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.linear_trend = nn.Linear(seq_len, horizon)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor | None = None,
        text_seq: torch.Tensor | None = None,
        **_kwargs,
    ) -> ModelOutput:
        del text, text_seq, _kwargs
        trend = self.moving_avg(x)
        seasonal = x - trend
        # [B, T, C] -> [B, C, T] -> linear on time -> [B, C, H] -> [B, H, C]
        seas = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        tren = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        y = seas + tren
        pred = y[:, :, self.target_idx]
        dummy_seg = x.new_zeros(x.size(0), 1, 1)
        return ModelOutput(pred=pred, proto_losses=None, segments=dummy_seg)

    def compute_loss(
        self,
        output: ModelOutput,
        target: torch.Tensor,
        lambda_c: float,
        lambda_e: float,
        lambda_d: float,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_pred_loss(output, target, lambda_c, lambda_e, lambda_d, **kwargs)
