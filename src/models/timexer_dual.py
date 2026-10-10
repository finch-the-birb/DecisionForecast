"""Canonical Dual-token TimeXer (NeurIPS 2024 / TSLib) with optional FFT integration.

Endogenous series [B, C_endo, L] is patched via Linear(patch_len, d_model) + PositionalEmbedding.
Optional FFT integration (fft_mode):
- "patch": Local additive Frequency Embedding (PatchFFT -> Linear(6, d_model)) added to price patches.
- "bridge": Global window rFFT (first 6 harmonic log-amplitudes) projected to tokens and concatenated to cross_ts.
- "none": Pure time series without FFT.
Two learnable global tokens (G_ts and G_text) are appended to patches: [P_1..P_N; G_ts; G_text].
In each encoder layer:
1. Full Self-Attention over all N+2 tokens (patches exchange time; G_ts and G_text see patches and each other).
2. Slicing: x_patches = x[:, :-2, :], g_ts = x[:, -2, :], g_text = x[:, -1, :].
3. Parallel Cross-Attention:
   G_ts cross-attends to inverted variate ts_exo [B, C_ts, d_model] (optionally + 6 FFT tokens).
   G_text cross-attends to projected text_exo [B, C_text, d_model].
   (Price patches are completely isolated from direct exogenous cross-attention).
4. Concatenation of patches, updated G_ts, and updated G_text.
5. Conv1d(1x1) -> GELU -> Conv1d(1x1) FFN over all N+2 tokens.
6. FlattenHead unfolds target variable tokens -> Linear -> horizon H=7.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.fft_patch import PatchFFT
from src.models.fusion import assert_fusion
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.prototypes import PrototypeLosses
from src.models.timexer_backbone import n_patches
from src.models.timexl_integration import PrototypeResidual


def dual_attention_mask(n_tokens_patches: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive mask [K+2, K+2] (kept for backward compatibility)."""
    width = int(n_tokens_patches) + 2
    mask = torch.zeros(width, width, device=device, dtype=dtype)
    ts_idx = int(n_tokens_patches)
    text_idx = ts_idx + 1
    mask[ts_idx, text_idx] = float("-inf")
    mask[text_idx, ts_idx] = float("-inf")
    return mask


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(-1)])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(-2)]


class EnEmbeddingDual(nn.Module):
    """Patch embedding with optional patch-level FFT and two learnable global tokens (G_ts, G_text)."""

    def __init__(
        self,
        c_endo: int,
        d_model: int,
        patch_len: int,
        patch_stride: int | None = None,
        dropout: float = 0.1,
        fft_mode: str = "patch",
        close_idx: int = 0,
    ) -> None:
        super().__init__()
        self.c_endo = int(c_endo)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride) if patch_stride is not None else int(patch_len)
        self.fft_mode = str(fft_mode)
        self.close_idx = int(close_idx)

        self.value_embedding = nn.Linear(self.patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)

        if self.fft_mode == "patch":
            self.fft = PatchFFT()
            self.freq_proj = nn.Linear(6, d_model)
        else:
            self.fft = None
            self.freq_proj = None

        self.glb_tokens = nn.Parameter(torch.randn(1, self.c_endo, 2, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        x: [B, C_endo, L]
        returns tokens: [B * C_endo, N_patches + 2, d_model], c_endo: int
        """
        if x.dim() != 3 or x.size(1) != self.c_endo:
            raise ValueError(
                f"EnEmbeddingDual expects [B, C_endo={self.c_endo}, L], got {tuple(x.shape)}"
            )
        B, C_endo, L = x.shape
        # Unfold time axis: [B, C_endo, N_patches, patch_len]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        n_p = patches.size(2)

        # 1. Temporal patch embedding
        val_emb = self.value_embedding(patches)  # [B, C_endo, N_patches, d_model]

        # 2. Positional embedding
        pos_emb = self.position_embedding.pe[:, :n_p].unsqueeze(1)  # [1, 1, N_patches, d_model]

        # 3. Frequency embedding (if fft_mode == "patch")
        if self.fft_mode == "patch" and self.fft is not None and self.freq_proj is not None:
            patch_close = patches[:, self.close_idx, :, :]  # [B, N_patches, patch_len]
            freq_bins = self.fft(patch_close)               # [B, N_patches, 6]
            freq_emb = self.freq_proj(freq_bins).unsqueeze(1)  # [B, 1, N_patches, d_model]
            patches_emb = val_emb + pos_emb + freq_emb
        else:
            patches_emb = val_emb + pos_emb

        # Expand global tokens: [B, C_endo, 2, d_model]
        glb = self.glb_tokens.expand(B, -1, -1, -1)

        # Concatenate patches and global tokens: [B, C_endo, N_patches + 2, d_model]
        tokens = torch.cat([patches_emb, glb], dim=2)

        # Reshape to [B * C_endo, N_patches + 2, d_model]
        tokens = tokens.reshape(B * C_endo, n_p + 2, -1)
        return self.dropout(tokens), C_endo


class FlattenHead(nn.Module):
    """Flatten head projecting [B, (patch_num + 2) * d_model] -> horizon."""

    def __init__(
        self,
        nf: int,
        target_window: int,
        head_dropout: float = 0.0,
        head_type: str = "linear",
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        if head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(nf, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, target_window),
                nn.Dropout(head_dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(nf, target_window),
                nn.Dropout(head_dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)


class _DualEncoderLayer(nn.Module):
    """Canonical TimeXer layer with parallel Cross-Attention for G_ts and G_text."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_ts = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ts = nn.LayerNorm(d_model)
        self.cross_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_text = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        tokens: torch.Tensor,
        cross_ts: torch.Tensor | None,
        cross_text: torch.Tensor | None,
        c_endo: int,
    ) -> torch.Tensor:
        B_enc, N2, D = tokens.shape
        B = B_enc // c_endo

        # 1. Self-Attention over entire sequence [P_1..P_N; G_ts; G_text]
        residual = tokens
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(residual + self.dropout(attn_out))

        # 2. Slicing tokens
        x_patches = tokens[:, :-2, :]
        g_ts_ori = tokens[:, -2:-1, :]
        g_text_ori = tokens[:, -1:, :]

        # 3. Parallel Cross-Attention
        if cross_ts is not None and cross_ts.size(1) > 0:
            g_ts_q = g_ts_ori.reshape(B, c_endo, D)
            ts_out, _ = self.cross_ts(g_ts_q, cross_ts, cross_ts, need_weights=False)
            ts_out = ts_out.reshape(B_enc, 1, D)
            g_ts = self.norm_ts(g_ts_ori + self.dropout(ts_out))
        else:
            g_ts = self.norm_ts(g_ts_ori)

        if cross_text is not None and cross_text.size(1) > 0:
            g_text_q = g_text_ori.reshape(B, c_endo, D)
            text_out, _ = self.cross_text(g_text_q, cross_text, cross_text, need_weights=False)
            text_out = text_out.reshape(B_enc, 1, D)
            g_text = self.norm_text(g_text_ori + self.dropout(text_out))
        else:
            g_text = self.norm_text(g_text_ori)

        # 4. Concatenation
        tokens = torch.cat([x_patches, g_ts, g_text], dim=1)

        # 5. FFN: Conv1d(1x1) -> Activation -> Conv1d(1x1) + Norm
        y = tokens.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        return self.norm_ff(tokens + y)


class TimeXerDual(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        horizon: int,
        d_model: int,
        n_heads: int,
        e_layers: int = 2,
        patch_len: int = 16,
        patch_stride: int = 12,
        dropout: float = 0.1,
        text_dim: int = 15,
        fusion: Any = None,
        n_ts_features: int = 0,
        close_idx: int = 0,
        use_prototypes: bool = False,
        n_prototypes: int = 10,
        d_min: float = 0.5,
        d_ff: int | None = None,
        head_type: str = "linear",
        head_hidden: int = 128,
        head_dropout: float = 0.0,
        head_pool: str = "last",
        fft_mode: str = "patch",
    ) -> None:
        super().__init__()
        del head_pool
        if e_layers < 1:
            raise ValueError("e_layers must be >= 1")
        if int(n_features) not in (1, 2, 5):
            raise ValueError(f"TimeXerDual expects 1, 2, or 5 endogenous channels, got {n_features}")
        if int(text_dim) != 15:
            raise ValueError(f"TimeXerDual expects text_dim 15, got {text_dim}")
        if not 0 <= int(close_idx) < int(n_features):
            raise ValueError("close_idx must point at one endogenous channel")
        if fft_mode not in ("patch", "bridge", "none"):
            raise ValueError(f"fft_mode must be 'patch', 'bridge', or 'none', got {fft_mode}")

        if fusion is not None:
            self.fusion = assert_fusion(
                fusion,
                kind="c1_dual",
                text_at_head=False,
                text_to_patches=False,
                text_as_exogenous=True,
                global_to_patch=True,
            )
        else:
            self.fusion = {
                "kind": "c1_dual",
                "text_at_head": False,
                "text_to_patches": False,
                "text_as_exogenous": True,
                "global_to_patch": True,
            }

        self.n_features = int(n_features)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.text_dim = int(text_dim)
        self.n_ts_features = int(n_ts_features)
        self.close_idx = int(close_idx)
        self.use_prototypes = bool(use_prototypes)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.fft_mode = str(fft_mode)
        self.n_patches = n_patches(self.seq_len, self.patch_len, self.patch_stride)

        d_model = int(d_model)
        d_ff = 4 * d_model if d_ff is None else int(d_ff)

        # 1. EnEmbeddingDual (with optional patch-level FFT)
        self.en_embedding = EnEmbeddingDual(
            c_endo=self.n_features,
            d_model=d_model,
            patch_len=self.patch_len,
            patch_stride=self.patch_stride,
            dropout=dropout,
            fft_mode=self.fft_mode,
            close_idx=self.close_idx,
        )

        # 2. Exogenous projections
        self.ts_proj = nn.Linear(self.seq_len, d_model) if self.n_ts_features > 0 else None
        self.text_proj = nn.Linear(self.text_dim, d_model)

        # Bridge FFT projection: 6 harmonic log-amplitudes -> 6 variate tokens
        if self.fft_mode == "bridge":
            self.fft_variate_proj = nn.Linear(1, d_model)
        else:
            self.fft_variate_proj = None

        # 3. Encoder layers
        self.layers = nn.ModuleList(
            [_DualEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)]
        )

        # 4. Optional prototype bank
        if self.use_prototypes:
            if n_prototypes < 1:
                raise ValueError("n_prototypes must be >= 1 when use_prototypes is true")
            self.prototype_block = PrototypeResidual(int(n_prototypes), d_model, float(d_min))

        # 5. FlattenHead
        self.head_nf = (self.n_patches + 2) * d_model
        self.head = FlattenHead(
            nf=self.head_nf,
            target_window=self.horizon,
            head_dropout=head_dropout,
            head_type=head_type,
            head_hidden=head_hidden,
        )

    @property
    def proto(self):
        if not self.use_prototypes:
            raise AttributeError("TimeXerDual has no prototype bank when use_prototypes is false")
        return self.prototype_block.proto

    @property
    def glb_tokens(self) -> nn.Parameter:
        return self.en_embedding.glb_tokens

    @property
    def text_token(self) -> torch.Tensor:
        """Slice of global token 1 (text) for inspection/backward compatibility."""
        return self.en_embedding.glb_tokens[:, :, 1:2, :]

    @property
    def ts_token(self) -> torch.Tensor:
        """Slice of global token 0 (ts) for inspection/backward compatibility."""
        return self.en_embedding.glb_tokens[:, :, 0:1, :]

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor | None = None,
        text_seq: torch.Tensor | None = None,
        ts: torch.Tensor | None = None,
        proto_mode: str = "none",
        text_mode: str = "none",
        ablation_generator: torch.Generator | None = None,
        **_kwargs,
    ) -> ModelOutput:
        del _kwargs
        if x.dim() != 3 or x.size(-1) != self.n_features or x.size(1) != self.seq_len:
            raise ValueError(
                f"expected x [B, {self.seq_len}, {self.n_features}], got {tuple(x.shape)}"
            )
        if self.n_ts_features > 0 and ts is None:
            ts = torch.zeros(
                x.size(0), self.seq_len, self.n_ts_features, device=x.device, dtype=x.dtype
            )

        # 1. Endogenous embedding [B, C_endo, L]
        x_endo = x.permute(0, 2, 1)
        tokens, _ = self.en_embedding(x_endo)  # [B * C_endo, N_patches + 2, d_model]

        # 2. Exogenous projections (including bridge FFT if enabled)
        cross_ts = self._project_ts(ts, x)
        cross_text = self._project_text(text, text_seq, text_mode, ablation_generator)

        # 3. Encoder layers
        for layer in self.layers:
            tokens = layer(tokens, cross_ts=cross_ts, cross_text=cross_text, c_endo=self.n_features)

        # 4. Extract close channel tokens
        enc_out = tokens.reshape(x.size(0), self.n_features, self.n_patches + 2, -1)
        enc_close = enc_out[:, self.close_idx, :, :]  # [B, N_patches + 2, d_model]
        bank = enc_close

        proto_losses: PrototypeLosses | None = None
        if self.use_prototypes:
            enc_close, proto_losses = self.prototype_block(
                enc_close, proto_mode=proto_mode, ablation_generator=ablation_generator
            )

        # 5. Head
        pred = self.head(enc_close)
        return ModelOutput(pred=pred, proto_losses=proto_losses, segments=bank)

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

    def _project_ts(self, ts: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor | None:
        ts_tokens: torch.Tensor | None = None
        if ts is not None and self.ts_proj is not None:
            if ts.dim() == 2:
                ts = ts.unsqueeze(1).expand(-1, self.seq_len, -1)
            if ts.dim() != 3 or ts.size(-1) != self.n_ts_features:
                raise ValueError(
                    f"ts must be [B, T, {self.n_ts_features}] or [B, {self.n_ts_features}], got {tuple(ts.shape)}"
                )
            # Inverted variate embedding: [B, C_ts, seq_len] -> [B, C_ts, d_model]
            ts_tokens = self.ts_proj(ts.transpose(1, 2))

        fft_tokens: torch.Tensor | None = None
        if self.fft_mode == "bridge" and self.fft_variate_proj is not None:
            close_series = x[:, :, self.close_idx]  # [B, T]
            centered = close_series - close_series.mean(dim=1, keepdim=True)
            spectrum = torch.fft.rfft(centered, dim=1)
            # First 6 harmonic log-amplitudes: [B, 6]
            fft_variates = torch.log(spectrum[:, 1:7].abs() + 1e-6)
            # Project each harmonic: [B, 6, 1] -> [B, 6, d_model]
            fft_tokens = self.fft_variate_proj(fft_variates.unsqueeze(-1))

        if ts_tokens is not None and fft_tokens is not None:
            return torch.cat([ts_tokens, fft_tokens], dim=1)
        elif ts_tokens is not None:
            return ts_tokens
        elif fft_tokens is not None:
            return fft_tokens
        return None

    def _project_text(
        self,
        text: torch.Tensor | None,
        text_seq: torch.Tensor | None,
        text_mode: str,
        ablation_generator: torch.Generator | None,
    ) -> torch.Tensor:
        text_in = _exo_sequence(text, text_seq, self.text_dim)
        if text_mode != "none":
            text_in = apply_feature_ablation(text_in, text_mode, ablation_generator)
        return self.text_proj(text_in)


def _exo_sequence(
    text: torch.Tensor | None,
    text_seq: torch.Tensor | None,
    text_dim: int,
) -> torch.Tensor:
    if text_seq is not None:
        exo = text_seq
    elif text is not None:
        exo = text.unsqueeze(1) if text.dim() == 2 else text
    else:
        raise ValueError("TimeXerDual requires text [B, 15] or text_seq [B, L, 15]")
    if exo.dim() != 3 or exo.size(-1) != text_dim:
        raise ValueError(f"text exo must have shape [B, L, {text_dim}], got {tuple(exo.shape)}")
    return exo
