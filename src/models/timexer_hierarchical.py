"""Canonical Hierarchical TimeXer (NeurIPS 2024 / TSLib) with optional FFT integration.

3-layer cross-modal hierarchy:
- Layer 1 (Technical Micro-structure):
  Self-Attention over [P; G_ts; G_text].
  Cross-Attention: ONLY G_ts queries cross_ts (optionally + 6 global FFT tokens in bridge mode).
  G_text skips cross-attention.
- Layer 2 (Semantic Macro-structure):
  Self-Attention: patches and G_text absorb enriched technical token G_ts^(1).
  Cross-Attention: ONLY G_text queries cross_text. G_ts skips cross-attention.
- Layer 3 (Cross-modal Consolidation):
  Self-Attention: all price patches receive consolidated info from G_text^(2) and G_ts^(2).
  Final Conv1d FFN.
- FlattenHead: unfolds target variable tokens -> Linear -> horizon H=7.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.ablate import apply_feature_ablation
from src.models.fusion import assert_fusion
from src.models.outputs import ModelOutput, compute_pred_loss
from src.models.prototypes import PrototypeLosses
from src.models.timexer_backbone import n_patches
from src.models.timexer_dual import EnEmbeddingDual, FlattenHead
from src.models.timexl_integration import PrototypeResidual


class _HierarchicalLayer1(nn.Module):
    """Layer 1: Self-attention on [P; G_ts; G_text], cross-attention ONLY G_ts <- cross_ts."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_ts = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ts = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tokens: torch.Tensor, cross_ts: torch.Tensor | None, c_endo: int) -> torch.Tensor:
        B_enc, N2, D = tokens.shape
        B = B_enc // c_endo

        # 1. Self-Attention
        residual = tokens
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(residual + self.dropout(attn_out))

        # 2. Slicing
        x_patches = tokens[:, :-2, :]
        g_ts_ori = tokens[:, -2:-1, :]
        g_text_ori = tokens[:, -1:, :]

        # 3. Cross-attention ONLY G_ts
        if cross_ts is not None and cross_ts.size(1) > 0:
            g_ts_q = g_ts_ori.reshape(B, c_endo, D)
            ts_out, _ = self.cross_ts(g_ts_q, cross_ts, cross_ts, need_weights=False)
            ts_out = ts_out.reshape(B_enc, 1, D)
            g_ts = self.norm_ts(g_ts_ori + self.dropout(ts_out))
        else:
            g_ts = self.norm_ts(g_ts_ori)
        g_text = g_text_ori

        # 4. Concatenation
        tokens = torch.cat([x_patches, g_ts, g_text], dim=1)

        # 5. FFN
        y = tokens.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        return self.norm_ff(tokens + y)


class _HierarchicalLayer2(nn.Module):
    """Layer 2: Self-attention on [P; G_ts; G_text], cross-attention ONLY G_text <- cross_text."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_text = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tokens: torch.Tensor, cross_text: torch.Tensor | None, c_endo: int) -> torch.Tensor:
        B_enc, N2, D = tokens.shape
        B = B_enc // c_endo

        # 1. Self-Attention (patches and G_text absorb enriched G_ts from Layer 1)
        residual = tokens
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(residual + self.dropout(attn_out))

        # 2. Slicing
        x_patches = tokens[:, :-2, :]
        g_ts_ori = tokens[:, -2:-1, :]
        g_text_ori = tokens[:, -1:, :]

        # 3. Cross-attention ONLY G_text
        if cross_text is not None and cross_text.size(1) > 0:
            g_text_q = g_text_ori.reshape(B, c_endo, D)
            text_out, _ = self.cross_text(g_text_q, cross_text, cross_text, need_weights=False)
            text_out = text_out.reshape(B_enc, 1, D)
            g_text = self.norm_text(g_text_ori + self.dropout(text_out))
        else:
            g_text = self.norm_text(g_text_ori)
        g_ts = g_ts_ori

        # 4. Concatenation
        tokens = torch.cat([x_patches, g_ts, g_text], dim=1)

        # 5. FFN
        y = tokens.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        return self.norm_ff(tokens + y)


class _HierarchicalLayer3(nn.Module):
    """Layer 3: Cross-modal consolidation via Self-Attention on [P; G_ts; G_text] + Conv1d FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention: patches receive consolidated info from G_text and G_ts
        residual = tokens
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(residual + self.dropout(attn_out))

        # 2. Final FFN
        y = tokens.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y)).transpose(1, 2)
        return self.norm_ff(tokens + y)


class TimeXerHierarchical(nn.Module):
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
        n_ts_features: int = 25,
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
        if int(e_layers) != 3:
            raise ValueError(
                f"Canonical TimeXerHierarchical requires exactly 3 layers (ts -> text -> consolidation), got e_layers={e_layers}"
            )
        if int(n_features) not in (1, 2, 5):
            raise ValueError(
                f"TimeXerHierarchical expects 1, 2, or 5 endogenous channels, got {n_features}"
            )
        if int(text_dim) != 15:
            raise ValueError(f"TimeXerHierarchical expects text_dim 15, got {text_dim}")
        if not 0 <= int(close_idx) < int(n_features):
            raise ValueError("close_idx must point at one endogenous channel")
        if fft_mode not in ("patch", "bridge", "none"):
            raise ValueError(f"fft_mode must be 'patch', 'bridge', or 'none', got {fft_mode}")

        if fusion is not None:
            self.fusion = assert_fusion(
                fusion,
                kind="c1_hierarchical",
                text_at_head=False,
                text_to_patches=False,
                text_as_exogenous=True,
                global_to_patch=True,
            )
        else:
            self.fusion = {
                "kind": "c1_hierarchical",
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

        # 3. 3-Layer Hierarchy
        self.layer1 = _HierarchicalLayer1(d_model, n_heads, d_ff, dropout)
        self.layer2 = _HierarchicalLayer2(d_model, n_heads, d_ff, dropout)
        self.layer3 = _HierarchicalLayer3(d_model, n_heads, d_ff, dropout)

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
            raise AttributeError(
                "TimeXerHierarchical has no prototype bank when use_prototypes is false"
            )
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

        # 3. 3-Layer Hierarchy
        tokens = self.layer1(tokens, cross_ts=cross_ts, c_endo=self.n_features)
        tokens = self.layer2(tokens, cross_text=cross_text, c_endo=self.n_features)
        tokens = self.layer3(tokens)

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
        raise ValueError("TimeXerHierarchical requires text [B, 15] or text_seq [B, L, 15]")
    if exo.dim() != 3 or exo.size(-1) != text_dim:
        raise ValueError(f"text exo must have shape [B, L, {text_dim}], got {tuple(exo.shape)}")
    return exo
