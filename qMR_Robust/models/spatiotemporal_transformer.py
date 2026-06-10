"""
Spatio-Temporal Transformer for 1-D MR Signals.

Combines temporal self-attention (across the signal time axis) with a
spatial cross-attention mechanism (across the real/imaginary channels)
to capture both temporal dynamics and channel interactions in complex-valued
MR signals.

Architecture:
  Conv1D stem → Patch embedding → [Temporal Transformer blocks × N]
  → Channel cross-attention → [Temporal Transformer blocks × M]
  → Pool → Evidential regression head
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ConvStem(nn.Module):
    """1-D convolutional stem that converts raw signal to hidden features."""

    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size, stride=2, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size, stride=2, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B, D, L')


class _TemporalTransformerBlock(nn.Module):
    """Standard pre-norm Transformer block operating on the time axis."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class _ChannelCrossAttention(nn.Module):
    """Cross-attention between real and imaginary channel features.

    Treats the 2 channels as separate 'views' and lets them attend to each
    other, producing a fused representation.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> torch.Tensor:
        # Real attends to Imaginary, then fuse
        q = self.norm_q(x_real)
        kv = self.norm_kv(x_imag)
        attended, _ = self.cross_attn(q, kv, kv)
        fused = x_real + attended
        fused = fused + self.ff(self.norm_ff(fused))
        return fused


class SpatioTemporalTransformer(nn.Module):
    """Spatio-Temporal Transformer for 1-D MR signals.

    This architecture processes complex-valued (2-channel) MR signals through:
    1. Convolutional stem for local feature extraction
    2. Temporal Transformer blocks for long-range temporal dependencies
    3. Channel cross-attention for real/imaginary interaction
    4. Second-stage temporal Transformer blocks
    5. Pooling and evidential regression head

    Parameters
    ----------
    in_channels : int
        Number of input channels (2 for real/imag).
    seq_len : int
        Input signal length.
    hidden_dim : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_temporal_layers_1 : int
        Number of temporal Transformer blocks before cross-attention.
    n_temporal_layers_2 : int
        Number of temporal Transformer blocks after cross-attention.
    output_dim : int
        Number of regression targets.
    dropout : float
        Dropout rate.
    evidential : bool
        If True, output NIG parameters (γ, ν, α, β).
    """

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 1000,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_temporal_layers_1: int = 3,
        n_temporal_layers_2: int = 2,
        output_dim: int = 2,
        dropout: float = 0.1,
        evidential: bool = False,
    ):
        super().__init__()
        self.evidential = evidential
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Conv stem: reduces seq_len by 4×
        self.stem = _ConvStem(in_channels, hidden_dim)
        self.seq_len_after_stem = seq_len // 4

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len_after_stem, hidden_dim) * 0.02)

        # First-stage temporal Transformer
        self.temporal_blocks_1 = nn.ModuleList([
            _TemporalTransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_temporal_layers_1)
        ])

        # Channel cross-attention
        self.channel_cross_attn = _ChannelCrossAttention(hidden_dim, n_heads, dropout)

        # Second-stage temporal Transformer
        self.temporal_blocks_2 = nn.ModuleList([
            _TemporalTransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_temporal_layers_2)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self._feature_dim = hidden_dim

        # Regression head
        head_out = output_dim * 4 if evidential else output_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_out),
        )

        self.softplus = nn.Softplus()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate feature vector (B, D).

        Parameters
        ----------
        x : Tensor (B, 2, L)
            Input signal with real and imaginary channels.
        """
        B = x.shape[0]

        # Conv stem: (B, 2, L) → (B, D, L')
        h = self.stem(x)  # (B, D, L')
        h = h.transpose(1, 2)  # (B, L', D)

        # Add positional encoding
        h = h + self.pos_embed[:, :h.size(1), :]

        # First-stage temporal self-attention
        for blk in self.temporal_blocks_1:
            h = blk(h)

        # Channel cross-attention: treat time-steps as the sequence,
        # and use real/imag features from the conv stem to create
        # two 'channel views' by splitting the hidden dim
        half = self.hidden_dim // 2
        x_real_view = h[..., :half]
        x_imag_view = h[..., half:]
        # Pad to full dim for cross-attention
        x_real_full = torch.cat([x_real_view, x_real_view], dim=-1)
        x_imag_full = torch.cat([x_imag_view, x_imag_view], dim=-1)
        h = self.channel_cross_attn(x_real_full, x_imag_full)

        # Second-stage temporal self-attention
        for blk in self.temporal_blocks_2:
            h = blk(h)

        h = self.norm(h)
        return h.mean(dim=1)  # global average pool over time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions or NIG parameters."""
        features = self.encode(x)
        raw = self.head(features)

        if not self.evidential:
            return raw

        return self._nig_activation(raw)

    def _nig_activation(self, raw: torch.Tensor) -> torch.Tensor:
        """Split raw head output into (γ, ν, α, β) and apply constraints."""
        B = raw.shape[0]
        D = self.output_dim
        raw = raw.view(B, D, 4)

        gamma = raw[..., 0]
        nu = self.softplus(raw[..., 1])
        alpha = self.softplus(raw[..., 2]) + 1.0
        beta = self.softplus(raw[..., 3])

        return torch.stack([gamma, nu, alpha, beta], dim=-1)
