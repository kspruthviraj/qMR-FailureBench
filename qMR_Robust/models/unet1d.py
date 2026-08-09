"""
U-Net 1D — Encoder-decoder with skip connections for 1-D MR signals.

The U-Net architecture uses a contracting path (encoder) to capture context
and an expanding path (decoder) to enable precise localization. Skip
connections between encoder and decoder preserve fine-grained temporal details.

When ``evidential=True`` the regression head outputs four NIG distribution
parameters (γ, ν, α, β) instead of a single point estimate.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _EncoderBlock(nn.Module):
    """Conv → BN → GELU → Conv → BN → GELU with skip connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor):
        skip = self.block(x)
        down = self.pool(skip)
        return down, skip


class _DecoderBlock(nn.Module):
    """Upsample → concat skip → Conv → BN → GELU → Conv → BN → GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.block = nn.Sequential(
            nn.Conv1d(out_ch * 2, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad to match skip size
        diff = skip.shape[2] - x.shape[2]
        if diff > 0:
            x = nn.functional.pad(x, [diff // 2, diff - diff // 2])
        elif diff < 0:
            x = x[:, :, :skip.shape[2]]
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNet1D(nn.Module):
    """U-Net for 1-D MR signal regression.

    Architecture:
        Encoder (4 levels) → Bottleneck → Decoder (4 levels) → Pool → Head

    Parameters
    ----------
    in_channels : int
        Number of input channels (2 for real/imag).
    hidden_dim : int
        Base channel dimension (doubled at each encoder level).
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
        hidden_dim: int = 64,
        output_dim: int = 2,
        dropout: float = 0.1,
        evidential: bool = False,
    ):
        super().__init__()
        self.evidential = evidential
        self.output_dim = output_dim

        # Encoder
        self.enc1 = _EncoderBlock(in_channels, hidden_dim)
        self.enc2 = _EncoderBlock(hidden_dim, hidden_dim * 2)
        self.enc3 = _EncoderBlock(hidden_dim * 2, hidden_dim * 4)
        self.enc4 = _EncoderBlock(hidden_dim * 4, hidden_dim * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dim * 8, hidden_dim * 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim * 16),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 16, hidden_dim * 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim * 16),
            nn.GELU(),
        )

        # Decoder
        self.dec4 = _DecoderBlock(hidden_dim * 16, hidden_dim * 8)
        self.dec3 = _DecoderBlock(hidden_dim * 8, hidden_dim * 4)
        self.dec2 = _DecoderBlock(hidden_dim * 4, hidden_dim * 2)
        self.dec1 = _DecoderBlock(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Regression head
        self._feature_dim = hidden_dim
        head_out = output_dim * 4 if evidential else output_dim
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_out),
        )

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input signal.

        Parameters
        ----------
        x : Tensor (B, C, L)

        Returns
        -------
        Tensor (B, hidden_dim)
        """
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Pool
        x = nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, C, L)

        Returns
        -------
        Tensor (B, output_dim) or (B, output_dim * 4) if evidential
        """
        features = self.encode(x)
        # features: (B, hidden_dim) → need (B, hidden_dim, 1) for head
        return self.head(features.unsqueeze(-1)).squeeze(-1)
