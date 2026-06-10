"""
ViT-1D — Vision Transformer adapted for 1-D MR signals.

When ``evidential=True`` the regression head outputs four NIG distribution
parameters (γ, ν, α, β) instead of a single point estimate.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _PatchEmbed1D(nn.Module):
    """Non-overlapping 1-D patch embedding."""

    def __init__(self, in_channels: int, patch_size: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).transpose(1, 2)
        return self.norm(x)


class _TransformerBlock(nn.Module):
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


class ViT1D(nn.Module):
    """
    Vision Transformer for 1-D MR signals.

    When *evidential* is True, the final layer outputs 4 × output_dim values
    corresponding to the Normal-Inverse-Gamma parameters (γ, ν, α, β) for
    each predicted quantity.  Softplus is applied to ν, α, β and α is
    incremented by 1 to satisfy the NIG constraint α > 1.
    """

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 2048,
        patch_size: int = 32,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        output_dim: int = 2,
        dropout: float = 0.1,
        pooling: str = "cls",
        evidential: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.n_patches = seq_len // patch_size
        self.evidential = evidential
        self.output_dim = output_dim

        self.patch_embed = _PatchEmbed1D(in_channels, patch_size, hidden_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        n_tokens = self.n_patches + 1 if pooling == "cls" else self.n_patches
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, hidden_dim) * 0.02)
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            _TransformerBlock(hidden_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        self._feature_dim = hidden_dim

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
        """Return the penultimate feature vector (B, D)."""
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.pooling == "cls":
            return x[:, 0]
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return regression predictions or NIG parameters (B, output_dim [* 4])."""
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
