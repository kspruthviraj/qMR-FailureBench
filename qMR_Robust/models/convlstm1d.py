"""
ConvLSTM-1D — Convolutional LSTM for 1-D MR signals.

Combines convolutional feature extraction with LSTM temporal modeling.
The LSTM processes the signal sequentially, maintaining a hidden state
that can capture long-range temporal dependencies in the MRF signal.

When ``evidential=True`` the regression head outputs four NIG distribution
parameters (γ, ν, α, β) instead of a single point estimate.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTM1D(nn.Module):
    """Convolutional LSTM for 1-D MR signal regression.

    Architecture:
        Conv1D stem → LSTM → Pool → Evidential regression head

    Parameters
    ----------
    in_channels : int
        Number of input channels (2 for real/imag).
    hidden_dim : int
        LSTM hidden dimension.
    n_lstm_layers : int
        Number of stacked LSTM layers.
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
        hidden_dim: int = 128,
        n_lstm_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.1,
        evidential: bool = False,
    ):
        super().__init__()
        self.evidential = evidential
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Conv stem: extract local features
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Regression head
        head_out = output_dim * 4 if evidential else output_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, head_out),
        )

        self._feature_dim = hidden_dim * 2

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
        Tensor (B, hidden_dim*2)
        """
        # Conv stem: (B, C, L) → (B, hidden_dim, L//4)
        h = self.conv_stem(x)

        # Reshape for LSTM: (B, hidden_dim, L//4) → (B, L//4, hidden_dim)
        h = h.transpose(1, 2)

        # LSTM: (B, L//4, hidden_dim) → (B, L//4, hidden_dim*2)
        h, _ = self.lstm(h)

        # Pool over time: (B, L//4, hidden_dim*2) → (B, hidden_dim*2)
        h = h.mean(dim=1)

        return self.dropout(h)

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
        return self.head(features)
