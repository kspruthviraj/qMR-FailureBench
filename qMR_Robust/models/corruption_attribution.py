"""
Explainable Corruption Attribution Head.

Extends the evidential regression model with a second classification head
that predicts the *source* of the corruption (B0, B1+, motion) from the
same feature representation.  This enables the model to not only say
"This prediction will fail" but also "because of B0 off-resonance."

Architecture:
  Backbone → features
    ├─ Evidential Head → (γ, ν, α, β) per target
    └─ Attribution Head → (p_B0, p_B1, p_motion) corruption source probabilities
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorruptionAttributionHead(nn.Module):
    """Classification head that predicts corruption source probabilities.

    Given features from the backbone, predicts a 3-class probability
    distribution over corruption sources: {B0, B1+, motion}.

    During training, the labels are derived from the PhysicsCorruptor
    metadata (which corruptions were applied to each sample).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 3 corruption sources
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities for corruption sources (B, 3)."""
        return F.log_softmax(self.fc(features), dim=-1)


class EvidentialWithAttribution(nn.Module):
    """Wraps any backbone to produce both NIG parameters and corruption attribution.

    This is the core "Explainable Failure Forecasting" model.

    Parameters
    ----------
    backbone : nn.Module
        Must have `encode(x) -> (B, D)` and `feature_dim` property.
    output_dim : int
        Number of regression targets.
    hidden_dim : int
        Hidden dimension for heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        backbone: nn.Module,
        output_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim
        feature_dim = backbone.feature_dim

        # Evidential regression head → (γ, ν, α, β)
        self.evidential_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * 4),
        )

        # Corruption attribution head → (p_B0, p_B1, p_motion)
        self.attribution_head = CorruptionAttributionHead(feature_dim, hidden_dim, dropout)

        self.softplus = nn.Softplus()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)

    def forward(
        self, x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return NIG parameters and corruption attribution.

        Returns
        -------
        dict with:
            'nig': Tensor (B, D, 4) — [γ, ν, α, β]
            'attribution': Tensor (B, 3) — log-probabilities [B0, B1, motion]
            'features': Tensor (B, feature_dim) — backbone features
        """
        features = self.encode(x)

        # Evidential head
        raw = self.evidential_head(features)
        B = raw.shape[0]
        D = self.output_dim
        raw = raw.view(B, D, 4)

        gamma = raw[..., 0]
        nu = self.softplus(raw[..., 1])
        alpha = self.softplus(raw[..., 2]) + 1.0
        beta = self.softplus(raw[..., 3])
        nig = torch.stack([gamma, nu, alpha, beta], dim=-1)

        # Attribution head
        attribution = self.attribution_head(features)

        return {
            "nig": nig,
            "attribution": attribution,
            "features": features,
        }


def compute_attribution_targets(
    b0_hz: torch.Tensor,
    b1_scale: torch.Tensor,
    motion_shift: torch.Tensor,
    threshold_b0: float = 1.0,
    threshold_b1: float = 0.01,
    threshold_motion: float = 0.5,
) -> torch.Tensor:
    """Convert corruption metadata to multi-label attribution targets.

    Each sample can have multiple active corruption sources.
    Returns soft targets (probabilities) rather than hard labels.

    Parameters
    ----------
    b0_hz : (B,) — applied B0 shift in Hz
    b1_scale : (B,) — applied B1 scale factor
    motion_shift : (B,) — applied motion shift in voxels

    Returns
    -------
    targets : (B, 3) — soft probability targets for [B0, B1, motion]
    """
    p_b0 = (b0_hz.abs() > threshold_b0).float()
    p_b1 = ((b1_scale - 1.0).abs() > threshold_b1).float()
    p_motion = (motion_shift.abs() > threshold_motion).float()

    targets = torch.stack([p_b0, p_b1, p_motion], dim=-1)

    # Normalize to sum to 1 (soft targets)
    row_sum = targets.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return targets / row_sum


def attribution_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """KL divergence loss for corruption attribution.

    Parameters
    ----------
    log_probs : (B, 3) — predicted log-probabilities
    targets : (B, 3) — soft target probabilities

    Returns
    -------
    loss : scalar
    """
    return F.kl_div(log_probs, targets, reduction="batchmean")


def joint_loss(
    nig_params: torch.Tensor,
    targets_reg: torch.Tensor,
    attribution_log_probs: torch.Tensor,
    attribution_targets: torch.Tensor,
    epoch: int = 0,
    reg_coeff: float = 1.0,
    attr_coeff: float = 0.5,
    annealing_epochs: int = 10,
) -> Dict[str, torch.Tensor]:
    """Joint loss: evidential regression + corruption attribution.

    Combines:
    1. NIG NLL loss for regression
    2. Evidential regularizer
    3. Attribution KL loss (with annealing)
    """
    from qMR_Robust.models.losses import evidential_regression_loss

    # Regression loss
    reg_result = evidential_regression_loss(
        targets_reg, nig_params,
        coeff=reg_coeff, epoch=epoch, annealing_epochs=annealing_epochs,
    )

    # Attribution loss
    attr_loss = attribution_loss(attribution_log_probs, attribution_targets)

    # Anneal attribution loss too
    anneal_weight = min(1.0, epoch / max(annealing_epochs, 1))
    total_loss = reg_result["loss"] + attr_coeff * anneal_weight * attr_loss

    return {
        "loss": total_loss,
        "reg_loss": reg_result["loss"],
        "nll": reg_result["nll"],
        "er": reg_result["reg"],
        "attr_loss": attr_loss.detach(),
    }
