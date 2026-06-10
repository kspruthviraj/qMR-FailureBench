"""
Baseline Uncertainty Quantification Methods.

Implements:
  - MCDropout: Standard model with dropout active at inference (approx. Bayesian)
  - DeepEnsemble: Average predictions across N independently trained models
  - QuantileRegression: Predicts 10th, 50th, 90th percentiles
  - HeteroscedasticGaussian: Predicts mean and log-variance (aleatoric only)
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MC-Dropout Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MCDropoutModel(nn.Module):
    """Wraps any model to keep dropout active during inference.

    At inference time, run T forward passes and compute mean + variance
    of the predictions.
    """

    def __init__(self, base_model: nn.Module, n_samples: int = 30):
        super().__init__()
        self.base_model = base_model
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run T stochastic forward passes.

        Returns
        -------
        mean : Tensor (B, D)
        variance : Tensor (B, D)
        """
        self.train()  # Keep dropout active
        preds = []
        for _ in range(self.n_samples):
            preds.append(self.base_model(x).cpu())
        preds = torch.stack(preds, dim=0)  # (T, B, D)
        mean = preds.mean(dim=0)
        variance = preds.var(dim=0)
        self.eval()
        return mean, variance


# ──────────────────────────────────────────────────────────────────────────────
# Deep Ensemble Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DeepEnsemble:
    """Trains and manages N independent models for ensemble uncertainty."""

    def __init__(self, model_fn, n_models: int = 5, device: str = "cuda"):
        self.model_fn = model_fn
        self.n_models = n_models
        self.device = device
        self.models: List[nn.Module] = []

    def train_single(
        self,
        model_idx: int,
        train_loader,
        val_loader,
        n_epochs: int = 50,
        lr: float = 5e-4,
    ) -> dict:
        """Train a single ensemble member with MSE loss."""
        model = self.model_fn().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            epoch_loss, n_batches = 0.0, 0
            for signals, targets in train_loader:
                signals, targets = signals.to(self.device), targets.to(self.device)
                pred = model(signals)
                loss = F.mse_loss(pred, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            history["train_loss"].append(epoch_loss / n_batches)

            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for signals, targets in val_loader:
                    signals, targets = signals.to(self.device), targets.to(self.device)
                    pred = model(signals)
                    val_loss_sum += F.mse_loss(pred, targets).item() * signals.size(0)
                    val_n += signals.size(0)
            val_loss = val_loss_sum / val_n
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        self.models.append(model.to(self.device))
        return history

    def train_all(self, train_loader, val_loader, n_epochs: int = 50, lr: float = 5e-4):
        """Train all ensemble members."""
        for i in range(self.n_models):
            logger.info("Training ensemble member %d/%d", i + 1, self.n_models)
            self.train_single(i, train_loader, val_loader, n_epochs, lr)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, signals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ensemble mean and variance.

        Returns
        -------
        mean : Tensor (B, D)
        variance : Tensor (B, D)
        """
        preds = []
        for model in self.models:
            model.eval()
            preds.append(model(signals.to(self.device)).cpu())
        preds = torch.stack(preds, dim=0)  # (M, B, D)
        return preds.mean(dim=0), preds.var(dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Quantile Regression Model
# ──────────────────────────────────────────────────────────────────────────────

class QuantileRegressionHead(nn.Module):
    """Head that predicts 3 quantiles (10th, 50th, 90th) per target."""

    def __init__(self, in_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.quantiles = [0.1, 0.5, 0.9]
        self.output_dim = output_dim
        out_features = output_dim * len(self.quantiles)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_features),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return (B, D, 3) quantile predictions."""
        raw = self.fc(features)
        return raw.view(features.shape[0], self.output_dim, len(self.quantiles))


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """Pinball loss for quantile regression.

    Parameters
    ----------
    pred : (B, D, Q) quantile predictions
    target : (B, D) ground truth
    quantiles : list of quantile levels
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = target - pred[..., i]
        loss = torch.max(q * error, (q - 1) * error)
        losses.append(loss)
    return torch.stack(losses, dim=-1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Heteroscedastic Gaussian Model
# ──────────────────────────────────────────────────────────────────────────────

class HeteroscedasticGaussianHead(nn.Module):
    """Predicts mean and log-variance for heteroscedastic aleatoric uncertainty."""

    def __init__(self, in_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, output_dim * 2),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_var) each (B, D)."""
        raw = self.fc(features)
        mean = raw[..., :self.output_dim]
        log_var = raw[..., self.output_dim:]
        return mean, log_var


def heteroscedastic_nll(
    mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL for heteroscedastic regression."""
    var = torch.exp(log_var).clamp(min=1e-6)
    nll = 0.5 * (torch.log(var) + (target - mean).pow(2) / var)
    return nll


# ──────────────────────────────────────────────────────────────────────────────
# Model factory for baselines
# ──────────────────────────────────────────────────────────────────────────────

def build_baseline_model(
    backbone_name: str,
    backbone_fn,
    baseline_type: str,
    output_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    """Build a baseline model by attaching a non-evidential head to a backbone.

    Parameters
    ----------
    backbone_name : str
        Name for logging.
    backbone_fn : callable
        Returns a backbone module with `encode()` method and `feature_dim` property.
    baseline_type : str
        One of 'deterministic', 'mc_dropout', 'quantile', 'heteroscedastic'.
    output_dim : int
        Number of prediction targets.
    hidden_dim : int
        Hidden dimension for the head.
    dropout : float
        Dropout rate (kept active for mc_dropout at inference).
    """
    backbone = backbone_fn()

    if baseline_type == "deterministic":
        head = nn.Sequential(
            nn.Linear(backbone.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        return _BaselineModel(backbone, head)

    elif baseline_type == "mc_dropout":
        head = nn.Sequential(
            nn.Linear(backbone.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        base = _BaselineModel(backbone, head)
        return MCDropoutModel(base, n_samples=30)

    elif baseline_type == "quantile":
        head = QuantileRegressionHead(backbone.feature_dim, output_dim, dropout)
        return _BaselineModel(backbone, head, head_type="quantile")

    elif baseline_type == "heteroscedastic":
        head = HeteroscedasticGaussianHead(backbone.feature_dim, output_dim, dropout)
        return _BaselineModel(backbone, head, head_type="heteroscedastic")

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


class _BaselineModel(nn.Module):
    """Backbone + head wrapper for baseline models."""

    def __init__(self, backbone: nn.Module, head: nn.Module, head_type: str = "deterministic"):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.head_type = head_type

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(x)

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        if self.head_type == "quantile":
            return self.head(features)  # (B, D, 3)
        elif self.head_type == "heteroscedastic":
            return self.head(features)  # (mean, log_var)
        else:
            return self.head(features)  # (B, D)
