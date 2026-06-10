"""
Continuous Corruption Severity Regression + Counterfactual Correction.

Instead of classifying corruption *type* (3-class), this module predicts
the continuous corruption *parameters*:
  - Δf (B0 off-resonance in Hz)
  - λ  (B1+ scaling factor, dimensionless)
  - δ  (motion shift in voxels)

This enables:
  1. "Failure due to B0 = 63 Hz" (precise actionable diagnosis)
  2. Counterfactual correction: invert estimated corruption → re-predict → lower error
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qMR_Robust.simulators.corruptor import PhysicsCorruptor


class SeverityRegressionHead(nn.Module):
    """Predicts continuous corruption parameters (Δf, λ-1, δ) with NIG uncertainty.

    Outputs 4 values per corruption parameter:
      - γ: predicted magnitude
      - ν, α, β: NIG uncertainty parameters (optional evidential mode)
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, evidential: bool = True):
        super().__init__()
        self.evidential = evidential
        out = 3 * 4 if evidential else 3
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out),
        )
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
          'delta_f': (B,) predicted B0 shift
          'lambda_b1': (B,) predicted B1 scale
          'delta_motion': (B,) predicted motion shift
          'severity_nig': (B, 3, 4) if evidential, else None
        """
        raw = self.fc(features)
        if not self.evidential:
            return {
                "delta_f": raw[:, 0],
                "lambda_b1": raw[:, 1],
                "delta_motion": raw[:, 2],
                "severity_nig": None,
            }

        B = features.shape[0]
        raw = raw.view(B, 3, 4)
        gamma = raw[..., 0]
        nu = self.softplus(raw[..., 1])
        alpha = self.softplus(raw[..., 2]) + 1.0
        beta = self.softplus(raw[..., 3])
        nig = torch.stack([gamma, nu, alpha, beta], dim=-1)

        return {
            "delta_f": gamma[:, 0],
            "lambda_b1": gamma[:, 1],
            "delta_motion": gamma[:, 2],
            "severity_nig": nig,
        }


class DualHeadWithSeverity(nn.Module):
    """Backbone → evidential regression head + severity regression head.

    This is the "Explainable + Correctable" failure forecasting model.
    """

    def __init__(self, backbone: nn.Module, output_dim: int = 2, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim
        fd = backbone.feature_dim

        # Evidential regression head
        self.reg_head = nn.Sequential(
            nn.Linear(fd, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * 4),
        )

        # Severity regression head
        self.severity_head = SeverityRegressionHead(fd, hidden_dim, evidential=False)

        self.softplus = nn.Softplus()

    def encode(self, x): return self.backbone.encode(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encode(x)

        # NIG regression
        raw = self.reg_head(features)
        B = raw.shape[0]
        D = self.output_dim
        raw = raw.view(B, D, 4)
        gamma = raw[..., 0]
        nu = self.softplus(raw[..., 1])
        alpha = self.softplus(raw[..., 2]) + 1.0
        beta = self.softplus(raw[..., 3])
        nig = torch.stack([gamma, nu, alpha, beta], dim=-1)

        # Severity
        severity = self.severity_head(features)

        return {"nig": nig, "severity": severity, "features": features}


def severity_regression_loss(pred: Dict, b0_true, b1_true, motion_true) -> torch.Tensor:
    """MSE loss for corruption severity prediction."""
    loss_b0 = F.mse_loss(pred["delta_f"], b0_true)
    loss_b1 = F.mse_loss(pred["lambda_b1"], b1_true - 1.0)  # predict deviation from 1
    loss_mot = F.mse_loss(pred["delta_motion"], motion_true)
    return loss_b0 + loss_b1 + loss_mot


def counterfactual_correction(
    signal: np.ndarray,
    estimated_b0: float,
    estimated_b1: float,
    estimated_motion: float,
) -> np.ndarray:
    """Apply inverse corruption to correct a signal.

    Given estimated corruption parameters, apply the inverse operation
    to recover the (approximately) clean signal.

    Parameters
    ----------
    signal : ndarray (L,) complex64
        Corrupted signal.
    estimated_b0 : float
        Estimated B0 shift in Hz (will apply -Δf).
    estimated_b1 : float
        Estimated B1 scale (will apply 1/λ).
    estimated_motion : float
        Estimated motion shift in voxels (will apply -δ).

    Returns
    -------
    corrected : ndarray (L,) complex64
        Corrected signal.
    """
    corruptor = PhysicsCorruptor.__new__(PhysicsCorruptor)

    corrected = signal.copy()

    # Invert B1 (scale by 1/λ)
    if abs(estimated_b1 - 1.0) > 0.01:
        corrected = corrected / max(estimated_b1, 0.1)

    # Invert B0 (apply -Δf)
    if abs(estimated_b0) > 1.0:
        corrected = corruptor.apply_b0_off_resonance(corrected, -estimated_b0)

    # Invert motion (apply -δ)
    if abs(estimated_motion) > 0.5:
        corrected = corruptor.apply_kspace_motion_artifact(corrected, shift_y=-int(estimated_motion))

    return corrected


def run_counterfactual_experiment(
    model: nn.Module,
    val_signals: np.ndarray,
    val_targets: np.ndarray,
    val_b0: np.ndarray,
    val_b1: np.ndarray,
    val_motion: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: str = "cuda",
    n_samples: int = 500,
) -> Dict[str, float]:
    """Run the full counterfactual correction experiment.

    For each sample:
    1. Predict qMRI parameters + corruption severity
    2. Apply inverse corruption to the signal
    3. Re-predict on corrected signal
    4. Compare error before vs after correction

    Returns dict with before/after MAE and per-sample results.
    """
    from qMR_Robust.models.resnet1d import ResNet1D

    model.eval()
    n = min(n_samples, len(val_signals))

    # Convert to tensors
    sig_tensor = torch.from_numpy(val_signals[:n]).to(device)

    with torch.no_grad():
        out = model(sig_tensor)
        nig = out["nig"]
        severity = out["severity"]

    gamma_before = nig[..., 0].cpu().numpy()
    pred_b0 = severity["delta_f"].cpu().numpy()
    pred_b1 = severity["lambda_b1"].cpu().numpy() + 1.0  # model predicts deviation
    pred_motion = severity["delta_motion"].cpu().numpy()

    # Denormalize predictions
    gamma_before_denorm = gamma_before * target_std + target_mean
    targets_denorm = val_targets[:n] * target_std + target_mean

    mae_before = np.abs(targets_denorm - gamma_before_denorm).mean()

    # Counterfactual correction
    corrected_signals = []
    for i in range(n):
        sig_complex = val_signals[i, 0] + 1j * val_signals[i, 1]  # reconstruct complex
        corrected = counterfactual_correction(sig_complex, pred_b0[i], pred_b1[i], pred_motion[i])
        corrected_signals.append(np.stack([corrected.real, corrected.imag], axis=0).astype(np.float32))

    corrected_tensor = torch.from_numpy(np.stack(corrected_signals)).to(device)

    with torch.no_grad():
        out_corrected = model(corrected_tensor)
        nig_corrected = out_corrected["nig"]

    gamma_after = nig_corrected[..., 0].cpu().numpy()
    gamma_after_denorm = gamma_after * target_std + target_mean
    mae_after = np.abs(targets_denorm - gamma_after_denorm).mean()

    # Per-corruption-type analysis
    results = {
        "mae_before_ms": float(mae_before),
        "mae_after_ms": float(mae_after),
        "improvement_pct": float((mae_before - mae_after) / mae_before * 100),
        "n_samples": n,
    }

    # Severity estimation accuracy
    b0_mae = float(np.abs(pred_b0 - val_b0[:n]).mean())
    b1_mae = float(np.abs(pred_b1 - val_b1[:n]).mean())
    mot_mae = float(np.abs(pred_motion - val_motion[:n]).mean())
    results["b0_estimation_mae_hz"] = b0_mae
    results["b1_estimation_mae"] = b1_mae
    results["motion_estimation_mae_voxels"] = mot_mae

    return results
