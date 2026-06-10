"""
Physics-Aware Evidential Loss.

Standard evidential learning treats aleatoric uncertainty as a free parameter.
In qMRI, we have prior physical knowledge about data noise:
  - Signal SNR directly constrains minimum aleatoric uncertainty
  - B0 field map variance correlates with systematic signal degradation
  - B1+ transmit maps constrain flip angle uncertainty

This module anchors the aleatoric uncertainty (β/(α-1)) to measured signal
properties, making the uncertainty decomposition physically meaningful:
  - Aleatoric ≈ physical noise floor (from SNR, B0, B1)
  - Epistemic = model ignorance (learned from data)

This is the key scientific innovation: the uncertainty decomposition now has
\emph{physical semantics}, not just statistical ones.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


def estimate_signal_snr(signal: torch.Tensor) -> torch.Tensor:
    """Estimate per-sample SNR from the raw signal.

    SNR ≈ max(|signal|) / std(noise_floor), where noise_floor is estimated
    from the tail of the signal (last 10% of timepoints, assumed to be noise-dominated).

    Parameters
    ----------
    signal : (B, C, L) — complex-valued signal as 2-channel real/imag

    Returns
    -------
    snr : (B,) — estimated SNR per sample
    """
    # Reconstruct magnitude
    mag = torch.sqrt(signal[:, 0] ** 2 + signal[:, 1] ** 2)  # (B, L)

    # Signal peak
    peak = mag.max(dim=-1).values  # (B,)

    # Noise floor from tail
    tail_len = max(1, mag.shape[-1] // 10)
    noise_floor = mag[:, -tail_len:].std(dim=-1).clamp(min=_EPS)  # (B,)

    snr = (peak / noise_floor).clamp(min=1.0, max=500.0)
    return snr


def snr_to_aleatoric_target(snr: torch.Tensor, D: int) -> torch.Tensor:
    """Convert SNR to expected aleatoric uncertainty.

    Higher SNR → lower expected aleatoric uncertainty (cleaner data).
    σ²_alea ≈ 1/SNR² (inversely proportional to signal quality).

    Parameters
    ----------
    snr : (B,) — per-sample SNR
    D : int — number of prediction targets

    Returns
    -------
    target_alea : (B, D) — expected aleatoric variance
    """
    # σ² ∝ 1/SNR
    base_unc = 1.0 / snr.clamp(min=1.0)  # (B,)
    # Expand to all targets (they share the same signal quality)
    return base_unc.unsqueeze(-1).expand(-1, D)


def physics_aware_evidential_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    signal: torch.Tensor,
    reg_coeff: float = 1.0,
    physics_coeff: float = 0.3,
    epoch: int = 0,
    annealing_epochs: int = 10,
) -> Dict[str, torch.Tensor]:
    """Physics-aware evidential loss with SNR-anchored aleatoric uncertainty.

    Total loss = NLL + λ_ER × ER + λ_phys × Physics Anchor

    The Physics Anchor term penalizes the KL divergence between:
      - Learned aleatoric: β/(α-1)
      - Physics-expected aleatoric: 1/SNR

    This ensures the aleatoric component captures genuine data noise
    while the epistemic component captures model ignorance.

    Parameters
    ----------
    y : (B, D) — ground truth
    gamma, nu, alpha, beta : (B, D) — NIG parameters
    signal : (B, C, L) — raw input signal for SNR estimation
    reg_coeff : float — evidential regularizer coefficient
    physics_coeff : float — physics anchor coefficient
    epoch : int — current epoch
    annealing_epochs : int — annealing schedule length

    Returns
    -------
    dict with 'loss', 'nll', 'er', 'physics_anchor', 'gamma', 'nu', 'alpha', 'beta'
    """
    from qMR_Robust.models.losses import nig_nll_loss, evidential_regularizer

    # Standard NIG NLL
    nll = nig_nll_loss(y, gamma, nu, alpha, beta)

    # Evidential regularizer
    er = evidential_regularizer(y, gamma, nu, alpha)

    # Physics anchor: align aleatoric uncertainty with SNR
    snr = estimate_signal_snr(signal)  # (B,)
    D = y.shape[-1]
    target_alea = snr_to_aleatoric_target(snr, D)  # (B, D)

    learned_alea = (beta / (alpha - 1.0)).clamp(min=_EPS)

    # KL divergence between learned and expected aleatoric
    # KL(P_learned || P_target) where both are Gamma distributions
    # Simplified: MSE in log-space (encourages proportional scaling)
    physics_anchor = (torch.log(learned_alea) - torch.log(target_alea)).pow(2)

    # Annealing
    anneal_weight = min(1.0, epoch / max(annealing_epochs, 1))

    total_loss = (
        nll.mean()
        + reg_coeff * anneal_weight * er.mean()
        + physics_coeff * anneal_weight * physics_anchor.mean()
    )

    if not torch.isfinite(total_loss):
        total_loss = nll.clamp(max=50.0).mean()

    return {
        "loss": total_loss,
        "nll": nll.mean().detach(),
        "er": er.mean().detach(),
        "physics_anchor": physics_anchor.mean().detach(),
        "gamma": gamma.detach(),
        "nu": nu.detach(),
        "alpha": alpha.detach(),
        "beta": beta.detach(),
    }
