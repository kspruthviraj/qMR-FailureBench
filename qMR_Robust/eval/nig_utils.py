"""Helpers for NIG parameter → uncertainty conversion (numerically stable)."""

from __future__ import annotations

import numpy as np
import torch

_EPS = 1e-6


def nig_epistemic_np(nu, alpha, beta) -> np.ndarray:
    """Epistemic uncertainty β / (ν (α - 1)), NumPy, stable."""
    nu = np.clip(np.asarray(nu, dtype=np.float64), _EPS, None)
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 1.0 + _EPS, None)
    beta = np.clip(np.asarray(beta, dtype=np.float64), _EPS, None)
    return (beta / (nu * (alpha - 1.0))).astype(np.float64)


def nig_aleatoric_np(alpha, beta) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 1.0 + _EPS, None)
    beta = np.clip(np.asarray(beta, dtype=np.float64), _EPS, None)
    return (beta / (alpha - 1.0)).astype(np.float64)


def nig_predictive_var_np(nu, alpha, beta) -> np.ndarray:
    """β(1+ν) / (ν(α-1))."""
    nu = np.clip(np.asarray(nu, dtype=np.float64), _EPS, None)
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 1.0 + _EPS, None)
    beta = np.clip(np.asarray(beta, dtype=np.float64), _EPS, None)
    return (beta * (1.0 + nu) / (nu * (alpha - 1.0))).astype(np.float64)


def denorm_gamma(gamma: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Map normalized NIG mean to physical units (ms)."""
    return gamma * std + mean


@torch.no_grad()
def predict_nig_batches(model, signals: np.ndarray, device: str, batch_size: int = 256):
    """Run model on (N, 2, L) float array → dict of numpy NIG arrays."""
    model.eval()
    all_g, all_nu, all_a, all_b = [], [], [], []
    n = len(signals)
    for i in range(0, n, batch_size):
        x = torch.from_numpy(signals[i : i + batch_size]).to(device)
        out = model(x)
        if isinstance(out, dict):
            nig = out["nig"]
        else:
            nig = out.view(out.shape[0], -1, 4)
        all_g.append(nig[..., 0].cpu().numpy())
        all_nu.append(nig[..., 1].cpu().numpy())
        all_a.append(nig[..., 2].cpu().numpy())
        all_b.append(nig[..., 3].cpu().numpy())
    return {
        "gamma": np.concatenate(all_g, axis=0),
        "nu": np.concatenate(all_nu, axis=0),
        "alpha": np.concatenate(all_a, axis=0),
        "beta": np.concatenate(all_b, axis=0),
    }
