"""
Proper regression calibration metrics.

Unlike a raw |unc - err| ECE with mismatched scales, these metrics are
scale-aware and comparable across models.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, norm


def spearman_uncertainty_error(
    uncertainty: np.ndarray,
    error: np.ndarray,
) -> Tuple[float, float]:
    """Spearman ρ between uncertainty scores and absolute errors."""
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(error, dtype=np.float64).ravel()
    valid = np.isfinite(u) & np.isfinite(e)
    if valid.sum() < 10 or np.std(u[valid]) < 1e-12 or np.std(e[valid]) < 1e-12:
        return 0.0, 1.0
    rho, p = spearmanr(u[valid], e[valid])
    if not np.isfinite(rho):
        return 0.0, 1.0
    return float(rho), float(p)


def normalized_ece(
    uncertainty: np.ndarray,
    error: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Scale-normalized ECE in [0, ~1+].

    Both uncertainty and error are divided by the mean absolute error so the
    metric is dimensionless. Values near 0 indicate good scale calibration of
    rank-binned uncertainty vs error.
    """
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    e = np.asarray(error, dtype=np.float64).ravel()
    valid = np.isfinite(u) & np.isfinite(e) & (u >= 0) & (e >= 0)
    u, e = u[valid], e[valid]
    if len(u) < n_bins:
        return float("nan")

    scale = max(float(e.mean()), 1e-8)
    u_n = u / scale
    e_n = e / scale

    edges = np.percentile(u_n, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-8
    edges[0] -= 1e-8
    idx = np.clip(np.digitize(u_n, edges) - 1, 0, n_bins - 1)

    ece = 0.0
    n = len(u_n)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(u_n[m].mean() - e_n[m].mean())
    return float(ece)


def z_score_calibration(
    mean: np.ndarray,
    std: np.ndarray,
    target: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Check if (target-mean)/std ~ N(0,1).

    Returns RMS of |empirical_std_of_z - 1| across magnitude bins, plus
    fraction of points within 1σ / 2σ predictive intervals.
    """
    mu = np.asarray(mean, dtype=np.float64).ravel()
    s = np.asarray(std, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()
    valid = np.isfinite(mu) & np.isfinite(s) & np.isfinite(y) & (s > 1e-8)
    mu, s, y = mu[valid], s[valid], y[valid]
    if len(y) < 20:
        return {"z_rms": float("nan"), "cov_1sigma": float("nan"), "cov_2sigma": float("nan")}

    z = (y - mu) / s
    cov1 = float(np.mean(np.abs(z) <= 1.0))
    cov2 = float(np.mean(np.abs(z) <= 2.0))

    # Expected under N(0,1): ~0.683 / 0.954
    return {
        "z_rms": float(np.sqrt(np.mean(z ** 2))),  # should be ~1
        "cov_1sigma": cov1,
        "cov_2sigma": cov2,
        "expected_cov_1sigma": 2 * norm.cdf(1) - 1,
        "expected_cov_2sigma": 2 * norm.cdf(2) - 1,
    }


def gaussian_nll_numpy(
    mean: np.ndarray,
    variance: np.ndarray,
    target: np.ndarray,
) -> float:
    """Mean Gaussian NLL (nats)."""
    mu = np.asarray(mean, dtype=np.float64)
    var = np.clip(np.asarray(variance, dtype=np.float64), 1e-8, None)
    y = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(mu) & np.isfinite(var) & np.isfinite(y)
    if valid.sum() == 0:
        return float("nan")
    nll = 0.5 * (np.log(2 * np.pi * var[valid]) + (y[valid] - mu[valid]) ** 2 / var[valid])
    return float(np.mean(nll))


def categorize_rho(rho: float) -> str:
    """Shared taxonomy for seed studies."""
    if rho > 0.8:
        return "good"
    if rho > 0.3:
        return "moderate"
    if rho > -0.1:
        return "near_zero"
    return "degenerate"


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))
