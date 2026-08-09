"""
Calibration and Failure Detection Metrics.

Implements:
  - Expected Calibration Error (ECE) for regression
  - Negative Log-Likelihood (NLL) under Gaussian assumption
  - Continuous Ranked Probability Score (CRPS)
  - Reliability diagram (binned uncertainty vs error)
  - Failure detection AUROC / AUPRC
  - Selective prediction (coverage vs error)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Calibration metrics
# ──────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    uncertainties: np.ndarray,
    residuals: np.ndarray,
    n_bins: int = 15,
    normalize: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """ECE for regression: how well does predicted uncertainty match actual error?

    Bins samples by predicted uncertainty, then compares mean predicted
    uncertainty vs mean actual residual in each bin.

    Parameters
    ----------
    normalize :
        If True (default), both uncertainty and residual are divided by the
        mean residual so ECE is dimensionless and not dominated by ms-scale
        absolute errors (legacy raw ECE could exceed 1000).

    Returns
    -------
    ece : float
        Weighted average absolute gap (normalized if ``normalize=True``).
    bin_means_pred : ndarray (n_bins,)
    bin_means_actual : ndarray (n_bins,)
    """
    unc_flat = uncertainties.flatten().astype(np.float64)
    res_flat = residuals.flatten().astype(np.float64)

    valid = np.isfinite(unc_flat) & np.isfinite(res_flat) & (unc_flat >= 0) & (res_flat >= 0)
    unc_flat = unc_flat[valid]
    res_flat = res_flat[valid]

    if len(unc_flat) == 0:
        return 0.0, np.zeros(n_bins), np.zeros(n_bins)

    if normalize:
        scale = max(float(res_flat.mean()), 1e-8)
        unc_flat = unc_flat / scale
        res_flat = res_flat / scale

    # Use quantile-based bins for equal population
    bin_edges = np.percentile(unc_flat, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-8
    bin_edges[0] -= 1e-8

    bin_indices = np.digitize(unc_flat, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means_pred = np.zeros(n_bins)
    bin_means_actual = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_means_pred[b] = unc_flat[mask].mean()
            bin_means_actual[b] = res_flat[mask].mean()
            bin_counts[b] = mask.sum()

    total = bin_counts.sum()
    weights = bin_counts / max(total, 1)
    ece = float(np.sum(weights * np.abs(bin_means_pred - bin_means_actual)))

    return ece, bin_means_pred, bin_means_actual


def gaussian_nll(
    mean: np.ndarray, variance: np.ndarray, target: np.ndarray,
) -> float:
    """Negative log-likelihood under a Gaussian with given mean and variance."""
    var = np.clip(variance, 1e-8, None)
    nll = 0.5 * (np.log(2 * np.pi * var) + (target - mean).pow(2) / var) if hasattr(target, 'pow') \
        else 0.5 * (np.log(2 * np.pi * var) + (target - mean) ** 2 / var)
    return float(np.mean(nll))


def continuous_ranked_probability_score(
    samples: np.ndarray, target: np.ndarray,
) -> float:
    """CRPS for ensemble or MC-Dropout samples.

    Parameters
    ----------
    samples : ndarray (T, N, D)
        T prediction samples for N data points, D targets.
    target : ndarray (N, D)
        Ground truth.
    """
    T = samples.shape[0]
    # Term 1: E|X - y|
    term1 = np.mean(np.abs(samples - target[np.newaxis]), axis=0)
    # Term 2: 0.5 * E|X - X'|
    diffs = []
    for i in range(min(T, 20)):  # subsample for efficiency
        j = np.random.randint(T)
        diffs.append(np.abs(samples[i] - samples[j]))
    term2 = 0.5 * np.mean(diffs, axis=0)
    return float(np.mean(term1 - term2))


# ──────────────────────────────────────────────────────────────────────────────
# Failure detection metrics
# ──────────────────────────────────────────────────────────────────────────────

def failure_detection_metrics(
    epistemic_unc: np.ndarray,
    residuals: np.ndarray,
    tolerance: float,
) -> Dict[str, float]:
    """Compute AUROC and AUPRC for failure detection.

    A sample is a 'failure' if its residual exceeds the tolerance.
    The 'score' for detection is the epistemic uncertainty.
    """
    max_epistemic = epistemic_unc.max(axis=-1) if epistemic_unc.ndim > 1 else epistemic_unc
    max_residual = residuals.max(axis=-1) if residuals.ndim > 1 else residuals

    # Binary labels: failure = residual > tolerance
    labels = (max_residual > tolerance).astype(int)

    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc": float("nan"), "auprc": float("nan"), "tolerance": tolerance}

    # Clip epistemic for numerical stability
    scores = np.clip(max_epistemic, 0, np.percentile(max_epistemic, 99.9))

    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    return {"auroc": auroc, "auprc": auprc, "tolerance": tolerance}


def compute_sensitivity_specificity(
    epistemic_unc: np.ndarray,
    residuals: np.ndarray,
    threshold: float,
    tolerance: float,
) -> Dict[str, float]:
    """Compute sensitivity and specificity at a given threshold."""
    max_epistemic = epistemic_unc.max(axis=-1) if epistemic_unc.ndim > 1 else epistemic_unc
    max_residual = residuals.max(axis=-1) if residuals.ndim > 1 else residuals

    actual_fail = max_residual > tolerance
    predicted_fail = max_epistemic > threshold

    tp = (actual_fail & predicted_fail).sum()
    fn = (actual_fail & ~predicted_fail).sum()
    fp = (~actual_fail & predicted_fail).sum()
    tn = (~actual_fail & ~predicted_fail).sum()

    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    precision = float(tp / max(tp + fp, 1))
    f1 = float(2 * precision * sensitivity / max(precision + sensitivity, 1e-8))

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "threshold": threshold,
        "tolerance": tolerance,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Selective prediction
# ──────────────────────────────────────────────────────────────────────────────

def selective_prediction_curve(
    epistemic_unc: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    coverage_levels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RMSE at different coverage levels (reject high-uncertainty voxels).

    Parameters
    ----------
    epistemic_unc : ndarray (N, D)
    predictions : ndarray (N, D)
    targets : ndarray (N, D)
    coverage_levels : ndarray, optional
        Fraction of samples to keep (default: 0.5 to 1.0 in 0.05 steps).

    Returns
    -------
    coverages : ndarray (K,)
    rmses : ndarray (K,)
    """
    if coverage_levels is None:
        coverage_levels = np.arange(0.5, 1.01, 0.05)

    max_unc = epistemic_unc.max(axis=-1) if epistemic_unc.ndim > 1 else epistemic_unc
    residuals = np.abs(targets - predictions)
    max_resid = residuals.max(axis=-1) if residuals.ndim > 1 else residuals

    rmses = []
    for cov in coverage_levels:
        n_keep = max(1, int(len(max_unc) * cov))
        # Keep samples with lowest uncertainty
        idx = np.argsort(max_unc)[:n_keep]
        rmse = float(np.sqrt(np.mean(max_resid[idx] ** 2)))
        rmses.append(rmse)

    return coverage_levels, np.array(rmses)


# ──────────────────────────────────────────────────────────────────────────────
# OOD Severity Curves
# ──────────────────────────────────────────────────────────────────────────────

def severity_curve(
    severities: np.ndarray,
    mean_epistemics: np.ndarray,
    mean_residuals: np.ndarray,
    corruption_name: str,
    output_dir: Path,
) -> None:
    """Plot epistemic uncertainty and error vs corruption severity."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "steelblue"
    ax1.set_xlabel(f"{corruption_name} Severity")
    ax1.set_ylabel("Mean Epistemic Uncertainty", color=color1)
    ax1.plot(severities, mean_epistemics, "o-", color=color1, linewidth=2, markersize=6)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.set_ylabel("Mean Absolute Residual", color=color2)
    ax2.plot(severities, mean_residuals, "s--", color=color2, linewidth=2, markersize=6)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(f"OOD Severity Curve — {corruption_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"severity_curve_{corruption_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2D Phantom Brain Maps
# ──────────────────────────────────────────────────────────────────────────────

def generate_2d_phantom(
    grid_size: int = 64,
    n_tissues: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic 2D brain phantom with tissue-specific T1/T2 values.

    Returns a dict with:
        't1_map', 't2_map', 'mask', 'tissue_labels',
        'signals', 'corrupted_signals', 'corruption_map'
    """
    rng = np.random.RandomState(seed)

    # Create circular brain mask
    y, x = np.ogrid[-grid_size // 2:grid_size // 2, -grid_size // 2:grid_size // 2]
    mask = (x ** 2 + y ** 2) < (grid_size // 2 - 2) ** 2

    # Tissue types: CSF, WM, GM, Lesion, Boundary
    tissue_t1 = [4000.0, 800.0, 1200.0, 1500.0, 1000.0]  # ms
    tissue_t2 = [2000.0, 80.0, 100.0, 120.0, 90.0]  # ms

    # Create tissue map with concentric regions
    tissue_map = np.zeros((grid_size, grid_size), dtype=int)
    dist = np.sqrt(x.astype(float) ** 2 + y.astype(float) ** 2)
    tissue_map[mask & (dist < grid_size * 0.15)] = 0  # CSF center
    tissue_map[mask & (dist >= grid_size * 0.15) & (dist < grid_size * 0.25)] = 1  # WM
    tissue_map[mask & (dist >= grid_size * 0.25) & (dist < grid_size * 0.38)] = 2  # GM
    tissue_map[mask & (dist >= grid_size * 0.38)] = 3  # Outer

    # Add a lesion
    ly, lx = grid_size // 4, grid_size // 3
    lesion_mask = ((y - ly) ** 2 + (x - lx) ** 2) < 8
    tissue_map[lesion_mask & mask] = 3

    t1_map = np.zeros((grid_size, grid_size))
    t2_map = np.zeros((grid_size, grid_size))
    for i, (t1, t2) in enumerate(zip(tissue_t1, tissue_t2)):
        t1_map[tissue_map == i] = t1 + rng.randn((tissue_map == i).sum()) * 50
        t2_map[tissue_map == i] = t2 + rng.randn((tissue_map == i).sum()) * 5

    t1_map[~mask] = 0
    t2_map[~mask] = 0

    # Generate synthetic MRF signals per voxel (simplified)
    n_time = 100
    signals = np.zeros((grid_size, grid_size, n_time), dtype=np.complex64)
    corruption_map = np.zeros((grid_size, grid_size), dtype=float)

    for iy in range(grid_size):
        for ix in range(grid_size):
            if not mask[iy, ix]:
                continue
            t1, t2 = t1_map[iy, ix], t2_map[iy, ix]
            fa = np.linspace(5, 70, n_time) + rng.randn(n_time) * 2
            tr = np.ones(n_time) * 12.0
            fa_rad = np.deg2rad(fa)
            sig = np.zeros(n_time, dtype=np.complex128)
            mz = 1.0
            for t in range(n_time):
                mz_pre = mz
                mz = mz_pre * np.cos(fa_rad[t])
                mxy = mz_pre * np.sin(fa_rad[t]) * np.exp(-2.0 / max(t2, 1e-3))
                mz = mz * np.exp(-tr[t] / max(t1, 1e-3)) + (1 - np.exp(-tr[t] / max(t1, 1e-3)))
                sig[t] = mxy
            noise = (rng.randn(n_time) + 1j * rng.randn(n_time)) * 0.01
            signals[iy, ix] = (sig + noise).astype(np.complex64)

    # Apply entangled corruption to ~30% of voxels
    corrupted_signals = signals.copy()
    corruption_mask = rng.random((grid_size, grid_size)) < 0.3
    corruption_mask &= mask

    from qMR_Robust.simulators.corruptor import PhysicsCorruptor
    cfg_stub = {"simulation": {"corruptor": {
        "b0_off_resonance_range": [-80, 80],
        "b1_transmit_scale_range": [0.6, 1.4],
        "motion_kspace_max_shift": [8, 8],
        "motion_kspace_rotation_range": [-15, 15],
        "probability_weights": {"b0": 1.0, "b1": 1.0, "motion": 1.0},
    }}}
    corruptor = PhysicsCorruptor(cfg_stub)

    for iy in range(grid_size):
        for ix in range(grid_size):
            if corruption_mask[iy, ix]:
                c_rng = np.random.RandomState(seed + iy * grid_size + ix)
                corrupted, meta = corruptor.corrupt_mrf_signal(signals[iy, ix], c_rng)
                corrupted_signals[iy, ix] = corrupted
                corruption_map[iy, ix] = abs(meta["b0_hz"]) + abs(meta["b1_scale"] - 1.0) * 50 + abs(meta["motion_shift"])

    return {
        "t1_map": t1_map,
        "t2_map": t2_map,
        "mask": mask,
        "tissue_labels": tissue_map,
        "signals": signals,
        "corrupted_signals": corrupted_signals,
        "corruption_mask": corruption_mask,
        "corruption_map": corruption_map,
    }


def plot_2d_brain_maps(
    phantom: Dict[str, np.ndarray],
    predicted_t1: np.ndarray,
    predicted_t2: np.ndarray,
    epistemic_t1: np.ndarray,
    epistemic_t2: np.ndarray,
    failure_mask: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate a multi-panel brain map diagnostic."""
    mask = phantom["mask"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: T1
    vmax_t1 = phantom["t1_map"][mask].max()
    im0 = axes[0, 0].imshow(phantom["t1_map"], cmap="hot", vmin=0, vmax=vmax_t1)
    axes[0, 0].set_title("Ground Truth T₁ (ms)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    pred_t1_masked = np.where(mask, predicted_t1, np.nan)
    im1 = axes[0, 1].imshow(pred_t1_masked, cmap="hot", vmin=0, vmax=vmax_t1)
    axes[0, 1].set_title("Predicted T₁ (ms)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    err_t1 = np.where(mask, np.abs(phantom["t1_map"] - predicted_t1), np.nan)
    im2 = axes[0, 2].imshow(err_t1, cmap="YlOrRd", vmin=0, vmax=vmax_t1 * 0.3)
    axes[0, 2].set_title("|T₁ Error| (ms)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    ep_t1_masked = np.where(mask, epistemic_t1, np.nan)
    im3 = axes[0, 3].imshow(ep_t1_masked, cmap="viridis")
    axes[0, 3].set_title("T₁ Epistemic Uncertainty")
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # Row 2: T2
    vmax_t2 = phantom["t2_map"][mask].max()
    im4 = axes[1, 0].imshow(phantom["t2_map"], cmap="hot", vmin=0, vmax=vmax_t2)
    axes[1, 0].set_title("Ground Truth T₂ (ms)")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    pred_t2_masked = np.where(mask, predicted_t2, np.nan)
    im5 = axes[1, 1].imshow(pred_t2_masked, cmap="hot", vmin=0, vmax=vmax_t2)
    axes[1, 1].set_title("Predicted T₂ (ms)")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    err_t2 = np.where(mask, np.abs(phantom["t2_map"] - predicted_t2), np.nan)
    im6 = axes[1, 2].imshow(err_t2, cmap="YlOrRd", vmin=0, vmax=vmax_t2 * 0.3)
    axes[1, 2].set_title("|T₂ Error| (ms)")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    # Failure mask overlay
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [0.2, 0.8, 0.2, 0.3]  # green for safe
    overlay[failure_mask & mask] = [1.0, 0.0, 0.0, 0.7]  # red for failure
    axes[1, 3].imshow(phantom["t1_map"], cmap="gray", alpha=0.5, vmin=0, vmax=vmax_t1)
    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title("Failure Mask (Red = Expected to Fail)")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle("2D Phantom Brain Maps: T₁/T₂ Prediction with Evidential Failure Forecasting", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "brain_maps_2d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Brain maps saved → %s", output_dir / "brain_maps_2d.png")


# ──────────────────────────────────────────────────────────────────────────────
# Reliability diagram plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    bin_means_pred: np.ndarray,
    bin_means_actual: np.ndarray,
    ece: float,
    split_name: str,
    output_dir: Path,
) -> None:
    """Plot reliability diagram for regression calibration."""
    n_bins = len(bin_means_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    x = np.arange(n_bins)
    width = 0.35
    ax.bar(x - width / 2, bin_means_pred, width, label="Predicted Uncertainty", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, bin_means_actual, width, label="Actual |Residual|", color="coral", alpha=0.8)

    # Perfect calibration line
    max_val = max(bin_means_pred.max(), bin_means_actual.max(), 1e-6)
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")

    ax.set_xlabel("Bin Index (sorted by predicted uncertainty)")
    ax.set_ylabel("Value")
    ax.set_title(f"Reliability Diagram — {split_name} (ECE = {ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_reliability_diagram.png", dpi=150)
    plt.close(fig)


def plot_failure_detection_roc(
    epistemic_unc: np.ndarray,
    residuals: np.ndarray,
    tolerances: List[float],
    split_name: str,
    output_dir: Path,
) -> None:
    """Plot ROC curves for failure detection at multiple tolerance levels."""
    from sklearn.metrics import roc_curve

    max_epistemic = epistemic_unc.max(axis=-1) if epistemic_unc.ndim > 1 else epistemic_unc
    max_residual = residuals.max(axis=-1) if residuals.ndim > 1 else residuals

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(tolerances)))

    for tol, color in zip(tolerances, colors):
        labels = (max_residual > tol).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        fpr, tpr, _ = roc_curve(labels, max_epistemic)
        auroc = roc_auc_score(labels, max_epistemic)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"τ={tol:.2f} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Failure Detection ROC — {split_name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_failure_detection_roc.png", dpi=150)
    plt.close(fig)


def plot_selective_prediction(
    coverages: np.ndarray,
    rmses: np.ndarray,
    method_name: str,
    split_name: str,
    output_dir: Path,
) -> None:
    """Plot coverage vs RMSE for selective prediction."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(coverages * 100, rmses, "o-", linewidth=2, markersize=6, color="steelblue")
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Selective Prediction — {split_name} ({method_name})")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_selective_prediction_{method_name}.png", dpi=150)
    plt.close(fig)
