"""
Forecaster — Evidential Uncertainty Quantification and Failure Prediction.

Takes NIG outputs from an evidential neural network and computes:
  1. Aleatoric Uncertainty  (data noise):      β / (α − 1)
  2. Epistemic Uncertainty  (model ignorance): β / (ν · (α − 1))

Includes evaluation loops for cMRF phantom and Big GABA test sets, plotting
of epistemic uncertainty vs. absolute residual error, and a thresholding
function that flags voxels as "Expected to Fail."
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty decomposition
# ──────────────────────────────────────────────────────────────────────────────

def aleatoric_uncertainty(
    beta: torch.Tensor, alpha: torch.Tensor,
) -> torch.Tensor:
    """Data (aleatoric) uncertainty: β / (α − 1).

    Parameters
    ----------
    beta : Tensor (..., D)
        NIG scale parameter.
    alpha : Tensor (..., D)
        NIG evidence for variance (α > 1).

    Returns
    -------
    Tensor (..., D)
        Aleatoric uncertainty estimate.
    """
    return beta / (alpha - 1.0)


def epistemic_uncertainty(
    beta: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor,
) -> torch.Tensor:
    """Model (epistemic) uncertainty: β / (ν · (α − 1)).

    Parameters
    ----------
    beta : Tensor (..., D)
        NIG scale parameter.
    nu : Tensor (..., D)
        NIG evidence for the mean.
    alpha : Tensor (..., D)
        NIG evidence for variance (α > 1).

    Returns
    -------
    Tensor (..., D)
        Epistemic uncertainty estimate.
    """
    return beta / (nu * (alpha - 1.0))


def expected_nig_variance(
    nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    """Expected predictive variance under the NIG posterior: β(1 + ν) / (ν(α − 1))."""
    return beta * (1.0 + nu) / (nu * (alpha - 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Failure flagging
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FailureFlag:
    """Result of failure-thresholding a single prediction."""
    voxel_idx: int
    predicted_mean: float
    epistemic_unc: float
    aleatoric_unc: float
    expected_to_fail: bool
    absolute_residual: Optional[float] = None


def flag_failures(
    epistemic: np.ndarray,
    threshold: float,
    predicted_mean: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[FailureFlag]]:
    """Flag voxels whose epistemic uncertainty exceeds the clinical tolerance.

    Parameters
    ----------
    epistemic : ndarray (N, D)
        Per-voxel per-target epistemic uncertainty.
    threshold : float
        Clinical tolerance; voxels with max epistemic unc > threshold are flagged.
    predicted_mean : ndarray (N, D), optional
        Predicted means (for reporting).
    ground_truth : ndarray (N, D), optional
        Ground-truth targets (for computing residuals).

    Returns
    -------
    mask : ndarray (N,)
        Boolean mask of voxels expected to fail.
    flags : list of FailureFlag
        Detailed per-voxel failure information.
    """
    n = epistemic.shape[0]
    max_epistemic = epistemic.max(axis=-1)
    mask = max_epistemic > threshold

    flags = []
    for i in range(n):
        resid = None
        if predicted_mean is not None and ground_truth is not None:
            resid = float(np.abs(predicted_mean[i] - ground_truth[i]).mean())
        flags.append(FailureFlag(
            voxel_idx=i,
            predicted_mean=float(predicted_mean[i].mean()) if predicted_mean is not None else 0.0,
            epistemic_unc=float(max_epistemic[i]),
            aleatoric_unc=0.0,
            expected_to_fail=bool(mask[i]),
            absolute_residual=resid,
        ))
    return mask, flags


# ──────────────────────────────────────────────────────────────────────────────
# Forecaster — evaluation orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class Forecaster:
    """Evaluates an evidential model and produces failure-forecast outputs.

    Parameters
    ----------
    model : nn.Module
        Trained evidential neural network (outputs NIG parameters).
    device : str
        Torch device string.
    epistemic_threshold : float
        Clinical tolerance for failure flagging.
    output_dir : str
        Directory for saving plots and metrics JSON.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        epistemic_threshold: float = 0.1,
        output_dir: str = "results/forecast",
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = epistemic_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _predict_batch(
        self, signals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a batch through the model and decompose NIG parameters."""
        signals = signals.to(self.device)
        nig = self.model(signals)  # (B, D, 4)

        gamma = nig[..., 0].cpu()
        nu = nig[..., 1].cpu()
        alpha = nig[..., 2].cpu()
        beta = nig[..., 3].cpu()
        return gamma, nu, alpha, beta

    def evaluate(
        self,
        dataloader: DataLoader,
        split_name: str = "test",
    ) -> Dict[str, Any]:
        """Run full evaluation on a dataset.

        Returns a dict with aggregated metrics and per-sample predictions.
        """
        all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
        all_targets = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                signals, targets = batch[0], batch[1]
            else:
                raise ValueError("Expected (signals, targets) tuple from dataloader")

            gamma, nu, alpha, beta = self._predict_batch(signals)
            all_gamma.append(gamma)
            all_nu.append(nu)
            all_alpha.append(alpha)
            all_beta.append(beta)
            all_targets.append(targets.cpu() if isinstance(targets, torch.Tensor) else targets)

        gamma = torch.cat(all_gamma, dim=0).numpy()
        nu = torch.cat(all_nu, dim=0).numpy()
        alpha = torch.cat(all_alpha, dim=0).numpy()
        beta = torch.cat(all_beta, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy() if isinstance(all_targets[0], torch.Tensor) \
            else np.concatenate(all_targets, axis=0)

        aleatoric = aleatoric_uncertainty(
            torch.from_numpy(beta), torch.from_numpy(alpha),
        ).numpy()
        epistemic = epistemic_uncertainty(
            torch.from_numpy(beta), torch.from_numpy(nu), torch.from_numpy(alpha),
        ).numpy()

        residuals = np.abs(targets - gamma)

        # Aggregate metrics
        mae = float(np.mean(residuals))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        mask, flags = flag_failures(
            epistemic, self.threshold,
            predicted_mean=gamma, ground_truth=targets,
        )
        n_flagged = int(mask.sum())
        failure_rate = float(mask.mean())

        metrics = {
            "split": split_name,
            "n_samples": len(gamma),
            "mae": mae,
            "rmse": rmse,
            "mean_aleatoric": float(np.mean(aleatoric)),
            "mean_epistemic": float(np.mean(epistemic)),
            "n_flagged": n_flagged,
            "failure_rate": failure_rate,
            "epistemic_threshold": self.threshold,
        }

        self._plot_epistemic_vs_residual(epistemic, residuals, split_name)
        self._plot_uncertainty_histograms(aleatoric, epistemic, split_name)

        metrics_path = self.output_dir / f"{split_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved → %s", metrics_path)

        return {
            "metrics": metrics,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "residuals": residuals,
            "failure_mask": mask,
            "flags": flags,
        }

    def _plot_epistemic_vs_residual(
        self, epistemic: np.ndarray, residuals: np.ndarray, split_name: str,
    ):
        """Scatter plot: predicted epistemic uncertainty vs actual absolute residual."""
        fig, axes = plt.subplots(1, epistemic.shape[-1], figsize=(7 * epistemic.shape[-1], 6), squeeze=False)
        target_names = ["T1", "T2"] if epistemic.shape[-1] == 2 else [f"Target {i}" for i in range(epistemic.shape[-1])]

        for d in range(epistemic.shape[-1]):
            ax = axes[0, d]
            ax.scatter(epistemic[:, d], residuals[:, d], alpha=0.3, s=8, c="steelblue", edgecolors="none")
            ax.axvline(self.threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={self.threshold}")
            ax.set_xlabel("Predicted Epistemic Uncertainty")
            ax.set_ylabel("Absolute Residual Error")
            ax.set_title(f"{split_name} — {target_names[d]}")
            ax.legend()

            # Pearson correlation
            if epistemic[:, d].std() > 1e-12 and residuals[:, d].std() > 1e-12:
                corr = np.corrcoef(epistemic[:, d], residuals[:, d])[0, 1]
                ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=11)

        fig.tight_layout()
        path = self.output_dir / f"{split_name}_epistemic_vs_residual.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Plot saved → %s", path)

    def _plot_uncertainty_histograms(
        self, aleatoric: np.ndarray, epistemic: np.ndarray, split_name: str,
    ):
        """Histogram of aleatoric and epistemic uncertainty distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(aleatoric.flatten(), bins=80, color="teal", alpha=0.7, edgecolor="white")
        axes[0].set_title(f"{split_name} — Aleatoric Uncertainty")
        axes[0].set_xlabel("β / (α − 1)")
        axes[0].set_ylabel("Count")

        axes[1].hist(epistemic.flatten(), bins=80, color="coral", alpha=0.7, edgecolor="white")
        axes[1].axvline(self.threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={self.threshold}")
        axes[1].set_title(f"{split_name} — Epistemic Uncertainty")
        axes[1].set_xlabel("β / (ν(α − 1))")
        axes[1].set_ylabel("Count")
        axes[1].legend()

        fig.tight_layout()
        path = self.output_dir / f"{split_name}_uncertainty_histograms.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Plot saved → %s", path)
