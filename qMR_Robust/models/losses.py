"""
Evidential Loss Functions for Normal-Inverse-Gamma (NIG) Regression.

The NIG distribution is parameterised by (γ, ν, α, β):
  - γ : predicted mean
  - ν : evidence for the mean          (ν > 0, enforced via Softplus)
  - α : evidence for the variance      (α > 1, enforced via Softplus + 1)
  - β : scale parameter                (β > 0, enforced via Softplus)

References:
  * Sensoy et al., "Evidential Deep Learning to Quantify Classification
    Uncertainty", NeurIPS 2018.
  * Soleimany et al., "Evidential Deep Learning for Guided Molecular
    Property Prediction and Discovery", ACS Central Science 2021.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-6


def nig_nll_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood of the Normal-Inverse-Gamma predictive distribution.

    The NLL for a single observation y under NIG(γ, ν, α, β) is:

        0.5 * log(π / ν)
      - α * log(2 * β * (1 + ν))
      + (α + 0.5) * log(ν * (y − γ)² + 2 * β * (1 + ν))
      + log Γ(α)
      − log Γ(α + 0.5)

    Parameters
    ----------
    y : Tensor (B, D)
        Ground-truth targets.
    gamma : Tensor (B, D)
        Predicted mean.
    nu : Tensor (B, D)
        Evidence for the mean (strictly positive).
    alpha : Tensor (B, D)
        Evidence for the variance (strictly > 1).
    beta : Tensor (B, D)
        Scale parameter (strictly positive).

    Returns
    -------
    Tensor (B, D)
        Per-element NLL loss.
    """
    nu = nu.clamp(min=_EPS)
    alpha = alpha.clamp(min=1.0 + _EPS)
    beta = beta.clamp(min=_EPS)

    two_beta_nu = 2.0 * beta * (1.0 + nu)
    two_beta_nu = two_beta_nu.clamp(min=_EPS)
    error_sq = (y - gamma).pow(2)

    nll = (
        0.5 * torch.log(torch.tensor(torch.pi, device=y.device) / nu)
        - alpha * torch.log(two_beta_nu)
        + (alpha + 0.5) * torch.log(nu * error_sq + two_beta_nu)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    return nll.clamp(max=50.0)


def evidential_regularizer(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Evidential regularizer that penalises confident errors.

    The penalty is |y − γ| × (2ν + α), i.e. the absolute error scaled by
    the total evidence.  This drives the model to express high epistemic
    uncertainty (low evidence) when it encounters out-of-distribution
    physical artifacts, preventing overconfident wrong predictions.

    Parameters
    ----------
    y : Tensor (B, D)
        Ground-truth targets.
    gamma : Tensor (B, D)
        Predicted mean.
    nu : Tensor (B, D)
        Evidence for the mean.
    alpha : Tensor (B, D)
        Evidence for the variance.

    Returns
    -------
    Tensor (B, D)
        Per-element regulariser penalty.
    """
    abs_error = torch.abs(y - gamma)
    total_evidence = 2.0 * nu + alpha
    return abs_error * total_evidence


def evidential_regression_loss(
    y: torch.Tensor,
    nig_params: torch.Tensor,
    coeff: float = 1.0,
    epoch: int = 0,
    annealing_epochs: int = 10,
) -> dict:
    """Combined evidential regression loss with KL-based annealing.

    Parameters
    ----------
    y : Tensor (B, D)
        Ground-truth targets.
    nig_params : Tensor (B, D, 4)
        Stacked NIG parameters [γ, ν, α, β].
    coeff : float
        Coefficient for the evidential regularizer.
    epoch : int
        Current training epoch (for annealing).
    annealing_epochs : int
        Number of epochs over which to anneal the regularizer weight from 0→1.

    Returns
    -------
    dict with keys 'loss', 'nll', 'reg', 'gamma', 'nu', 'alpha', 'beta'.
    """
    gamma = nig_params[..., 0]
    nu = nig_params[..., 1]
    alpha = nig_params[..., 2]
    beta = nig_params[..., 3]

    nll = nig_nll_loss(y, gamma, nu, alpha, beta)
    reg = evidential_regularizer(y, gamma, nu, alpha)

    anneal_weight = min(1.0, epoch / max(annealing_epochs, 1))

    loss = nll.mean() + coeff * anneal_weight * reg.mean()

    if not torch.isfinite(loss):
        loss = nll.clamp(max=50.0).mean()

    return {
        "loss": loss,
        "nll": nll.mean().detach(),
        "reg": reg.mean().detach(),
        "gamma": gamma.detach(),
        "nu": nu.detach(),
        "alpha": alpha.detach(),
        "beta": beta.detach(),
    }
