#!/usr/bin/env python3
"""
compute_calibration_metrics.py — Compute ECE and NLL for all 20 seeds on real qMRLab data.

ECE: bins samples by predicted uncertainty, measures calibration within each bin.
NLL: negative log-likelihood under the NIG evidential model.
"""
import json
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.losses import nig_nll_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT_DIR = ROOT / "results" / "checkpoints"



def load_real_data():
    """Unit-safe qMRLab VFA loader (T1 in ms, zero-pad cross-sequence protocol)."""
    from qMR_Robust.data.loaders import load_qmrlab_vfa
    from pathlib import Path as _P
    qmrlab_dir = _P(__file__).resolve().parent.parent / "data" / "real" / "qmrlab" / "vfa_t1_data"
    data = load_qmrlab_vfa(qmrlab_dir, pad_mode="zeropad")
    return data.signals, data.t1_ms



def compute_ece(uncertainty, error, n_bins=15):
    """Expected Calibration Error: bins by uncertainty, measures correlation within bins."""
    bins = np.linspace(uncertainty.min(), uncertainty.max(), n_bins + 1)
    bin_idx = np.digitize(uncertainty, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    ece = 0.0
    n_total = len(uncertainty)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_unc = uncertainty[mask].mean()
        bin_err = error[mask].mean()
        ece += (mask.sum() / n_total) * abs(bin_unc - bin_err)
    return float(ece)


def compute_nll_for_seed(ckpt_path, X_real, real_t1, t_mean, t_std):
    """Compute NLL on real data for a single seed."""
    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X_real), 256):
            batch = torch.from_numpy(X_real[i:i+256]).to(DEVICE)
            out = model(batch).view(-1, 2, 4)
            all_gamma.append(out[..., 0].cpu().numpy())
            all_nu.append(out[..., 1].cpu().numpy())
            all_alpha.append(out[..., 2].cpu().numpy())
            all_beta.append(out[..., 3].cpu().numpy())

    gamma = np.concatenate(all_gamma)
    nu = np.concatenate(all_nu)
    alpha = np.concatenate(all_alpha)
    beta = np.concatenate(all_beta)

    gamma_d = gamma * t_std + t_mean
    epistemic = beta / (nu * (alpha - 1.0))
    max_ep = epistemic.max(axis=-1)
    error = np.abs(real_t1 - gamma_d[:, 0])

    # NLL: compute using the evidential loss
    gamma_t = torch.from_numpy(gamma).float().to(DEVICE)
    nu_t = torch.from_numpy(nu).float().to(DEVICE)
    alpha_t = torch.from_numpy(alpha).float().to(DEVICE)
    beta_t = torch.from_numpy(beta).float().to(DEVICE)

    # Normalize targets
    tgt_norm = (torch.from_numpy(real_t1).float().unsqueeze(1).to(DEVICE) - t_mean[0]) / t_std[0]
    tgt_norm = tgt_norm.expand_as(gamma_t)

    nig_params = torch.stack([gamma_t, nu_t, alpha_t, beta_t], dim=-1)
    nll = nig_nll_loss(tgt_norm, gamma_t, nu_t, alpha_t, beta_t).mean().item()

    rho, _ = spearmanr(max_ep, error) if max_ep.std() > 1e-12 else (0.0, 1.0)
    ece = compute_ece(max_ep, error)

    return {
        "nll": float(nll),
        "ece": float(ece),
        "rho": float(rho),
        "mean_epistemic": float(max_ep.mean()),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    # Load normalization from training data
    hf = h5py.File(mrf_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)
    t_mean = hf["parameters"][:n_train, :2].astype(np.float32).mean(0)
    t_std = hf["parameters"][:n_train, :2].astype(np.float32).std(0) + 1e-8
    hf.close()

    # Load real data
    print("Loading real qMRLab data...")
    X_real, real_t1 = load_real_data()
    print(f"  {len(X_real)} voxels loaded")

    # Process all seeds
    seed_data = json.load(open(FIG / "seed_study.json"))
    seeds = [r["seed"] for r in seed_data["results"]]

    results = []
    for seed in seeds:
        ckpt = CKPT_DIR / f"seed_study_{seed}.pt"
        if not ckpt.exists():
            print(f"  Seed {seed}: no checkpoint, skipping")
            continue

        metrics = compute_nll_for_seed(ckpt, X_real, real_t1, t_mean, t_std)
        metrics["seed"] = seed
        results.append(metrics)
        rho_label = "GOOD" if metrics["rho"] > 0.8 else "DEGENERATE" if metrics["rho"] < -0.1 else "MODERATE"
        print(f"  Seed {seed:3d}: NLL={metrics['nll']:.3f}  ECE={metrics['ece']:.4f}  ρ={metrics['rho']:.3f}  [{rho_label}]")

    # Summary
    good = [r for r in results if r["rho"] > 0.8]
    degenerate = [r for r in results if r["rho"] < -0.1]

    if good and degenerate:
        print(f"\n=== SUMMARY ===")
        print(f"Good seeds (n={len(good)}):")
        print(f"  NLL:  {np.mean([r['nll'] for r in good]):.3f} ± {np.std([r['nll'] for r in good]):.3f}")
        print(f"  ECE:  {np.mean([r['ece'] for r in good]):.4f} ± {np.std([r['ece'] for r in good]):.4f}")
        print(f"  ρ:    {np.mean([r['rho'] for r in good]):.3f} ± {np.std([r['rho'] for r in good]):.3f}")
        print(f"Degenerate seeds (n={len(degenerate)}):")
        print(f"  NLL:  {np.mean([r['nll'] for r in degenerate]):.3f} ± {np.std([r['nll'] for r in degenerate]):.3f}")
        print(f"  ECE:  {np.mean([r['ece'] for r in degenerate]):.4f} ± {np.std([r['ece'] for r in degenerate]):.4f}")
        print(f"  ρ:    {np.mean([r['rho'] for r in degenerate]):.3f} ± {np.std([r['rho'] for r in degenerate]):.3f}")

    # Save
    with open(FIG / "calibration_metrics.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {FIG / 'calibration_metrics.json'}")


if __name__ == "__main__":
    main()
