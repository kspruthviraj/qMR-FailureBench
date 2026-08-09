#!/usr/bin/env python3
"""
run_training_dynamics.py — Log per-epoch metrics for all 20 seeds.

Captures: MAE, NLL, ECE, Spearman ρ, mean ν, α, β, epistemic uncertainty
on real qMRLab data at every epoch.

Goal: discover WHEN degenerate seeds diverge and whether collapse can be predicted early.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.losses import evidential_regression_loss, nig_nll_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT_DIR = ROOT / "results" / "checkpoints"
FIG.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class MRFMetaDataset(Dataset):
    def __init__(self, h5_path, split="train", train_ratio=0.8):
        hf = h5py.File(h5_path, "r")
        n = hf.attrs["n_signals"]
        n_train = int(n * train_ratio)
        s, e = (0, n_train) if split == "train" else (n_train, n)
        self.signals = np.stack(
            [hf["corrupted_signals"][s:e].real, hf["corrupted_signals"][s:e].imag], axis=1
        ).astype(np.float32)
        self.params = hf["parameters"][s:e, :2].astype(np.float32)
        hf.close()
        if split == "train":
            self.mean = self.params.mean(0)
            self.std = self.params.std(0) + 1e-8
        else:
            self.mean = np.zeros(2, dtype=np.float32)
            self.std = np.ones(2, dtype=np.float32)

    def set_norm(self, mean, std):
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.signals[i]),
            (torch.from_numpy(self.params[i]) - self.mean) / self.std,
        )



def load_real_data():
    """Unit-safe qMRLab VFA loader (T1 in ms, zero-pad cross-sequence protocol)."""
    from qMR_Robust.data.loaders import load_qmrlab_vfa
    from pathlib import Path as _P
    qmrlab_dir = _P(__file__).resolve().parent.parent / "data" / "real" / "qmrlab" / "vfa_t1_data"
    data = load_qmrlab_vfa(qmrlab_dir, pad_mode="zeropad")
    return data.signals, data.t1_ms



def evaluate_real_data(model, X_real, real_t1, t_mean, t_std):
    """Evaluate model on real qMRLab data and return metrics."""
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

    rho, _ = spearmanr(max_ep, error) if max_ep.std() > 1e-12 else (0.0, 1.0)

    # NLL
    gamma_t = torch.from_numpy(gamma).float().to(DEVICE)
    nu_t = torch.from_numpy(nu).float().to(DEVICE)
    alpha_t = torch.from_numpy(alpha).float().to(DEVICE)
    beta_t = torch.from_numpy(beta).float().to(DEVICE)
    tgt_norm = (torch.from_numpy(real_t1).float().unsqueeze(1).to(DEVICE) - t_mean[0]) / t_std[0]
    tgt_norm = tgt_norm.expand_as(gamma_t)
    nll = nig_nll_loss(tgt_norm, gamma_t, nu_t, alpha_t, beta_t).mean().item()

    # ECE
    bins = np.linspace(max_ep.min(), max_ep.max(), 16)
    bin_idx = np.clip(np.digitize(max_ep, bins) - 1, 0, 14)
    ece = 0.0
    for b in range(15):
        mask = bin_idx == b
        if mask.sum() > 0:
            ece += (mask.sum() / len(max_ep)) * abs(max_ep[mask].mean() - error[mask].mean())

    return {
        "mae": float(np.mean(error)),
        "nll": float(nll),
        "ece": float(ece),
        "rho": float(rho),
        "mean_nu": float(np.mean(nu)),
        "mean_alpha": float(np.mean(alpha)),
        "mean_beta": float(np.mean(beta)),
        "mean_epistemic": float(max_ep.mean()),
    }


def train_seed_with_dynamics(seed, mrf_path, X_real, real_t1, t_mean, t_std, n_epochs=30):
    """Train one seed and log per-epoch metrics on real data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)

    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    epoch_log = []
    for epoch in range(n_epochs):
        model.train()
        anneal = min(1.0, epoch / 10)
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            out = model(x)
            nig = out.view(out.shape[0], 2, 4)
            result = evidential_regression_loss(y, nig, coeff=1.0, epoch=epoch, annealing_epochs=10)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        # Evaluate on real data every epoch
        metrics = evaluate_real_data(model, X_real, real_t1, t_mean, t_std)
        metrics["epoch"] = epoch + 1
        epoch_log.append(metrics)

    return model, epoch_log


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    # Load normalization
    hf = h5py.File(mrf_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)
    t_mean = hf["parameters"][:n_train, :2].astype(np.float32).mean(0)
    t_std = hf["parameters"][:n_train, :2].astype(np.float32).std(0) + 1e-8
    hf.close()

    # Load real data once
    print("Loading real qMRLab data...")
    X_real, real_t1 = load_real_data()
    print(f"  {len(X_real)} voxels loaded")

    seeds = [7, 13, 21, 42, 123, 3, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

    all_dynamics = {}
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()
        model, epoch_log = train_seed_with_dynamics(seed, mrf_path, X_real, real_t1, t_mean, t_std)
        elapsed = time.time() - t0

        final_rho = epoch_log[-1]["rho"]
        label = "GOOD" if final_rho > 0.8 else "DEGENERATE" if final_rho < -0.1 else "MODERATE"
        print(f"  Done in {elapsed:.0f}s  final ρ={final_rho:.3f}  [{label}]")

        all_dynamics[f"seed_{seed}"] = {
            "seed": seed,
            "final_rho": final_rho,
            "category": label,
            "epochs": epoch_log,
        }

        # Save checkpoint
        torch.save(model.state_dict(), CKPT_DIR / f"dynamics_seed_{seed}.pt")

    # Save all dynamics
    with open(FIG / "training_dynamics.json", "w") as f:
        json.dump(all_dynamics, f, indent=2)
    print(f"\nSaved to {FIG / 'training_dynamics.json'}")

    # Quick analysis
    print("\n=== DIVERGENCE ANALYSIS ===")
    for seed_key, data in all_dynamics.items():
        rhos = [e["rho"] for e in data["epochs"]]
        # Find epoch where rho first drops below -0.1
        divergence_epoch = None
        for i, r in enumerate(rhos):
            if r < -0.1:
                divergence_epoch = i + 1
                break
        if divergence_epoch:
            print(f"  {seed_key}: diverged at epoch {divergence_epoch}/30 (ρ={rhos[divergence_epoch-1]:.3f})")
        else:
            print(f"  {seed_key}: stable (final ρ={rhos[-1]:.3f})")


if __name__ == "__main__":
    main()
