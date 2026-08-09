#!/usr/bin/env python3
"""
run_alpha_safeguard.py — Test whether preventing α→1 collapse reduces degeneracy.

Compares:
  1. Standard NIG training (baseline)
  2. NIG with α-clipping (α > 1 + ε)
  3. NIG with stronger ER (higher λ_ER)

Across 10 seeds each to test if α-collapse prevention reduces failure rate.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.losses import evidential_regression_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT_DIR = ROOT / "results" / "checkpoints"


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



def evaluate_real(model, X_real, real_t1, t_mean, t_std):
    model.eval()
    all_ep = []
    all_gamma = []
    with torch.no_grad():
        for i in range(0, len(X_real), 256):
            batch = torch.from_numpy(X_real[i:i+256]).to(DEVICE)
            out = model(batch).view(-1, 2, 4)
            ep = (out[..., 3] / (out[..., 1] * (out[..., 2] - 1.0))).max(dim=-1).values
            all_ep.append(ep.cpu().numpy())
            all_gamma.append(out[..., 0].cpu().numpy())
    ep = np.concatenate(all_ep)
    gamma = np.concatenate(all_gamma) * t_std + t_mean
    error = np.abs(real_t1 - gamma[:, 0])
    # Replace inf with large finite for correlation
    ep_clean = np.where(np.isfinite(ep), ep, 1e6)
    rho, _ = spearmanr(ep_clean, error) if np.std(ep_clean) > 1e-12 else (0.0, 1.0)
    return float(rho), float(np.mean(np.isfinite(ep)))


def train_and_evaluate(seed, mrf_path, X_real, real_t1, t_mean, t_std, config):
    """Train one seed with given config and return final ρ."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)

    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["n_epochs"])

    for epoch in range(config["n_epochs"]):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            out = model(x)
            nig = out.view(out.shape[0], 2, 4)

            result = evidential_regression_loss(y, nig, coeff=config["er_coeff"], epoch=epoch, annealing_epochs=10)
            loss = result["loss"]

            # Add α-penalty: penalize α approaching 1.0
            if config.get("alpha_penalty", 0) > 0:
                alpha = nig[..., 2]
                alpha_penalty = config["alpha_penalty"] * torch.mean(torch.exp(-alpha + 1.0))
                loss = loss + alpha_penalty
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    rho, finite_frac = evaluate_real(model, X_real, real_t1, t_mean, t_std)
    return rho, finite_frac


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    hf = h5py.File(mrf_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)
    t_mean = hf["parameters"][:n_train, :2].astype(np.float32).mean(0)
    t_std = hf["parameters"][:n_train, :2].astype(np.float32).std(0) + 1e-8
    hf.close()

    print("Loading real data...")
    X_real, real_t1 = load_real_data()

    seeds = [7, 13, 21, 42, 123, 3, 11, 17, 23, 29]

    configs = {
        "baseline": {"er_coeff": 1.0, "alpha_penalty": 0, "n_epochs": 30},
        "alpha_penalty_0.1": {"er_coeff": 1.0, "alpha_penalty": 0.1, "n_epochs": 30},
        "alpha_penalty_1.0": {"er_coeff": 1.0, "alpha_penalty": 1.0, "n_epochs": 30},
        "strong_er": {"er_coeff": 5.0, "alpha_penalty": 0, "n_epochs": 30},
    }

    results = {}
    for config_name, config in configs.items():
        print(f"\n=== {config_name} ===")
        rhos = []
        for seed in seeds:
            rho, finite_frac = train_and_evaluate(seed, mrf_path, X_real, real_t1, t_mean, t_std, config)
            rhos.append(rho)
            label = "GOOD" if rho > 0.8 else "DEG" if rho < -0.1 else "MOD"
            print(f"  Seed {seed:3d}: ρ={rho:+.3f} [{label}] finite_ep={finite_frac:.2%}")

        good = sum(1 for r in rhos if r > 0.8)
        degenerate = sum(1 for r in rhos if r < -0.1)
        results[config_name] = {
            "rhos": rhos,
            "good": good,
            "degenerate": degenerate,
            "n": len(seeds),
            "failure_rate": degenerate / len(seeds),
        }
        print(f"  Summary: {good}/{len(seeds)} good, {degenerate}/{len(seeds)} degenerate")

    # Save
    with open(FIG / "alpha_safeguard.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {FIG / 'alpha_safeguard.json'}")

    # Comparison table
    print("\n=== COMPARISON ===")
    print(f"{'Config':<20} {'Good':>5} {'Deg':>5} {'Fail%':>6}")
    for name, r in results.items():
        print(f"{name:<20} {r['good']:>5} {r['degenerate']:>5} {r['failure_rate']*100:>5.0f}%")


if __name__ == "__main__":
    main()
