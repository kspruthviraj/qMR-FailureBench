#!/usr/bin/env python3
"""
run_strong_er_validation.py — Run Strong ER (5.0) across 50 seeds to validate mitigation.

Uses the same 50 seeds as the main seed study for direct comparison.
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
    all_ep, all_gamma = [], []
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
    ep_clean = np.where(np.isfinite(ep), ep, 1e6)
    rho, _ = spearmanr(ep_clean, error) if np.std(ep_clean) > 1e-12 else (0.0, 1.0)
    rho_cal = rho  # no repair for this experiment
    return float(rho), float(rho_cal)


def train_strong_er(seed, mrf_path, X_real, real_t1, t_mean, t_std, n_epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = CKPT_DIR / f"strong_er_seed_{seed}.pt"
    if ckpt_path.exists():
        model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        rho, rho_cal = evaluate_real(model, X_real, real_t1, t_mean, t_std)
        return rho, rho_cal

    train_ds = MRFMetaDataset(mrf_path, split="train")
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)
    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            out = model(x)
            nig = out.view(out.shape[0], 2, 4)
            result = evidential_regression_loss(y, nig, coeff=5.0, epoch=epoch, annealing_epochs=10)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), ckpt_path)
    rho, rho_cal = evaluate_real(model, X_real, real_t1, t_mean, t_std)
    return rho, rho_cal


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

    seeds = [7, 13, 21, 42, 123, 3, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
             483, 486, 286, 130, 421, 426, 383, 358, 547, 542, 770, 119, 370, 316, 655, 106,
             92, 760, 375, 209, 883, 714, 268, 78, 651, 82, 723, 167, 264, 44]

    results = []
    for seed in seeds:
        rho, rho_cal = train_strong_er(seed, mrf_path, X_real, real_t1, t_mean, t_std)
        results.append({"seed": seed, "real_rho_zero_shot": rho, "real_rho_5pct": rho_cal})
        label = "GOOD" if rho > 0.8 else "DEG" if rho < -0.1 else "MOD"
        print(f"  Seed {seed:3d}: ρ={rho:+.3f} [{label}]")

    # Summary
    rhos = [r["real_rho_zero_shot"] for r in results]
    good = sum(1 for r in rhos if r > 0.8)
    degenerate = sum(1 for r in rhos if r < -0.1)
    n = len(rhos)

    from scipy.stats import norm
    p_hat = degenerate / n
    z = 1.96
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denom
    lo = max(0, center-margin)
    hi = min(1, center+margin)

    print(f"\n=== STRONG ER (5.0) — 50 SEEDS ===")
    print(f"Good: {good}/{n} ({good/n*100:.0f}%)")
    print(f"Degenerate: {degenerate}/{n} ({degenerate/n*100:.0f}%)")
    print(f"Failure rate: {p_hat*100:.0f}% (95% Wilson CI: [{lo*100:.1f}%, {hi*100:.1f}%])")

    # Save
    with open(FIG / "strong_er_validation.json", "w") as f:
        json.dump({"seeds": seeds, "results": results,
                   "summary": {"n": n, "good": good, "degenerate": degenerate,
                               "failure_rate": float(p_hat),
                               "ci_lo": float(lo), "ci_hi": float(hi)}}, f, indent=2)
    print(f"\nSaved to {FIG / 'strong_er_validation.json'}")


if __name__ == "__main__":
    main()
