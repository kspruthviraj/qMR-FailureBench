#!/usr/bin/env python3
"""
run_seed_study.py — 5-seed adaptation study to characterize seed sensitivity.

Trains evidential models with 5 different seeds, evaluates:
  1. Zero-shot ρ on real qMRLab data
  2. 5% calibration repair ρ
  3. Synthetic test MAE and AUROC

This determines whether calibration repair is robust or seed-dependent.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.losses import evidential_regression_loss
from qMR_Robust.reproducibility import seed_everything

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
        self.b0 = hf["b0_hz_applied"][s:e].astype(np.float32)
        self.b1 = hf["b1_scale_applied"][s:e].astype(np.float32)
        self.motion = hf["motion_shift_applied"][s:e].astype(np.float32)
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


def train_evidential(mrf_path, seed, n_epochs=30, lr=1e-3):
    """Train evidential model with given seed. Returns checkpoint path."""
    ckpt_path = CKPT_DIR / f"seed_study_{seed}.pt"
    if ckpt_path.exists():
        print(f"  Seed {seed}: checkpoint exists, skipping training")
        return ckpt_path

    seed_everything(seed)

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val, best_state = float("inf"), None
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                out = model(x)
                nig = out.view(out.shape[0], 2, 4)
                result = evidential_regression_loss(y, nig, coeff=1.0, epoch=epoch, annealing_epochs=10)
                val_loss += result["loss"].item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def evaluate_adaptation(mrf_path, ckpt_path, seed):
    """Evaluate zero-shot and 5% calibration repair on synthetic + real data."""
    seed_everything(seed)

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std

    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Synthetic test evaluation
    n_eval = min(2000, len(val_ds))
    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
    all_tgt = []
    with torch.no_grad():
        for i in range(0, n_eval, 256):
            batch_x = torch.from_numpy(val_ds.signals[i:i+256]).to(DEVICE)
            batch_y = torch.from_numpy(val_ds.params[i:i+256]).numpy()
            out = model(batch_x)
            nig = out.view(out.shape[0], 2, 4)
            all_gamma.append(nig[..., 0].cpu().numpy())
            all_nu.append(nig[..., 1].cpu().numpy())
            all_alpha.append(nig[..., 2].cpu().numpy())
            all_beta.append(nig[..., 3].cpu().numpy())
            all_tgt.append(batch_y)

    gamma = np.concatenate(all_gamma)
    nu = np.concatenate(all_nu)
    alpha = np.concatenate(all_alpha)
    beta = np.concatenate(all_beta)
    tgt = np.concatenate(all_tgt)

    gamma_d = gamma * t_std + t_mean
    tgt_d = tgt  # params are already raw ms, no denorm needed
    resid = np.abs(tgt_d - gamma_d).max(axis=-1)
    epistemic = beta / (nu * (alpha - 1.0))
    max_ep = epistemic.max(axis=-1)

    labels = (resid > 300).astype(int)
    synth_auroc = float(roc_auc_score(labels, max_ep)) if 0 < labels.sum() < n_eval else float("nan")
    synth_mae = float(np.mean(resid))
    synth_rho, _ = spearmanr(max_ep, resid) if max_ep.std() > 1e-12 else (0.0, 1.0)

    # Real data evaluation (qMRLab VFA) — unit-safe, T1 in milliseconds
    from qMR_Robust.data.loaders import load_qmrlab_vfa
    qmrlab_dir = ROOT / "data" / "real" / "qmrlab" / "vfa_t1_data"
    if not (qmrlab_dir / "VFAData.nii.gz").exists():
        print(f"  Seed {seed}: No qMRLab data, using synthetic-only metrics")
        return {
            "seed": seed,
            "synth_mae": synth_mae,
            "synth_auroc": synth_auroc,
            "synth_rho": float(synth_rho),
            "real_rho_zero_shot": None,
            "real_rho_5pct": None,
        }

    real = load_qmrlab_vfa(qmrlab_dir, pad_mode="zeropad")
    X_real = real.signals
    real_t1 = real.t1_ms  # milliseconds
    all_ep_real = []
    all_gamma_real = []
    with torch.no_grad():
        for i in range(0, len(X_real), 256):
            batch = torch.from_numpy(X_real[i:i+256]).to(DEVICE)
            out = model(batch)
            nig = out.view(out.shape[0], 2, 4)
            ep = (nig[..., 3] / (nig[..., 1] * (nig[..., 2] - 1.0))).max(dim=-1).values
            g = nig[..., 0, 0] * t_std[0] + t_mean[0]
            all_ep_real.append(ep.cpu().numpy())
            all_gamma_real.append(g.cpu().numpy())

    ep_real = np.concatenate(all_ep_real)
    gamma_real = np.concatenate(all_gamma_real)
    error_real = np.abs(real_t1 - gamma_real)

    rho_zero, _ = spearmanr(ep_real, error_real) if ep_real.std() > 1e-12 else (0.0, 1.0)

    # 5% calibration repair
    n_cal = int(0.05 * len(ep_real))
    idx = np.random.permutation(len(ep_real))
    idx_cal, idx_test = idx[:n_cal], idx[n_cal:]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(ep_real[idx_cal], error_real[idx_cal])
    calibrated = iso.transform(ep_real[idx_test])
    rho_cal, _ = spearmanr(calibrated, error_real[idx_test])

    return {
        "seed": seed,
        "synth_mae": synth_mae,
        "synth_auroc": synth_auroc,
        "synth_rho": float(synth_rho),
        "real_rho_zero_shot": float(rho_zero),
        "real_rho_5pct": float(rho_cal),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    seeds = [7, 13, 21, 42, 123, 3, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
             483, 486, 286, 130, 421, 426, 383, 358, 547, 542, 770, 119, 370, 316, 655, 106,
             92, 760, 375, 209, 883, 714, 268, 78, 651, 82, 723, 167, 264, 44]
    results = []

    print("=" * 60)
    print("  5-SEED ADAPTATION STUDY")
    print("=" * 60)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()

        # Train
        ckpt = train_evidential(mrf_path, seed)
        print(f"  Training: {time.time()-t0:.0f}s")

        # Evaluate
        result = evaluate_adaptation(mrf_path, ckpt, seed)
        results.append(result)
        print(f"  synth_mae={result['synth_mae']:.1f}  synth_rho={result['synth_rho']:.3f}  "
              f"real_rho_0%={result['real_rho_zero_shot']}  real_rho_5%={result['real_rho_5pct']}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    # Check for qmrlab data availability
    has_real = all(r["real_rho_zero_shot"] is not None for r in results)

    if has_real:
        rhos_zero = [r["real_rho_zero_shot"] for r in results]
        rhos_cal = [r["real_rho_5pct"] for r in results]

        print(f"\n  Zero-shot ρ: {[f'{r:.3f}' for r in rhos_zero]}")
        print(f"  Mean±SD: {np.mean(rhos_zero):.3f} ± {np.std(rhos_zero):.3f}")
        print(f"\n  5% repair ρ: {[f'{r:.3f}' for r in rhos_cal]}")
        print(f"  Mean±SD: {np.mean(rhos_cal):.3f} ± {np.std(rhos_cal):.3f}")

        n_degenerate = sum(1 for r in rhos_zero if r < -0.5)
        print(f"\n  Degenerate seeds (ρ < -0.5): {n_degenerate}/{len(seeds)}")
    else:
        print("\n  No qMRLab real data available — synthetic-only metrics")
        synth_rhos = [r["synth_rho"] for r in results]
        print(f"  Synthetic ρ: {[f'{r:.3f}' for r in synth_rhos]}")
        print(f"  Mean±SD: {np.mean(synth_rhos):.3f} ± {np.std(synth_rhos):.3f}")

    # Save
    output = {
        "seeds": seeds,
        "results": results,
        "summary": {
            "n_degenerate": sum(1 for r in results if r.get("real_rho_zero_shot", 0) is not None and r["real_rho_zero_shot"] < -0.5),
            "mean_zero_shot_rho": float(np.mean([r["real_rho_zero_shot"] for r in results if r["real_rho_zero_shot"] is not None])) if has_real else None,
            "std_zero_shot_rho": float(np.std([r["real_rho_zero_shot"] for r in results if r["real_rho_zero_shot"] is not None])) if has_real else None,
            "mean_5pct_rho": float(np.mean([r["real_rho_5pct"] for r in results if r["real_rho_5pct"] is not None])) if has_real else None,
            "std_5pct_rho": float(np.std([r["real_rho_5pct"] for r in results if r["real_rho_5pct"] is not None])) if has_real else None,
        }
    }
    with open(FIG / "seed_study.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {FIG / 'seed_study.json'}")


if __name__ == "__main__":
    main()
