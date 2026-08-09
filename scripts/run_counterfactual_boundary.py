#!/usr/bin/env python3
"""
run_counterfactual_boundary.py — Characterize when counterfactual repair works vs fails.

Analyze the n=200 vs n=1000 discrepancy:
  - What corruption severity distinguishes the easy subset (n=200) from the hard set?
  - At what severity threshold does correction start degrading?
  - Which corruption type drives the degradation?
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.severity_regression import DualHeadWithSeverity, counterfactual_correction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"


class MRFMetaDataset:
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


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    # Load checkpoint
    ckpt_candidates = [
        ROOT / "results" / "backup_20260619" / "checkpoints" / "v2_severity.pt",
        ROOT / "results" / "checkpoints" / "v2_severity.pt",
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        print("ERROR: No severity checkpoint")
        sys.exit(1)

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std

    backbone = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1)
    model = DualHeadWithSeverity(backbone, output_dim=2, hidden_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    n = min(1000, len(val_ds))

    # Get normalized targets
    all_targets = []
    for i in range(n):
        _, tgt_norm = val_ds[i]
        all_targets.append(tgt_norm.numpy())
    targets_norm = np.stack(all_targets)
    targets_denorm = targets_norm * t_std + t_mean

    # Model predictions + severity
    batch = torch.from_numpy(val_ds.signals[:n]).to(DEVICE)
    with torch.no_grad():
        out = model(batch)
        gamma = out["nig"][..., 0].cpu().numpy()
        sev_b0 = out["severity"]["delta_f"].cpu().numpy()
        sev_b1 = out["severity"]["lambda_b1"].cpu().numpy() + 1.0
        sev_mot = out["severity"]["delta_motion"].cpu().numpy()

    gamma_denorm = gamma * t_std + t_mean
    resid_before = np.abs(targets_denorm - gamma_denorm)

    # Counterfactual correction
    resid_after = []
    for i in range(n):
        sig_complex = val_ds.signals[i, 0] + 1j * val_ds.signals[i, 1]
        corrected = counterfactual_correction(
            sig_complex.astype(np.complex64),
            float(sev_b0[i]), float(sev_b1[i]), float(sev_mot[i])
        )
        sig_2ch = torch.from_numpy(
            np.stack([corrected.real, corrected.imag], axis=0).astype(np.float32)
        ).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_corr = model(sig_2ch)
            g_corr = out_corr["nig"][..., 0].cpu().numpy()
        g_corr_denorm = g_corr[0] * t_std + t_mean
        resid_after.append(np.abs(targets_denorm[i] - g_corr_denorm))

    resid_after = np.array(resid_after)

    # Per-sample improvement
    improvement = (resid_before.mean(axis=-1) - resid_after.mean(axis=-1))

    # Corruption parameters
    b0 = val_ds.b0[:n]
    b1 = val_ds.b1[:n]
    mot = val_ds.motion[:n]
    b0_severity = np.abs(b0)
    b1_severity = np.abs(b1 - 1.0)
    mot_severity = np.abs(mot)
    total_severity = b0_severity / 80 + b1_severity / 0.4 + mot_severity / 8

    # Sort by improvement and analyze quartiles
    sorted_idx = np.argsort(improvement)

    print("=" * 60)
    print("  COUNTERFACTUAL FAILURE BOUNDARY ANALYSIS")
    print("=" * 60)

    # Quartile analysis
    for q_label, q_idx in [("Best 25%", slice(None, n//4)),
                             ("Worst 25%", slice(3*n//4, None))]:
        idx = sorted_idx[q_idx]
        print(f"\n{q_label} (n={len(idx)}):")
        print(f"  Mean improvement: {improvement[idx].mean():.1f} ms")
        print(f"  Mean |B0|: {b0_severity[idx].mean():.1f} Hz")
        print(f"  Mean |B1-1|: {b1_severity[idx].mean():.3f}")
        print(f"  Mean |motion|: {mot_severity[idx].mean():.1f} voxels")
        print(f"  Mean total severity: {total_severity[idx].mean():.3f}")
        print(f"  Pct with strong B0 (>20Hz): {(b0_severity[idx] > 20).mean()*100:.0f}%")
        print(f"  Pct with strong B1 (>0.1): {(b1_severity[idx] > 0.1).mean()*100:.0f}%")
        print(f"  Pct with strong motion (>3): {(mot_severity[idx] > 3).mean()*100:.0f}%")

    # Severity threshold analysis
    print("\n\nImprovement by severity quartile:")
    for q in range(4):
        lo, hi = np.percentile(total_severity, q * 25), np.percentile(total_severity, (q+1) * 25)
        mask = (total_severity >= lo) & (total_severity < hi)
        if mask.sum() > 10:
            print(f"  Q{q+1} (severity {lo:.3f}-{hi:.3f}): n={mask.sum()}, "
                  f"mean_improvement={improvement[mask].mean():.1f} ms, "
                  f"pct_positive={(improvement[mask] > 0).mean()*100:.0f}%")

    # Save
    output = {
        "n_eval": n,
        "overall_improvement_pct": float((resid_before.mean() - resid_after.mean()) / resid_before.mean() * 100),
        "pct_positive_improvement": float((improvement > 0).mean() * 100),
        "best_quartile_severity": {
            "mean_b0": float(b0_severity[sorted_idx[:n//4]].mean()),
            "mean_b1": float(b1_severity[sorted_idx[:n//4]].mean()),
            "mean_mot": float(mot_severity[sorted_idx[:n//4]].mean()),
        },
        "worst_quartile_severity": {
            "mean_b0": float(b0_severity[sorted_idx[3*n//4:]].mean()),
            "mean_b1": float(b1_severity[sorted_idx[3*n//4:]].mean()),
            "mean_mot": float(mot_severity[sorted_idx[3*n//4:]].mean()),
        },
    }
    with open(FIG / "counterfactual_boundary.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {FIG / 'counterfactual_boundary.json'}")


if __name__ == "__main__":
    main()
