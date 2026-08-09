#!/usr/bin/env python3
"""
validate_counterfactual.py — Reproduce the original 39.6% counterfactual result.

Matches the deleted run_v2_upgrades.py logic exactly:
  - Uses dataset's normalized targets (NOT raw HDF5 params)
  - n=200 first (original), then n=1000 (extended)
  - Reports per-corruption and overall improvement
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.severity_regression import DualHeadWithSeverity, counterfactual_correction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MRFMetaDataset:
    """Minimal dataset matching run_v3_final.py's MRFMetaDataset."""

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
            torch.tensor(self.b0[i]),
            torch.tensor(self.b1[i]),
            torch.tensor(self.motion[i]),
        )


def run_counterfactual(mrf_path, ckpt_path, n_eval, label):
    """Run counterfactual correction on n_eval samples.

    Matches the original run_v2_upgrades.py logic:
      - targets come from the dataset (normalized), then denormalized
      - NOT from raw HDF5 params
    """
    print(f"\n{'='*60}")
    print(f"  Counterfactual validation: n={n_eval} ({label})")
    print(f"{'='*60}")

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std

    backbone = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1)
    model = DualHeadWithSeverity(backbone, output_dim=2, hidden_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    n = min(n_eval, len(val_ds))

    # Get normalized targets from dataset (matching original code)
    all_targets = []
    for i in range(n):
        _, tgt_norm, _, _, _ = val_ds[i]
        all_targets.append(tgt_norm.numpy())
    targets_norm = np.stack(all_targets)  # (n, 2) normalized

    # Denormalize targets to raw ms (CORRECT: targets are normalized)
    targets_denorm = targets_norm * t_std + t_mean

    # Get model predictions + severity estimates
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

    mae_before = float(resid_before.mean())
    mae_after = float(resid_after.mean())
    improvement_pct = (mae_before - mae_after) / mae_before * 100

    # Per-corruption breakdown
    b0_true = val_ds.b0[:n]
    b1_true = val_ds.b1[:n]
    mot_true = val_ds.motion[:n]

    b0_dom = np.abs(b0_true) > 20
    b1_dom = np.abs(b1_true - 1.0) > 0.1
    mot_dom = np.abs(mot_true) > 3

    per_corruption = {}
    for name, mask in [("B0-dominant", b0_dom), ("B1-dominant", b1_dom), ("Motion-dominant", mot_dom)]:
        if mask.sum() > 5:
            before = float(resid_before[mask].mean())
            after = float(resid_after[mask].mean())
            pct = (before - after) / before * 100
            improved = float((resid_after[mask].max(axis=-1) < resid_before[mask].max(axis=-1)).mean() * 100)
            per_corruption[name] = {
                "n": int(mask.sum()),
                "mae_before": before,
                "mae_after": after,
                "improvement_pct": pct,
                "pct_improved": improved,
            }

    result = {
        "n_eval": n,
        "mae_before_ms": mae_before,
        "mae_after_ms": mae_after,
        "improvement_pct": improvement_pct,
        "per_corruption": per_corruption,
    }

    print(f"  MAE before: {mae_before:.1f} ms")
    print(f"  MAE after:  {mae_after:.1f} ms")
    print(f"  Overall improvement: {improvement_pct:.1f}%")
    for k, v in per_corruption.items():
        print(f"  {k}: n={v['n']}, before={v['mae_before']:.1f}, after={v['mae_after']:.1f}, "
              f"improvement={v['improvement_pct']:.1f}%, pct_improved={v['pct_improved']:.1f}%")

    return result


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    # Try backup checkpoint first, then re-run checkpoint
    ckpt_candidates = [
        ROOT / "results" / "backup_20260619" / "checkpoints" / "v2_severity.pt",
        ROOT / "results" / "checkpoints" / "v2_severity.pt",
        ROOT / "results" / "checkpoints" / "leaderboard_severity.pt",
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        print("ERROR: No severity checkpoint found. Run run_novel_experiments.py first.")
        sys.exit(1)
    print(f"Using checkpoint: {ckpt_path}")

    # Run n=200 first (matching original)
    r200 = run_counterfactual(mrf_path, ckpt_path, n_eval=200, label="original n=200")

    # Run n=1000 (extended)
    r1000 = run_counterfactual(mrf_path, ckpt_path, n_eval=1000, label="extended n=1000")

    # Save results
    output = {"n200": r200, "n1000": r1000}
    out_path = ROOT / "results" / "figures" / "counterfactual_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  n=200:  {r200['improvement_pct']:.1f}% overall improvement")
    print(f"  n=1000: {r1000['improvement_pct']:.1f}% overall improvement")
    print(f"  Original claim: 39.6%")
    print(f"  Match? {'YES' if abs(r200['improvement_pct'] - 39.6) < 5 else 'CLOSE' if abs(r200['improvement_pct'] - 39.6) < 15 else 'NO'}")


if __name__ == "__main__":
    main()
