#!/usr/bin/env python3
"""
run_mc_dropout_baseline.py — Train and evaluate MC Dropout under entangled corruptions.

Answers: Does the same calibration collapse occur with a different UQ method?
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.baselines import build_baseline_model

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


def train_mc_dropout(mrf_path, seed=42, n_epochs=30):
    """Train MC Dropout model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = CKPT_DIR / "mc_dropout_baseline.pt"
    if ckpt_path.exists():
        print("  Checkpoint exists, loading")
        backbone_fn = lambda: ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=False)
        model = build_baseline_model("resnet_mc", backbone_fn, "mc_dropout", output_dim=2).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        return model

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)

    backbone_fn = lambda: ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=False)
    model = build_baseline_model("resnet_mc", backbone_fn, "mc_dropout", output_dim=2).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), ckpt_path)
    return model


def evaluate_mc_dropout(model, val_ds, t_mean, t_std, n_forward=20):
    """Evaluate MC Dropout with multiple forward passes."""
    model.train()  # Keep dropout active
    X = val_ds.signals

    all_preds = []
    with torch.no_grad():
        for _ in range(n_forward):
            preds = []
            for i in range(0, len(X), 512):
                batch = torch.from_numpy(X[i:i+512]).to(DEVICE)
                preds.append(model(batch).cpu().numpy())
            all_preds.append(np.concatenate(preds))

    preds = np.stack(all_preds)  # (n_forward, N, D)
    mean_pred = preds.mean(axis=0)
    variance = preds.var(axis=0)
    max_var = variance.max(axis=-1)

    # Denormalize
    mean_denorm = mean_pred * t_std + t_mean
    tgt_denorm = val_ds.params  # already raw ms
    resid = np.abs(tgt_denorm - mean_denorm)
    max_resid = resid.max(axis=-1)

    # AUROC
    labels = (max_resid > 300).astype(int)
    auroc = float(roc_auc_score(labels, max_var)) if 0 < labels.sum() < len(labels) else float("nan")
    rho, _ = spearmanr(max_var, max_resid) if max_var.std() > 1e-12 else (0.0, 1.0)

    return {
        "mae_ms": float(np.mean(resid)),
        "auroc": auroc,
        "rho": float(rho),
        "mean_variance": float(np.mean(variance)),
    }


def evaluate_per_corruption(model, val_ds, t_mean, t_std, n_forward=20):
    """Evaluate MC Dropout per corruption type."""
    model.train()
    X = val_ds.signals

    all_preds = []
    with torch.no_grad():
        for _ in range(n_forward):
            preds = []
            for i in range(0, len(X), 512):
                batch = torch.from_numpy(X[i:i+512]).to(DEVICE)
                preds.append(model(batch).cpu().numpy())
            all_preds.append(np.concatenate(preds))

    preds = np.stack(all_preds)
    mean_pred = preds.mean(axis=0)
    variance = preds.var(axis=0)
    max_var = variance.max(axis=-1)

    mean_denorm = mean_pred * t_std + t_mean
    tgt_denorm = val_ds.params
    resid = np.abs(tgt_denorm - mean_denorm).max(axis=-1)

    corruption_types = {
        "B0 only": (np.abs(val_ds.b0) > 1.0) & (np.abs(val_ds.b1 - 1.0) < 0.01) & (np.abs(val_ds.motion) < 1),
        "B1 only": (np.abs(val_ds.b0) < 1.0) & (np.abs(val_ds.b1 - 1.0) > 0.01) & (np.abs(val_ds.motion) < 1),
        "Motion only": (np.abs(val_ds.b0) < 1.0) & (np.abs(val_ds.b1 - 1.0) < 0.01) & (np.abs(val_ds.motion) > 1),
        "Entangled": (np.abs(val_ds.b0) > 1.0) & (np.abs(val_ds.b1 - 1.0) > 0.01) & (np.abs(val_ds.motion) > 1),
    }

    results = {}
    for name, mask in corruption_types.items():
        n_active = mask.sum()
        if n_active < 10:
            results[name] = {"n": int(n_active), "auroc": float("nan")}
            continue
        res_sub = resid[mask]
        var_sub = max_var[mask]
        labels = (res_sub > 300).astype(int)
        auroc = float(roc_auc_score(labels, var_sub)) if 0 < labels.sum() < n_active else float("nan")
        results[name] = {"n": int(n_active), "mae": float(np.mean(res_sub)), "auroc": auroc}

    return results


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std

    print("Training MC Dropout baseline...")
    model = train_mc_dropout(mrf_path, seed=42)

    print("Evaluating overall...")
    overall = evaluate_mc_dropout(model, val_ds, t_mean, t_std)
    print(f"  MAE={overall['mae_ms']:.1f}  AUROC={overall['auroc']:.3f}  ρ={overall['rho']:.3f}")

    print("Evaluating per-corruption...")
    per_corr = evaluate_per_corruption(model, val_ds, t_mean, t_std)
    for name, metrics in per_corr.items():
        print(f"  {name}: n={metrics['n']}  MAE={metrics.get('mae',0):.1f}  AUROC={metrics.get('auroc',float('nan')):.3f}")

    # Save
    output = {"overall": overall, "per_corruption": per_corr}
    with open(FIG / "mc_dropout_baseline.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {FIG / 'mc_dropout_baseline.json'}")


if __name__ == "__main__":
    main()
