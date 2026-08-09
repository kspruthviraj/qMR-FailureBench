#!/usr/bin/env python3
"""
run_architecture_comparison.py — Compare ResNet-1D, ViT-1D, and SpatioTemporal Transformer.

Tests whether the calibration collapse is architecture-specific or a general phenomenon.
Each architecture is trained with the same setup (seed 42, NLL+ER, 30 epochs) and evaluated
on synthetic test data and real qMRLab data.
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.vit1d import ViT1D
from qMR_Robust.models.spatiotemporal_transformer import SpatioTemporalTransformer
from qMR_Robust.models.losses import evidential_regression_loss

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



def load_real_data():
    """Unit-safe qMRLab VFA loader (T1 in ms, zero-pad cross-sequence protocol)."""
    from qMR_Robust.data.loaders import load_qmrlab_vfa
    from pathlib import Path as _P
    qmrlab_dir = _P(__file__).resolve().parent.parent / "data" / "real" / "qmrlab" / "vfa_t1_data"
    data = load_qmrlab_vfa(qmrlab_dir, pad_mode="zeropad")
    return data.signals, data.t1_ms



def build_model(arch_name, evidential=True):
    """Build model by architecture name."""
    if arch_name == "resnet1d":
        return ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=evidential)
    elif arch_name == "vit1d":
        return ViT1D(in_channels=2, hidden_dim=128, output_dim=2, patch_size=50, n_heads=4, n_layers=4, dropout=0.1, evidential=evidential)
    elif arch_name == "spatiotemporal":
        return SpatioTemporalTransformer(in_channels=2, seq_len=1000, hidden_dim=128, output_dim=2, n_heads=4, n_temporal_layers_1=3, n_temporal_layers_2=2, dropout=0.1, evidential=evidential)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


def train_model(arch_name, mrf_path, seed=42, n_epochs=30, er_coeff=1.0):
    """Train a model with given architecture."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = CKPT_DIR / f"arch_{arch_name}_seed{seed}.pt"
    if ckpt_path.exists():
        print(f"  Checkpoint exists, loading")
        model = build_model(arch_name, evidential=True).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        return model

    train_ds = MRFMetaDataset(mrf_path, split="train")
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)

    model = build_model(arch_name, evidential=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            out = model(x)
            nig = out.view(out.shape[0], 2, 4)
            result = evidential_regression_loss(y, nig, coeff=er_coeff, epoch=epoch, annealing_epochs=10)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), ckpt_path)
    return model


def evaluate_synthetic(model, val_ds, t_mean, t_std):
    """Evaluate on synthetic validation data."""
    model.eval()
    X = val_ds.signals
    n_eval = min(2000, len(X))

    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
    with torch.no_grad():
        for i in range(0, n_eval, 256):
            batch = torch.from_numpy(X[i:i+256]).to(DEVICE)
            out = model(batch).view(-1, 2, 4)
            all_gamma.append(out[..., 0].cpu().numpy())
            all_nu.append(out[..., 1].cpu().numpy())
            all_alpha.append(out[..., 2].cpu().numpy())
            all_beta.append(out[..., 3].cpu().numpy())

    gamma = np.concatenate(all_gamma)[:n_eval] * t_std + t_mean
    nu = np.concatenate(all_nu)[:n_eval]
    alpha = np.concatenate(all_alpha)[:n_eval]
    beta = np.concatenate(all_beta)[:n_eval]
    tgt = val_ds.params[:n_eval]

    resid = np.abs(tgt - gamma).max(axis=-1)
    epistemic = beta / (nu * (alpha - 1.0))
    max_ep = epistemic.max(axis=-1)

    labels = (resid > 300).astype(int)
    auroc = float(roc_auc_score(labels, max_ep)) if 0 < labels.sum() < n_eval else float("nan")
    rho, _ = spearmanr(max_ep, resid) if max_ep.std() > 1e-12 else (0.0, 1.0)

    return {
        "mae_ms": float(np.mean(resid)),
        "auroc": auroc,
        "rho": float(rho),
    }


def evaluate_real(model, X_real, real_t1, t_mean, t_std):
    """Evaluate on real qMRLab data."""
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
    return {"rho": float(rho), "mae_ms": float(np.mean(error))}


def evaluate_per_corruption(model, val_ds, t_mean, t_std):
    """Evaluate AUROC per corruption type."""
    model.eval()
    X = val_ds.signals
    n = len(X)

    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
    with torch.no_grad():
        for i in range(0, n, 256):
            batch = torch.from_numpy(X[i:i+256]).to(DEVICE)
            out = model(batch).view(-1, 2, 4)
            all_gamma.append(out[..., 0].cpu().numpy())
            all_nu.append(out[..., 1].cpu().numpy())
            all_alpha.append(out[..., 2].cpu().numpy())
            all_beta.append(out[..., 3].cpu().numpy())

    gamma = np.concatenate(all_gamma) * t_std + t_mean
    nu = np.concatenate(all_nu)
    alpha = np.concatenate(all_alpha)
    beta = np.concatenate(all_beta)
    tgt = val_ds.params

    resid = np.abs(tgt - gamma).max(axis=-1)
    epistemic = beta / (nu * (alpha - 1.0))
    max_ep = epistemic.max(axis=-1)

    masks = {
        "B0 only": (np.abs(val_ds.b0) > 1.0) & (np.abs(val_ds.b1 - 1.0) < 0.01) & (np.abs(val_ds.motion) < 1),
        "B1 only": (np.abs(val_ds.b0) < 1.0) & (np.abs(val_ds.b1 - 1.0) > 0.01) & (np.abs(val_ds.motion) < 1),
        "Motion only": (np.abs(val_ds.b0) < 1.0) & (np.abs(val_ds.b1 - 1.0) < 0.01) & (np.abs(val_ds.motion) > 1),
        "Entangled": (np.abs(val_ds.b0) > 1.0) & (np.abs(val_ds.b1 - 1.0) > 0.01) & (np.abs(val_ds.motion) > 1),
    }

    results = {}
    for name, mask in masks.items():
        n_active = mask.sum()
        if n_active < 10:
            results[name] = float("nan")
            continue
        res_sub = resid[mask]
        ep_sub = max_ep[mask]
        labels = (res_sub > 300).astype(int)
        auroc = float(roc_auc_score(labels, ep_sub)) if 0 < labels.sum() < n_active else float("nan")
        results[name] = auroc

    return results


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    hf = h5py.File(mrf_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)
    t_mean = hf["parameters"][:n_train, :2].astype(np.float32).mean(0)
    t_std = hf["parameters"][:n_train, :2].astype(np.float32).std(0) + 1e-8
    hf.close()

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)

    print("Loading real data...")
    X_real, real_t1 = load_real_data()

    architectures = ["resnet1d", "vit1d", "spatiotemporal"]
    arch_labels = ["ResNet-1D", "ViT-1D", "SpatioTemporal"]

    all_results = {}

    for arch, label in zip(architectures, arch_labels):
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        t0 = time.time()
        model = train_model(arch, mrf_path, seed=42, n_epochs=30, er_coeff=1.0)
        train_time = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())

        print(f"  Training: {train_time:.0f}s, Parameters: {n_params:,}")

        # Synthetic evaluation
        synth = evaluate_synthetic(model, val_ds, t_mean, t_std)
        print(f"  Synthetic: MAE={synth['mae_ms']:.1f} AUROC={synth['auroc']:.3f} ρ={synth['rho']:.3f}")

        # Real evaluation
        real = evaluate_real(model, X_real, real_t1, t_mean, t_std)
        print(f"  Real: MAE={real['mae_ms']:.1f} ρ={real['rho']:.3f}")

        # Per-corruption
        per_corr = evaluate_per_corruption(model, val_ds, t_mean, t_std)
        print(f"  Per-corruption AUROC:")
        for name, auroc in per_corr.items():
            print(f"    {name}: {auroc:.3f}" if not np.isnan(auroc) else f"    {name}: NaN")

        all_results[arch] = {
            "label": label,
            "n_params": n_params,
            "train_time_s": train_time,
            "synthetic": synth,
            "real": real,
            "per_corruption": per_corr,
        }

    # Save
    with open(FIG / "architecture_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {FIG / 'architecture_comparison.json'}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  ARCHITECTURE COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Architecture':<20} {'Params':>10} {'Synth MAE':>10} {'Synth AUROC':>12} {'Real ρ':>8} {'Ent. AUROC':>10}")
    for arch in architectures:
        r = all_results[arch]
        ent = r["per_corruption"].get("Entangled", float("nan"))
        ent_str = f"{ent:.3f}" if not np.isnan(ent) else "NaN"
        print(f"{r['label']:<20} {r['n_params']:>10,} {r['synthetic']['mae_ms']:>10.1f} {r['synthetic']['auroc']:>12.3f} {r['real']['rho']:>8.3f} {ent_str:>10}")


if __name__ == "__main__":
    main()
