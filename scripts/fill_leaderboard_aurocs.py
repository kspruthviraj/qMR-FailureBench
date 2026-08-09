#!/usr/bin/env python3
"""Compute failure-detection AUROC for leaderboard baselines on synthetic val.

Uses existing checkpoints when available; otherwise trains lightweight baselines
for a few epochs so AUROC cells are not null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.data.loaders import MRFMetaDataset, load_training_norm
from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.losses import evidential_regression_loss
from qMR_Robust.eval.nig_utils import predict_nig_batches, denorm_gamma, nig_epistemic_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT = ROOT / "results" / "checkpoints"
FAIL_THRESH = 300.0


class HeteroscedasticHead(nn.Module):
    def __init__(self, backbone, out_dim=2):
        super().__init__()
        self.backbone = backbone
        fd = backbone.feature_dim
        self.head = nn.Linear(fd, out_dim * 2)  # mean, log_var

    def forward(self, x):
        f = self.backbone.encode(x)
        raw = self.head(f)
        mean, log_var = raw.chunk(2, dim=-1)
        return mean, log_var


def auroc_from_scores(resid, scores):
    labels = (resid > FAIL_THRESH).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def eval_evidential(ckpt, signals, params, t_mean, t_std):
    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    preds = predict_nig_batches(model, signals, DEVICE)
    gamma = denorm_gamma(preds["gamma"], t_mean, t_std)
    resid = np.abs(params - gamma).max(axis=-1)
    ep = nig_epistemic_np(preds["nu"], preds["alpha"], preds["beta"]).max(axis=-1)
    ep = np.where(np.isfinite(ep), ep, 1e6)
    rho, _ = spearmanr(ep, resid) if ep.std() > 1e-12 else (0.0, 1.0)
    return {
        "mae_ms": float(resid.mean()),
        "auroc": auroc_from_scores(resid, ep),
        "rho": float(rho),
    }


def train_hetero(train_loader, n_epochs=15):
    backbone = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=False).to(DEVICE)
    model = HeteroscedasticHead(backbone).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mean, log_var = model(x)
            var = torch.exp(log_var).clamp(min=1e-6)
            loss = (0.5 * (torch.log(var) + (y - mean) ** 2 / var)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def eval_hetero(model, signals, params, t_mean, t_std):
    model.eval()
    means, vars_ = [], []
    for i in range(0, len(signals), 256):
        x = torch.from_numpy(signals[i : i + 256]).to(DEVICE)
        m, lv = model(x)
        means.append(m.cpu().numpy())
        vars_.append(torch.exp(lv).cpu().numpy())
    mean = np.concatenate(means) * t_std + t_mean
    var = np.concatenate(vars_) * (t_std ** 2)
    resid = np.abs(params - mean).max(axis=-1)
    unc = np.sqrt(var).max(axis=-1)
    rho, _ = spearmanr(unc, resid) if unc.std() > 1e-12 else (0.0, 1.0)
    return {
        "mae_ms": float(resid.mean()),
        "auroc": auroc_from_scores(resid, unc),
        "rho": float(rho),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    mrf = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    t_mean, t_std = load_training_norm(mrf)

    train_ds = MRFMetaDataset(mrf, split="train")
    val_ds = MRFMetaDataset(mrf, split="val")
    val_ds.set_norm(t_mean, t_std)
    n_eval = min(2000, len(val_ds))
    signals = val_ds.signals[:n_eval]
    params = val_ds.params[:n_eval]

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)

    leaderboard = {}

    # Ours: prefer abl_NLL_ER or seed 42
    ours_ckpt = CKPT / "abl_NLL_ER.pt"
    if not ours_ckpt.exists():
        ours_ckpt = CKPT / "seed_study_42.pt"
    if ours_ckpt.exists():
        leaderboard["Ours (NLL+ER)"] = eval_evidential(ours_ckpt, signals, params, t_mean, t_std)
        leaderboard["Ours (NLL+ER)"]["ckpt"] = ours_ckpt.name
        print("Ours", leaderboard["Ours (NLL+ER)"])

    # Deep ensemble from existing members if present
    ens = sorted(CKPT.glob("ensemble_*.pt")) + sorted(CKPT.glob("v3_ensemble_*.pt"))
    if len(ens) >= 2:
        preds = []
        for p in ens[:5]:
            model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
            try:
                model.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
            except Exception:
                continue
            model.eval()
            pr = predict_nig_batches(model, signals, DEVICE)
            preds.append(denorm_gamma(pr["gamma"], t_mean, t_std))
        if preds:
            stack = np.stack(preds, axis=0)
            mean = stack.mean(0)
            std = stack.std(0).max(-1)
            resid = np.abs(params - mean).max(-1)
            leaderboard["Deep Ensemble"] = {
                "mae_ms": float(resid.mean()),
                "auroc": auroc_from_scores(resid, std),
                "rho": float(spearmanr(std, resid)[0]) if std.std() > 1e-12 else 0.0,
                "n_members": len(preds),
            }
            print("Ensemble", leaderboard["Deep Ensemble"])
    else:
        # fallback to saved JSON
        de = json.load(open(FIG / "v3_deep_ensemble.json"))
        leaderboard["Deep Ensemble"] = {
            "mae_ms": de["mae_ms"],
            "auroc": de["auroc"],
            "rho": de.get("correlation"),
            "source": "v3_deep_ensemble.json",
        }

    # Heteroscedastic
    print("Training heteroscedastic baseline (15 epochs)...")
    torch.manual_seed(42)
    hetero = train_hetero(train_loader, n_epochs=15)
    leaderboard["Heteroscedastic"] = eval_hetero(hetero, signals, params, t_mean, t_std)
    print("Hetero", leaderboard["Heteroscedastic"])

    # Deterministic
    print("Training deterministic baseline (10 epochs)...")
    torch.manual_seed(42)
    det = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=False).to(DEVICE)
    opt = torch.optim.AdamW(det.parameters(), lr=1e-3)
    for _ in range(10):
        det.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = det(x)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    det.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, n_eval, 256):
            out = det(torch.from_numpy(signals[i : i + 256]).to(DEVICE)).cpu().numpy()
            preds.append(out)
    pred = np.concatenate(preds) * t_std + t_mean
    resid = np.abs(params - pred).max(-1)
    leaderboard["Deterministic"] = {
        "mae_ms": float(resid.mean()),
        "auroc": None,
        "rho": None,
        "note": "no uncertainty head",
    }
    print("Det", leaderboard["Deterministic"])

    # Merge prior conformal/quantile MAE if present
    old = json.load(open(FIG / "leaderboard.json")) if (FIG / "leaderboard.json").exists() else {}
    for k in ["Quantile", "Conformal (90%)"]:
        if k in old and isinstance(old[k], dict):
            leaderboard[k] = {
                "mae_ms": old[k].get("mae_ms"),
                "auroc": old[k].get("auroc"),
                "note": old[k].get("auroc_note", "from prior run"),
            }

    # Dictionary baseline if available
    if (FIG / "dictionary_baseline.json").exists():
        db = json.load(open(FIG / "dictionary_baseline.json"))
        leaderboard["Dictionary matching"] = {
            "mae_ms": db["mae_ms"],
            "auroc": db["auroc"],
            "rho": db["spearman_rho"],
        }

    leaderboard["_meta"] = {
        "n_eval": n_eval,
        "fail_threshold_ms": FAIL_THRESH,
        "t1_units": "ms",
        "split": "synthetic_val",
    }

    path = FIG / "leaderboard.json"
    path.write_text(json.dumps(leaderboard, indent=2, default=str))
    print("Wrote", path)
    print(json.dumps(leaderboard, indent=2, default=str))


if __name__ == "__main__":
    main()
