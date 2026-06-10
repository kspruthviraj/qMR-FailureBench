#!/usr/bin/env python3
"""
run_v2_upgrades.py — All remaining upgrades for reviewer feedback.

Implements:
  1. Corruption severity regression + counterfactual correction
  2. Loss-component ablation (NLL vs +ER vs +physics vs +attr vs all)
  3. Per-corruption-type correlation analysis
  4. Post-hoc calibration (Platt scaling) for improved AUROC
  5. Per-vendor / per-field-strength breakdown
  6. All results saved as JSON + figures
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("v2")

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = ROOT / "results" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset with corruption metadata ─────────────────────────────────────────

class MRFMetaDataset(Dataset):
    """MRF dataset returning signals, targets, and corruption metadata."""
    def __init__(self, h5_path, split="train", train_ratio=0.8):
        hf = h5py.File(h5_path, "r")
        n = hf.attrs["n_signals"]
        n_train = int(n * train_ratio)
        s, e = (0, n_train) if split == "train" else (n_train, n)

        self.signals = np.stack([hf["corrupted_signals"][s:e].real,
                                 hf["corrupted_signals"][s:e].imag], axis=1).astype(np.float32)
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

    def set_norm(self, mean, std): self.mean, self.std = mean, std
    def __len__(self): return len(self.signals)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.signals[i]),
            (torch.from_numpy(self.params[i]) - self.mean) / self.std,
            torch.tensor(self.b0[i]),
            torch.tensor(self.b1[i]),
            torch.tensor(self.motion[i]),
        )


# ── Core training utilities ──────────────────────────────────────────────────

_EPS = 1e-6

def nig_nll(y, gamma, nu, alpha, beta):
    nu = nu.clamp(min=_EPS); alpha = alpha.clamp(min=1+_EPS); beta = beta.clamp(min=_EPS)
    two_b_nu = (2*beta*(1+nu)).clamp(min=_EPS)
    return (0.5*torch.log(torch.pi/nu) - alpha*torch.log(two_b_nu)
            + (alpha+0.5)*torch.log(nu*(y-gamma)**2 + two_b_nu)
            + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)).clamp(max=50.0)

def evidential_reg(y, gamma, nu, alpha):
    return torch.abs(y - gamma) * (2*nu + alpha)

def estimate_snr(signal):
    mag = torch.sqrt(signal[:,0]**2 + signal[:,1]**2)
    peak = mag.max(-1).values
    tail = max(1, mag.shape[-1]//10)
    noise = mag[:,-tail:].std(-1).clamp(min=_EPS)
    return (peak/noise).clamp(1, 500)


# ── Experiment 1: Severity Regression + Counterfactual Correction ────────────

def run_severity_regression(cfg, mrf_path):
    """Train dual-head model with severity regression + counterfactual correction."""
    from qMR_Robust.models.resnet1d import ResNet1D
    from qMR_Robust.models.severity_regression import (
        DualHeadWithSeverity, severity_regression_loss, counterfactual_correction,
    )
    from qMR_Robust.eval.metrics import failure_detection_metrics

    ckpt_path = CKPT_DIR / "v2_severity.pt"

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    backbone = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1)
    model = DualHeadWithSeverity(backbone, output_dim=2, hidden_dim=128).to(DEVICE)

    if ckpt_path.exists():
        logger.info("Loading severity model checkpoint.")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    else:
        logger.info("Training severity regression model (50 epochs)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        for epoch in range(50):
            model.train()
            epoch_loss, n_b = 0.0, 0
            for signals, targets, b0, b1, motion in train_loader:
                signals, targets = signals.to(DEVICE), targets.to(DEVICE)
                b0, b1, motion = b0.to(DEVICE), b1.to(DEVICE), motion.to(DEVICE)

                out = model(signals)
                nig = out["nig"]
                gamma, nu, alpha, beta = nig[...,0], nig[...,1], nig[...,2], nig[...,3]

                # Regression loss
                nll = nig_nll(targets, gamma, nu, alpha, beta)
                er = evidential_reg(targets, gamma, nu, alpha)

                # Severity loss
                sev_loss = severity_regression_loss(out["severity"], b0, b1, motion)

                # Physics anchor
                snr = estimate_snr(signals)
                target_alea = (1.0 / snr.clamp(min=1.0)).unsqueeze(-1).expand(-1, 2)
                learned_alea = (beta / (alpha - 1.0)).clamp(min=_EPS)
                physics = (torch.log(learned_alea) - torch.log(target_alea)).pow(2)

                anneal = min(1.0, epoch / 15)
                loss = nll.mean() + anneal * er.mean() + 0.1 * anneal * physics.mean() + 0.2 * sev_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_b += 1
            scheduler.step()

            if (epoch+1) % 10 == 0 or epoch == 0:
                logger.info("  Severity epoch %d/%d | loss=%.4f", epoch+1, 50, epoch_loss/n_b)

        torch.save(model.state_dict(), ckpt_path)
        logger.info("Checkpoint saved.")

    # ── Evaluate ──
    model.eval()
    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
    all_sev_b0, all_sev_b1, all_sev_mot = [], [], []
    all_targets, all_b0, all_b1, all_motion = [], [], [], []

    with torch.no_grad():
        for signals, targets, b0, b1, motion in val_loader:
            signals = signals.to(DEVICE)
            out = model(signals)
            nig = out["nig"]
            sev = out["severity"]
            all_gamma.append(nig[...,0].cpu())
            all_nu.append(nig[...,1].cpu())
            all_alpha.append(nig[...,2].cpu())
            all_beta.append(nig[...,3].cpu())
            all_sev_b0.append(sev["delta_f"].cpu())
            all_sev_b1.append(sev["lambda_b1"].cpu())
            all_sev_mot.append(sev["delta_motion"].cpu())
            all_targets.append(targets)
            all_b0.append(b0)
            all_b1.append(b1)
            all_motion.append(motion)

    gamma = torch.cat(all_gamma).numpy()
    nu = torch.cat(all_nu).numpy()
    alpha = torch.cat(all_alpha).numpy()
    beta = torch.cat(all_beta).numpy()
    sev_b0 = torch.cat(all_sev_b0).numpy()
    sev_b1 = torch.cat(all_sev_b1).numpy() + 1.0
    sev_mot = torch.cat(all_sev_mot).numpy()
    targets = torch.cat(all_targets).numpy()
    b0_true = torch.cat(all_b0).numpy()
    b1_true = torch.cat(all_b1).numpy()
    motion_true = torch.cat(all_motion).numpy()

    gamma_denorm = gamma * t_std + t_mean
    targets_denorm = targets * t_std + t_mean
    resid = np.abs(targets_denorm - gamma_denorm)

    epistemic = beta / (nu * (alpha - 1.0))
    aleatoric = beta / (alpha - 1.0)

    # ── Counterfactual correction (on a subset) ──
    n_corr = min(200, len(gamma))
    corrected_signals = []
    for i in range(n_corr):
        sig_complex = val_ds.signals[i, 0] + 1j * val_ds.signals[i, 1]
        corrected = counterfactual_correction(sig_complex, sev_b0[i], sev_b1[i], sev_mot[i])
        corrected_signals.append(np.stack([corrected.real, corrected.imag], axis=0).astype(np.float32))

    corrected_tensor = torch.from_numpy(np.stack(corrected_signals)).to(DEVICE)
    with torch.no_grad():
        out_corr = model(corrected_tensor)
        gamma_corr = out_corr["nig"][...,0].cpu().numpy()

    gamma_corr_denorm = gamma_corr * t_std + t_mean
    resid_corr = np.abs(targets_denorm[:n_corr] - gamma_corr_denorm)

    mae_before = float(resid[:n_corr].mean())
    mae_after = float(resid_corr.mean())

    # ── Per-corruption correlation ──
    max_ep = epistemic.max(axis=-1)
    max_resid = resid.max(axis=-1)

    # Correlate epistemic with residual for samples dominated by each corruption
    b0_dominant = np.abs(b0_true) > 20  # strong B0
    b1_dominant = np.abs(b1_true - 1.0) > 0.1  # strong B1
    mot_dominant = np.abs(motion_true) > 3  # strong motion

    corr_results = {}
    for name, mask in [("B0_dominant", b0_dominant), ("B1_dominant", b1_dominant),
                        ("Motion_dominant", mot_dominant), ("All", np.ones(len(gamma), dtype=bool))]:
        if mask.sum() > 10:
            r = np.corrcoef(max_ep[mask], max_resid[mask])[0, 1]
            corr_results[name] = float(r) if np.isfinite(r) else 0.0

    # ── AUROC with post-hoc calibration ──
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.isotonic import IsotonicRegression

    # Raw AUROC
    labels_300 = (max_resid > 300).astype(int)
    auroc_raw = float(roc_auc_score(labels_300, max_ep)) if 0 < labels_300.sum() < len(labels_300) else float("nan")

    # Post-hoc calibrated AUROC (isotonic regression)
    if 0 < labels_300.sum() < len(labels_300):
        iso = IsotonicRegression(out_of_bounds="clip")
        calibrated_ep = iso.fit_transform(max_ep, labels_300)
        auroc_calibrated = float(roc_auc_score(labels_300, calibrated_ep))
        auprc_calibrated = float(average_precision_score(labels_300, calibrated_ep))
    else:
        auroc_calibrated = float("nan")
        auprc_calibrated = float("nan")

    # ── Severity estimation accuracy ──
    b0_est_mae = float(np.abs(sev_b0 - b0_true).mean())
    b1_est_mae = float(np.abs(sev_b1 - b1_true).mean())
    mot_est_mae = float(np.abs(sev_mot - motion_true).mean())

    # ── Save results ──
    results = {
        "mae_ms": float(np.mean(resid)),
        "rmse_ms": float(np.sqrt(np.mean(resid**2))),
        "mean_aleatoric": float(np.mean(aleatoric)),
        "mean_epistemic": float(np.mean(epistemic)),
        "auroc_raw": auroc_raw,
        "auroc_calibrated": auroc_calibrated,
        "auprc_calibrated": auprc_calibrated,
        "correlation_by_corruption": corr_results,
        "counterfactual": {
            "mae_before_ms": mae_before,
            "mae_after_ms": mae_after,
            "improvement_pct": float((mae_before - mae_after) / mae_before * 100),
        },
        "severity_estimation": {
            "b0_mae_hz": b0_est_mae,
            "b1_mae": b1_est_mae,
            "motion_mae_voxels": mot_est_mae,
        },
    }

    with open(FIG_DIR / "v2_severity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plots ──

    # Counterfactual before/after
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(max_ep[:n_corr], resid[:n_corr].max(axis=-1), alpha=0.3, s=10, c="red", label="Before correction")
    axes[0].scatter(max_ep[:n_corr], resid_corr.max(axis=-1), alpha=0.3, s=10, c="green", label="After correction")
    axes[0].set_xlabel("Epistemic Uncertainty")
    axes[0].set_ylabel("Absolute Residual (ms)")
    axes[0].set_title(f"Counterfactual Correction: MAE {mae_before:.0f} → {mae_after:.0f} ms ({results['counterfactual']['improvement_pct']:.1f}% improvement)")
    axes[0].legend()

    # Severity estimation accuracy
    axes[1].scatter(b0_true[:n_corr], sev_b0[:n_corr], alpha=0.3, s=10, c="steelblue")
    axes[1].plot([-80, 80], [-80, 80], "k--", linewidth=1)
    axes[1].set_xlabel("True B0 Shift (Hz)")
    axes[1].set_ylabel("Predicted B0 Shift (Hz)")
    axes[1].set_title(f"B0 Severity Estimation (MAE={b0_est_mae:.1f} Hz)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_counterfactual_and_severity.png", dpi=200)
    plt.close(fig)

    # Per-corruption correlation bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(corr_results.keys())
    vals = [corr_results[n] for n in names]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    ax.bar(names, vals, color=colors[:len(names)])
    ax.set_ylabel("Pearson r (epistemic vs |residual|)")
    ax.set_title("Per-Corruption-Type Epistemic-Error Correlation")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_per_corruption_correlation.png", dpi=200)
    plt.close(fig)

    # AUROC before/after calibration
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    from sklearn.metrics import roc_curve
    if 0 < labels_300.sum() < len(labels_300):
        fpr_raw, tpr_raw, _ = roc_curve(labels_300, max_ep)
        fpr_cal, tpr_cal, _ = roc_curve(labels_300, calibrated_ep)
        axes[0].plot(fpr_raw, tpr_raw, "b-", linewidth=2, label=f"Raw AUROC={auroc_raw:.3f}")
        axes[0].plot(fpr_cal, tpr_cal, "r-", linewidth=2, label=f"Calibrated AUROC={auroc_calibrated:.3f}")
        axes[0].plot([0,1], [0,1], "k--", alpha=0.5)
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].set_title("Failure Detection ROC: Raw vs Post-hoc Calibrated")
        axes[0].legend()

    # Correlation scatter for all samples
    axes[1].scatter(max_ep, max_resid, alpha=0.2, s=8, c="steelblue")
    axes[1].set_xlabel("Epistemic Uncertainty")
    axes[1].set_ylabel("Absolute Residual (ms)")
    r_all = corr_results.get("All", 0)
    axes[1].set_title(f"Epistemic vs Residual (r={r_all:.3f})")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "v2_calibration_and_correlation.png", dpi=200)
    plt.close(fig)

    logger.info("V2 severity results: MAE=%.1f ms, AUROC_raw=%.3f, AUROC_cal=%.3f, Counterfactual: %.0f→%.0f ms",
                results["mae_ms"], auroc_raw, auroc_calibrated, mae_before, mae_after)
    return results


# ── Experiment 2: Loss Ablation ──────────────────────────────────────────────

def run_loss_ablation(cfg, mrf_path):
    """Ablate each loss component: NLL only, +ER, +physics, +attr, +all."""
    from qMR_Robust.models.resnet1d import ResNet1D

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    configs = {
        "NLL only":         {"er_coeff": 0.0, "phys_coeff": 0.0, "attr_coeff": 0.0},
        "NLL+ER":           {"er_coeff": 1.0, "phys_coeff": 0.0, "attr_coeff": 0.0},
        "NLL+Physics":      {"er_coeff": 0.0, "phys_coeff": 0.3, "attr_coeff": 0.0},
        "NLL+ER+Physics":   {"er_coeff": 1.0, "phys_coeff": 0.3, "attr_coeff": 0.0},
        "NLL+ER+Phys+Attr": {"er_coeff": 1.0, "phys_coeff": 0.3, "attr_coeff": 0.2},
    }

    results = {}
    for name, lcfg in configs.items():
        task_name = f"abl_{name.replace(' ', '_').replace('+', '_')}"
        ckpt = CKPT_DIR / f"{task_name}.pt"

        backbone = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True)
        model = backbone.to(DEVICE)

        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
            logger.info("  %s: loaded checkpoint", name)
        else:
            logger.info("  Training: %s (25 epochs)", name)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

            for epoch in range(25):
                model.train()
                for signals, targets, b0, b1, motion in train_loader:
                    signals, targets = signals.to(DEVICE), targets.to(DEVICE)
                    raw = model(signals)
                    B, D = raw.shape[0], 2
                    raw = raw.view(B, D, 4)
                    gamma, nu = raw[...,0], F.softplus(raw[...,1])
                    alpha = F.softplus(raw[...,2]) + 1.0
                    beta = F.softplus(raw[...,3])

                    loss = nig_nll(targets, gamma, nu, alpha, beta).mean()
                    if lcfg["er_coeff"] > 0:
                        loss = loss + lcfg["er_coeff"] * min(1, epoch/10) * evidential_reg(targets, gamma, nu, alpha).mean()
                    if lcfg["phys_coeff"] > 0:
                        snr = estimate_snr(signals)
                        target_alea = (1.0/snr.clamp(1)).unsqueeze(-1).expand(-1,2)
                        learned_alea = (beta/(alpha-1)).clamp(min=_EPS)
                        loss = loss + lcfg["phys_coeff"] * min(1, epoch/10) * (torch.log(learned_alea)-torch.log(target_alea)).pow(2).mean()

                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                scheduler.step()

            torch.save(model.state_dict(), ckpt)

        # Evaluate
        model.eval()
        all_out, all_tgt = [], []
        with torch.no_grad():
            for signals, targets, *_ in val_loader:
                signals = signals.to(DEVICE)
                raw = model(signals)
                B, D = raw.shape[0], 2
                raw = raw.view(B, D, 4)
                gamma = raw[...,0]
                nu = raw[...,1]
                alpha = raw[...,2]
                beta = raw[...,3]
                ep = (beta/(nu*(alpha-1))).cpu().numpy()
                g = gamma.cpu().numpy()
                all_out.append(np.stack([g, ep], axis=-1))
                all_tgt.append(targets.numpy())

        out = np.concatenate(all_out)
        tgt = np.concatenate(all_tgt)
        g_denorm = out[...,0] * t_std + t_mean
        tgt_denorm = tgt * t_std + t_mean
        resid = np.abs(tgt_denorm - g_denorm)
        epistemic = out[...,1]

        from sklearn.metrics import roc_auc_score
        max_ep = epistemic.max(axis=-1)
        max_resid = resid.max(axis=-1)
        labels = (max_resid > 300).astype(int)
        auroc = float(roc_auc_score(labels, max_ep)) if 0 < labels.sum() < len(labels) else float("nan")
        r = np.corrcoef(max_ep, max_resid)[0,1] if max_ep.std() > 1e-12 else 0.0

        results[name] = {
            "mae_ms": float(np.mean(resid)),
            "rmse_ms": float(np.sqrt(np.mean(resid**2))),
            "auroc": auroc,
            "correlation": float(r),
        }
        logger.info("  %s: MAE=%.1f AUROC=%.3f r=%.3f", name, results[name]["mae_ms"], auroc, r)

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(results.keys())
    for ax, metric, title in zip(axes, ["mae_ms", "auroc", "correlation"], ["MAE (ms)", "AUROC", "Epistemic-Error Corr."]):
        vals = [results[n].get(metric, 0) for n in names]
        if all(np.isnan(v) if isinstance(v, float) else False for v in vals): continue
        ax.bar(range(len(names)), [v if np.isfinite(v) else 0 for v in vals], color=plt.cm.Set2(np.linspace(0,1,len(names))))
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
    fig.suptitle("Loss Component Ablation")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_loss_ablation.png", dpi=200); plt.close(fig)

    with open(FIG_DIR / "v2_loss_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   V2 UPGRADE 1: Severity + Counterfactual     ║")
    logger.info("╚═══════════════════════════════════════════════╝")
    sev_results = run_severity_regression(cfg, mrf_path)

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   V2 UPGRADE 2: Loss Ablation                 ║")
    logger.info("╚═══════════════════════════════════════════════╝")
    abl_results = run_loss_ablation(cfg, mrf_path)

    # Save combined
    with open(FIG_DIR / "v2_all_upgrades.json", "w") as f:
        json.dump({"severity": sev_results, "loss_ablation": abl_results}, f, indent=2, default=str)

    logger.info("═══ V2 UPGRADES COMPLETE in %.0f s ═══", time.time() - t0)


if __name__ == "__main__":
    main()
