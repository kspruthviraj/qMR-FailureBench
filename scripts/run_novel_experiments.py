#!/usr/bin/env python3
"""
run_novel_experiments.py — Novel Failure Forecasting with all three innovations.

Innovations:
  1. Explainable Corruption Attribution (dual-head: regression + corruption source)
  2. Physics-Aware Evidential Loss (SNR-anchored aleatoric uncertainty)
  3. qMR-FailureBench benchmark packaging

Experiments:
  A. Standard Evidential vs Physics-Aware Evidential
  B. Corruption Attribution accuracy
  C. Joint model evaluation
  D. Benchmark packaging
"""

from __future__ import annotations
import sys

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
import yaml
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("novel")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset with corruption metadata ─────────────────────────────────────────

class MRFNovelDataset(Dataset):
    """MRF dataset that also returns corruption metadata for attribution training."""

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

    def set_norm(self, mean, std):
        self.mean, self.std = mean, std

    def __len__(self): return len(self.signals)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.signals[i]),
            (torch.from_numpy(self.params[i]) - self.mean) / self.std,
            torch.tensor(self.b0[i]),
            torch.tensor(self.b1[i]),
            torch.tensor(self.motion[i]),
        )


# ── Losses ───────────────────────────────────────────────────────────────────

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


# ── Training ─────────────────────────────────────────────────────────────────

def train_novel_model(cfg, mrf_path, n_epochs=30):
    """Train the dual-head model with physics-aware loss + attribution."""
    from qMR_Robust.models.resnet1d import ResNet1D
    from qMR_Robust.models.corruption_attribution import EvidentialWithAttribution

    ckpt_path = ROOT / "results" / "checkpoints" / "novel_dual_head_multilabel_v2.pt"
    if ckpt_path.exists():
        logger.info("Novel model checkpoint exists, loading.")
        backbone = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1)
        model = EvidentialWithAttribution(backbone, output_dim=2, hidden_dim=128)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model._train_mean = np.zeros(2)
        model._train_std = np.ones(2)
        return model.to(DEVICE)

    train_ds = MRFNovelDataset(mrf_path, split="train")
    val_ds = MRFNovelDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    backbone = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1)
    model = EvidentialWithAttribution(backbone, output_dim=2, hidden_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val, best_state = float("inf"), None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_b = 0.0, 0

        for signals, targets, b0, b1, motion in train_loader:
            signals = signals.to(DEVICE)
            targets = targets.to(DEVICE)
            b0, b1, motion = b0.to(DEVICE), b1.to(DEVICE), motion.to(DEVICE)

            out = model(signals)
            nig = out["nig"]
            attr_logits = out["attribution"]

            # NIG regression loss
            gamma, nu, alpha, beta = nig[...,0], nig[...,1], nig[...,2], nig[...,3]
            nll = nig_nll(targets, gamma, nu, alpha, beta)
            er = evidential_reg(targets, gamma, nu, alpha)

            # Physics-aware anchor
            snr = estimate_snr(signals)
            target_alea = (1.0 / snr.clamp(min=1.0)).unsqueeze(-1).expand(-1, 2)
            learned_alea = (beta / (alpha - 1.0)).clamp(min=_EPS)
            physics_anchor = (torch.log(learned_alea) - torch.log(target_alea)).pow(2)

            # Independent multi-label attribution loss. Entangled samples
            # retain multiple positive labels instead of being normalised to one class.
            attr_targets = torch.stack([
                (b0.abs() > 1.0).float(),
                ((b1 - 1.0).abs() > 0.01).float(),
                (motion.abs() > 0.5).float(),
            ], dim=-1)
            attr_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                attr_logits, attr_targets
            )

            anneal = min(1.0, epoch / max(15, 1))
            # Regression loss gets full weight; attribution is auxiliary with small coeff
            loss = nll.mean() + anneal * 1.0 * er.mean() + 0.1 * anneal * physics_anchor.mean() + 0.1 * anneal * attr_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        scheduler.step()

        # Validation
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for signals, targets, b0, b1, motion in val_loader:
                signals, targets = signals.to(DEVICE), targets.to(DEVICE)
                out = model(signals)
                nig = out["nig"]
                gamma, nu, alpha, beta = nig[...,0], nig[...,1], nig[...,2], nig[...,3]
                nll = nig_nll(targets, gamma, nu, alpha, beta)
                er = evidential_reg(targets, gamma, nu, alpha)
                vloss += (nll.mean() + er.mean()).item() * signals.size(0)
                vn += signals.size(0)
        vloss /= vn

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch+1) % 5 == 0 or epoch == 0:
            logger.info("  Novel epoch %d/%d | train=%.4f val=%.4f", epoch+1, n_epochs, epoch_loss/n_b, vloss)

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt_path)
    return model.to(DEVICE)


def evaluate_novel_model(model, mrf_path):
    """Evaluate the novel model: regression + attribution + physics-aware uncertainty."""
    from qMR_Robust.eval.metrics import failure_detection_metrics, expected_calibration_error

    train_ds = MRFNovelDataset(mrf_path, split="train")
    val_ds = MRFNovelDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
    all_attr_pred, all_attr_true = [], []
    all_targets, all_b0, all_b1, all_motion = [], [], [], []

    with torch.no_grad():
        for signals, targets, b0, b1, motion in val_loader:
            signals = signals.to(DEVICE)
            out = model(signals)
            nig = out["nig"]
            all_gamma.append(nig[...,0].cpu())
            all_nu.append(nig[...,1].cpu())
            all_alpha.append(nig[...,2].cpu())
            all_beta.append(nig[...,3].cpu())
            all_attr_pred.append(torch.sigmoid(out["attribution"]).cpu())
            all_targets.append(targets)
            all_b0.append(b0)
            all_b1.append(b1)
            all_motion.append(motion)

    gamma = torch.cat(all_gamma).numpy()
    nu = torch.cat(all_nu).numpy()
    alpha = torch.cat(all_alpha).numpy()
    beta = torch.cat(all_beta).numpy()
    attr_pred = torch.cat(all_attr_pred).numpy()
    targets = torch.cat(all_targets).numpy()
    b0 = torch.cat(all_b0).numpy()
    b1 = torch.cat(all_b1).numpy()
    motion = torch.cat(all_motion).numpy()

    # Denormalize
    gamma_denorm = gamma * t_std + t_mean
    targets_denorm = targets * t_std + t_mean
    resid = np.abs(targets_denorm - gamma_denorm)

    # Uncertainty
    epistemic = beta / (nu * (alpha - 1.0))
    aleatoric = beta / (alpha - 1.0)

    # Attribution ground truth
    attr_true = np.stack([
        (np.abs(b0) > 1.0).astype(float),
        (np.abs(b1 - 1.0) > 0.01).astype(float),
        (np.abs(motion) > 0.5).astype(float),
    ], axis=-1)

    # Attribution accuracy
    attr_pred_binary = (attr_pred > 0.5).astype(int)
    attr_true_binary = attr_true.astype(int)
    attr_acc = float((attr_pred_binary == attr_true_binary).mean())

    # Per-source attribution accuracy
    source_names = ["B0", "B1", "Motion"]
    per_source_acc = {}
    for i, name in enumerate(source_names):
        mask = attr_true[:, i] > 0.5
        if mask.sum() > 0:
            per_source_acc[name] = float((attr_pred_binary[mask, i] == 1).mean())
        else:
            per_source_acc[name] = float("nan")

    # Failure detection
    max_ep = epistemic.max(axis=-1)
    max_resid = resid.max(axis=-1)
    fdm = failure_detection_metrics(epistemic, resid, tolerance=300.0)

    # ECE
    ece, bin_pred, bin_act = expected_calibration_error(epistemic, resid, n_bins=10)

    results = {
        "mae_ms": float(np.mean(resid)),
        "rmse_ms": float(np.sqrt(np.mean(resid**2))),
        "mean_aleatoric": float(np.mean(aleatoric)),
        "mean_epistemic": float(np.mean(epistemic)),
        "auroc": fdm["auroc"],
        "auprc": fdm["auprc"],
        "ece": ece,
        "attribution_accuracy": attr_acc,
        "per_source_attribution": per_source_acc,
    }

    # ── Plots ──

    # Attribution confusion-style bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (name, ax) in enumerate(zip(source_names, axes)):
        mask_true = attr_true[:, i] > 0.5
        mask_false = ~mask_true
        if mask_true.sum() > 0:
            ax.hist(attr_pred[mask_true, i], bins=30, alpha=0.7, color="green", label=f"True {name}", density=True)
        if mask_false.sum() > 0:
            ax.hist(attr_pred[mask_false, i], bins=30, alpha=0.7, color="red", label=f"Not {name}", density=True)
        ax.set_title(f"Attribution: {name}")
        ax.set_xlabel("Predicted Probability")
        ax.legend()
    fig.suptitle("Corruption Attribution: Predicted Probability by True Label")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "novel_attribution_distributions.png", dpi=200)
    plt.close(fig)

    # Epistemic vs residual with attribution coloring
    fig, ax = plt.subplots(figsize=(8, 6))
    max_attr = attr_pred.argmax(axis=-1)
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    for i, (name, color) in enumerate(zip(source_names, colors)):
        mask = max_attr == i
        if mask.sum() > 0:
            ax.scatter(max_ep[mask], max_resid[mask], alpha=0.3, s=10, c=color, label=f"Attributed: {name}")
    ax.set_xlabel("Epistemic Uncertainty")
    ax.set_ylabel("Absolute Residual (ms)")
    ax.set_title("Failure Forecasting with Corruption Attribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "novel_attributed_scatter.png", dpi=200)
    plt.close(fig)

    # Physics-aware: aleatoric vs SNR
    val_loader2 = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)
    snrs = []
    with torch.no_grad():
        for batch in val_loader2:
            signals = batch[0].to(DEVICE)
            snrs.append(estimate_snr(signals).cpu().numpy())
    snrs = np.concatenate(snrs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(snrs, aleatoric.mean(axis=-1), alpha=0.3, s=8, c="steelblue")
    ax.set_xlabel("Estimated Signal SNR")
    ax.set_ylabel("Learned Aleatoric Uncertainty")
    ax.set_title("Physics-Aware: Aleatoric Uncertainty vs Signal SNR")
    # Fit line
    valid = np.isfinite(snrs) & np.isfinite(aleatoric.mean(-1))
    if valid.sum() > 10:
        z = np.polyfit(snrs[valid], aleatoric.mean(-1)[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(snrs[valid].min(), snrs[valid].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Slope={z[0]:.4f}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "novel_physics_aware_snr.png", dpi=200)
    plt.close(fig)

    # Reliability diagram
    from qMR_Robust.eval.metrics import plot_reliability_diagram, plot_failure_detection_roc
    plot_reliability_diagram(bin_pred, bin_act, ece, "novel_mrf", FIG_DIR)
    plot_failure_detection_roc(epistemic, resid, [100, 200, 300, 500], "novel_mrf", FIG_DIR)

    with open(FIG_DIR / "novel_experiment_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Novel model results: MAE=%.1f ms, AUROC=%.3f, Attribution Acc=%.3f",
                results["mae_ms"], results["auroc"], results["attribution_accuracy"])
    return results


# ── Benchmark packaging ──────────────────────────────────────────────────────

def package_benchmark(cfg):
    """Package qMR-FailureBench."""
    from qMR_Robust.benchmark import package_benchmark as pkg

    bench_dir = ROOT / "qMR-FailureBench"
    if (bench_dir / "metadata.json").exists():
        logger.info("Benchmark already packaged.")
        return

    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    mrs_path = str(ROOT / cfg["paths"]["failure_forecast_mrs"])
    pkg(mrf_path, mrs_path, str(bench_dir))
    logger.info("Benchmark packaged → %s", bench_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    # 1. Train novel dual-head model
    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   NOVEL: Dual-Head Model Training              ║")
    logger.info("╚═══════════════════════════════════════════════╝")
    model = train_novel_model(cfg, mrf_path, n_epochs=50)

    # 2. Evaluate
    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   NOVEL: Evaluation + Attribution              ║")
    logger.info("╚═══════════════════════════════════════════════╝")
    results = evaluate_novel_model(model, mrf_path)

    # 3. Package benchmark
    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   NOVEL: qMR-FailureBench Packaging            ║")
    logger.info("╚═══════════════════════════════════════════════╝")
    package_benchmark(cfg)

    elapsed = time.time() - t_start
    logger.info("═══ NOVEL EXPERIMENTS COMPLETE in %.0f s ═══", elapsed)


if __name__ == "__main__":
    main()
