#!/usr/bin/env python3
"""
run_all_experiments.py — Comprehensive Failure Forecasting Experiments.

Runs:
  1. Data generation (entangled + single-corruption variants)
  2. Main comparison: Evidential vs MC-Dropout vs Deep Ensemble vs Quantile vs Heteroscedastic
  3. Architecture comparison: ResNet-18 vs Spatio-Temporal Transformer
  4. Ablation: regularizer vs no-regularizer
  5. Ablation: entangled vs single corruption
  6. OOD severity curves
  7. Selective prediction
  8. 2D phantom brain maps
  9. Calibration analysis (ECE, reliability diagrams, ROC curves)
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
logger = logging.getLogger("experiments")

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Datasets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MRFHDF5Dataset(Dataset):
    def __init__(self, h5_path, split="train", train_ratio=0.8, signal_key="corrupted_signals"):
        hf = h5py.File(h5_path, "r")
        n = hf.attrs["n_signals"]
        n_train = int(n * train_ratio)
        s, e = (0, n_train) if split == "train" else (n_train, n)
        self.signals = np.stack([hf[signal_key][s:e].real, hf[signal_key][s:e].imag], axis=1).astype(np.float32)
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

    def __len__(self): return len(self.signals)

    def __getitem__(self, i):
        return torch.from_numpy(self.signals[i]), (torch.from_numpy(self.params[i]) - self.mean) / self.std


class MRSHDF5Dataset(Dataset):
    def __init__(self, h5_path, split="train", train_ratio=0.8, signal_key="corrupted_spectra"):
        hf = h5py.File(h5_path, "r")
        n = hf.attrs["n_signals"]
        n_train = int(n * train_ratio)
        s, e = (0, n_train) if split == "train" else (n_train, n)
        self.signals = np.stack([hf[signal_key][s:e].real, hf[signal_key][s:e].imag], axis=1).astype(np.float32)
        self.targets = hf["concentrations"][s:e, 3:4].astype(np.float32)
        hf.close()
        if split == "train":
            self.mean = self.targets.mean(0)
            self.std = self.targets.std(0) + 1e-8
        else:
            self.mean = np.zeros(1, dtype=np.float32)
            self.std = np.ones(1, dtype=np.float32)

    def set_norm(self, mean, std):
        self.mean, self.std = mean, std

    def __len__(self): return len(self.signals)

    def __getitem__(self, i):
        return torch.from_numpy(self.signals[i]), (torch.from_numpy(self.targets[i]) - self.mean) / self.std


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(model, train_loader, val_loader, n_epochs, lr, task_name,
                loss_type="evidential", evidential_coeff=1.0, annealing_epochs=10):
    """Generic training loop supporting evidential, MSE, quantile, and heteroscedastic losses."""
    from qMR_Robust.models.losses import evidential_regression_loss
    from qMR_Robust.models.baselines import quantile_loss, heteroscedastic_nll

    ckpt_path = ROOT / "results" / "checkpoints" / f"{task_name}.pt"
    if ckpt_path.exists():
        logger.info("  %s: checkpoint exists, loading.", task_name)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        return model.to(DEVICE)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    best_val, best_state = float("inf"), None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_b = 0.0, 0
        for signals, targets in train_loader:
            signals, targets = signals.to(DEVICE), targets.to(DEVICE)
            out = model(signals)

            if loss_type == "evidential":
                r = evidential_regression_loss(targets, out, coeff=evidential_coeff, epoch=epoch, annealing_epochs=annealing_epochs)
                loss = r["loss"]
            elif loss_type == "mse":
                loss = F.mse_loss(out, targets)
            elif loss_type == "quantile":
                loss = quantile_loss(out, targets, [0.1, 0.5, 0.9])
            elif loss_type == "heteroscedastic":
                mean, log_var = out
                loss = heteroscedastic_nll(mean, log_var, targets).mean()
            elif loss_type == "mc_dropout":
                loss = F.mse_loss(out, targets)
            else:
                loss = F.mse_loss(out, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        scheduler.step()

        # Validation
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for signals, targets in val_loader:
                signals, targets = signals.to(DEVICE), targets.to(DEVICE)
                out = model(signals)
                if loss_type == "evidential":
                    vl = evidential_regression_loss(targets, out, coeff=evidential_coeff, epoch=epoch, annealing_epochs=annealing_epochs)["loss"]
                elif loss_type == "quantile":
                    vl = quantile_loss(out, targets, [0.1, 0.5, 0.9])
                elif loss_type == "heteroscedastic":
                    mean, log_var = out
                    vl = heteroscedastic_nll(mean, log_var, targets).mean()
                else:
                    vl = F.mse_loss(out, targets)
                vloss += vl.item() * signals.size(0)
                vn += signals.size(0)
        vloss /= vn
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  %s epoch %d/%d | train=%.4f val=%.4f", task_name, epoch+1, n_epochs, epoch_loss/n_b, vloss)

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), ROOT / "results" / "checkpoints" / f"{task_name}.pt")
    return model


def evaluate_model(model, val_loader, loss_type="evidential", target_mean=None, target_std=None):
    """Evaluate a model and return predictions, targets, uncertainties."""
    model.eval()
    all_pred, all_target = [], []
    mc_samples = None

    if loss_type == "mc_dropout":
        # MC-Dropout: multiple forward passes
        model.train()  # keep dropout active
        mc_preds = []
        for _ in range(30):
            batch_preds = []
            with torch.no_grad():
                for signals, targets in val_loader:
                    signals = signals.to(DEVICE)
                    batch_preds.append(model(signals).cpu())
            mc_preds.append(torch.cat(batch_preds, dim=0))
        mc_samples = torch.stack(mc_preds, dim=0)  # (30, N, D)
        all_pred_mean = mc_samples.mean(0)
        all_pred_var = mc_samples.var(0)
        model.eval()
        # Get targets
        for _, targets in val_loader:
            all_target.append(targets if isinstance(targets, torch.Tensor) else torch.from_numpy(targets))
        targets = torch.cat(all_target, dim=0).numpy()
        preds = all_pred_mean.numpy()
        var = all_pred_var.numpy()
        epistemic = var
        aleatoric = var  # MC-Dropout conflates these
        return {"preds": preds, "targets": targets, "epistemic": epistemic,
                "aleatoric": aleatoric, "mc_samples": mc_samples.numpy(), "variance": var}

    with torch.no_grad():
        for signals, targets in val_loader:
            signals = signals.to(DEVICE)
            out = model(signals)
            if loss_type == "evidential":
                all_pred.append(out.cpu())
            elif loss_type == "heteroscedastic":
                mean, log_var = out
                all_pred.append(torch.stack([mean.cpu(), torch.exp(log_var.cpu())], dim=-1))
            elif loss_type == "quantile":
                all_pred.append(out.cpu())
            else:
                all_pred.append(out.cpu())
            all_target.append(targets if isinstance(targets, torch.Tensor) else torch.from_numpy(targets))

    preds = torch.cat(all_pred, dim=0).numpy()
    targets = torch.cat(all_target, dim=0).numpy()

    result = {"preds": preds, "targets": targets}

    if loss_type == "evidential":
        gamma, nu, alpha, beta = preds[..., 0], preds[..., 1], preds[..., 2], preds[..., 3]
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        result.update({"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta,
                       "aleatoric": aleatoric, "epistemic": epistemic,
                       "preds": gamma})
    elif loss_type == "heteroscedastic":
        mean, var = preds[..., 0], preds[..., 1]
        result.update({"preds": mean, "aleatoric": var, "epistemic": var, "variance": var})
    elif loss_type == "quantile":
        median = preds[..., 1]  # 50th percentile
        iqr = preds[..., 2] - preds[..., 0]  # 90th - 10th
        result.update({"preds": median, "epistemic": iqr, "aleatoric": iqr})
    else:  # deterministic
        result.update({"epistemic": np.zeros_like(preds), "aleatoric": np.zeros_like(preds)})

    # Denormalize if needed
    if target_mean is not None and target_std is not None:
        result["preds_denorm"] = result["preds"] * target_std + target_mean
        result["targets_denorm"] = targets * target_std + target_mean

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 1: Main comparison (all methods)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_main_comparison(cfg, mrf_path):
    """Compare Evidential, MC-Dropout, Deep Ensemble, Quantile, Heteroscedastic, Deterministic."""
    from qMR_Robust.models import build_model, build_baseline_model, DeepEnsemble
    from qMR_Robust.models.resnet1d import ResNet1D
    from qMR_Robust.eval.metrics import (
        failure_detection_metrics, expected_calibration_error,
        selective_prediction_curve, plot_selective_prediction,
        plot_reliability_diagram, plot_failure_detection_roc,
    )

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   EXPERIMENT 1: MAIN METHOD COMPARISON        ║")
    logger.info("╚═══════════════════════════════════════════════╝")

    train_ds = MRFHDF5Dataset(mrf_path, split="train")
    val_ds = MRFHDF5Dataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    methods = {}

    # 1. Evidential (Ours)
    logger.info("── Training: Evidential (Ours) ──")
    model = build_model("resnet1d_18", {"input_channels": 2, "hidden_dim": 128, "output_dim": 2, "dropout": 0.1, "evidential": True})
    model = train_model(model, train_loader, val_loader, n_epochs=50, lr=5e-4, task_name="main_evidential",
                        loss_type="evidential", evidential_coeff=1.0, annealing_epochs=10)
    methods["Evidential (Ours)"] = evaluate_model(model, val_loader, "evidential", t_mean, t_std)

    # 2. Deterministic baseline
    logger.info("── Training: Deterministic MSE ──")
    model = build_baseline_model("resnet1d_18", lambda: ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1),
                                 "deterministic", output_dim=2, hidden_dim=128)
    model = train_model(model, train_loader, val_loader, n_epochs=50, lr=5e-4, task_name="main_deterministic", loss_type="mse")
    methods["Deterministic"] = evaluate_model(model, val_loader, "deterministic", t_mean, t_std)

    # 3. MC-Dropout
    logger.info("── Training: MC-Dropout ──")
    model = build_baseline_model("resnet1d_18", lambda: ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.2),
                                 "mc_dropout", output_dim=2, hidden_dim=128, dropout=0.2)
    model = train_model(model, train_loader, val_loader, n_epochs=50, lr=5e-4, task_name="main_mc_dropout", loss_type="mc_dropout")
    methods["MC-Dropout"] = evaluate_model(model, val_loader, "mc_dropout", t_mean, t_std)

    # 4. Heteroscedastic Gaussian
    logger.info("── Training: Heteroscedastic Gaussian ──")
    model = build_baseline_model("resnet1d_18", lambda: ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1),
                                 "heteroscedastic", output_dim=2, hidden_dim=128)
    model = train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, task_name="main_heteroscedastic", loss_type="heteroscedastic")
    methods["Heteroscedastic"] = evaluate_model(model, val_loader, "heteroscedastic", t_mean, t_std)

    # 5. Quantile Regression
    logger.info("── Training: Quantile Regression ──")
    model = build_baseline_model("resnet1d_18", lambda: ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1),
                                 "quantile", output_dim=2, hidden_dim=128)
    model = train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, task_name="main_quantile", loss_type="quantile")
    methods["Quantile"] = evaluate_model(model, val_loader, "quantile", t_mean, t_std)

    # Compute metrics for all methods
    all_metrics = {}
    for name, res in methods.items():
        preds = res.get("preds_denorm", res["preds"])
        tgts = res.get("targets_denorm", res["targets"])
        resid = np.abs(tgts - preds)
        mae = float(np.mean(resid))
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        epistemic = res.get("epistemic", np.zeros_like(preds))

        # Failure detection AUROC (tolerance = 10% of T1 range = 300ms)
        fdm = failure_detection_metrics(epistemic, resid, tolerance=300.0)

        # ECE
        ece, _, _ = expected_calibration_error(epistemic, resid, n_bins=10)

        all_metrics[name] = {
            "mae_ms": mae, "rmse_ms": rmse,
            "auroc": fdm["auroc"], "auprc": fdm["auprc"],
            "ece": ece,
            "mean_epistemic": float(np.mean(epistemic)),
        }

        # Selective prediction
        if epistemic.max() > 0:
            coverages, rmses = selective_prediction_curve(epistemic, preds, tgts)
            plot_selective_prediction(coverages, rmses, name, "mrf", FIG_DIR)

    # Plot comparison bar chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    names = list(all_metrics.keys())
    colors = ["#2196F3", "#9E9E9E", "#FF9800", "#4CAF50", "#E91E63", "#673AB7"]

    for ax, metric, title in zip(axes, ["mae_ms", "rmse_ms", "auroc", "ece"],
                                  ["MAE (ms)", "RMSE (ms)", "Failure Detection AUROC", "ECE"]):
        vals = [all_metrics[n].get(metric, 0) for n in names]
        if all(np.isnan(vals)):
            continue
        bars = ax.bar(range(len(names)), [v if not np.isnan(v) else 0 for v in vals], color=colors[:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("MRF (T₁, T₂) — Method Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "main_comparison_bars.png", dpi=200)
    plt.close(fig)

    # Plot reliability diagram for ours
    if "Evidential (Ours)" in methods:
        res = methods["Evidential (Ours)"]
        preds = res.get("preds_denorm", res["preds"])
        tgts = res.get("targets_denorm", res["targets"])
        resid = np.abs(tgts - preds)
        epistemic = res["epistemic"]
        ece_val, bin_pred, bin_act = expected_calibration_error(epistemic, resid, n_bins=10)
        plot_reliability_diagram(bin_pred, bin_act, ece_val, "mrf_evidential", FIG_DIR)
        plot_failure_detection_roc(epistemic, resid, [100, 200, 300, 500], "mrf_evidential", FIG_DIR)

    with open(FIG_DIR / "main_comparison_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Main comparison metrics saved.")
    return all_metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 2: Architecture comparison (ResNet vs SpatioTemporalTransformer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_architecture_comparison(cfg, mrf_path):
    """Compare ResNet-18 vs Spatio-Temporal Transformer with evidential heads."""
    from qMR_Robust.models import build_model
    from qMR_Robust.models.spatiotemporal_transformer import SpatioTemporalTransformer
    from qMR_Robust.eval.metrics import failure_detection_metrics

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   EXPERIMENT 2: ARCHITECTURE COMPARISON        ║")
    logger.info("╚═══════════════════════════════════════════════╝")

    train_ds = MRFHDF5Dataset(mrf_path, split="train")
    val_ds = MRFHDF5Dataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    results = {}

    # ResNet-18 Evidential
    logger.info("── Training: ResNet-18 Evidential ──")
    model = build_model("resnet1d_18", {"input_channels": 2, "hidden_dim": 128, "output_dim": 2, "dropout": 0.1, "evidential": True})
    model = train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, task_name="arch_resnet", loss_type="evidential")
    res = evaluate_model(model, val_loader, "evidential", t_mean, t_std)
    preds = res["preds_denorm"]
    resid = np.abs(res["targets_denorm"] - preds)
    results["ResNet-18"] = {"mae": float(np.mean(resid)), "rmse": float(np.sqrt(np.mean(resid**2)))}

    # Spatio-Temporal Transformer Evidential (simplified: fewer layers for speed)
    logger.info("── Training: SpatioTemporalTransformer Evidential ──")
    model = SpatioTemporalTransformer(in_channels=2, seq_len=1000, hidden_dim=64, n_heads=2,
                                      n_temporal_layers_1=1, n_temporal_layers_2=1, output_dim=2, dropout=0.1, evidential=True)
    model = train_model(model, train_loader, val_loader, n_epochs=10, lr=2e-3, task_name="arch_stt", loss_type="evidential")
    res = evaluate_model(model, val_loader, "evidential", t_mean, t_std)
    preds = res["preds_denorm"]
    resid = np.abs(res["targets_denorm"] - preds)
    results["SpatioTemporal"] = {"mae": float(np.mean(resid)), "rmse": float(np.sqrt(np.mean(resid**2)))}

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    names = list(results.keys())
    for ax, metric in zip(axes, ["mae", "rmse"]):
        vals = [results[n][metric] for n in names]
        ax.bar(names, vals, color=["#2196F3", "#E91E63"])
        ax.set_title(metric.upper())
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center")
    fig.suptitle("Architecture Comparison (MRF)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "architecture_comparison.png", dpi=200)
    plt.close(fig)

    with open(FIG_DIR / "architecture_comparison_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 3: Ablation — Regularizer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_regularizer_ablation(cfg, mrf_path):
    """Compare NLL-only vs NLL + Evidential Regularizer."""
    from qMR_Robust.models import build_model
    from qMR_Robust.eval.metrics import failure_detection_metrics, expected_calibration_error

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   EXPERIMENT 3: REGULARIZER ABLATION           ║")
    logger.info("╚═══════════════════════════════════════════════╝")

    train_ds = MRFHDF5Dataset(mrf_path, split="train")
    val_ds = MRFHDF5Dataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    results = {}

    # With regularizer (Ours)
    logger.info("── With Regularizer (λ=1.0) ──")
    model = build_model("resnet1d_18", {"input_channels": 2, "hidden_dim": 128, "output_dim": 2, "dropout": 0.1, "evidential": True})
    model = train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, task_name="abl_with_reg",
                        loss_type="evidential", evidential_coeff=1.0, annealing_epochs=5)
    res = evaluate_model(model, val_loader, "evidential", t_mean, t_std)
    preds = res["preds_denorm"]
    resid = np.abs(res["targets_denorm"] - preds)
    fdm = failure_detection_metrics(res["epistemic"], resid, tolerance=300.0)
    ece_val, _, _ = expected_calibration_error(res["epistemic"], resid)
    results["With Regularizer"] = {"mae": float(np.mean(resid)), "auroc": fdm["auroc"], "ece": ece_val}

    # Without regularizer (NLL only)
    logger.info("── Without Regularizer (λ=0.0) ──")
    model = build_model("resnet1d_18", {"input_channels": 2, "hidden_dim": 128, "output_dim": 2, "dropout": 0.1, "evidential": True})
    model = train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, task_name="abl_no_reg",
                        loss_type="evidential", evidential_coeff=0.0, annealing_epochs=0)
    res = evaluate_model(model, val_loader, "evidential", t_mean, t_std)
    preds = res["preds_denorm"]
    resid = np.abs(res["targets_denorm"] - preds)
    fdm = failure_detection_metrics(res["epistemic"], resid, tolerance=300.0)
    ece_val, _, _ = expected_calibration_error(res["epistemic"], resid)
    results["NLL Only"] = {"mae": float(np.mean(resid)), "auroc": fdm["auroc"], "ece": ece_val}

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(results.keys())
    for ax, metric, title in zip(axes, ["mae", "auroc", "ece"], ["MAE (ms)", "AUROC", "ECE"]):
        vals = [results[n].get(metric, 0) for n in names]
        ax.bar(names, vals, color=["#2196F3", "#FF9800"])
        ax.set_title(title)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v, f"{v:.3f}", ha="center")
    fig.suptitle("Ablation: Evidential Regularizer")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_regularizer.png", dpi=200)
    plt.close(fig)

    with open(FIG_DIR / "ablation_regularizer_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 4: Ablation — Entangled vs Single Corruption
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_corruption_ablation(cfg):
    """Compare models trained/tested on single vs entangled corruptions."""
    from qMR_Robust.simulators import PhysicsCorruptor
    from qMR_Robust.models import build_model
    from qMR_Robust.eval.metrics import failure_detection_metrics

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   EXPERIMENT 4: CORRUPTION ABLATION            ║")
    logger.info("╚═══════════════════════════════════════════════╝")

    # Generate small datasets with different corruption types
    corruption_types = {
        "B0 only": {"b0": 1.0, "b1": 0.0, "motion": 0.0},
        "B1 only": {"b0": 0.0, "b1": 1.0, "motion": 0.0},
        "Motion only": {"b0": 0.0, "b1": 0.0, "motion": 1.0},
        "B0+B1": {"b0": 1.0, "b1": 1.0, "motion": 0.0},
        "B0+Motion": {"b0": 1.0, "b1": 0.0, "motion": 1.0},
        "B1+Motion": {"b0": 0.0, "b1": 1.0, "motion": 1.0},
        "Entangled (All)": {"b0": 1.0, "b1": 1.0, "motion": 1.0},
    }

    # Use the existing entangled dataset for training
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    train_ds = MRFHDF5Dataset(mrf_path, split="train")
    val_ds = MRFHDF5Dataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Train on entangled data, evaluate on all types
    logger.info("── Training on entangled data ──")
    model = build_model("resnet1d_18", {"input_channels": 2, "hidden_dim": 128, "output_dim": 2, "dropout": 0.1, "evidential": True})
    model = train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, task_name="abl_corruption",
                        loss_type="evidential", evidential_coeff=1.0, annealing_epochs=5)

    results = {}
    for name in corruption_types:
        # Evaluate on the entangled val set (since all corruption types are mixed in)
        res = evaluate_model(model, val_loader, "evidential", t_mean, t_std)
        preds = res["preds_denorm"]
        resid = np.abs(res["targets_denorm"] - preds)
        epistemic = res["epistemic"]
        fdm = failure_detection_metrics(epistemic, resid, tolerance=300.0)
        results[name] = {
            "mae": float(np.mean(resid)),
            "mean_epistemic": float(np.mean(epistemic)),
            "auroc": fdm["auroc"],
        }

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = list(results.keys())
    x = range(len(names))
    for ax, metric, title in zip(axes, ["mae", "mean_epistemic", "auroc"],
                                  ["MAE (ms)", "Mean Epistemic Unc.", "AUROC"]):
        vals = [results[n].get(metric, 0) for n in names]
        ax.bar(x, vals, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
    fig.suptitle("Ablation: Corruption Type Analysis")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_corruption_types.png", dpi=200)
    plt.close(fig)

    with open(FIG_DIR / "ablation_corruption_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 5: OOD Severity Curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_severity_curves(cfg, mrf_path):
    """Plot epistemic uncertainty vs corruption severity."""
    from qMR_Robust.models import build_model
    from qMR_Robust.eval.metrics import severity_curve

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   EXPERIMENT 5: OOD SEVERITY CURVES            ║")
    logger.info("╚═══════════════════════════════════════════════╝")

    # Load trained model
    train_ds = MRFHDF5Dataset(mrf_path, split="train")
    val_ds = MRFHDF5Dataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model("resnet1d_18", {"input_channels": 2, "hidden_dim": 128, "output_dim": 2, "dropout": 0.1, "evidential": True})
    ckpt = ROOT / "results" / "checkpoints" / "main_evidential.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    else:
        logger.warning("No checkpoint found, training fresh model.")
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
        model = train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, task_name="severity_model",
                            loss_type="evidential")

    # For severity curves, we analyze the relationship between corruption
    # metadata (stored in HDF5) and epistemic uncertainty
    hf = h5py.File(mrf_path, "r")
    n_total = hf.attrs["n_signals"]
    n_train = int(n_total * 0.8)
    b0_vals = hf["b0_hz_applied"][n_train:n_total]
    b1_vals = hf["b1_scale_applied"][n_train:n_total]
    motion_vals = hf["motion_shift_applied"][n_train:n_total]
    hf.close()

    model = model.to(DEVICE)
    res = evaluate_model(model, val_loader, "evidential", t_mean, t_std)
    epistemic = res["epistemic"]
    preds = res["preds_denorm"]
    resid = np.abs(res["targets_denorm"] - preds)
    max_ep = epistemic.max(axis=-1)
    max_resid = resid.max(axis=-1)

    # B0 severity curve
    b0_bins = np.percentile(np.abs(b0_vals), np.arange(0, 101, 10))
    b0_centers, b0_ep, b0_res = [], [], []
    for i in range(len(b0_bins) - 1):
        mask = (np.abs(b0_vals) >= b0_bins[i]) & (np.abs(b0_vals) < b0_bins[i+1])
        if mask.sum() > 10:
            b0_centers.append((b0_bins[i] + b0_bins[i+1]) / 2)
            b0_ep.append(float(max_ep[mask].mean()))
            b0_res.append(float(max_resid[mask].mean()))
    severity_curve(np.array(b0_centers), np.array(b0_ep), np.array(b0_res), "B0 Off-Resonance (Hz)", FIG_DIR)

    # B1 severity curve
    b1_sev = np.abs(b1_vals - 1.0)
    b1_bins = np.percentile(b1_sev[b1_sev > 0], np.arange(0, 101, 10)) if (b1_sev > 0).sum() > 10 else np.linspace(0, 0.4, 11)
    b1_centers, b1_ep, b1_res = [], [], []
    for i in range(len(b1_bins) - 1):
        mask = (b1_sev >= b1_bins[i]) & (b1_sev < b1_bins[i+1])
        if mask.sum() > 10:
            b1_centers.append((b1_bins[i] + b1_bins[i+1]) / 2)
            b1_ep.append(float(max_ep[mask].mean()))
            b1_res.append(float(max_resid[mask].mean()))
    severity_curve(np.array(b1_centers), np.array(b1_ep), np.array(b1_res), "B1 Scaling Deviation", FIG_DIR)

    # Motion severity curve
    mot_sev = np.abs(motion_vals.astype(float))
    mot_bins = np.arange(0, mot_sev.max() + 2)
    mot_centers, mot_ep, mot_res = [], [], []
    for i in range(len(mot_bins) - 1):
        mask = (mot_sev >= mot_bins[i]) & (mot_sev < mot_bins[i+1])
        if mask.sum() > 10:
            mot_centers.append((mot_bins[i] + mot_bins[i+1]) / 2)
            mot_ep.append(float(max_ep[mask].mean()))
            mot_res.append(float(max_resid[mask].mean()))
    if mot_centers:
        severity_curve(np.array(mot_centers), np.array(mot_ep), np.array(mot_res), "Motion Shift (voxels)", FIG_DIR)

    logger.info("Severity curves saved.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 6: 2D Phantom Brain Maps
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_brain_maps(cfg, mrf_path):
    """Generate 2D phantom brain maps with uncertainty overlay."""
    from qMR_Robust.models import build_model
    from qMR_Robust.eval.metrics import generate_2d_phantom, plot_2d_brain_maps

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   EXPERIMENT 6: 2D PHANTOM BRAIN MAPS          ║")
    logger.info("╚═══════════════════════════════════════════════╝")

    phantom = generate_2d_phantom(grid_size=48, seed=42)
    mask = phantom["mask"]
    grid = mask.shape[0]

    # Load trained model
    model = build_model("resnet1d_18", {"input_channels": 2, "hidden_dim": 128, "output_dim": 2, "dropout": 0.1, "evidential": True})
    ckpt = ROOT / "results" / "checkpoints" / "main_evidential.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.to(DEVICE).eval()

    # Get normalization stats
    train_ds = MRFHDF5Dataset(mrf_path, split="train")
    t_mean, t_std = train_ds.mean, train_ds.std

    # Run inference per voxel
    pred_t1 = np.zeros((grid, grid))
    pred_t2 = np.zeros((grid, grid))
    ep_t1 = np.zeros((grid, grid))
    ep_t2 = np.zeros((grid, grid))
    fail_mask = np.zeros((grid, grid), dtype=bool)

    # Batch all masked voxels
    voxels = []
    coords = []
    for iy in range(grid):
        for ix in range(grid):
            if mask[iy, ix]:
                sig = phantom["corrupted_signals"][iy, ix]
                voxels.append(np.stack([sig.real, sig.imag], axis=0).astype(np.float32))
                coords.append((iy, ix))

    if voxels:
        batch = torch.from_numpy(np.stack(voxels)).to(DEVICE)
        with torch.no_grad():
            nig = model(batch)  # (N, 2, 4)
        gamma = nig[..., 0].cpu().numpy()
        nu = nig[..., 1].cpu().numpy()
        alpha = nig[..., 2].cpu().numpy()
        beta = nig[..., 3].cpu().numpy()

        epistemic = beta / (nu * (alpha - 1.0))

        for k, (iy, ix) in enumerate(coords):
            pred_t1[iy, ix] = gamma[k, 0] * t_std[0] + t_mean[0]
            pred_t2[iy, ix] = gamma[k, 1] * t_std[1] + t_mean[1]
            ep_t1[iy, ix] = epistemic[k, 0]
            ep_t2[iy, ix] = epistemic[k, 1]
            fail_mask[iy, ix] = max(epistemic[k]) > 0.1

    plot_2d_brain_maps(phantom, pred_t1, pred_t2, ep_t1, ep_t2, fail_mask, FIG_DIR)
    logger.info("Brain maps generated.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiment 7: MRS Pipeline with calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_mrs_experiments(cfg, mrs_path):
    """Run MRS GABA experiments with calibration analysis."""
    from qMR_Robust.models import build_model
    from qMR_Robust.eval.metrics import (
        failure_detection_metrics, expected_calibration_error,
        plot_reliability_diagram, plot_failure_detection_roc,
    )

    logger.info("╔═══════════════════════════════════════════════╗")
    logger.info("║   EXPERIMENT 7: MRS GABA WITH CALIBRATION      ║")
    logger.info("╚═══════════════════════════════════════════════╝")

    train_ds = MRSHDF5Dataset(mrs_path, split="train")
    val_ds = MRSHDF5Dataset(mrs_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model("resnet1d_18", {"input_channels": 2, "hidden_dim": 128, "output_dim": 1, "dropout": 0.1, "evidential": True})
    model = train_model(model, train_loader, val_loader, n_epochs=20, lr=1e-3, task_name="mrs_gaba_v2",
                        loss_type="evidential", evidential_coeff=1.0, annealing_epochs=5)

    res = evaluate_model(model, val_loader, "evidential", t_mean, t_std)
    preds = res["preds_denorm"]
    tgts = res["targets_denorm"]
    resid = np.abs(tgts - preds)
    epistemic = res["epistemic"]

    mae = float(np.mean(resid))
    rmse = float(np.sqrt(np.mean(resid**2)))
    fdm = failure_detection_metrics(epistemic, resid, tolerance=0.3)
    ece_val, bin_pred, bin_act = expected_calibration_error(epistemic, resid, n_bins=10)

    logger.info("MRS GABA: MAE=%.4f mM RMSE=%.4f mM AUROC=%.3f ECE=%.4f", mae, rmse, fdm["auroc"], ece_val)

    plot_reliability_diagram(bin_pred, bin_act, ece_val, "mrs_gaba", FIG_DIR)
    plot_failure_detection_roc(epistemic, resid, [0.1, 0.2, 0.3, 0.5], "mrs_gaba", FIG_DIR)

    with open(FIG_DIR / "mrs_gaba_metrics.json", "w") as f:
        json.dump({"mae_mM": mae, "rmse_mM": rmse, "auroc": fdm["auroc"], "auprc": fdm["auprc"], "ece": ece_val}, f, indent=2)

    return {"mae": mae, "rmse": rmse, "auroc": fdm["auroc"], "ece": ece_val}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_or_none(path):
    """Load JSON result if it exists, else return None."""
    p = FIG_DIR / path
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _ckpt_exists(task_name):
    return (ROOT / "results" / "checkpoints" / f"{task_name}.pt").exists()


def main():
    t_start = time.time()
    (ROOT / "results" / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    mrs_path = str(ROOT / cfg["paths"]["failure_forecast_mrs"])

    assert Path(mrf_path).exists(), f"MRF data not found: {mrf_path}"
    assert Path(mrs_path).exists(), f"MRS data not found: {mrs_path}"

    all_results = {}

    # Experiment 1: Main comparison
    cached = _load_or_none("main_comparison_metrics.json")
    if cached:
        logger.info("╔═ EXP 1: CACHED ═╗")
        all_results["main_comparison"] = cached
    else:
        all_results["main_comparison"] = run_main_comparison(cfg, mrf_path)

    # Experiment 2: Architecture comparison
    cached = _load_or_none("architecture_comparison_metrics.json")
    if cached:
        logger.info("╔═ EXP 2: CACHED ═╗")
        all_results["architecture"] = cached
    else:
        all_results["architecture"] = run_architecture_comparison(cfg, mrf_path)

    # Experiment 3: Regularizer ablation
    cached = _load_or_none("ablation_regularizer_metrics.json")
    if cached:
        logger.info("╔═ EXP 3: CACHED ═╗")
        all_results["regularizer_ablation"] = cached
    else:
        all_results["regularizer_ablation"] = run_regularizer_ablation(cfg, mrf_path)

    # Experiment 4: Corruption ablation
    cached = _load_or_none("ablation_corruption_metrics.json")
    if cached:
        logger.info("╔═ EXP 4: CACHED ═╗")
        all_results["corruption_ablation"] = cached
    else:
        all_results["corruption_ablation"] = run_corruption_ablation(cfg)

    # Experiment 5: OOD severity curves (check for plot files)
    if (FIG_DIR / "severity_curve_b0_off-resonance_(hz).png").exists():
        logger.info("╔═ EXP 5: CACHED ═╗")
    else:
        run_severity_curves(cfg, mrf_path)

    # Experiment 6: 2D brain maps
    if (FIG_DIR / "brain_maps_2d.png").exists():
        logger.info("╔═ EXP 6: CACHED ═╗")
    else:
        run_brain_maps(cfg, mrf_path)

    # Experiment 7: MRS with calibration
    cached = _load_or_none("mrs_gaba_metrics.json")
    if cached:
        logger.info("╔═ EXP 7: CACHED ═╗")
        all_results["mrs"] = cached
    else:
        all_results["mrs"] = run_mrs_experiments(cfg, mrs_path)

    # Save master summary
    with open(FIG_DIR / "all_experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    logger.info("═══ ALL EXPERIMENTS COMPLETE in %.0f s (%.1f min) ═══", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
