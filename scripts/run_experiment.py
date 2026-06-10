#!/usr/bin/env python3
"""
run_experiment.py — End-to-end Failure Forecasting pipeline.

Steps:
  1. Generate synthetic MRF + MRS data with entangled corruptions
  2. Train evidential ViT on MRF (T1, T2 prediction)
  3. Train evidential ViT on MRS (GABA concentration prediction)
  4. Evaluate with Forecaster: uncertainty decomposition, failure flagging, plots
  5. Save all metrics + figures for the paper
"""

from __future__ import annotations

import json
import logging
import os
import sys
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
from torch.utils.data import DataLoader, TensorDataset, random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiment")

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Data Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_data(cfg: dict):
    """Generate clean + entangled-corrupted MRF and MRS datasets."""
    from qMR_Robust.simulators import SimulationManager, PhysicsCorruptor

    paths = cfg["paths"]
    mgr = SimulationManager(cfg)
    corruptor = PhysicsCorruptor(cfg)

    # --- MRF ---
    mrf_path = ROOT / paths["failure_forecast_mrf"]
    if not mrf_path.exists():
        logger.info("═══ Generating failure-forecast MRF data ═══")
        t0 = time.time()
        corruptor.generate_failure_forecast_mrf(str(mrf_path), n_signals=50_000)
        logger.info("MRF generation: %.1f s", time.time() - t0)
    else:
        logger.info("MRF data already exists: %s", mrf_path)

    # --- MRS ---
    mrs_path = ROOT / paths["failure_forecast_mrs"]
    if not mrs_path.exists():
        logger.info("═══ Generating failure-forecast MRS data ═══")
        t0 = time.time()
        corruptor.generate_failure_forecast_mrs(str(mrs_path), n_signals=10_000)
        logger.info("MRS generation: %.1f s", time.time() - t0)
    else:
        logger.info("MRS data already exists: %s", mrs_path)

    return str(mrf_path), str(mrs_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Dataset classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MRFHDF5Dataset(torch.utils.data.Dataset):
    """Loads corrupted MRF signals and clean parameters from HDF5."""

    def __init__(self, h5_path: str, split: str = "train", train_ratio: float = 0.8):
        super().__init__()
        hf = h5py.File(h5_path, "r")
        n = hf.attrs["n_signals"]
        n_train = int(n * train_ratio)

        if split == "train":
            self.signals = hf["corrupted_signals"][0:n_train]
            self.params = hf["parameters"][0:n_train]
        else:
            self.signals = hf["corrupted_signals"][n_train:n]
            self.params = hf["parameters"][n_train:n]

        self.signals_real = np.stack([self.signals.real, self.signals.imag], axis=1).astype(np.float32)
        self.params = self.params[:, :2].astype(np.float32)  # T1, T2

        # Normalize targets to zero-mean unit-variance for stable evidential training
        if split == "train":
            self.target_mean = self.params.mean(axis=0)
            self.target_std = self.params.std(axis=0) + 1e-8
        else:
            # Will be set externally via set_norm()
            self.target_mean = np.zeros(2, dtype=np.float32)
            self.target_std = np.ones(2, dtype=np.float32)

    def set_norm(self, mean: np.ndarray, std: np.ndarray):
        self.target_mean = mean
        self.target_std = std

    def __len__(self):
        return len(self.signals_real)

    def __getitem__(self, idx):
        sig = torch.from_numpy(self.signals_real[idx])
        tgt = (torch.from_numpy(self.params[idx]) - self.target_mean) / self.target_std
        return sig, tgt


class MRSHDF5Dataset(torch.utils.data.Dataset):
    """Loads corrupted MRS spectra and GABA concentration from HDF5."""

    def __init__(self, h5_path: str, split: str = "train", train_ratio: float = 0.8):
        super().__init__()
        hf = h5py.File(h5_path, "r")
        n = hf.attrs["n_signals"]
        n_train = int(n * train_ratio)

        if split == "train":
            self.spectra = hf["corrupted_spectra"][0:n_train]
            self.conc = hf["concentrations"][0:n_train]
        else:
            self.spectra = hf["corrupted_spectra"][n_train:n]
            self.conc = hf["concentrations"][n_train:n]

        self.spectra_real = np.stack([self.spectra.real, self.spectra.imag], axis=1).astype(np.float32)
        # GABA is index 3 in the metabolite list
        self.gaba = self.conc[:, 3:4].astype(np.float32)

    def __len__(self):
        return len(self.spectra_real)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.spectra_real[idx]),
            torch.from_numpy(self.gaba[idx]),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Training loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_evidential(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    task_name: str,
    n_epochs: int = 30,
    lr: float = 3e-4,
    evidential_coeff: float = 1.0,
    annealing_epochs: int = 10,
) -> dict:
    """Train an evidential model and return training history."""
    from qMR_Robust.models.losses import evidential_regression_loss

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": [], "train_nll": [], "train_reg": []}

    best_val = float("inf")
    best_state = None

    logger.info("═══ Training %s (%d epochs) ═══", task_name, n_epochs)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, epoch_nll, epoch_reg, n_batches = 0.0, 0.0, 0.0, 0

        for signals, targets in train_loader:
            signals, targets = signals.to(DEVICE), targets.to(DEVICE)
            nig = model(signals)
            result = evidential_regression_loss(
                targets, nig,
                coeff=evidential_coeff,
                epoch=epoch,
                annealing_epochs=annealing_epochs,
            )

            optimizer.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += result["loss"].item()
            epoch_nll += result["nll"].item()
            epoch_reg += result["reg"].item()
            n_batches += 1

        scheduler.step()

        train_loss = epoch_loss / n_batches
        history["train_loss"].append(train_loss)
        history["train_nll"].append(epoch_nll / n_batches)
        history["train_reg"].append(epoch_reg / n_batches)

        # Validation
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for signals, targets in val_loader:
                signals, targets = signals.to(DEVICE), targets.to(DEVICE)
                nig = model(signals)
                result = evidential_regression_loss(targets, nig, coeff=evidential_coeff, epoch=epoch, annealing_epochs=annealing_epochs)
                val_loss_sum += result["loss"].item() * signals.size(0)
                val_n += signals.size(0)
        val_loss = val_loss_sum / val_n
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "  Epoch %3d/%d | train_loss=%.4f nll=%.4f reg=%.4f | val_loss=%.4f",
                epoch + 1, n_epochs, train_loss, epoch_nll / n_batches, epoch_reg / n_batches, val_loss,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    # Save checkpoint
    ckpt_dir = ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{task_name}_best.pt")
    logger.info("Checkpoint saved → %s", ckpt_dir / f"{task_name}_best.pt")

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title(f"{task_name} — Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_nll"], label="NLL", color="tab:blue")
    axes[1].set_title(f"{task_name} — NLL Component")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(history["train_reg"], label="Reg", color="tab:orange")
    axes[2].set_title(f"{task_name} — Evidential Regularizer")
    axes[2].set_xlabel("Epoch")

    fig.tight_layout()
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{task_name}_training_curves.png", dpi=150)
    plt.close(fig)

    return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_mrf_pipeline(cfg: dict, mrf_path: str):
    """Train evidential ResNet on MRF → T1, T2 prediction."""
    from qMR_Robust.models import build_model
    from qMR_Robust.eval.forecaster import Forecaster

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║   MRF FAILURE FORECASTING PIPELINE   ║")
    logger.info("╚══════════════════════════════════════╝")

    train_ds = MRFHDF5Dataset(mrf_path, split="train")
    val_ds = MRFHDF5Dataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.target_mean, train_ds.target_std)
    logger.info("MRF train=%d  val=%d | target_mean=%s target_std=%s",
                len(train_ds), len(val_ds), train_ds.target_mean, train_ds.target_std)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model_cfg = {
        "architecture": "resnet1d_18",
        "input_channels": 2,
        "hidden_dim": 128,
        "output_dim": 2,
        "dropout": 0.1,
        "evidential": True,
    }
    model = build_model("resnet1d_18", model_cfg)

    history = train_evidential(
        model, train_loader, val_loader, cfg,
        task_name="mrf_t1t2",
        n_epochs=10,
        lr=5e-4,
        evidential_coeff=cfg["training"]["evidential_coeff"],
        annealing_epochs=5,
    )

    # Evaluate — we need to denormalize predictions back to original scale
    forecaster = Forecaster(
        model, device=DEVICE,
        epistemic_threshold=cfg["evaluation"]["epistemic_threshold"],
        output_dir=str(ROOT / "results" / "figures"),
    )
    raw_results = forecaster.evaluate(val_loader, split_name="mrf_test")

    # Denormalize predictions and targets for reporting
    t_mean = train_ds.target_mean
    t_std = raw_results["gamma"] * 0 + train_ds.target_std  # broadcast
    gamma_denorm = raw_results["gamma"] * train_ds.target_std + train_ds.target_mean
    targets_denorm = val_ds.params  # raw unnormalized targets from dataset

    # Recompute residuals in original scale
    residuals_denorm = np.abs(targets_denorm - gamma_denorm)
    mae_orig = float(np.mean(residuals_denorm))
    rmse_orig = float(np.sqrt(np.mean(residuals_denorm ** 2)))

    raw_results["metrics"]["mae_original_scale"] = mae_orig
    raw_results["metrics"]["rmse_original_scale"] = rmse_orig
    raw_results["gamma_denorm"] = gamma_denorm
    raw_results["residuals_denorm"] = residuals_denorm

    logger.info("MRF Results (normalized): MAE=%.4f RMSE=%.4f", raw_results["metrics"]["mae"], raw_results["metrics"]["rmse"])
    logger.info("MRF Results (original ms): MAE=%.1f RMSE=%.1f | Failure Rate=%.2f%%",
                mae_orig, rmse_orig, raw_results["metrics"]["failure_rate"] * 100)

    return raw_results


def run_mrs_pipeline(cfg: dict, mrs_path: str):
    """Train evidential ResNet on MRS → GABA concentration prediction."""
    from qMR_Robust.models import build_model
    from qMR_Robust.eval.forecaster import Forecaster

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║   MRS FAILURE FORECASTING PIPELINE   ║")
    logger.info("╚══════════════════════════════════════╝")

    train_ds = MRSHDF5Dataset(mrs_path, split="train")
    val_ds = MRSHDF5Dataset(mrs_path, split="val")
    logger.info("MRS train=%d  val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    model_cfg = {
        "architecture": "resnet1d_18",
        "input_channels": 2,
        "hidden_dim": 128,
        "output_dim": 1,
        "dropout": 0.1,
        "evidential": True,
    }
    model = build_model("resnet1d_18", model_cfg)

    history = train_evidential(
        model, train_loader, val_loader, cfg,
        task_name="mrs_gaba",
        n_epochs=10,
        lr=5e-4,
        evidential_coeff=cfg["training"]["evidential_coeff"],
        annealing_epochs=5,
    )

    forecaster = Forecaster(
        model, device=DEVICE,
        epistemic_threshold=cfg["evaluation"]["epistemic_threshold"],
        output_dir=str(ROOT / "results" / "figures"),
    )
    results = forecaster.evaluate(val_loader, split_name="mrs_test")

    logger.info("MRS Results: MAE=%.4f RMSE=%.4f | Failure Rate=%.2f%%",
                results["metrics"]["mae"], results["metrics"]["rmse"],
                results["metrics"]["failure_rate"] * 100)

    return results


def save_summary(mrf_results: dict, mrs_results: dict):
    """Save combined results summary for the paper."""
    fig_dir = ROOT / "results" / "figures"

    summary = {
        "mrf": mrf_results["metrics"],
        "mrs": mrs_results["metrics"],
    }
    with open(fig_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Combined failure-rate comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MRF epistemic vs residual (use denormalized residuals for meaningful units)
    ep_mrf = mrf_results["epistemic"].reshape(-1)
    res_mrf = mrf_results.get("residuals_denorm", mrf_results["residuals"]).reshape(-1)
    axes[0].scatter(ep_mrf, res_mrf, alpha=0.2, s=6, c="steelblue", edgecolors="none")
    axes[0].axvline(summary["mrf"]["epistemic_threshold"], color="red", ls="--", lw=1.5)
    axes[0].set_xlabel("Epistemic Uncertainty")
    axes[0].set_ylabel("Absolute Residual")
    axes[0].set_title(f"MRF (T1, T2) — Failure Rate {summary['mrf']['failure_rate']*100:.1f}% | MAE={summary['mrf'].get('mae_original_scale', summary['mrf']['mae']):.1f} ms")
    if np.std(ep_mrf) > 1e-12 and np.std(res_mrf) > 1e-12:
        r = np.corrcoef(ep_mrf, res_mrf)[0, 1]
        axes[0].annotate(f"r = {r:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)

    # MRS epistemic vs residual
    ep_mrs = mrs_results["epistemic"].reshape(-1)
    res_mrs = mrs_results["residuals"].reshape(-1)
    axes[1].scatter(ep_mrs, res_mrs, alpha=0.3, s=8, c="coral", edgecolors="none")
    axes[1].axvline(summary["mrs"]["epistemic_threshold"], color="red", ls="--", lw=1.5)
    axes[1].set_xlabel("Epistemic Uncertainty")
    axes[1].set_ylabel("Absolute Residual")
    axes[1].set_title(f"MRS (GABA) — Failure Rate {summary['mrs']['failure_rate']*100:.1f}%")
    if np.std(ep_mrs) > 1e-12 and np.std(res_mrs) > 1e-12:
        r = np.corrcoef(ep_mrs, res_mrs)[0, 1]
        axes[1].annotate(f"r = {r:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=12)

    fig.tight_layout()
    fig.savefig(fig_dir / "combined_epistemic_vs_residual.png", dpi=200)
    plt.close(fig)

    # Summary table as bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tasks = ["MRF (T1,T2)", "MRS (GABA)"]
    mae_vals = [summary["mrf"].get("mae_original_scale", summary["mrf"]["mae"]), summary["mrs"]["mae"]]
    rmse_vals = [summary["mrf"].get("rmse_original_scale", summary["mrf"]["rmse"]), summary["mrs"]["rmse"]]
    fail_vals = [summary["mrf"]["failure_rate"] * 100, summary["mrs"]["failure_rate"] * 100]
    colors = ["steelblue", "coral"]

    for ax, vals, title, ylabel in zip(
        axes,
        [mae_vals, rmse_vals, fail_vals],
        ["MAE", "RMSE", "Failure Rate (%)"],
        ["MAE", "RMSE", "% Flagged"],
    ):
        bars = ax.bar(tasks, vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                    f"{v:.4f}" if v < 1 else f"{v:.1f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(fig_dir / "summary_metrics.png", dpi=200)
    plt.close(fig)

    logger.info("Summary saved → %s", fig_dir / "experiment_summary.json")
    return summary


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t_start = time.time()

    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Step 1: Generate data
    mrf_path, mrs_path = generate_data(cfg)

    # Step 2-3: Train & evaluate MRF
    mrf_results = run_mrf_pipeline(cfg, mrf_path)

    # Step 4-5: Train & evaluate MRS
    mrs_results = run_mrs_pipeline(cfg, mrs_path)

    # Step 6: Save combined summary + plots
    summary = save_summary(mrf_results, mrs_results)

    elapsed = time.time() - t_start
    logger.info("═══ FULL PIPELINE COMPLETE in %.0f s ═══", elapsed)
    logger.info("MRF  — MAE=%.4f  RMSE=%.4f  FailureRate=%.2f%%",
                summary["mrf"]["mae"], summary["mrf"]["rmse"], summary["mrf"]["failure_rate"] * 100)
    logger.info("MRS  — MAE=%.4f  RMSE=%.4f  FailureRate=%.2f%%",
                summary["mrs"]["mae"], summary["mrs"]["rmse"], summary["mrs"]["failure_rate"] * 100)

    return summary


if __name__ == "__main__":
    main()
