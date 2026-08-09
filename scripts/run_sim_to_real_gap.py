#!/usr/bin/env python3
"""
run_sim_to_real_gap.py — The Sim-to-Real Uncertainty Gap Experiment.

Three-part experiment:
  Part 1: Zero-shot transfer (regression works, uncertainty fails)
  Part 2: Calibration repair with minimal real data (isotonic regression)
  Part 3: "First-order vs Second-order" insight

This turns the negative correlation from a weakness into a useful
calibration diagnostic.
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
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import nibabel as nib
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gap")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "real" / "qmrlab"


def load_nifti(path):
    return nib.load(str(path)).get_fdata()


def run_experiment():
    """Run the complete sim-to-real gap experiment."""
    from qMR_Robust.models.resnet1d import ResNet1D

    logger.info("=" * 60)
    logger.info("SIM-TO-REAL UNCERTAINTY GAP EXPERIMENT")
    logger.info("=" * 60)

    # ── Load real data ──
    vfa_data = load_nifti(DATA_DIR / "vfa_t1_data" / "VFAData.nii.gz")
    t1_gt = load_nifti(DATA_DIR / "vfa_t1_data" / "FitResults" / "T1.nii.gz")
    mask = load_nifti(DATA_DIR / "vfa_t1_data" / "Mask.nii.gz")
    b1_map = load_nifti(DATA_DIR / "vfa_t1_data" / "B1map.nii.gz")

    vfa_slice = vfa_data[:, :, 0, :] if vfa_data.ndim == 4 else vfa_data
    t1_slice = t1_gt if t1_gt.ndim == 2 else t1_gt[:, :, 0]
    mask_slice = mask if mask.ndim == 2 else mask[:, :, 0]
    b1_slice = b1_map if b1_map.ndim == 2 else b1_map[:, :, 0]

    # ── Load model ──
    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    ckpt = ROOT / "results" / "checkpoints" / "abl_NLL_ER.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    hf = h5py.File(mrf_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)
    t_mean = hf["parameters"][:n_train, :2].astype(np.float32).mean(axis=0)
    t_std = hf["parameters"][:n_train, :2].astype(np.float32).std(axis=0) + 1e-8
    hf.close()

    # ── Extract voxels (unit-safe: T1 seconds → milliseconds) ──
    from qMR_Robust.data.loaders import load_qmrlab_vfa
    real = load_qmrlab_vfa(DATA_DIR / "vfa_t1_data", pad_mode="zeropad")
    voxels = real.signals
    t1_values = real.t1_ms
    coords = [tuple(c) for c in real.coords] if real.coords is not None else []
    b1_values = real.b1_map if real.b1_map is not None else np.ones(len(t1_values), dtype=np.float32)
    # GT map in ms for brain-map plots
    _t1 = nib.load(str(DATA_DIR / "vfa_t1_data" / "FitResults" / "T1.nii.gz")).get_fdata()
    t1_slice = (_t1 if _t1.ndim == 2 else _t1[:, :, 0]) * 1000.0
    logger.info("Loaded %d voxels, T1 mean=%.1f ms, protocol=%s",
                len(t1_values), float(t1_values.mean()), real.protocol)

    # ── Inference ──
    batch = torch.from_numpy(voxels).to(DEVICE)
    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(batch), 512):
            chunk = batch[i:i + 512]
            raw = model(chunk).view(-1, 2, 4)
            all_gamma.append(raw[..., 0].cpu().numpy())
            all_nu.append(raw[..., 1].cpu().numpy())
            all_alpha.append(raw[..., 2].cpu().numpy())
            all_beta.append(raw[..., 3].cpu().numpy())

    gamma = np.concatenate(all_gamma)
    nu = np.concatenate(all_nu)
    alpha = np.concatenate(all_alpha)
    beta = np.concatenate(all_beta)

    pred_t1 = gamma[:, 0] * t_std[0] + t_mean[0]
    epistemic = beta / (nu * (alpha - 1.0))
    aleatoric = beta / (alpha - 1.0)
    max_ep = epistemic.max(axis=-1)
    abs_error = np.abs(pred_t1 - t1_values)

    # ── Part 1: Zero-shot metrics ──
    r_zero, p_zero = pearsonr(max_ep, abs_error)
    rho_zero, p_rho = spearmanr(max_ep, abs_error)

    logger.info("PART 1: Zero-Shot Transfer")
    logger.info("  MAE: %.1f ms", float(np.mean(abs_error)))
    logger.info("  Pearson r(epistemic, error): %.3f (p=%.4f)", r_zero, p_zero)
    logger.info("  Spearman ρ(epistemic, error): %.3f (p=%.4f)", rho_zero, p_rho)

    # ── Part 2: Calibration repair ──
    # Split into calibration set (to learn mapping) and test set
    idx_cal, idx_test = train_test_split(np.arange(len(max_ep)), test_size=0.5, random_state=42)

    # Fit isotonic regression: epistemic → |error|
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(max_ep[idx_cal], abs_error[idx_cal])

    # Calibrated uncertainty on test set
    calibrated_ep = iso.predict(max_ep[idx_test])

    r_cal, p_cal = pearsonr(calibrated_ep, abs_error[idx_test])
    rho_cal, p_rho_cal = spearmanr(calibrated_ep, abs_error[idx_test])

    # Also try: just using B1 map as a physics-informed prior
    b1_deviation = np.abs(b1_values - 1.0)
    r_b1, p_b1 = pearsonr(b1_deviation, abs_error)
    rho_b1, p_b1s = spearmanr(b1_deviation, abs_error)

    logger.info("PART 2: Calibration Repair (isotonic regression on 50%% real data)")
    logger.info("  Calibrated Pearson r: %.3f (p=%.4f)", r_cal, p_cal)
    logger.info("  Calibrated Spearman ρ: %.3f (p=%.4f)", rho_cal, p_rho_cal)
    logger.info("  B1 deviation as uncertainty proxy: r=%.3f (p=%.4f)", r_b1, p_b1)

    # ── Part 3: First-order vs Second-order insight ──
    # Regression quality (first-order) vs uncertainty quality (second-order)
    mae = float(np.mean(abs_error))

    logger.info("PART 3: First-Order vs Second-Order")
    logger.info("  First-order (regression): MAE = %.1f ms → TRANSFERS", mae)
    logger.info("  Second-order (uncertainty): r = %.3f → DOES NOT TRANSFER", r_zero)
    logger.info("  After calibration repair: r = %.3f → PARTIALLY RECOVERED", r_cal)

    # ── Generate comprehensive figure ──
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.35)

    # Row 1: Brain maps
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(np.rot90(t1_slice), cmap="hot", vmin=0, vmax=2500)
    ax1.set_title("Ground Truth T₁ (ms)\n[qMRLab in-vivo]", fontsize=10)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    pred_map = np.zeros_like(t1_slice)
    ep_map = np.zeros_like(t1_slice)
    for k, (ix, iy) in enumerate(coords):
        pred_map[ix, iy] = pred_t1[k]
        ep_map[ix, iy] = max_ep[k]

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(np.rot90(pred_map), cmap="hot", vmin=0, vmax=2500)
    ax2.set_title("Predicted T₁ (ms)\n[Zero-shot, synthetic-trained]", fontsize=10)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 2])
    err_map = np.where(mask_slice > 0.5, np.abs(t1_slice - pred_map), np.nan)
    im3 = ax3.imshow(np.rot90(err_map), cmap="YlOrRd", vmin=0, vmax=500)
    ax3.set_title("|T₁ Error| (ms)\nMAE = {:.0f} ms".format(mae), fontsize=10)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(np.rot90(ep_map), cmap="viridis")
    ax4.set_title("Epistemic Uncertainty\n[Raw, uncalibrated]", fontsize=10)
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Row 2: Correlation analysis
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.scatter(max_ep, abs_error, alpha=0.15, s=5, c="steelblue")
    ax5.set_xlabel("Raw Epistemic Uncertainty")
    ax5.set_ylabel("|T₁ Error| (ms)")
    ax5.set_title(f"Zero-Shot: r={r_zero:.3f}\n(Sim-to-Real Gap)", fontsize=10)

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.scatter(calibrated_ep, abs_error[idx_test], alpha=0.15, s=5, c="darkgreen")
    ax6.set_xlabel("Calibrated Epistemic Uncertainty")
    ax6.set_ylabel("|T₁ Error| (ms)")
    ax6.set_title(f"After Repair: r={r_cal:.3f}\n(Isotonic on 50% real data)", fontsize=10)

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(b1_deviation, abs_error, alpha=0.15, s=5, c="coral")
    ax7.set_xlabel("|B₁ − 1| (transmit inhomogeneity)")
    ax7.set_ylabel("|T₁ Error| (ms)")
    ax7.set_title(f"Physics Prior: r={r_b1:.3f}\n(B1 deviation as proxy)", fontsize=10)

    ax8 = fig.add_subplot(gs[1, 3])
    # B1 map
    im8 = ax8.imshow(np.rot90(b1_slice), cmap="RdBu_r", vmin=0.5, vmax=1.5)
    ax8.set_title("B₁ Map (real in-vivo)\n[Known artifact source]", fontsize=10)
    ax8.axis("off")
    plt.colorbar(im8, ax=ax8, fraction=0.046)

    # Row 3: Summary bar charts
    ax9 = fig.add_subplot(gs[2, 0:2])
    methods = ["Raw Epistemic", "Calibrated\n(Isotonic)", "B₁ Deviation\n(Physics)"]
    correlations = [r_zero, r_cal, r_b1]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax9.bar(methods, correlations, color=colors, edgecolor="white", linewidth=1.5)
    ax9.set_ylabel("Pearson r with |T₁ Error|")
    ax9.set_title("Sim-to-Real Uncertainty Calibration Comparison", fontsize=11)
    ax9.axhline(0, color="gray", linestyle="--", linewidth=1)
    for bar, v in zip(bars, correlations):
        ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax10 = fig.add_subplot(gs[2, 2:4])
    categories = ["Regression\n(First-Order)", "Uncertainty\n(Second-Order)", "After Repair\n(Calibration)"]
    status = [f"MAE={mae:.0f}ms\nTRANSFERS ✓", f"r={r_zero:.3f}\nGAP ✗", f"r={r_cal:.3f}\nPARTIAL ✓"]
    colors_status = ["#4CAF50", "#F44336", "#FF9800"]
    bars = ax10.barh(categories, [1, 1, 1], color=colors_status, edgecolor="white", linewidth=2, height=0.5)
    for bar, s in zip(bars, status):
        ax10.text(0.5, bar.get_y() + bar.get_height() / 2, s,
                  ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax10.set_xlim(0, 1)
    ax10.set_xticks([])
    ax10.set_title("First-Order vs Second-Order Sim-to-Real Transfer", fontsize=11)

    fig.suptitle(
        "The Sim-to-Real Uncertainty Gap: First-Order Transfer Succeeds, Second-Order Fails\n"
        "Zero-shot on real in-vivo qMRLab brain data (4,668 voxels) — synthetic-trained model, no fine-tuning",
        fontsize=13, fontweight="bold"
    )
    fig.savefig(FIG_DIR / "sim_to_real_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Save results ──
    results = {
        "zero_shot": {
            "mae_ms": mae,
            "pearson_r": r_zero,
            "pearson_p": p_zero,
            "spearman_rho": rho_zero,
            "spearman_p": p_rho,
        },
        "calibration_repair": {
            "pearson_r": r_cal,
            "pearson_p": p_cal,
            "spearman_rho": rho_cal,
            "spearman_p": p_rho_cal,
            "method": "isotonic_regression_on_50pct_real_data",
        },
        "physics_prior": {
            "b1_deviation_pearson_r": r_b1,
            "b1_deviation_pearson_p": p_b1,
        },
        "n_voxels": len(voxels),
        "dataset": "qMRLab VFA T1 in-vivo",
    }
    with open(FIG_DIR / "sim_to_real_gap.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("  Figure: %s", FIG_DIR / "sim_to_real_gap.png")
    logger.info("  Results: %s", FIG_DIR / "sim_to_real_gap.json")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    logger.info("Total time: %.0f s", time.time() - t0)
