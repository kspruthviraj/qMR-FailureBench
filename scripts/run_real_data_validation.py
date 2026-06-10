#!/usr/bin/env python3
"""
run_real_data_validation.py — ZERO-SHOT real in-vivo validation.

Loads real brain qMRI data from qMRLab public datasets (VFA T1 mapping,
B0 field map, B1 map) and runs our pre-trained model WITHOUT any
fine-tuning or PhysicsCorruptor injection.

The goal: show that epistemic uncertainty naturally highlights:
- Air-tissue interfaces (where B0 artifacts are worst)
- Regions with known B1 inhomogeneity
- Areas where the model has never seen real scanner noise

This is the experiment that silences the "synthetic data only" critique.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import nibabel as nib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("real")

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "real" / "qmrlab"


def load_nifti(path):
    """Load a NIfTI file and return the data array."""
    img = nib.load(str(path))
    return img.get_fdata()


def run_real_validation():
    """Run zero-shot validation on real in-vivo qMRLab data."""
    import h5py
    import yaml

    logger.info("=" * 60)
    logger.info("ZERO-SHOT REAL IN-VIVO VALIDATION")
    logger.info("=" * 60)

    # ── Load real data ──
    logger.info("Loading real in-vivo qMRI data...")

    # VFA data: multi-flip-angle brain images
    vfa_data = load_nifti(DATA_DIR / "vfa_t1_data" / "VFAData.nii.gz")
    logger.info("  VFA data shape: %s", vfa_data.shape)

    # Ground truth T1 map (from qMRLab fitting)
    t1_gt = load_nifti(DATA_DIR / "vfa_t1_data" / "FitResults" / "T1.nii.gz")
    logger.info("  T1 ground truth shape: %s", t1_gt.shape)

    # Brain mask
    mask = load_nifti(DATA_DIR / "vfa_t1_data" / "Mask.nii.gz")
    logger.info("  Mask shape: %s, nonzero voxels: %d", mask.shape, int(mask.sum()))

    # B1 map (transmit field)
    b1_map = load_nifti(DATA_DIR / "vfa_t1_data" / "B1map.nii.gz")
    logger.info("  B1 map shape: %s", b1_map.shape)

    # B0 field map
    b0_fmap = load_nifti(DATA_DIR / "b0_map_data" / "FitResults" / "B0map.nii.gz")
    logger.info("  B0 field map shape: %s", b0_fmap.shape)

    # ── Extract 2D data ──
    # VFA data is (x, y, 1, n_flip_angles) — squeeze the slice dimension
    if vfa_data.ndim == 4:
        vfa_slice = vfa_data[:, :, 0, :]  # (x, y, n_fa)
    elif vfa_data.ndim == 3:
        vfa_slice = vfa_data
    else:
        vfa_slice = vfa_data

    t1_slice = t1_gt if t1_gt.ndim == 2 else t1_gt[:, :, 0]
    mask_slice = mask if mask.ndim == 2 else mask[:, :, 0]
    b1_slice = b1_map if b1_map.ndim == 2 else b1_map[:, :, 0]
    b0_slice = b0_fmap if b0_fmap.ndim == 2 else b0_fmap[:, :, 0]

    logger.info("  Slice shapes: VFA=%s, T1=%s, Mask=%s", vfa_slice.shape, t1_slice.shape, mask_slice.shape)

    # ── Load our pre-trained model ──
    from qMR_Robust.models.resnet1d import ResNet1D

    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)

    # Find best checkpoint
    ckpt_dir = ROOT / "results" / "checkpoints"
    candidates = [
        "abl_NLL_ER.pt",
        "v3_phys_NLL_ER.pt",
        "v3_phys_NLL_ER_Monotonicity.pt",
        "main_evidential.pt",
    ]
    ckpt = None
    for c in candidates:
        p = ckpt_dir / c
        if p.exists():
            ckpt = p
            break
    if ckpt is None:
        logger.error("No checkpoint found!")
        return

    logger.info("Loading model: %s", ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    # Get normalization stats from training data
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    hf = h5py.File(mrf_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)
    t_mean = hf["parameters"][:n_train, :2].astype(np.float32).mean(axis=0)
    t_std = hf["parameters"][:n_train, :2].astype(np.float32).std(axis=0) + 1e-8
    hf.close()

    # ── Prepare voxels for inference ──
    # The VFA data has n_flip_angles per voxel.
    # Our model expects 1000-timepoint complex signals.
    # Strategy: convert VFA multi-angle data into a 2-channel signal
    # (real=signal magnitude, imag=angle-dependent phase)
    # and tile to match the expected input length.

    n_fa = vfa_slice.shape[-1]
    target_len = 1000

    voxels = []
    coords = []
    t1_values = []

    for ix in range(vfa_slice.shape[0]):
        for iy in range(vfa_slice.shape[1]):
            if mask_slice[ix, iy] < 0.5:
                continue

            # Extract VFA signal for this voxel
            sig = vfa_slice[ix, iy, :]  # (n_fa,)

            # Normalize signal
            sig_max = np.abs(sig).max()
            if sig_max < 1e-6:
                continue
            sig_norm = sig / sig_max

            # Create 2-channel representation:
            # Channel 1: signal magnitude (repeated to fill target_len)
            # Channel 2: phase progression across flip angles
            tiled_mag = np.tile(sig_norm.real, target_len // n_fa + 1)[:target_len]
            phase = np.linspace(0, 2 * np.pi, n_fa)
            tiled_phase = np.tile(np.sin(phase), target_len // n_fa + 1)[:target_len]

            sig_2ch = np.stack([tiled_mag, tiled_phase], axis=0).astype(np.float32)
            voxels.append(sig_2ch)
            coords.append((ix, iy))
            t1_values.append(t1_slice[ix, iy])

    if not voxels:
        logger.error("No valid voxels found!")
        return

    logger.info("Running inference on %d real brain voxels...", len(voxels))

    # Batch inference
    batch = torch.from_numpy(np.stack(voxels)).to(DEVICE)
    batch_size = 512
    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(batch), batch_size):
            chunk = batch[i:i + batch_size]
            raw = model(chunk)
            B, D = raw.shape[0], 2
            raw = raw.view(B, D, 4)
            all_gamma.append(raw[..., 0].cpu().numpy())
            all_nu.append(raw[..., 1].cpu().numpy())
            all_alpha.append(raw[..., 2].cpu().numpy())
            all_beta.append(raw[..., 3].cpu().numpy())

    gamma = np.concatenate(all_gamma)
    nu = np.concatenate(all_nu)
    alpha = np.concatenate(all_alpha)
    beta = np.concatenate(all_beta)

    # Compute uncertainties
    epistemic = beta / (nu * (alpha - 1.0))
    aleatoric = beta / (alpha - 1.0)

    # Denormalize predictions
    pred_t1 = gamma[:, 0] * t_std[0] + t_mean[0]

    t1_gt_arr = np.array(t1_values)

    # ── Build 2D maps ──
    nx, ny = vfa_slice.shape[:2]
    pred_t1_map = np.zeros((nx, ny))
    ep_t1_map = np.zeros((nx, ny))
    alea_t1_map = np.zeros((nx, ny))
    gt_t1_map = t1_slice.copy()

    for k, (ix, iy) in enumerate(coords):
        pred_t1_map[ix, iy] = pred_t1[k]
        ep_t1_map[ix, iy] = epistemic[k, 0]
        alea_t1_map[ix, iy] = aleatoric[k, 0]

    # ── Compute metrics ──
    valid = np.array(t1_values) > 0
    mae = float(np.abs(pred_t1[valid] - t1_gt_arr[valid]).mean())
    max_ep = epistemic[valid].max(axis=-1)
    resid = np.abs(pred_t1[valid] - t1_gt_arr[valid])
    r = float(np.corrcoef(max_ep, resid)[0, 1]) if max_ep.std() > 1e-12 else 0.0

    # B0 correlation: does uncertainty correlate with B0 field inhomogeneity?
    b0_at_coords = np.array([b0_slice[ix, iy] if ix < b0_slice.shape[0] and iy < b0_slice.shape[1] else 0
                              for ix, iy in coords])
    b0_valid = np.abs(b0_at_coords[valid]) > 0
    r_b0 = float(np.corrcoef(max_ep[b0_valid], np.abs(b0_at_coords[valid][b0_valid]))[0, 1]) if b0_valid.sum() > 10 else 0.0

    logger.info("  MAE: %.1f ms", mae)
    logger.info("  Epistemic-error correlation: r=%.3f", r)
    logger.info("  Epistemic-B0 correlation: r=%.3f", r_b0)

    # ── Generate Figure ──
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.3)

    # Row 1: Input data
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(np.rot90(gt_t1_map), cmap="hot", vmin=0, vmax=2500)
    ax1.set_title("Ground Truth T₁ (ms)\n[qMRLab in-vivo]", fontsize=11)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(np.rot90(pred_t1_map), cmap="hot", vmin=0, vmax=2500)
    ax2.set_title("Predicted T₁ (ms)\n[Zero-shot, no fine-tuning]", fontsize=11)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 2])
    err_map = np.where(mask_slice > 0.5, np.abs(gt_t1_map - pred_t1_map), np.nan)
    im3 = ax3.imshow(np.rot90(err_map), cmap="YlOrRd", vmin=0, vmax=500)
    ax3.set_title("|T₁ Error| (ms)", fontsize=11)
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(np.rot90(b1_slice), cmap="RdBu_r", vmin=0.5, vmax=1.5)
    ax4.set_title("B₁ Map (real in-vivo)\n[Shows transmit inhomogeneity]", fontsize=11)
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Row 2: Uncertainty and analysis
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(np.rot90(ep_t1_map), cmap="viridis")
    ax5.set_title("Epistemic Uncertainty\n[Model ignorance]", fontsize=11)
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(np.rot90(alea_t1_map), cmap="magma")
    ax6.set_title("Aleatoric Uncertainty\n[Data noise]", fontsize=11)
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    # Scatter: epistemic vs error
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(max_ep, resid, alpha=0.2, s=5, c="steelblue")
    ax7.set_xlabel("Epistemic Uncertainty")
    ax7.set_ylabel("|T₁ Error| (ms)")
    ax7.set_title(f"Epistemic vs Error (r={r:.3f})", fontsize=11)

    # Scatter: epistemic vs B0
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.scatter(np.abs(b0_at_coords[valid]), max_ep, alpha=0.2, s=5, c="coral")
    ax8.set_xlabel("|B₀ Field Inhomogeneity|")
    ax8.set_ylabel("Epistemic Uncertainty")
    ax8.set_title(f"Epistemic vs B₀ (r={r_b0:.3f})", fontsize=11)

    fig.suptitle(
        "Zero-Shot Real In-Vivo Validation: qMRLab VFA T₁ Mapping\n"
        "Model trained on synthetic data only — tested on real brain MRI without fine-tuning",
        fontsize=14, fontweight="bold"
    )
    fig.savefig(FIG_DIR / "real_data_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Save results ──
    results = {
        "dataset": "qMRLab VFA T1 (real in-vivo)",
        "n_voxels": int(len(voxels)),
        "mae_ms": mae,
        "epistemic_error_correlation": r,
        "epistemic_b0_correlation": r_b0,
        "model_checkpoint": str(ckpt),
        "zero_shot": True,
    }
    with open(FIG_DIR / "real_data_validation.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("REAL DATA VALIDATION COMPLETE")
    logger.info("  MAE: %.1f ms", mae)
    logger.info("  Epistemic-error r: %.3f", r)
    logger.info("  Epistemic-B0 r: %.3f", r_b0)
    logger.info("  Figure: %s", FIG_DIR / "real_data_validation.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    t0 = time.time()
    run_real_validation()
    logger.info("Total time: %.0f s", time.time() - t0)
