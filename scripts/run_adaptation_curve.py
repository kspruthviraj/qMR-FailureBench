#!/usr/bin/env python3
import sys
"""
run_adaptation_curve.py — Calibration repair scaling law.

Shows how Spearman ρ improves as we increase the % of real data
used for calibration repair (0%, 10%, 20%, 30%, 40%, 50%).

This transforms "there is a problem" into "here is the scaling law."
"""
import json, logging
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel as nib
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("adapt")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
DATA_DIR = ROOT / "data" / "real" / "qmrlab"


def run():
    from qMR_Robust.models.resnet1d import ResNet1D

    logger.info("Adaptation curve experiment")

    # Load real data (unit-safe: T1 in ms, zero-pad VFA protocol)
    from qMR_Robust.data.loaders import load_qmrlab_vfa
    real = load_qmrlab_vfa(DATA_DIR / "vfa_t1_data", pad_mode="zeropad")
    voxels = real.signals
    t1_values = real.t1_ms
    b1_values = real.b1_map if real.b1_map is not None else np.ones(len(t1_values))
    n_total = len(voxels)
    logger.info("Loaded %d voxels, T1 mean=%.1f ms, protocol=%s", n_total, t1_values.mean(), real.protocol)

    # Load model
    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    hf = h5py.File(mrf_path, "r")
    nt = int(hf.attrs["n_signals"] * 0.8)
    t_mean = hf["parameters"][:nt, :2].astype(np.float32).mean(0)
    t_std = hf["parameters"][:nt, :2].astype(np.float32).std(0) + 1e-8
    hf.close()

    ckpt = ROOT / "results" / "checkpoints" / "abl_NLL_ER.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    # Get all predictions
    all_gamma, all_nu, all_alpha, all_beta = [], [], [], []
    batch_t = torch.from_numpy(voxels).to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(batch_t), 512):
            chunk = batch_t[i:i + 512]
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
    max_ep = epistemic.max(axis=-1)
    abs_error = np.abs(pred_t1 - t1_values)

    # Zero-shot baseline
    r_zero, _ = spearmanr(max_ep, abs_error)
    p_zero, _ = pearsonr(max_ep, abs_error)
    logger.info("  Zero-shot: Spearman=%.3f, Pearson=%.3f", r_zero, p_zero)

    # Adaptation curve: vary % of real data used for calibration
    cal_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    results = []

    for frac in cal_fracs:
        if frac == 0.0:
            # No calibration
            rho_s, rho_p = r_zero, p_zero
        else:
            # Split: frac for calibration, rest for test
            idx_cal, idx_test = train_test_split(
                np.arange(n_total), test_size=1.0 - frac, random_state=42
            )
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(max_ep[idx_cal], abs_error[idx_cal])
            calibrated = iso.predict(max_ep[idx_test])
            rho_s, _ = spearmanr(calibrated, abs_error[idx_test])
            rho_p, _ = pearsonr(calibrated, abs_error[idx_test])

        results.append({
            "cal_frac": frac,
            "n_cal": int(frac * n_total),
            "n_test": int((1 - frac) * n_total),
            "spearman_rho": float(rho_s),
            "pearson_r": float(rho_p),
        })
        logger.info("  Cal %.0f%%: Spearman=%.3f, Pearson=%.3f (n_cal=%d, n_test=%d)",
                     frac * 100, rho_s, rho_p, int(frac * n_total), int((1 - frac) * n_total))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    fracs = [r["cal_frac"] * 100 for r in results]
    rhos = [r["spearman_rho"] for r in results]
    ax.plot(fracs, rhos, "o-", lw=2, ms=8, color="#2196F3", label="Spearman ρ")
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("% of Real Data Used for Calibration")
    ax.set_ylabel("Spearman ρ (Epistemic vs Error)")
    ax.set_title("Calibration Repair Scaling Law\nHow much real data is needed to fix the sim-to-real gap?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for r in results:
        if r["cal_frac"] > 0:
            ax.annotate(f"ρ={r['spearman_rho']:.3f}",
                        (r["cal_frac"] * 100, r["spearman_rho"]),
                        textcoords="offset points", xytext=(0, 10), fontsize=9, ha="center")
    fig.tight_layout()
    fig.savefig(FIG / "adaptation_curve.png", dpi=200)
    plt.close(fig)

    with open(FIG / "adaptation_curve.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Adaptation curve saved.")
    return results


if __name__ == "__main__":
    run()
