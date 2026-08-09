#!/usr/bin/env python3
"""Re-score Strong-ER / alpha-safeguard checkpoints with unit-safe real eval."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.data.loaders import load_qmrlab_vfa, load_training_norm
from qMR_Robust.eval.calibration import categorize_rho, spearman_uncertainty_error, wilson_ci
from qMR_Robust.eval.nig_utils import nig_epistemic_np, predict_nig_batches, denorm_gamma
from qMR_Robust.models.resnet1d import ResNet1D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = ROOT / "results" / "checkpoints"
FIG = ROOT / "results" / "figures"


def eval_ckpt(path, real, t_mean, t_std):
    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    preds = predict_nig_batches(model, real.signals, DEVICE)
    gamma = denorm_gamma(preds["gamma"], t_mean, t_std)
    err = np.abs(real.t1_ms - gamma[:, 0])
    ep = nig_epistemic_np(preds["nu"], preds["alpha"], preds["beta"]).max(axis=-1)
    ep = np.where(np.isfinite(ep), ep, 1e6)
    rho, _ = spearman_uncertainty_error(ep, err)
    return {
        "ckpt": path.name,
        "rho": float(rho),
        "mae_ms": float(err.mean()),
        "category": categorize_rho(float(rho)),
    }


def summarize(rows):
    rhos = [r["rho"] for r in rows]
    n_deg = sum(1 for r in rhos if r < -0.1)
    n = len(rhos)
    lo, hi = wilson_ci(n_deg, n) if n else (0, 1)
    return {
        "n": n,
        "n_degenerate": n_deg,
        "failure_rate": n_deg / n if n else 0,
        "wilson_ci": [lo, hi],
        "mean_rho": float(np.mean(rhos)) if rhos else None,
        "mean_mae_ms": float(np.mean([r["mae_ms"] for r in rows])) if rows else None,
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    mrf = ROOT / cfg["paths"]["failure_forecast_mrf"]
    t_mean, t_std = load_training_norm(mrf)
    real = load_qmrlab_vfa(ROOT / "data/real/qmrlab/vfa_t1_data", pad_mode="zeropad")

    patterns = {
        "baseline_seed_study": "seed_study_*.pt",
        "strong_er": "strong_er_*.pt",
        "alpha_safeguard": "alpha_*.pt",
        "er5": "*er5*.pt",
        "safeguard": "*safeguard*.pt",
    }
    # discover actual names
    all_ckpts = list(CKPT.glob("*.pt"))
    groups = {
        "baseline_seed_study": sorted(CKPT.glob("seed_study_*.pt")),
        "strong_er": sorted([p for p in all_ckpts if "strong_er" in p.name.lower() or "er5" in p.name.lower() or "er_5" in p.name.lower()]),
        "alpha": sorted([p for p in all_ckpts if "alpha" in p.name.lower()]),
    }
    # also list unique prefixes for user visibility
    print("Checkpoint name samples:", [p.name for p in all_ckpts[:20]])

    out = {}
    for name, paths in groups.items():
        print(f"\n=== {name}: {len(paths)} ckpts ===")
        rows = []
        for p in paths:
            try:
                row = eval_ckpt(p, real, t_mean, t_std)
                rows.append(row)
                print(f"  {p.name}: ρ={row['rho']:+.3f} MAE={row['mae_ms']:.0f} [{row['category']}]")
            except Exception as e:
                print(f"  skip {p.name}: {e}")
        if rows:
            out[name] = {"results": rows, "summary": summarize(rows)}
            print(" summary", out[name]["summary"])

    path = FIG / "mitigation_rescore_v2.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print("Wrote", path)


if __name__ == "__main__":
    main()
