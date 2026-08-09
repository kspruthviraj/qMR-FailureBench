#!/usr/bin/env python3
"""Fast multi-arch unit-safe reeval (uses all arch30_*.pt + seed_study for resnet)."""
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
from qMR_Robust.models.vit1d import ViT1D
from qMR_Robust.models.spatiotemporal_transformer import SpatioTemporalTransformer
from qMR_Robust.models.convlstm1d import ConvLSTM1D
from qMR_Robust.models.unet1d import UNet1D
from qMR_Robust.data.loaders import MRFMetaDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT = ROOT / "results" / "checkpoints"


def build_model(arch: str):
    if arch == "resnet1d":
        return ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True)
    if arch == "vit1d":
        return ViT1D(in_channels=2, hidden_dim=128, output_dim=2, patch_size=50,
                     n_heads=4, n_layers=4, dropout=0.1, evidential=True)
    if arch == "spatiotemporal":
        return SpatioTemporalTransformer(
            in_channels=2, seq_len=1000, hidden_dim=128, output_dim=2, n_heads=4,
            n_temporal_layers_1=3, n_temporal_layers_2=2, dropout=0.1, evidential=True,
        )
    if arch == "convlstm":
        return ConvLSTM1D(in_channels=2, hidden_dim=128, n_lstm_layers=2, output_dim=2,
                          dropout=0.1, evidential=True)
    if arch == "unet1d":
        return UNet1D(in_channels=2, hidden_dim=64, output_dim=2, dropout=0.1, evidential=True)
    raise ValueError(arch)


def summarize(results, key):
    rhos = [r[key] for r in results if r.get(key) is not None and np.isfinite(r[key])]
    n = len(rhos)
    n_good = sum(1 for r in rhos if r > 0.8)
    n_deg = sum(1 for r in rhos if r < -0.1)
    n_mod = sum(1 for r in rhos if 0.3 < r <= 0.8)
    n_near = sum(1 for r in rhos if -0.1 < r <= 0.3)
    lo, hi = wilson_ci(n_deg, n) if n else (0.0, 1.0)
    return {
        "n": n,
        "n_good": n_good,
        "n_moderate": n_mod,
        "n_near_zero": n_near,
        "n_degenerate": n_deg,
        "failure_rate": n_deg / n if n else 0.0,
        "wilson_ci": [lo, hi],
        "mean_rho": float(np.mean(rhos)) if rhos else None,
        "std_rho": float(np.std(rhos)) if rhos else None,
        "mean_good_rho": float(np.mean([r for r in rhos if r > 0.8])) if n_good else None,
    }


def eval_one(model, signals_s, params_s, real, t_mean, t_std):
    # synthetic
    pr = predict_nig_batches(model, signals_s, DEVICE)
    g = denorm_gamma(pr["gamma"], t_mean, t_std)
    resid = np.abs(params_s - g).max(-1)
    ep = nig_epistemic_np(pr["nu"], pr["alpha"], pr["beta"]).max(-1)
    ep = np.where(np.isfinite(ep), ep, 1e6)
    srho, _ = spearman_uncertainty_error(ep, resid)
    # real
    prr = predict_nig_batches(model, real.signals, DEVICE)
    gr = denorm_gamma(prr["gamma"], t_mean, t_std)
    err = np.abs(real.t1_ms - gr[:, 0])
    epr = nig_epistemic_np(prr["nu"], prr["alpha"], prr["beta"]).max(-1)
    epr = np.where(np.isfinite(epr), epr, 1e6)
    rrho, _ = spearman_uncertainty_error(epr, err)
    return {
        "synth_rho": float(srho),
        "synth_mae_ms": float(resid.mean()),
        "real_rho_zero_shot": float(rrho),
        "real_mae_ms": float(err.mean()),
        "category": categorize_rho(float(rrho)),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    mrf = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    t_mean, t_std = load_training_norm(mrf)
    real = load_qmrlab_vfa(ROOT / "data/real/qmrlab/vfa_t1_data", pad_mode="zeropad")
    val = MRFMetaDataset(mrf, split="val")
    n_s = min(2000, len(val))
    signals_s, params_s = val.signals[:n_s], val.params[:n_s]

    arch_summary = {}
    for arch in ["resnet1d", "vit1d", "convlstm", "unet1d", "spatiotemporal"]:
        if arch == "resnet1d":
            ckpts = sorted(CKPT.glob("seed_study_*.pt"))
        else:
            ckpts = sorted(CKPT.glob(f"arch30_{arch}_seed*.pt"))
        print(f"\n=== {arch}: {len(ckpts)} checkpoints ===")
        rows = []
        for ckpt in ckpts:
            try:
                seed = int("".join([c for c in ckpt.stem.split("seed")[-1] if c.isdigit()] or ["-1"]))
            except Exception:
                seed = -1
            try:
                model = build_model(arch).to(DEVICE)
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
                model.eval()
                m = eval_one(model, signals_s, params_s, real, t_mean, t_std)
                row = {"seed": seed, "ckpt": ckpt.name, **m}
                rows.append(row)
                print(f"  {ckpt.name}: synthρ={m['synth_rho']:+.3f} realρ={m['real_rho_zero_shot']:+.3f}")
            except Exception as e:
                print(f"  FAIL {ckpt.name}: {e}")
        if rows:
            arch_summary[arch] = {
                "n": len(rows),
                "summary_real": summarize(rows, "real_rho_zero_shot"),
                "summary_synth": summarize(rows, "synth_rho"),
                "results": rows,
            }
            print("  real", arch_summary[arch]["summary_real"])
            print("  synth", arch_summary[arch]["summary_synth"])

    out = FIG / "arch_summary_v2.json"
    out.write_text(json.dumps(arch_summary, indent=2, default=str))
    print("\nWrote", out)


if __name__ == "__main__":
    main()
