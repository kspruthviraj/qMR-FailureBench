#!/usr/bin/env python3
"""
Re-evaluate existing checkpoints with unit-safe real-data loading.

Produces:
  results/figures/seed_study_v2.json          — 50-seed ResNet (if ckpts exist)
  results/figures/calibration_metrics_v2.json
  results/figures/arch_summary_v2.json        — multi-arch if ckpts exist
  results/figures/synth_seed_rho_v2.json      — synthetic-only ρ (unit-safe primary claim)

This does NOT retrain — it reloads existing weights and recomputes metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.data.loaders import (
    MRFMetaDataset,
    load_qmrlab_vfa,
    load_training_norm,
    assert_t1_units_ms,
)
from qMR_Robust.eval.calibration import (
    categorize_rho,
    normalized_ece,
    spearman_uncertainty_error,
    wilson_ci,
    gaussian_nll_numpy,
    z_score_calibration,
)
from qMR_Robust.eval.nig_utils import (
    nig_epistemic_np,
    nig_predictive_var_np,
    predict_nig_batches,
    denorm_gamma,
)
from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.vit1d import ViT1D
from qMR_Robust.models.spatiotemporal_transformer import SpatioTemporalTransformer
from qMR_Robust.models.convlstm1d import ConvLSTM1D
from qMR_Robust.models.unet1d import UNet1D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT = ROOT / "results" / "checkpoints"
FIG.mkdir(parents=True, exist_ok=True)

SEED_LIST = [
    7, 13, 21, 42, 123, 3, 11, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    483, 486, 286, 130, 421, 426, 383, 358, 547, 542, 770, 119, 370, 316, 655, 106,
    92, 760, 375, 209, 883, 714, 268, 78, 651, 82, 723, 167, 264, 44,
]


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


def eval_synthetic(model, mrf_path, t_mean, t_std, n_eval=2000, fail_thresh=300.0):
    val_ds = MRFMetaDataset(mrf_path, split="val", load_corruption_meta=True)
    val_ds.set_norm(t_mean, t_std)
    n = min(n_eval, len(val_ds))
    preds = predict_nig_batches(model, val_ds.signals[:n], DEVICE)
    gamma = denorm_gamma(preds["gamma"], t_mean, t_std)
    tgt = val_ds.params[:n]
    assert_t1_units_ms(tgt[:, 0], "synthetic val T1")
    resid = np.abs(tgt - gamma).max(axis=-1)
    ep = nig_epistemic_np(preds["nu"], preds["alpha"], preds["beta"]).max(axis=-1)
    ep = np.where(np.isfinite(ep), ep, 1e6)
    rho, _ = spearman_uncertainty_error(ep, resid)
    labels = (resid > fail_thresh).astype(int)
    auroc = float("nan")
    if 0 < labels.sum() < n:
        auroc = float(roc_auc_score(labels, ep))
    mae = float(resid.mean())
    return {
        "synth_mae_ms": mae,
        "synth_rho": float(rho),
        "synth_auroc": auroc,
        "mean_alpha": float(preds["alpha"].mean()),
        "mean_epistemic": float(ep.mean()),
    }


def eval_real(model, real, t_mean, t_std, do_isotonic=True):
    preds = predict_nig_batches(model, real.signals, DEVICE)
    gamma = denorm_gamma(preds["gamma"], t_mean, t_std)
    pred_t1 = gamma[:, 0]
    err = np.abs(real.t1_ms - pred_t1)
    ep = nig_epistemic_np(preds["nu"], preds["alpha"], preds["beta"]).max(axis=-1)
    ep = np.where(np.isfinite(ep), ep, 1e6)
    rho0, p0 = spearman_uncertainty_error(ep, err)
    ece = normalized_ece(ep, err)
    # Predictive variance for NLL (T1 only, denorm scale)
    # In normalized space compute Gaussian NLL then report
    var_norm = nig_predictive_var_np(preds["nu"][:, 0], preds["alpha"][:, 0], preds["beta"][:, 0])
    tgt_norm = (real.t1_ms - t_mean[0]) / t_std[0]
    nll = gaussian_nll_numpy(preds["gamma"][:, 0], var_norm, tgt_norm)
    zcal = z_score_calibration(preds["gamma"][:, 0], np.sqrt(var_norm), tgt_norm)

    rho5 = float("nan")
    if do_isotonic and len(ep) > 50:
        n_cal = max(20, int(0.05 * len(ep)))
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(ep))
        iso = IsotonicRegression(out_of_bounds="clip")
        try:
            iso.fit(ep[idx[:n_cal]], err[idx[:n_cal]])
            cal = iso.transform(ep[idx[n_cal:]])
            rho5, _ = spearman_uncertainty_error(cal, err[idx[n_cal:]])
        except Exception:
            rho5 = float("nan")

    return {
        "real_mae_ms": float(err.mean()),
        "real_rho_zero_shot": float(rho0),
        "real_rho_p": float(p0),
        "real_rho_5pct": float(rho5) if np.isfinite(rho5) else None,
        "normalized_ece": float(ece) if np.isfinite(ece) else None,
        "nll": float(nll) if np.isfinite(nll) else None,
        "z_rms": zcal.get("z_rms"),
        "cov_1sigma": zcal.get("cov_1sigma"),
        "pred_t1_mean": float(pred_t1.mean()),
        "pred_t1_std": float(pred_t1.std()),
        "category": categorize_rho(float(rho0)),
        "protocol": real.protocol,
    }


def summarize(results, rho_key="real_rho_zero_shot"):
    rhos = [r[rho_key] for r in results if r.get(rho_key) is not None and np.isfinite(r[rho_key])]
    cats = [categorize_rho(r) for r in rhos]
    n = len(rhos)
    n_good = sum(c == "good" for c in cats)
    n_deg = sum(c == "degenerate" for c in cats)
    n_mod = sum(c == "moderate" for c in cats)
    n_near = sum(c == "near_zero" for c in cats)
    fail_rate = n_deg / n if n else 0.0
    lo, hi = wilson_ci(n_deg, n) if n else (0.0, 1.0)
    good_rhos = [r for r, c in zip(rhos, cats) if c == "good"]
    deg_rhos = [r for r, c in zip(rhos, cats) if c == "degenerate"]
    return {
        "n": n,
        "n_good": n_good,
        "n_moderate": n_mod,
        "n_near_zero": n_near,
        "n_degenerate": n_deg,
        "failure_rate": fail_rate,
        "wilson_ci": [lo, hi],
        "mean_good_rho": float(np.mean(good_rhos)) if good_rhos else None,
        "std_good_rho": float(np.std(good_rhos)) if good_rhos else None,
        "mean_deg_rho": float(np.mean(deg_rhos)) if deg_rhos else None,
        "std_deg_rho": float(np.std(deg_rhos)) if deg_rhos else None,
        "mean_rho": float(np.mean(rhos)) if rhos else None,
        "std_rho": float(np.std(rhos)) if rhos else None,
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    t_mean, t_std = load_training_norm(mrf_path)
    print(f"Training norm: mean={t_mean}, std={t_std}")

    qmrlab = ROOT / cfg["paths"].get("qmrlab_vfa", "data/real/qmrlab/vfa_t1_data")
    real = load_qmrlab_vfa(qmrlab, pad_mode="zeropad")
    print(f"Real VFA: n={len(real.t1_ms)}, T1 mean={real.t1_ms.mean():.1f} ms, protocol={real.protocol}")

    # ── ResNet 50-seed study ──
    results = []
    for seed in SEED_LIST:
        ckpt = CKPT / f"seed_study_{seed}.pt"
        if not ckpt.exists():
            print(f"  skip seed {seed}: no checkpoint")
            continue
        model = build_model("resnet1d").to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()
        synth = eval_synthetic(model, mrf_path, t_mean, t_std)
        real_m = eval_real(model, real, t_mean, t_std)
        row = {"seed": seed, **synth, **real_m}
        results.append(row)
        print(
            f"  seed {seed:4d}: synth ρ={synth['synth_rho']:+.3f} MAE={synth['synth_mae_ms']:.0f} | "
            f"real ρ={real_m['real_rho_zero_shot']:+.3f} MAE={real_m['real_mae_ms']:.0f} [{real_m['category']}]"
        )

    summary_real = summarize(results, "real_rho_zero_shot")
    summary_synth = summarize(results, "synth_rho")
    out = {
        "version": "v2_unit_safe",
        "t1_units": "ms",
        "real_protocol": real.protocol,
        "n_fa_original": real.n_fa_original,
        "results": results,
        "summary_real": summary_real,
        "summary_synth": summary_synth,
    }
    path = FIG / "seed_study_v2.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {path}")
    print("REAL summary:", json.dumps(summary_real, indent=2))
    print("SYNTH summary:", json.dumps(summary_synth, indent=2))

    # Calibration table
    cal_rows = []
    for r in results:
        cal_rows.append({
            "seed": r["seed"],
            "rho": r["real_rho_zero_shot"],
            "nll": r.get("nll"),
            "normalized_ece": r.get("normalized_ece"),
            "category": r["category"],
            "z_rms": r.get("z_rms"),
        })
    good = [c for c in cal_rows if c["category"] == "good"]
    deg = [c for c in cal_rows if c["category"] == "degenerate"]

    def mean_std(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    cal_out = {
        "results": cal_rows,
        "good": {
            "n": len(good),
            "nll": mean_std(good, "nll"),
            "normalized_ece": mean_std(good, "normalized_ece"),
            "rho": mean_std(good, "rho"),
        },
        "degenerate": {
            "n": len(deg),
            "nll": mean_std(deg, "nll"),
            "normalized_ece": mean_std(deg, "normalized_ece"),
            "rho": mean_std(deg, "rho"),
        },
    }
    (FIG / "calibration_metrics_v2.json").write_text(json.dumps(cal_out, indent=2, default=str))

    # ── Multi-arch from existing arch30 checkpoints ──
    arch_summary = {}
    for arch in ["vit1d", "convlstm", "unet1d", "spatiotemporal", "resnet1d"]:
        rows = []
        ckpts = sorted(CKPT.glob(f"arch30_{arch}_seed*.pt"))
        # also seed_study for resnet
        if arch == "resnet1d" and not ckpts:
            ckpts = [CKPT / f"seed_study_{s}.pt" for s in SEED_LIST if (CKPT / f"seed_study_{s}.pt").exists()]
        for ckpt in ckpts:
            try:
                seed = int(str(ckpt.stem).split("seed")[-1].replace("_", "") if "seed" in ckpt.stem else ckpt.stem.split("_")[-1])
            except Exception:
                seed = -1
            try:
                model = build_model(arch).to(DEVICE)
                model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
                model.eval()
                synth = eval_synthetic(model, mrf_path, t_mean, t_std)
                real_m = eval_real(model, real, t_mean, t_std, do_isotonic=False)
                rows.append({"seed": seed, "ckpt": ckpt.name, **synth, **real_m})
            except Exception as e:
                print(f"  arch {arch} {ckpt.name} failed: {e}")
        if rows:
            arch_summary[arch] = {
                "n": len(rows),
                "summary_real": summarize(rows, "real_rho_zero_shot"),
                "summary_synth": summarize(rows, "synth_rho"),
                "results": rows,
            }
            print(f"\n{arch}: real {arch_summary[arch]['summary_real']}")
            print(f"{arch}: synth {arch_summary[arch]['summary_synth']}")

    (FIG / "arch_summary_v2.json").write_text(json.dumps(arch_summary, indent=2, default=str))
    print("\nDone. All v2 metrics use T1 in milliseconds + zeropad VFA protocol.")


if __name__ == "__main__":
    main()
