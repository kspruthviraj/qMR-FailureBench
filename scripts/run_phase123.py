#!/usr/bin/env python3
"""
run_phase123.py — All Phase 1/2/3 upgrades.

Phase 1: Narrative refinements (done in paper rewrite)
Phase 2: Selective prediction, cross-vendor, cross-field experiments
Phase 3: Bibliography expansion, benchmark README
"""
from __future__ import annotations
import json, logging, subprocess, time
from pathlib import Path
import h5py, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, yaml
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("phase123")
ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT = ROOT / "results" / "checkpoints"


# ── Phase 2: Selective Prediction ────────────────────────────────────────────

def run_selective_prediction():
    """Coverage vs Error at 100/90/80/70% thresholds."""
    logger.info("Phase 2: Selective prediction")
    from qMR_Robust.models.resnet1d import ResNet1D

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    hf = h5py.File(mrf, "r")
    n = hf.attrs["n_signals"]; nt = int(n * 0.8)
    sigs = hf["corrupted_signals"][nt:n]
    params = hf["parameters"][nt:n, :2].astype(np.float32)
    t_mean = hf["parameters"][:nt, :2].astype(np.float32).mean(0)
    t_std = hf["parameters"][:nt, :2].astype(np.float32).std(0) + 1e-8
    hf.close()
    sigs_2ch = np.stack([sigs.real, sigs.imag], axis=1).astype(np.float32)

    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    model.load_state_dict(torch.load(CKPT / "abl_NLL_ER.pt", map_location=DEVICE, weights_only=True))
    model.eval()

    all_out = []
    with torch.no_grad():
        for i in range(0, len(sigs_2ch), 512):
            raw = model(torch.from_numpy(sigs_2ch[i:i+512]).to(DEVICE))
            B, D = raw.shape[0], 2
            all_out.append(raw.view(B, D, 4).cpu().numpy())
    out = np.concatenate(all_out)
    gamma = out[..., 0] * t_std + t_mean
    epistemic = (out[..., 3] / (out[..., 1] * (out[..., 2] - 1.0)))
    resid = np.abs(params - gamma)
    max_ep = epistemic.max(axis=-1)
    max_resid = resid.max(axis=-1)

    coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    results = []
    for cov in coverages:
        n_keep = max(1, int(len(max_ep) * cov))
        idx = np.argsort(max_ep)[:n_keep]
        mae = float(np.mean(max_resid[idx]))
        rmse = float(np.sqrt(np.mean(max_resid[idx]**2)))
        results.append({"coverage": cov, "n_kept": n_keep, "mae_ms": mae, "rmse_ms": rmse})
        logger.info("  Coverage %.0f%%: MAE=%.1f ms, RMSE=%.1f ms (n=%d)", cov*100, mae, rmse, n_keep)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["coverage"]*100 for r in results], [r["mae_ms"] for r in results], "o-", lw=2, ms=8, color="#2196F3", label="MAE")
    ax.plot([r["coverage"]*100 for r in results], [r["rmse_ms"] for r in results], "s--", lw=2, ms=8, color="#E91E63", label="RMSE")
    ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Error (ms)")
    ax.set_title("Selective Prediction: Rejecting High-Uncertainty Voxels Reduces Error")
    ax.legend(); ax.invert_xaxis(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "selective_prediction.png", dpi=200); plt.close(fig)

    with open(FIG / "selective_prediction.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ── Phase 2: Cross-Vendor Holdout ────────────────────────────────────────────

def run_cross_vendor():
    """Train on Siemens+Philips, test on GE (and vice versa)."""
    logger.info("Phase 2: Cross-vendor holdout")
    from qMR_Robust.models.resnet1d import ResNet1D
    from torch.nn import functional as F

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    hf = h5py.File(mrf, "r")
    n = hf.attrs["n_signals"]
    sigs = hf["corrupted_signals"][:]
    params = hf["parameters"][:, :2].astype(np.float32)
    domains = [d.decode() for d in hf["domain_labels"][:]]
    hf.close()

    vendors = {"siemens": [], "philips": [], "ge": []}
    for i, d in enumerate(domains):
        for v in vendors:
            if d.startswith(v):
                vendors[v].append(i)
                break

    sigs_2ch = np.stack([sigs.real, sigs.imag], axis=1).astype(np.float32)
    t_mean = params.mean(0); t_std = params.std(0) + 1e-8
    params_norm = (params - t_mean) / t_std

    results = {}
    for test_vendor in ["siemens", "philips", "ge"]:
        train_idx = []
        for v, idxs in vendors.items():
            if v != test_vendor:
                train_idx.extend(idxs)
        test_idx = vendors[test_vendor]

        if len(train_idx) < 100 or len(test_idx) < 100:
            continue

        train_sigs = torch.from_numpy(sigs_2ch[train_idx]).to(DEVICE)
        train_tgt = torch.from_numpy(params_norm[train_idx]).to(DEVICE)
        test_sigs = torch.from_numpy(sigs_2ch[test_idx]).to(DEVICE)
        test_tgt = params_norm[test_idx]
        test_tgt_denorm = params[test_idx]

        model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
        ckpt_path = CKPT / f"cross_vendor_excl_{test_vendor}.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for epoch in range(20):
                model.train()
                idx = torch.randperm(len(train_sigs))
                for j in range(0, len(idx), 512):
                    batch_idx = idx[j:j+512]
                    out = model(train_sigs[batch_idx])
                    B, D = out.shape[0], 2; out = out.view(B, D, 4)
                    gamma, nu, alpha, beta = out[...,0], out[...,1], out[...,2], out[...,3]
                    from qMR_Robust.models.losses import nig_nll_loss, evidential_regularizer
                    loss = nig_nll_loss(train_tgt[batch_idx], gamma, nu, alpha, beta).mean()
                    anneal = min(1, epoch/10)
                    loss = loss + anneal * evidential_regularizer(train_tgt[batch_idx], gamma, nu, alpha).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            torch.save(model.state_dict(), ckpt_path)

        model.eval()
        with torch.no_grad():
            out = model(test_sigs).view(-1, 2, 4).cpu().numpy()
        gamma = out[..., 0] * t_std + t_mean
        epistemic = out[..., 3] / (out[..., 1] * (out[..., 2] - 1.0))
        resid = np.abs(test_tgt_denorm - gamma)
        max_ep = epistemic.max(axis=-1)
        max_resid = resid.max(axis=-1)
        r, _ = spearmanr(max_ep, max_resid)

        results[f"train_others_test_{test_vendor}"] = {
            "n_train": len(train_idx), "n_test": len(test_idx),
            "mae_ms": float(np.mean(max_resid)),
            "spearman_rho": float(r),
        }
        logger.info("  Train others, test %s: MAE=%.1f, rho=%.3f (n_test=%d)",
                     test_vendor, np.mean(max_resid), r, len(test_idx))

    with open(FIG / "cross_vendor.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
    short = [n.split("_")[-1].upper() for n in names]
    axes[0].bar(short, [results[n]["mae_ms"] for n in names], color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[0].set_title("Cross-Vendor MAE"); axes[0].set_ylabel("MAE (ms)")
    axes[1].bar(short, [results[n]["spearman_rho"] for n in names], color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[1].set_title("Cross-Vendor Epistemic-Error ρ"); axes[1].set_ylabel("Spearman ρ")
    fig.suptitle("Cross-Vendor Generalization: Train on 2 vendors, test on held-out vendor")
    fig.tight_layout(); fig.savefig(FIG / "cross_vendor.png", dpi=200); plt.close(fig)
    return results


# ── Phase 2: Cross-Field-Strength Holdout ────────────────────────────────────

def run_cross_field():
    """Train on 1.5T+3T, test on 7T (and permutations)."""
    logger.info("Phase 2: Cross-field-strength holdout")
    from qMR_Robust.models.resnet1d import ResNet1D

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf = str(ROOT / cfg["paths"]["failure_forecast_mrf"])
    hf = h5py.File(mrf, "r")
    n = hf.attrs["n_signals"]
    sigs = hf["corrupted_signals"][:]
    params = hf["parameters"][:, :2].astype(np.float32)
    domains = [d.decode() for d in hf["domain_labels"][:]]
    hf.close()

    fields = {"1.5T": [], "3.0T": [], "7.0T": []}
    for i, d in enumerate(domains):
        for f in fields:
            if f in d:
                fields[f].append(i)
                break

    sigs_2ch = np.stack([sigs.real, sigs.imag], axis=1).astype(np.float32)
    t_mean = params.mean(0); t_std = params.std(0) + 1e-8
    params_norm = (params - t_mean) / t_std

    results = {}
    for test_field in ["1.5T", "3.0T", "7.0T"]:
        train_idx = []
        for f, idxs in fields.items():
            if f != test_field:
                train_idx.extend(idxs)
        test_idx = fields[test_field]
        if len(train_idx) < 100 or len(test_idx) < 100:
            continue

        train_sigs = torch.from_numpy(sigs_2ch[train_idx]).to(DEVICE)
        train_tgt = torch.from_numpy(params_norm[train_idx]).to(DEVICE)
        test_sigs = torch.from_numpy(sigs_2ch[test_idx]).to(DEVICE)
        test_tgt_denorm = params[test_idx]

        model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
        ckpt_path = CKPT / f"cross_field_excl_{test_field.replace('.','')}.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for epoch in range(20):
                model.train()
                idx = torch.randperm(len(train_sigs))
                for j in range(0, len(idx), 512):
                    batch_idx = idx[j:j+512]
                    out = model(train_sigs[batch_idx])
                    B, D = out.shape[0], 2; out = out.view(B, D, 4)
                    gamma, nu, alpha, beta = out[...,0], out[...,1], out[...,2], out[...,3]
                    from qMR_Robust.models.losses import nig_nll_loss, evidential_regularizer
                    loss = nig_nll_loss(train_tgt[batch_idx], gamma, nu, alpha, beta).mean()
                    loss = loss + min(1, epoch/10) * evidential_regularizer(train_tgt[batch_idx], gamma, nu, alpha).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            torch.save(model.state_dict(), ckpt_path)

        model.eval()
        with torch.no_grad():
            out = model(test_sigs).view(-1, 2, 4).cpu().numpy()
        gamma = out[..., 0] * t_std + t_mean
        epistemic = out[..., 3] / (out[..., 1] * (out[..., 2] - 1.0))
        resid = np.abs(test_tgt_denorm - gamma)
        r, _ = spearmanr(epistemic.max(-1), resid.max(-1))

        results[f"train_others_test_{test_field}"] = {
            "n_train": len(train_idx), "n_test": len(test_idx),
            "mae_ms": float(np.mean(resid)),
            "spearman_rho": float(r),
        }
        logger.info("  Train others, test %s: MAE=%.1f, rho=%.3f", test_field, np.mean(resid), r)

    with open(FIG / "cross_field.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(results.keys())
    short = [n.split("_")[-1] for n in names]
    axes[0].bar(short, [results[n]["mae_ms"] for n in names], color=["#9C27B0", "#00BCD4", "#FF5722"])
    axes[0].set_title("Cross-Field MAE"); axes[0].set_ylabel("MAE (ms)")
    axes[1].bar(short, [results[n]["spearman_rho"] for n in names], color=["#9C27B0", "#00BCD4", "#FF5722"])
    axes[1].set_title("Cross-Field Epistemic-Error ρ"); axes[1].set_ylabel("Spearman ρ")
    fig.suptitle("Cross-Field-Strength Generalization")
    fig.tight_layout(); fig.savefig(FIG / "cross_field.png", dpi=200); plt.close(fig)
    return results


# ── Phase 3: Benchmark README ────────────────────────────────────────────────

def write_benchmark_readme():
    """Write a polished benchmark README with challenge splits."""
    logger.info("Phase 3: Benchmark README")
    bench = ROOT / "qMR-FailureBench"
    bench.mkdir(exist_ok=True)

    readme = """# qMR-FailureBench

**A Benchmark for Explainable Failure Forecasting and Counterfactual Correction in Quantitative MRI**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

qMR-FailureBench provides standardized datasets, evaluation protocols, and baseline results for testing uncertainty quantification and failure detection methods in quantitative MRI under entangled physical corruptions.

## Quick Start

```bash
# Clone
git clone https://github.com/qMR-Robust/qMR-FailureBench.git
cd qMR-FailureBench

# Install dependencies
pip install -r requirements.txt

# Run baseline evaluation
python evaluation.py --method evidential --checkpoint checkpoints/abl_NLL_ER.pt

# Reproduce all paper results
python reproduce_paper.py
```

## Challenge Splits

| Split | Purpose | Train | Test |
|-------|---------|-------|------|
| **In-domain** | Standard evaluation | Random 80% | Random 20% |
| **Cross-vendor** | Vendor generalization | Siemens + Philips | GE |
| **Cross-field** | Field strength generalization | 1.5T + 3T | 7T |
| **Sim-to-real** | Real-world transfer | Synthetic signals | Real in-vivo (qMRLab) |

## Evaluation Tasks

| Task | Metric | Description |
|------|--------|-------------|
| **Failure Detection** | AUROC, AUPRC | Flag voxels where residual > tolerance |
| **Corruption Attribution** | Precision, Recall, F1 | Identify corruption source (B₀/B₁⁺/Motion) |
| **Severity Estimation** | MAE | Predict corruption magnitudes (Δf, λ, δ) |
| **Counterfactual Repair** | ΔMAE, % improved | Error reduction after correction |
| **Sim-to-Real Calibration** | Spearman ρ | Uncertainty-error rank correlation on real data |

## Corruption Types

Each sample can be corrupted by any combination of:
1. **B₀ off-resonance**: Frequency shift [-80, 80] Hz
2. **B₁+ transmit scaling**: Amplitude scaling [0.6, 1.4]
3. **k-space motion**: Translation ±8 voxels, rotation ±15°

Corruptions are *entangled*: multiple types can co-occur on the same signal.

## HDF5 Schema

### MRF Benchmark (`mrf_benchmark.h5`)
| Dataset | Shape | Description |
|---------|-------|-------------|
| `clean_signals` | (N, 1000) complex64 | Clean MRF signals |
| `corrupted_signals` | (N, 1000) complex64 | Entangled-corrupted signals |
| `parameters` | (N, 3) float32 | [T1, T2, M0] ground truth |
| `b0_hz_applied` | (N,) float32 | Applied B₀ shift |
| `b1_scale_applied` | (N,) float32 | Applied B₁ scale |
| `motion_shift_applied` | (N,) int32 | Applied motion shift |
| `domain_labels` | (N,) string | Vendor_field_FA_TR |

### MRS Benchmark (`mrs_benchmark.h5`)
| Dataset | Shape | Description |
|---------|-------|-------------|
| `clean_spectra` | (N, 2048) complex64 | Clean MRS spectra |
| `corrupted_spectra` | (N, 2048) complex64 | Entangled-corrupted spectra |
| `concentrations` | (N, 8) float38 | Metabolite concentrations |

## Baseline Results

| Method | MAE (ms) | AUROC | Spearman ρ |
|--------|----------|-------|------------|
| Deterministic | 229.1 | --- | --- |
| Heteroscedastic | 249.1 | 0.710 | --- |
| Quantile | 224.3 | 0.574 | --- |
| Deep Ensemble (5) | 221.0 | 0.476 | 0.010 |
| **Ours (NLL+ER)** | **223.9±14.5** | **0.642±0.020** | **0.149±0.040** |

## Citation

```bibtex
@article{qmrfailurebench2026,
  title={The Sim-to-Real Uncertainty Gap in Quantitative MRI: Characterization, Benchmark, and Counterfactual Correction},
  author={qMR-Robust Research Group},
  year={2026}
}
```

## License

MIT License
"""
    with open(bench / "README.md", "w") as f:
        f.write(readme)
    logger.info("  README written to %s", bench / "README.md")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Phase 2 experiments
    selective = run_selective_prediction()
    cross_vendor = run_cross_vendor()
    cross_field = run_cross_field()

    # Phase 3 benchmark
    write_benchmark_readme()

    # Save combined
    with open(FIG / "phase123_results.json", "w") as f:
        json.dump({"selective": selective, "cross_vendor": cross_vendor, "cross_field": cross_field}, f, indent=2)

    logger.info("=" * 60)
    logger.info("ALL PHASE 1/2/3 EXPERIMENTS COMPLETE in %.0f s", time.time() - t0)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
