#!/usr/bin/env python3
"""
run_failure_taxonomy.py — Analyze what predicts seed collapse.

For each of the 5 seeds, compute:
  - Training MAE
  - Validation MAE
  - Evidence parameters (mean ν, α, β)
  - Epistemic variance
  - Real-data ρ (zero-shot)
  - Whether repair works (5% ρ > 0)

Then analyze: what distinguishes healthy seeds from degenerate ones?
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from scripts.run_seed_study import MRFMetaDataset, evaluate_adaptation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT_DIR = ROOT / "results" / "checkpoints"


def extract_evidence_params(mrf_path, ckpt_path):
    """Extract evidence parameters (ν, α, β) from a trained model."""
    torch.manual_seed(0)

    train_ds = MRFMetaDataset(mrf_path, split="train")
    val_ds = MRFMetaDataset(mrf_path, split="val")
    val_ds.set_norm(train_ds.mean, train_ds.std)
    t_mean, t_std = train_ds.mean, train_ds.std

    model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    n_eval = min(2000, len(val_ds))
    all_nu, all_alpha, all_beta = [], [], []
    all_gamma, all_tgt = [], []

    with torch.no_grad():
        for i in range(0, n_eval, 256):
            batch_x = torch.from_numpy(val_ds.signals[i:i+256]).to(DEVICE)
            batch_y = val_ds.params[i:i+256]
            out = model(batch_x)
            nig = out.view(out.shape[0], 2, 4)
            all_gamma.append(nig[..., 0].cpu().numpy())
            all_nu.append(nig[..., 1].cpu().numpy())
            all_alpha.append(nig[..., 2].cpu().numpy())
            all_beta.append(nig[..., 3].cpu().numpy())
            all_tgt.append(batch_y)

    gamma = np.concatenate(all_gamma)
    nu = np.concatenate(all_nu)
    alpha = np.concatenate(all_alpha)
    beta = np.concatenate(all_beta)
    tgt = np.concatenate(all_tgt)

    gamma_d = gamma * t_std + t_mean
    tgt_d = tgt * t_std + t_mean
    resid = np.abs(tgt_d - gamma_d)

    epistemic = beta / (nu * (alpha - 1.0))
    aleatoric = beta / (alpha - 1.0)

    return {
        "train_mae": float(np.mean(resid)),
        "mean_nu": float(np.mean(nu)),
        "mean_alpha": float(np.mean(alpha)),
        "mean_beta": float(np.mean(beta)),
        "mean_epistemic": float(np.mean(epistemic)),
        "mean_aleatoric": float(np.mean(aleatoric)),
        "std_epistemic": float(np.std(epistemic)),
        "nu_range": float(np.percentile(nu, 95) - np.percentile(nu, 5)),
        "alpha_range": float(np.percentile(alpha, 95) - np.percentile(alpha, 5)),
        "max_nu": float(np.max(nu)),
        "max_alpha": float(np.max(alpha)),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    seeds = [7, 13, 21, 42, 123]

    # Try to load seed study results first
    seed_study = None
    seed_study_path = FIG / "seed_study.json"
    if seed_study_path.exists():
        seed_study = json.load(open(seed_study_path))

    results = []
    for seed in seeds:
        ckpt = CKPT_DIR / f"seed_study_{seed}.pt"
        if not ckpt.exists():
            print(f"Seed {seed}: no checkpoint, skipping")
            continue

        print(f"Analyzing seed {seed}...")
        evidence = extract_evidence_params(mrf_path, ckpt)

        # Get adaptation results from seed study if available
        rho_zero = rho_cal = None
        if seed_study and "results" in seed_study:
            for r in seed_study["results"]:
                if r["seed"] == seed:
                    rho_zero = r.get("real_rho_zero_shot")
                    rho_cal = r.get("real_rho_5pct")
                    break

        result = {"seed": seed, **evidence}
        if rho_zero is not None:
            result["real_rho_zero_shot"] = rho_zero
            result["real_rho_5pct"] = rho_cal
            result["degenerate"] = rho_zero < -0.5 if rho_zero is not None else None
        results.append(result)

        print(f"  nu={evidence['mean_nu']:.4f} alpha={evidence['mean_alpha']:.4f} "
              f"beta={evidence['mean_beta']:.4f} max_nu={evidence['max_nu']:.2f}")
        if rho_zero is not None:
            print(f"  rho_zero={rho_zero:.3f} degenerate={result['degenerate']}")

    if len(results) < 2:
        print("Not enough seeds with checkpoints for taxonomy analysis")
        return

    # Analysis: what predicts degeneracy?
    print("\n" + "=" * 60)
    print("  FAILURE TAXONOMY")
    print("=" * 60)

    has_rho = all(r.get("real_rho_zero_shot") is not None for r in results)
    if has_rho:
        good = [r for r in results if not r.get("degenerate", True)]
        bad = [r for r in results if r.get("degenerate", False)]

        if good and bad:
            print(f"\nGood seeds ({len(good)}): {[r['seed'] for r in good]}")
            print(f"Bad seeds ({len(bad)}): {[r['seed'] for r in bad]}")

            for metric in ["mean_nu", "mean_alpha", "mean_beta", "mean_epistemic",
                           "mean_aleatoric", "max_nu", "max_alpha", "train_mae"]:
                good_vals = [r[metric] for r in good]
                bad_vals = [r[metric] for r in bad]
                print(f"  {metric}: good={np.mean(good_vals):.4f}±{np.std(good_vals):.4f}  "
                      f"bad={np.mean(bad_vals):.4f}±{np.std(bad_vals):.4f}")

            # Key predictor: evidence saturation
            good_max_nu = np.mean([r["max_nu"] for r in good])
            bad_max_nu = np.mean([r["max_nu"] for r in bad])
            good_max_alpha = np.mean([r["max_alpha"] for r in good])
            bad_max_alpha = np.mean([r["max_alpha"] for r in bad])

            print(f"\n  KEY FINDING:")
            print(f"  Good seeds: max_ν={good_max_nu:.2f}, max_α={good_max_alpha:.2f}")
            print(f"  Bad seeds:  max_ν={bad_max_nu:.2f}, max_α={bad_max_alpha:.2f}")
            if bad_max_nu > good_max_nu * 2:
                print(f"  → Evidence saturation (high ν) predicts collapse")
            elif bad_max_alpha > good_max_alpha * 2:
                print(f"  → Aleatoric evidence saturation (high α) predicts collapse")
            else:
                print(f"  → Pattern not clear from these seeds alone")
        else:
            print("\nAll seeds are good or all are bad — no taxonomy possible")
    else:
        print("\nNo real-data ρ available — showing evidence params only")
        for r in results:
            print(f"  Seed {r['seed']}: nu={r['mean_nu']:.4f} alpha={r['mean_alpha']:.4f} "
                  f"beta={r['mean_beta']:.4f} max_nu={r['max_nu']:.2f}")

    # Save
    output = {"taxonomy": results}
    with open(FIG / "failure_taxonomy.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {FIG / 'failure_taxonomy.json'}")


if __name__ == "__main__":
    main()
