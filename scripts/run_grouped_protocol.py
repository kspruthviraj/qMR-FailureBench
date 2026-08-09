#!/usr/bin/env python3
"""Run an explicit domain-grouped train/calibration/test protocol.

This runner is intentionally separate from the legacy scripts. It selects a
checkpoint using the calibration partition and reports final metrics only on
the untouched grouped test partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.data.loaders import MRFMetaDataset
from qMR_Robust.data.splits import build_split_manifest
from qMR_Robust.models.losses import evidential_regression_loss
from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.reproducibility import seed_everything


def _predict(model, loader, mean, std, device):
    model.eval()
    gammas, epis, targets = [], [], []
    with torch.no_grad():
        for signals, y in loader:
            out = model(signals.to(device))
            nig = out if out.ndim == 3 else out.view(out.shape[0], 2, 4)
            gamma, nu, alpha, beta = nig[..., 0], nig[..., 1], nig[..., 2], nig[..., 3]
            gammas.append(gamma.cpu().numpy())
            epis.append((beta / (nu * (alpha - 1.0)).clamp(min=1e-8)).cpu().numpy())
            targets.append(y.numpy())
    gamma = np.concatenate(gammas)
    epistemic = np.concatenate(epis)
    target = np.concatenate(targets)
    return gamma * std + mean, target * std + mean, epistemic


def _scores(pred, target, epistemic, tolerance_ms=300.0):
    residual = np.abs(pred - target)
    labels = (residual.max(axis=1) > tolerance_ms).astype(np.uint8)
    score = epistemic.max(axis=1)
    result = {
        "n_samples": int(len(pred)),
        "mae_ms": float(residual.mean()),
        "rmse_ms": float(np.sqrt(np.mean(residual ** 2))),
        "failure_rate": float(labels.mean()),
        "predicted_t1_std_ms": float(pred[:, 0].std()),
        "residual_uncertainty_spearman": None,
        "auroc": None,
        "auprc": None,
    }
    if labels.min() != labels.max():
        result["auroc"] = float(roc_auc_score(labels, score))
        result["auprc"] = float(average_precision_score(labels, score))
    rho = spearmanr(score, residual.max(axis=1))
    if np.isfinite(rho.statistic):
        result["residual_uncertainty_spearman"] = float(rho.statistic)
    return result


def _calibration_nll(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for signals, y in loader:
            y = y.to(device)
            out = model(signals.to(device))
            nig = out if out.ndim == 3 else out.view(out.shape[0], 2, 4)
            result = evidential_regression_loss(
                y, nig, coeff=1.0, epoch=1, annealing_epochs=1
            )
            total += float(result["nll"].mean()) * len(y)
            count += len(y)
    return total / max(count, 1)



def _domain_scores(pred, target, epistemic, domains):
    """Return sample-weighted test metrics for each held-out acquisition domain."""
    result = []
    domains = np.asarray(domains)
    for domain in sorted(set(str(value) for value in domains)):
        mask = np.asarray([str(value) == domain for value in domains], dtype=bool)
        metrics = _scores(pred[mask], target[mask], epistemic[mask])
        metrics["domain"] = domain
        result.append(metrics)
    return result


def _bootstrap_domain_metric(run, metric, repetitions=2000):
    """Bootstrap complete held-out domains, preserving domain-level independence."""
    domains = run.get("test_by_domain", [])
    if not domains:
        return None
    values = np.asarray([item.get(metric, np.nan) for item in domains], dtype=float)
    weights = np.asarray([item.get("n_samples", 0) for item in domains], dtype=float)
    valid = np.isfinite(values) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if len(values) < 2:
        return None
    rng = np.random.default_rng(20260809 + int(run["seed"]))
    estimates = []
    for _ in range(int(repetitions)):
        sampled = rng.integers(0, len(values), size=len(values))
        if metric == "residual_uncertainty_spearman":
            estimates.append(float(np.mean(values[sampled])))
        else:
            estimates.append(float(np.average(values[sampled], weights=weights[sampled])))
    return {
        "statistic": float(np.average(values, weights=weights))
        if metric != "residual_uncertainty_spearman"
        else float(np.mean(values)),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n_domains": int(len(values)),
        "repetitions": int(repetitions),
    }


def _summarize_runs(runs):
    metrics = (
        "mae_ms",
        "rmse_ms",
        "failure_rate",
        "predicted_t1_std_ms",
        "residual_uncertainty_spearman",
        "auroc",
        "auprc",
    )
    summary = {"seed_mean_std": {}, "grouped_bootstrap_ci95": {}}
    for metric in metrics:
        values = np.asarray(
            [run["test"].get(metric, np.nan) for run in runs], dtype=float
        )
        values = values[np.isfinite(values)]
        if len(values):
            summary["seed_mean_std"][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n_seeds": int(len(values)),
            }
        per_seed = {}
        for run in runs:
            interval = _bootstrap_domain_metric(run, metric)
            if interval is not None:
                per_seed[str(run["seed"])] = interval
        if per_seed:
            summary["grouped_bootstrap_ci95"][metric] = {
                "per_seed": per_seed,
                "mean_ci95": [
                    float(np.mean([item["ci95"][0] for item in per_seed.values()])),
                    float(np.mean([item["ci95"][1] for item in per_seed.values()])),
                ],
            }
    return summary

def run_one_seed(
    h5_path: Path,
    manifest: dict,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict:
    seed_everything(seed, deterministic=True)
    indices = manifest["grouped_indices"]
    train_ds = MRFMetaDataset(
        h5_path, split="train", indices=np.asarray(indices["train"], dtype=np.int64)
    )
    calibration_ds = MRFMetaDataset(
        h5_path, split="calibration",
        indices=np.asarray(indices["calibration"], dtype=np.int64),
    )
    test_ds = MRFMetaDataset(
        h5_path, split="test", indices=np.asarray(indices["test"], dtype=np.int64)
    )
    calibration_ds.set_norm(train_ds.mean, train_ds.std)
    test_ds.set_norm(train_ds.mean, train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    calibration_loader = DataLoader(calibration_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNet1D(
        in_channels=2, hidden_dim=128, output_dim=2,
        dropout=0.1, evidential=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_state = None
    best_calibration = float("inf")

    for epoch in range(epochs):
        model.train()
        for signals, y in train_loader:
            signals, y = signals.to(device), y.to(device)
            nig = model(signals)
            loss = evidential_regression_loss(
                y, nig, coeff=1.0, epoch=epoch, annealing_epochs=max(10, epochs)
            )["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        calibration_nll = _calibration_nll(model, calibration_loader, device)
        if calibration_nll < best_calibration:
            best_calibration = calibration_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("no calibration checkpoint was selected")
    model.load_state_dict(best_state)
    pred, target, epistemic = _predict(
        model, test_loader, train_ds.mean, train_ds.std, device
    )
    with h5py.File(h5_path, "r") as hf:
        test_domains = np.asarray(
            hf["domain_labels"][np.asarray(indices["test"], dtype=np.int64)]
        )
    return {
        "seed": int(seed),
        "calibration_nll": float(best_calibration),
        "test": _scores(pred, target, epistemic),
        "test_by_domain": _domain_scores(pred, target, epistemic, test_domains),
        "split_protocol": manifest["grouped_protocol"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=ROOT / "data/synthetic/failure_forecast_mrf.h5")
    parser.add_argument("--manifest", type=Path, default=ROOT / "frozen_results/grouped_split_manifest_v1.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "frozen_results/grouped_protocol_results.json")
    args = parser.parse_args()

    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
    else:
        manifest = build_split_manifest(args.h5, seed=42)
        args.manifest.write_text(json.dumps(manifest, indent=2))

    runs = [
        run_one_seed(
            args.h5, manifest, seed, args.epochs, args.batch_size, args.device
        )
        for seed in args.seeds
    ]
    results = {
        "protocol_version": "grouped_train_calibration_test_v3_deterministic",
        "h5": str(args.h5),
        "manifest": str(args.manifest),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": args.device,
        "deterministic_training": True,
        "split_counts": manifest["grouped_sample_counts"],
        "runs": runs,
        "summary": _summarize_runs(runs),
    }
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
