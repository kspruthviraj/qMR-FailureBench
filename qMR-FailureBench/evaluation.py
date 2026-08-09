#!/usr/bin/env python3
"""Standard evaluation script for qMR-FailureBench.

Failure labels are derived from prediction residuals and stored tolerances.
They are not read from severity proxies in the HDF5 file.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _ranking_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=float)
    if labels.min() == labels.max():
        return {}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _validate_array(name, value, expected_shape):
    value = np.asarray(value)
    if value.shape != expected_shape:
        raise ValueError(f"{name} shape {value.shape} != expected {expected_shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def evaluate(predictions_path: str, benchmark_dir: str = ".", modality: str = "mrf"):
    benchmark = Path(benchmark_dir)
    with open(predictions_path) as f:
        preds = json.load(f)

    file_name = "mrf/mrf_benchmark.h5" if modality == "mrf" else "mrs/mrs_benchmark.h5"
    with h5py.File(benchmark / file_name, "r") as hf:
        if modality == "mrf":
            gt = np.asarray(hf["parameters"][:, :2], dtype=float)
            tolerances = np.array([
                float(hf.attrs["tolerance_t1_ms"]),
                float(hf.attrs["tolerance_t2_ms"]),
            ])
            flags = np.stack([
                hf["corruption_b0"][:],
                hf["corruption_b1"][:],
                hf["corruption_motion"][:],
            ], axis=-1).astype(np.uint8)
            target_names = ["t1", "t2"]
            unit = "ms"
        else:
            gt = np.asarray(hf["concentrations"][:], dtype=float)
            tolerances = np.array([float(hf.attrs["tolerance_gaba_mM"])])
            flags = np.stack([
                hf["corruption_b0"][:],
                hf["corruption_b1"][:],
                hf["corruption_motion"][:],
            ], axis=-1).astype(np.uint8)
            target_names = ["gaba"]
            unit = "mM"

    n = gt.shape[0]
    if "predictions" not in preds:
        raise ValueError("predictions JSON must contain a predictions field")
    pred = _validate_array("predictions", preds["predictions"], gt.shape)
    residual = np.abs(gt - pred)

    if modality == "mrf":
        failure_by_target = residual > tolerances[None, :]
        results = {
            "modality": "mrf",
            "n_samples": int(n),
            "mae_ms": float(residual.mean()),
            "rmse_ms": float(np.sqrt(np.mean(residual ** 2))),
            "mae_t1_ms": float(residual[:, 0].mean()),
            "mae_t2_ms": float(residual[:, 1].mean()),
            "failure_rate_t1": float(failure_by_target[:, 0].mean()),
            "failure_rate_t2": float(failure_by_target[:, 1].mean()),
        }
        failure_any = failure_by_target.any(axis=1)
    else:
        failure_any = residual[:, 3] > tolerances[0]
        results = {
            "modality": "mrs",
            "n_samples": int(n),
            "mae_mM": float(residual.mean()),
            "rmse_mM": float(np.sqrt(np.mean(residual ** 2))),
            "mae_gaba_mM": float(residual[:, 3].mean()),
            "failure_rate_gaba": float(failure_any.mean()),
        }

    results["failure_label_source"] = "derived from prediction residuals"
    results["tolerances"] = tolerances.tolist()
    results["unit"] = unit

    if "epistemic_uncertainty" in preds:
        uncertainty = _validate_array(
            "epistemic_uncertainty", preds["epistemic_uncertainty"], gt.shape
        )
        if modality == "mrf":
            for i, name in enumerate(target_names):
                results[f"{name}_failure_detection"] = _ranking_metrics(
                    failure_by_target[:, i], uncertainty[:, i]
                )
            results["failure_any_detection"] = _ranking_metrics(
                failure_any, uncertainty.max(axis=1)
            )
        else:
            results["gaba_failure_detection"] = _ranking_metrics(
                failure_any, uncertainty[:, 3]
            )

    if "attribution" in preds:
        attr = np.asarray(preds["attribution"], dtype=float)
        if attr.shape != flags.shape:
            raise ValueError(f"attribution shape {attr.shape} != expected {flags.shape}")
        if not np.isfinite(attr).all() or (attr < 0).any() or (attr > 1).any():
            raise ValueError("attribution must contain finite probabilities in [0, 1]")
        pred_binary = (attr >= 0.5).astype(np.uint8)
        results["attribution_exact_match"] = float(
            np.all(pred_binary == flags, axis=1).mean()
        )
        results["attribution_macro_f1"] = float(
            f1_score(flags, pred_binary, average="macro", zero_division=0)
        )
        results["attribution_micro_f1"] = float(
            f1_score(flags, pred_binary, average="micro", zero_division=0)
        )
        results["attribution_per_source"] = {}
        for i, name in enumerate(["b0", "b1", "motion"]):
            results["attribution_per_source"][name] = {
                "precision": float(precision_score(flags[:, i], pred_binary[:, i], zero_division=0)),
                "recall": float(recall_score(flags[:, i], pred_binary[:, i], zero_division=0)),
                "f1": float(f1_score(flags[:, i], pred_binary[:, i], zero_division=0)),
            }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("benchmark_dir", nargs="?", default=".")
    parser.add_argument("--modality", choices=["mrf", "mrs"], default="mrf")
    args = parser.parse_args()
    evaluate(args.predictions, args.benchmark_dir, args.modality)
