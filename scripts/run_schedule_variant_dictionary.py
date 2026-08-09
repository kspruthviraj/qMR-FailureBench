#!/usr/bin/env python3
"""Evaluate a schedule-variant matched classical MRF dictionary baseline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qMR_Robust.baselines.dictionary_match import (
    build_dictionary,
    default_grids,
    match_signals,
)
from qMR_Robust.simulators.manager import _generate_fa_schedule, _generate_tr_schedule


def evaluate(h5_path: Path, output_path: Path) -> dict:
    with h5py.File(h5_path, "r") as hf:
        signals = np.asarray(hf["corrupted_signals"][:], dtype=np.complex64)
        params = np.asarray(hf["parameters"][:, :2], dtype=np.float32)
        fa_ids = np.asarray(hf["fa_schedule_variant"][:], dtype=np.int16)
        tr_ids = np.asarray(hf["tr_schedule_variant"][:], dtype=np.int16)

    t1_grid, t2_grid = default_grids(coarse=True)
    predictions = np.empty((len(signals), 2), dtype=np.float32)
    scores = np.empty(len(signals), dtype=np.float32)
    per_schedule = {}
    for fa_id, tr_id in sorted(set(zip(fa_ids.tolist(), tr_ids.tolist()))):
        indices = np.flatnonzero((fa_ids == fa_id) & (tr_ids == tr_id))
        rng = np.random.RandomState(1000 + 31 * int(fa_id) + int(tr_id))
        fa = _generate_fa_schedule(int(fa_id), signals.shape[1], rng)
        tr = _generate_tr_schedule(int(tr_id), signals.shape[1], rng)
        dictionary, t1_dict, t2_dict = build_dictionary(
            t1_grid, t2_grid, fa, tr
        )
        matched = match_signals(
            signals[indices], dictionary, t1_dict, t2_dict, batch=128
        )
        predictions[indices, 0] = matched["t1"]
        predictions[indices, 1] = matched["t2"]
        scores[indices] = matched["match_score"]
        per_schedule[f"fa{fa_id}_tr{tr_id}"] = {
            "n_samples": int(len(indices)),
            "dictionary_size": int(len(dictionary)),
            "mean_match_score": float(matched["match_score"].mean()),
        }

    residual = np.abs(params - predictions)
    max_residual = residual.max(axis=1)
    uncertainty = 1.0 - scores
    labels = (max_residual > 300.0).astype(np.uint8)
    rho = spearmanr(uncertainty, max_residual).statistic
    result = {
        "method": "dictionary_matching_schedule_variant",
        "n_eval": int(len(signals)),
        "mae_ms": float(residual.mean()),
        "mae_t1_ms": float(residual[:, 0].mean()),
        "mae_t2_ms": float(residual[:, 1].mean()),
        "rmse_ms": float(np.sqrt(np.mean(residual ** 2))),
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "failure_rate": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, uncertainty))
        if labels.min() != labels.max()
        else None,
        "mean_match_score": float(scores.mean()),
        "schedule_variants": int(len(per_schedule)),
        "per_schedule": per_schedule,
        "note": (
            "Dictionary is matched to FA/TR variant IDs using deterministic "
            "representative schedules. It is not exact per-sample schedule "
            "matching because the simulator adds per-sample schedule noise."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + chr(10))
    return result


def main() -> None:
    h5_path = ROOT / "qMR-FailureBench" / "mrf" / "mrf_benchmark.h5"
    output_path = ROOT / "results" / "figures" / "dictionary_baseline_schedule_variant_v3.json"
    result = evaluate(h5_path, output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
