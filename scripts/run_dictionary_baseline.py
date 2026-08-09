#!/usr/bin/env python3
"""Classical dictionary-matching baseline on synthetic entangled MRF val set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.baselines.dictionary_match import (
    build_dictionary,
    match_signals,
    default_grids,
)
from qMR_Robust.simulators.manager import _generate_fa_schedule, _generate_tr_schedule

FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
    mrf_path = ROOT / cfg["paths"]["failure_forecast_mrf"]

    hf = h5py.File(mrf_path, "r")
    n = int(hf.attrs["n_signals"])
    n_train = int(n * 0.8)
    # Use a manageable val subset
    n_eval = min(1500, n - n_train)
    sigs = hf["corrupted_signals"][n_train : n_train + n_eval]
    params = hf["parameters"][n_train : n_train + n_eval, :2].astype(np.float32)
    hf.close()

    # Dictionary under nominal FA/TR variant 0 (representative schedule)
    rng = np.random.RandomState(0)
    n_tp = sigs.shape[1]
    fa = _generate_fa_schedule(0, n_tp, rng)
    tr = _generate_tr_schedule(0, n_tp, rng)
    t1_grid, t2_grid = default_grids(coarse=True)
    print(f"Building dictionary: T1={len(t1_grid)} T2={len(t2_grid)} pts...")
    D, t1d, t2d = build_dictionary(t1_grid, t2_grid, fa, tr)
    print(f"  Dictionary size M={len(D)}")

    print(f"Matching n={n_eval} val signals...")
    out = match_signals(sigs.astype(np.complex64), D, t1d, t2d, batch=128)
    err_t1 = np.abs(params[:, 0] - out["t1"])
    err_t2 = np.abs(params[:, 1] - out["t2"])
    mae = float(np.mean(np.maximum(err_t1, err_t2)))
    mae_t1 = float(err_t1.mean())
    # Use 1 - match_score as uncertainty proxy (lower match → higher unc)
    unc = 1.0 - out["match_score"]
    resid = np.maximum(err_t1, err_t2)
    rho, _ = spearmanr(unc, resid) if unc.std() > 1e-12 else (0.0, 1.0)
    labels = (resid > 300).astype(int)
    auroc = float(roc_auc_score(labels, unc)) if 0 < labels.sum() < n_eval else float("nan")

    results = {
        "method": "dictionary_matching",
        "n_eval": n_eval,
        "dictionary_size": int(len(D)),
        "mae_ms": mae,
        "mae_t1_ms": mae_t1,
        "mae_t2_ms": float(err_t2.mean()),
        "spearman_rho": float(rho),
        "auroc": auroc,
        "mean_match_score": float(out["match_score"].mean()),
        "note": (
            "Coarse dictionary under single FA/TR schedule; not schedule-matched per sample. "
            "Uncertainty proxy = 1 - correlation match score."
        ),
    }
    path = FIG / "dictionary_baseline.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("Wrote", path)


if __name__ == "__main__":
    main()
