#!/usr/bin/env python3
"""Add exact per-sample simulator seeds and schedule IDs to a v3 MRF file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qMR_Robust.reproducibility import stable_seed

def augment(path: Path, config_path: Path, base_seed: int = 42) -> None:
    config = yaml.safe_load(config_path.read_text())
    cfg = config["simulation"]["mrf"]
    vendors = cfg.get("vendors", ["siemens", "philips", "ge"])
    fields = cfg.get("field_strengths", [1.5, 3.0, 7.0])
    fa_vars = int(cfg.get("fa_schedule_variants", 5))
    tr_vars = int(cfg.get("tr_schedule_variants", 3))
    combos = [
        (vendor, field, fa_var, tr_var)
        for vendor in vendors
        for field in fields
        for fa_var in range(fa_vars)
        for tr_var in range(tr_vars)
    ]
    with h5py.File(path, "r+") as hf:
        n = int(hf.attrs["n_signals"])
        if n % len(combos):
            raise ValueError("n_signals is not an integer number of complete domains")
        per_domain = n // len(combos)
        seeds = np.empty(n, dtype=np.int64)
        fa_ids = np.empty(n, dtype=np.int16)
        tr_ids = np.empty(n, dtype=np.int16)
        domain_ids = np.empty(n, dtype=np.int16)
        for domain_id, (vendor, field, fa_var, tr_var) in enumerate(combos):
            start = domain_id * per_domain
            stop = start + per_domain
            for k, index in enumerate(range(start, stop)):
                seeds[index] = stable_seed(base_seed, vendor, field, fa_var, tr_var, k)
                fa_ids[index] = fa_var
                tr_ids[index] = tr_var
                domain_ids[index] = domain_id
        for name in ("simulation_seed", "fa_schedule_variant", "tr_schedule_variant", "domain_id"):
            if name in hf:
                del hf[name]
        hf.create_dataset("simulation_seed", data=seeds)
        hf.create_dataset("fa_schedule_variant", data=fa_ids)
        hf.create_dataset("tr_schedule_variant", data=tr_ids)
        hf.create_dataset("domain_id", data=domain_ids)
        hf.attrs["base_seed"] = int(base_seed)
        hf.attrs["schedule_reconstruction"] = (
            "Use simulation_seed with manager._generate_mrf_sample; exact per-sample "
            "FA/TR arrays are generated after the tissue-parameter RNG draws."
        )
        hf.attrs["schedule_variants"] = f"fa={fa_vars},tr={tr_vars}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=ROOT / "data/synthetic/failure_forecast_mrf_v3.h5")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yaml")
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    augment(args.h5, args.config, args.base_seed)
    print("Added schedule reconstruction metadata to", args.h5)


if __name__ == "__main__":
    main()
