"""Audit exact reconstruction of synthetic v3 MRF rows from provenance metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qMR_Robust.simulators.manager import reconstruct_mrf_sample_from_metadata


def audit(h5_path: Path, config_path: Path, n_samples: int = 32, atol: float = 1e-6) -> dict:
    config = yaml.safe_load(config_path.read_text())
    sim_cfg = config["simulation"]
    with h5py.File(h5_path, "r") as hf:
        required = {
            "clean_signals",
            "parameters",
            "base_b0_hz",
            "base_b1_scale",
            "sim_parameters",
            "simulation_seed",
            "domain_id",
        }
        missing = sorted(required.difference(hf.keys()))
        if missing:
            raise ValueError(
                "HDF5 file is missing provenance datasets: " + ", ".join(missing)
            )
        n_total = int(hf.attrs["n_signals"])
        n_check = min(max(int(n_samples), 1), n_total)
        indices = np.linspace(0, n_total - 1, n_check, dtype=np.int64)
        max_errors = {
            "clean_signal": 0.0,
            "parameters": 0.0,
            "base_b0_hz": 0.0,
            "base_b1_scale": 0.0,
            "sim_parameters": 0.0,
        }
        failures = []
        for index in indices:
            sample = reconstruct_mrf_sample_from_metadata(
                sim_cfg,
                int(hf["simulation_seed"][index]),
                int(hf["domain_id"][index]),
            )
            comparisons = {
                "clean_signal": np.max(
                    np.abs(np.asarray(hf["clean_signals"][index]) - sample["signal"])
                ),
                "parameters": np.max(
                    np.abs(np.asarray(hf["parameters"][index]) - sample["params"])
                ),
                "base_b0_hz": abs(
                    float(hf["base_b0_hz"][index]) - float(sample["base_b0_hz"])
                ),
                "base_b1_scale": abs(
                    float(hf["base_b1_scale"][index]) - float(sample["base_b1_scale"])
                ),
                "sim_parameters": np.max(
                    np.abs(
                        np.asarray(hf["sim_parameters"][index])
                        - np.asarray(
                            [sample["t1_sim_ms"], sample["t2_sim_ms"]],
                            dtype=np.float32,
                        )
                    )
                ),
            }
            for name, error in comparisons.items():
                max_errors[name] = max(max_errors[name], float(error))
                if float(error) > atol:
                    failures.append({"index": int(index), "field": name, "error": float(error)})
        return {
            "h5": str(h5_path),
            "config": str(config_path),
            "n_total": n_total,
            "n_checked": n_check,
            "atol": float(atol),
            "max_abs_error": max_errors,
            "passed": not failures,
            "failures": failures[:20],
            "failure_count": len(failures),
            "interpretation": (
                "Stored clean signals and simulator metadata reproduce exactly "
                "for the audited rows; this does not validate external scanner data."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=ROOT / "data/synthetic/failure_forecast_mrf_v3.h5",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/config.yaml",
    )
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance; float32 metadata is stored with finite precision.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = audit(args.h5, args.config, args.n_samples, args.atol)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + chr(10))
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
