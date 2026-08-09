#!/usr/bin/env python3
"""Generate versioned failure-forecast datasets without overwriting legacy files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qMR_Robust.simulators import PhysicsCorruptor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "config.yaml")
    parser.add_argument("--modality", choices=("mrf", "mrs", "both"), default="both")
    parser.add_argument("--n-signals", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--mrf-output", type=Path, default=ROOT / "data/synthetic/failure_forecast_mrf_v3.h5")
    parser.add_argument("--mrs-output", type=Path, default=ROOT / "data/synthetic/failure_forecast_mrs_v3.h5")
    args = parser.parse_args()

    with args.config.open() as handle:
        cfg = yaml.safe_load(handle)

    if args.workers is not None:
        cfg["simulation"]["mrf"]["n_workers"] = args.workers
        cfg["simulation"]["mrs"]["n_workers"] = args.workers

    corruptor = PhysicsCorruptor(cfg)
    if args.modality in ("mrf", "both"):
        corruptor.generate_failure_forecast_mrf(str(args.mrf_output), args.n_signals)
    if args.modality in ("mrs", "both"):
        corruptor.generate_failure_forecast_mrs(str(args.mrs_output), args.n_signals)


if __name__ == "__main__":
    main()
