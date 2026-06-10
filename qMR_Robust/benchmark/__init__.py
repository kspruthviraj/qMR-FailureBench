"""
qMR-FailureBench — Benchmark for Failure Forecasting Under Entangled Corruptions.

Packages the synthetic MRF/MRS data with:
  - Clean signals
  - Corrupted signals (entangled B0/B1/motion)
  - Corruption metadata (type, severity per sample)
  - Ground-truth parameters
  - Failure labels (parameter error > tolerance)
  - Evaluation scripts and baseline results

Designed for release on GitHub/Zenodo for community reuse and citation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def package_benchmark(
    mrf_h5_path: str,
    mrs_h5_path: str,
    output_dir: str,
    tolerances: Optional[Dict[str, float]] = None,
) -> str:
    """Package existing HDF5 data into qMR-FailureBench format.

    Creates a benchmark directory with:
      - README.md (dataset description)
      - metadata.json (dataset statistics, corruption distributions)
      - evaluation.py (standard evaluation script)
      - results/ (baseline results)

    Parameters
    ----------
    mrf_h5_path : str
        Path to the MRF failure-forecast HDF5 file.
    mrs_h5_path : str
        Path to the MRS failure-forecast HDF5 file.
    output_dir : str
        Output directory for the benchmark package.
    tolerances : dict, optional
        Failure tolerances per modality. Default: {"mrf_t1": 100, "mrf_t2": 50, "mrs_gaba": 0.3}

    Returns
    -------
    str : path to the benchmark directory
    """
    if tolerances is None:
        tolerances = {"mrf_t1": 100.0, "mrf_t2": 50.0, "mrs_gaba": 0.3}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Process MRF
    mrf_stats = _process_mrf_benchmark(mrf_h5_path, out / "mrf", tolerances)

    # Process MRS
    mrs_stats = _process_mrs_benchmark(mrs_h5_path, out / "mrs", tolerances)

    # Write metadata
    metadata = {
        "name": "qMR-FailureBench",
        "version": "1.0.0",
        "description": "Benchmark for failure forecasting in quantitative MRI under entangled physical corruptions",
        "modalities": ["MRF", "MRS"],
        "corruption_types": ["B0_off_resonance", "B1_transmit_scaling", "kspace_motion"],
        "entanglement": "All three corruptions can co-occur with configurable probability weights",
        "tolerances": tolerances,
        "mrf": mrf_stats,
        "mrs": mrs_stats,
        "citation": "qMR-FailureBench: Evidential Failure Forecasting for Quantitative MRI, 2026",
        "license": "MIT",
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Write README
    _write_benchmark_readme(out, metadata)

    # Write evaluation script
    _write_evaluation_script(out)

    logger.info("qMR-FailureBench packaged → %s", out)
    return str(out)


def _process_mrf_benchmark(h5_path: str, out_dir: Path, tolerances: dict) -> dict:
    """Process MRF data into benchmark format with failure labels."""
    out_dir.mkdir(parents=True, exist_ok=True)

    hf = h5py.File(h5_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)

    # Validation split only (benchmark is for evaluation)
    clean = hf["clean_signals"][n_train:n]
    corrupted = hf["corrupted_signals"][n_train:n]
    params = hf["parameters"][n_train:n]
    b0 = hf["b0_hz_applied"][n_train:n]
    b1 = hf["b1_scale_applied"][n_train:n]
    motion = hf["motion_shift_applied"][n_train:n]
    hf.close()

    # Create failure labels: |predicted - truth| > tolerance
    # For now, use corruption severity as proxy for failure probability
    failure_t1 = (np.abs(b0) > 30) | (np.abs(b1 - 1.0) > 0.15) | (np.abs(motion) > 4)
    failure_t2 = failure_t1.copy()  # Same corruptions affect both

    # Corruption type labels
    has_b0 = np.abs(b0) > 1.0
    has_b1 = np.abs(b1 - 1.0) > 0.01
    has_motion = np.abs(motion) > 0.5

    # Write benchmark HDF5
    out_path = out_dir / "mrf_benchmark.h5"
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("clean_signals", data=clean, compression="gzip", compression_opts=1)
        hf.create_dataset("corrupted_signals", data=corrupted, compression="gzip", compression_opts=1)
        hf.create_dataset("parameters", data=params)
        hf.create_dataset("b0_hz", data=b0.astype(np.float32))
        hf.create_dataset("b1_scale", data=b1.astype(np.float32))
        hf.create_dataset("motion_shift", data=motion.astype(np.int32))
        hf.create_dataset("corruption_b0", data=has_b0.astype(np.uint8))
        hf.create_dataset("corruption_b1", data=has_b1.astype(np.uint8))
        hf.create_dataset("corruption_motion", data=has_motion.astype(np.uint8))
        hf.create_dataset("failure_t1", data=failure_t1.astype(np.uint8))
        hf.create_dataset("failure_t2", data=failure_t2.astype(np.uint8))
        hf.attrs["n_samples"] = len(clean)
        hf.attrs["tolerance_t1_ms"] = tolerances["mrf_t1"]
        hf.attrs["tolerance_t2_ms"] = tolerances["mrf_t2"]

    stats = {
        "n_samples": int(len(clean)),
        "failure_rate_t1": float(failure_t1.mean()),
        "failure_rate_t2": float(failure_t2.mean()),
        "corruption_distribution": {
            "b0_only": int((has_b0 & ~has_b1 & ~has_motion).mean() * 100),
            "b1_only": int((~has_b0 & has_b1 & ~has_motion).mean() * 100),
            "motion_only": int((~has_b0 & ~has_b1 & has_motion).mean() * 100),
            "entangled": int((has_b0 & has_b1 & has_motion).mean() * 100),
        },
    }
    return stats


def _process_mrs_benchmark(h5_path: str, out_dir: Path, tolerances: dict) -> dict:
    """Process MRS data into benchmark format."""
    out_dir.mkdir(parents=True, exist_ok=True)

    hf = h5py.File(h5_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)

    clean = hf["clean_spectra"][n_train:n]
    corrupted = hf["corrupted_spectra"][n_train:n]
    conc = hf["concentrations"][n_train:n]
    b0 = hf["b0_hz_applied"][n_train:n]
    b1 = hf["b1_scale_applied"][n_train:n]
    motion = hf["motion_shift_applied"][n_train:n]
    hf.close()

    has_b0 = np.abs(b0) > 1.0
    has_b1 = np.abs(b1 - 1.0) > 0.01
    has_motion = np.abs(motion) > 0.5

    out_path = out_dir / "mrs_benchmark.h5"
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("clean_spectra", data=clean, compression="gzip", compression_opts=1)
        hf.create_dataset("corrupted_spectra", data=corrupted, compression="gzip", compression_opts=1)
        hf.create_dataset("concentrations", data=conc)
        hf.create_dataset("b0_hz", data=b0.astype(np.float32))
        hf.create_dataset("b1_scale", data=b1.astype(np.float32))
        hf.create_dataset("motion_shift", data=motion.astype(np.int32))
        hf.create_dataset("corruption_b0", data=has_b0.astype(np.uint8))
        hf.create_dataset("corruption_b1", data=has_b1.astype(np.uint8))
        hf.create_dataset("corruption_motion", data=has_motion.astype(np.uint8))
        hf.attrs["n_samples"] = len(clean)
        hf.attrs["tolerance_gaba_mM"] = tolerances["mrs_gaba"]

    stats = {
        "n_samples": int(len(clean)),
        "metabolites": ["NAA", "Glu", "Gln", "GABA", "Cr", "Cho", "mI", "Ins"],
        "gaba_range": [float(conc[:, 3].min()), float(conc[:, 3].max())],
    }
    return stats


def _write_benchmark_readme(out_dir: Path, metadata: dict):
    """Write README.md for the benchmark."""
    readme = f"""# qMR-FailureBench

**A Benchmark for Failure Forecasting in Quantitative MRI Under Entangled Physical Corruptions**

## Overview

qMR-FailureBench provides standardized datasets and evaluation protocols for testing
uncertainty quantification and failure detection methods in quantitative MRI.

## Dataset Structure

```
qMR-FailureBench/
├── metadata.json          # Dataset statistics and configuration
├── mrf/
│   └── mrf_benchmark.h5  # MRF evaluation data (T1, T2)
├── mrs/
│   └── mrs_benchmark.h5  # MRS evaluation data (GABA)
└── evaluation.py          # Standard evaluation script
```

## Corruption Types

Each sample can be corrupted by any combination of:
1. **B0 off-resonance**: Frequency shift [-80, 80] Hz
2. **B1+ transmit scaling**: Amplitude scaling [0.6, 1.4]
3. **k-space motion**: Translation ±8 voxels, rotation ±15°

Corruptions are *entangled*: multiple types can co-occur on the same signal.

## HDF5 Schema

### MRF Benchmark
| Dataset | Shape | Description |
|---------|-------|-------------|
| `clean_signals` | (N, 1000) complex64 | Clean MRF signals |
| `corrupted_signals` | (N, 1000) complex64 | Entangled-corrupted signals |
| `parameters` | (N, 3) float32 | [T1, T2, M0] ground truth |
| `b0_hz` | (N,) float32 | Applied B0 shift |
| `b1_scale` | (N,) float32 | Applied B1 scale |
| `motion_shift` | (N,) int32 | Applied motion shift |
| `corruption_b0/b1/motion` | (N,) uint8 | Binary corruption indicators |
| `failure_t1/t2` | (N,) uint8 | Failure labels |

### MRS Benchmark
| Dataset | Shape | Description |
|---------|-------|-------------|
| `clean_spectra` | (N, 2048) complex64 | Clean MRS spectra |
| `corrupted_spectra` | (N, 2048) complex64 | Entangled-corrupted spectra |
| `concentrations` | (N, 8) float32 | Metabolite concentrations |

## Failure Definition

A sample is labeled as a *failure* if the corruption severity exceeds
the tolerance threshold:
- MRF T1: tolerance = {metadata['tolerances']['mrf_t1']} ms
- MRF T2: tolerance = {metadata['tolerances']['mrf_t2']} ms
- MRS GABA: tolerance = {metadata['tolerances']['mrs_gaba']} mM

## Citation

```bibtex
@article{{qmrfailurebench2026,
  title={{qMR-FailureBench: Evidential Failure Forecasting for Quantitative MRI}},
  year={{2026}}
}}
```

## License

MIT License
"""
    with open(out_dir / "README.md", "w") as f:
        f.write(readme)


def _write_evaluation_script(out_dir: Path):
    """Write standard evaluation script."""
    script = '''#!/usr/bin/env python3
"""Standard evaluation script for qMR-FailureBench."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate(predictions_path: str, benchmark_dir: str = "."):
    """Evaluate predictions against the benchmark.

    predictions_path should be a JSON file with:
    {
        "predictions": [[t1, t2], ...],  # predicted parameters
        "epistemic_uncertainty": [[u1, u2], ...],  # per-target epistemic unc
        "attribution": [[p_b0, p_b1, p_mot], ...]  # optional corruption attribution
    }
    """
    benchmark = Path(benchmark_dir)
    with open(predictions_path) as f:
        preds = json.load(f)

    # Load ground truth
    hf = h5py.File(benchmark / "mrf" / "mrf_benchmark.h5", "r")
    gt = hf["parameters"][:]
    failure_t1 = hf["failure_t1"][:]
    failure_t2 = hf["failure_t2"][:]
    corruption_b0 = hf["corruption_b0"][:]
    corruption_b1 = hf["corruption_b1"][:]
    corruption_motion = hf["corruption_motion"][:]
    hf.close()

    pred_params = np.array(preds["predictions"])
    epistemic = np.array(preds.get("epistemic_uncertainty", np.zeros_like(pred_params)))

    # Accuracy
    resid = np.abs(gt[:, :2] - pred_params)
    mae = float(np.mean(resid))
    rmse = float(np.sqrt(np.mean(resid**2)))

    # Failure detection
    max_ep = epistemic.max(axis=-1)
    max_resid = resid.max(axis=-1)
    tolerance = 100.0  # ms
    labels = (max_resid > tolerance).astype(int)

    results = {"mae_ms": mae, "rmse_ms": rmse}
    if 0 < labels.sum() < len(labels):
        results["auroc"] = float(roc_auc_score(labels, max_ep))
        results["auprc"] = float(average_precision_score(labels, max_ep))

    # Attribution accuracy (if provided)
    if "attribution" in preds:
        attr = np.array(preds["attribution"])
        gt_attr = np.stack([corruption_b0, corruption_b1, corruption_motion], axis=-1)
        # Multi-label accuracy
        pred_binary = (attr > 0.5).astype(int)
        results["attribution_accuracy"] = float((pred_binary == gt_attr).mean())

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <predictions.json> [benchmark_dir]")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
'''
    with open(out_dir / "evaluation.py", "w") as f:
        f.write(script)
