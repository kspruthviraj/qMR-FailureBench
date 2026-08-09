#!/usr/bin/env python3
"""Audit qMR-FailureBench data and evaluation integrity without changing inputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _attrs(hf: h5py.File) -> dict:
    return {str(k): _scalar(v) for k, v in hf.attrs.items()}


def _labels(values: np.ndarray) -> np.ndarray:
    return np.asarray([_scalar(v) for v in values], dtype=object)


def _combination_counts(b0, b1, motion) -> dict:
    flags = np.stack([
        np.abs(b0) > 1.0,
        np.abs(b1 - 1.0) > 0.01,
        np.abs(motion) > 0.5,
    ], axis=1)
    names = ["b0_only", "b1_only", "motion_only", "b0_b1", "b0_motion", "b1_motion", "all_three"]
    masks = [
        flags[:, 0] & ~flags[:, 1] & ~flags[:, 2],
        ~flags[:, 0] & flags[:, 1] & ~flags[:, 2],
        ~flags[:, 0] & ~flags[:, 1] & flags[:, 2],
        flags[:, 0] & flags[:, 1] & ~flags[:, 2],
        flags[:, 0] & ~flags[:, 1] & flags[:, 2],
        ~flags[:, 0] & flags[:, 1] & flags[:, 2],
        flags.all(axis=1),
    ]
    return {name: int(mask.sum()) for name, mask in zip(names, masks)}


def audit_mrf(path: Path) -> dict:
    required = {
        "clean_signals",
        "corrupted_signals",
        "parameters",
        "b0_hz_applied",
        "b1_scale_applied",
        "motion_shift_applied",
    }
    with h5py.File(path, "r") as hf:
        keys = sorted(hf.keys())
        n = int(hf.attrs.get("n_signals", hf["parameters"].shape[0]))
        missing = sorted(required - set(keys))
        b0 = hf["b0_hz_applied"][:]
        b1 = hf["b1_scale_applied"][:]
        motion = hf["motion_shift_applied"][:]
        labels = _labels(hf["domain_labels"][:]) if "domain_labels" in hf else None
        sample = hf["corrupted_signals"][: min(n, 1000)]
        attrs = _attrs(hf)

    result = {
        "path": str(path),
        "n_samples": n,
        "keys": keys,
        "missing_required_keys": missing,
        "attrs": attrs,
        "combination_counts": _combination_counts(b0, b1, motion),
        "metadata_unique_counts": {
            "b0_hz": int(np.unique(b0).size),
            "b1_scale": int(np.unique(b1).size),
            "motion_shift": int(np.unique(motion).size),
        },
        "signal_magnitude_quantiles_first_1000": np.quantile(
            np.abs(sample), [0.0, 0.5, 0.95, 1.0]
        ).tolist(),
        "expected_provenance_attrs_missing": [
            key for key in ("t1_t2_units", "physics_version", "seed_scheme")
            if key not in attrs
        ],
        "has_schedule_reconstruction_metadata": all(
            key in keys
            for key in (
                "simulation_seed",
                "fa_schedule_variant",
                "tr_schedule_variant",
                "domain_id",
            )
        ),
    }

    if labels is not None:
        cut = int(n * 0.8)
        train_labels = np.unique(labels[:cut])
        val_labels = np.unique(labels[cut:])
        result["domain_count"] = int(np.unique(labels).size)
        result["legacy_ordered_split"] = {
            "train_samples": cut,
            "validation_samples": n - cut,
            "train_domains": int(train_labels.size),
            "validation_domains": int(val_labels.size),
            "domain_overlap": int(np.intersect1d(train_labels, val_labels).size),
            "train_label_examples": [str(x) for x in train_labels[:8]],
            "validation_label_examples": [str(x) for x in val_labels[:8]],
        }
        vendors = np.asarray([str(x).split("_", 1)[0] for x in labels], dtype=object)
        fields = np.asarray([
            (re.search(r"_([0-9.]+)T$", str(x)) or [None, "unknown"])[1]
            for x in labels
        ], dtype=object)
        result["legacy_ordered_split"]["train_vendor_counts"] = {
            str(k): int(v) for k, v in zip(*np.unique(vendors[:cut], return_counts=True))
        }
        result["legacy_ordered_split"]["validation_vendor_counts"] = {
            str(k): int(v) for k, v in zip(*np.unique(vendors[cut:], return_counts=True))
        }
        result["legacy_ordered_split"]["train_field_counts"] = {
            str(k): int(v) for k, v in zip(*np.unique(fields[:cut], return_counts=True))
        }
        result["legacy_ordered_split"]["validation_field_counts"] = {
            str(k): int(v) for k, v in zip(*np.unique(fields[cut:], return_counts=True))
        }
    else:
        result["domain_count"] = None
        result["legacy_ordered_split"] = {
            "warning": "domain_labels missing; domain-grouped evaluation is unavailable"
        }
    return result


def audit_mrs(path: Path) -> dict:
    with h5py.File(path, "r") as hf:
        conc = hf["concentrations"][:]
        b0 = hf["b0_hz_applied"][:]
        b1 = hf["b1_scale_applied"][:]
        motion = hf["motion_shift_applied"][:]
        result = {
            "path": str(path),
            "n_samples": int(hf.attrs.get("n_signals", len(conc))),
            "attrs": _attrs(hf),
            "keys": sorted(hf.keys()),
            "unique_counts": {
                "b0_hz": int(np.unique(b0).size),
                "b1_scale": int(np.unique(b1).size),
                "motion_shift": int(np.unique(motion).size),
                "gaba": int(np.unique(conc[:, 3]).size),
            },
            "gaba_range": [float(conc[:, 3].min()), float(conc[:, 3].max())],
        }
    warnings = []
    if result["unique_counts"]["b0_hz"] <= 1:
        warnings.append("B0 metadata is constant or absent")
    if result["unique_counts"]["b1_scale"] <= 1:
        warnings.append("B1 metadata is constant or absent")
    if result["unique_counts"]["motion_shift"] <= 1:
        warnings.append("motion metadata is constant or absent")
    if np.isclose(*result["gaba_range"]):
        warnings.append("GABA concentration is constant")
    result["release_ready"] = not warnings
    result["warnings"] = warnings
    return result


def audit_vfa(path: Path) -> dict:
    try:
        from qMR_Robust.data.loaders import load_qmrlab_vfa

        data = load_qmrlab_vfa(path, pad_mode="zeropad")
        signal = data.signals
        tail = signal[:, :, data.n_fa_original:]
        return {
            "path": str(path),
            "status": "loaded",
            "n_voxels": int(len(data.t1_ms)),
            "signal_shape": list(signal.shape),
            "protocol": data.protocol,
            "n_fa_original": int(data.n_fa_original),
            "t1_ms_mean": float(data.t1_ms.mean()),
            "normalized_signal_max_range": [
                float(np.abs(signal).max(axis=(1, 2)).min()),
                float(np.abs(signal).max(axis=(1, 2)).max()),
            ],
            "nonzero_tail_fraction": float(np.count_nonzero(tail) / max(tail.size, 1)),
        }
    except Exception as exc:
        return {"path": str(path), "status": "error", "error": str(exc)}


def audit_benchmark(path: Path) -> dict:
    result = {"path": str(path), "status": "missing"}
    metadata_path = path / "metadata.json"
    mrf_path = path / "mrf" / "mrf_benchmark.h5"
    mrs_path = path / "mrs" / "mrs_benchmark.h5"
    if metadata_path.exists():
        result["metadata"] = json.loads(metadata_path.read_text())
    if mrf_path.exists():
        with h5py.File(mrf_path, "r") as hf:
            result["mrf_keys"] = sorted(hf.keys())
            result["mrf_failure_labels_available"] = bool(
                hf.attrs.get("failure_labels_available", "failure_t1" in hf)
            )
    if mrs_path.exists():
        with h5py.File(mrs_path, "r") as hf:
            result["mrs_keys"] = sorted(hf.keys())
            result["mrs_failure_labels_available"] = bool(
                hf.attrs.get("failure_labels_available", False)
            )
    if metadata_path.exists() or mrf_path.exists() or mrs_path.exists():
        result["status"] = "present"
    return result


def audit_source(root: Path) -> dict:
    hazards = []
    for path in root.rglob("*.py"):
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(errors="replace")
        if "hash(" in text and "stable_seed" not in text:
            hazards.append(str(path.relative_to(root)))
    requirement_files = [root / "requirements.lock", root / "requirements.txt"]
    pinned_requirement_files = [
        path
        for path in requirement_files
        if path.exists()
        and any(
            "==" in line and not line.strip().startswith("#")
            for line in path.read_text().splitlines()
        )
    ]
    return {
        "builtin_hash_seed_hazards": sorted(hazards),
        "requirements_are_pinned": bool(pinned_requirement_files),
        "pinned_requirement_files": [
            str(path.relative_to(root)) for path in pinned_requirement_files
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrf", type=Path, default=root / "data/synthetic/failure_forecast_mrf.h5")
    parser.add_argument("--mrf-v3", type=Path, default=root / "data/synthetic/failure_forecast_mrf_v3.h5")
    parser.add_argument("--mrs", type=Path, default=root / "data/synthetic/failure_forecast_mrs.h5")
    parser.add_argument("--vfa", type=Path, default=root / "data/real/qmrlab/vfa_t1_data")
    parser.add_argument("--benchmark", type=Path, default=root / "qMR-FailureBench")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "audit_version": "scientific_integrity_v1",
        "project_root": str(root),
        "mrf": audit_mrf(args.mrf) if args.mrf.exists() else {"status": "missing", "path": str(args.mrf)},
        "mrf_v3": audit_mrf(args.mrf_v3) if args.mrf_v3.exists() else {"status": "missing", "path": str(args.mrf_v3)},
        "mrs": audit_mrs(args.mrs) if args.mrs.exists() else {"status": "missing", "path": str(args.mrs)},
        "real_vfa": audit_vfa(args.vfa) if args.vfa.exists() else {"status": "missing", "path": str(args.vfa)},
        "benchmark": audit_benchmark(args.benchmark),
        "source": audit_source(root),
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
