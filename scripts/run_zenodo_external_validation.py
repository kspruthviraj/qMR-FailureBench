#!/usr/bin/env python3
"""Reproducible audit of the two public Zenodo datasets.

The 8234101 release contains final MRF-derived maps, not raw MRF fingerprints.
It is therefore used only for an unregistered scan--rescan distribution audit.
The 8419809 release contains four-point inversion-recovery data and processed
T1 maps.  It is evaluated as an explicitly cross-sequence, zero-padded OOD
stress test; it is not independent real-MRF validation and has no acquisition
failure labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
MODEL_SEQ_LEN = 1000
FAILURE_TOLERANCE_MS = 300.0
QUANTILES = np.asarray([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)

ARCHIVES = {
    "zenodo_8234101_mrf_maps": {
        "record": "https://zenodo.org/records/8234101",
        "file": "analysis_images_share.zip",
        "path": ROOT / "data/external/zenodo_8234101/analysis_images_share.zip",
        "expected_md5": "f8563b61503542cffdd22ed8e4c46f67",
        "size_bytes": 3831813910,
    },
    "zenodo_8419809_nist_phantom": {
        "record": "https://zenodo.org/records/8419809",
        "file": "Dataset_10.55458_NeuroLibre_00014_2654c1.zip",
        "path": ROOT
        / "data/external/zenodo_8419809/Dataset_10.55458_NeuroLibre_00014_2654c1.zip",
        "expected_md5": "4742c9ef2d05bbad9802c1404fea3495",
        "size_bytes": 217142333,
    },
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_audit() -> dict[str, dict[str, Any]]:
    result = {}
    for name, spec in ARCHIVES.items():
        path = Path(spec["path"])
        if not path.exists():
            result[name] = {"path": str(path), "present": False}
            continue
        actual_size = path.stat().st_size
        actual_md5 = _md5(path)
        result[name] = {
            "record": spec["record"],
            "file": spec["file"],
            "path": str(path),
            "present": True,
            "size_bytes": int(actual_size),
            "published_size_bytes": int(spec["size_bytes"]),
            "md5": actual_md5,
            "published_md5": spec["expected_md5"],
            "verified": actual_size == spec["size_bytes"]
            and actual_md5 == spec["expected_md5"],
        }
    return result


def _map_stats(path: Path, low: float, high: float) -> dict[str, Any]:
    values = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32)).ravel()
    values = values[np.isfinite(values) & (values >= low) & (values <= high)]
    if values.size == 0:
        raise ValueError(f"no valid map voxels in {path}")
    q = np.quantile(values, QUANTILES)
    return {
        "n_voxels": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "quantiles_ms": [float(v) for v in q],
    }


def _mrf_scan_rescan_audit() -> dict[str, Any]:
    """Compare final-map distributions without pretending maps are registered."""
    root = (
        ROOT
        / "data/external/zenodo_8234101/extracted/analysis_images_share"
    )
    pairs = []
    skipped = []
    for subject in sorted(root.glob("sub_*")):
        for scanner in sorted(subject.glob("*_scanner_*")):
            acquisitions = {}
            for acq in (1, 2):
                folder = scanner / f"{scanner.name}_acq_{acq}" / "Unprocessed_NIFTI"
                t1_path = folder / "MRF_T1.nii.gz"
                t2_path = folder / "MRF_T2.nii.gz"
                if not (t1_path.exists() and t2_path.exists()):
                    skipped.append(
                        {"subject": subject.name, "scanner": scanner.name, "acq": acq}
                    )
                    continue
                acquisitions[acq] = {
                    "t1": _map_stats(t1_path, 10.0, 5000.0),
                    "t2": _map_stats(t2_path, 1.0, 2000.0),
                    "paths": {
                        "t1": str(t1_path.relative_to(ROOT)),
                        "t2": str(t2_path.relative_to(ROOT)),
                    },
                }
            if 1 not in acquisitions or 2 not in acquisitions:
                continue
            comparison = {
                "subject": subject.name,
                "scanner": scanner.name,
                "acq_1": acquisitions[1],
                "acq_2": acquisitions[2],
                "metrics": {},
            }
            for kind in ("t1", "t2"):
                q1 = np.asarray(acquisitions[1][kind]["quantiles_ms"])
                q2 = np.asarray(acquisitions[2][kind]["quantiles_ms"])
                diff = np.abs(q1 - q2)
                comparison["metrics"][kind] = {
                    "absolute_quantile_difference_ms": [float(v) for v in diff],
                    "mean_absolute_quantile_difference_ms": float(diff.mean()),
                    "median_difference_ms": float(q1[2] - q2[2]),
                }
            pairs.append(comparison)

    aggregate = {}
    for kind in ("t1", "t2"):
        diffs = np.asarray(
            [p["metrics"][kind]["absolute_quantile_difference_ms"] for p in pairs],
            dtype=np.float64,
        )
        scalar = np.asarray(
            [
                p["metrics"][kind][
                    "mean_absolute_quantile_difference_ms"
                ]
                for p in pairs
            ],
            dtype=np.float64,
        )
        aggregate[kind] = {
            "mean_absolute_quantile_difference_ms": float(scalar.mean())
            if scalar.size
            else None,
            "median_absolute_quantile_difference_ms": float(np.median(scalar))
            if scalar.size
            else None,
            "pairwise_quantile_difference_mean_ms": [
                float(v) for v in diffs.mean(axis=0)
            ]
            if diffs.size
            else [],
        }

    return {
        "dataset_role": "map_only_scan_rescan_distribution_audit",
        "direct_model_input_available": False,
        "voxelwise_comparison": False,
        "reason": "The release contains final maps but no raw MRF fingerprint time courses, and the scans are not spatially registered for this audit.",
        "quantiles": [float(v) for v in QUANTILES],
        "n_pairs": int(len(pairs)),
        "n_skipped_acquisitions": int(len(skipped)),
        "skipped": skipped,
        "aggregate": aggregate,
        "pairs": pairs,
    }


def _metadata_index() -> dict[str, Path]:
    root = ROOT / "data/external/zenodo_8419809/extracted/analysis/3T_NIST"
    return {
        path.name: path
        for path in root.rglob("*.json")
        if path.name != "data_requirement.json"
    }


def _metadata_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"sidecar_found": False}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"sidecar_found": False, "sidecar": str(path.relative_to(ROOT))}
    site = data.get("site", {})
    sample = data.get("sample", {})
    sequence = data.get("sequence", {})
    return {
        "sidecar_found": True,
        "sidecar": str(path.relative_to(ROOT)),
        "submitter": data.get("submitter", {}).get("contact"),
        "site": site.get("name"),
        "manufacturer": site.get("manufacturer"),
        "field_t": site.get("field"),
        "sample_serial_number": sample.get("serial_number"),
        "temperature_c": sample.get("temperature"),
        "sequence_type": sequence.get("type"),
        "inversion_times_ms": sequence.get("inversion_times"),
        "repetition_time_ms": sequence.get("repetition_time"),
        "echo_time_ms": sequence.get("echo_time"),
    }


def _nist_entries() -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_root = ROOT / "data/external/zenodo_8419809/extracted/analysis/3T_NIST_pooled"
    map_root = ROOT / "data/external/zenodo_8419809/extracted/analysis/3T_NIST_T1maps_pooled"
    sidecars = _metadata_index()
    entries = []
    magnitude_count = 0
    complex_count = 0

    for raw in sorted(raw_root.glob("*_Magnitude.nii.gz")):
        target = map_root / raw.name.replace(
            "_Magnitude.nii.gz", "_Magnitude_T1map.nii.gz"
        )
        entries.append(
            {
                "family": "magnitude",
                "raw": raw,
                "target": target,
                "metadata": _metadata_summary(
                    sidecars.get(raw.name.replace(".nii.gz", ".json"))
                ),
            }
        )
        magnitude_count += 1

    for real in sorted(raw_root.glob("*_Real.nii.gz")):
        imaginary = raw_root / real.name.replace("_Real.nii.gz", "_Imaginary.nii.gz")
        target = map_root / real.name.replace(
            "_Real.nii.gz", "_Complex_T1map.nii.gz"
        )
        entries.append(
            {
                "family": "complex",
                "raw": real,
                "imaginary": imaginary,
                "target": target,
                "metadata": _metadata_summary(
                    sidecars.get(real.name.replace("_Real.nii.gz", ".json"))
                ),
            }
        )
        complex_count += 1
    return entries, {"magnitude_raw": magnitude_count, "complex_real_raw": complex_count}


def _stable_record_seed(name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def _load_ir_record(
    entry: dict[str, Any], max_voxels: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.squeeze(nib.load(str(entry["raw"])).get_fdata(dtype=np.float32))
    if entry["family"] == "magnitude":
        curve = raw.astype(np.complex64)
    else:
        imaginary_path = entry["imaginary"]
        if not imaginary_path.exists():
            raise FileNotFoundError(f"missing imaginary companion: {imaginary_path}")
        imaginary = np.squeeze(nib.load(str(imaginary_path)).get_fdata(dtype=np.float32))
        if raw.shape != imaginary.shape:
            raise ValueError(f"real/imaginary shape mismatch: {raw.shape} vs {imaginary.shape}")
        curve = raw.astype(np.complex64) + 1j * imaginary.astype(np.complex64)

    target = np.squeeze(nib.load(str(entry["target"])).get_fdata(dtype=np.float32))
    if curve.ndim != 3:
        raise ValueError(f"expected X,Y,time data after squeeze, got {curve.shape}")
    if target.shape != curve.shape[:-1]:
        raise ValueError(f"raw/map shape mismatch: {curve.shape} vs {target.shape}")

    finite = np.isfinite(curve.real).all(axis=-1) & np.isfinite(curve.imag).all(axis=-1)
    finite &= np.isfinite(target)
    amplitude = np.max(np.abs(curve), axis=-1)
    valid = finite & (amplitude > 1e-8) & (target >= 10.0) & (target <= 5000.0)
    coords = np.flatnonzero(valid.ravel())
    if not len(coords):
        raise ValueError("no valid voxels after finite/signal/T1 filtering")

    if max_voxels > 0 and len(coords) > max_voxels:
        rng = np.random.default_rng(_stable_record_seed(entry["raw"].name))
        coords = np.sort(rng.choice(coords, size=max_voxels, replace=False))

    curves = curve.reshape(-1, curve.shape[-1])[coords]
    targets = target.ravel()[coords].astype(np.float32)
    n_time = min(curves.shape[-1], MODEL_SEQ_LEN)
    curves = curves[:, :n_time]
    scale = np.max(np.abs(curves), axis=1, keepdims=True).astype(np.float32)
    scale = np.maximum(scale, 1e-8)

    signals = np.zeros((len(curves), 2, MODEL_SEQ_LEN), dtype=np.float32)
    signals[:, 0, :n_time] = (curves.real / scale).astype(np.float32)
    signals[:, 1, :n_time] = (curves.imag / scale).astype(np.float32)
    metadata = {
        "family": entry["family"],
        "raw": str(entry["raw"].relative_to(ROOT)),
        "target_map": str(entry["target"].relative_to(ROOT)),
        "n_timepoints": int(n_time),
        "input_transform": "per-voxel complex amplitude normalization plus zero-padding to 1000 samples",
        "target_units": "milliseconds",
        "reference_kind": "challenge processed T1 map from the same four-point inversion-recovery acquisition",
        **entry["metadata"],
    }
    return signals, targets, metadata


def _scalar_scores(
    prediction: np.ndarray,
    target: np.ndarray,
    epistemic_norm: np.ndarray,
    train_std_t1: float,
) -> dict[str, Any]:
    residual = prediction - target
    absolute = np.abs(residual)
    failure = (absolute > FAILURE_TOLERANCE_MS).astype(np.uint8)
    score = np.asarray(epistemic_norm, dtype=np.float64)
    result: dict[str, Any] = {
        "n_voxels": int(len(target)),
        "mae_ms": float(absolute.mean()),
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "bias_ms": float(residual.mean()),
        "median_absolute_error_ms": float(np.median(absolute)),
        "failure_definition": "absolute T1 error > 300 ms relative to the supplied processed T1 map; this is an error-detection proxy, not an acquisition-QC label",
        "failure_rate": float(failure.mean()),
        "predicted_t1_mean_ms": float(prediction.mean()),
        "predicted_t1_std_ms": float(prediction.std()),
        "target_t1_mean_ms": float(target.mean()),
        "target_t1_std_ms": float(target.std()),
        "mean_epistemic_t1_std_proxy_ms": float(
            np.sqrt(np.maximum(score, 0.0)).mean() * float(train_std_t1)
        ),
        "residual_uncertainty_spearman": None,
        "auroc": None,
        "auprc": None,
    }
    rho = spearmanr(score, absolute)
    if np.isfinite(rho.statistic):
        result["residual_uncertainty_spearman"] = float(rho.statistic)
    if len(np.unique(failure)) == 2:
        result["auroc"] = float(roc_auc_score(failure, score))
        result["auprc"] = float(average_precision_score(failure, score))
    return result


def _predict_external(
    model: torch.nn.Module,
    signals: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    epistemic = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).to(device)
            out = model(batch)
            nig = out if out.ndim == 3 else out.view(out.shape[0], 2, 4)
            gamma = nig[..., 0]
            nu = nig[..., 1]
            alpha = nig[..., 2]
            beta = nig[..., 3]
            pred_t1 = gamma[:, 0].cpu().numpy() * float(std[0]) + float(mean[0])
            epistemic_t1 = (
                beta[:, 0]
                / (nu[:, 0] * (alpha[:, 0] - 1.0)).clamp(min=1e-8)
            ).cpu().numpy()
            predictions.append(pred_t1)
            epistemic.append(epistemic_t1)
    return np.concatenate(predictions), np.concatenate(epistemic)


def _nist_cross_sequence_evaluation(
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    max_voxels: int,
) -> dict[str, Any]:
    entries, inventory = _nist_entries()
    records = []
    skipped = []
    all_predictions = []
    all_targets = []
    all_epistemic = []
    for entry in entries:
        try:
            signals, targets, metadata = _load_ir_record(entry, max_voxels)
            prediction, epistemic = _predict_external(
                model, signals, mean, std, device
            )
            record = {
                **metadata,
                **_scalar_scores(prediction, targets, epistemic, float(std[0])),
            }
            records.append(record)
            all_predictions.append(prediction)
            all_targets.append(targets)
            all_epistemic.append(epistemic)
        except (FileNotFoundError, OSError, ValueError) as exc:
            skipped.append(
                {
                    "family": entry["family"],
                    "raw": str(entry["raw"].relative_to(ROOT)),
                    "target_map": str(entry["target"].relative_to(ROOT)),
                    "reason": str(exc),
                }
            )

    aggregate = None
    if all_predictions:
        aggregate = _scalar_scores(
            np.concatenate(all_predictions),
            np.concatenate(all_targets),
            np.concatenate(all_epistemic),
            float(std[0]),
        )
    by_family = {}
    for family in ("magnitude", "complex"):
        family_records = [r for r in records if r["family"] == family]
        if family_records:
            by_family[family] = {
                "n_records": len(family_records),
                "mean_mae_ms": float(np.mean([r["mae_ms"] for r in family_records])),
                "mean_failure_rate": float(
                    np.mean([r["failure_rate"] for r in family_records])
                ),
                "mean_spearman": float(
                    np.nanmean(
                        [
                            r["residual_uncertainty_spearman"]
                            if r["residual_uncertainty_spearman"] is not None
                            else np.nan
                            for r in family_records
                        ]
                    )
                ),
            }

    return {
        "dataset_role": "cross_sequence_ood_stress_test",
        "sequence": "four-point spin-echo inversion recovery; first four points retained, remaining 996 samples zero-padded",
        "direct_real_mrf_validation": False,
        "ground_truth_status": "The supplied processed T1 map is a reference for this stress test, not an independent MRF ground truth or an acquisition-failure label.",
        "inventory": inventory,
        "n_evaluated_records": len(records),
        "n_skipped_records": len(skipped),
        "skipped": skipped,
        "by_family": by_family,
        "aggregate": aggregate,
        "records": records,
    }


def _load_training_normalization(h5_path: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    train_indices = np.asarray(manifest["grouped_indices"]["train"], dtype=np.int64)
    with h5py.File(h5_path, "r") as handle:
        parameters = handle["parameters"][train_indices, :2].astype(np.float32)
    mean = parameters.mean(axis=0)
    std = parameters.std(axis=0) + 1e-8
    return manifest, mean, std


def _calibration_nll(model, loader, device, evidential_regression_loss) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for signals, targets in loader:
            targets = targets.to(device)
            output = model(signals.to(device))
            nig = output if output.ndim == 3 else output.view(output.shape[0], 2, 4)
            loss = evidential_regression_loss(
                targets, nig, coeff=1.0, epoch=1, annealing_epochs=1
            )
            total += float(loss["nll"]) * len(targets)
            count += len(targets)
    return total / max(count, 1)


def _train_or_load_model(
    h5_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    epochs: int,
    batch_size: int,
    device: str,
    force_retrain: bool,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from qMR_Robust.data.loaders import MRFMetaDataset
    from qMR_Robust.models.losses import evidential_regression_loss
    from qMR_Robust.models.resnet1d import ResNet1D
    from qMR_Robust.reproducibility import seed_everything

    manifest, mean, std = _load_training_normalization(h5_path, manifest_path)
    model = ResNet1D(
        in_channels=2,
        hidden_dim=128,
        output_dim=2,
        dropout=0.1,
        evidential=True,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and not force_retrain:
        payload = torch.load(str(checkpoint_path), map_location="cpu")
        state = payload["state_dict"] if "state_dict" in payload else payload
        model.load_state_dict(state)
        model.to(device)
        return model, mean, std, {
            "checkpoint": str(checkpoint_path),
            "loaded_existing": True,
            "epochs": payload.get("epochs") if isinstance(payload, dict) else None,
            "calibration_nll": payload.get("calibration_nll")
            if isinstance(payload, dict)
            else None,
            "h5": str(h5_path),
            "manifest": str(manifest_path),
            "seed": 42,
        }

    seed_everything(42, deterministic=True)
    indices = manifest["grouped_indices"]
    train_ds = MRFMetaDataset(
        h5_path, split="train", indices=np.asarray(indices["train"], dtype=np.int64)
    )
    calibration_ds = MRFMetaDataset(
        h5_path,
        split="calibration",
        indices=np.asarray(indices["calibration"], dtype=np.int64),
    )
    calibration_ds.set_norm(train_ds.mean, train_ds.std)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    calibration_loader = DataLoader(
        calibration_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    best_state = None
    best_calibration = float("inf")
    for epoch in range(epochs):
        model.train()
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            nig = model(signals)
            loss = evidential_regression_loss(
                targets,
                nig,
                coeff=1.0,
                epoch=epoch,
                annealing_epochs=max(10, epochs),
            )["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        calibration_nll = _calibration_nll(
            model, calibration_loader, device, evidential_regression_loss
        )
        print(
            f"epoch {epoch + 1:02d}/{epochs}: calibration_nll={calibration_nll:.6f}",
            flush=True,
        )
        if calibration_nll < best_calibration:
            best_calibration = calibration_nll
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no calibration-selected model state was produced")
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "epochs": int(epochs),
            "calibration_nll": float(best_calibration),
            "seed": 42,
            "protocol": "grouped v3 train/calibration/test model reused for external stress test",
            "h5": str(h5_path),
            "manifest": str(manifest_path),
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
        },
        str(checkpoint_path),
    )
    return model, mean, std, {
        "checkpoint": str(checkpoint_path),
        "loaded_existing": False,
        "epochs": int(epochs),
        "calibration_nll": float(best_calibration),
        "h5": str(h5_path),
        "manifest": str(manifest_path),
        "seed": 42,
    }



def _make_figure(mrf: dict[str, Any], nist: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, kind, title in (
        (axes[0, 0], "t1", "MRF map scan–rescan T1 distribution differences"),
        (axes[0, 1], "t2", "MRF map scan–rescan T2 distribution differences"),
    ):
        values = [
            p["metrics"][kind]["absolute_quantile_difference_ms"]
            for p in mrf["pairs"]
        ]
        if values:
            matrix = np.asarray(values)
            axis.boxplot([matrix[:, i] for i in range(matrix.shape[1])], tick_labels=["P10", "P25", "P50", "P75", "P90"])
        axis.set_title(title)
        axis.set_ylabel("absolute difference (ms)")
        axis.grid(axis="y", alpha=0.25)

    records = nist.get("records", [])
    families = ("magnitude", "complex")
    means = [
        np.mean([r["mae_ms"] for r in records if r["family"] == family])
        if any(r["family"] == family for r in records)
        else np.nan
        for family in families
    ]
    axes[1, 0].bar(families, means, color=["#4472c4", "#ed7d31"])
    axes[1, 0].set_title("NIST cross-sequence OOD error by input family")
    axes[1, 0].set_ylabel("mean voxel MAE (ms)")
    axes[1, 0].grid(axis="y", alpha=0.25)

    for family, color in zip(families, ("#4472c4", "#ed7d31")):
        family_records = [r for r in records if r["family"] == family]
        axes[1, 1].scatter(
            [r["target_t1_mean_ms"] for r in family_records],
            [r["predicted_t1_mean_ms"] for r in family_records],
            label=family,
            color=color,
            alpha=0.8,
        )
    if records:
        low = min(r["target_t1_mean_ms"] for r in records)
        high = max(r["target_t1_mean_ms"] for r in records)
        axes[1, 1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1, 1].set_title("NIST record means: supplied map vs prediction")
    axes[1, 1].set_xlabel("supplied T1-map mean (ms)")
    axes[1, 1].set_ylabel("model T1 prediction mean (ms)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle("External Zenodo audit (not direct real-MRF validation)", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=ROOT / "data/synthetic/failure_forecast_mrf_v3.h5",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "frozen_results/grouped_split_manifest_v3.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "results/checkpoints/zenodo_external_grouped_seed42.pt",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=4096,
        help="per-record deterministic voxel cap; use 0 for all valid voxels",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/external/zenodo_external_validation.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "results/figures/zenodo_external_validation.png",
    )
    args = parser.parse_args()

    archives = _archive_audit()
    if not all(item.get("verified", False) for item in archives.values()):
        raise RuntimeError(f"archive verification failed: {archives}")
    mrf = _mrf_scan_rescan_audit()
    model_info: dict[str, Any]
    if args.skip_model:
        nist = {
            "dataset_role": "cross_sequence_ood_stress_test",
            "skipped": "model evaluation disabled by --skip-model",
        }
        model_info = {"skipped": True}
    else:
        model, mean, std, model_info = _train_or_load_model(
            args.h5,
            args.manifest,
            args.checkpoint,
            args.epochs,
            args.batch_size,
            args.device,
            args.force_retrain,
        )
        nist = _nist_cross_sequence_evaluation(
            model, mean, std, args.device, args.max_voxels
        )
    _make_figure(mrf, nist, args.figure)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "protocol": "Zenodo external audit v1",
        "archives": archives,
        "mrf_map_scan_rescan": mrf,
        "nist_cross_sequence": nist,
        "model": model_info,
        "figure": str(args.figure),
        "interpretation_guardrail": "Neither Zenodo release supplies sequence-matched raw MRF fingerprints plus independent failure labels. The MRF release supports map-level scan–rescan distribution analysis; the NIST release supports only a clearly labeled four-point inversion-recovery cross-sequence stress test.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "figure": str(args.figure),
                "mrf_pairs": mrf["n_pairs"],
                "nist_records": nist.get("n_evaluated_records"),
                "nist_skipped": nist.get("n_skipped_records"),
                "nist_aggregate": nist.get("aggregate"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
