"""Grouped split utilities for leakage-resistant qMRI evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import h5py
import numpy as np


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_labels(values: Iterable[object]) -> np.ndarray:
    labels = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        labels.append(str(value))
    return np.asarray(labels, dtype=object)


def grouped_split_indices(
    domain_labels: Iterable[object],
    seed: int = 42,
    fractions: Mapping[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    """Split samples by domain without allowing a domain across partitions.

    The default creates train/calibration/test partitions with approximate
    70/15/15 sample proportions. The assignment is deterministic for a seed and
    returns sorted sample indices for reproducible HDF5 slicing.
    """
    if fractions is None:
        fractions = {"train": 0.70, "calibration": 0.15, "test": 0.15}
    names = tuple(fractions)
    if not names or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("fractions must be non-empty and sum to one")

    labels = _decode_labels(domain_labels)
    if labels.size == 0:
        raise ValueError("domain_labels is empty")
    unique = np.unique(labels)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unique))

    target = np.asarray([fractions[name] * len(labels) for name in names])
    assigned = {name: [] for name in names}
    counts = np.zeros(len(names), dtype=float)

    for group_index in order:
        group = unique[group_index]
        sample_indices = np.flatnonzero(labels == group)
        # Largest remaining deficit gets the next complete domain.
        deficits = target - counts
        split_index = int(np.argmax(deficits))
        assigned[names[split_index]].extend(sample_indices.tolist())
        counts[split_index] += len(sample_indices)

    result = {
        name: np.asarray(sorted(indices), dtype=np.int64)
        for name, indices in assigned.items()
    }
    all_indices = np.concatenate(list(result.values()))
    if np.unique(all_indices).size != len(labels):
        raise RuntimeError("grouped split produced overlapping or missing indices")
    return result


def legacy_ordered_indices(
    n_samples: int,
    train_fraction: float = 0.8,
) -> Dict[str, np.ndarray]:
    """Return the historical ordered train/validation split for audit only."""
    cut = int(n_samples * train_fraction)
    return {
        "train": np.arange(0, cut, dtype=np.int64),
        "validation": np.arange(cut, n_samples, dtype=np.int64),
    }


def build_split_manifest(
    h5_path: str | Path,
    seed: int = 42,
) -> dict:
    """Build a JSON-serialisable manifest for both legacy and grouped splits."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as hf:
        if "domain_labels" not in hf:
            raise ValueError("HDF5 file has no domain_labels; grouped splitting is unsafe")
        labels = _decode_labels(hf["domain_labels"][:])
        n_samples = int(hf.attrs.get("n_signals", len(labels)))
        attrs = {}
        for key in ("t1_t2_units", "physics_version", "seed_scheme", "split"):
            if key in hf.attrs:
                value = hf.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                elif isinstance(value, np.generic):
                    value = value.item()
                attrs[key] = value

    grouped = grouped_split_indices(labels, seed=seed)
    legacy = legacy_ordered_indices(n_samples)
    return {
        "schema_version": "grouped_split_manifest_v1",
        "source_file": str(h5_path),
        "source_size_bytes": int(h5_path.stat().st_size),
        "source_sha256": _sha256_file(h5_path),
        "seed": int(seed),
        "n_samples": n_samples,
        "n_domains": int(np.unique(labels).size),
        "source_attrs": attrs,
        "grouped_protocol": "domain_grouped_70_15_15",
        "grouped_domain_counts": {
            name: int(np.unique(labels[idx]).size) for name, idx in grouped.items()
        },
        "grouped_sample_counts": {
            name: int(len(idx)) for name, idx in grouped.items()
        },
        "grouped_indices": {
            name: idx.tolist() for name, idx in grouped.items()
        },
        "legacy_protocol": "ordered_first_80_last_20_audit_only",
        "legacy_sample_counts": {name: int(len(idx)) for name, idx in legacy.items()},
        "legacy_indices": {name: idx.tolist() for name, idx in legacy.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path")
    parser.add_argument("output_json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_split_manifest(args.h5_path, seed=args.seed)
    Path(args.output_json).write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "output": args.output_json,
        "n_samples": manifest["n_samples"],
        "grouped_sample_counts": manifest["grouped_sample_counts"],
        "grouped_domain_counts": manifest["grouped_domain_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
