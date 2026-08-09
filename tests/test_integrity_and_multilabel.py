"""Tests for reproducibility, grouped evaluation, and multi-label attribution."""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch


def test_stable_seed_is_independent_of_python_hash_randomization():
    from qMR_Robust.reproducibility import stable_seed

    first = stable_seed(42, "ge", 3.0, 2, 1, 17)
    second = stable_seed(42, "ge", 3.0, 2, 1, 17)
    other = stable_seed(42, "ge", 3.0, 2, 1, 18)
    assert first == second
    assert first != other
    assert 0 < first < 2**31 - 1


def test_grouped_splits_are_disjoint_and_domain_atomic():
    from qMR_Robust.data.splits import grouped_split_indices

    labels = np.repeat(np.array(["a", "b", "c", "d", "e"], dtype=object), 4)
    splits = grouped_split_indices(labels, seed=7)
    all_indices = np.concatenate(list(splits.values()))
    assert np.unique(all_indices).size == len(labels)
    assert np.sort(all_indices).tolist() == list(range(len(labels)))

    for split_name, indices in splits.items():
        split_domains = set(labels[indices])
        for other_name, other_indices in splits.items():
            if split_name == other_name:
                continue
            assert split_domains.isdisjoint(set(labels[other_indices]))


def test_attribution_targets_preserve_entangled_labels():
    from qMR_Robust.models.corruption_attribution import (
        attribution_loss,
        compute_attribution_targets,
    )

    targets = compute_attribution_targets(
        torch.tensor([50.0]),
        torch.tensor([0.70]),
        torch.tensor([3.0]),
    )
    torch.testing.assert_close(targets, torch.ones(1, 3))
    assert torch.isfinite(attribution_loss(torch.zeros(1, 3), targets))


def test_benchmark_does_not_publish_proxy_failure_labels(tmp_path):
    from qMR_Robust.benchmark import _process_mrf_benchmark

    source = tmp_path / "source.h5"
    out_dir = tmp_path / "out"
    n = 10
    with h5py.File(source, "w") as hf:
        hf.create_dataset("clean_signals", data=np.zeros((n, 4), dtype=np.complex64))
        hf.create_dataset("corrupted_signals", data=np.ones((n, 4), dtype=np.complex64))
        hf.create_dataset("parameters", data=np.ones((n, 3), dtype=np.float32) * 1000)
        hf.create_dataset("b0_hz_applied", data=np.linspace(0, 50, n, dtype=np.float32))
        hf.create_dataset("b1_scale_applied", data=np.ones(n, dtype=np.float32))
        hf.create_dataset("motion_shift_applied", data=np.zeros(n, dtype=np.float32))
        hf.create_dataset(
            "domain_labels",
            data=np.asarray([f"d{i // 2}".encode() for i in range(n)], dtype="S8"),
        )
        hf.attrs["n_signals"] = n
        hf.attrs["t1_t2_units"] = "ms"
        hf.attrs["physics_version"] = "test"
        hf.attrs["seed_scheme"] = "stable_blake2b_v1"

    stats = _process_mrf_benchmark(
        str(source), out_dir, {"mrf_t1": 100.0, "mrf_t2": 50.0}
    )
    assert stats["failure_labels_available"] is False

    with h5py.File(out_dir / "mrf_benchmark.h5", "r") as hf:
        assert "failure_t1" not in hf
        assert "severity_proxy_t1" in hf
        assert hf.attrs["failure_labels_available"] == 0


def test_audit_script_is_json_serializable(tmp_path):
    from scripts.audit_scientific_integrity import audit_source

    report = audit_source(tmp_path)
    json.dumps(report)


def test_mrf_meta_dataset_accepts_explicit_indices(tmp_path):
    from qMR_Robust.data.loaders import MRFMetaDataset

    source = tmp_path / "mrf.h5"
    n = 6
    with h5py.File(source, "w") as hf:
        hf.create_dataset("corrupted_signals", data=np.ones((n, 8), dtype=np.complex64))
        hf.create_dataset(
            "parameters",
            data=np.asarray([[500.0, 50.0, 1.0], [600.0, 60.0, 1.0],
                             [700.0, 70.0, 1.0], [800.0, 80.0, 1.0],
                             [900.0, 90.0, 1.0], [1000.0, 100.0, 1.0]],
                            dtype=np.float32),
        )
        hf.attrs["n_signals"] = n

    train = MRFMetaDataset(source, split="train", indices=np.array([0, 2, 4]))
    test = MRFMetaDataset(source, split="test", indices=np.array([1, 3, 5]))
    test.set_norm(train.mean, train.std)
    assert len(train) == 3
    assert len(test) == 3
    assert train.mean[0] == np.mean([500.0, 700.0, 900.0])

def test_split_manifest_records_source_fingerprint(tmp_path):
    from qMR_Robust.data.splits import build_split_manifest

    source = tmp_path / "mrf.h5"
    with h5py.File(source, "w") as hf:
        hf.create_dataset(
            "domain_labels",
            data=np.asarray([b"a", b"a", b"b", b"b", b"c", b"c"]),
        )
        hf.create_dataset("parameters", data=np.ones((6, 3), dtype=np.float32))
        hf.attrs["n_signals"] = 6
    manifest = build_split_manifest(source, seed=42)
    assert manifest["source_size_bytes"] == source.stat().st_size
    assert len(manifest["source_sha256"]) == 64
    assert manifest["grouped_sample_counts"]["train"] + manifest["grouped_sample_counts"]["calibration"] + manifest["grouped_sample_counts"]["test"] == 6
