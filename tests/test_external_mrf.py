"""Tests for the future raw-MRF external-data ingestion contract."""

from __future__ import annotations

import numpy as np
import pytest


def _write_case(root):
    raw = (
        np.arange(8 * 4 * 6, dtype=np.float32).reshape(8, 4, 6)
        + 1j * np.ones((8, 4, 6), dtype=np.float32)
    ).astype(np.complex64)
    np.save(root / "raw_mrf.npy", raw)
    np.save(root / "raw_gre.npy", np.zeros((2, 4, 5), dtype=np.complex64))
    np.save(root / "noise.npy", np.zeros((4, 5), dtype=np.complex64))


def test_delics_loader_validates_axes_and_preserves_complex_phase(tmp_path):
    from qMR_Robust.data.external_mrf import load_delics_case

    _write_case(tmp_path)
    case = load_delics_case(
        tmp_path,
        expected_coils=4,
        n_tr=3,
        n_repeats=2,
    )

    assert case.raw_mrf.shape == (8, 4, 6)
    assert np.iscomplexobj(case.raw_mrf)
    assert case.gre_mrf is not None
    assert case.noise is not None
    assert case.summary()["labels"]["failure_labels_available"] is False
    assert case.summary()["protocol"]["tr_ms"] == 12.0


def test_delics_loader_uses_memory_mapping_by_default(tmp_path):
    from qMR_Robust.data.external_mrf import load_delics_case

    _write_case(tmp_path)
    case = load_delics_case(tmp_path, expected_coils=4, n_tr=3, n_repeats=2)
    assert isinstance(case.raw_mrf, np.memmap)


def test_delics_loader_rejects_magnitude_only_data(tmp_path):
    from qMR_Robust.data.external_mrf import load_delics_case

    np.save(tmp_path / "raw_mrf.npy", np.zeros((8, 4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="complex-valued"):
        load_delics_case(tmp_path, expected_coils=4, n_tr=3, n_repeats=2)


def test_delics_loader_rejects_wrong_spiral_count(tmp_path):
    from qMR_Robust.data.external_mrf import load_delics_case

    np.save(tmp_path / "raw_mrf.npy", np.zeros((8, 4, 5), dtype=np.complex64))
    with pytest.raises(ValueError, match="final axis mismatch"):
        load_delics_case(tmp_path, expected_coils=4, n_tr=3, n_repeats=2)


def test_delics_loader_requires_extracted_case_directory(tmp_path):
    from qMR_Robust.data.external_mrf import load_delics_case

    archive = tmp_path / "case.tar.gz"
    archive.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="extract"):
        load_delics_case(archive)
