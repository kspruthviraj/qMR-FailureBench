"""Tests for exact synthetic MRF schedule provenance reconstruction."""

from __future__ import annotations

import numpy as np


def _sim_cfg():
    return {
        "mrf": {
            "t1_range": [50.0, 3000.0],
            "t2_range": [5.0, 500.0],
            "m0_range": [0.5, 1.5],
            "n_timepoints": 32,
            "b0_shift_range": [-50.0, 50.0],
            "b1_scale_range": [0.8, 1.2],
            "snr_range": [10.0, 100.0],
            "fa_schedule_variants": 2,
            "tr_schedule_variants": 2,
            "vendors": ["siemens", "ge"],
            "field_strengths": [1.5, 3.0],
        }
    }


def test_domain_table_order_is_deterministic():
    from qMR_Robust.simulators.manager import mrf_domain_table

    table = mrf_domain_table(_sim_cfg())
    assert table[0] == ("siemens", 1.5, 0, 0)
    assert table[1] == ("siemens", 1.5, 0, 1)
    assert table[-1] == ("ge", 3.0, 1, 1)
    assert len(table) == 16


def test_sample_reconstructs_exactly_from_seed_and_domain():
    from qMR_Robust.simulators.manager import (
        _generate_mrf_sample,
        reconstruct_mrf_sample_from_metadata,
    )

    cfg = _sim_cfg()
    original = _generate_mrf_sample(12345, cfg, "ge", 3.0, 1, 0)
    reconstructed = reconstruct_mrf_sample_from_metadata(cfg, 12345, 14)

    np.testing.assert_array_equal(original["signal"], reconstructed["signal"])
    np.testing.assert_array_equal(original["clean_signal"], reconstructed["clean_signal"])
    np.testing.assert_array_equal(original["params"], reconstructed["params"])
    assert original["domain_name"] == reconstructed["domain_name"]
    assert original["base_b0_hz"] == reconstructed["base_b0_hz"]
    assert original["base_b1_scale"] == reconstructed["base_b1_scale"]


def test_invalid_domain_id_is_rejected():
    from qMR_Robust.simulators.manager import reconstruct_mrf_sample_from_metadata

    try:
        reconstruct_mrf_sample_from_metadata(_sim_cfg(), 1, 16)
    except ValueError as exc:
        assert "domain_id" in str(exc)
    else:
        raise AssertionError("invalid domain_id was accepted")
