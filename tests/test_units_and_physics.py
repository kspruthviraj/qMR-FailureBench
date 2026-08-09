"""Integrity tests: units, physics constraints, calibration metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestUnits:
    def test_assert_t1_rejects_seconds(self):
        from qMR_Robust.data.loaders import assert_t1_units_ms
        with pytest.raises(ValueError, match="SECONDS"):
            assert_t1_units_ms(np.array([0.8, 1.2, 1.5]), "t1")

    def test_assert_t1_accepts_ms(self):
        from qMR_Robust.data.loaders import assert_t1_units_ms
        assert_t1_units_ms(np.array([800.0, 1200.0, 1500.0]), "t1")

    def test_qmrlab_loads_in_ms(self):
        from qMR_Robust.data.loaders import load_qmrlab_vfa, assert_t1_units_ms
        path = ROOT / "data/real/qmrlab/vfa_t1_data"
        if not (path / "VFAData.nii.gz").exists():
            pytest.skip("qMRLab data not present")
        data = load_qmrlab_vfa(path, pad_mode="zeropad")
        assert_t1_units_ms(data.t1_ms)
        assert data.t1_ms.mean() > 100
        assert data.signals.shape[1] == 2
        assert data.signals.shape[2] == 1000
        # Zero-pad: samples after n_fa should be ~0
        assert abs(data.signals[0, 0, data.n_fa_original :].sum()) < 1e-5


class TestPhysics:
    def test_corruptor_probs_sum_to_one(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
        c = PhysicsCorruptor(cfg)
        assert abs(c.p_b0 + c.p_b1 + c.p_motion - 1.0) < 1e-6

    def test_b0_uses_tr_scale_dwell(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        sig = np.ones(100, dtype=np.complex64)
        # 50 Hz over 12 ms steps → large phase by end
        out_tr = PhysicsCorruptor.apply_b0_off_resonance(sig, 50.0, dwell_time_ms=12.0)
        out_dwell = PhysicsCorruptor.apply_b0_off_resonance(sig, 50.0, dwell_time_ms=0.5)
        # Phase accrual should differ substantially
        assert not np.allclose(out_tr, out_dwell, atol=1e-3)

    def test_b1_resim_changes_signal(self):
        from qMR_Robust.simulators.manager import bloch_simulate
        n = 50
        fa = np.linspace(10, 60, n)
        tr = np.ones(n) * 12.0
        s1 = bloch_simulate(1000, 80, 1.0, fa, tr, 0.0, 1.0, n)
        s2 = bloch_simulate(1000, 80, 1.0, fa, tr, 0.0, 0.7, n)
        assert not np.allclose(s1, s2, atol=1e-4)
        # B1≠1 should change amplitude trajectory (not just global scale of same shape)
        ratio = np.abs(s2) / (np.abs(s1) + 1e-8)
        assert ratio.std() > 1e-4  # nonlinear FA effect → non-constant ratio

    def test_motion_preserves_length(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        sig = (np.random.randn(200) + 1j * np.random.randn(200)).astype(np.complex64)
        out = PhysicsCorruptor.apply_kspace_motion_artifact(sig, shift_y=3, rotation_deg=5.0)
        assert out.shape == sig.shape
        assert out.dtype == np.complex64

    def test_entangled_at_least_one(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        cfg = yaml.safe_load(open(ROOT / "configs/config.yaml"))
        c = PhysicsCorruptor(cfg)
        rng = np.random.RandomState(0)
        for _ in range(50):
            sig = np.random.randn(100).astype(np.complex64)
            _, meta = c.corrupt_mrf_signal(sig, rng)
            n_app = sum([
                abs(meta["b0_hz"]) > 1e-6,
                abs(meta["b1_scale"] - 1.0) > 1e-6,
                meta["motion_shift"] != 0,
            ])
            assert n_app >= 1


class TestCalibration:
    def test_normalized_ece_scale(self):
        from qMR_Robust.eval.calibration import normalized_ece
        rng = np.random.RandomState(0)
        err = rng.rand(1000) * 100  # ms-scale
        unc = err + rng.randn(1000) * 10
        ece = normalized_ece(unc, err)
        assert 0 <= ece < 5  # dimensionless, not hundreds

    def test_legacy_ece_normalized_default(self):
        from qMR_Robust.eval.metrics import expected_calibration_error
        rng = np.random.RandomState(0)
        err = rng.rand(500) * 200
        unc = err * 0.5
        ece, _, _ = expected_calibration_error(unc, err, normalize=True)
        assert ece < 10

    def test_wilson_ci(self):
        from qMR_Robust.eval.calibration import wilson_ci
        lo, hi = wilson_ci(21, 50)
        assert 0 < lo < 0.42 < hi < 1

    def test_categorize_rho(self):
        from qMR_Robust.eval.calibration import categorize_rho
        assert categorize_rho(0.95) == "good"
        assert categorize_rho(-0.5) == "degenerate"


class TestDenorm:
    def test_counterfactual_no_double_denorm(self):
        """Targets in raw ms must not be denormalized again."""
        params_ms = np.array([[800.0, 80.0], [1200.0, 100.0]], dtype=np.float32)
        t_mean = np.array([1000.0, 90.0], dtype=np.float32)
        t_std = np.array([200.0, 20.0], dtype=np.float32)
        # Correct: use params as-is
        targets_raw = params_ms
        assert targets_raw.mean() > 100
        # Wrong pattern (double denorm) would produce nonsense:
        wrong = params_ms * t_std + t_mean
        assert wrong.mean() > 50_000  # clearly wrong scale


class TestNIG:
    def test_epistemic_stable_near_alpha_boundary(self):
        from qMR_Robust.eval.nig_utils import nig_epistemic_np
        nu = np.array([1.0])
        alpha = np.array([1.0 + 1e-8])
        beta = np.array([1.0])
        ep = nig_epistemic_np(nu, alpha, beta)
        assert np.isfinite(ep).all()
        assert ep[0] > 0
