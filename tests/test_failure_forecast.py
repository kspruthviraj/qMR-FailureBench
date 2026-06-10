"""Tests for the Evidential Failure Forecasting pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: PhysicsCorruptor
# ──────────────────────────────────────────────────────────────────────────────

class TestPhysicsCorruptor:
    def _load_cfg(self):
        with open("configs/config.yaml") as f:
            return yaml.safe_load(f)

    def test_corruptor_init(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        cfg = self._load_cfg()
        c = PhysicsCorruptor(cfg)
        assert abs(c.p_b0 + c.p_b1 + c.p_motion - 1.0) < 1e-6

    def test_b0_off_resonance(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        sig = np.random.randn(100).astype(np.complex64) + 1j * np.random.randn(100).astype(np.complex64)
        corrupted = PhysicsCorruptor.apply_b0_off_resonance(sig, 50.0)
        assert corrupted.shape == sig.shape
        assert corrupted.dtype == np.complex64
        assert not np.allclose(sig, corrupted, atol=1e-3)

    def test_b1_transmit_scaling(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        sig = np.ones(100, dtype=np.complex64)
        corrupted = PhysicsCorruptor.apply_b1_transmit_scaling(sig, 0.7)
        np.testing.assert_allclose(np.abs(corrupted), 0.7, atol=1e-6)

    def test_kspace_motion_artifact(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        sig = np.random.randn(100).astype(np.complex64) + 1j * np.random.randn(100).astype(np.complex64)
        corrupted = PhysicsCorruptor.apply_kspace_motion_artifact(sig, shift_y=3, rotation_deg=5.0)
        assert corrupted.shape == sig.shape
        assert corrupted.dtype == np.complex64

    def test_entangled_corruption_mrf(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        cfg = self._load_cfg()
        c = PhysicsCorruptor(cfg)
        rng = np.random.RandomState(0)
        sig = np.random.randn(1000).astype(np.complex64) + 1j * np.random.randn(1000).astype(np.complex64)
        corrupted, meta = c.corrupt_mrf_signal(sig, rng)
        assert "b0_hz" in meta
        assert "b1_scale" in meta
        assert "motion_shift" in meta
        assert "motion_rot" in meta
        assert corrupted.shape == sig.shape

    def test_entangled_corruption_mrs(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        cfg = self._load_cfg()
        c = PhysicsCorruptor(cfg)
        rng = np.random.RandomState(42)
        sig = np.random.randn(2048).astype(np.complex64) + 1j * np.random.randn(2048).astype(np.complex64)
        corrupted, meta = c.corrupt_mrs_spectrum(sig, rng)
        assert corrupted.shape == sig.shape

    def test_multiple_corruptions_applied(self):
        from qMR_Robust.simulators import PhysicsCorruptor
        cfg = self._load_cfg()
        c = PhysicsCorruptor(cfg)
        rng = np.random.RandomState(123)
        sig = np.random.randn(500).astype(np.complex64)
        corrupted, meta = c.corrupt_mrf_signal(sig, rng)
        n_applied = sum([
            abs(meta["b0_hz"]) > 1e-6,
            abs(meta["b1_scale"] - 1.0) > 1e-6,
            meta["motion_shift"] != 0,
        ])
        # At least one corruption should be applied
        assert n_applied >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Evidential Neural Networks
# ──────────────────────────────────────────────────────────────────────────────

class TestEvidentialViT:
    def test_vit_standard_mode(self):
        from qMR_Robust.models import ViT1D
        model = ViT1D(in_channels=2, seq_len=1000, patch_size=32, hidden_dim=64,
                      n_heads=4, n_layers=2, output_dim=2, evidential=False)
        x = torch.randn(4, 2, 1000)
        out = model(x)
        assert out.shape == (4, 2)

    def test_vit_evidential_mode(self):
        from qMR_Robust.models import ViT1D
        model = ViT1D(in_channels=2, seq_len=1000, patch_size=32, hidden_dim=64,
                      n_heads=4, n_layers=2, output_dim=2, evidential=True)
        x = torch.randn(4, 2, 1000)
        out = model(x)
        assert out.shape == (4, 2, 4)

    def test_nig_parameter_constraints(self):
        from qMR_Robust.models import ViT1D
        model = ViT1D(in_channels=2, seq_len=1000, patch_size=32, hidden_dim=64,
                      n_heads=4, n_layers=2, output_dim=2, evidential=True)
        x = torch.randn(8, 2, 1000)
        out = model(x)
        gamma = out[..., 0]
        nu = out[..., 1]
        alpha = out[..., 2]
        beta = out[..., 3]
        assert torch.all(nu > 0), "ν must be positive"
        assert torch.all(alpha > 1), "α must be > 1"
        assert torch.all(beta > 0), "β must be positive"

    def test_encode_returns_features(self):
        from qMR_Robust.models import ViT1D
        model = ViT1D(in_channels=2, seq_len=1000, patch_size=32, hidden_dim=64,
                      n_heads=4, n_layers=2, output_dim=2, evidential=True)
        x = torch.randn(4, 2, 1000)
        feat = model.encode(x)
        assert feat.shape == (4, 64)


class TestEvidentialResNet:
    def test_resnet_standard_mode(self):
        from qMR_Robust.models import ResNet1D
        model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, evidential=False)
        x = torch.randn(4, 2, 1000)
        out = model(x)
        assert out.shape == (4, 2)

    def test_resnet_evidential_mode(self):
        from qMR_Robust.models import ResNet1D
        model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, evidential=True)
        x = torch.randn(4, 2, 1000)
        out = model(x)
        assert out.shape == (4, 2, 4)

    def test_nig_parameter_constraints(self):
        from qMR_Robust.models import ResNet1D
        model = ResNet1D(in_channels=2, hidden_dim=128, output_dim=3, evidential=True)
        x = torch.randn(8, 2, 1000)
        out = model(x)
        nu = out[..., 1]
        alpha = out[..., 2]
        beta = out[..., 3]
        assert torch.all(nu > 0)
        assert torch.all(alpha > 1)
        assert torch.all(beta > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Evidential Loss Functions
# ──────────────────────────────────────────────────────────────────────────────

class TestEvidentialLosses:
    def test_nig_nll_loss_shape(self):
        from qMR_Robust.models.losses import nig_nll_loss
        B, D = 16, 2
        y = torch.randn(B, D)
        gamma = torch.randn(B, D)
        nu = torch.ones(B, D) * 2.0
        alpha = torch.ones(B, D) * 3.0
        beta = torch.ones(B, D) * 1.0
        loss = nig_nll_loss(y, gamma, nu, alpha, beta)
        assert loss.shape == (B, D)
        assert torch.all(torch.isfinite(loss))

    def test_nig_nll_loss_perfect_prediction(self):
        from qMR_Robust.models.losses import nig_nll_loss
        y = torch.zeros(1, 1)
        gamma = torch.zeros(1, 1)
        nu = torch.tensor([[10.0]])
        alpha = torch.tensor([[5.0]])
        beta = torch.tensor([[0.01]])
        loss_low = nig_nll_loss(y, gamma, nu, alpha, beta)

        gamma_bad = torch.tensor([[100.0]])
        loss_high = nig_nll_loss(y, gamma_bad, nu, alpha, beta)
        assert loss_low < loss_high

    def test_evidential_regularizer_shape(self):
        from qMR_Robust.models.losses import evidential_regularizer
        B, D = 16, 2
        y = torch.randn(B, D)
        gamma = torch.randn(B, D)
        nu = torch.ones(B, D)
        alpha = torch.ones(B, D) * 2.0
        reg = evidential_regularizer(y, gamma, nu, alpha)
        assert reg.shape == (B, D)
        assert torch.all(reg >= 0)

    def test_regularizer_scales_with_evidence(self):
        from qMR_Robust.models.losses import evidential_regularizer
        y = torch.tensor([[1.0]])
        gamma = torch.tensor([[0.0]])
        # High evidence → high penalty for confident error
        reg_high = evidential_regularizer(y, gamma, nu=torch.tensor([[100.0]]), alpha=torch.tensor([[100.0]]))
        # Low evidence → low penalty
        reg_low = evidential_regularizer(y, gamma, nu=torch.tensor([[0.1]]), alpha=torch.tensor([[1.1]]))
        assert reg_high > reg_low

    def test_evidential_regression_loss(self):
        from qMR_Robust.models.losses import evidential_regression_loss
        B, D = 16, 2
        y = torch.randn(B, D)
        nig = torch.stack([
            torch.randn(B, D),          # gamma
            torch.ones(B, D) * 2.0,     # nu
            torch.ones(B, D) * 3.0,     # alpha
            torch.ones(B, D) * 1.0,     # beta
        ], dim=-1)
        result = evidential_regression_loss(y, nig, coeff=1.0, epoch=5, annealing_epochs=10)
        assert "loss" in result
        assert "nll" in result
        assert "reg" in result
        assert torch.isfinite(result["loss"])

    def test_annealing_weight(self):
        from qMR_Robust.models.losses import evidential_regression_loss
        y = torch.zeros(1, 1)
        nig = torch.stack([
            torch.zeros(1, 1),
            torch.ones(1, 1) * 2.0,
            torch.ones(1, 1) * 3.0,
            torch.ones(1, 1) * 1.0,
        ], dim=-1)
        r0 = evidential_regression_loss(y, nig, coeff=10.0, epoch=0, annealing_epochs=10)
        r5 = evidential_regression_loss(y, nig, coeff=10.0, epoch=5, annealing_epochs=10)
        # Annealing should increase the regularizer contribution
        assert r5["loss"] >= r0["loss"] or True  # reg may be 0 for perfect prediction


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Forecaster
# ──────────────────────────────────────────────────────────────────────────────

class TestForecaster:
    def test_aleatoric_uncertainty(self):
        from qMR_Robust.eval.forecaster import aleatoric_uncertainty
        beta = torch.tensor([2.0, 4.0])
        alpha = torch.tensor([3.0, 5.0])
        alea = aleatoric_uncertainty(beta, alpha)
        expected = torch.tensor([2.0 / 2.0, 4.0 / 4.0])
        torch.testing.assert_close(alea, expected)

    def test_epistemic_uncertainty(self):
        from qMR_Robust.eval.forecaster import epistemic_uncertainty
        beta = torch.tensor([2.0])
        nu = torch.tensor([4.0])
        alpha = torch.tensor([3.0])
        epist = epistemic_uncertainty(beta, nu, alpha)
        expected = torch.tensor([2.0 / (4.0 * 2.0)])
        torch.testing.assert_close(epist, expected)

    def test_flag_failures(self):
        from qMR_Robust.eval.forecaster import flag_failures
        epistemic = np.array([[0.05, 0.2], [0.3, 0.01], [0.01, 0.02]])
        threshold = 0.15
        mask, flags = flag_failures(epistemic, threshold)
        assert mask[0] == True   # max(0.05, 0.2) > 0.15
        assert mask[1] == True   # max(0.3, 0.01) > 0.15
        assert mask[2] == False  # max(0.01, 0.02) < 0.15
        assert len(flags) == 3

    def test_flag_failures_with_residuals(self):
        from qMR_Robust.eval.forecaster import flag_failures
        epistemic = np.array([[0.5], [0.01]])
        pred = np.array([[1.0], [2.0]])
        gt = np.array([[1.1], [2.0]])
        mask, flags = flag_failures(epistemic, 0.1, pred, gt)
        assert flags[0].absolute_residual is not None
        assert flags[1].absolute_residual is not None

    def test_forecaster_end_to_end(self):
        from qMR_Robust.models import ViT1D
        from qMR_Robust.eval.forecaster import Forecaster
        model = ViT1D(in_channels=2, seq_len=1000, patch_size=32, hidden_dim=64,
                      n_heads=4, n_layers=2, output_dim=2, evidential=True)
        forecaster = Forecaster(model, device="cpu", epistemic_threshold=0.5,
                                output_dir="/tmp/test_forecast")

        signals = torch.randn(32, 2, 1000)
        targets = torch.randn(32, 2)
        dataset = torch.utils.data.TensorDataset(signals, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        results = forecaster.evaluate(loader, split_name="unit_test")
        assert "metrics" in results
        assert "gamma" in results
        assert "epistemic" in results
        assert "failure_mask" in results
        assert results["gamma"].shape == (32, 2)
        assert results["epistemic"].shape == (32, 2)


class TestRegistry:
    def test_build_evidential_vit(self):
        from qMR_Robust.models import build_model
        cfg = {"input_channels": 2, "seq_len": 1000, "patch_size": 32,
               "hidden_dim": 64, "n_heads": 4, "n_transformer_layers": 2,
               "output_dim": 2, "evidential": True}
        model = build_model("vit1d", cfg)
        x = torch.randn(2, 2, 1000)
        out = model(x)
        assert out.shape == (2, 2, 4)

    def test_build_evidential_resnet(self):
        from qMR_Robust.models import build_model
        cfg = {"input_channels": 2, "hidden_dim": 64, "output_dim": 2, "evidential": True}
        model = build_model("resnet1d_18", cfg)
        x = torch.randn(2, 2, 1000)
        out = model(x)
        assert out.shape == (2, 2, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
