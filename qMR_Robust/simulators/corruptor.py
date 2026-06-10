"""
PhysicsCorruptor — Entangled artifact injection for Failure Forecasting.

Instead of single isolated artifacts, this module injects *entangled*
corruptions — simultaneous B₀ off-resonance, B₁⁺ transmit scaling errors,
and synthetic k-space motion artifacts — into the same signal.  The
probability weights control which combination of artifacts is active for
each sample.

Output is written to HDF5 files in the same schema used by the training
pipeline so corrupted datasets can be loaded transparently.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml
from tqdm import tqdm

from .manager import (
    FIELD_FACTORS,
    METABOLITE_CONC_RANGE,
    METABOLITE_SHIFTS_PPM,
    VENDOR_BIAS,
    _generate_fa_schedule,
    _generate_mrf_sample,
    _generate_mrs_sample,
    _generate_tr_schedule,
    _lorentzian,
    bloch_simulate,
)

logger = logging.getLogger(__name__)


class PhysicsCorruptor:
    """Inject entangled physical corruptions into MRF and MRS signals.

    Parameters
    ----------
    cfg : dict
        Full project configuration (must include ``simulation.corruptor``).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        cor = cfg.get("simulation", {}).get("corruptor", {})
        self.b0_range = cor.get("b0_off_resonance_range", [-80.0, 80.0])
        self.b1_range = cor.get("b1_transmit_scale_range", [0.6, 1.4])
        self.motion_shift = cor.get("motion_kspace_max_shift", [8, 8])
        self.motion_rot_range = cor.get("motion_kspace_rotation_range", [-15.0, 15.0])
        pw = cor.get("probability_weights", {"b0": 1.0, "b1": 1.0, "motion": 1.0})
        total_w = pw["b0"] + pw["b1"] + pw["motion"]
        self.p_b0 = pw["b0"] / total_w
        self.p_b1 = pw["b1"] / total_w
        self.p_motion = pw["motion"] / total_w

    # ── individual corruption primitives ──────────────────────────────────

    @staticmethod
    def apply_b0_off_resonance(
        signal: np.ndarray, b0_shift_hz: float, dwell_time_ms: float = 0.5,
    ) -> np.ndarray:
        """Apply B₀ off-resonance phase ramp across the time-domain signal."""
        n = signal.shape[-1]
        t = np.arange(n, dtype=np.float64) * dwell_time_ms * 1e-3
        phase = np.exp(1j * 2 * np.pi * b0_shift_hz * t).astype(np.complex64)
        return (signal * phase).astype(np.complex64)

    @staticmethod
    def apply_b1_transmit_scaling(
        signal: np.ndarray, b1_scale: float,
    ) -> np.ndarray:
        """Scale signal amplitude to simulate B₁⁺ transmit inhomogeneity."""
        return (signal * b1_scale).astype(np.complex64)

    @staticmethod
    def apply_kspace_motion_artifact(
        signal: np.ndarray,
        shift_y: int,
        shift_x: int = 0,
        rotation_deg: float = 0.0,
    ) -> np.ndarray:
        """Inject synthetic k-space motion artifacts.

        For 1-D signals the artifact is modelled as a combination of:
        1.  Random k-space line displacement (phase-shift in Fourier domain).
        2.  Random rotation-induced phase roll.

        For multi-dimensional signals the first axis is treated as the
        readout / spectral dimension.
        """
        sig = signal.astype(np.complex64)
        n = sig.shape[-1]
        k = np.fft.fft(sig, axis=-1)

        # phase-shift ↔ spatial / spectral translation
        freqs = np.fft.fftfreq(n, d=1.0 / n)
        shift_phase = np.exp(-1j * 2 * np.pi * freqs * shift_y / n)
        k = k * shift_phase

        # rotation-induced linear phase ramp (simplified model)
        if abs(rotation_deg) > 1e-3:
            rot_phase = np.exp(-1j * np.deg2rad(rotation_deg) * freqs)
            k = k * rot_phase

        return np.fft.ifft(k, axis=-1).astype(np.complex64)

    # ── entangled corruption dispatcher ───────────────────────────────────

    def corrupt_mrf_signal(
        self, signal: np.ndarray, rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inject an entangled combination of B₀, B₁, and motion artifacts.

        Returns the corrupted signal and a metadata dict recording which
        corruptions were applied and their magnitudes.
        """
        corrupted = signal.copy()
        meta: Dict[str, Any] = {"b0_hz": 0.0, "b1_scale": 1.0, "motion_shift": 0, "motion_rot": 0.0}

        if rng.random() < (self.p_b0 + self.p_b1 + self.p_motion):
            # Decide which subset of corruptions to entangle
            apply_b0 = rng.random() < self.p_b0
            apply_b1 = rng.random() < self.p_b1
            apply_motion = rng.random() < self.p_motion

            # Guarantee at least one corruption is applied
            if not (apply_b0 or apply_b1 or apply_motion):
                choice = rng.randint(3)
                apply_b0 = choice == 0
                apply_b1 = choice == 1
                apply_motion = choice == 2

            if apply_b0:
                b0_shift = rng.uniform(*self.b0_range)
                corrupted = self.apply_b0_off_resonance(corrupted, b0_shift)
                meta["b0_hz"] = float(b0_shift)

            if apply_b1:
                b1_scale = rng.uniform(*self.b1_range)
                corrupted = self.apply_b1_transmit_scaling(corrupted, b1_scale)
                meta["b1_scale"] = float(b1_scale)

            if apply_motion:
                sy = rng.randint(-self.motion_shift[1], self.motion_shift[1] + 1)
                rot = rng.uniform(*self.motion_rot_range)
                corrupted = self.apply_kspace_motion_artifact(corrupted, shift_y=sy, rotation_deg=rot)
                meta["motion_shift"] = int(sy)
                meta["motion_rot"] = float(rot)

        return corrupted, meta

    def corrupt_mrs_spectrum(
        self, spectrum: np.ndarray, rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inject entangled corruptions into an MRS spectrum."""
        corrupted = spectrum.copy()
        meta: Dict[str, Any] = {"b0_hz": 0.0, "b1_scale": 1.0, "motion_shift": 0, "motion_rot": 0.0}

        if rng.random() < (self.p_b0 + self.p_b1 + self.p_motion):
            apply_b0 = rng.random() < self.p_b0
            apply_b1 = rng.random() < self.p_b1
            apply_motion = rng.random() < self.p_motion

            if not (apply_b0 or apply_b1 or apply_motion):
                choice = rng.randint(3)
                apply_b0 = choice == 0
                apply_b1 = choice == 1
                apply_motion = choice == 2

            if apply_b0:
                b0_shift = rng.uniform(*self.b0_range)
                corrupted = self.apply_b0_off_resonance(corrupted, b0_shift)
                meta["b0_hz"] = float(b0_shift)

            if apply_b1:
                b1_scale = rng.uniform(*self.b1_range)
                corrupted = self.apply_b1_transmit_scaling(corrupted, b1_scale)
                meta["b1_scale"] = float(b1_scale)

            if apply_motion:
                sy = rng.randint(-self.motion_shift[1], self.motion_shift[1] + 1)
                rot = rng.uniform(*self.motion_rot_range)
                corrupted = self.apply_kspace_motion_artifact(corrupted, shift_y=sy, rotation_deg=rot)
                meta["motion_shift"] = int(sy)
                meta["motion_rot"] = float(rot)

        return corrupted, meta

    # ── batch generation for Failure Forecasting split ─────────────────────

    def generate_failure_forecast_mrf(
        self, output_path: str, n_signals: Optional[int] = None,
    ) -> str:
        """Generate corrupted MRF signals for the Failure Forecasting training split.

        Each sample contains the *clean* signal (for reference), the
        *corrupted* signal (for training), the ground-truth parameters,
        and per-sample corruption metadata.
        """
        mc = self.cfg.get("simulation", {}).get("mrf", {})
        n = n_signals or mc.get("n_signals", 500_000)
        vendors = mc.get("vendors", ["siemens", "philips", "ge"])
        fields = mc.get("field_strengths", [1.5, 3.0, 7.0])
        fa_vars = mc.get("fa_schedule_variants", 5)
        tr_vars = mc.get("tr_schedule_variants", 3)
        n_time = mc.get("n_timepoints", 1000)

        combos = [(v, f, fa, tr) for v in vendors for f in fields
                  for fa in range(fa_vars) for tr in range(tr_vars)]
        per_combo = max(1, n // len(combos))
        total = per_combo * len(combos)

        logger.info(
            "Generating %d failure-forecast MRF samples (%d domains) …",
            total, len(combos),
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        tasks = []
        for v, f, fa, tr in combos:
            for _ in range(per_combo):
                tasks.append((v, f, fa, tr))

        worker_fn = partial(
            _corrupt_mrf_worker,
            sim_cfg=self.cfg.get("simulation", {}),
            corruptor_cfg=self.cfg.get("simulation", {}).get("corruptor", {}),
            base_seed=self.seed,
        )

        with h5py.File(out, "w") as hf:
            clean_ds = hf.create_dataset(
                "clean_signals", shape=(total, n_time), dtype=np.complex64,
                chunks=(min(1024, total), n_time), compression="gzip", compression_opts=1,
            )
            corrupt_ds = hf.create_dataset(
                "corrupted_signals", shape=(total, n_time), dtype=np.complex64,
                chunks=(min(1024, total), n_time), compression="gzip", compression_opts=1,
            )
            param_ds = hf.create_dataset("parameters", shape=(total, 3), dtype=np.float32)
            dom_ds = hf.create_dataset("domain_labels", shape=(total,), dtype="S64")
            b0_ds = hf.create_dataset("b0_hz_applied", shape=(total,), dtype=np.float32)
            b1_ds = hf.create_dataset("b1_scale_applied", shape=(total,), dtype=np.float32)
            mot_ds = hf.create_dataset("motion_shift_applied", shape=(total,), dtype=np.int32)
            rot_ds = hf.create_dataset("motion_rotation_applied", shape=(total,), dtype=np.float32)

            n_workers = mc.get("n_workers", mp.cpu_count())
            if n_workers > 1:
                ctx = mp.get_context("spawn")
                with ctx.Pool(n_workers) as pool:
                    for i, result in enumerate(
                        tqdm(pool.imap(worker_fn, tasks, chunksize=64),
                             total=total, desc="FailureForecastMRF")
                    ):
                        clean_ds[i] = result["clean"]
                        corrupt_ds[i] = result["corrupted"]
                        param_ds[i] = result["params"]
                        dom_ds[i] = result["domain_name"].encode()
                        b0_ds[i] = result["meta"]["b0_hz"]
                        b1_ds[i] = result["meta"]["b1_scale"]
                        mot_ds[i] = result["meta"]["motion_shift"]
                        rot_ds[i] = result["meta"]["motion_rot"]
            else:
                for i, task in enumerate(tqdm(tasks, desc="FailureForecastMRF")):
                    result = worker_fn(task)
                    clean_ds[i] = result["clean"]
                    corrupt_ds[i] = result["corrupted"]
                    param_ds[i] = result["params"]
                    dom_ds[i] = result["domain_name"].encode()
                    b0_ds[i] = result["meta"]["b0_hz"]
                    b1_ds[i] = result["meta"]["b1_scale"]
                    mot_ds[i] = result["meta"]["motion_shift"]
                    rot_ds[i] = result["meta"]["motion_rot"]

            hf.attrs["n_signals"] = total
            hf.attrs["n_domains"] = len(combos)
            hf.attrs["n_timepoints"] = n_time
            hf.attrs["split"] = "failure_forecast"
            hf.attrs["description"] = "Entangled corruption training data for evidential failure forecasting"

        logger.info("Failure-forecast MRF saved → %s  (%d signals)", out, total)
        return str(out)

    def generate_failure_forecast_mrs(
        self, output_path: str, n_signals: Optional[int] = None,
    ) -> str:
        """Generate corrupted MRS spectra for the Failure Forecasting training split."""
        mc = self.cfg.get("simulation", {}).get("mrs", {})
        n = n_signals or mc.get("n_signals", 100_000)
        te_values = mc.get("te_values", [30.0, 68.0, 80.0, 144.0])
        n_pts = mc.get("n_points", 2048)
        metabolites = mc.get("metabolites", list(METABOLITE_SHIFTS_PPM.keys()))
        n_met = len(metabolites)

        per_te = max(1, n // len(te_values))
        total = per_te * len(te_values)

        logger.info("Generating %d failure-forecast MRS samples …", total)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        tasks = []
        for te in te_values:
            for _ in range(per_te):
                tasks.append(te)

        worker_fn = partial(
            _corrupt_mrs_worker,
            sim_cfg=self.cfg.get("simulation", {}),
            corruptor_cfg=self.cfg.get("simulation", {}).get("corruptor", {}),
            base_seed=self.seed,
        )

        with h5py.File(out, "w") as hf:
            clean_ds = hf.create_dataset(
                "clean_spectra", shape=(total, n_pts), dtype=np.complex64,
                chunks=(min(1024, total), n_pts), compression="gzip", compression_opts=1,
            )
            corrupt_ds = hf.create_dataset(
                "corrupted_spectra", shape=(total, n_pts), dtype=np.complex64,
                chunks=(min(1024, total), n_pts), compression="gzip", compression_opts=1,
            )
            conc_ds = hf.create_dataset("concentrations", shape=(total, n_met), dtype=np.float32)
            dom_ds = hf.create_dataset("domain_labels", shape=(total,), dtype="S32")
            b0_ds = hf.create_dataset("b0_hz_applied", shape=(total,), dtype=np.float32)
            b1_ds = hf.create_dataset("b1_scale_applied", shape=(total,), dtype=np.float32)
            mot_ds = hf.create_dataset("motion_shift_applied", shape=(total,), dtype=np.int32)
            rot_ds = hf.create_dataset("motion_rotation_applied", shape=(total,), dtype=np.float32)

            n_workers = mc.get("n_workers", mp.cpu_count())
            if n_workers > 1:
                ctx = mp.get_context("spawn")
                with ctx.Pool(n_workers) as pool:
                    for i, result in enumerate(
                        tqdm(pool.imap(worker_fn, tasks, chunksize=64),
                             total=total, desc="FailureForecastMRS")
                    ):
                        clean_ds[i] = result["clean"]
                        corrupt_ds[i] = result["corrupted"]
                        conc_ds[i] = result["concentrations"]
                        dom_ds[i] = result["domain_name"].encode()
                        b0_ds[i] = result["meta"]["b0_hz"]
                        b1_ds[i] = result["meta"]["b1_scale"]
                        mot_ds[i] = result["meta"]["motion_shift"]
                        rot_ds[i] = result["meta"]["motion_rot"]
            else:
                for i, te_val in enumerate(tqdm(tasks, desc="FailureForecastMRS")):
                    result = worker_fn(te_val)
                    clean_ds[i] = result["clean"]
                    corrupt_ds[i] = result["corrupted"]
                    conc_ds[i] = result["concentrations"]
                    dom_ds[i] = result["domain_name"].encode()
                    b0_ds[i] = result["meta"]["b0_hz"]
                    b1_ds[i] = result["meta"]["b1_scale"]
                    mot_ds[i] = result["meta"]["motion_shift"]
                    rot_ds[i] = result["meta"]["motion_rot"]

            hf.attrs["n_signals"] = total
            hf.attrs["metabolites"] = metabolites
            hf.attrs["n_points"] = n_pts
            hf.attrs["te_values"] = te_values
            hf.attrs["split"] = "failure_forecast"
            hf.attrs["description"] = "Entangled corruption training data for evidential failure forecasting"

        logger.info("Failure-forecast MRS saved → %s  (%d signals)", out, total)
        return str(out)


# ── picklable top-level workers for multiprocessing ───────────────────────────

def _corrupt_mrf_worker(task, sim_cfg, corruptor_cfg, base_seed):
    v, f, fa, tr = task
    seed = base_seed ^ hash((v, f, fa, tr)) % (2**31)
    rng = np.random.RandomState(seed)

    sample = _generate_mrf_sample(seed, sim_cfg, v, f, fa, tr)
    corruptor = PhysicsCorruptor({"simulation": {"corruptor": corruptor_cfg}}, seed=seed)
    corrupted, meta = corruptor.corrupt_mrf_signal(sample["signal"], rng)

    return {
        "clean": sample["signal"],
        "corrupted": corrupted,
        "params": sample["params"],
        "domain_name": sample["domain_name"],
        "meta": meta,
    }


def _corrupt_mrs_worker(te, sim_cfg, corruptor_cfg, base_seed):
    seed = base_seed ^ hash(te) % (2**31)
    rng = np.random.RandomState(seed)

    sample = _generate_mrs_sample(seed, sim_cfg, te)
    corruptor = PhysicsCorruptor({"simulation": {"corruptor": corruptor_cfg}}, seed=seed)
    corrupted, meta = corruptor.corrupt_mrs_spectrum(sample["signal"], rng)

    return {
        "clean": sample["signal"],
        "corrupted": corrupted,
        "concentrations": sample["concentrations"],
        "domain_name": sample["domain_name"],
        "meta": meta,
    }
