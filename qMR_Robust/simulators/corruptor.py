"""
PhysicsCorruptor — Entangled artifact injection for Failure Forecasting.

Corruptions
-----------
1. **B0 off-resonance** — phase ramp using the true inter-sample interval.
   For MRF fingerprints the default Δt is the mean TR (ms), not an arbitrary
   dwell time. Prefer applying B0 inside ``bloch_simulate`` when regenerating
   data; the post-hoc ramp remains for post-processing / counterfactual invert.

2. **B1+ transmit** — when FA schedules are available, re-simulate with scaled
   FAs (physically correct). Fallback: amplitude scale (legacy, approximate).

3. **Motion** — multi-shot phase/segment errors on the 1-D fingerprint
   (view-to-view phase jumps), not a pure circular shift of the whole train.

At least one corruption is always applied when ``force_at_least_one=True``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from .manager import (
    METABOLITE_SHIFTS_PPM,
    _generate_fa_schedule,
    _generate_mrf_sample,
    _generate_mrs_sample,
    _generate_tr_schedule,
    bloch_simulate,
)
from qMR_Robust.reproducibility import stable_seed

logger = logging.getLogger(__name__)


class PhysicsCorruptor:
    """Inject entangled physical corruptions into MRF and MRS signals."""

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        cor = cfg.get("simulation", {}).get("corruptor", {})
        self.b0_range = cor.get("b0_off_resonance_range", [-80.0, 80.0])
        self.b1_range = cor.get("b1_transmit_scale_range", [0.6, 1.4])
        self.motion_shift = cor.get("motion_kspace_max_shift", [8, 8])
        self.motion_rot_range = cor.get("motion_kspace_rotation_range", [-15.0, 15.0])
        self.default_tr_ms = float(cor.get("default_tr_ms", 12.0))
        self.n_motion_segments = int(cor.get("n_motion_segments", 8))
        pw = cor.get("probability_weights", {"b0": 1.0, "b1": 1.0, "motion": 1.0})
        total_w = max(pw.get("b0", 1.0) + pw.get("b1", 1.0) + pw.get("motion", 1.0), 1e-8)
        # Normalized relative weights (sum to 1) — kept for tests / logging
        self.p_b0 = pw.get("b0", 1.0) / total_w
        self.p_b1 = pw.get("b1", 1.0) / total_w
        self.p_motion = pw.get("motion", 1.0) / total_w
        # Independent application probability (equal weights → ~0.5 each so
        # pairwise and triple entanglement occur with non-trivial frequency)
        self.p_apply = float(cor.get("independent_apply_prob", 0.5))

    # ── individual corruption primitives ──────────────────────────────────

    @staticmethod
    def apply_b0_off_resonance(
        signal: np.ndarray,
        b0_shift_hz: float,
        dwell_time_ms: float = 12.0,
    ) -> np.ndarray:
        """Apply B₀ off-resonance phase ramp across the time-domain signal.

        Parameters
        ----------
        dwell_time_ms :
            Time between successive samples in milliseconds. For MRF
            fingerprints this should be the TR (typically 8–15 ms), NOT
            a readout dwell of 0.5 ms.
        """
        n = signal.shape[-1]
        t = np.arange(n, dtype=np.float64) * float(dwell_time_ms) * 1e-3
        phase = np.exp(1j * 2 * np.pi * b0_shift_hz * t).astype(np.complex64)
        return (signal * phase).astype(np.complex64)

    @staticmethod
    def apply_b1_transmit_scaling(
        signal: np.ndarray, b1_scale: float,
    ) -> np.ndarray:
        """Legacy amplitude scale (approximate B1 proxy only).

        Prefer ``apply_b1_via_flip_angles`` / re-simulation when possible.
        """
        return (signal * b1_scale).astype(np.complex64)

    @staticmethod
    def apply_b1_via_resimulate(
        t1: float,
        t2: float,
        m0: float,
        fa: np.ndarray,
        tr: np.ndarray,
        b0_hz: float,
        b1_scale: float,
    ) -> np.ndarray:
        """Physically correct B1+: re-run Bloch with scaled flip angles."""
        n = len(fa)
        return bloch_simulate(t1, t2, m0, fa, tr, b0_hz, b1_scale, n).astype(np.complex64)

    @staticmethod
    def apply_kspace_motion_artifact(
        signal: np.ndarray,
        shift_y: int,
        shift_x: int = 0,
        rotation_deg: float = 0.0,
        n_segments: int = 8,
    ) -> np.ndarray:
        """Inject multi-shot motion errors into a 1-D fingerprint.

        Models view-to-view (segment-wise) phase jumps and a mild global
        linear phase, approximating interrupted k-space / shot inconsistency
        without claiming full 2-D k-space fidelity.
        """
        sig = signal.astype(np.complex64).copy()
        n = sig.shape[-1]
        n_segments = max(1, min(int(n_segments), n))
        seg_len = int(np.ceil(n / n_segments))

        # Per-segment phase offset proportional to shift + rotation
        base_phase = 2 * np.pi * (shift_y / max(n, 1))
        rot_phase = np.deg2rad(rotation_deg)

        for s in range(n_segments):
            a, b = s * seg_len, min((s + 1) * seg_len, n)
            # Alternating / progressive phase errors across shots
            shot_phase = base_phase * (s - n_segments / 2.0) + rot_phase * (s / n_segments)
            # Random-looking but deterministic from shift: mix with segment index
            shot_phase += 0.15 * shift_y * np.sin(s * 1.7)
            sig[..., a:b] = sig[..., a:b] * np.exp(1j * shot_phase).astype(np.complex64)

        # Mild global linear phase (translation residual)
        if abs(shift_y) > 0:
            freqs = np.fft.fftfreq(n, d=1.0 / n)
            k = np.fft.fft(sig, axis=-1)
            k = k * np.exp(-1j * 2 * np.pi * freqs * (0.25 * shift_y) / n)
            sig = np.fft.ifft(k, axis=-1).astype(np.complex64)

        return sig.astype(np.complex64)

    def _draw_corruption_mask(self, rng: np.random.RandomState):
        """Independent Bernoulli draws with ≥1 corruption guaranteed.

        Each source is included with probability ``p_apply`` (default 0.5),
        modulated by its relative weight so unequal configs still work.
        """
        apply_b0 = rng.random() < min(0.95, self.p_apply * (self.p_b0 * 3.0))
        apply_b1 = rng.random() < min(0.95, self.p_apply * (self.p_b1 * 3.0))
        apply_motion = rng.random() < min(0.95, self.p_apply * (self.p_motion * 3.0))
        if not (apply_b0 or apply_b1 or apply_motion):
            choice = int(rng.randint(3))
            apply_b0 = choice == 0
            apply_b1 = choice == 1
            apply_motion = choice == 2
        return apply_b0, apply_b1, apply_motion

    def corrupt_mrf_signal(
        self,
        signal: np.ndarray,
        rng: np.random.RandomState,
        tr_ms: Optional[float] = None,
        fa: Optional[np.ndarray] = None,
        tr_schedule: Optional[np.ndarray] = None,
        tissue: Optional[Dict[str, float]] = None,
        clean_signal: Optional[np.ndarray] = None,
        base_b0: float = 0.0,
        base_b1: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inject an entangled combination of B₀, B₁, and motion artifacts.

        If ``tissue``, ``fa``, and ``tr_schedule`` are provided, B1 is applied
        by re-simulation (physically correct). Otherwise falls back to
        post-hoc amplitude scaling for backward compatibility.
        """
        corrupted = signal.copy().astype(np.complex64)
        meta: Dict[str, Any] = {
            "b0_hz": 0.0,
            "b1_scale": 1.0,
            "motion_shift": 0,
            "motion_rot": 0.0,
            "b1_mode": "none",
            "b0_mode": "none",
        }

        apply_b0, apply_b1, apply_motion = self._draw_corruption_mask(rng)
        dwell = float(tr_ms) if tr_ms is not None else self.default_tr_ms

        b0_extra = 0.0
        b1_scale = 1.0

        if apply_b0:
            b0_extra = float(rng.uniform(*self.b0_range))
            meta["b0_hz"] = b0_extra

        if apply_b1:
            b1_scale = float(rng.uniform(*self.b1_range))
            meta["b1_scale"] = b1_scale

        # Prefer re-simulation when physics context is available
        can_resim = (
            tissue is not None
            and fa is not None
            and tr_schedule is not None
            and (apply_b0 or apply_b1)
        )
        if can_resim:
            resimulated = self.apply_b1_via_resimulate(
                tissue["t1"], tissue["t2"], tissue["m0"],
                fa, tr_schedule,
                base_b0 + b0_extra,
                base_b1 * b1_scale,
            )
            if clean_signal is not None:
                noise_residual = (
                    signal.astype(np.complex64) - clean_signal.astype(np.complex64)
                )
                corrupted = (resimulated + noise_residual).astype(np.complex64)
                meta["noise_preserved"] = True
            else:
                corrupted = resimulated.astype(np.complex64)
                meta["noise_preserved"] = False
            meta["b1_mode"] = "resim_fa" if apply_b1 else "resim_b0_only"
            meta["b0_mode"] = "resim" if apply_b0 else "none"
        else:
            if apply_b1:
                corrupted = self.apply_b1_transmit_scaling(corrupted, b1_scale)
                meta["b1_mode"] = "amplitude_legacy"
            if apply_b0:
                corrupted = self.apply_b0_off_resonance(corrupted, b0_extra, dwell_time_ms=dwell)
                meta["b0_mode"] = "phase_ramp"

        if apply_motion:
            # Exclude zero shift so "motion applied" is always detectable
            sy = 0
            while sy == 0:
                sy = int(rng.randint(-self.motion_shift[1], self.motion_shift[1] + 1))
            rot = float(rng.uniform(*self.motion_rot_range))
            if abs(rot) < 1.0:
                rot = 5.0 if rng.rand() < 0.5 else -5.0
            corrupted = self.apply_kspace_motion_artifact(
                corrupted, shift_y=sy, rotation_deg=rot, n_segments=self.n_motion_segments,
            )
            meta["motion_shift"] = sy
            meta["motion_rot"] = rot

        return corrupted.astype(np.complex64), meta

    def corrupt_mrs_spectrum(
        self, spectrum: np.ndarray, rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inject entangled corruptions into an MRS spectrum."""
        corrupted = spectrum.copy().astype(np.complex64)
        meta: Dict[str, Any] = {
            "b0_hz": 0.0, "b1_scale": 1.0, "motion_shift": 0, "motion_rot": 0.0,
            "b1_mode": "amplitude_legacy", "b0_mode": "phase_ramp",
        }

        apply_b0, apply_b1, apply_motion = self._draw_corruption_mask(rng)

        if apply_b0:
            # Spectral dwell: for 2000 Hz SW over 2048 pts → ~0.5 ms
            b0_shift = float(rng.uniform(*self.b0_range))
            corrupted = self.apply_b0_off_resonance(corrupted, b0_shift, dwell_time_ms=0.5)
            meta["b0_hz"] = b0_shift

        if apply_b1:
            b1_scale = float(rng.uniform(*self.b1_range))
            corrupted = self.apply_b1_transmit_scaling(corrupted, b1_scale)
            meta["b1_scale"] = b1_scale

        if apply_motion:
            sy = 0
            while sy == 0:
                sy = int(rng.randint(-self.motion_shift[1], self.motion_shift[1] + 1))
            rot = float(rng.uniform(*self.motion_rot_range))
            if abs(rot) < 1.0:
                rot = 5.0 if rng.rand() < 0.5 else -5.0
            corrupted = self.apply_kspace_motion_artifact(
                corrupted, shift_y=sy, rotation_deg=rot, n_segments=self.n_motion_segments,
            )
            meta["motion_shift"] = sy
            meta["motion_rot"] = rot

        return corrupted, meta

    # ── batch generation for Failure Forecasting split ─────────────────────

    def generate_failure_forecast_mrf(
        self, output_path: str, n_signals: Optional[int] = None,
    ) -> str:
        """Generate corrupted MRF signals for the Failure Forecasting split.

        Stores clean + corrupted signals, tissue parameters (ms), and
        per-sample corruption metadata.
        """
        mc = self.cfg.get("simulation", {}).get("mrf", {})
        n = n_signals or mc.get("n_signals", 50_000)
        vendors = mc.get("vendors", ["siemens", "philips", "ge"])
        fields = mc.get("field_strengths", [1.5, 3.0, 7.0])
        fa_vars = mc.get("fa_schedule_variants", 5)
        tr_vars = mc.get("tr_schedule_variants", 3)
        n_time = mc.get("n_timepoints", 1000)

        combos = [
            (v, f, fa, tr)
            for v in vendors for f in fields
            for fa in range(fa_vars) for tr in range(tr_vars)
        ]
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
            for k in range(per_combo):
                tasks.append((v, f, fa, tr, k))

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
            base_b0_ds = hf.create_dataset("base_b0_hz", shape=(total,), dtype=np.float32)
            base_b1_ds = hf.create_dataset("base_b1_scale", shape=(total,), dtype=np.float32)
            sim_param_ds = hf.create_dataset("sim_parameters", shape=(total, 2), dtype=np.float32)

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
                        base_b0_ds[i] = result["base_b0_hz"]
                        base_b1_ds[i] = result["base_b1_scale"]
                        sim_param_ds[i] = result["sim_parameters"]
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
                    base_b0_ds[i] = result["base_b0_hz"]
                    base_b1_ds[i] = result["base_b1_scale"]
                    sim_param_ds[i] = result["sim_parameters"]

            hf.attrs["n_signals"] = total
            hf.attrs["n_domains"] = len(combos)
            hf.attrs["n_timepoints"] = n_time
            hf.attrs["split"] = "failure_forecast"
            hf.attrs["t1_t2_units"] = "ms"
            hf.attrs["physics_version"] = "v3_stable_seed_noise_preserved"
            hf.attrs["seed_scheme"] = "stable_blake2b_v1"
            hf.attrs["target_parameter_definition"] = (
                "parameters are nominal tissue values; sim_parameters are field-adjusted values used in the signal"
            )
            hf.attrs["description"] = (
                "Entangled corruption training data for evidential failure forecasting. "
                "T1/T2 in milliseconds. B1 via flip-angle re-simulation when possible."
            )

        logger.info("Failure-forecast MRF saved → %s  (%d signals)", out, total)
        return str(out)

    def generate_failure_forecast_mrs(
        self, output_path: str, n_signals: Optional[int] = None,
    ) -> str:
        """Generate corrupted MRS spectra for the Failure Forecasting split."""
        mc = self.cfg.get("simulation", {}).get("mrs", {})
        n = n_signals or mc.get("n_signals", 10_000)
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
            for k in range(per_te):
                tasks.append((te, k))

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
                for i, task in enumerate(tqdm(tasks, desc="FailureForecastMRS")):
                    result = worker_fn(task)
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
            hf.attrs["seed_scheme"] = "stable_blake2b_v1"
            hf.attrs["physics_version"] = "v3_stable_seed_noise_preserved"
            hf.attrs["description"] = "Entangled corruption training data for evidential failure forecasting"

        logger.info("Failure-forecast MRS saved → %s  (%d signals)", out, total)
        return str(out)


# ── picklable top-level workers for multiprocessing ───────────────────────────

def _corrupt_mrf_worker(task, sim_cfg, corruptor_cfg, base_seed):
    """Generate one clean MRF sample and apply physically-aware corruptions."""
    if len(task) == 5:
        v, f, fa_var, tr_var, k = task
    else:
        v, f, fa_var, tr_var = task
        k = 0
    seed = stable_seed(base_seed, v, f, fa_var, tr_var, k)
    rng = np.random.RandomState(seed)

    sample = _generate_mrf_sample(seed, sim_cfg, v, f, fa_var, tr_var)
    corruptor = PhysicsCorruptor({"simulation": {"corruptor": corruptor_cfg}}, seed=seed)

    # Use the *same* FA/TR schedules as the clean simulation
    fa = sample["fa"]
    tr = sample["tr"]
    mean_tr = float(np.mean(tr))
    tissue = {
        "t1": float(sample["t1_sim_ms"]),
        "t2": float(sample["t2_sim_ms"]),
        "m0": float(sample["params"][2]),
    }

    corrupted, meta = corruptor.corrupt_mrf_signal(
        sample["signal"],
        rng,
        tr_ms=mean_tr,
        fa=fa,
        tr_schedule=tr,
        tissue=tissue,
        clean_signal=sample["clean_signal"],
        base_b0=float(sample["base_b0_hz"]),
        base_b1=float(sample["base_b1_scale"]),
    )

    return {
        "clean": sample["signal"],
        "corrupted": corrupted,
        "params": sample["params"],
        "domain_name": sample["domain_name"],
        "base_b0_hz": sample["base_b0_hz"],
        "base_b1_scale": sample["base_b1_scale"],
        "sim_parameters": np.array(
            [sample["t1_sim_ms"], sample["t2_sim_ms"]], dtype=np.float32
        ),
        "meta": meta,
    }



def _corrupt_mrs_worker(task, sim_cfg, corruptor_cfg, base_seed):
    if isinstance(task, tuple):
        te, k = task if len(task) == 2 else (task[0], 0)
    else:
        te, k = task, 0
    seed = stable_seed(base_seed, float(te), k)
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
