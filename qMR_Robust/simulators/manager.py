"""
SimulationManager — Orchestrates synthetic qMRI data generation.

Supports two modes:
  1. Pre-compute: Generate large HDF5 files offline (multi-process).
  2. On-the-fly:  Generate batches during training (single-process).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

VENDOR_BIAS = {
    "siemens": {"b0_hz": 0.0, "b1_scale": 1.00, "noise_scale": 1.00},
    "philips": {"b0_hz": 5.0, "b1_scale": 0.95, "noise_scale": 1.10},
    "ge":      {"b0_hz": -3.0, "b1_scale": 1.05, "noise_scale": 1.20},
}

FIELD_FACTORS = {
    1.5: {"t1_scale": 0.85, "t2_scale": 1.10, "snr_scale": 0.60},
    3.0: {"t1_scale": 1.00, "t2_scale": 1.00, "snr_scale": 1.00},
    7.0: {"t1_scale": 1.30, "t2_scale": 0.75, "snr_scale": 1.80},
}

METABOLITE_SHIFTS_PPM = {
    "NAA": 2.02, "Glu": 2.04, "Gln": 2.12, "GABA": 3.01,
    "Cr": 3.03, "Cho": 3.22, "mI": 3.56, "Ins": 3.56,
}

METABOLITE_CONC_RANGE = {
    "NAA": (5.0, 15.0), "Glu": (5.0, 15.0), "Gln": (2.0, 8.0), "GABA": (0.5, 3.0),
    "Cr": (5.0, 12.0), "Cho": (0.5, 3.0), "mI": (3.0, 10.0), "Ins": (3.0, 10.0),
}


def _generate_fa_schedule(variant: int, n: int, rng: np.random.RandomState) -> np.ndarray:
    if variant == 0:
        fa = np.linspace(5, 70, n) + rng.randn(n) * 2
    elif variant == 1:
        fa = np.abs(rng.randn(n) * 15 + 30)
    elif variant == 2:
        seg = n // 3
        fa = np.concatenate([np.ones(seg)*10, np.ones(seg)*50, np.ones(n-2*seg)*30]).astype(float)
        fa += rng.randn(n) * 2
    elif variant == 3:
        fa = 20 + 20 * np.sin(np.linspace(0, 4*np.pi, n)) + rng.randn(n)
    else:
        fa = rng.uniform(5, 75, n)
    return np.clip(fa, 1.0, 90.0)


def _generate_tr_schedule(variant: int, n: int, rng: np.random.RandomState) -> np.ndarray:
    if variant == 0:
        return np.clip(np.ones(n) * 12.0 + rng.randn(n) * 0.5, 5, 50)
    if variant == 1:
        return np.clip(np.linspace(8, 20, n) + rng.randn(n) * 0.3, 5, 50)
    return np.clip(np.ones(n) * 15.0, 5, 50)


def bloch_simulate(
    t1: float, t2: float, m0: float,
    fa: np.ndarray, tr: np.ndarray,
    b0_shift: float, b1_scale: float, n: int,
) -> np.ndarray:
    signal = np.zeros(n, dtype=np.complex128)
    mz = m0
    fa_rad = np.deg2rad(fa) * b1_scale
    e1 = np.exp(-tr / max(t1, 1e-6))
    te = 2.0
    e2_te = np.exp(-te / max(t2, 1e-6))
    b0_phase = np.exp(1j * 2 * np.pi * b0_shift * te / 1000.0)

    for i in range(n):
        mz_pre = mz
        mz = mz_pre * np.cos(fa_rad[i])
        mxy = mz_pre * np.sin(fa_rad[i]) * e2_te * b0_phase
        mz = mz * e1[i] + m0 * (1 - e1[i])
        signal[i] = mxy
    return signal


def _generate_mrf_sample(
    seed: int, cfg: dict, vendor: str, field: float, fa_var: int, tr_var: int,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    vr = cfg["mrf"]

    t1 = rng.uniform(*vr["t1_range"])
    t2 = min(rng.uniform(*vr["t2_range"]), t1 * 0.95)
    m0 = rng.uniform(*vr["m0_range"])

    ff = FIELD_FACTORS[field]
    vb = VENDOR_BIAS[vendor]
    t1_s = t1 * ff["t1_scale"]
    t2_s = t2 * ff["t2_scale"]

    n = vr.get("n_timepoints", 1000)
    fa = _generate_fa_schedule(fa_var, n, rng)
    tr = _generate_tr_schedule(tr_var, n, rng)

    b0 = vb["b0_hz"] + rng.uniform(*vr["b0_shift_range"])
    b1 = vb["b1_scale"] * rng.uniform(*vr["b1_scale_range"])

    sig = bloch_simulate(t1_s, t2_s, m0, fa, tr, b0, b1, n)

    snr = rng.uniform(*vr["snr_range"]) * ff["snr_scale"]
    noise_std = np.abs(sig).max() / max(snr, 1.0)
    noise = (rng.randn(n) + 1j * rng.randn(n)) * noise_std
    sig = (sig + noise).astype(np.complex64)

    domain_name = f"{vendor}_fa{fa_var}_tr{tr_var}_{field}T"
    return {
        "signal": sig,
        "params": np.array([t1, t2, m0], dtype=np.float32),
        "domain_name": domain_name,
        "vendor": vendor,
        "field_strength": field,
    }


def _lorentzian(n: int, sw: float, center_hz: float, fwhm_hz: float) -> np.ndarray:
    freq = np.linspace(-sw / 2, sw / 2, n)
    gamma = fwhm_hz / 2.0
    return (gamma**2 / ((freq - center_hz) ** 2 + gamma**2)).astype(np.float32)


def _generate_mrs_sample(
    seed: int, cfg: dict, te: float,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    mr = cfg["mrs"]
    metabolites = mr["metabolites"]
    n_pts = mr.get("n_points", 2048)
    sw = mr.get("spectral_width", 2000.0)
    field = mr.get("field_strength", 3.0)

    larmor = 42.577e6 * field * 1e-6

    concentrations = np.array(
        [rng.uniform(*METABOLITE_CONC_RANGE.get(m, (1.0, 10.0))) for m in metabolites],
        dtype=np.float32,
    )

    linewidth = rng.uniform(*mr["linewidth_range"])
    spectrum = np.zeros(n_pts, dtype=np.float32)

    for met, conc in zip(metabolites, concentrations):
        ppm = METABOLITE_SHIFTS_PPM.get(met, 3.0)
        center_hz = ppm * larmor
        basis = _lorentzian(n_pts, sw, center_hz, linewidth)
        j_mod = 1.0
        if met in ("GABA", "Glu", "Gln") and te > 0:
            j_mod = abs(np.cos(np.pi * 7.5 * te / 1000.0))
        spectrum += conc * basis * j_mod

    phase_rad = rng.uniform(*mr.get("phase_error_range", [-30, 30])) * np.pi / 180
    spectrum_c = spectrum * np.exp(1j * phase_rad).astype(np.complex64)

    snr = rng.uniform(*mr["snr_range"])
    noise_std = np.abs(spectrum_c).max() / max(snr, 1.0)
    noise = (rng.randn(n_pts) + 1j * rng.randn(n_pts)) * noise_std
    spectrum_noisy = (spectrum_c + noise).astype(np.complex64)

    domain_name = f"TE{te}"
    return {
        "signal": spectrum_noisy,
        "concentrations": concentrations,
        "domain_name": domain_name,
        "te": te,
    }


class SimulationManager:
    """Public API for synthetic qMRI data generation."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sim_cfg = cfg.get("simulation", {})
        self.seed = cfg.get("project", {}).get("seed", 42)
        self.n_workers = self.sim_cfg.get("mrf", {}).get("n_workers", mp.cpu_count())

    def generate_mrf(self, output_path: str, n_signals: Optional[int] = None):
        mc = self.sim_cfg.get("mrf", {})
        n = n_signals or mc.get("n_signals", 100_000)
        vendors = mc.get("vendors", ["siemens", "philips", "ge"])
        fields = mc.get("field_strengths", [1.5, 3.0, 7.0])
        fa_vars = mc.get("fa_schedule_variants", 5)
        tr_vars = mc.get("tr_schedule_variants", 3)

        combos = [(v, f, fa, tr) for v in vendors for f in fields
                  for fa in range(fa_vars) for tr in range(tr_vars)]
        per_combo = max(1, n // len(combos))
        total = per_combo * len(combos)
        n_time = mc.get("n_timepoints", 1000)

        logger.info("Generating %d MRF signals across %d domains …", total, len(combos))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        tasks = []
        for v, f, fa, tr in combos:
            for _ in range(per_combo):
                tasks.append((v, f, fa, tr))

        worker_fn = partial(_mrf_worker, cfg=self.sim_cfg, base_seed=self.seed)

        with h5py.File(out, "w") as hf:
            sig_ds = hf.create_dataset(
                "signals", shape=(total, n_time), dtype=np.complex64,
                chunks=(min(1024, total), n_time), compression="gzip", compression_opts=1,
            )
            param_ds = hf.create_dataset("parameters", shape=(total, 3), dtype=np.float32)
            dom_ds = hf.create_dataset("domain_labels", shape=(total,), dtype="S64")

            if self.n_workers > 1:
                ctx = mp.get_context("spawn")
                with ctx.Pool(self.n_workers) as pool:
                    for i, result in enumerate(
                        tqdm(pool.imap(worker_fn, tasks, chunksize=64),
                             total=total, desc="MRF")
                    ):
                        sig_ds[i] = result["signal"]
                        param_ds[i] = result["params"]
                        dom_ds[i] = result["domain_name"].encode()
            else:
                for i, task in enumerate(tqdm(tasks, desc="MRF")):
                    result = _mrf_worker(task, self.sim_cfg, self.seed + i)
                    sig_ds[i] = result["signal"]
                    param_ds[i] = result["params"]
                    dom_ds[i] = result["domain_name"].encode()

            hf.attrs["n_signals"] = total
            hf.attrs["n_domains"] = len(combos)
            hf.attrs["n_timepoints"] = n_time
            hf.attrs["vendors"] = vendors
            hf.attrs["field_strengths"] = fields

        logger.info("MRF dictionary saved → %s  (%d signals)", out, total)
        return str(out)

    def generate_mrs(self, output_path: str, n_signals: Optional[int] = None):
        mc = self.sim_cfg.get("mrs", {})
        n = n_signals or mc.get("n_signals", 100_000)
        te_values = mc.get("te_values", [30.0, 68.0, 80.0, 144.0])
        n_pts = mc.get("n_points", 2048)
        metabolites = mc.get("metabolites", list(METABOLITE_SHIFTS_PPM.keys()))
        n_met = len(metabolites)

        per_te = max(1, n // len(te_values))
        total = per_te * len(te_values)

        logger.info("Generating %d MRS spectra across %d TE values …", total, len(te_values))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        tasks = []
        for te in te_values:
            for _ in range(per_te):
                tasks.append(te)

        worker_fn = partial(_mrs_worker, cfg=self.sim_cfg, base_seed=self.seed)

        with h5py.File(out, "w") as hf:
            spec_ds = hf.create_dataset(
                "spectra", shape=(total, n_pts), dtype=np.complex64,
                chunks=(min(1024, total), n_pts), compression="gzip", compression_opts=1,
            )
            conc_ds = hf.create_dataset("concentrations", shape=(total, n_met), dtype=np.float32)
            dom_ds = hf.create_dataset("domain_labels", shape=(total,), dtype="S32")

            if self.n_workers > 1:
                ctx = mp.get_context("spawn")
                with ctx.Pool(self.n_workers) as pool:
                    for i, result in enumerate(
                        tqdm(pool.imap(worker_fn, tasks, chunksize=64),
                             total=total, desc="MRS")
                    ):
                        spec_ds[i] = result["signal"]
                        conc_ds[i] = result["concentrations"]
                        dom_ds[i] = result["domain_name"].encode()
            else:
                for i, te_val in enumerate(tqdm(tasks, desc="MRS")):
                    result = _mrs_worker(te_val, self.sim_cfg, self.seed + i)
                    spec_ds[i] = result["signal"]
                    conc_ds[i] = result["concentrations"]
                    dom_ds[i] = result["domain_name"].encode()

            hf.attrs["n_signals"] = total
            hf.attrs["metabolites"] = metabolites
            hf.attrs["n_points"] = n_pts
            hf.attrs["te_values"] = te_values

        logger.info("MRS spectra saved → %s  (%d signals)", out, total)
        return str(out)

    def generate_batch_mrf(
        self, batch_size: int, vendor: str = "random", field: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        signals, params, domains = [], [], []
        for _ in range(batch_size):
            v = vendor if vendor != "random" else np.random.choice(
                self.sim_cfg.get("mrf", {}).get("vendors", ["siemens"])
            )
            f = field if field > 0 else np.random.choice(
                self.sim_cfg.get("mrf", {}).get("field_strengths", [3.0])
            )
            result = _generate_mrf_sample(
                np.random.randint(2**31), self.sim_cfg, v, f,
                np.random.randint(5), np.random.randint(3),
            )
            signals.append(result["signal"])
            params.append(result["params"])
            domains.append(result["domain_name"])
        return np.stack(signals), np.stack(params), domains

    def generate_batch_mrs(
        self, batch_size: int, te: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        te_values = self.sim_cfg.get("mrs", {}).get("te_values", [68.0])
        signals, concs, domains = [], [], []
        for _ in range(batch_size):
            t = te if te > 0 else np.random.choice(te_values)
            result = _generate_mrs_sample(np.random.randint(2**31), self.sim_cfg, t)
            signals.append(result["signal"])
            concs.append(result["concentrations"])
            domains.append(result["domain_name"])
        return np.stack(signals), np.stack(concs), domains


def _mrf_worker(task, cfg, base_seed):
    v, f, fa, tr = task
    seed = base_seed ^ hash((v, f, fa, tr)) % (2**31)
    return _generate_mrf_sample(seed, cfg, v, f, fa, tr)


def _mrs_worker(te, cfg, base_seed):
    seed = base_seed ^ hash(te) % (2**31)
    return _generate_mrs_sample(seed, cfg, te)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic qMRI data")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--modality", choices=["mrf", "mrs", "both"], default="both")
    parser.add_argument("--n-signals", type=int, default=None)
    parser.add_argument("--output-mrf", default=None)
    parser.add_argument("--output-mrs", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mgr = SimulationManager(cfg)
    paths = cfg.get("paths", {})

    if args.modality in ("mrf", "both"):
        out = args.output_mrf or paths.get("synthetic_mrf", "data/synthetic/mrf_dictionary.h5")
        mgr.generate_mrf(out, args.n_signals)

    if args.modality in ("mrs", "both"):
        out = args.output_mrs or paths.get("synthetic_mrs", "data/synthetic/mrs_spectra.h5")
        mgr.generate_mrs(out, args.n_signals)


if __name__ == "__main__":
    main()
