"""Validated ingestion helpers for future raw MRF external validation.

The first supported public format is the DeliCS SPI-TGAS-MRF release.  DeliCS
stores raw multichannel k-space in NumPy arrays, so this module deliberately
stops at validation and provenance capture.  It does not silently convert
k-space into voxel fingerprints or invent failure labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


DELICS_N_READOUT = 2000
DELICS_N_COILS = 48
DELICS_N_TR = 500
DELICS_N_REPEATS = 48
DELICS_TI_MS = 20.0
DELICS_TE_MS = 0.7
DELICS_TR_MS = 12.0


def _load_numeric_npy(
    path: Path,
    *,
    name: str,
    expected_ndim: int,
    mmap_mode: Optional[str],
) -> np.ndarray:
    """Load a numeric NumPy array without allowing object deserialization."""
    try:
        array = np.load(str(path), mmap_mode=mmap_mode, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not load {name} from {path}: {exc}") from exc

    if array.ndim != expected_ndim:
        raise ValueError(
            f"{name} must have {expected_ndim} dimensions; got "
            f"shape={array.shape} from {path}"
        )
    if array.dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise ValueError(
            f"{name} must be numeric; got dtype={array.dtype} from {path}"
        )
    return array


def _optional_array(
    root: Path,
    names: tuple[str, ...],
    *,
    name: str,
    expected_ndim: int,
    mmap_mode: Optional[str],
) -> tuple[Optional[np.ndarray], Optional[Path]]:
    """Load the first present optional array under one of several filenames."""
    for filename in names:
        path = root / filename
        if path.exists():
            return (
                _load_numeric_npy(
                    path,
                    name=name,
                    expected_ndim=expected_ndim,
                    mmap_mode=mmap_mode,
                ),
                path,
            )
    return None, None


@dataclass(frozen=True)
class DeliCSCase:
    """Validated view of one extracted DeliCS case.

    raw_mrf uses the documented axis order
    (readout_samples, receive_channels, flattened_spiral_repetitions).
    The last axis contains n_tr * n_repeats acquisitions.  It remains
    flattened here because the public archive does not encode a reconstruction
    trajectory object in each case directory.
    """

    root: Path
    raw_mrf: np.ndarray
    gre_mrf: Optional[np.ndarray]
    noise: Optional[np.ndarray]
    raw_mrf_path: Path
    gre_mrf_path: Optional[Path]
    noise_path: Optional[Path]
    n_tr: int
    n_repeats: int

    @property
    def n_readout(self) -> int:
        return int(self.raw_mrf.shape[0])

    @property
    def n_coils(self) -> int:
        return int(self.raw_mrf.shape[1])

    @property
    def n_spirals(self) -> int:
        return int(self.raw_mrf.shape[2])

    def summary(self) -> dict:
        """Return JSON-serialisable provenance and validation metadata."""
        return {
            "dataset_family": "DeliCS SPI-TGAS-MRF+GRE",
            "case_root": str(self.root),
            "raw_mrf_path": str(self.raw_mrf_path),
            "raw_mrf_shape": list(self.raw_mrf.shape),
            "raw_mrf_dtype": str(self.raw_mrf.dtype),
            "raw_mrf_is_complex": bool(np.iscomplexobj(self.raw_mrf)),
            "axis_order": [
                "readout_samples",
                "receive_channels",
                "flattened_spiral_repetitions",
            ],
            "n_readout": self.n_readout,
            "n_coils": self.n_coils,
            "n_tr": int(self.n_tr),
            "n_repeats": int(self.n_repeats),
            "n_spirals": self.n_spirals,
            "protocol": {
                "field_strength_t": 3.0,
                "vendor": "GE",
                "scanner": "Premier",
                "coil_channels": self.n_coils,
                "sequence": "SPI-TGAS-MRF",
                "ti_ms": DELICS_TI_MS,
                "te_ms": DELICS_TE_MS,
                "tr_ms": DELICS_TR_MS,
                "flip_angle_range_deg": [10.0, 75.0],
                "acquisition_duration_min": 6.0,
            },
            "gre_prescan": {
                "present": self.gre_mrf is not None,
                "path": str(self.gre_mrf_path) if self.gre_mrf_path else None,
                "shape": list(self.gre_mrf.shape) if self.gre_mrf is not None else None,
                "dtype": str(self.gre_mrf.dtype) if self.gre_mrf is not None else None,
            },
            "noise_scan": {
                "present": self.noise is not None,
                "path": str(self.noise_path) if self.noise_path else None,
                "shape": list(self.noise.shape) if self.noise is not None else None,
                "dtype": str(self.noise.dtype) if self.noise is not None else None,
            },
            "labels": {
                "failure_labels_available": False,
                "reference_t1_t2_available": False,
                "b0_b1_maps_loaded": False,
            },
            "next_step": (
                "Reconstruct or otherwise derive voxel fingerprints with the "
                "published trajectory/reconstruction pipeline before model evaluation."
            ),
        }


def load_delics_case(
    case_dir: str | Path,
    *,
    mmap_mode: Optional[str] = "r",
    expected_coils: Optional[int] = DELICS_N_COILS,
    n_tr: int = DELICS_N_TR,
    n_repeats: int = DELICS_N_REPEATS,
    require_complex: bool = True,
) -> DeliCSCase:
    """Validate and memory-map one extracted DeliCS raw-MRF case.

    Parameters
    ----------
    case_dir:
        Directory containing raw_mrf.npy.  The archive must be extracted
        first; compressed tarballs cannot be memory-mapped safely.
    mmap_mode:
        Passed to numpy.load.  Keep the default "r" for multi-gigabyte
        cases.  Use None only when a full in-memory array is intentional.
    expected_coils:
        Expected receive-channel count.  Set to None only for exploratory
        inspection of a non-DeliCS format.
    n_tr, n_repeats:
        Expected protocol dimensions used to validate the flattened final axis.
    require_complex:
        Reject magnitude-only arrays so a future evaluation cannot silently
        discard phase information.
    """
    root = Path(case_dir)
    if root.suffixes[-2:] == [".tar", ".gz"]:
        raise ValueError(
            f"{root} is compressed; extract the case before loading so NumPy "
            "can memory-map raw_mrf.npy"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"DeliCS case directory not found: {root}")

    raw_path = root / "raw_mrf.npy"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Expected an extracted DeliCS case containing raw_mrf.npy."
        )

    raw = _load_numeric_npy(
        raw_path,
        name="raw_mrf",
        expected_ndim=3,
        mmap_mode=mmap_mode,
    )
    if require_complex and not np.iscomplexobj(raw):
        raise ValueError(
            f"raw_mrf must be complex-valued to preserve phase; got dtype={raw.dtype}"
        )
    if expected_coils is not None and raw.shape[1] != int(expected_coils):
        raise ValueError(
            f"raw_mrf coil axis mismatch: expected {expected_coils}, got {raw.shape[1]}"
        )
    expected_spirals = int(n_tr) * int(n_repeats)
    if raw.shape[2] != expected_spirals:
        raise ValueError(
            "raw_mrf final axis mismatch: expected "
            f"n_tr*n_repeats={expected_spirals}, got {raw.shape[2]}"
        )
    if min(int(n_tr), int(n_repeats)) <= 0:
        raise ValueError("n_tr and n_repeats must be positive")

    gre, gre_path = _optional_array(
        root,
        ("raw_gre.npy", "gre_mrf.npy"),
        name="GRE prescan",
        expected_ndim=3,
        mmap_mode=mmap_mode,
    )
    noise, noise_path = _optional_array(
        root,
        ("noise.npy",),
        name="noise scan",
        expected_ndim=2,
        mmap_mode=mmap_mode,
    )
    if gre is not None and gre.shape[1] != raw.shape[1]:
        raise ValueError(
            f"GRE prescan coil axis mismatch: expected {raw.shape[1]}, got {gre.shape[1]}"
        )
    if noise is not None and noise.shape[0] != raw.shape[1]:
        raise ValueError(
            f"noise scan coil axis mismatch: expected {raw.shape[1]}, got {noise.shape[0]}"
        )

    return DeliCSCase(
        root=root,
        raw_mrf=raw,
        gre_mrf=gre,
        noise=noise,
        raw_mrf_path=raw_path,
        gre_mrf_path=gre_path,
        noise_path=noise_path,
        n_tr=int(n_tr),
        n_repeats=int(n_repeats),
    )


def validate_delics_case(case_dir: str | Path, **kwargs) -> dict:
    """Load a DeliCS case and return a JSON-ready validation report."""
    return load_delics_case(case_dir, **kwargs).summary()


__all__ = [
    "DeliCSCase",
    "DELICS_N_READOUT",
    "DELICS_N_COILS",
    "DELICS_N_TR",
    "DELICS_N_REPEATS",
    "load_delics_case",
    "validate_delics_case",
]
