"""
Unit-safe data loaders for synthetic MRF and real qMRLab VFA.

CRITICAL UNIT CONVENTION
------------------------
All tissue parameters in this project are stored and evaluated in **milliseconds**
for T1/T2. qMRLab VFA T1 maps are provided in **seconds** and MUST be converted
with ``* 1000.0`` at load time.

Real VFA evaluation is a *cross-sequence OOD* protocol: the model was trained on
synthetic MRF fingerprints (1000 TRs), while VFA provides only a few flip-angle
samples. We zero-pad (not tile-repeat with fake phase) and mark the protocol
explicitly so results are not over-claimed as in-vivo MRF transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Typical brain T1 mean in ms; used as a safety check
T1_MS_MIN_MEAN = 100.0
T1_MS_MAX_MEAN = 5000.0


def assert_t1_units_ms(t1: np.ndarray, name: str = "T1") -> None:
    """Raise if array looks like seconds instead of milliseconds."""
    t1 = np.asarray(t1, dtype=np.float64)
    finite = t1[np.isfinite(t1) & (t1 > 0)]
    if finite.size == 0:
        raise ValueError(f"{name}: no positive finite values")
    mean = float(finite.mean())
    if mean < T1_MS_MIN_MEAN:
        raise ValueError(
            f"{name} mean={mean:.3f} looks like SECONDS, not milliseconds. "
            f"Convert with *1000 before evaluation."
        )
    if mean > T1_MS_MAX_MEAN:
        raise ValueError(
            f"{name} mean={mean:.1f} is implausibly large for brain T1 in ms."
        )


def complex_to_2ch(signals: np.ndarray) -> np.ndarray:
    """(N, L) complex → (N, 2, L) float32 real/imag."""
    return np.stack([signals.real, signals.imag], axis=1).astype(np.float32)


def load_training_norm(
    h5_path: str | Path,
    train_ratio: float = 0.8,
    n_params: int = 2,
    indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) of training-split parameters in raw physical units (ms)."""
    hf = h5py.File(h5_path, "r")
    n = int(hf.attrs["n_signals"])
    n_train = int(n * train_ratio)
    if indices is None:
        params = hf["parameters"][:n_train, :n_params].astype(np.float32)
    else:
        params = hf["parameters"][np.asarray(indices, dtype=np.int64), :n_params].astype(np.float32)
    hf.close()
    mean = params.mean(axis=0)
    std = params.std(axis=0) + 1e-8
    assert_t1_units_ms(params[:, 0], "synthetic training T1")
    return mean, std


class MRFMetaDataset(Dataset):
    """HDF5-backed corrupted MRF dataset with optional target normalization."""

    def __init__(
        self,
        h5_path: str | Path,
        split: str = "train",
        train_ratio: float = 0.8,
        n_params: int = 2,
        load_corruption_meta: bool = False,
        indices: Optional[np.ndarray] = None,
    ):
        hf = h5py.File(h5_path, "r")
        n = int(hf.attrs["n_signals"])
        n_train = int(n * train_ratio)
        if indices is None:
            s, e = (0, n_train) if split == "train" else (n_train, n)
            selection = slice(s, e)
        else:
            selection = np.asarray(indices, dtype=np.int64)

        sig = hf["corrupted_signals"][selection]
        self.signals = complex_to_2ch(sig)
        self.params = hf["parameters"][selection, :n_params].astype(np.float32)

        self.b0 = self.b1 = self.motion = None
        if load_corruption_meta:
            self.b0 = hf["b0_hz_applied"][selection].astype(np.float32)
            self.b1 = hf["b1_scale_applied"][selection].astype(np.float32)
            self.motion = hf["motion_shift_applied"][selection].astype(np.float32)
        hf.close()

        if split == "train":
            self.mean = self.params.mean(0)
            self.std = self.params.std(0) + 1e-8
            assert_t1_units_ms(self.params[:, 0], "MRFMetaDataset train T1")
        else:
            self.mean = np.zeros(n_params, dtype=np.float32)
            self.std = np.ones(n_params, dtype=np.float32)

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, i: int):
        x = torch.from_numpy(self.signals[i])
        y = (torch.from_numpy(self.params[i]) - self.mean) / self.std
        if self.b0 is None:
            return x, y
        return (
            x,
            y,
            float(self.b0[i]),
            float(self.b1[i]),
            float(self.motion[i]),
        )


@dataclass
class RealVFAData:
    """Cross-sequence OOD evaluation pack (VFA → MRF-shaped input)."""

    signals: np.ndarray  # (N, 2, L) float32
    t1_ms: np.ndarray  # (N,) milliseconds
    coords: Optional[np.ndarray] = None  # (N, 2) voxel indices
    b1_map: Optional[np.ndarray] = None  # (N,) relative B1 if available
    protocol: str = "vfa_cross_sequence_zeropad"
    n_fa_original: int = 0
    seq_len: int = 1000


def load_qmrlab_vfa(
    qmrlab_dir: str | Path,
    seq_len: int = 1000,
    pad_mode: str = "zeropad",
    require_mask: bool = True,
) -> RealVFAData:
    """Load qMRLab VFA T1 data with correct units (seconds → milliseconds).

    Parameters
    ----------
    qmrlab_dir :
        Path to ``vfa_t1_data`` containing VFAData.nii.gz, Mask.nii.gz,
        FitResults/T1.nii.gz, optional B1map.nii.gz.
    seq_len :
        Target temporal length expected by MRF models (default 1000).
    pad_mode :
        ``zeropad`` (recommended, honest) or ``tile`` (legacy; discouraged).
        Fake phase injection is never used.
    """
    import nibabel as nib

    qmrlab_dir = Path(qmrlab_dir)
    vfa_path = qmrlab_dir / "VFAData.nii.gz"
    t1_path = qmrlab_dir / "FitResults" / "T1.nii.gz"
    mask_path = qmrlab_dir / "Mask.nii.gz"
    b1_path = qmrlab_dir / "B1map.nii.gz"

    if not vfa_path.exists():
        raise FileNotFoundError(f"Missing VFA data: {vfa_path}")
    if not t1_path.exists():
        raise FileNotFoundError(f"Missing T1 map: {t1_path}")

    vfa_data = nib.load(str(vfa_path)).get_fdata()
    t1_gt = nib.load(str(t1_path)).get_fdata()  # seconds in qMRLab
    mask = (
        nib.load(str(mask_path)).get_fdata()
        if mask_path.exists()
        else np.ones(t1_gt.shape[:2] if t1_gt.ndim >= 2 else t1_gt.shape)
    )
    b1_map = nib.load(str(b1_path)).get_fdata() if b1_path.exists() else None

    # Slice selection
    if vfa_data.ndim == 4:
        vfa_slice = vfa_data[:, :, 0, :]
    elif vfa_data.ndim == 3:
        # (X, Y, n_fa) or (X, Y, Z)
        vfa_slice = vfa_data
    else:
        raise ValueError(f"Unexpected VFA shape: {vfa_data.shape}")

    t1_slice = t1_gt if t1_gt.ndim == 2 else t1_gt[:, :, 0]
    mask_slice = mask if mask.ndim == 2 else mask[:, :, 0]
    b1_slice = None
    if b1_map is not None:
        b1_slice = b1_map if b1_map.ndim == 2 else b1_map[:, :, 0]

    n_fa = vfa_slice.shape[-1]
    voxels, t1_values, coords, b1_values = [], [], [], []

    for ix in range(vfa_slice.shape[0]):
        for iy in range(vfa_slice.shape[1]):
            if require_mask and mask_slice[ix, iy] < 0.5:
                continue
            sig = np.asarray(vfa_slice[ix, iy, :], dtype=np.float64)
            sig_max = np.abs(sig).max()
            if sig_max < 1e-6:
                continue
            t1_s = float(t1_slice[ix, iy])
            if not np.isfinite(t1_s) or t1_s <= 0:
                continue

            # Normalize magnitude curve
            sig_norm = (sig / sig_max).astype(np.float32)

            # Build length-seq_len real channel; imag = 0 (magnitude VFA)
            if pad_mode == "tile":
                mag = np.tile(sig_norm, seq_len // n_fa + 1)[:seq_len]
            else:  # zeropad — honest: only first n_fa samples are data
                mag = np.zeros(seq_len, dtype=np.float32)
                mag[: min(n_fa, seq_len)] = sig_norm[: min(n_fa, seq_len)]

            phase = np.zeros(seq_len, dtype=np.float32)
            voxels.append(np.stack([mag, phase], axis=0))
            # CRITICAL: convert seconds → milliseconds
            t1_values.append(t1_s * 1000.0)
            coords.append((ix, iy))
            if b1_slice is not None:
                b1_values.append(float(b1_slice[ix, iy]))

    if not voxels:
        raise RuntimeError(f"No valid voxels found in {qmrlab_dir}")

    t1_ms = np.asarray(t1_values, dtype=np.float32)
    assert_t1_units_ms(t1_ms, "qMRLab VFA T1 (after s→ms conversion)")

    return RealVFAData(
        signals=np.stack(voxels).astype(np.float32),
        t1_ms=t1_ms,
        coords=np.asarray(coords, dtype=np.int32),
        b1_map=np.asarray(b1_values, dtype=np.float32) if b1_values else None,
        protocol=f"vfa_cross_sequence_{pad_mode}",
        n_fa_original=n_fa,
        seq_len=seq_len,
    )
