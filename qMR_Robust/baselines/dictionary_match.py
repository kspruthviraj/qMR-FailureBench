"""
Classical dictionary-matching baseline for synthetic MRF fingerprints.

Builds a discrete (T1, T2) dictionary via Bloch simulation and matches
corrupted signals by maximum absolute correlation (complex inner product).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from qMR_Robust.simulators.manager import bloch_simulate, _generate_fa_schedule, _generate_tr_schedule


def build_dictionary(
    t1_grid: np.ndarray,
    t2_grid: np.ndarray,
    fa: np.ndarray,
    tr: np.ndarray,
    b0: float = 0.0,
    b1: float = 1.0,
    m0: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dict_signals [M,L] complex, t1[M], t2[M])."""
    entries = []
    t1s, t2s = [], []
    n = len(fa)
    for t1 in t1_grid:
        for t2 in t2_grid:
            if t2 >= t1:
                continue
            sig = bloch_simulate(float(t1), float(t2), m0, fa, tr, b0, b1, n)
            # Normalize for correlation matching
            norm = np.linalg.norm(sig) + 1e-8
            entries.append((sig / norm).astype(np.complex64))
            t1s.append(t1)
            t2s.append(t2)
    return np.stack(entries), np.asarray(t1s, dtype=np.float32), np.asarray(t2s, dtype=np.float32)


def match_signals(
    signals: np.ndarray,
    dictionary: np.ndarray,
    t1_dict: np.ndarray,
    t2_dict: np.ndarray,
    batch: int = 256,
) -> Dict[str, np.ndarray]:
    """Match (N,L) complex signals to dictionary via max |<s,d>|."""
    n = len(signals)
    pred_t1 = np.zeros(n, dtype=np.float32)
    pred_t2 = np.zeros(n, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)

    # dictionary: (M, L)
    D = dictionary  # already normalized
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        S = signals[i0:i1]
        # normalize queries
        sn = np.linalg.norm(S, axis=1, keepdims=True) + 1e-8
        S = S / sn
        # correlation: (B, M)
        # use real part of Hermitian product magnitude
        corr = np.abs(S @ D.conj().T)
        idx = corr.argmax(axis=1)
        pred_t1[i0:i1] = t1_dict[idx]
        pred_t2[i0:i1] = t2_dict[idx]
        scores[i0:i1] = corr[np.arange(i1 - i0), idx]
    return {"t1": pred_t1, "t2": pred_t2, "match_score": scores}


def default_grids(coarse: bool = True):
    if coarse:
        t1 = np.concatenate([
            np.arange(200, 1000, 40),
            np.arange(1000, 3000, 80),
        ]).astype(np.float32)
        t2 = np.concatenate([
            np.arange(10, 100, 5),
            np.arange(100, 500, 20),
        ]).astype(np.float32)
    else:
        t1 = np.arange(100, 3000, 20).astype(np.float32)
        t2 = np.arange(10, 500, 5).astype(np.float32)
    return t1, t2
