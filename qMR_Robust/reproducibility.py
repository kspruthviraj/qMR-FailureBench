"""Reproducibility helpers shared by simulation and training code.

The project previously used Python's built-in hash for simulation seeds.
That hash is intentionally salted per interpreter process, so it is not a
valid reproducibility primitive. This module provides a stable, serialisable
seed derivation function and an opt-in training seeding helper.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Derive the same 32-bit seed in every Python process.

    JSON encoding keeps the derivation independent of Python's hash randomisation.
    The function is deterministic across multiprocessing workers and does not
    mutate global RNG state.
    """
    payload = json.dumps(
        [int(base_seed), *parts],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**31 - 1)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and (when installed) PyTorch.

    deterministic=True is intended for validation runs, not necessarily for
    production training, because deterministic kernels can be slower.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
