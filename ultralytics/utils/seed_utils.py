"""Reproducibility utilities for YOLOv14.

Centralised seed setting so that every component (Python ``random``,
``numpy``, ``torch`` CPU & CUDA) shares a single deterministic state.
This is essential for fair ablation studies and for reproducing the
results reported in the NeurIPS 2024 paper.

Usage::

    from ultralytics.utils.seed_utils import set_seed, get_seed
    set_seed(42)          # seed everything
    seed = get_seed()     # retrieve current seed
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ("set_seed", "get_seed", "seed_worker")

_GLOBAL_SEED: int = 42


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> int:
    """Seed all RNG sources for reproducibility.

    Parameters
    ----------
    seed : int
        Master seed value (default 42, matching common ML convention).
    deterministic : bool
        If True, force cuDNN to use deterministic algorithms.
        May reduce performance but guarantees reproducibility.
    benchmark : bool
        If True, enable cuDNN auto-tuner (incompatible with deterministic).
        Set to True for production inference where input sizes are fixed.

    Returns
    -------
    int
        The seed that was set (echoed for logging).
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def get_seed() -> int:
    """Return the current global seed."""
    return _GLOBAL_SEED


def seed_worker(worker_id: int) -> None:
    """Callable for DataLoader ``worker_init_fn``.

    Ensures each worker gets a deterministic but distinct seed derived
    from the global seed, preventing duplicate augmentations across
    workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)