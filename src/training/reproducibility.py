"""Reproducibility utilities.

Ensures deterministic behavior across runs by seeding all
random number generators and configuring CuDNN.
"""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set seeds for all RNGs to ensure reproducible results.

    Covers: Python stdlib, NumPy, PyTorch CPU/GPU, CuDNN.
    Setting CUBLAS_WORKSPACE_CONFIG ensures deterministic CuBLAS operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Required for deterministic CUDA operations in PyTorch >= 1.8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info("All random seeds set to %d for reproducibility", seed)
