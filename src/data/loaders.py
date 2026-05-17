"""DataLoader factory with performance-optimized defaults.

Encapsulates best practices for efficient GPU-bound training:
- pin_memory for faster host-to-device transfers
- persistent_workers to avoid re-spawning overhead
- prefetch_factor for overlapping data loading with computation
"""

import logging
from typing import Optional

from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    train_split: float = 0.8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with optimized settings.

    Args:
        dataset: Full dataset to split.
        batch_size: Samples per batch.
        num_workers: Parallel data loading workers.
            Higher values overlap CPU preprocessing with GPU computation.
        pin_memory: Pre-allocate samples in pinned (page-locked) memory for
            faster CPU→GPU transfers via DMA.
        persistent_workers: Keep worker processes alive between epochs to
            avoid re-initialization overhead.
        prefetch_factor: Number of batches prefetched per worker, enabling
            overlap of data loading with model forward/backward passes.
        train_split: Fraction of data used for training.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    import torch

    total = len(dataset)
    if total == 0:
        raise ValueError(
            "Dataset is empty (0 samples). Check data.root_dir and dataset folder structure."
        )

    train_size = int(total * train_split)
    if train_size == 0:
        raise ValueError(
            f"Training split produced 0 samples (total={total}, train_split={train_split}). "
            "Increase dataset size or train_split."
        )

    val_size = total - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # pin_memory only helps with GPU; disable on CPU to avoid warnings
    use_pin_memory = pin_memory and torch.cuda.is_available()

    # Disable persistent_workers when num_workers=0 (single-process loading)
    use_persistent = persistent_workers and num_workers > 0
    use_prefetch: Optional[int] = prefetch_factor if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=use_prefetch,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=use_prefetch,
        drop_last=False,
    )

    logger.info(
        "DataLoaders created: train=%d batches, val=%d batches, workers=%d",
        len(train_loader),
        len(val_loader),
        num_workers,
    )
    return train_loader, val_loader
