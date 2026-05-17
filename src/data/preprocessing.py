"""Parallel preprocessing with Ray for biometric images.

Scales from local sequential execution to multi-node clusters
when Ray is available.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_single_image(image_path: str, target_size: int = 128) -> dict:
    """Preprocess a single image: resize, normalize, extract metadata.

    Returns dict with processed array and metadata.
    """
    path = Path(image_path)
    img = Image.open(path).convert("RGB")
    original_size = img.size
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    return {
        "path": str(path),
        "array": arr,
        "original_size": original_size,
        "target_size": (target_size, target_size),
        "person_id": path.parent.parent.name,
        "modality": path.parent.name,
    }


def preprocess_dataset_sequential(image_paths: list[str], target_size: int = 128) -> list[dict]:
    """Preprocess all images sequentially (baseline)."""
    results = []
    for p in image_paths:
        results.append(preprocess_single_image(p, target_size))
    return results


def preprocess_dataset_parallel(
    image_paths: list[str], target_size: int = 128, num_workers: int = 4
) -> list[dict]:
    """Preprocess all images in parallel using Ray.

    Falls back to sequential processing if Ray is unavailable.
    """
    try:
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=num_workers, logging_level=logging.WARNING)

        @ray.remote
        def _preprocess_remote(path: str, size: int) -> dict:
            return preprocess_single_image(path, size)

        futures = [_preprocess_remote.remote(p, target_size) for p in image_paths]
        results = ray.get(futures)
        return results

    except ImportError:
        logger.warning("Ray not installed, falling back to sequential preprocessing")
        return preprocess_dataset_sequential(image_paths, target_size)
