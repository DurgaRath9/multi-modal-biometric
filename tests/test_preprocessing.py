"""Tests for preprocessing and caching modules."""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

from src.data.cache import (
    build_metadata_table,
    get_dataset_summary,
    save_metadata_cache,
    load_metadata_cache,
)
from src.data.preprocessing import preprocess_single_image, preprocess_dataset_sequential


def _create_test_images(root: Path, num_persons: int = 2) -> list[str]:
    """Create test images and return their paths."""
    paths = []
    for pid in range(1, num_persons + 1):
        for modality in ["left", "fingerprint"]:
            d = root / str(pid) / modality
            d.mkdir(parents=True, exist_ok=True)
            img_path = d / "test.bmp"
            img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
            img.save(img_path)
            paths.append(str(img_path))
    return paths


class TestPreprocessing:
    def test_preprocess_single_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "1" / "left"
            root.mkdir(parents=True)
            img_path = root / "test.bmp"
            Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)).save(img_path)

            result = preprocess_single_image(str(img_path), target_size=64)
            assert result["array"].shape == (64, 64, 3)
            assert result["original_size"] == (100, 100)
            assert 0.0 <= result["array"].min()
            assert result["array"].max() <= 1.0

    def test_sequential_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _create_test_images(Path(tmpdir))
            results = preprocess_dataset_sequential(paths, target_size=32)
            assert len(results) == len(paths)
            for r in results:
                assert r["array"].shape == (32, 32, 3)


class TestCache:
    def test_build_metadata_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_images(Path(tmpdir), num_persons=3)
            table = build_metadata_table(tmpdir)
            assert isinstance(table, pa.Table)
            assert table.num_rows > 0
            assert "person_id" in table.column_names

    def test_save_and_load_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_images(Path(tmpdir), num_persons=2)
            table = build_metadata_table(tmpdir)

            cache_path = str(Path(tmpdir) / "cache" / "metadata.parquet")
            save_metadata_cache(table, cache_path)
            loaded = load_metadata_cache(cache_path)
            assert loaded.num_rows == table.num_rows

    def test_dataset_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_test_images(Path(tmpdir), num_persons=2)
            table = build_metadata_table(tmpdir)
            summary = get_dataset_summary(table)
            assert summary["num_persons"] == 2
            assert summary["total_images"] > 0
