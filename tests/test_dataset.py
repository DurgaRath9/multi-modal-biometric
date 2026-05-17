"""Tests for the multimodal biometric dataset."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import MultiModalBiometricDataset
from src.data.transforms import get_eval_transforms, get_train_transforms


def _create_test_dataset(root: Path, num_persons: int = 3, img_size: int = 64) -> None:
    """Create a minimal test dataset on disk."""
    for pid in range(1, num_persons + 1):
        person_dir = root / str(pid)

        # Create iris images
        for side in ["left", "right"]:
            iris_dir = person_dir / side
            iris_dir.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                img = Image.fromarray(
                    np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
                )
                img.save(iris_dir / f"iris_{i}.bmp")

        # Create fingerprint images
        fp_dir = person_dir / "fingerprint"
        fp_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            )
            img.save(fp_dir / f"fp_{i}.bmp")


class TestMultiModalDataset:
    def test_dataset_loads_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_test_dataset(root, num_persons=3)
            ds = MultiModalBiometricDataset(root_dir=str(root))
            assert len(ds) > 0

    def test_sample_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_test_dataset(root, num_persons=2)
            ds = MultiModalBiometricDataset(root_dir=str(root))
            sample = ds[0]

            assert "iris" in sample
            assert "fingerprint" in sample
            assert "label" in sample
            assert isinstance(sample["iris"], torch.Tensor)
            assert isinstance(sample["fingerprint"], torch.Tensor)
            assert isinstance(sample["label"], torch.Tensor)

    def test_transform_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_test_dataset(root, num_persons=2)
            tf = get_eval_transforms(image_size=64)
            ds = MultiModalBiometricDataset(
                root_dir=str(root), transform_iris=tf, transform_fingerprint=tf
            )
            sample = ds[0]
            # Check normalized tensor shape
            assert sample["iris"].shape == (3, 64, 64)
            assert sample["fingerprint"].shape == (3, 64, 64)

    def test_person_id_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_test_dataset(root, num_persons=5)
            ds = MultiModalBiometricDataset(root_dir=str(root), person_ids=[1, 3])
            person_ids = {s["person_id"] for s in [ds[i] for i in range(len(ds))]}
            assert person_ids <= {1, 3}

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            MultiModalBiometricDataset(root_dir="/nonexistent/path")

    def test_labels_zero_indexed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _create_test_dataset(root, num_persons=3)
            ds = MultiModalBiometricDataset(root_dir=str(root))
            labels = {ds[i]["label"].item() for i in range(len(ds))}
            assert 0 in labels  # person_id=1 → label=0

    def test_raises_for_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "not-a-person-folder").mkdir(parents=True, exist_ok=True)

            with pytest.raises(ValueError, match="No dataset samples found"):
                MultiModalBiometricDataset(root_dir=str(root))


class TestTransforms:
    def test_train_transforms_output_shape(self):
        tf = get_train_transforms(image_size=128)
        img = Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8))
        result = tf(img)
        assert result.shape == (3, 128, 128)

    def test_eval_transforms_output_shape(self):
        tf = get_eval_transforms(image_size=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = tf(img)
        assert result.shape == (3, 64, 64)

    def test_normalized_range(self):
        tf = get_eval_transforms(image_size=64)
        # Use a white image (255) - after ImageNet normalization, values will exceed 1.0
        img = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255)
        result = tf(img)
        # After ImageNet normalization of a white image, values should exceed [0,1] range
        assert result.max() > 1.0
