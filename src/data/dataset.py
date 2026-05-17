"""Dataset abstraction for multimodal biometric data (Iris + Fingerprint).

Supports lazy loading, configurable transforms, and missing modality handling.
Designed to be extensible for additional modalities (e.g., radar, temporal sensors).
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class MultiModalBiometricDataset(Dataset):
    """PyTorch Dataset for multimodal iris + fingerprint biometric data.

    Dataset layout expected:
        root_dir/
        ├── 1/
        │   ├── fingerprint/   (10 images)
        │   ├── left/          (5 left-eye iris images)
        │   └── right/         (5 right-eye iris images)
        ├── 2/
        ...

    Each sample pairs one iris image with one fingerprint image for a person.
    """

    IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        transform_iris: Optional[transforms.Compose] = None,
        transform_fingerprint: Optional[transforms.Compose] = None,
        person_ids: Optional[list[int]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform_iris = transform_iris
        self.transform_fingerprint = transform_fingerprint
        self.samples: list[dict] = []

        self._scan_dataset(person_ids)
        logger.info("Loaded %d samples from %s", len(self.samples), self.root_dir)

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    def _scan_dataset(self, person_ids: Optional[list[int]] = None) -> None:
        """Scan directory tree and build sample index."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        person_dirs = sorted(
            [d for d in self.root_dir.iterdir() if d.is_dir()],
            key=lambda d: int(d.name) if d.name.isdigit() else d.name,
        )

        for person_dir in person_dirs:
            person_id = int(person_dir.name) if person_dir.name.isdigit() else None
            if person_id is None:
                continue
            if person_ids is not None and person_id not in person_ids:
                continue

            label = person_id - 1  # zero-indexed label

            iris_images = self._collect_iris_images(person_dir)
            fingerprint_images = self._collect_fingerprint_images(person_dir)

            if not iris_images and not fingerprint_images:
                logger.warning("No images found for person %d", person_id)
                continue

            # Pair iris and fingerprint images: zip to shortest, or handle missing
            max_pairs = max(len(iris_images), len(fingerprint_images), 1)
            for i in range(max_pairs):
                sample = {
                    "person_id": person_id,
                    "label": label,
                    "iris_path": iris_images[i % len(iris_images)] if iris_images else None,
                    "fingerprint_path": (
                        fingerprint_images[i % len(fingerprint_images)]
                        if fingerprint_images
                        else None
                    ),
                }
                self.samples.append(sample)

        if not self.samples:
            subdirs = [d.name for d in self.root_dir.iterdir() if d.is_dir()][:20]
            raise ValueError(
                f"No dataset samples found at {self.root_dir}. "
                f"Subdirectories present: {subdirs}"
            )

    def _collect_iris_images(self, person_dir: Path) -> list[Path]:
        """Collect iris images from left/ and right/ subdirectories."""
        images = []
        children = [d for d in person_dir.iterdir() if d.is_dir()]
        by_lower = {d.name.lower(): d for d in children}
        for sub in ["left", "right"]:
            iris_dir = by_lower.get(sub)
            if iris_dir is None:
                continue
            if iris_dir.exists():
                images.extend(sorted(f for f in iris_dir.iterdir() if self._is_image(f)))
        return images

    def _collect_fingerprint_images(self, person_dir: Path) -> list[Path]:
        """Collect fingerprint images."""
        fp_dir = None
        for child in person_dir.iterdir():
            if child.is_dir() and child.name.lower() == "fingerprint":
                fp_dir = child
                break

        if fp_dir is None:
            return []
        if not fp_dir.exists():
            return []
        return sorted(f for f in fp_dir.iterdir() if self._is_image(f))

    def _load_image(self, path: Optional[Path]) -> Optional[Image.Image]:
        if path is None:
            return None
        return Image.open(path).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        iris_img = self._load_image(sample["iris_path"])
        fp_img = self._load_image(sample["fingerprint_path"])

        # Apply transforms
        if iris_img is not None and self.transform_iris is not None:
            iris_img = self.transform_iris(iris_img)
        elif iris_img is not None:
            iris_img = transforms.ToTensor()(iris_img)

        if fp_img is not None and self.transform_fingerprint is not None:
            fp_img = self.transform_fingerprint(fp_img)
        elif fp_img is not None:
            fp_img = transforms.ToTensor()(fp_img)

        # Handle missing modalities with zero tensors
        if iris_img is None:
            iris_img = torch.zeros(3, 128, 128)
        if fp_img is None:
            fp_img = torch.zeros(3, 128, 128)

        return {
            "iris": iris_img,
            "fingerprint": fp_img,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "person_id": sample["person_id"],
        }
