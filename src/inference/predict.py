"""Inference pipeline for the multimodal biometric model."""

import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.multimodal_model import MultiModalBiometricModel

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _validate_image_path(path: str, label: str) -> Path:
    """Validate that a path points to an existing image file."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} image not found: {p}")
    if not p.is_file():
        raise ValueError(f"{label} path is not a file: {p}")
    if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"{label} has unsupported extension '{p.suffix}'. "
            f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
        )
    return p


def _validate_checkpoint_path(path: str) -> Path:
    """Validate that a checkpoint file exists."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    if not p.is_file():
        raise ValueError(f"Checkpoint path is not a file: {p}")
    return p


def load_model(
    checkpoint_path: str, device: torch.device, **model_kwargs
) -> MultiModalBiometricModel:
    """Load a trained model from a checkpoint."""
    ckpt_path = _validate_checkpoint_path(checkpoint_path)
    model = MultiModalBiometricModel(**model_kwargs)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", ckpt_path)
    return model


def predict_single(
    model: MultiModalBiometricModel,
    iris_path: str,
    fingerprint_path: str,
    device: torch.device,
    image_size: int = 128,
) -> dict:
    """Run inference on a single iris + fingerprint pair.

    Returns predicted person ID and confidence scores.
    """
    iris_p = _validate_image_path(iris_path, "Iris")
    fp_p = _validate_image_path(fingerprint_path, "Fingerprint")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    iris_img = Image.open(iris_p).convert("RGB")
    fp_img = Image.open(fp_p).convert("RGB")

    iris_tensor = transform(iris_img).unsqueeze(0).to(device)
    fp_tensor = transform(fp_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(iris_tensor, fp_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = probs.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

    return {
        "predicted_person_id": predicted_class + 1,  # 1-indexed
        "confidence": round(confidence, 4),
        "top5": _top_k(probs[0], k=5),
    }


def _top_k(probs: torch.Tensor, k: int = 5) -> list[dict]:
    """Return top-k predictions with probabilities."""
    values, indices = torch.topk(probs, k)
    return [
        {"person_id": idx.item() + 1, "probability": round(val.item(), 4)}
        for val, idx in zip(values, indices)
    ]
