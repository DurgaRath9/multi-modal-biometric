"""Interactive Gradio demo for multimodal biometric recognition.

Launches a web UI where users can upload iris + fingerprint images
and get real-time identity predictions with confidence scores.

Usage:
    pip install gradio
    python demo.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from src.models.multimodal_model import MultiModalBiometricModel

logger = logging.getLogger(__name__)

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/best_model.pt"
IMAGE_SIZE = 128
NUM_CLASSES = 45
DEVICE = torch.device("cpu")


def _detect_backbone(state_dict: dict) -> str:
    """Infer backbone type from checkpoint state_dict keys."""
    for key in state_dict:
        if ".base.layer1." in key:
            return "resnet18"
    return "simple_cnn"


def _detect_fusion_strategy(state_dict: dict) -> str:
    """Infer fusion strategy from checkpoint state_dict keys."""
    for key in state_dict:
        if key.startswith("fusion.gate.") or key.startswith("fusion.iris_proj."):
            return "attention"
    return "concat"


def _detect_num_classes(state_dict: dict, fusion_strategy: str) -> int:
    """Infer num_classes from the final classifier layer in the checkpoint."""
    # For concat: fusion.classifier.3.weight; for attention: fusion.classifier.2.weight
    for key in sorted(state_dict, reverse=True):
        if key.startswith("fusion.classifier.") and key.endswith(".weight"):
            return state_dict[key].shape[0]
    return NUM_CLASSES


def _load_model() -> MultiModalBiometricModel:
    """Load the trained model from the best checkpoint."""
    ckpt_path = Path(CHECKPOINT_PATH)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        sd = ckpt["model_state_dict"]
        # Use saved config if available, otherwise detect from state_dict keys
        model_cfg = ckpt.get("model_config", {})
        backbone = model_cfg.get("backbone", _detect_backbone(sd))
        fusion_strategy = model_cfg.get("fusion_strategy", _detect_fusion_strategy(sd))
        num_classes = model_cfg.get("num_classes", _detect_num_classes(sd, fusion_strategy))

        model = MultiModalBiometricModel(
            num_classes=num_classes,
            fusion_strategy=fusion_strategy,
            iris_backbone=backbone,
            fp_backbone=backbone,
        )
        model.load_state_dict(sd)
        logger.info(
            "Loaded model from %s (backbone=%s, fusion=%s)", ckpt_path, backbone, fusion_strategy
        )
    else:
        logger.warning("No checkpoint found at %s — using untrained model", ckpt_path)
        model = MultiModalBiometricModel(num_classes=NUM_CLASSES, fusion_strategy="attention")

    model.to(DEVICE)
    model.eval()
    return model


_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_model = _load_model()


def predict(iris_image: Image.Image, fingerprint_image: Image.Image) -> dict[str, float]:
    """Run inference and return top-5 predictions as {label: confidence}."""
    if iris_image is None or fingerprint_image is None:
        return {"Error: Please upload both images": 1.0}

    iris_tensor = _transform(iris_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    fp_tensor = _transform(fingerprint_image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(iris_tensor, fp_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # Top-5 predictions
    values, indices = torch.topk(probs, k=min(5, NUM_CLASSES))
    results = {}
    for val, idx in zip(values, indices):
        person_id = idx.item() + 1  # 1-indexed
        results[f"Person {person_id}"] = round(val.item(), 4)

    return results


# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Iris Image", sources=["upload"]),
        gr.Image(type="pil", label="Fingerprint Image", sources=["upload"]),
    ],
    outputs=gr.Label(num_top_classes=5, label="Predicted Identity"),
    title="Multimodal Biometric Recognition",
    description=(
        "Upload an **iris scan** and a **fingerprint scan** to identify the person. "
        "The model uses gated attention fusion to combine both modalities "
        "and outputs a confidence-ranked list of predicted identities."
    ),
    examples=[],
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
