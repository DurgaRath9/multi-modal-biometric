"""Grad-CAM visualization for multimodal biometric model.

Generates heatmaps showing which image regions drive the model's prediction.
Useful for interpretability: verifying the model attends to iris texture
and fingerprint ridge patterns rather than background artifacts.

Usage:
    python -m src.inference.gradcam \
        --checkpoint checkpoints/best_model.pt \
        --iris path/to/iris.bmp \
        --fingerprint path/to/fingerprint.bmp \
        --output gradcam_output.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.multimodal_model import MultiModalBiometricModel

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM for a target convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Generate Grad-CAM heatmap.

        Args:
            input_tensor: Model input (will be forwarded through the model).
            class_idx: Target class. If None, uses the predicted class.

        Returns:
            Heatmap as a numpy array of shape (H, W), values in [0, 1].
        """
        output = (
            self.model(*input_tensor)
            if isinstance(input_tensor, tuple)
            else self.model(input_tensor)
        )

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a Grad-CAM heatmap on the original image."""
    import matplotlib.cm as cm

    # Resize heatmap to image size
    heatmap_resized = (
        np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        )
        / 255.0
    )

    colormap = cm.jet(heatmap_resized)[:, :, :3]  # (H, W, 3)
    colormap = (colormap * 255).astype(np.uint8)

    img_array = np.array(image)
    blended = (alpha * colormap + (1 - alpha) * img_array).astype(np.uint8)
    return Image.fromarray(blended)


def _validate_path(path: str, label: str, must_be_file: bool = True) -> Path:
    """Validate that a path exists and is a file."""
    p = Path(path).resolve()
    if must_be_file:
        if not p.is_file():
            raise FileNotFoundError(f"{label} not found: {p}")
    return p


def visualize_gradcam(
    checkpoint_path: str,
    iris_path: str,
    fingerprint_path: str,
    output_path: str = "gradcam_output.png",
    image_size: int = 128,
    **model_kwargs,
) -> None:
    """Generate and save Grad-CAM visualizations for both modalities."""
    ckpt_p = _validate_path(checkpoint_path, "Checkpoint")
    iris_p = _validate_path(iris_path, "Iris image")
    fp_p = _validate_path(fingerprint_path, "Fingerprint image")
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    model = MultiModalBiometricModel(**model_kwargs)
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    iris_img = Image.open(iris_p).convert("RGB")
    fp_img = Image.open(fp_p).convert("RGB")
    iris_tensor = transform(iris_img).unsqueeze(0)
    fp_tensor = transform(fp_img).unsqueeze(0)

    # Grad-CAM on iris encoder (last conv layer)
    iris_target = (
        model.iris_encoder.features[-2]
        if hasattr(model.iris_encoder, "features")
        else list(model.iris_encoder.base.children())[-2]
    )

    cam_iris = GradCAM(model, iris_target)
    heatmap_iris = cam_iris.generate((iris_tensor, fp_tensor))

    # Grad-CAM on fingerprint encoder
    fp_target = (
        model.fingerprint_encoder.features[-2]
        if hasattr(model.fingerprint_encoder, "features")
        else list(model.fingerprint_encoder.base.children())[-2]
    )

    cam_fp = GradCAM(model, fp_target)
    heatmap_fp = cam_fp.generate((iris_tensor, fp_tensor))

    # Create side-by-side visualization
    iris_resized = iris_img.resize((image_size, image_size))
    fp_resized = fp_img.resize((image_size, image_size))

    iris_overlay = overlay_heatmap(iris_resized, heatmap_iris)
    fp_overlay = overlay_heatmap(fp_resized, heatmap_fp)

    # Combine: [original_iris | gradcam_iris | original_fp | gradcam_fp]
    combined_width = image_size * 4
    combined = Image.new("RGB", (combined_width, image_size))
    combined.paste(iris_resized, (0, 0))
    combined.paste(iris_overlay, (image_size, 0))
    combined.paste(fp_resized, (image_size * 2, 0))
    combined.paste(fp_overlay, (image_size * 3, 0))
    combined.save(str(out_p))

    logger.info("Grad-CAM visualization saved to %s", out_p)
    print(f"Saved: {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--iris", required=True, help="Path to iris image")
    parser.add_argument("--fingerprint", required=True, help="Path to fingerprint image")
    parser.add_argument("--output", default="gradcam_output.png", help="Output image path")
    parser.add_argument("--num-classes", type=int, default=45)
    args = parser.parse_args()

    visualize_gradcam(
        checkpoint_path=args.checkpoint,
        iris_path=args.iris,
        fingerprint_path=args.fingerprint,
        output_path=args.output,
        num_classes=args.num_classes,
    )
