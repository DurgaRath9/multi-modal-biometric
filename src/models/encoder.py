"""Generic modality encoder with configurable backbone.

A single class replaces the duplicate IrisEncoder/FingerprintEncoder
wrappers. Each modality gets its own instance via Hydra config.
"""

import torch
import torch.nn as nn

from src.models.backbones import build_encoder


class ModalityEncoder(nn.Module):
    """Encodes an image into a fixed-size embedding using a configurable backbone."""

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        backbone: str = "simple_cnn",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            backbone=backbone,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
