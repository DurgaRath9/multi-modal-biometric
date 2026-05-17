"""End-to-end multimodal biometric model.

Composes iris and fingerprint encoders with a fusion classifier
into a single nn.Module for clean training/inference.
"""

import torch
import torch.nn as nn

from src.models.backbones import build_encoder
from src.models.fusion import build_fusion


class MultiModalBiometricModel(nn.Module):
    """Full multimodal model: encodes both modalities and fuses for classification."""

    def __init__(
        self,
        iris_in_channels: int = 3,
        iris_embedding_dim: int = 128,
        iris_backbone: str = "simple_cnn",
        iris_pretrained: bool = True,
        fp_in_channels: int = 3,
        fp_embedding_dim: int = 128,
        fp_backbone: str = "simple_cnn",
        fp_pretrained: bool = True,
        fusion_strategy: str = "concat",
        hidden_dim: int = 128,
        num_classes: int = 45,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.iris_encoder = build_encoder(
            backbone=iris_backbone,
            in_channels=iris_in_channels,
            embedding_dim=iris_embedding_dim,
            pretrained=iris_pretrained,
        )
        self.fingerprint_encoder = build_encoder(
            backbone=fp_backbone,
            in_channels=fp_in_channels,
            embedding_dim=fp_embedding_dim,
            pretrained=fp_pretrained,
        )
        self.fusion = build_fusion(
            strategy=fusion_strategy,
            iris_embedding_dim=iris_embedding_dim,
            fingerprint_embedding_dim=fp_embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, iris: torch.Tensor, fingerprint: torch.Tensor) -> torch.Tensor:
        iris_emb = self.iris_encoder(iris)
        fp_emb = self.fingerprint_encoder(fingerprint)
        return self.fusion(iris_emb, fp_emb)
