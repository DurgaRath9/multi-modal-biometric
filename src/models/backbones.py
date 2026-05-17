"""Configurable encoder backbones for biometric modalities.

Supports simple_cnn (default) and resnet18.
All backbones expose the same interface: (in_channels, embedding_dim) -> nn.Module
that maps an image tensor to a fixed-size embedding vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SimpleCNNBackbone(nn.Module):
    """Lightweight 3-layer CNN for fast iteration on small datasets."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone with optional ImageNet-pretrained weights."""

    def __init__(
        self, in_channels: int = 3, embedding_dim: int = 128, *, pretrained: bool = True
    ) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet18(weights=weights)

        # Adapt first conv if input is not 3-channel
        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original classification head
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.fc = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        return self.fc(features)


def build_encoder(
    backbone: str = "simple_cnn",
    in_channels: int = 3,
    embedding_dim: int = 128,
    pretrained: bool = True,
) -> nn.Module:
    """Build an encoder from a backbone name ('simple_cnn' or 'resnet18')."""
    if backbone == "simple_cnn":
        return SimpleCNNBackbone(in_channels=in_channels, embedding_dim=embedding_dim)
    elif backbone == "resnet18":
        return ResNet18Backbone(
            in_channels=in_channels, embedding_dim=embedding_dim, pretrained=pretrained
        )
    raise ValueError(f"Unknown backbone '{backbone}'. Choose from: 'simple_cnn', 'resnet18'")
