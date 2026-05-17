"""Multimodal fusion: combines iris and fingerprint embeddings for classification.

Supports two strategies:
  - concat: Simple concatenation + MLP (baseline).
  - attention: Gated attention fusion that learns per-sample modality importance.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    """Concatenates modality embeddings and classifies via a small MLP."""

    def __init__(
        self,
        iris_embedding_dim: int = 128,
        fingerprint_embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 45,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        combined_dim = iris_embedding_dim + fingerprint_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, iris_emb: torch.Tensor, fp_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([iris_emb, fp_emb], dim=1)
        return self.classifier(combined)


class GatedAttentionFusion(nn.Module):
    """Gated attention fusion that learns per-sample modality importance.

    Each modality embedding is projected, then a learned gate decides how
    much weight each modality receives before classification. This lets
    the model rely more on whichever modality is more informative per sample.
    """

    def __init__(
        self,
        iris_embedding_dim: int = 128,
        fingerprint_embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 45,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Project both modalities to the same dimension
        self.iris_proj = nn.Linear(iris_embedding_dim, hidden_dim)
        self.fp_proj = nn.Linear(fingerprint_embedding_dim, hidden_dim)

        # Gating network: takes both embeddings, outputs attention weights
        self.gate = nn.Sequential(
            nn.Linear(iris_embedding_dim + fingerprint_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, iris_emb: torch.Tensor, fp_emb: torch.Tensor) -> torch.Tensor:
        # Project to shared space
        iris_h = self.iris_proj(iris_emb)  # (B, hidden_dim)
        fp_h = self.fp_proj(fp_emb)  # (B, hidden_dim)

        # Compute per-sample gating weights
        gate_input = torch.cat([iris_emb, fp_emb], dim=1)
        weights = self.gate(gate_input)  # (B, 2)

        # Weighted combination
        fused = weights[:, 0:1] * iris_h + weights[:, 1:2] * fp_h  # (B, hidden_dim)

        return self.classifier(fused)


def build_fusion(
    strategy: str = "concat",
    iris_embedding_dim: int = 128,
    fingerprint_embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_classes: int = 45,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a fusion module from a strategy name ('concat' or 'attention')."""
    kwargs = dict(
        iris_embedding_dim=iris_embedding_dim,
        fingerprint_embedding_dim=fingerprint_embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    if strategy == "concat":
        return FusionClassifier(**kwargs)
    elif strategy == "attention":
        return GatedAttentionFusion(**kwargs)
    raise ValueError(f"Unknown fusion strategy '{strategy}'. Choose from: 'concat', 'attention'")
