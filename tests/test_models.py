"""Tests for model architecture."""

import pytest
import torch

from src.models.backbones import SimpleCNNBackbone, ResNet18Backbone, build_encoder
from src.models.fusion import FusionClassifier, GatedAttentionFusion, build_fusion
from src.models.multimodal_model import MultiModalBiometricModel


class TestBackboneFactory:
    def test_simple_cnn(self):
        enc = build_encoder("simple_cnn", in_channels=3, embedding_dim=64)
        assert isinstance(enc, SimpleCNNBackbone)
        out = enc(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 64)

    def test_resnet18(self):
        enc = build_encoder("resnet18", in_channels=3, embedding_dim=64, pretrained=False)
        assert isinstance(enc, ResNet18Backbone)
        out = enc(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 64)

    def test_unknown_backbone_raises(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            build_encoder("unknown_model")


class TestBuildEncoder:
    def test_output_shape(self):
        model = build_encoder("simple_cnn", in_channels=3, embedding_dim=128)
        x = torch.randn(4, 3, 128, 128)
        out = model(x)
        assert out.shape == (4, 128)

    def test_different_embedding_dim(self):
        model = build_encoder("simple_cnn", in_channels=3, embedding_dim=64)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 64)

    def test_resnet18_backbone(self):
        model = build_encoder("resnet18", in_channels=3, embedding_dim=128, pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 128)


class TestFusionClassifier:
    def test_output_shape(self):
        model = FusionClassifier(
            iris_embedding_dim=128,
            fingerprint_embedding_dim=128,
            hidden_dim=64,
            num_classes=45,
        )
        iris_emb = torch.randn(4, 128)
        fp_emb = torch.randn(4, 128)
        out = model(iris_emb, fp_emb)
        assert out.shape == (4, 45)


class TestGatedAttentionFusion:
    def test_output_shape(self):
        model = GatedAttentionFusion(
            iris_embedding_dim=128,
            fingerprint_embedding_dim=128,
            hidden_dim=64,
            num_classes=45,
        )
        iris_emb = torch.randn(4, 128)
        fp_emb = torch.randn(4, 128)
        out = model(iris_emb, fp_emb)
        assert out.shape == (4, 45)

    def test_gate_weights_sum_to_one(self):
        model = GatedAttentionFusion(iris_embedding_dim=64, fingerprint_embedding_dim=64)
        iris_emb = torch.randn(2, 64)
        fp_emb = torch.randn(2, 64)
        gate_input = torch.cat([iris_emb, fp_emb], dim=1)
        weights = model.gate(gate_input)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


class TestFusionFactory:
    def test_build_concat(self):
        module = build_fusion(strategy="concat", num_classes=10)
        assert isinstance(module, FusionClassifier)

    def test_build_attention(self):
        module = build_fusion(strategy="attention", num_classes=10)
        assert isinstance(module, GatedAttentionFusion)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            build_fusion(strategy="nonexistent")


class TestMultiModalModel:
    def test_forward_pass(self):
        model = MultiModalBiometricModel(num_classes=45)
        iris = torch.randn(4, 3, 128, 128)
        fp = torch.randn(4, 3, 128, 128)
        out = model(iris, fp)
        assert out.shape == (4, 45)

    def test_attention_fusion(self):
        model = MultiModalBiometricModel(num_classes=10, fusion_strategy="attention")
        iris = torch.randn(2, 3, 64, 64)
        fp = torch.randn(2, 3, 64, 64)
        out = model(iris, fp)
        assert out.shape == (2, 10)

    def test_gradient_flow(self):
        model = MultiModalBiometricModel(num_classes=10)
        iris = torch.randn(2, 3, 64, 64)
        fp = torch.randn(2, 3, 64, 64)
        labels = torch.tensor([0, 1])

        out = model(iris, fp)
        loss = torch.nn.CrossEntropyLoss()(out, labels)
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
