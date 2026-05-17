"""Tests for training components."""

import torch

from src.training.metrics import (
    compute_accuracy,
    compute_top_k_accuracy,
    compute_epoch_metrics,
)
from src.training.reproducibility import seed_everything


class TestMetrics:
    def test_perfect_accuracy(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        labels = torch.tensor([0, 1, 2])
        assert compute_accuracy(logits, labels) == 1.0

    def test_zero_accuracy(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        labels = torch.tensor([1, 2])
        assert compute_accuracy(logits, labels) == 0.0

    def test_partial_accuracy(self):
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
        labels = torch.tensor([0, 1])
        assert compute_accuracy(logits, labels) == 0.5


class TestTopKAccuracy:
    def test_top5_perfect(self):
        logits = torch.tensor([[10.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
        labels = torch.tensor([0])
        assert compute_top_k_accuracy(logits, labels, k=5) == 1.0

    def test_top5_in_range(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 10.0]])
        labels = torch.tensor([2])  # class 2 has score 3.0, within top-5
        assert compute_top_k_accuracy(logits, labels, k=5) == 1.0

    def test_top1_miss_top5_hit(self):
        logits = torch.tensor([[1.0, 5.0, 4.0, 3.0, 2.0, 0.0]])
        labels = torch.tensor([3])  # class 3 has score 3.0, rank 4
        assert compute_accuracy(logits, labels) == 0.0
        assert compute_top_k_accuracy(logits, labels, k=5) == 1.0


class TestEpochMetrics:
    def test_returns_all_keys(self):
        logits = torch.randn(20, 5)
        labels = torch.randint(0, 5, (20,))
        metrics = compute_epoch_metrics(logits, labels)
        assert "top1_acc" in metrics
        assert "top5_acc" in metrics
        assert "f1_macro" in metrics
        assert "roc_auc" in metrics
        assert "eer" in metrics

    def test_perfect_predictions(self):
        # Create logits where each sample has a high score for the correct class
        logits = torch.zeros(10, 3)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        for i, label in enumerate(labels):
            logits[i, label] = 10.0
        metrics = compute_epoch_metrics(logits, labels)
        assert metrics["top1_acc"] == 1.0
        assert metrics["f1_macro"] == 1.0


class TestReproducibility:
    def test_seed_produces_same_results(self):
        seed_everything(123)
        a = torch.randn(10)
        seed_everything(123)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = torch.randn(10)
        seed_everything(2)
        b = torch.randn(10)
        assert not torch.equal(a, b)
