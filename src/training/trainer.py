"""Training loop with TensorBoard logging, LR scheduling, early stopping, and AMP."""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.training.metrics import compute_accuracy, compute_epoch_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Handles the training and validation loop for the multimodal model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "runs",
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        patience: int = 0,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler
        self.patience = patience  # 0 = disabled
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in loader:
            iris = batch["iris"].to(self.device)
            fingerprint = batch["fingerprint"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(iris, fingerprint)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_acc += compute_accuracy(logits, labels)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        logger.info("Epoch %d | Train Loss: %.4f | Train Acc: %.4f", epoch, avg_loss, avg_acc)
        return {"train_loss": avg_loss, "train_acc": avg_acc}

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> dict:
        """Run validation with comprehensive biometric metrics."""
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in loader:
            iris = batch["iris"].to(self.device)
            fingerprint = batch["fingerprint"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(iris, fingerprint)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        all_logits_t = torch.cat(all_logits)
        all_labels_t = torch.cat(all_labels)
        avg_loss = total_loss / max(len(loader), 1)

        metrics = compute_epoch_metrics(all_logits_t, all_labels_t)
        metrics["val_loss"] = avg_loss

        logger.info(
            "Epoch %d | Val Loss: %.4f | Top1: %.4f | Top5: %.4f | F1: %.4f | AUC: %.4f | EER: %.4f",
            epoch,
            avg_loss,
            metrics["top1_acc"],
            metrics["top5_acc"],
            metrics["f1_macro"],
            metrics["roc_auc"],
            metrics["eer"],
        )
        return metrics

    def save_checkpoint(self, epoch: int, metrics: dict, filename: str | None = None) -> None:
        """Save model checkpoint."""
        name = filename or f"checkpoint_epoch_{epoch}.pt"
        path = self.checkpoint_dir / name
        # Extract model config for reproducible loading
        model_config = {}
        if hasattr(self.model, "iris_encoder"):
            encoder = self.model.iris_encoder
            model_config["backbone"] = "resnet18" if hasattr(encoder, "base") else "simple_cnn"
        if hasattr(self.model, "fusion"):
            fusion = self.model.fusion
            model_config["fusion_strategy"] = "attention" if hasattr(fusion, "gate") else "concat"
            if hasattr(fusion, "classifier"):
                last_layer = list(fusion.classifier.children())[-1]
                if hasattr(last_layer, "out_features"):
                    model_config["num_classes"] = last_layer.out_features
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "model_config": model_config,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def _log_to_tensorboard(self, train_m: dict, val_m: dict, epoch: int) -> None:
        """Write scalars to TensorBoard."""
        self.writer.add_scalar("Loss/train", train_m["train_loss"], epoch)
        self.writer.add_scalar("Loss/val", val_m["val_loss"], epoch)
        self.writer.add_scalar("Accuracy/train", train_m["train_acc"], epoch)
        self.writer.add_scalar("Accuracy/val_top1", val_m["top1_acc"], epoch)
        self.writer.add_scalar("Accuracy/val_top5", val_m["top5_acc"], epoch)
        self.writer.add_scalar("Metrics/f1_macro", val_m["f1_macro"], epoch)
        self.writer.add_scalar("Metrics/roc_auc", val_m["roc_auc"], epoch)
        self.writer.add_scalar("Metrics/eer", val_m["eer"], epoch)
        lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("LR", lr, epoch)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        save_every: int = 5,
    ) -> list[dict]:
        """Full training loop with optional early stopping."""
        history: list[dict] = []
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()

            train_metrics = self.train_one_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.perf_counter() - epoch_start
            combined = {
                **train_metrics,
                **val_metrics,
                "epoch": epoch,
                "epoch_time_sec": epoch_time,
            }
            history.append(combined)

            # TensorBoard logging
            self._log_to_tensorboard(train_metrics, val_metrics, epoch)

            # Best model tracking
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                epochs_without_improvement = 0
                self.save_checkpoint(epoch, combined, filename="best_model.pt")
            else:
                epochs_without_improvement += 1

            # Periodic checkpoint
            if epoch % save_every == 0 or epoch == epochs:
                self.save_checkpoint(epoch, combined)

            # Early stopping
            if self.patience > 0 and epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch,
                    self.patience,
                )
                break

        total_time = time.perf_counter() - start_time
        logger.info("Training complete in %.1f seconds", total_time)
        self.writer.close()
        return history
